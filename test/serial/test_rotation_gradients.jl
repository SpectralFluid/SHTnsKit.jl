# SHTnsKit.jl - Rotation AD-adjoint gradient tests
#
# Validates the reverse-mode adjoints (ChainRules rrules in SHTnsKitAdvancedADExt
# and the Zygote @adjoints in SHTnsKitZygoteExt) for the packed real-field
# rotations against finite differences.
#
# Regression guard: the Qlm-gradients were previously WRONG. SH_Zrotate/SH_Yrotate
# conjugated / mis-weighted the cotangent, and SH_Xrotate90 used non-inverse ZYZ
# angles. The correct standard-inner-product adjoint of a packed rotation R is
#   Q̄ = W · R⁻¹ · (W⁻¹ ȳ),   W = diag(wm),  wm = 2 for m>0, 1 for m=0
# because the m>0 packed modes carry double weight in the physical field inner
# product while Zygote/ChainRules use the unweighted packed inner product.
# The angle (dα) gradients were already correct; they are checked here too.

using Test
using Random
using SHTnsKit
using ChainRulesCore

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

const _HAS_ZYGOTE_ROT = try
    @eval using Zygote
    true
catch
    false
end

@testset "Z-rotation pullbacks preserve primal coefficients" begin
    function check_saved_rotation(make_pullback)
        for mres in (1, 2), storage in (:separate, :inplace, :overlapping_views)
            @testset "mres=$mres, storage=$storage" begin
                cfg = create_gauss_config(4, 6; nlon=9, mres=mres)
                rng = MersenneTwister(4260 + mres)
                Q = randn(rng, ComplexF64, cfg.nlm)
                C = randn(rng, ComplexF64, cfg.nlm)
                h = randn(rng, ComplexF64, cfg.nlm)
                alpha, epsilon = 0.7, 1e-6
                loss(q, a) = real(sum(conj(C) .* SH_Zrotate(cfg, q, a, similar(q))))
                fd_alpha = (loss(Q, alpha + epsilon) - loss(Q, alpha - epsilon)) / (2epsilon)
                fd_q = (loss(Q .+ epsilon .* h, alpha) - loss(Q .- epsilon .* h, alpha)) / (2epsilon)

                if storage === :overlapping_views
                    # The output precedes the input so the primal's forward
                    # traversal reads each coefficient before overwriting it.
                    buffer = vcat(zero(eltype(Q)), Q)
                    q, out = view(buffer, 2:length(buffer)), view(buffer, 1:cfg.nlm)
                else
                    q = copy(Q)
                    out = storage === :inplace ? q : similar(q)
                end
                y, back = make_pullback(cfg, q, alpha, out)
                @test y === out
                @test y ≈ SH_Zrotate(cfg, Q, alpha, similar(Q))
                _, qbar, alphabar, _ = back(C)
                @test real(sum(conj(qbar) .* h)) ≈ fd_q rtol=1e-6 atol=1e-8
                @test alphabar ≈ fd_alpha rtol=1e-6 atol=1e-8

                # Callers may reuse the input/output buffers after the primal.
                # Repeated pullbacks must still use the values at that call.
                fill!(q, 0)
                fill!(out, 0)
                _, qbar_again, alphabar_again, _ = back(C)
                @test qbar_again ≈ qbar
                @test alphabar_again ≈ fd_alpha rtol=1e-6 atol=1e-8
                _, qbar_scaled, alphabar_scaled, _ = back(2 .* C)
                @test qbar_scaled ≈ 2 .* qbar
                @test alphabar_scaled ≈ 2fd_alpha rtol=1e-6 atol=1e-8
            end
        end
    end

    @testset "ChainRules" begin
        check_saved_rotation() do cfg, q, alpha, out
            y, back = ChainRulesCore.rrule(SH_Zrotate, cfg, q, alpha, out)
            y, cotangent -> Base.tail(back(cotangent))
        end
    end
    if _HAS_ZYGOTE_ROT
        @testset "Zygote" begin
            check_saved_rotation((cfg, q, alpha, out) -> Zygote.pullback(SH_Zrotate, cfg, q, alpha, out))
        end
    end
end

@testset "Angle gradients survive an in-place primal" begin
    # `SH_Zrotate` is covered above. Its siblings capture the primal INPUT and
    # read it lazily in the pullback, so the same hazard applies to them: an
    # in-place call (Rlm === Qlm) overwrites those coefficients, and so does any
    # caller that reuses the buffer before the pullback runs. Each rrule must
    # snapshot what it needs at primal time.
    rng = MersenneTwister(20931)
    ε = 1e-6

    @testset "SH_Yrotate dα" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        Q = randn(rng, ComplexF64, cfg.nlm)
        C = randn(rng, ComplexF64, cfg.nlm)
        α = 0.7
        loss(q, a) = real(sum(conj(C) .* SH_Yrotate(cfg, copy(q), a, similar(q))))
        fd = (loss(Q, α + ε) - loss(Q, α - ε)) / (2ε)

        q = copy(Q)
        _, back_sep = ChainRulesCore.rrule(SH_Yrotate, cfg, q, α, similar(q))
        @test back_sep(C)[4] ≈ fd rtol=1e-6 atol=1e-8

        inplace = copy(Q)                      # Rlm === Qlm
        _, back_ip = ChainRulesCore.rrule(SH_Yrotate, cfg, inplace, α, inplace)
        @test back_ip(C)[4] ≈ fd rtol=1e-6 atol=1e-8

        reused = copy(Q)                       # caller clobbers the input afterwards
        _, back_reuse = ChainRulesCore.rrule(SH_Yrotate, cfg, reused, α, similar(reused))
        fill!(reused, 0)
        @test back_reuse(C)[4] ≈ fd rtol=1e-6 atol=1e-8

        if _HAS_ZYGOTE_ROT
            zq = copy(Q)
            _, zback = Zygote.pullback(SH_Yrotate, cfg, zq, α, zq)
            @test zback(C)[3] ≈ fd rtol=1e-6 atol=1e-8
        end
    end

    @testset "shtns_rotation_apply_cplx dβ" begin
        lmax = mmax = 4
        mkrot(β) = (r = SHTnsKit.SHTRotation(lmax, mmax);
                    SHTnsKit.shtns_rotation_set_angles_ZYZ(r, 0.3, β, 0.2); r)
        n = SHTnsKit.nlm_cplx_calc(lmax, mmax, 1)
        Z = randn(rng, ComplexF64, n)
        C = randn(rng, ComplexF64, n)
        β = 0.7
        loss(r, z) = (R = similar(z);
                      SHTnsKit.shtns_rotation_apply_cplx(r, copy(z), R);
                      real(sum(conj(C) .* R)))
        fd = (loss(mkrot(β + ε), Z) - loss(mkrot(β - ε), Z)) / (2ε)

        z = copy(Z)
        _, back_sep = ChainRulesCore.rrule(SHTnsKit.shtns_rotation_apply_cplx, mkrot(β), z, similar(z))
        @test back_sep(C)[2].β ≈ fd rtol=1e-6 atol=1e-8

        zi = copy(Z)                            # Rlm === Zlm
        _, back_ip = ChainRulesCore.rrule(SHTnsKit.shtns_rotation_apply_cplx, mkrot(β), zi, zi)
        @test back_ip(C)[2].β ≈ fd rtol=1e-6 atol=1e-8
    end

    @testset "shtns_rotation_apply_real dβ" begin
        cfg = create_gauss_config(4, 7; nlon=11)
        mkrot(β) = (r = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax);
                    SHTnsKit.shtns_rotation_set_angles_ZYZ(r, 0.3, β, 0.2); r)
        Q = randn(rng, ComplexF64, cfg.nlm)
        C = randn(rng, ComplexF64, cfg.nlm)
        β = 0.7
        loss(r, q) = (R = similar(q);
                      SHTnsKit.shtns_rotation_apply_real(r, copy(q), R);
                      real(sum(conj(C) .* R)))
        fd = (loss(mkrot(β + ε), Q) - loss(mkrot(β - ε), Q)) / (2ε)

        q = copy(Q)
        _, back_sep = ChainRulesCore.rrule(SHTnsKit.shtns_rotation_apply_real, mkrot(β), q, similar(q))
        @test back_sep(C)[2].β ≈ fd rtol=1e-6 atol=1e-8

        qi = copy(Q)                            # Rlm === Qlm
        _, back_ip = ChainRulesCore.rrule(SHTnsKit.shtns_rotation_apply_real, mkrot(β), qi, qi)
        @test back_ip(C)[2].β ≈ fd rtol=1e-6 atol=1e-8
    end
end

@testset "Order-mixing rotations reject mres > 1 with a usable message" begin
    cfg = create_gauss_config(4, 6; mres=2)
    Q = randn(MersenneTwister(5), ComplexF64, cfg.nlm)
    err = try
        SH_Yrotate(cfg, Q, 0.3, similar(Q)); nothing
    catch e
        e
    end
    # Either guard is fine — SH_Yrotate's own mres check or the packed-length
    # check inside the Wigner engine — as long as the message names `mres`, so
    # the caller is not left reverse-engineering a bare size mismatch.
    @test err isa Union{ArgumentError,DimensionMismatch}
    @test occursin("mres", sprint(showerror, err))
end

@testset "Complex-packed analysis rrule respects configured convention" begin
    lmax = 4
    cfg = create_gauss_config(lmax, 7; nlon=11, norm=:schmidt,
                              real_norm=true, cs_phase=false)
    rng = MersenneTwister(8491)
    z = randn(rng, ComplexF64, cfg.nlat, cfg.nlon)
    h = randn(rng, ComplexF64, size(z))
    cotangent = randn(rng, ComplexF64, nlm_cplx_calc(lmax, lmax, 1))

    _, pullback = ChainRulesCore.rrule(analysis_packed_cplx, cfg, z)
    _, _, zbar = pullback(cotangent)
    loss(zv) = real(sum(conj(cotangent) .* analysis_packed_cplx(cfg, zv)))
    epsilon = 1e-6
    fd = (loss(z .+ epsilon .* h) - loss(z .- epsilon .* h)) / (2epsilon)
    ad = real(sum(conj(zbar) .* h))
    @test isapprox(ad, fd; rtol=2e-6, atol=2e-8)
end

@testset "Setter rotation rrules use effective Euler angles" begin
    lmax = 4
    cfg = create_gauss_config(lmax, 7; nlon=11)
    rng = MersenneTwister(0xAD37)
    real_input = randn(rng, ComplexF64, cfg.nlm)
    real_cotangent = randn(rng, ComplexF64, cfg.nlm)
    for index in eachindex(real_input)
        cfg.mi[index] == 0 && (real_input[index] = real(real_input[index]))
    end
    complex_input = randn(rng, ComplexF64, nlm_cplx_calc(lmax, lmax, 1))
    complex_cotangent = randn(rng, ComplexF64, length(complex_input))
    real_direction = randn(rng, ComplexF64, cfg.nlm)
    for index in eachindex(real_direction)
        cfg.mi[index] == 0 && (real_direction[index] = real(real_direction[index]))
    end
    complex_direction = randn(rng, ComplexF64, length(complex_input))
    angles = (0.23, 0.41, -0.17)
    epsilon = 1e-6

    function setter_rotation(convention, values)
        rotation = SHTRotation(lmax, lmax)
        if convention === :ZYZ
            shtns_rotation_set_angles_ZYZ(rotation, values...)
        else
            shtns_rotation_set_angles_ZXZ(rotation, values...)
        end
        return rotation
    end

    function equivalent_direct(convention, values)
        alpha, beta, gamma = values
        effective = convention === :ZYZ ?
            (gamma, beta, alpha) :
            (gamma - pi / 2, beta, alpha + pi / 2)
        return SHTRotation(lmax, lmax; α=effective[1], β=effective[2], γ=effective[3])
    end

    function primal_and_pullback(apply, rotation, input, cotangent)
        output = similar(input)
        primal, pullback = ChainRulesCore.rrule(apply, rotation, input, output)
        _, rotation_bar, input_bar, _ = pullback(cotangent)
        return copy(primal), rotation_bar, input_bar
    end

    for (name, apply, input, cotangent, direction) in (
        ("real", shtns_rotation_apply_real, real_input, real_cotangent,
         real_direction),
        ("complex", shtns_rotation_apply_cplx, complex_input,
         complex_cotangent, complex_direction),
    ), convention in (:ZYZ, :ZXZ)
        @testset "$name $convention" begin
            setter = setter_rotation(convention, angles)
            direct = equivalent_direct(convention, angles)
            setter_y, setter_rbar, setter_input_bar =
                primal_and_pullback(apply, setter, input, cotangent)
            direct_y, direct_rbar, direct_input_bar =
                primal_and_pullback(apply, direct, input, cotangent)

            @test setter_y ≈ direct_y atol=2e-12 rtol=2e-12
            @test setter_input_bar ≈ direct_input_bar atol=2e-12 rtol=2e-12
            @test [setter_rbar.α, setter_rbar.β, setter_rbar.γ] ≈
                  [direct_rbar.γ, direct_rbar.β, direct_rbar.α] atol=2e-12 rtol=2e-12

            coefficient_loss(candidate) = real(sum(
                conj(cotangent) .* apply(setter, candidate, similar(candidate)),
            ))
            coefficient_fd = (
                coefficient_loss(input .+ epsilon .* direction) -
                coefficient_loss(input .- epsilon .* direction)
            ) / (2epsilon)
            coefficient_ad = real(sum(conj(setter_input_bar) .* direction))
            @test coefficient_ad ≈ coefficient_fd rtol=2e-5 atol=2e-7

            loss(values) = begin
                rotation = setter_rotation(convention, values)
                output = apply(rotation, input, similar(input))
                real(sum(conj(cotangent) .* output))
            end
            finite_difference(index) = begin
                plus = ntuple(i -> angles[i] + (i == index ? epsilon : 0.0), 3)
                minus = ntuple(i -> angles[i] - (i == index ? epsilon : 0.0), 3)
                (loss(plus) - loss(minus)) / (2epsilon)
            end
            stored_tangent = (setter_rbar.α, setter_rbar.β, setter_rbar.γ)
            for index in 1:3
                @test stored_tangent[index] ≈ finite_difference(index) rtol=2e-5 atol=2e-7
            end
        end
    end

    @testset "Direct and angle-axis rotations retain canonical rrules" begin
        angle_axis = SHTRotation(lmax, lmax)
        shtns_rotation_set_angle_axis(angle_axis, 0.37, 0.2, -0.4, 0.7)
        direct = SHTRotation(lmax, lmax; α=angles[1], β=angles[2], γ=angles[3])

        for (name, apply, input, cotangent, direction) in (
            ("real", shtns_rotation_apply_real, real_input, real_cotangent,
             real_direction),
            ("complex", shtns_rotation_apply_cplx, complex_input,
             complex_cotangent, complex_direction),
        ), (origin, rotation) in (("direct", direct), ("angle-axis", angle_axis))
            @testset "$name $origin" begin
                _, rotation_bar, input_bar =
                    primal_and_pullback(apply, rotation, input, cotangent)
                coefficient_loss(candidate) = real(sum(
                    conj(cotangent) .* apply(rotation, candidate, similar(candidate)),
                ))
                coefficient_fd = (
                    coefficient_loss(input .+ epsilon .* direction) -
                    coefficient_loss(input .- epsilon .* direction)
                ) / (2epsilon)
                coefficient_ad = real(sum(conj(input_bar) .* direction))
                @test coefficient_ad ≈ coefficient_fd rtol=2e-5 atol=2e-7

                stored = (rotation.α, rotation.β, rotation.γ)
                stored_tangent = (rotation_bar.α, rotation_bar.β, rotation_bar.γ)
                angle_loss(values) = begin
                    candidate = SHTRotation(
                        lmax, lmax; α=values[1], β=values[2], γ=values[3],
                    )
                    real(sum(conj(cotangent) .* apply(
                        candidate, input, similar(input),
                    )))
                end
                for index in 1:3
                    plus = ntuple(i -> stored[i] + (i == index ? epsilon : 0.0), 3)
                    minus = ntuple(i -> stored[i] - (i == index ? epsilon : 0.0), 3)
                    fd = (angle_loss(plus) - angle_loss(minus)) / (2epsilon)
                    @test stored_tangent[index] ≈ fd rtol=2e-5 atol=2e-7
                end
            end
        end
    end
end

@testset "Configured axis-wrapper rrules" begin
    lmax = 5
    cfg = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1,
                              norm=:schmidt, real_norm=true, cs_phase=false)
    rng = MersenneTwister(4257)
    real_field_vector() = begin
        value = randn(rng, ComplexF64, cfg.nlm)
        @inbounds for index in eachindex(value)
            cfg.mi[index] == 0 && (value[index] = real(value[index]))
        end
        value
    end
    Q = real_field_vector()
    cotangent = real_field_vector()
    direction = real_field_vector()
    alpha = 0.43
    epsilon = 1e-6

    for (name, apply, arguments) in (
        ("SH_Zrotate", SH_Zrotate, (alpha, similar(Q))),
        ("SH_Yrotate", SH_Yrotate, (alpha, similar(Q))),
        ("SH_Yrotate90", SH_Yrotate90, (similar(Q),)),
        ("SH_Xrotate90", SH_Xrotate90, (similar(Q),)),
    )
        _, pullback = ChainRulesCore.rrule(apply, cfg, Q, arguments...)
        result = pullback(cotangent)
        Qbar = result[3]
        loss(candidate) = real(sum(conj(cotangent) .* apply(
            cfg, candidate, arguments...,
        )))
        fd = (loss(Q .+ epsilon .* direction) -
              loss(Q .- epsilon .* direction)) / (2epsilon)
        ad = real(sum(conj(Qbar) .* direction))
        @testset "$name coefficients" begin
            @test ad ≈ fd rtol=1e-4 atol=1e-8
        end
    end

    for (name, apply) in (("SH_Zrotate", SH_Zrotate),
                          ("SH_Yrotate", SH_Yrotate))
        _, pullback = ChainRulesCore.rrule(apply, cfg, Q, alpha, similar(Q))
        angle_bar = pullback(cotangent)[4]
        loss(angle) = real(sum(conj(cotangent) .* apply(
            cfg, Q, angle, similar(Q),
        )))
        fd = (loss(alpha + epsilon) - loss(alpha - epsilon)) / (2epsilon)
        @testset "$name angle" begin
            @test angle_bar ≈ fd rtol=1e-4 atol=1e-8
        end
    end
end

if _HAS_ZYGOTE_ROT
@testset "Rotation AD adjoints vs finite differences" begin
    # Real-field-compatible packed vector: m=0 entries must be real.
    function _rfvec(rng, cfg)
        v = randn(rng, ComplexF64, cfg.nlm)
        @inbounds for k in 1:cfg.nlm
            cfg.mi[k] == 0 && (v[k] = real(v[k]))
        end
        return v
    end

    for lmax in (5, 8)
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(4242 + lmax)

        Q = _rfvec(rng, cfg)
        C = _rfvec(rng, cfg)   # fixed target -> phase-sensitive linear loss
        h = _rfvec(rng, cfg)   # FD direction (real-field-compatible)
        alpha = 0.7
        ϵ = 1e-6

        # ---- gradient w.r.t. Qlm ----
        # loss(Q) = real(Σ conj(C) · rot(Q)); Zygote convention:
        # L(Q+ϵh) ≈ L(Q) + ϵ Re(Σ conj(g) · h)
        function check_dQ(name, rot)
            loss(q) = real(sum(conj(C) .* rot(q)))
            g = Zygote.gradient(loss, Q)[1]
            @test g !== nothing
            dL_ad = real(sum(conj(g) .* h))
            dL_fd = (loss(Q .+ ϵ .* h) - loss(Q .- ϵ .* h)) / (2ϵ)
            VERBOSE && @info "rotation dQ" name dL_ad dL_fd
            @test isapprox(dL_ad, dL_fd; rtol=1e-4, atol=1e-8)
        end

        check_dQ("SH_Zrotate",   q -> SH_Zrotate(cfg, q, alpha, similar(q)))
        check_dQ("SH_Yrotate",   q -> SH_Yrotate(cfg, q, alpha, similar(q)))
        check_dQ("SH_Yrotate90", q -> SH_Yrotate90(cfg, q, similar(q)))
        check_dQ("SH_Xrotate90", q -> SH_Xrotate90(cfg, q, similar(q)))

        # ---- gradient w.r.t. rotation angle α ----
        function check_dα(name, rot)
            lossα(a) = real(sum(conj(C) .* rot(Q, a)))
            gα = Zygote.gradient(lossα, alpha)[1]
            @test gα !== nothing
            fd = (lossα(alpha + ϵ) - lossα(alpha - ϵ)) / (2ϵ)
            VERBOSE && @info "rotation dα" name gα fd
            @test isapprox(gα, fd; rtol=1e-4, atol=1e-8)
        end

        check_dα("SH_Zrotate α", (q, a) -> SH_Zrotate(cfg, q, a, similar(q)))
        check_dα("SH_Yrotate α", (q, a) -> SH_Yrotate(cfg, q, a, similar(q)))
    end

    @testset "configured axis-wrapper adjoints" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1,
                                  norm=:schmidt, real_norm=true, cs_phase=false)
        rng = MersenneTwister(4257)
        Q = _rfvec(rng, cfg)
        C = _rfvec(rng, cfg)
        h = _rfvec(rng, cfg)
        alpha = 0.43
        ϵ = 1e-6

        function check_configured_dQ(name, rot)
            loss(q) = real(sum(conj(C) .* rot(q)))
            g = Zygote.gradient(loss, Q)[1]
            dL_ad = real(sum(conj(g) .* h))
            dL_fd = (loss(Q .+ ϵ .* h) - loss(Q .- ϵ .* h)) / (2ϵ)
            VERBOSE && @info "configured rotation dQ" name dL_ad dL_fd
            @test isapprox(dL_ad, dL_fd; rtol=1e-4, atol=1e-8)
        end

        check_configured_dQ("SH_Yrotate", q -> SH_Yrotate(cfg, q, alpha, similar(q)))
        check_configured_dQ("SH_Yrotate90", q -> SH_Yrotate90(cfg, q, similar(q)))
        check_configured_dQ("SH_Xrotate90", q -> SH_Xrotate90(cfg, q, similar(q)))

        loss_alpha(a) = real(sum(conj(C) .* SH_Yrotate(cfg, Q, a, similar(Q))))
        g_alpha = Zygote.gradient(loss_alpha, alpha)[1]
        fd_alpha = (loss_alpha(alpha + ϵ) - loss_alpha(alpha - ϵ)) / (2ϵ)
        @test isapprox(g_alpha, fd_alpha; rtol=1e-4, atol=1e-8)

        # Exercise the ChainRules rule directly too: Zygote has a dedicated
        # adjoint for SH_Yrotate, while fixed-angle wrappers use this rrule.
        _, pullback = ChainRulesCore.rrule(SH_Yrotate, cfg, Q, alpha, similar(Q))
        _, _, Qbar, alpha_bar, _ = pullback(C)
        directional = real(sum(conj(Qbar) .* h))
        loss_q(q) = real(sum(conj(C) .* SH_Yrotate(cfg, q, alpha, similar(q))))
        fd_directional = (loss_q(Q .+ ϵ .* h) - loss_q(Q .- ϵ .* h)) / (2ϵ)
        @test isapprox(directional, fd_directional; rtol=1e-4, atol=1e-8)
        @test isapprox(alpha_bar, fd_alpha; rtol=1e-4, atol=1e-8)
    end
end
else
    @info "Skipping rotation-adjoint FD check (Zygote not available in this test context)"
end
