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
# Angle gradients and configured-convention boundary maps are checked here too.

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

        # Exercise the ChainRules rule directly as well; SH_Yrotate also has a
        # Zygote-specific adjoint, while the fixed-angle wrappers use this path.
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

@testset "ZXZ real-rotation rrule angle gradients" begin
    lmax = 4
    cfg = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1)
    rng = MersenneTwister(4258)
    Q = randn(rng, ComplexF64, cfg.nlm)
    C = randn(rng, ComplexF64, cfg.nlm)
    @inbounds for k in eachindex(Q)
        if cfg.mi[k] == 0
            Q[k] = real(Q[k])
            C[k] = real(C[k])
        end
    end

    alpha, beta, gamma = 0.31, 0.57, -0.28
    r = SHTRotation(lmax, lmax)
    shtns_rotation_set_angles_ZXZ(r, alpha, beta, gamma)
    _, pullback = ChainRulesCore.rrule(shtns_rotation_apply_real, r, Q, similar(Q))
    rotation_tangent = ChainRulesCore.unthunk(pullback(C)[2])

    function loss_at(a, b, g)
        rp = SHTRotation(lmax, lmax)
        shtns_rotation_set_angles_ZXZ(rp, a, b, g)
        out = similar(Q)
        shtns_rotation_apply_real(rp, Q, out)
        return real(sum(conj(C) .* out))
    end

    ϵ = 1e-6
    fd_alpha = (loss_at(alpha + ϵ, beta, gamma) -
                loss_at(alpha - ϵ, beta, gamma)) / (2ϵ)
    fd_beta = (loss_at(alpha, beta + ϵ, gamma) -
               loss_at(alpha, beta - ϵ, gamma)) / (2ϵ)
    fd_gamma = (loss_at(alpha, beta, gamma + ϵ) -
                loss_at(alpha, beta, gamma - ϵ)) / (2ϵ)
    @test rotation_tangent.α ≈ fd_alpha rtol=1e-4 atol=1e-8
    @test rotation_tangent.β ≈ fd_beta rtol=1e-4 atol=1e-8
    @test rotation_tangent.γ ≈ fd_gamma rtol=1e-4 atol=1e-8
end
