# SHTnsKit.jl - Energy Gradient Tests
# Tests for analytic gradients validated against finite differences

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

const _HAS_CHAINRULES = try
    @eval using ChainRulesCore: rrule
    true
catch
    false
end

# Only run the synthesis-adjoint FD check if Zygote is reachable. `test_serial.jl`
# (used in some CI profiles) does not load Zygote; skip cleanly instead of
# erroring on the `using` statement.
const _HAS_ZYGOTE = try
    @eval using Zygote
    true
catch
    false
end

if _HAS_ZYGOTE
@testset "synthesis rrule: adjoint consistency with finite-difference" begin
    # Verifies that the rrule for `synthesis` implements the true mathematical
    # adjoint (not just `analysis`, which differs by Gauss-Legendre weights).
    # Before this test existed, the rrule returned `analysis(cfg, ȳ)` which is
    # off by a lat-dependent factor of w_i * cphi — silent on dot-product-only
    # tests but visible in any direct FD check against alm.
    for lmax in (4, 8)
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(1 + lmax)

        # Real-field-compatible alm (m=0 real, l<m zeroed)
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        loss(a) = sum(abs2, synthesis(cfg, a; real_output=true))
        g = Zygote.gradient(loss, alm)[1]

        # Directional FD check
        h = randn(rng, ComplexF64, lmax+1, lmax+1)
        h[:, 1] .= real.(h[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            h[l+1, m+1] = 0
        end
        ϵ = 1e-6
        L(ξ) = loss(alm .+ ξ .* h)
        dL_fd = (L(ϵ) - L(-ϵ)) / (2ϵ)
        # Zygote/ChainRules convention: L(a + ϵh) ≈ L(a) + ϵ Re(sum(conj(g) ⋅ h))
        dL_ad = real(sum(conj(g) .* h))
        @test isapprox(dL_ad, dL_fd; rtol=5e-4, atol=1e-7)
    end
end

@testset "batch rrules: adjoint consistency with finite-difference" begin
    # synthesis_batch's rrule must use the true synthesis adjoint per slice, NOT
    # `analysis` (off by Gauss w_i·cphi weights). analysis_batch's adjoint is
    # `_adjoint_analysis` per slice. Both FD-checked here.
    lmax = 6; nlat = lmax + 2; nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(313)
    rfalm() = begin
        a = randn(rng, ComplexF64, lmax+1, lmax+1); a[:, 1] .= real.(a[:, 1])
        for m in 0:lmax, l in 0:(m-1); a[l+1, m+1] = 0; end
        a
    end
    ϵ = 1e-6

    # synthesis_batch (spectral -> spatial), 2 fields
    ab = cat(rfalm(), rfalm(); dims=3)
    hb = cat(rfalm(), rfalm(); dims=3)
    sloss(a) = sum(abs2, synthesis_batch(cfg, a; real_output=true))
    gs = Zygote.gradient(sloss, ab)[1]
    s_ad = real(sum(conj(gs) .* hb))
    s_fd = (sloss(ab .+ ϵ .* hb) - sloss(ab .- ϵ .* hb)) / (2ϵ)
    @test isapprox(s_ad, s_fd; rtol=5e-4, atol=1e-7)

    # analysis_batch (spatial -> spectral), 2 fields
    fb = randn(rng, nlat, nlon, 2); hf = randn(rng, nlat, nlon, 2)
    aloss(f) = sum(abs2, analysis_batch(cfg, f))
    ga = Zygote.gradient(aloss, fb)[1]
    a_ad = sum(ga .* hf)
    a_fd = (aloss(fb .+ ϵ .* hf) - aloss(fb .- ϵ .* hf)) / (2ϵ)
    @test isapprox(a_ad, a_fd; rtol=5e-4, atol=1e-7)
end
else
    @info "Skipping synthesis-rrule FD check (Zygote not available in this test context)"
end

@testset "analysis_sphtor adjoint preserves complex cotangents" begin
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(119)

    Sbar = zeros(ComplexF64, lmax + 1, lmax + 1)
    Tbar = zeros(ComplexF64, lmax + 1, lmax + 1)
    for m in 0:lmax, l in max(1, m):lmax
        Sbar[l+1, m+1] = randn(rng) + im * randn(rng)
        Tbar[l+1, m+1] = randn(rng) + im * randn(rng)
    end

    Vtbar, Vpbar = SHTnsKit._adjoint_analysis_sphtor(cfg, Sbar, Tbar)
    @test eltype(Vtbar) === ComplexF64
    @test eltype(Vpbar) === ComplexF64
    @test !all(iszero, imag.(Vtbar))
    @test !all(iszero, imag.(Vpbar))

    Vt = randn(rng, ComplexF64, nlat, nlon)
    Vp = randn(rng, ComplexF64, nlat, nlon)
    S, T = analysis_sphtor(cfg, Vt, Vp)
    lhs = real(sum(conj.(Sbar) .* S) + sum(conj.(Tbar) .* T))
    rhs = real(sum(conj.(Vtbar) .* Vt) + sum(conj.(Vpbar) .* Vp))
    @test isapprox(rhs, lhs; rtol=1e-10, atol=1e-11)

    theta_rows = [2, 5]
    phi_window = 3:7
    Vtwindow, Vpwindow = SHTnsKit._adjoint_analysis_sphtor(
        cfg, Sbar, Tbar; θ_globals=theta_rows, φ_window=phi_window)
    @test eltype(Vtwindow) === ComplexF64
    @test eltype(Vpwindow) === ComplexF64
    @test Vtwindow ≈ Vtbar[theta_rows, phi_window]
    @test Vpwindow ≈ Vpbar[theta_rows, phi_window]
end

if _HAS_CHAINRULES
@testset "analysis_sphtor rrule projects onto each primal tangent space" begin
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(120)

    Vt_real = randn(rng, nlat, nlon)
    Vp_real = randn(rng, nlat, nlon)
    Vt_complex = complex.(Vt_real, randn(rng, nlat, nlon))
    Vp_complex = complex.(Vp_real, randn(rng, nlat, nlon))
    Sbar = randn(rng, ComplexF64, lmax + 1, lmax + 1)
    Tbar = randn(rng, ComplexF64, lmax + 1, lmax + 1)

    raw_Vtbar, raw_Vpbar = SHTnsKit._adjoint_analysis_sphtor(cfg, Sbar, Tbar)

    # The primitive deliberately preserves complex cotangents.  The rrule must
    # then project each result into the tangent space of its corresponding
    # primal: real arrays get real cotangents, complex arrays retain both parts.
    _, pullback_real = rrule(analysis_sphtor, cfg, Vt_real, Vp_real)
    _, _, Vtbar_real, Vpbar_real = pullback_real((Sbar, Tbar))
    @test eltype(Vtbar_real) <: Real
    @test eltype(Vpbar_real) <: Real
    @test Vtbar_real ≈ real.(raw_Vtbar)
    @test Vpbar_real ≈ real.(raw_Vpbar)

    _, pullback_complex = rrule(analysis_sphtor, cfg, Vt_complex, Vp_complex)
    _, _, Vtbar_complex, Vpbar_complex = pullback_complex((Sbar, Tbar))
    @test eltype(Vtbar_complex) <: Complex
    @test eltype(Vpbar_complex) <: Complex
    @test Vtbar_complex ≈ raw_Vtbar
    @test Vpbar_complex ≈ raw_Vpbar

    # Projection is per argument, not a single decision for the input pair.
    _, pullback_mixed = rrule(analysis_sphtor, cfg, Vt_real, Vp_complex)
    _, _, Vtbar_mixed, Vpbar_mixed = pullback_mixed((Sbar, Tbar))
    @test eltype(Vtbar_mixed) <: Real
    @test eltype(Vpbar_mixed) <: Complex
end
else
    @info "Skipping analysis_sphtor tangent-projection rrule test (ChainRulesCore not available)"
end

@testset "Energy Gradients" begin
    @testset "Scalar energy gradient (matrix form)" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(120)

        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # ChainRules complex-gradient convention: grad = wm * alm, with
        # dE(a)[h] = real(sum(conj(grad) .* h)).
        grad = grad_energy_scalar_alm(cfg, alm)

        # Finite difference validation
        ϵ = 1e-7
        h = randn(rng, ComplexF64, lmax+1, lmax+1)
        h[:, 1] .= real.(h[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            h[l+1, m+1] = 0
        end

        E_plus = energy_scalar(cfg, alm .+ ϵ .* h)
        E_minus = energy_scalar(cfg, alm .- ϵ .* h)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(grad) .* h))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Scalar energy gradient multiple directions" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(121)

        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        grad = grad_energy_scalar_alm(cfg, alm)
        ϵ = 1e-7

        # Test multiple random directions
        for _ in 1:5
            h = randn(rng, ComplexF64, lmax+1, lmax+1)
            h[:, 1] .= real.(h[:, 1])
            for m in 0:lmax, l in 0:(m-1)
                h[l+1, m+1] = 0
            end

            E_plus = energy_scalar(cfg, alm .+ ϵ .* h)
            E_minus = energy_scalar(cfg, alm .- ϵ .* h)
            dE_fd = (E_plus - E_minus) / (2ϵ)
            dE_ad = real(sum(conj(grad) .* h))

            @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
        end
    end

    @testset "Vector energy gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(122)

        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Slm[1, :] .= 0; Tlm[1, :] .= 0

        # Analytic gradient
        gS, gT = grad_energy_vector_Slm_Tlm(cfg, Slm, Tlm)

        # Finite difference
        ϵ = 1e-7
        hS = randn(rng, ComplexF64, lmax+1, lmax+1)
        hT = randn(rng, ComplexF64, lmax+1, lmax+1)
        hS[:, 1] .= real.(hS[:, 1])
        hT[:, 1] .= real.(hT[:, 1])
        hS[1, :] .= 0; hT[1, :] .= 0

        E_plus = energy_vector(cfg, Slm .+ ϵ .* hS, Tlm .+ ϵ .* hT)
        E_minus = energy_vector(cfg, Slm .- ϵ .* hS, Tlm .- ϵ .* hT)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(gS) .* hS) + sum(conj(gT) .* hT))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Enstrophy gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(123)

        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Tlm[1, :] .= 0

        # Analytic gradient
        gT = grad_enstrophy_Tlm(cfg, Tlm)

        # Finite difference
        ϵ = 1e-7
        h = randn(rng, ComplexF64, lmax+1, lmax+1)
        h[:, 1] .= real.(h[:, 1])
        h[1, :] .= 0

        Z_plus = enstrophy(cfg, Tlm .+ ϵ .* h)
        Z_minus = enstrophy(cfg, Tlm .- ϵ .* h)
        dZ_fd = (Z_plus - Z_minus) / (2ϵ)
        dZ_ad = real(sum(conj(gT) .* h))

        @test isapprox(dZ_ad, dZ_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Packed scalar gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(124)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        gQ = grad_energy_scalar_packed(cfg, Qlm)
        @test length(gQ) == cfg.nlm

        ϵ = 1e-7
        h = randn(rng, ComplexF64, cfg.nlm)
        h[1:lmax+1] .= real.(h[1:lmax+1])

        E_plus = energy_scalar_packed(cfg, Qlm .+ ϵ .* h)
        E_minus = energy_scalar_packed(cfg, Qlm .- ϵ .* h)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(gQ) .* h))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Packed vector gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(125)

        Sp = randn(rng, ComplexF64, cfg.nlm)
        Tp = randn(rng, ComplexF64, cfg.nlm)
        Sp[1:lmax+1] .= real.(Sp[1:lmax+1])
        Tp[1:lmax+1] .= real.(Tp[1:lmax+1])
        # l=0 should be zero for vectors
        Sp[1] = 0; Tp[1] = 0

        gS, gT = grad_energy_vector_packed(cfg, Sp, Tp)

        ϵ = 1e-7
        hS = randn(rng, ComplexF64, cfg.nlm)
        hT = randn(rng, ComplexF64, cfg.nlm)
        hS[1:lmax+1] .= real.(hS[1:lmax+1])
        hT[1:lmax+1] .= real.(hT[1:lmax+1])
        hS[1] = 0; hT[1] = 0

        E_plus = energy_vector_packed(cfg, Sp .+ ϵ .* hS, Tp .+ ϵ .* hT)
        E_minus = energy_vector_packed(cfg, Sp .- ϵ .* hS, Tp .- ϵ .* hT)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(gS) .* hS) + sum(conj(gT) .* hT))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Grid energy scalar gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(126)

        f = randn(rng, ComplexF64, nlat, nlon)
        gf = grad_grid_energy_scalar_field(cfg, f)

        @test size(gf) == (nlat, nlon)

        ϵ = 1e-7
        h = randn(rng, ComplexF64, nlat, nlon)

        E_plus = grid_energy_scalar(cfg, f .+ ϵ .* h)
        E_minus = grid_energy_scalar(cfg, f .- ϵ .* h)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(gf) .* h))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

    @testset "Grid energy vector gradient" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(127)

        Vt = randn(rng, ComplexF64, nlat, nlon)
        Vp = randn(rng, ComplexF64, nlat, nlon)
        gVt, gVp = grad_grid_energy_vector_fields(cfg, Vt, Vp)

        ϵ = 1e-7
        ht = randn(rng, ComplexF64, nlat, nlon)
        hp = randn(rng, ComplexF64, nlat, nlon)

        E_plus = grid_energy_vector(cfg, Vt .+ ϵ .* ht, Vp .+ ϵ .* hp)
        E_minus = grid_energy_vector(cfg, Vt .- ϵ .* ht, Vp .- ϵ .* hp)
        dE_fd = (E_plus - E_minus) / (2ϵ)
        dE_ad = real(sum(conj(gVt) .* ht) + sum(conj(gVp) .* hp))

        @test isapprox(dE_ad, dE_fd; rtol=5e-4, atol=1e-8)
    end

end


if _HAS_CHAINRULES
@testset "QST rrules and implementation-detail kwargs" begin
    # The QST family had no rrule at all, so differentiating a QST pipeline fell
    # through to source tracing and crashed inside FFTW. And every rrule that
    # declared fewer kwargs than its primal was skipped the moment a caller
    # passed one, even at its default value.
    cfg = create_gauss_config(5, 7; nlon=12)
    rng = MersenneTwister(9412)
    rnd() = begin
        a = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        for m in 0:cfg.mmax, l in m:cfg.lmax
            a[l+1, m+1] = m == 0 ? randn(rng) : complex(randn(rng), randn(rng))
        end
        a
    end
    ε = 1e-6

    @testset "synthesis_qst pullback vs finite differences" begin
        Q, S, T = rnd(), rnd(), rnd()
        Cr, Ct, Cp = randn(rng, cfg.nlat, cfg.nlon), randn(rng, cfg.nlat, cfg.nlon), randn(rng, cfg.nlat, cfg.nlon)
        loss(q, s, t) = (V = synthesis_qst(cfg, q, s, t);
                         sum(Cr .* V[1]) + sum(Ct .* V[2]) + sum(Cp .* V[3]))
        _, back = rrule(synthesis_qst, cfg, Q, S, T)
        _, _, Q̄, S̄, T̄, _ = back((Cr, Ct, Cp))
        for (slot, ḡ, perturb) in ((:Q, Q̄, (h, q, s, t) -> (q .+ h, s, t)),
                                   (:S, S̄, (h, q, s, t) -> (q, s .+ h, t)),
                                   (:T, T̄, (h, q, s, t) -> (q, s, t .+ h)))
            h = rnd()
            fd = (loss(perturb(ε .* h, Q, S, T)...) - loss(perturb(-ε .* h, Q, S, T)...)) / (2ε)
            @test real(sum(conj(ḡ) .* h)) ≈ fd rtol=1e-6 atol=1e-8
        end
    end

    @testset "analysis_qst pullback vs finite differences" begin
        Vr, Vt, Vp = randn(rng, cfg.nlat, cfg.nlon), randn(rng, cfg.nlat, cfg.nlon), randn(rng, cfg.nlat, cfg.nlon)
        G1, G2, G3 = rnd(), rnd(), rnd()
        loss(vr, vt, vp) = (A = analysis_qst(cfg, vr, vt, vp);
                            real(sum(conj(G1) .* A[1]) + sum(conj(G2) .* A[2]) + sum(conj(G3) .* A[3])))
        _, back = rrule(analysis_qst, cfg, Vr, Vt, Vp)
        _, _, V̄r, V̄t, V̄p = back((G1, G2, G3))
        for (ḡ, perturb) in ((V̄r, (h, a, b, c) -> (a .+ h, b, c)),
                             (V̄t, (h, a, b, c) -> (a, b .+ h, c)),
                             (V̄p, (h, a, b, c) -> (a, b, c .+ h)))
            h = randn(rng, cfg.nlat, cfg.nlon)
            fd = (loss(perturb(ε .* h, Vr, Vt, Vp)...) - loss(perturb(-ε .* h, Vr, Vt, Vp)...)) / (2ε)
            @test sum(real(ḡ) .* h) ≈ fd rtol=1e-6 atol=1e-8
        end
    end

    @testset "rrules accept the primal's implementation-detail kwargs" begin
        alm = rnd()
        f = randn(rng, cfg.nlat, cfg.nlon)
        Vt, Vp = randn(rng, cfg.nlat, cfg.nlon), randn(rng, cfg.nlat, cfg.nlon)
        @test rrule(synthesis, cfg, alm; use_rfft=false) !== nothing
        @test rrule(synthesis, cfg, alm; real_output=true, fft_scratch=nothing) !== nothing
        @test rrule(analysis, cfg, f; use_rfft=false) !== nothing
        @test rrule(analysis_sphtor, cfg, Vt, Vp; use_rfft=false) !== nothing
        @test rrule(synthesis_sphtor, cfg, alm, alm; use_rfft=false) !== nothing
        # and the rfft path gives the same adjoint as the complex one
        ȳ = randn(rng, cfg.nlat, cfg.nlon)
        _, b1 = rrule(synthesis, cfg, alm)
        _, b2 = rrule(synthesis, cfg, alm; use_rfft=true)
        @test b1(ȳ)[3] ≈ b2(ȳ)[3] rtol=1e-10
    end
end
else
    @info "Skipping QST rrule tests (ChainRulesCore not available)"
end
