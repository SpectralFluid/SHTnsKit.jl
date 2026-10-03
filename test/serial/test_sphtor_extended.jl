# SHTnsKit.jl - Extended spheroidal/toroidal tests
# Covers code paths not hit by test_vector_transforms.jl:
# - Robert form roundtrip
# - Helmholtz-style decomposition: curl-free (S-only) vs div-free (T-only)
# - PLM table path for sphtor
# - Complex spheroidal/toroidal transforms
# - Div/vorticity conversions consistency

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

function _rand_vec_alm(rng, lmax, mmax)
    S = randn(rng, ComplexF64, lmax + 1, mmax + 1)
    T = randn(rng, ComplexF64, lmax + 1, mmax + 1)
    S[:, 1] .= real.(S[:, 1]); T[:, 1] .= real.(T[:, 1])
    S[1, 1] = 0; T[1, 1] = 0  # no l=0 for vectors
    for m in 0:mmax, l in 0:max(0, m - 1)
        S[l + 1, m + 1] = 0; T[l + 1, m + 1] = 0
    end
    return S, T
end

@testset "Sphtor transforms (extended)" begin
    @testset "Real coefficient matrices retain complex Fourier storage" begin
        for T in (Float32, Float64), norm in (:orthonormal, :schmidt)
            cfg = create_gauss_config(2, 4; nlon=7, norm)
            S = zeros(T, 3, 3); S[2, 2] = T(0.75)
            Tlm = zero(S); Tlm[3, 2] = T(-0.25)
            Sc, Tc = complex.(S), complex.(Tlm)
            tol = T === Float32 ? 2f-6 : 2e-14
            for real_output in (false, true)
                reference = synthesis_sphtor(cfg, Sc, Tc; real_output)
                expected_type = real_output ? T : Complex{T}
                for actual in (synthesis_sphtor(cfg, S, Tlm; real_output),
                               synthesis_sphtor_l(cfg, S, Tlm, 2; real_output))
                    @test all(eltype(v) === expected_type for v in actual)
                    @test actual[1] ≈ reference[1] atol=tol rtol=tol
                    @test actual[2] ≈ reference[2] atol=tol rtol=tol
                end
            end
            reference = synthesis_sphtor(cfg, Sc, Tc)
            rfft_result = synthesis_sphtor(cfg, S, Tlm; use_rfft=true)
            @test rfft_result[1] ≈ reference[1] atol=tol rtol=tol
            @test rfft_result[2] ≈ reference[2] atol=tol rtol=tol
            qst = synthesis_qst(cfg, S, S, Tlm)
            @test qst[1] ≈ synthesis(cfg, Sc) atol=tol rtol=tol
            @test qst[2] ≈ reference[1] atol=tol rtol=tol
            @test qst[3] ≈ reference[2] atol=tol rtol=tol
            for (actual, expected) in ((synthesis_sph(cfg, S), synthesis_sph(cfg, Sc)),
                                       (synthesis_tor(cfg, Tlm), synthesis_tor(cfg, Tc)))
                @test actual[1] ≈ expected[1] atol=tol rtol=tol
                @test actual[2] ≈ expected[2] atol=tol rtol=tol
            end
        end
    end

    @testset "Robert form roundtrip" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1, robert_form=true)
        @test cfg.robert_form
        rng = MersenneTwister(300)

        Slm, Tlm = _rand_vec_alm(rng, lmax, lmax)
        Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        Slm_rec, Tlm_rec = analysis_sphtor(cfg, Vt, Vp)

        @test isapprox(Slm_rec, Slm; rtol=1e-9, atol=1e-11)
        @test isapprox(Tlm_rec, Tlm; rtol=1e-9, atol=1e-11)
    end

    @testset "Robert analysis rejects nonzero-weight m=1 poles" begin
        cfg = create_regular_config(2, 9; nlon=7, include_poles=true,
                                    robert_form=true)
        S = zeros(ComplexF64, 3, 3); S[2, 2] = 1
        Tlm = zero(S)
        Vt, Vp = synthesis_sphtor(cfg, S, Tlm)
        @test all(isfinite, Vt) && all(isfinite, Vp)
        for tables in (false, true)
            tables ? set_use_tables!(cfg) : set_on_the_fly!(cfg)
            @test_throws ArgumentError analysis_sphtor(cfg, Vt, Vp)
            @test_throws ArgumentError analysis_sphtor(cfg, Vt, Vp; use_rfft=true)
            @test_throws ArgumentError analysis_sphtor_l(cfg, Vt, Vp, 2)
            @test_throws ArgumentError analysis_sphtor_cplx(cfg, complex.(Vt), complex.(Vp))
            @test_throws ArgumentError analysis_qst(cfg, zero(Vt), Vt, Vp)
            @test_throws ArgumentError analysis_sphtor_batch(
                cfg, reshape(Vt, cfg.nlat, cfg.nlon, 1), reshape(Vp, cfg.nlat, cfg.nlon, 1))
            plan = SHTPlan(cfg)
            Sout, Tout = fill(7.0 + 2im, 3, 3), fill(8.0 + 3im, 3, 3)
            @test_throws ArgumentError analysis_sphtor!(plan, Sout, Tout, Vt, Vp)
            @test all(==(7 + 2im), Sout) && all(==(8 + 3im), Tout)
        end
        mode = zeros(ComplexF64, cfg.nlat)
        @test_throws ArgumentError analysis_sphtor_ml(cfg, 1, mode, mode, 2)
        @test all(iszero, first(analysis_sphtor_ml(cfg, 0, mode, mode, 2)))
        @test all(iszero, first(analysis_sphtor_ml(cfg, 2, mode, mode, 2)))
        @test all(iszero, first(analysis_sphtor_l(cfg, Vt, Vp, 0)))
        @test all(value -> all(iszero, value), analysis_qst_l(cfg, zero(Vt), Vt, Vp, 0))

        # Zero-weight DH poles and layouts without m=1 retain their inverse.
        for safe_cfg in (
            create_regular_config(2, 6; nlon=7, include_poles=true,
                                  use_dh_weights=true, robert_form=true),
            create_regular_config(2, 9; nlon=7, include_poles=true,
                                  mmax=0, robert_form=true),
            create_regular_config(2, 9; nlon=7, include_poles=true,
                                  mres=2, robert_form=true),
        )
            safe_S = zeros(ComplexF64, safe_cfg.lmax+1, safe_cfg.mmax+1)
            safe_S[2, 1] = 0.5
            safe_cfg.mres == 1 && safe_cfg.mmax > 0 && (safe_S[2, 2] = 1)
            safe_cfg.mres == 2 && (safe_S[3, 3] = 0.2 - 0.1im)
            safe_T = zero(safe_S)
            v = synthesis_sphtor(safe_cfg, safe_S, safe_T)
            recovered = analysis_sphtor(safe_cfg, v...)
            @test recovered[1] ≈ safe_S atol=2e-14 rtol=2e-14
            @test recovered[2] ≈ safe_T atol=2e-14 rtol=2e-14
        end
    end

    @testset "PLM tables path: sphtor roundtrip" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        prepare_plm_tables!(cfg)
        @test cfg.use_plm_tables
        rng = MersenneTwister(301)

        Slm, Tlm = _rand_vec_alm(rng, lmax, lmax)
        Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        Slm_r, Tlm_r = analysis_sphtor(cfg, Vt, Vp)

        @test isapprox(Slm_r, Slm; rtol=1e-10, atol=1e-12)
        @test isapprox(Tlm_r, Tlm; rtol=1e-10, atol=1e-12)

        # Compare with on-the-fly cfg (same data) → same spatial field
        cfg_otf = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        Vt_otf, Vp_otf = synthesis_sphtor(cfg_otf, Slm, Tlm; real_output=true)
        @test isapprox(Vt, Vt_otf; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp, Vp_otf; rtol=1e-10, atol=1e-12)
    end

    @testset "Helmholtz decomposition: sph-only + tor-only = full" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(302)
        Slm, Tlm = _rand_vec_alm(rng, lmax, lmax)

        Vt_full, Vp_full = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        Vt_s, Vp_s = synthesis_sphtor(cfg, Slm, zero(Tlm); real_output=true)
        Vt_t, Vp_t = synthesis_sphtor(cfg, zero(Slm), Tlm; real_output=true)

        @test isapprox(Vt_full, Vt_s .+ Vt_t; rtol=1e-12, atol=1e-14)
        @test isapprox(Vp_full, Vp_s .+ Vp_t; rtol=1e-12, atol=1e-14)

        # synthesis_sph and synthesis_tor should agree with partial results
        Vt_sph, Vp_sph = synthesis_sph(cfg, Slm; real_output=true)
        Vt_tor, Vp_tor = synthesis_tor(cfg, Tlm; real_output=true)
        @test isapprox(Vt_sph, Vt_s; rtol=1e-12, atol=1e-14)
        @test isapprox(Vp_sph, Vp_s; rtol=1e-12, atol=1e-14)
        @test isapprox(Vt_tor, Vt_t; rtol=1e-12, atol=1e-14)
        @test isapprox(Vp_tor, Vp_t; rtol=1e-12, atol=1e-14)
    end

    @testset "Toroidal-only field is divergence-free" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(303)
        _, Tlm = _rand_vec_alm(rng, lmax, lmax)
        Slm_zero = zeros(ComplexF64, lmax + 1, lmax + 1)

        # Divergence from a pure-toroidal field: S=0 → div = 0
        δ = divergence_from_spheroidal(cfg, Slm_zero)
        @test maximum(abs, δ) < 1e-14
    end

    @testset "Spheroidal-only field is curl-free (ζ=0)" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        Tlm_zero = zeros(ComplexF64, lmax + 1, lmax + 1)
        ζ = vorticity_from_toroidal(cfg, Tlm_zero)
        @test maximum(abs, ζ) < 1e-14
    end

    @testset "div ∘ S⁻¹ = identity on spectra (round-trip)" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(304)
        Slm, _ = _rand_vec_alm(rng, lmax, lmax)

        δ = divergence_from_spheroidal(cfg, Slm)
        Slm_back = spheroidal_from_divergence(cfg, δ)
        # l=0 stays zero (no monopole in vector)
        @test isapprox(Slm_back, Slm; rtol=1e-12, atol=1e-14)
    end

    @testset "vort ∘ T⁻¹ = identity" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(305)
        _, Tlm = _rand_vec_alm(rng, lmax, lmax)

        ζ = vorticity_from_toroidal(cfg, Tlm)
        Tlm_back = toroidal_from_vorticity(cfg, ζ)
        @test isapprox(Tlm_back, Tlm; rtol=1e-12, atol=1e-14)
    end

    @testset "In-place div/vort variants match" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(306)
        Slm, Tlm = _rand_vec_alm(rng, lmax, lmax)

        δ_ref = divergence_from_spheroidal(cfg, Slm)
        δ_in = similar(Slm)
        divergence_from_spheroidal!(cfg, δ_in, Slm)
        @test δ_in == δ_ref

        ζ_ref = vorticity_from_toroidal(cfg, Tlm)
        ζ_in = similar(Tlm)
        vorticity_from_toroidal!(cfg, ζ_in, Tlm)
        @test ζ_in == ζ_ref

        S_out = similar(δ_ref)
        spheroidal_from_divergence!(cfg, S_out, δ_ref)
        @test S_out == spheroidal_from_divergence(cfg, δ_ref)

        T_out = similar(ζ_ref)
        toroidal_from_vorticity!(cfg, T_out, ζ_ref)
        @test T_out == toroidal_from_vorticity(cfg, ζ_ref)
    end

    @testset "Complex sphtor transforms roundtrip" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(307)
        Slm, Tlm = _rand_vec_alm(rng, lmax, lmax)

        Vt, Vp = synthesis_sphtor_cplx(cfg, Slm, Tlm)
        @test eltype(Vt) <: Complex
        @test eltype(Vp) <: Complex

        Slm_b, Tlm_b = analysis_sphtor_cplx(cfg, Vt, Vp)
        @test isapprox(Slm_b, Slm; rtol=1e-10, atol=1e-12)
        @test isapprox(Tlm_b, Tlm; rtol=1e-10, atol=1e-12)
    end

    @testset "Regular grid (non-gauss) sphtor runs and is finite" begin
        # Regular (equiangular) quadrature is not exact for Legendre integrals,
        # so the sphtor analysis∘synthesis roundtrip has noticeably larger error
        # than Gauss. Just exercise the code path and verify boundedness.
        lmax = 5
        cfg = create_regular_config(lmax, lmax + 4; nlon=2*lmax + 1, include_poles=false)
        rng = MersenneTwister(308)
        Slm, Tlm = _rand_vec_alm(rng, lmax, lmax)

        Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        @test all(isfinite, Vt)
        @test all(isfinite, Vp)

        Slm_r, Tlm_r = analysis_sphtor(cfg, Vt, Vp)
        @test all(isfinite, Slm_r)
        @test all(isfinite, Tlm_r)
        # The dominant (high-l, high-m) modes are usually recovered to a few %
        @test isapprox(Slm_r[end, end], Slm[end, end]; rtol=1e-3, atol=1e-5)
        @test isapprox(Tlm_r[end, end], Tlm[end, end]; rtol=1e-3, atol=1e-5)
    end
end
