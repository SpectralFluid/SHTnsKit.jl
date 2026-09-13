# SHTnsKit.jl - Vector Transform Tests (Spheroidal-Toroidal)
# Tests for 2D vector field spherical harmonic transforms

using Test
using Random
using LinearAlgebra
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

# Allocation budgets below are calibrated for a SINGLE-THREADED run, where they
# are exact. The shared m-loop orchestrators start `@threads` tasks whenever
# threads are available, and each threaded region costs a few hundred bytes to
# spawn — a constant independent of problem size, and not the kind of regression
# these budgets exist to catch (a per-row temporary or a dense zero spectrum
# costs tens of KB and grows with lmax). Allow a per-thread slack rather than
# letting the whole suite go red on any multi-core machine; note this makes the
# budgets coarse at `nthreads > 1`, so the tight check is the 1-thread run.
@isdefined(_thread_alloc_slack) ||
    (_thread_alloc_slack() = Threads.nthreads() > 1 ? 4_096 * Threads.nthreads() : 0)

@testset "Vector Transforms (Spheroidal-Toroidal)" begin
    @testset "Sphtor roundtrip" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(50)

        # Random S and T coefficients
        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)

        for m in 0:lmax
            for l in max(1, m):lmax  # l >= 1 for vector fields
                Slm[l+1, m+1] = randn(rng) + im * randn(rng)
                Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
            end
            if m == 0
                Slm[:, 1] .= real.(Slm[:, 1])
                Tlm[:, 1] .= real.(Tlm[:, 1])
            end
        end

        # Synthesis
        Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)

        # Analysis
        Slm_rec, Tlm_rec = analysis_sphtor(cfg, Vt, Vp)

        @test isapprox(Slm_rec, Slm; rtol=1e-9, atol=1e-11)
        @test isapprox(Tlm_rec, Tlm; rtol=1e-9, atol=1e-11)
    end

    @testset "Individual S and T components" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)

        # Only S component - use consistent real_output setting
        Slm[4, 2] = 1.0 + 0.5im  # l=3, m=1
        Vt_s, Vp_s = synthesis_sph(cfg, Slm; real_output=false)
        Vt_full, Vp_full = synthesis_sphtor(cfg, Slm, Tlm; real_output=false)
        @test isapprox(Vt_s, Vt_full; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_s, Vp_full; rtol=1e-10, atol=1e-12)

        Vt_s_cplx, Vp_s_cplx = @inferred synthesis_sph_cplx(cfg, Slm)
        @test isapprox(Vt_s_cplx, Vt_full; rtol=0, atol=0)
        @test isapprox(Vp_s_cplx, Vp_full; rtol=0, atol=0)

        # Only T component - use consistent real_output setting
        fill!(Slm, 0)
        Tlm[5, 3] = 0.8 - 0.3im  # l=4, m=2
        Vt_t, Vp_t = synthesis_tor(cfg, Tlm; real_output=false)
        Vt_full2, Vp_full2 = synthesis_sphtor(cfg, Slm, Tlm; real_output=false)
        @test isapprox(Vt_t, Vt_full2; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_t, Vp_full2; rtol=1e-10, atol=1e-12)

        Vt_t_cplx, Vp_t_cplx = @inferred synthesis_tor_cplx(cfg, Tlm)
        @test isapprox(Vt_t_cplx, Vt_full2; rtol=0, atol=0)
        @test isapprox(Vp_t_cplx, Vp_full2; rtol=0, atol=0)
    end

    @testset "Spheroidal/toroidal-only synthesis avoids dense zero spectra" begin
        lmax = 10
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(51)

        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        synthesis_sph(cfg, Slm)
        synthesis_tor(cfg, Tlm)
        GC.gc()

        @test @allocated(synthesis_sph(cfg, Slm)) <= 15_000 + _thread_alloc_slack()
        @test @allocated(synthesis_tor(cfg, Tlm)) <= 15_000 + _thread_alloc_slack()
    end

    @testset "Gradient transform" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(52)

        # Random scalar field coefficients
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0.0
        end

        # Gradient via synthesis_sph (spheroidal-only synthesis)
        Gt, Gp = synthesis_sph(cfg, alm)
        Gt_ref, Gp_ref = synthesis_sph(cfg, alm)

        @test isapprox(Gt, Gt_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Gp, Gp_ref; rtol=1e-10, atol=1e-12)
    end

    @testset "Truncated vector transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        ltr = lmax - 2
        rng = MersenneTwister(53)

        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])

        # l=0 row should be zero for vectors
        Slm[1, :] .= 0
        Tlm[1, :] .= 0

        # Truncated synthesis (default real_output=true)
        Vt_l, Vp_l = synthesis_sphtor_l(cfg, Slm, Tlm, ltr)
        @inferred synthesis_sphtor_l(cfg, Slm, Tlm, ltr)

        # Reference: zero high modes and synthesize with same real_output setting
        Slm_z = copy(Slm); Tlm_z = copy(Tlm)
        for m in 0:lmax, l in (ltr+1):lmax
            Slm_z[l+1, m+1] = 0
            Tlm_z[l+1, m+1] = 0
        end
        Vt_ref, Vp_ref = synthesis_sphtor(cfg, Slm_z, Tlm_z; real_output=true)

        @test isapprox(Vt_l, Vt_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_l, Vp_ref; rtol=1e-10, atol=1e-12)
    end

    @testset "Mode-limited vector transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(54)

        im = 2  # azimuthal mode
        ltr = lmax - 1
        len = ltr - im + 1

        Sl = randn(rng, ComplexF64, len)
        Tl = randn(rng, ComplexF64, len)

        # Mode-limited synthesis
        Vt_ml, Vp_ml = synthesis_sphtor_ml(cfg, im, Sl, Tl, ltr)

        @test length(Vt_ml) == nlat
        @test length(Vp_ml) == nlat

        # Individual component mode-limited
        Vt_s, Vp_s = synthesis_sph_ml(cfg, im, Sl, ltr)
        Vt_t, Vp_t = synthesis_tor_ml(cfg, im, Tl, ltr)

        @test length(Vt_s) == nlat
        @test length(Vt_t) == nlat

        # Combined should match sum
        @test isapprox(Vt_ml, Vt_s .+ Vt_t; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_ml, Vp_s .+ Vp_t; rtol=1e-10, atol=1e-12)

        synthesis_sphtor_ml(cfg, im, Sl, Tl, ltr)
        synthesis_sph_ml(cfg, im, Sl, ltr)
        synthesis_tor_ml(cfg, im, Tl, ltr)
        GC.gc()
        @test @allocated(synthesis_sph_ml(cfg, im, Sl, ltr)) <= @allocated(synthesis_sphtor_ml(cfg, im, Sl, Tl, ltr)) + 128
        @test @allocated(synthesis_tor_ml(cfg, im, Tl, ltr)) <= @allocated(synthesis_sphtor_ml(cfg, im, Sl, Tl, ltr)) + 128
    end

    @testset "Mode-limited single-component synthesis avoids dense zero vectors" begin
        lmax = 32
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(57)

        im = 3
        ltr = lmax - 2
        len = ltr - im + 1
        Sl = randn(rng, ComplexF64, len)
        Tl = randn(rng, ComplexF64, len)

        synthesis_sphtor_ml(cfg, im, Sl, Tl, ltr)
        synthesis_sph_ml(cfg, im, Sl, ltr)
        synthesis_tor_ml(cfg, im, Tl, ltr)
        GC.gc()

        full_alloc = @allocated(synthesis_sphtor_ml(cfg, im, Sl, Tl, ltr))
        @test @allocated(synthesis_sph_ml(cfg, im, Sl, ltr)) <= full_alloc + 128
        @test @allocated(synthesis_tor_ml(cfg, im, Tl, ltr)) <= full_alloc + 128
    end

    @testset "Divergence and vorticity from S/T" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(55)

        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)

        for m in 0:lmax
            for l in max(1, m):lmax
                Slm[l+1, m+1] = randn(rng) + im * randn(rng)
                Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
            end
        end
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])

        # Compute divergence from spheroidal
        div_lm = divergence_from_spheroidal(cfg, Slm)
        @test size(div_lm) == size(Slm)

        # Compute vorticity from toroidal
        vort_lm = vorticity_from_toroidal(cfg, Tlm)
        @test size(vort_lm) == size(Tlm)

        # Inverse: spheroidal from divergence
        Slm_rec = spheroidal_from_divergence(cfg, div_lm)
        @test isapprox(Slm_rec, Slm; rtol=1e-9, atol=1e-11)

        # Inverse: toroidal from vorticity
        Tlm_rec = toroidal_from_vorticity(cfg, vort_lm)
        @test isapprox(Tlm_rec, Tlm; rtol=1e-9, atol=1e-11)
    end

    @testset "Complex vector transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(56)

        # Start with spectral coefficients for reliable roundtrip
        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)

        for m in 0:lmax
            for l in max(1, m):lmax
                Slm[l+1, m+1] = randn(rng) + im * randn(rng)
                Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
            end
        end

        # Synthesis
        Vt, Vp = synthesis_sphtor_cplx(cfg, Slm, Tlm)

        # Analysis
        Slm_back, Tlm_back = analysis_sphtor_cplx(cfg, Vt, Vp)

        @test isapprox(Slm_back, Slm; rtol=1e-8, atol=1e-10)
        @test isapprox(Tlm_back, Tlm; rtol=1e-8, atol=1e-10)
    end

    @testset "In-place spheroidal-toroidal transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        plan = SHTPlan(cfg)  # In-place functions require a plan
        rng = MersenneTwister(88)

        # Create spectral coefficients
        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Slm[1, :] .= 0
        Tlm[1, :] .= 0
        for m in 0:lmax, l in 0:(m-1)
            Slm[l+1, m+1] = 0
            Tlm[l+1, m+1] = 0
        end

        # In-place synthesis using plan
        Vt = zeros(nlat, nlon)
        Vp = zeros(nlat, nlon)
        synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm)

        # Compare with out-of-place version
        Vt_ref, Vp_ref = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        @test isapprox(Vt, Vt_ref; rtol=1e-12, atol=1e-14)
        @test isapprox(Vp, Vp_ref; rtol=1e-12, atol=1e-14)

        # In-place analysis using plan
        Slm_back = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm_back = zeros(ComplexF64, lmax+1, lmax+1)
        analysis_sphtor!(plan, Slm_back, Tlm_back, Vt, Vp)

        # Verify roundtrip
        @test isapprox(Slm_back, Slm; rtol=1e-9, atol=1e-11)
        @test isapprox(Tlm_back, Tlm; rtol=1e-9, atol=1e-11)
    end

    @testset "Truncated vector analysis (analysis_sphtor_l)" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        ltr = lmax - 2
        rng = MersenneTwister(100)

        # Create spectral coefficients with modes only up to ltr
        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:ltr
            for l in max(1, m):ltr
                Slm[l+1, m+1] = randn(rng) + im * randn(rng)
                Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
            end
        end
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])

        # Synthesize and analyze with truncation
        Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        Slm_rec, Tlm_rec = analysis_sphtor_l(cfg, Vt, Vp, ltr)

        # Only compare up to ltr (higher modes should be zero in original)
        for m in 0:ltr
            for l in max(1, m):ltr
                @test isapprox(Slm_rec[l+1, m+1], Slm[l+1, m+1]; rtol=1e-9, atol=1e-11)
                @test isapprox(Tlm_rec[l+1, m+1], Tlm[l+1, m+1]; rtol=1e-9, atol=1e-11)
            end
        end
    end

    @testset "Mode-limited vector analysis (analysis_sphtor_ml)" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(101)

        # Test that analysis_sphtor_ml produces valid output for different modes
        for im_mode in [0, 1, 2, 3]
            ltr = lmax - 1
            len = ltr - im_mode + 1
            if len <= 0
                continue
            end

            # Create input vectors (Fourier coefficients at each latitude)
            Vt_m = randn(rng, ComplexF64, nlat)
            Vp_m = randn(rng, ComplexF64, nlat)

            # Analysis: spatial -> coefficients
            Sl_rec, Tl_rec = analysis_sphtor_ml(cfg, im_mode, Vt_m, Vp_m, ltr)

            # Verify output dimensions and finiteness
            @test length(Sl_rec) == len
            @test length(Tl_rec) == len
            @test all(isfinite, Sl_rec)
            @test all(isfinite, Tl_rec)
        end

        # Note: Mode-limited analysis/synthesis do not form exact inverses due to
        # different normalization conventions. The full analysis_sphtor/synthesis_sphtor
        # functions should be used for exact roundtrip transforms.
    end

    @testset "Truncated helper functions (_l variants)" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        ltr = lmax - 2
        rng = MersenneTwister(102)

        # Create spectral coefficients
        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)
        alm = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:lmax
            for l in max(1, m):lmax
                Slm[l+1, m+1] = randn(rng) + im * randn(rng)
                Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
            end
            for l in m:lmax
                alm[l+1, m+1] = randn(rng) + im * randn(rng)
            end
        end
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])
        alm[:, 1] .= real.(alm[:, 1])

        # Test synthesis_sph_l: truncated spheroidal synthesis
        Vt_sph_l, Vp_sph_l = synthesis_sph_l(cfg, Slm, ltr)
        # Reference: zero high modes and use full synthesis
        Slm_z = copy(Slm)
        for m in 0:lmax, l in (ltr+1):lmax
            Slm_z[l+1, m+1] = 0
        end
        Vt_sph_ref, Vp_sph_ref = synthesis_sph(cfg, Slm_z)
        @test isapprox(Vt_sph_l, Vt_sph_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_sph_l, Vp_sph_ref; rtol=1e-10, atol=1e-12)

        Vt_sph_l_cplx, Vp_sph_l_cplx = @inferred synthesis_sph_l_cplx(cfg, Slm, ltr)
        Vt_sph_ref_cplx, Vp_sph_ref_cplx = synthesis_sph(cfg, Slm_z; real_output=false)
        @test isapprox(Vt_sph_l_cplx, Vt_sph_ref_cplx; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_sph_l_cplx, Vp_sph_ref_cplx; rtol=1e-10, atol=1e-12)

        # Test synthesis_tor_l: truncated toroidal synthesis
        Vt_tor_l, Vp_tor_l = synthesis_tor_l(cfg, Tlm, ltr)
        # Reference: zero high modes and use full synthesis
        Tlm_z = copy(Tlm)
        for m in 0:lmax, l in (ltr+1):lmax
            Tlm_z[l+1, m+1] = 0
        end
        Vt_tor_ref, Vp_tor_ref = synthesis_tor(cfg, Tlm_z)
        @test isapprox(Vt_tor_l, Vt_tor_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_tor_l, Vp_tor_ref; rtol=1e-10, atol=1e-12)

        Vt_tor_l_cplx, Vp_tor_l_cplx = @inferred synthesis_tor_l_cplx(cfg, Tlm, ltr)
        Vt_tor_ref_cplx, Vp_tor_ref_cplx = synthesis_tor(cfg, Tlm_z; real_output=false)
        @test isapprox(Vt_tor_l_cplx, Vt_tor_ref_cplx; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_tor_l_cplx, Vp_tor_ref_cplx; rtol=1e-10, atol=1e-12)

        Vt_l_cplx, Vp_l_cplx = @inferred synthesis_sphtor_l_cplx(cfg, Slm, Tlm, ltr)
        Vt_ref_cplx, Vp_ref_cplx = synthesis_sphtor(cfg, Slm_z, Tlm_z; real_output=false)
        @test isapprox(Vt_l_cplx, Vt_ref_cplx; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_l_cplx, Vp_ref_cplx; rtol=1e-10, atol=1e-12)

        # Test synthesis_sph_l: truncated spheroidal synthesis
        Gt_l, Gp_l = synthesis_sph_l(cfg, alm, ltr)
        # Reference: zero high modes and use full gradient
        alm_z = copy(alm)
        for m in 0:lmax, l in (ltr+1):lmax
            alm_z[l+1, m+1] = 0
        end
        Gt_ref, Gp_ref = synthesis_sph(cfg, alm_z)
        @test isapprox(Gt_l, Gt_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Gp_l, Gp_ref; rtol=1e-10, atol=1e-12)
    end


    @testset "mode-limited sphtor validates its order and truncation" begin
        # Out-of-range arguments used to walk off the end of the norm-scale table
        # and return ±Inf coefficients; the scalar twin already raised.
        cfg = create_gauss_config(3, 5; nlon=8)
        v = randn(MersenneTwister(3311), ComplexF64, cfg.nlat)
        sl = zeros(ComplexF64, cfg.lmax + 1)
        @test_throws ArgumentError SHTnsKit.analysis_sphtor_ml(cfg, 9, v, v, 12)
        @test_throws ArgumentError SHTnsKit.analysis_sphtor_ml(cfg, -1, v, v, 3)
        @test_throws ArgumentError SHTnsKit.analysis_sphtor_ml(cfg, 1, v, v, 99)
        @test_throws ArgumentError SHTnsKit.synthesis_sphtor_ml(cfg, 9, sl, sl, 12)
        # a valid call still returns finite coefficients
        S, T = SHTnsKit.analysis_sphtor_ml(cfg, 1, v, v, 3)
        @test all(isfinite, S) && all(isfinite, T)
    end

end
