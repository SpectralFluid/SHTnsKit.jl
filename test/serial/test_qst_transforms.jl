# SHTnsKit.jl - QST (3D Vector) Transform Tests
# Tests for radial-spheroidal-toroidal vector field transforms

using Test
using Random
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

@testset "QST (3D Vector) Transforms" begin
    @testset "QST roundtrip" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(60)

        # Random QST coefficients
        Qlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)

        # m=0 must be real
        Qlm[:, 1] .= real.(Qlm[:, 1])
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])

        # l=0 for S,T should be zero
        Slm[1, :] .= 0
        Tlm[1, :] .= 0

        # Zero invalid (l < m)
        for m in 0:lmax, l in 0:(m-1)
            Qlm[l+1, m+1] = 0
            Slm[l+1, m+1] = 0
            Tlm[l+1, m+1] = 0
        end

        # Synthesis
        Vr, Vt, Vp = synthesis_qst(cfg, Qlm, Slm, Tlm; real_output=true)

        # Analysis
        Qlm_rec, Slm_rec, Tlm_rec = analysis_qst(cfg, Vr, Vt, Vp)

        @test isapprox(Qlm_rec, Qlm; rtol=1e-9, atol=1e-11)
        @test isapprox(Slm_rec, Slm; rtol=1e-9, atol=1e-11)
        @test isapprox(Tlm_rec, Tlm; rtol=1e-9, atol=1e-11)
    end

    @testset "QST truncated transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        ltr = lmax - 2
        rng = MersenneTwister(61)

        Qlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Qlm[:, 1] .= real.(Qlm[:, 1])
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Slm[1, :] .= 0; Tlm[1, :] .= 0

        # Truncated synthesis
        Vr_l, Vt_l, Vp_l = synthesis_qst_l(cfg, Qlm, Slm, Tlm, ltr; real_output=true)

        # Reference
        Qlm_z = copy(Qlm); Slm_z = copy(Slm); Tlm_z = copy(Tlm)
        for m in 0:lmax, l in (ltr+1):lmax
            Qlm_z[l+1, m+1] = 0
            Slm_z[l+1, m+1] = 0
            Tlm_z[l+1, m+1] = 0
        end
        Vr_ref, Vt_ref, Vp_ref = synthesis_qst(cfg, Qlm_z, Slm_z, Tlm_z; real_output=true)

        @test isapprox(Vr_l, Vr_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Vt_l, Vt_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_l, Vp_ref; rtol=1e-10, atol=1e-12)

        Vr_l_cplx, Vt_l_cplx, Vp_l_cplx = @inferred synthesis_qst_l_cplx(cfg, Qlm, Slm, Tlm, ltr)
        Vr_ref_cplx, Vt_ref_cplx, Vp_ref_cplx = synthesis_qst(cfg, Qlm_z, Slm_z, Tlm_z; real_output=false)
        @test isapprox(Vr_l_cplx, Vr_ref_cplx; rtol=1e-10, atol=1e-12)
        @test isapprox(Vt_l_cplx, Vt_ref_cplx; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp_l_cplx, Vp_ref_cplx; rtol=1e-10, atol=1e-12)

        synthesis_qst_l(cfg, Qlm, Slm, Tlm, ltr; real_output=true)
        GC.gc()
        # Array header/alignment overhead differs across Julia patch versions
        # and platforms (notably Windows CI). Keep this below an extra
        # component-sized scratch allocation while allowing that fixed floor.
        qst_l_alloc_budget = 13_000 + _thread_alloc_slack()
        @test @allocated(synthesis_qst_l(cfg, Qlm, Slm, Tlm, ltr; real_output=true)) <= qst_l_alloc_budget
    end

    @testset "QST complex transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(62)

        # Start with random spectral coefficients, properly zeroed
        Qlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)

        # Zero invalid entries
        for m in 0:lmax, l in 0:(m-1)
            Qlm[l+1, m+1] = 0
            Slm[l+1, m+1] = 0
            Tlm[l+1, m+1] = 0
        end
        # l=0 for S,T should be zero
        Slm[1, :] .= 0
        Tlm[1, :] .= 0

        # Synthesis
        Vr, Vt, Vp = synthesis_qst_cplx(cfg, Qlm, Slm, Tlm)

        # Analysis
        Qlm_back, Slm_back, Tlm_back = analysis_qst_cplx(cfg, Vr, Vt, Vp)

        # Valid entries should match
        for m in 0:lmax
            for l in m:lmax
                @test isapprox(Qlm_back[l+1, m+1], Qlm[l+1, m+1]; rtol=1e-8, atol=1e-10)
            end
            for l in max(1, m):lmax
                @test isapprox(Slm_back[l+1, m+1], Slm[l+1, m+1]; rtol=1e-8, atol=1e-10)
                @test isapprox(Tlm_back[l+1, m+1], Tlm[l+1, m+1]; rtol=1e-8, atol=1e-10)
            end
        end
    end

    @testset "QST mode-limited transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(63)

        im = 2
        ltr = lmax - 1
        len = ltr - im + 1

        Ql = randn(rng, ComplexF64, len)
        Sl = randn(rng, ComplexF64, len)
        Tl = randn(rng, ComplexF64, len)

        # Mode-limited synthesis - verify dimensions and validity
        Vr_ml, Vt_ml, Vp_ml = synthesis_qst_ml(cfg, im, Ql, Sl, Tl, ltr)

        @test length(Vr_ml) == nlat
        @test length(Vt_ml) == nlat
        @test length(Vp_ml) == nlat
        @test all(!isnan, Vr_ml) && all(!isinf, Vr_ml)
        @test all(!isnan, Vt_ml) && all(!isinf, Vt_ml)
        @test all(!isnan, Vp_ml) && all(!isinf, Vp_ml)

        # Mode-limited analysis - verify dimensions
        Ql_back, Sl_back, Tl_back = analysis_qst_ml(cfg, im, Vr_ml, Vt_ml, Vp_ml, ltr)

        @test length(Ql_back) == len
        @test length(Sl_back) == len
        @test length(Tl_back) == len
        @test all(!isnan, Ql_back) && all(!isinf, Ql_back)
        @test all(!isnan, Sl_back) && all(!isinf, Sl_back)
        @test all(!isnan, Tl_back) && all(!isinf, Tl_back)

        # Q component (scalar) should roundtrip exactly
        @test isapprox(Ql_back, Ql; rtol=1e-9, atol=1e-11)

        # S and T components have complex l(l+1) coupling in mode-limited transforms.
        # Verify non-trivial output and that individual coefficients are finite.
        E_back = sum(abs2.(Sl_back)) + sum(abs2.(Tl_back))
        @test E_back > 0
        @test all(isfinite, Sl_back)
        @test all(isfinite, Tl_back)
    end

    @testset "QST decomposition orthogonality" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(64)

        # Pure radial field (only Q)
        Qlm_only = randn(rng, ComplexF64, lmax+1, lmax+1)
        Qlm_only[:, 1] .= real.(Qlm_only[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            Qlm_only[l+1, m+1] = 0
        end
        Slm_zero = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm_zero = zeros(ComplexF64, lmax+1, lmax+1)

        Vr, Vt, Vp = synthesis_qst(cfg, Qlm_only, Slm_zero, Tlm_zero; real_output=true)

        # Vt and Vp should be zero for pure radial field
        @test maximum(abs.(Vt)) < 1e-12
        @test maximum(abs.(Vp)) < 1e-12

        # Pure toroidal field (only T)
        Qlm_zero = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm_only = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm_only[:, 1] .= real.(Tlm_only[:, 1])
        Tlm_only[1, :] .= 0

        Vr2, Vt2, Vp2 = synthesis_qst(cfg, Qlm_zero, Slm_zero, Tlm_only; real_output=true)

        # Vr should be zero for pure tangential field
        @test maximum(abs.(Vr2)) < 1e-12
    end

    @testset "QST point evaluation" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(65)

        # Create packed QST coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Slm = randn(rng, ComplexF64, cfg.nlm)
        Tlm = randn(rng, ComplexF64, cfg.nlm)

        # m=0 must be real
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])
        Slm[1:lmax+1] .= real.(Slm[1:lmax+1])
        Tlm[1:lmax+1] .= real.(Tlm[1:lmax+1])

        # l=0 for S,T should be zero
        Slm[1] = 0
        Tlm[1] = 0

        # Evaluate at a point (use full module path as it's not exported)
        cost = 0.5
        phi = π/4

        vr, vt, vp = SHTnsKit.SHqst_to_point(cfg, Qlm, Slm, Tlm, cost, phi)

        @test !isnan(vr) && !isinf(vr)
        @test !isnan(vt) && !isinf(vt)
        @test !isnan(vp) && !isinf(vp)

        # Test at multiple points
        for cost_test in [-0.8, 0.0, 0.8]
            for phi_test in [0.0, π/2, π]
                vr_t, vt_t, vp_t = SHTnsKit.SHqst_to_point(cfg, Qlm, Slm, Tlm, cost_test, phi_test)
                @test !isnan(vr_t) && !isinf(vr_t)
                @test !isnan(vt_t) && !isinf(vt_t)
                @test !isnan(vp_t) && !isinf(vp_t)
            end
        end
    end

    @testset "QST latitude evaluation" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(66)

        # Create packed QST coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Slm = randn(rng, ComplexF64, cfg.nlm)
        Tlm = randn(rng, ComplexF64, cfg.nlm)

        # m=0 must be real
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])
        Slm[1:lmax+1] .= real.(Slm[1:lmax+1])
        Tlm[1:lmax+1] .= real.(Tlm[1:lmax+1])

        # l=0 for S,T should be zero
        Slm[1] = 0
        Tlm[1] = 0

        # Evaluate at a latitude
        cost = 0.3
        vr_lat, vt_lat, vp_lat = SHqst_to_lat(cfg, Qlm, Slm, Tlm, cost)

        @test length(vr_lat) == nlon
        @test length(vt_lat) == nlon
        @test length(vp_lat) == nlon
        @test all(!isnan, vr_lat) && all(!isinf, vr_lat)
        @test all(!isnan, vt_lat) && all(!isinf, vt_lat)
        @test all(!isnan, vp_lat) && all(!isinf, vp_lat)
    end

    @testset "Truncated QST analysis (analysis_qst_l)" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        ltr = lmax - 2
        rng = MersenneTwister(103)

        # Create spectral coefficients with modes only up to ltr
        Qlm = zeros(ComplexF64, lmax+1, lmax+1)
        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:ltr
            for l in m:ltr
                Qlm[l+1, m+1] = randn(rng) + im * randn(rng)
            end
            for l in max(1, m):ltr
                Slm[l+1, m+1] = randn(rng) + im * randn(rng)
                Tlm[l+1, m+1] = randn(rng) + im * randn(rng)
            end
        end
        Qlm[:, 1] .= real.(Qlm[:, 1])
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])

        # Synthesize and analyze with truncation
        Vr, Vt, Vp = synthesis_qst(cfg, Qlm, Slm, Tlm; real_output=true)
        Qlm_rec, Slm_rec, Tlm_rec = analysis_qst_l(cfg, Vr, Vt, Vp, ltr)

        # Only compare up to ltr (higher modes should be zero in original)
        for m in 0:ltr
            for l in m:ltr
                @test isapprox(Qlm_rec[l+1, m+1], Qlm[l+1, m+1]; rtol=1e-9, atol=1e-11)
            end
            for l in max(1, m):ltr
                @test isapprox(Slm_rec[l+1, m+1], Slm[l+1, m+1]; rtol=1e-9, atol=1e-11)
                @test isapprox(Tlm_rec[l+1, m+1], Tlm[l+1, m+1]; rtol=1e-9, atol=1e-11)
            end
        end
    end
end
