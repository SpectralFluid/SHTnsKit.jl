# SHTnsKit.jl - Basic Scalar Transform Tests
# Tests for analysis, synthesis, and SHTPlan

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

@testset "Basic Scalar Transforms" begin
    @testset "Real coefficient matrices" begin
        for T in (Float32, Float64, Int), norm in (:orthonormal, :schmidt)
            T === Int && norm !== :orthonormal && continue
            cfg = create_gauss_config(3, 5; norm)
            alm = zeros(T, cfg.lmax + 1, cfg.mmax + 1)
            alm[2, 1] = 1
            alm[3, 2] = 2
            alm[4, 3] = -1
            original = copy(alm)
            complex_alm = complex.(float.(alm))
            expected = synthesis(cfg, complex_alm)
            expected_cplx = synthesis_cplx(cfg, complex_alm)
            tolerance = T === Float32 ? 1e-5 : 1e-12

            @test synthesis(cfg, alm) ≈ expected rtol=tolerance
            @test synthesis_cplx(cfg, alm) ≈ expected_cplx rtol=tolerance
            @test synthesis(cfg, alm; real_output=false) ≈ expected_cplx rtol=tolerance
            for use_rfft in (false, true)
                @test synthesis(cfg, alm; use_rfft) ≈ expected rtol=tolerance
                out = similar(expected)
                @test synthesis!(cfg, out, alm; use_rfft) === out
                @test out ≈ expected rtol=tolerance
            end
            out_cplx = similar(expected_cplx)
            @test synthesis!(cfg, out_cplx, alm; real_output=false) === out_cplx
            @test out_cplx ≈ expected_cplx rtol=tolerance

            # QST degree truncation reuses the scalar synthesis helper.
            zero_coeffs = zeros(eltype(complex_alm), size(alm))
            expected_l = synthesis_qst_l(cfg, complex_alm, zero_coeffs, zero_coeffs, 2)[1]
            @test synthesis_qst_l(cfg, alm, zero_coeffs, zero_coeffs, 2)[1] ≈
                  expected_l rtol=tolerance
            @test alm == original
        end
    end

    @testset "Analysis-synthesis roundtrip" begin
        lmax = 10
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(42)

        # Create random spectral coefficients
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 must be real for real fields
        # Zero invalid entries (l < m)
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0.0
        end

        # Roundtrip: spectral -> spatial -> spectral
        f = synthesis(cfg, alm; real_output=true)
        alm_recovered = analysis(cfg, f)

        @test isapprox(alm_recovered, alm; rtol=1e-10, atol=1e-12)
        VERBOSE && @info "Scalar roundtrip" max_err=maximum(abs.(alm_recovered - alm))
    end

    @testset "SHTPlan optimized transforms" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(43)

        # Create plan
        plan = SHTPlan(cfg)

        # Start with spectral coefficients for reliable roundtrip
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 real
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # In-place synthesis
        f = zeros(nlat, nlon)
        synthesis!(plan, f, alm)

        # In-place analysis
        alm_back = zeros(ComplexF64, lmax+1, lmax+1)
        analysis!(plan, alm_back, f)

        # Compare recovered coefficients
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)

        # Also verify plan matches non-planned version
        alm_ref = analysis(cfg, f)
        @test isapprox(alm_back, alm_ref; rtol=1e-10, atol=1e-12)
    end

    @testset "Packed vector format" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(44)

        # Start with packed spectral coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])  # m=0 real (first lmax+1 indices)

        # Packed synthesis
        f = synthesis_packed(cfg, Qlm)
        @test length(f) == nlat * nlon

        # Packed analysis
        Qlm_back = analysis_packed(cfg, f)
        @test length(Qlm_back) == cfg.nlm

        # Verify roundtrip
        @test isapprox(Qlm_back, Qlm; rtol=1e-10, atol=1e-12)
    end

    @testset "Single spherical harmonic modes" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Test individual modes
        for l in 0:lmax
            for m in 0:l
                alm = zeros(ComplexF64, lmax+1, lmax+1)
                alm[l+1, m+1] = m == 0 ? 1.0 : 1.0 + 0.5im

                f = synthesis(cfg, alm; real_output=true)
                alm_rec = analysis(cfg, f)

                @test isapprox(alm_rec[l+1, m+1], alm[l+1, m+1]; rtol=1e-10, atol=1e-12)
            end
        end
    end

    @testset "Real field consistency" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(45)

        # Start with spectral coefficients, real for m=0
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])  # m=0 real
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Synthesize to real field
        f = synthesis(cfg, alm; real_output=true)

        # Analyze back
        alm_back = analysis(cfg, f)

        # m=0 coefficients should remain real
        @test all(abs.(imag.(alm_back[:, 1])) .< 1e-12)

        # Roundtrip should preserve coefficients
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "FFT scratch buffer reuse" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(47)

        # Preallocate scratch buffer
        scratch = zeros(ComplexF64, nlat, nlon)

        # Random spectral coefficients
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Synthesis with scratch buffer
        f_scratch = synthesis(cfg, alm; real_output=true, fft_scratch=scratch)
        f_no_scratch = synthesis(cfg, alm; real_output=true)
        @test isapprox(f_scratch, f_no_scratch; rtol=1e-12, atol=1e-14)

        # Analysis with scratch buffer
        alm_scratch = analysis(cfg, f_scratch; fft_scratch=scratch)
        alm_no_scratch = analysis(cfg, f_scratch)
        @test isapprox(alm_scratch, alm_no_scratch; rtol=1e-12, atol=1e-14)
    end

    @testset "Out-of-place synthesis inference" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(49)

        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        @inferred SHTnsKit.phi_inv_scale(cfg)
        @inferred synthesis(cfg, alm)

        f_kw = synthesis(cfg, alm; real_output=false)
        f_cplx = @inferred synthesis_cplx(cfg, alm)
        @test eltype(f_cplx) === ComplexF64
        @test isapprox(f_cplx, f_kw; rtol=0, atol=0)
    end

    @testset "PLM tables path" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1

        # Config with precomputed PLM tables (use prepare_plm_tables!)
        cfg_plm = create_gauss_config(lmax, nlat; nlon=nlon)
        prepare_plm_tables!(cfg_plm)  # Enable PLM tables

        cfg_otf = create_gauss_config(lmax, nlat; nlon=nlon)
        # Default is on-the-fly (no tables)

        @test cfg_plm.use_plm_tables == true
        @test cfg_otf.use_plm_tables == false

        rng = MersenneTwister(48)
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Both paths should give same results
        f_plm = synthesis(cfg_plm, alm; real_output=true)
        f_otf = synthesis(cfg_otf, alm; real_output=true)
        @test isapprox(f_plm, f_otf; rtol=1e-10, atol=1e-12)

        alm_plm = analysis(cfg_plm, f_plm)
        alm_otf = analysis(cfg_otf, f_otf)
        @test isapprox(alm_plm, alm_otf; rtol=1e-10, atol=1e-12)

        # Roundtrip should work with PLM tables
        @test isapprox(alm_plm, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "use_rfft=true matches complex path (scalar analysis/synthesis)" begin
        for (lmax, nlat, nlon) in ((6, 8, 13), (12, 14, 25), (8, 10, 20))
            cfg = create_gauss_config(lmax, nlat; nlon=nlon)
            rng = MersenneTwister(101 + lmax)

            # Random real-valued spatial field
            f = randn(rng, nlat, nlon)

            # Analysis: rfft path vs complex path
            alm_c = analysis(cfg, f)
            alm_r = analysis(cfg, f; use_rfft=true)
            @test isapprox(alm_c, alm_r; rtol=1e-10, atol=1e-12)

            # Synthesis: rfft path vs complex path (from same alm)
            f_c = synthesis(cfg, alm_c; real_output=true)
            f_r = synthesis(cfg, alm_c; real_output=true, use_rfft=true)
            @test isapprox(f_c, f_r; rtol=1e-10, atol=1e-12)

            # In-place variants mirror the out-of-place contract
            alm_out = zeros(ComplexF64, lmax+1, lmax+1)
            analysis!(cfg, alm_out, f; use_rfft=true)
            @test isapprox(alm_out, alm_r; rtol=1e-10, atol=1e-12)

            f_out = zeros(Float64, nlat, nlon)
            synthesis!(cfg, f_out, alm_c; real_output=true, use_rfft=true)
            @test isapprox(f_out, f_r; rtol=1e-10, atol=1e-12)

            rfft_scratch = zeros(ComplexF64, nlat, nlon ÷ 2 + 1)
            analysis!(cfg, alm_out, f; fft_scratch=rfft_scratch, use_rfft=true)
            @test isapprox(alm_out, alm_r; rtol=1e-10, atol=1e-12)
            synthesis!(cfg, f_out, alm_c; real_output=true, fft_scratch=rfft_scratch, use_rfft=true)
            @test isapprox(f_out, f_r; rtol=1e-10, atol=1e-12)

            analysis!(cfg, alm_out, f; fft_scratch=rfft_scratch, use_rfft=true)
            synthesis!(cfg, f_out, alm_c; real_output=true, fft_scratch=rfft_scratch, use_rfft=true)
            GC.gc()
            # Julia/FFTW patch versions can impose a small constant allocation
            # floor around plan execution. Keep the budget below scratch-sized
            # allocations while allowing the observed Julia 1.10 Linux floor.
            rfft_alloc_budget = 2_048 + _thread_alloc_slack()
            @test @allocated(analysis!(cfg, alm_out, f; fft_scratch=rfft_scratch, use_rfft=true)) <= rfft_alloc_budget
            @test @allocated(synthesis!(cfg, f_out, alm_c; real_output=true, fft_scratch=rfft_scratch, use_rfft=true)) <= rfft_alloc_budget

            # Round-trip via rfft path
            alm_rt = analysis(cfg, f_r; use_rfft=true)
            @test isapprox(alm_rt, alm_r; rtol=1e-10, atol=1e-12)

            # Error cases
            @test_throws ArgumentError analysis(cfg, complex.(f); use_rfft=true)
            @test_throws ArgumentError synthesis(cfg, alm_c; real_output=false, use_rfft=true)
            @test_throws ArgumentError synthesis(cfg, alm_c; fft_scratch=zeros(ComplexF64, nlat, nlon), use_rfft=true)
        end
    end

    @testset "Axisymmetric transforms (analysis_axisym/synthesis_axisym)" begin
        lmax = 10
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(104)

        # Create random m=0 only coefficients (axisymmetric field)
        Ql = complex.(randn(rng, lmax+1))

        # Axisymmetric synthesis: spectral -> latitude values
        f_lat = synthesis_axisym(cfg, Ql)

        @test length(f_lat) == nlat
        @test eltype(f_lat) <: Real
        @test all(isfinite, f_lat)

        # Axisymmetric analysis: latitude values -> spectral
        Ql_rec = analysis_axisym(cfg, f_lat)

        @test length(Ql_rec) == lmax + 1
        # m=0 coefficients should be real (imaginary part ~0)
        @test maximum(abs.(imag.(Ql_rec))) < 1e-10
        # The round trip is an IDENTITY, not a proportionality. This used to
        # divide out a fitted scale factor, which made the assertion blind to the
        # missing φ quadrature factor (cphi*nlon = 2π) in `analysis_axisym` — the
        # test passed while every returned coefficient was 1/2π too small. Assert
        # the absolute values so a revert is caught.
        @test isapprox(real.(Ql_rec), real.(Ql); rtol=1e-9, atol=1e-11)
        # And pin it to the full transform: axisym analysis must equal the m=0
        # column of `analysis` on the same field.
        f2d = repeat(f_lat, 1, cfg.nlon)
        @test isapprox(Ql_rec, analysis(cfg, f2d)[:, 1]; rtol=1e-9, atol=1e-11)
    end

    @testset "Axisymmetric truncated transforms (analysis_axisym_l/synthesis_axisym_l)" begin
        lmax = 10
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        ltr = lmax - 3
        rng = MersenneTwister(105)

        # Create random m=0 only coefficients
        Ql = complex.(randn(rng, lmax+1))
        # Zero high modes for reference
        Ql_z = copy(Ql)
        Ql_z[ltr+2:end] .= 0

        # Truncated axisymmetric synthesis should match full synthesis with zeroed high modes
        f_lat_l = synthesis_axisym_l(cfg, Ql, ltr)
        f_lat_ref = synthesis_axisym(cfg, Ql_z)

        @test length(f_lat_l) == nlat
        @test isapprox(f_lat_l, f_lat_ref; rtol=1e-10, atol=1e-12)

        # Truncated analysis should match full analysis up to ltr
        Ql_rec_l = analysis_axisym_l(cfg, f_lat_l, ltr)
        Ql_rec_full = analysis_axisym(cfg, f_lat_l)

        @test length(Ql_rec_l) == ltr + 1
        @test isapprox(Ql_rec_l, Ql_rec_full[1:ltr+1]; rtol=1e-10, atol=1e-12)
    end

end
