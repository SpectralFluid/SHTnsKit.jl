# SHTnsKit.jl - Batch Transform Tests
# Tests for multi-field batch transforms

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

@testset "Batch Transforms" begin
    @testset "Scalar batch analysis" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(70)

        nfields = 5

        # Multiple fields as 3D array
        fields = randn(rng, nlat, nlon, nfields)

        # Batch analysis
        alm_batch = analysis_batch(cfg, fields)
        @test size(alm_batch) == (lmax+1, lmax+1, nfields)

        # Verify matches individual analysis
        for k in 1:nfields
            alm_single = analysis(cfg, fields[:, :, k])
            @test isapprox(alm_batch[:, :, k], alm_single; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "Scalar batch synthesis" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(71)

        nfields = 4

        # Random coefficients
        alm_batch = randn(rng, ComplexF64, lmax+1, lmax+1, nfields)
        for k in 1:nfields
            alm_batch[:, 1, k] .= real.(alm_batch[:, 1, k])
            for m in 0:lmax, l in 0:(m-1)
                alm_batch[l+1, m+1, k] = 0
            end
        end

        # Batch synthesis
        fields = synthesis_batch(cfg, alm_batch)
        @test size(fields) == (nlat, nlon, nfields)

        # Verify matches individual synthesis
        for k in 1:nfields
            f_single = synthesis(cfg, alm_batch[:, :, k]; real_output=true)
            @test isapprox(fields[:, :, k], f_single; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "Scalar batch roundtrip" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(72)

        nfields = 3

        # Start with spectral coefficients
        alm_batch = randn(rng, ComplexF64, lmax+1, lmax+1, nfields)
        for k in 1:nfields
            alm_batch[:, 1, k] .= real.(alm_batch[:, 1, k])
            for m in 0:lmax, l in 0:(m-1)
                alm_batch[l+1, m+1, k] = 0
            end
        end

        # Synth then analysis
        fields = synthesis_batch(cfg, alm_batch)
        alm_back = analysis_batch(cfg, fields)

        @test isapprox(alm_back, alm_batch; rtol=1e-10, atol=1e-12)
    end

    @testset "Scalar batch in-place scratch keeps allocations bounded" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(75)
        nfields = 4

        fields = randn(rng, nlat, nlon, nfields)
        alm_batch = zeros(ComplexF64, lmax + 1, lmax + 1, nfields)
        fft_batch = zeros(ComplexF64, nlat, nlon, nfields)
        rfft_batch = zeros(ComplexF64, nlat, nlon ÷ 2 + 1, nfields)
        fields_out = zeros(Float64, nlat, nlon, nfields)

        analysis_batch!(cfg, alm_batch, fields; fft_batch=fft_batch)
        synthesis_batch!(cfg, fields_out, alm_batch; fft_batch=fft_batch)
        analysis_batch!(cfg, alm_batch, fields; fft_batch=rfft_batch, use_rfft=true)
        synthesis_batch!(cfg, fields_out, alm_batch; fft_batch=rfft_batch, use_rfft=true)
        GC.gc()

        @test @allocated(analysis_batch!(cfg, alm_batch, fields; fft_batch=fft_batch)) <= 8_000 + _thread_alloc_slack()
        @test @allocated(synthesis_batch!(cfg, fields_out, alm_batch; fft_batch=fft_batch)) <= 8_000 + _thread_alloc_slack()
        @test @allocated(analysis_batch!(cfg, alm_batch, fields; fft_batch=rfft_batch, use_rfft=true)) <= 8_000 + _thread_alloc_slack()
        @test @allocated(synthesis_batch!(cfg, fields_out, alm_batch; fft_batch=rfft_batch, use_rfft=true)) <= 8_000 + _thread_alloc_slack()
        @inferred synthesis_batch(cfg, alm_batch)

        fields_kw = synthesis_batch(cfg, alm_batch; real_output=false)
        fields_cplx = @inferred synthesis_batch_cplx(cfg, alm_batch)
        @test eltype(fields_cplx) === ComplexF64
        @test isapprox(fields_cplx, fields_kw; rtol=0, atol=0)
    end

    @testset "Noncanonical batch in-place conversion does not allocate a copy" begin
        lmax = 32
        nlat = lmax + 2
        nlon = 2lmax + 1
        nfields = 2
        canonical = create_gauss_config(lmax, nlat; nlon)
        configured = create_gauss_config(lmax, nlat; nlon, norm=:schmidt,
                                         real_norm=true, cs_phase=false)
        rng = MersenneTwister(76)
        alm = randn(rng, ComplexF64, lmax + 1, lmax + 1, nfields)
        fields = zeros(Float64, nlat, nlon, nfields)
        fft_batch = zeros(ComplexF64, nlat, nlon, nfields)
        rfft_batch = zeros(ComplexF64, nlat, nlon ÷ 2 + 1, nfields)

        for cfg in (canonical, configured)
            synthesis_batch!(cfg, fields, alm; fft_batch)
            synthesis_batch!(cfg, fields, alm; fft_batch=rfft_batch, use_rfft=true)
        end
        GC.gc()

        canonical_alloc = @allocated synthesis_batch!(canonical, fields, alm; fft_batch)
        configured_alloc = @allocated synthesis_batch!(configured, fields, alm; fft_batch)
        @test configured_alloc <= canonical_alloc + 512

        canonical_ralloc = @allocated synthesis_batch!(canonical, fields, alm;
                                                       fft_batch=rfft_batch, use_rfft=true)
        configured_ralloc = @allocated synthesis_batch!(configured, fields, alm;
                                                        fft_batch=rfft_batch, use_rfft=true)
        @test configured_ralloc <= canonical_ralloc + 512
    end

    @testset "Batch size configuration" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Test batch size controls
        original_size = get_batch_size(cfg)

        set_batch_size!(cfg, 4)
        @test get_batch_size(cfg) == 4

        set_batch_size!(cfg, 8)
        @test get_batch_size(cfg) == 8

        set_batch_size!(cfg, 16)
        @test get_batch_size(cfg) == 16

        reset_batch_size!(cfg)
        @test get_batch_size(cfg) == original_size
    end

    @testset "Vector batch transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(73)

        nfields = 3

        # Test individual transforms with spectral-to-spatial-to-spectral roundtrip
        Slm_batch = zeros(ComplexF64, lmax+1, lmax+1, nfields)
        Tlm_batch = zeros(ComplexF64, lmax+1, lmax+1, nfields)
        Vt_batch = randn(rng, nlat, nlon, nfields)
        Vp_batch = randn(rng, nlat, nlon, nfields)

        for k in 1:nfields
            Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Slm[:, 1] .= real.(Slm[:, 1])
            Tlm[:, 1] .= real.(Tlm[:, 1])
            Slm[1, :] .= 0  # l=0 must be zero for vectors
            Tlm[1, :] .= 0
            # Zero invalid (l < m) positions
            for m in 0:lmax, l in 0:(m-1)
                Slm[l+1, m+1] = 0
                Tlm[l+1, m+1] = 0
            end
            Slm_batch[:, :, k] .= Slm
            Tlm_batch[:, :, k] .= Tlm

            Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
            @test size(Vt) == (nlat, nlon)
            @test size(Vp) == (nlat, nlon)

            Slm_back, Tlm_back = analysis_sphtor(cfg, Vt, Vp)
            @test size(Slm_back) == (lmax+1, lmax+1)
            @test size(Tlm_back) == (lmax+1, lmax+1)

            # Verify roundtrip
            @test isapprox(Slm_back, Slm; rtol=1e-9, atol=1e-11)
            @test isapprox(Tlm_back, Tlm; rtol=1e-9, atol=1e-11)
        end

        @inferred synthesis_sphtor_batch(cfg, Slm_batch, Tlm_batch)
        Vt_kw, Vp_kw = synthesis_sphtor_batch(cfg, Slm_batch, Tlm_batch; real_output=false)
        Vt_cplx, Vp_cplx = @inferred synthesis_sphtor_batch_cplx(cfg, Slm_batch, Tlm_batch)
        @test eltype(Vt_cplx) === ComplexF64
        @test eltype(Vp_cplx) === ComplexF64
        @test isapprox(Vt_cplx, Vt_kw; rtol=0, atol=0)
        @test isapprox(Vp_cplx, Vp_kw; rtol=0, atol=0)
        synthesis_sphtor_batch(cfg, Slm_batch, Tlm_batch)
        analysis_sphtor_batch(cfg, Vt_batch, Vp_batch)
        GC.gc()
        @test @allocated(synthesis_sphtor_batch(cfg, Slm_batch, Tlm_batch)) <= 16_000 + _thread_alloc_slack()
        @test @allocated(analysis_sphtor_batch(cfg, Vt_batch, Vp_batch)) <= 18_000 + _thread_alloc_slack()
    end

    @testset "QST batch transforms" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(74)

        nfields = 2

        # Test individual transforms with spectral-to-spatial-to-spectral roundtrip
        for k in 1:nfields
            Qlm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Qlm[:, 1] .= real.(Qlm[:, 1])
            Slm[:, 1] .= real.(Slm[:, 1])
            Tlm[:, 1] .= real.(Tlm[:, 1])
            Slm[1, :] .= 0; Tlm[1, :] .= 0
            # Zero invalid (l < m) positions for all coefficients
            for m in 0:lmax, l in 0:(m-1)
                Qlm[l+1, m+1] = 0
                Slm[l+1, m+1] = 0
                Tlm[l+1, m+1] = 0
            end

            Vr, Vt, Vp = synthesis_qst(cfg, Qlm, Slm, Tlm; real_output=true)
            @test size(Vr) == (nlat, nlon)

            Qlm_back, Slm_back, Tlm_back = analysis_qst(cfg, Vr, Vt, Vp)
            @test size(Qlm_back) == (lmax+1, lmax+1)

            # Verify roundtrip
            @test isapprox(Qlm_back, Qlm; rtol=1e-9, atol=1e-11)
            @test isapprox(Slm_back, Slm; rtol=1e-9, atol=1e-11)
            @test isapprox(Tlm_back, Tlm; rtol=1e-9, atol=1e-11)
        end

        Qlm_batch = zeros(ComplexF64, lmax+1, lmax+1, nfields)
        Slm_batch = zeros(ComplexF64, lmax+1, lmax+1, nfields)
        Tlm_batch = zeros(ComplexF64, lmax+1, lmax+1, nfields)
        for k in 1:nfields
            Qlm_batch[:, :, k] .= randn(rng, ComplexF64, lmax+1, lmax+1)
            Slm_batch[:, :, k] .= randn(rng, ComplexF64, lmax+1, lmax+1)
            Tlm_batch[:, :, k] .= randn(rng, ComplexF64, lmax+1, lmax+1)
            Qlm_batch[:, 1, k] .= real.(Qlm_batch[:, 1, k])
            Slm_batch[:, 1, k] .= real.(Slm_batch[:, 1, k])
            Tlm_batch[:, 1, k] .= real.(Tlm_batch[:, 1, k])
            Slm_batch[1, :, k] .= 0
            Tlm_batch[1, :, k] .= 0
            for m in 0:lmax, l in 0:(m-1)
                Qlm_batch[l+1, m+1, k] = 0
                Slm_batch[l+1, m+1, k] = 0
                Tlm_batch[l+1, m+1, k] = 0
            end
        end

        Vr_kw, Vt_kw, Vp_kw = synthesis_qst_batch(cfg, Qlm_batch, Slm_batch, Tlm_batch; real_output=false)
        Vr_cplx, Vt_cplx, Vp_cplx = @inferred synthesis_qst_batch_cplx(cfg, Qlm_batch, Slm_batch, Tlm_batch)
        @test eltype(Vr_cplx) === ComplexF64
        @test eltype(Vt_cplx) === ComplexF64
        @test eltype(Vp_cplx) === ComplexF64
        @test isapprox(Vr_cplx, Vr_kw; rtol=0, atol=0)
        @test isapprox(Vt_cplx, Vt_kw; rtol=0, atol=0)
        @test isapprox(Vp_cplx, Vp_kw; rtol=0, atol=0)
    end
end
