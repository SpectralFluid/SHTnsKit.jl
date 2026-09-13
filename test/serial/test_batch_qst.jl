# SHTnsKit.jl - QST batch transform tests
# Exercises analysis_qst_batch / synthesis_qst_batch: roundtrip and per-field
# consistency with single-field analysis_qst / synthesis_qst.

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

function _real_alm(rng, lmax, mmax)
    a = randn(rng, ComplexF64, lmax + 1, mmax + 1)
    a[:, 1] .= real.(a[:, 1])
    for m in 0:mmax, l in 0:(m - 1)
        a[l + 1, m + 1] = 0
    end
    return a
end

@testset "Batch QST transforms" begin
    @testset "synthesis_qst_batch matches per-field synthesis_qst" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(950)
        nfields = 3

        Qb = randn(rng, ComplexF64, lmax + 1, lmax + 1, nfields)
        Sb = randn(rng, ComplexF64, lmax + 1, lmax + 1, nfields)
        Tb = randn(rng, ComplexF64, lmax + 1, lmax + 1, nfields)
        for k in 1:nfields
            Qb[:, 1, k] .= real.(Qb[:, 1, k])
            Sb[:, 1, k] .= real.(Sb[:, 1, k]); Sb[1, 1, k] = 0
            Tb[:, 1, k] .= real.(Tb[:, 1, k]); Tb[1, 1, k] = 0
            for m in 0:lmax, l in 0:(m - 1)
                Qb[l + 1, m + 1, k] = 0
                Sb[l + 1, m + 1, k] = 0
                Tb[l + 1, m + 1, k] = 0
            end
        end

        Vr_b, Vt_b, Vp_b = synthesis_qst_batch(cfg, Qb, Sb, Tb; real_output=true)
        @test size(Vr_b) == (cfg.nlat, cfg.nlon, nfields)
        @inferred synthesis_qst_batch(cfg, Qb, Sb, Tb)

        for k in 1:nfields
            Vr, Vt, Vp = synthesis_qst(cfg, Qb[:, :, k], Sb[:, :, k], Tb[:, :, k]; real_output=true)
            @test isapprox(Vr_b[:, :, k], Vr; rtol=1e-11, atol=1e-13)
            @test isapprox(Vt_b[:, :, k], Vt; rtol=1e-11, atol=1e-13)
            @test isapprox(Vp_b[:, :, k], Vp; rtol=1e-11, atol=1e-13)
        end

        synthesis_qst_batch(cfg, Qb, Sb, Tb)
        GC.gc()
        @test @allocated(synthesis_qst_batch(cfg, Qb, Sb, Tb)) <= 25_000 + _thread_alloc_slack()
    end

    @testset "analysis_qst_batch matches per-field analysis_qst" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(951)
        nfields = 3

        Vr_b = randn(rng, cfg.nlat, cfg.nlon, nfields)
        Vt_b = randn(rng, cfg.nlat, cfg.nlon, nfields)
        Vp_b = randn(rng, cfg.nlat, cfg.nlon, nfields)

        Qb, Sb, Tb = analysis_qst_batch(cfg, Vr_b, Vt_b, Vp_b)
        @test size(Qb) == (lmax + 1, lmax + 1, nfields)
        @test size(Sb) == size(Qb)
        @test size(Tb) == size(Qb)

        for k in 1:nfields
            Q, S, T = analysis_qst(cfg,
                Vr_b[:, :, k], Vt_b[:, :, k], Vp_b[:, :, k])
            @test isapprox(Qb[:, :, k], Q; rtol=1e-11, atol=1e-13)
            @test isapprox(Sb[:, :, k], S; rtol=1e-11, atol=1e-13)
            @test isapprox(Tb[:, :, k], T; rtol=1e-11, atol=1e-13)
        end

        analysis_qst_batch(cfg, Vr_b, Vt_b, Vp_b)
        GC.gc()
        @test @allocated(analysis_qst_batch(cfg, Vr_b, Vt_b, Vp_b)) <= 28_000 + _thread_alloc_slack()
    end

    @testset "Batch QST roundtrip" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(952)
        nfields = 4

        Qb = cat([_real_alm(rng, lmax, lmax) for _ in 1:nfields]...; dims=3)
        Sb = cat([_real_alm(rng, lmax, lmax) for _ in 1:nfields]...; dims=3)
        Tb = cat([_real_alm(rng, lmax, lmax) for _ in 1:nfields]...; dims=3)
        for k in 1:nfields
            Sb[1, 1, k] = 0; Tb[1, 1, k] = 0
        end

        Vr_b, Vt_b, Vp_b = synthesis_qst_batch(cfg, Qb, Sb, Tb; real_output=true)
        Qr, Sr, Tr = analysis_qst_batch(cfg, Vr_b, Vt_b, Vp_b)

        @test isapprox(Qr, Qb; rtol=1e-9, atol=1e-11)
        @test isapprox(Sr, Sb; rtol=1e-9, atol=1e-11)
        @test isapprox(Tr, Tb; rtol=1e-9, atol=1e-11)
    end

    @testset "Batch QST dimension checks" begin
        cfg = create_gauss_config(3, 5; nlon=7)
        good = (cfg.nlat, cfg.nlon, 2)
        Vr = zeros(good...); Vt = zeros(good...); Vp = zeros(good...)

        # Mismatched spatial dims
        @test_throws DimensionMismatch analysis_qst_batch(cfg,
            zeros(cfg.nlat + 1, cfg.nlon, 2), Vt, Vp)
        @test_throws DimensionMismatch analysis_qst_batch(cfg,
            Vr, zeros(cfg.nlat, cfg.nlon + 1, 2), Vp)
        @test_throws DimensionMismatch analysis_qst_batch(cfg,
            Vr, Vt, zeros(cfg.nlat, cfg.nlon, 3))

        # Synthesis: wrong spectral leading dims
        bad_spec = zeros(ComplexF64, cfg.lmax, cfg.mmax + 1, 2)
        good_spec = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1, 2)
        @test_throws DimensionMismatch synthesis_qst_batch(cfg,
            bad_spec, good_spec, good_spec)
    end
end
