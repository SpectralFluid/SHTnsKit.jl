# SHTnsKit.jl - Threading / parallel CPU tests (no MPI).
# Verifies correctness under Julia's native multithreading:
#  - Batch transforms give the same result with 1 vs N threads in user-level loops
#  - Per-thread SHTPlan pattern (recommended usage) is race-free
#  - analysis_batch / synthesis_batch are thread-consistent
#
# Run in a Julia session launched with e.g. `julia -t 4 --project`.

using Test
using Random
using Base.Threads
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

function _thread_rand_real_alm(rng, lmax, mmax)
    alm = randn(rng, ComplexF64, lmax + 1, mmax + 1)
    alm[:, 1] .= real.(alm[:, 1])
    for m in 0:mmax, l in 0:(m - 1)
        alm[l + 1, m + 1] = 0
    end
    return alm
end

function _threaded_collect(f, nitems)
    out = Vector{Any}(undef, nitems)
    err = try
        @threads for k in 1:nitems
            out[k] = f(k)
        end
        nothing
    catch caught
        caught
    end
    return out, err
end

function _simultaneous_threaded_collect(f)
    nitems = Threads.nthreads()
    out = Vector{Any}(undef, nitems)
    ready = Threads.Atomic{Int}(0)
    err = try
        @threads :static for k in 1:nitems
            Threads.atomic_add!(ready, 1)
            while ready[] < nitems
                GC.safepoint()
            end
            out[k] = f(k)
        end
        nothing
    catch caught
        caught
    end
    return out, err
end

_thread_result_isapprox(actual, expected; kwargs...) =
    isapprox(actual, expected; kwargs...)

function _thread_result_isapprox(actual::Tuple, expected::Tuple; kwargs...)
    return length(actual) == length(expected) &&
           all(_thread_result_isapprox(actual[i], expected[i]; kwargs...)
               for i in eachindex(actual))
end

function _test_threaded_matches(f, refs; kwargs...)
    out, err = _threaded_collect(f, length(refs))
    @test err === nothing
    if err === nothing
        @test all(_thread_result_isapprox(out[k], refs[k]; kwargs...)
                  for k in eachindex(refs))
    end
end

@testset "Threading / parallel CPU" begin
    nt = Threads.nthreads()
    VERBOSE && @info "Threads available" nt

    @testset "Per-thread SHTPlan: concurrent synthesis" begin
        # Canonical multi-thread pattern: one plan per thread.
        lmax = 8
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plans = [SHTPlan(cfg) for _ in 1:Threads.maxthreadid()]

        ntasks = 16
        rngs = [MersenneTwister(1000 + k) for k in 1:ntasks]
        alms = [_thread_rand_real_alm(rngs[k], lmax, lmax) for k in 1:ntasks]
        outs = [zeros(cfg.nlat, cfg.nlon) for _ in 1:ntasks]

        @threads :static for k in 1:ntasks
            plan = plans[Threads.threadid()]
            synthesis!(plan, outs[k], alms[k])
        end

        # Each output must match the serial synthesis of its own coefficients
        for k in 1:ntasks
            ref = synthesis(cfg, alms[k]; real_output=true)
            @test isapprox(outs[k], ref; rtol=1e-12, atol=1e-14)
        end
    end

    @testset "Per-thread SHTPlan: concurrent analysis" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plans = [SHTPlan(cfg) for _ in 1:Threads.maxthreadid()]

        ntasks = 12
        rng = MersenneTwister(1100)
        fields = [randn(rng, cfg.nlat, cfg.nlon) for _ in 1:ntasks]
        outs = [zeros(ComplexF64, lmax + 1, lmax + 1) for _ in 1:ntasks]

        @threads :static for k in 1:ntasks
            plan = plans[Threads.threadid()]
            analysis!(plan, outs[k], fields[k])
        end

        for k in 1:ntasks
            ref = analysis(cfg, fields[k])
            @test isapprox(outs[k], ref; rtol=1e-12, atol=1e-14)
        end
    end

    @testset "Batch analysis: result independent of thread count" begin
        # analysis_batch uses @threads :static internally over m; the result must
        # match a reference sequential run regardless of Threads.nthreads().
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(1200)
        nfields = 8
        fields = randn(rng, cfg.nlat, cfg.nlon, nfields)

        alm_batch = analysis_batch(cfg, fields)
        # Single-field reference
        for k in 1:nfields
            ref = analysis(cfg, fields[:, :, k])
            @test isapprox(alm_batch[:, :, k], ref; rtol=1e-11, atol=1e-13)
        end
    end

    @testset "Batch synthesis: result independent of thread count" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(1201)
        nfields = 8
        alm_batch = randn(rng, ComplexF64, lmax + 1, lmax + 1, nfields)
        for k in 1:nfields
            alm_batch[:, 1, k] .= real.(alm_batch[:, 1, k])
            for m in 0:lmax, l in 0:(m - 1)
                alm_batch[l + 1, m + 1, k] = 0
            end
        end

        fields = synthesis_batch(cfg, alm_batch; real_output=true)
        for k in 1:nfields
            ref = synthesis(cfg, alm_batch[:, :, k]; real_output=true)
            @test isapprox(fields[:, :, k], ref; rtol=1e-11, atol=1e-13)
        end
    end

    @testset "Determinism: repeated multi-threaded runs agree bitwise-close" begin
        lmax = 7
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(1300)
        nfields = 6
        fields = randn(rng, cfg.nlat, cfg.nlon, nfields)

        a1 = analysis_batch(cfg, fields)
        a2 = analysis_batch(cfg, fields)
        a3 = analysis_batch(cfg, fields)
        @test a1 == a2
        @test a1 == a3
    end

    @testset "Sequential unrelated configs don't interfere" begin
        # Verify that switching cfg between calls doesn't leak cached state.
        # Concurrent allocating-transform coverage lives in the testsets below.
        lmax_a, lmax_b = 5, 7
        cfg_a = create_gauss_config(lmax_a, lmax_a + 2; nlon=2*lmax_a + 1)
        cfg_b = create_gauss_config(lmax_b, lmax_b + 2; nlon=2*lmax_b + 1)

        rng = MersenneTwister(1400)
        f_a = randn(rng, cfg_a.nlat, cfg_a.nlon)
        f_b = randn(rng, cfg_b.nlat, cfg_b.nlon)

        ref_a = analysis(cfg_a, f_a)
        ref_b = analysis(cfg_b, f_b)
        # Interleave calls, expect identical results to the references
        got_a = analysis(cfg_a, f_a)
        got_b = analysis(cfg_b, f_b)
        got_a2 = analysis(cfg_a, f_a)

        @test got_a == ref_a
        @test got_b == ref_b
        @test got_a2 == ref_a
    end

    @testset "Allocating transforms compose with outer threading" begin
        lmax = 6
        config_builders = [
            ("on-the-fly", () -> create_gauss_config(
                lmax, lmax + 2; nlon=2*lmax + 1, norm=:schmidt,
                real_norm=true, cs_phase=false)),
            ("tables", () -> prepare_plm_tables!(create_gauss_config(
                lmax, lmax + 2; nlon=2*lmax + 1, norm=:schmidt,
                real_norm=true, cs_phase=false))),
        ]

        for (mode, build_config) in config_builders
            @testset "$mode Legendre evaluation" begin
                # Compute references with a different config so the config used
                # by the outer threaded calls retains cold lazy caches.
                cfg = build_config()
                ref_cfg = build_config()
                ntasks = max(2, 2 * nt)
                rng = MersenneTwister(mode == "tables" ? 1450 : 1475)

                fields = [randn(rng, cfg.nlat, cfg.nlon) for _ in 1:ntasks]
                analysis_refs = [analysis(ref_cfg, fields[k]) for k in 1:ntasks]
                @testset "scalar analysis" begin
                    _test_threaded_matches(k -> analysis(cfg, fields[k]), analysis_refs;
                                           rtol=1e-12, atol=1e-14)
                end

                alms = [_thread_rand_real_alm(rng, lmax, lmax) for _ in 1:ntasks]
                synthesis_refs = [synthesis(ref_cfg, alms[k]; real_output=true)
                                  for k in 1:ntasks]
                @testset "scalar synthesis" begin
                    _test_threaded_matches(
                        k -> synthesis(cfg, alms[k]; real_output=true), synthesis_refs;
                        rtol=1e-12, atol=1e-14)
                end

                Ss = [_thread_rand_real_alm(rng, lmax, lmax) for _ in 1:ntasks]
                Ts = [_thread_rand_real_alm(rng, lmax, lmax) for _ in 1:ntasks]
                vector_refs = [synthesis_sphtor(ref_cfg, Ss[k], Ts[k]; real_output=true)
                               for k in 1:ntasks]
                @testset "vector synthesis" begin
                    _test_threaded_matches(
                        k -> synthesis_sphtor(cfg, Ss[k], Ts[k]; real_output=true),
                        vector_refs; rtol=1e-11, atol=1e-13)
                end

                vector_analysis_refs = [analysis_sphtor(ref_cfg, vector_refs[k]...)
                                        for k in 1:ntasks]
                @testset "vector analysis" begin
                    _test_threaded_matches(
                        k -> analysis_sphtor(cfg, vector_refs[k]...), vector_analysis_refs;
                        rtol=1e-11, atol=1e-13)
                end
            end
        end
    end

    @testset "Cold OTF scratch initialization is concurrent-safe" begin
        # This is the intermediate state produced by `resize!` before its new
        # slots are populated. The cache initializer must never expose or choke
        # on such slots when several transforms first touch a config together.
        partial = Vector{Vector{Float64}}(undef, Threads.maxthreadid())
        @test SHTnsKit._ensure_otf_scratch!(partial, 8) === partial
        @test all(i -> isassigned(partial, i) && length(partial[i]) == 9,
                  eachindex(partial))

        large = [Vector{Float64}(undef, 101) for _ in 1:Threads.maxthreadid()]
        @test SHTnsKit._ensure_otf_scratch!(large, 10) === large
        @test all(buffer -> length(buffer) == 101, large)

        for trial in 1:(nt == 1 ? 1 : 8)
            cfg = create_gauss_config(8, 10; nlon=17, norm=:schmidt,
                                      real_norm=true, cs_phase=false)
            ref_cfg = create_gauss_config(8, 10; nlon=17, norm=:schmidt,
                                          real_norm=true, cs_phase=false)
            rng = MersenneTwister(1600 + trial)
            Ss = [_thread_rand_real_alm(rng, cfg.lmax, cfg.mmax) for _ in 1:nt]
            Ts = [_thread_rand_real_alm(rng, cfg.lmax, cfg.mmax) for _ in 1:nt]
            refs = [synthesis_sphtor(ref_cfg, Ss[k], Ts[k]; real_output=true)
                    for k in 1:nt]

            out, err = _simultaneous_threaded_collect(
                k -> synthesis_sphtor(cfg, Ss[k], Ts[k]; real_output=true))
            @test err === nothing
            if err === nothing
                @test all(_thread_result_isapprox(out[k], refs[k]; rtol=1e-11, atol=1e-13)
                          for k in eachindex(refs))
            end
        end
    end

    @testset "Per-thread sphtor plan: concurrent vector synthesis" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plans = [SHTPlan(cfg) for _ in 1:Threads.maxthreadid()]

        ntasks = 8
        rng = MersenneTwister(1500)
        Ss = [_thread_rand_real_alm(rng, lmax, lmax) for _ in 1:ntasks]
        Ts = [_thread_rand_real_alm(rng, lmax, lmax) for _ in 1:ntasks]
        for k in 1:ntasks
            Ss[k][1, 1] = 0; Ts[k][1, 1] = 0
        end
        Vts = [zeros(cfg.nlat, cfg.nlon) for _ in 1:ntasks]
        Vps = [zeros(cfg.nlat, cfg.nlon) for _ in 1:ntasks]

        @threads :static for k in 1:ntasks
            plan = plans[Threads.threadid()]
            synthesis_sphtor!(plan, Vts[k], Vps[k], Ss[k], Ts[k]; real_output=true)
        end

        for k in 1:ntasks
            Vt_ref, Vp_ref = synthesis_sphtor(cfg, Ss[k], Ts[k]; real_output=true)
            @test isapprox(Vts[k], Vt_ref; rtol=1e-11, atol=1e-13)
            @test isapprox(Vps[k], Vp_ref; rtol=1e-11, atol=1e-13)
        end
    end
end
