# SHTnsKit.jl - SHTPlan tests
# Exercises src/plan.jl: planned scalar/vector transforms, validation,
# normalization conversion paths, and robert_form handling.

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

function _rand_real_alm(rng, lmax, mmax)
    alm = randn(rng, ComplexF64, lmax + 1, mmax + 1)
    alm[:, 1] .= real.(alm[:, 1])
    for m in 0:mmax, l in 0:(m - 1)
        alm[l + 1, m + 1] = 0
    end
    return alm
end

@testset "SHTPlan" begin
    @testset "Planned rfft scalar matches complex plan" begin
        for (lmax, nlon) in ((6, 13), (8, 20), (12, 25))
            cfg = create_gauss_config(lmax, lmax + 2; nlon=nlon)
            plan_c = SHTPlan(cfg)
            plan_r = SHTPlan(cfg; use_rfft=true)
            rng = MersenneTwister(777 + lmax)

            alm = _rand_real_alm(rng, lmax, lmax)

            f_c = zeros(cfg.nlat, cfg.nlon); synthesis!(plan_c, f_c, alm)
            f_r = zeros(cfg.nlat, cfg.nlon); synthesis!(plan_r, f_r, alm)
            @test isapprox(f_c, f_r; rtol=1e-10, atol=1e-12)

            alm_c = zeros(ComplexF64, lmax+1, lmax+1); analysis!(plan_c, alm_c, f_c)
            alm_r = zeros(ComplexF64, lmax+1, lmax+1); analysis!(plan_r, alm_r, f_r)
            @test isapprox(alm_c, alm_r; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "Planned scalar matches non-planned" begin
        lmax = 8
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(201)

        alm = _rand_real_alm(rng, lmax, lmax)
        f = synthesis(cfg, alm; real_output=true)
        alm_ref = analysis(cfg, f)

        f_out = zeros(cfg.nlat, cfg.nlon)
        synthesis!(plan, f_out, alm)
        @test isapprox(f_out, f; rtol=1e-12, atol=1e-14)

        alm_out = zeros(ComplexF64, lmax + 1, lmax + 1)
        analysis!(plan, alm_out, f_out)
        @test isapprox(alm_out, alm_ref; rtol=1e-12, atol=1e-14)
        @test isapprox(alm_out, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "Planned scalar: reuse across calls (no state leakage)" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(202)

        alm_a = _rand_real_alm(rng, lmax, lmax)
        alm_b = _rand_real_alm(rng, lmax, lmax)

        f_a = zeros(cfg.nlat, cfg.nlon); f_b = zeros(cfg.nlat, cfg.nlon)
        synthesis!(plan, f_a, alm_a)
        synthesis!(plan, f_b, alm_b)
        # Running B must not pollute A's previous result
        @test isapprox(f_a, synthesis(cfg, alm_a; real_output=true); rtol=1e-12, atol=1e-14)
        @test isapprox(f_b, synthesis(cfg, alm_b; real_output=true); rtol=1e-12, atol=1e-14)

        # Run A again → identical to first call
        f_a2 = zeros(cfg.nlat, cfg.nlon)
        synthesis!(plan, f_a2, alm_a)
        @test f_a2 == f_a
    end

    @testset "Planned scalar: dimension mismatch throws" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        plan = SHTPlan(cfg)
        alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        f = zeros(cfg.nlat, cfg.nlon)

        @test_throws DimensionMismatch synthesis!(plan, zeros(1, cfg.nlon), alm)
        @test_throws DimensionMismatch synthesis!(plan, zeros(cfg.nlat, 1), alm)
        @test_throws DimensionMismatch synthesis!(plan, f, zeros(ComplexF64, 1, cfg.mmax + 1))
        @test_throws DimensionMismatch synthesis!(plan, f, zeros(ComplexF64, cfg.lmax + 1, 1))

        @test_throws DimensionMismatch analysis!(plan, alm, zeros(1, cfg.nlon))
        @test_throws DimensionMismatch analysis!(plan, alm, zeros(cfg.nlat, 1))
        @test_throws DimensionMismatch analysis!(plan, zeros(ComplexF64, 1, cfg.mmax + 1), f)
    end

    @testset "Planned vector (sphtor) matches non-planned" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(203)

        Slm = _rand_real_alm(rng, lmax, lmax); Slm[1, 1] = 0
        Tlm = _rand_real_alm(rng, lmax, lmax); Tlm[1, 1] = 0

        Vt = zeros(cfg.nlat, cfg.nlon); Vp = zeros(cfg.nlat, cfg.nlon)
        synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm; real_output=true)

        Vt_ref, Vp_ref = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        @test isapprox(Vt, Vt_ref; rtol=1e-11, atol=1e-13)
        @test isapprox(Vp, Vp_ref; rtol=1e-11, atol=1e-13)

        Slm_out = zeros(ComplexF64, lmax + 1, lmax + 1)
        Tlm_out = zeros(ComplexF64, lmax + 1, lmax + 1)
        analysis_sphtor!(plan, Slm_out, Tlm_out, Vt, Vp)

        Slm_ref, Tlm_ref = analysis_sphtor(cfg, Vt, Vp)
        @test isapprox(Slm_out, Slm_ref; rtol=1e-11, atol=1e-13)
        @test isapprox(Tlm_out, Tlm_ref; rtol=1e-11, atol=1e-13)
        @inferred synthesis_sphtor(cfg, Slm, Tlm)
    end

    @testset "Planned vector dimension checks" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        plan = SHTPlan(cfg)
        S = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        T = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        Vt = zeros(cfg.nlat, cfg.nlon); Vp = zeros(cfg.nlat, cfg.nlon)

        @test_throws DimensionMismatch analysis_sphtor!(plan, S, T, zeros(1, cfg.nlon), Vp)
        @test_throws DimensionMismatch analysis_sphtor!(plan, S, T, Vt, zeros(cfg.nlat, 1))
        @test_throws DimensionMismatch analysis_sphtor!(plan, zeros(ComplexF64, 1, 1), T, Vt, Vp)
        @test_throws DimensionMismatch analysis_sphtor!(plan, S, zeros(ComplexF64, 1, 1), Vt, Vp)

        @test_throws DimensionMismatch synthesis_sphtor!(plan, zeros(1, cfg.nlon), Vp, S, T)
        @test_throws DimensionMismatch synthesis_sphtor!(plan, Vt, zeros(cfg.nlat, 1), S, T)
    end

    @testset "Planned scalar matches the non-planned path exactly" begin
        # `SHTPlan`'s scalar path is a drop-in accelerator for `analysis`/
        # `synthesis`, so it must be ORTHONORMAL like them — not merely
        # self-consistent. A self-roundtrip test cannot tell the two apart:
        # plan∘plan closes under either convention, and only MIXING them breaks.
        # These assertions are what would catch a revert.
        #
        # They compare to a TOLERANCE, not bit-for-bit. The two paths reach the
        # same φ transform through different FFTW plans — `synthesis` uses the
        # shared cache (built UNALIGNED, since it reuses plans across arbitrary
        # caller arrays) while `SHTPlan` plans its own stably-aligned buffers —
        # so FFTW may pick different codelets and the results can differ in the
        # last ulp. That is not a convention bug and varies by CPU and FFTW
        # build: `==` passed on arm64 and failed on x86_64 CI at 2e-16.
        # The tolerance below is ~4 orders above roundoff and ~10 orders below a
        # normalization revert, which is an O(1) relative error (M[l,m] is 40-180%
        # off for :schmidt/:fourpi), so a revert is still caught cleanly.
        lmax = 6
        for (nrm, cs) in ((:orthonormal, true), (:schmidt, true), (:fourpi, false))
            cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1,
                                      norm=nrm, cs_phase=cs)
            plan = SHTPlan(cfg)
            rng = MersenneTwister(204)
            alm = _rand_real_alm(rng, lmax, lmax)

            f_plan = zeros(cfg.nlat, cfg.nlon)
            synthesis!(plan, f_plan, alm)
            @test all(isfinite, f_plan)
            # the non-planned transform to roundoff — same convention, not just
            # self-consistent (see the note above on why this is not `==`)
            @test isapprox(f_plan, synthesis(cfg, alm; real_output=true);
                           rtol=1e-12, atol=1e-14)

            alm_back = zeros(ComplexF64, lmax + 1, lmax + 1)
            analysis!(plan, alm_back, f_plan)
            @test isapprox(alm_back, analysis(cfg, f_plan); rtol=1e-12, atol=1e-14)
            # and the plan's own roundtrip still recovers the input
            @test isapprox(alm_back, alm; rtol=1e-9, atol=1e-11)
        end
    end

    @testset "Planned sphtor: robert_form path" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1, robert_form=true)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(205)

        Slm = _rand_real_alm(rng, lmax, lmax); Slm[1, 1] = 0
        Tlm = _rand_real_alm(rng, lmax, lmax); Tlm[1, 1] = 0

        Vt_ref, Vp_ref = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        Vt = zeros(cfg.nlat, cfg.nlon); Vp = zeros(cfg.nlat, cfg.nlon)
        synthesis_sphtor!(plan, Vt, Vp, Slm, Tlm; real_output=true)
        @test isapprox(Vt, Vt_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Vp, Vp_ref; rtol=1e-10, atol=1e-12)

        Slm_back = zeros(ComplexF64, lmax + 1, lmax + 1)
        Tlm_back = zeros(ComplexF64, lmax + 1, lmax + 1)
        analysis_sphtor!(plan, Slm_back, Tlm_back, Vt, Vp)
        Slm_ref, Tlm_ref = analysis_sphtor(cfg, Vt, Vp)
        @test isapprox(Slm_back, Slm_ref; rtol=1e-10, atol=1e-12)
        @test isapprox(Tlm_back, Tlm_ref; rtol=1e-10, atol=1e-12)
    end

    @testset "Planned sphtor: robert_form analysis does not allocate row slices" begin
        # The property that matters is that nothing is allocated PER ROW or per
        # (m, θ) pair — a per-row temporary would make the cost grow with the
        # problem. Measure two sizes and require the allocation not to grow:
        # an absolute byte bound cannot work here because the shared m-loop
        # starts `@threads :static` tasks when threads are available, and the
        # task-spawn overhead is a constant few KB independent of the grid.
        function planned_sphtor_bytes(lmax)
            cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1, robert_form=true)
            plan = SHTPlan(cfg)
            rng = MersenneTwister(207)
            Vt = randn(rng, cfg.nlat, cfg.nlon)
            Vp = randn(rng, cfg.nlat, cfg.nlon)
            Slm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
            Tlm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
            analysis_sphtor!(plan, Slm, Tlm, Vt, Vp)   # warm up
            GC.gc()
            return minimum(@allocated(analysis_sphtor!(plan, Slm, Tlm, Vt, Vp)) for _ in 1:3)
        end

        small = planned_sphtor_bytes(5)     #  7 lat ×  6 m
        large = planned_sphtor_bytes(20)    # 22 lat × 21 m  (11x the work)
        @test large <= small + 256
        if Threads.nthreads() == 1
            @test small <= 128              # serial path stays allocation-free
        end
    end

    @testset "Planned scalar: complex output path (real_output=false) runs" begin
        # real_output=false skips Hermitian symmetry enforcement, producing a
        # genuinely complex spatial field — just sanity-check that the path is
        # exercised and produces finite output.
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        plan = SHTPlan(cfg)
        rng = MersenneTwister(206)

        alm = _rand_real_alm(rng, lmax, lmax)
        f_cplx = zeros(ComplexF64, cfg.nlat, cfg.nlon)
        synthesis!(plan, f_cplx, alm; real_output=false)
        @test all(isfinite, f_cplx)
        @test eltype(f_cplx) <: Complex
    end

    @testset "planned transforms match the cfg form with Legendre tables" begin
        # Regression: `analysis!`/`synthesis!` hardwired the on-the-fly Legendre
        # kernel and never consulted `cfg.NP_tables`, so a plan built on a
        # table-enabled config was ~6x slower than the plain call it exists to
        # beat. The vector pair additionally ran the whole m/θ loop twice, once
        # per component. Both now route through the shared orchestrators; these
        # checks pin the numerical agreement in every buffer mode.
        rng = MersenneTwister(4821)
        for tables in (false, true), use_rfft in (false, true), robert in (false, true)
            cfgp = create_gauss_config(10, 12)
            cfgp.robert_form = robert
            tables ? SHTnsKit.prepare_plm_tables!(cfgp) : SHTnsKit.disable_plm_tables!(cfgp)

            alm = zeros(ComplexF64, cfgp.lmax + 1, cfgp.mmax + 1)
            for m in 0:cfgp.mmax, l in m:cfgp.lmax
                alm[l + 1, m + 1] = m == 0 ? randn(rng) : complex(randn(rng), randn(rng))
            end
            S = copy(alm); T = 0.5 .* alm
            S[1, 1] = 0; T[1, 1] = 0

            f = synthesis(cfgp, alm)
            Vt, Vp = synthesis_sphtor(cfgp, S, T)
            alm_ref = analysis(cfgp, f)
            S_ref, T_ref = analysis_sphtor(cfgp, Vt, Vp)

            plan = SHTPlan(cfgp; use_rfft=use_rfft)
            tol = use_rfft ? 1e-12 : 0.0   # complex path must agree bit-for-bit

            alm_out = similar(alm_ref)
            analysis!(plan, alm_out, f)
            @test maximum(abs, alm_out .- alm_ref) <= tol

            f_out = similar(f)
            synthesis!(plan, f_out, alm)
            @test maximum(abs, f_out .- f) <= tol

            S_out = similar(S_ref); T_out = similar(T_ref)
            SHTnsKit.analysis_sphtor!(plan, S_out, T_out, Vt, Vp)
            @test maximum(abs, S_out .- S_ref) <= tol
            @test maximum(abs, T_out .- T_ref) <= tol

            Vt_out = similar(Vt); Vp_out = similar(Vp)
            SHTnsKit.synthesis_sphtor!(plan, Vt_out, Vp_out, S, T)
            @test maximum(abs, Vt_out .- Vt) <= tol
            @test maximum(abs, Vp_out .- Vp) <= tol
        end
    end
end
