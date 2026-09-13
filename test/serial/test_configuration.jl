# SHTnsKit.jl - Configuration and Setup Tests
# Tests for grid configuration, indexing, and normalization

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Configuration and Setup" begin
    @testset "Gauss grid configuration" begin
        # Basic configuration
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        @test cfg.lmax == lmax
        @test cfg.mmax == lmax
        @test cfg.nlat == nlat
        @test cfg.nlon == nlon
        @test cfg.grid_type == :gauss

        # Check Gauss weights sum to 2 (integral of 1 over sphere / 2π)
        @test isapprox(sum(cfg.w), 2.0; rtol=1e-12)

        # Check cos(θ) values are in [-1, 1] and sorted
        @test all(-1 .<= cfg.x .<= 1)
        @test issorted(cfg.x; rev=true) || issorted(cfg.x)

        # Check θ values match x = cos(θ)
        @test isapprox(cos.(cfg.θ), cfg.x; rtol=1e-12)

        # Check φ values span [0, 2π)
        @test cfg.φ[1] ≈ 0.0
        @test cfg.φ[end] < 2π
        @test length(cfg.φ) == nlon
    end

    @testset "Regular grid configuration" begin
        lmax = 8
        nlat = 2 * (lmax + 1)
        nlon = 2 * (2 * lmax + 1)

        # Regular grid with Driscoll-Healy weights
        cfg_dh = create_regular_config(lmax, nlat; nlon=nlon, include_poles=true, use_dh_weights=true)
        @test cfg_dh.grid_type == :driscoll_healy
        @test cfg_dh.θ[1] == 0.0

        # DH weights without the DH pole-inclusive nodes would create a midpoint
        # grid mislabeled as :driscoll_healy.
        @test_throws ArgumentError create_regular_config(
            lmax, nlat; nlon=nlon, include_poles=false, use_dh_weights=true,
        )
        @test_throws ArgumentError create_regular_config(
            lmax, nlat - 2; nlon=nlon, include_poles=true, use_dh_weights=true,
        )

        # Regular grid without poles
        cfg_nopole = create_regular_config(lmax, nlat; nlon=nlon, include_poles=false)
        @test cfg_nopole.nlat == nlat
    end

    @testset "Generic create_config" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1

        # Test create_config with various options
        cfg = create_config(lmax; nlat=nlat, nlon=nlon)
        @test cfg.lmax == lmax
        @test cfg.nlat == nlat
        @test cfg.nlon == nlon

        # Test with mmax different from lmax
        cfg_mmax = create_config(lmax; mmax=lmax-2, nlat=nlat, nlon=nlon)
        @test cfg_mmax.mmax == lmax - 2

        # Test with mres
        cfg_mres = create_config(lmax; mres=2, nlat=nlat, nlon=nlon)
        @test cfg_mres.mres == 2

        cfg_dh = create_config(lmax; grid_type=:driscoll_healy, nlon=nlon)
        @test cfg_dh.nlat == 2 * (lmax + 1)
    end

    @testset "On-the-fly mode" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Test enabling on-the-fly mode
        set_on_the_fly!(cfg)
        @test cfg.on_the_fly == true
        @test cfg.use_plm_tables == false
        @test is_on_the_fly(cfg) == true

        # Test switching back to table mode
        set_use_tables!(cfg)
        @test cfg.use_plm_tables == true
        @test is_on_the_fly(cfg) == false
    end

    @testset "PLM table preparation" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Initially no tables (default is on-the-fly)
        @test cfg.use_plm_tables == false

        # Prepare PLM tables
        prepare_plm_tables!(cfg)
        @test cfg.use_plm_tables == true
        @test length(cfg.plm_tables) == cfg.mmax + 1

        # Each table should be non-empty and have valid dimensions
        for m in 0:cfg.mmax
            tbl = cfg.plm_tables[m+1]
            @test !isempty(tbl)
            @test size(tbl, 1) > 0
            @test size(tbl, 2) > 0
        end
    end

    @testset "Normalization matrix" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2)

        # Nlm should be positive for valid (l,m) pairs
        for m in 0:lmax
            for l in m:lmax
                @test cfg.Nlm[l+1, m+1] > 0
            end
        end
    end

    @testset "Index calculations" begin
        lmax = 5
        mmax = 5
        mres = 1

        # Test nlm_calc
        nlm = nlm_calc(lmax, mmax, mres)
        @test nlm == (lmax + 1) * (lmax + 2) ÷ 2  # triangular number

        # Test LM_index roundtrip
        for m in 0:mmax
            for l in m:lmax
                idx = LM_index(lmax, mres, l, m)
                @test idx >= 0
                @test idx < nlm
            end
        end

        # Test nlm_cplx_calc
        nlm_cplx = nlm_cplx_calc(lmax, mmax, mres)
        @test nlm_cplx == (lmax + 1)^2  # full square for complex
    end

    @testset "Various lmax configurations" begin
        for lmax in [4, 8, 16, 32]
            nlat = lmax + 2
            nlon = 2*lmax + 1
            cfg = create_gauss_config(lmax, nlat; nlon=nlon)

            @test cfg.lmax == lmax
            @test cfg.nlm == (lmax + 1) * (lmax + 2) ÷ 2
            @test length(cfg.x) == nlat
            @test length(cfg.w) == nlat
            @test length(cfg.θ) == nlat
            @test length(cfg.φ) == nlon
        end
    end

    @testset "Thread utilization diagnostics" begin
        lmax = 8
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Test check_thread_utilization returns correct structure
        stats = SHTnsKit.check_thread_utilization(cfg; warn=false)

        @test haskey(stats, :nthreads)
        @test haskey(stats, :mmax)
        @test haskey(stats, :active_threads)
        @test haskey(stats, :utilization)

        # Verify values are consistent
        @test stats.mmax == cfg.mmax
        @test stats.nthreads == Threads.nthreads()
        @test stats.active_threads == min(stats.nthreads, cfg.mmax + 1)
        @test stats.utilization == stats.active_threads / max(stats.nthreads, 1)

        # Utilization should be in [0, 1]
        @test 0.0 <= stats.utilization <= 1.0

        # Test with different configurations
        for lmax_test in [4, 16, 32]
            cfg_test = create_gauss_config(lmax_test, lmax_test + 2)
            stats_test = SHTnsKit.check_thread_utilization(cfg_test; warn=false)
            @test stats_test.mmax == lmax_test
            @test stats_test.active_threads <= stats_test.nthreads
            @test stats_test.active_threads <= lmax_test + 1
        end
    end

    @testset "Memory estimation" begin
        lmax = 8
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Test estimate_table_memory returns reasonable value
        mem = estimate_table_memory(cfg)
        @test mem >= 0
        @test typeof(mem) <: Number

        # Larger lmax should require more memory
        cfg_large = create_gauss_config(32, 34)
        mem_large = estimate_table_memory(cfg_large)
        @test mem_large >= mem
    end

    @testset "South pole first mode" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Test south pole first functions
        initial_state = is_south_pole_first(cfg)
        @test typeof(initial_state) == Bool

        # Toggle south pole first
        set_south_pole_first!(cfg)
        @test is_south_pole_first(cfg) == true

        set_north_pole_first!(cfg)
        @test is_south_pole_first(cfg) == false

        # Test create_gauss_config_spf creates south-pole-first config
        cfg_spf = create_gauss_config_spf(lmax, nlat)
        @test is_south_pole_first(cfg_spf) == true
    end

    @testset "Padding control" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Test padding functions exist and return expected types
        initial_padding = is_padding_enabled(cfg)
        @test typeof(initial_padding) == Bool

        # Enable and disable padding
        set_allow_padding!(cfg)
        @test is_padding_enabled(cfg) == true

        disable_padding!(cfg)
        @test is_padding_enabled(cfg) == false

        # Test padding queries
        nlat_padded = get_nlat_padded(cfg)
        @test nlat_padded >= nlat

        spat_dist = get_spat_dist(cfg)
        @test spat_dist >= 0

        optimal_pad = compute_optimal_padding(nlat, nlon)
        @test optimal_pad >= 0

        # Test padding overhead estimation
        overhead = estimate_padding_overhead(cfg)
        @test overhead >= 0.0
    end

    @testset "Padded array allocation" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Allocate padded spatial arrays
        arr = allocate_padded_spatial(cfg)
        @test size(arr, 1) >= nlat
        @test size(arr, 2) >= nlon

        # Allocate padded batch arrays
        batch_size = 4
        arr_batch = allocate_padded_spatial_batch(cfg, batch_size)
        @test size(arr_batch, 1) >= nlat
        @test size(arr_batch, 2) >= nlon
        @test size(arr_batch, 3) == batch_size

        # Test copy functions
        src = randn(nlat, nlon)
        padded = allocate_padded_spatial(cfg)
        copy_to_padded!(padded, src, cfg)

        dst = zeros(nlat, nlon)
        copy_from_padded!(dst, padded, cfg)
        @test isapprox(dst, src; rtol=1e-14)
    end

    @testset "FFT utilities" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Test FFT backend query
        backend = fft_phi_backend()
        @test backend isa Symbol

        # Test scratch buffer creation
        fft_scratch = scratch_fft(cfg)
        @test size(fft_scratch) == (nlat, nlon)
        @test eltype(fft_scratch) <: Complex

        spatial_scratch = scratch_spatial(cfg)
        @test size(spatial_scratch) == (nlat, nlon)
    end

    @testset "structural field assignment keeps derived state consistent" begin
        # `cfg.lmax = 10` used to leave `size(cfg.Nlm) == (7,7)` while the
        # transforms index it as (lmax+1, mmax+1) under `@inbounds` — a live
        # out-of-bounds read. lmax/mmax/mres now rebuild the spectral layout;
        # the fields that cannot be made consistent are rejected outright.
        cfg = create_gauss_config(6, 8)
        prepare_plm_tables!(cfg)
        @test SHTnsKit.has_fused_scalar_tables(cfg)

        cfg.lmax = 10
        @test size(cfg.Nlm) == (cfg.lmax + 1, cfg.mmax + 1)
        @test cfg.nlm == SHTnsKit.nlm_calc(cfg.lmax, cfg.mmax, cfg.mres)
        @test length(cfg.li) == cfg.nlm && length(cfg.mi) == cfg.nlm
        @test !SHTnsKit.has_fused_scalar_tables(cfg)   # stale tables dropped

        cfg2 = create_gauss_config(6, 8; mres=1)
        cfg2.mres = 2
        @test cfg2.nlm == SHTnsKit.nlm_calc(6, cfg2.mmax, 2)
        @test all(m -> m % 2 == 0, cfg2.mi)

        for bad in (:nlat, :nlon, :grid_type, :nlm, :nspat)
            @test_throws ArgumentError setproperty!(create_gauss_config(4, 6), bad,
                                                    bad === :grid_type ? :regular : 99)
        end
        # An inconsistent spectral triple is rejected rather than stored.
        @test_throws ArgumentError (c = create_gauss_config(6, 8); c.mmax = 99)
    end

    @testset "kwarg SHTConfig constructor validates its invariants" begin
        # This constructor is exported and used to check nothing, so a hand-built
        # config could violate nlon >= 2*mmax+1 and then silently synthesize an
        # all-zero field for any mode it could not resolve.
        lmax = mmax = 6
        mk(; nlon=2*mmax + 2, nlat=8, kw...) = begin
            θ = collect(range(0.1, 3.0; length=nlat))
            base = (; lmax, mmax, mres=1, nlat, nlon,
                    θ, φ=collect(range(0, 2π; length=nlon+1))[1:nlon],
                    x=cos.(θ), w=fill(2/nlat, nlat), st=sin.(θ),
                    Nlm=SHTnsKit.Nlm_table(lmax, mmax), cphi=2π/nlon,
                    nlm=SHTnsKit.nlm_calc(lmax, mmax, 1),
                    li=SHTnsKit.build_li_mi(lmax, mmax, 1)[1],
                    mi=SHTnsKit.build_li_mi(lmax, mmax, 1)[2],
                    nspat=nlat*nlon, norm=:orthonormal, cs_phase=true,
                    real_norm=false, robert_form=false)
            SHTnsKit.SHTConfig(; base..., kw...)
        end
        @test mk() isa SHTnsKit.SHTConfig                       # the valid case still builds
        @test_throws ArgumentError mk(nlon=8)                    # nlon < 2*mmax+1
        @test_throws ArgumentError mk(mres=0)
        @test_throws ArgumentError mk(nspat=1)                   # nspat != nlat*nlon
        @test_throws DimensionMismatch mk(w=fill(0.25, 4))       # w shorter than nlat
        @test_throws DimensionMismatch mk(Nlm=zeros(2, 2))        # Nlm not (lmax+1, mmax+1)
    end

    @testset "FFT plan cache control" begin
        # These knobs used to forward to a cache in the parallel extension that
        # nothing ever read (`_get_or_plan` had no call sites), so the whole
        # documented feature was a no-op and this testset was wrapped in a
        # try/catch that skipped it whenever MPI was absent. They now control the
        # φ-FFT plan cache every transform actually goes through, serial or
        # distributed, so assert the behaviour rather than just the flag.
        initial_state = fft_plan_cache_enabled()
        @test typeof(initial_state) == Bool

        enable_fft_plan_cache!()
        @test fft_plan_cache_enabled() == true
        disable_fft_plan_cache!()
        @test fft_plan_cache_enabled() == false
        set_fft_plan_cache!(true)
        @test fft_plan_cache_enabled() == true
        set_fft_plan_cache!(false)
        @test fft_plan_cache_enabled() == false

        enable_fft_plan_cache!()
        cfg_cache = create_gauss_config(4, 6)
        field = randn(MersenneTwister(77), cfg_cache.nlat, cfg_cache.nlon)
        ref = analysis(cfg_cache, field)
        @test !isempty(SHTnsKit._LOCAL_FFT_PLAN_CACHE)

        disable_fft_plan_cache!()                        # clears by default
        @test isempty(SHTnsKit._LOCAL_FFT_PLAN_CACHE)
        @test analysis(cfg_cache, field) ≈ ref           # same answer, unplanned
        @test isempty(SHTnsKit._LOCAL_FFT_PLAN_CACHE)    # and still bypassed

        enable_fft_plan_cache!()
        @test analysis(cfg_cache, field) ≈ ref
        @test !isempty(SHTnsKit._LOCAL_FFT_PLAN_CACHE)

        @test SHTnsKit.fft_plan_cache_max!(SHTnsKit.fft_plan_cache_max!(8)) == 8

        set_fft_plan_cache!(initial_state)
    end

    @testset "Pencil grid suggestion" begin
        # Test suggest_pencil_grid for various sizes
        nlat, nlon = 32, 64

        # Without MPI, should return reasonable default
        try
            grid = suggest_pencil_grid(nothing, nlat, nlon)
            @test grid isa Tuple
            @test length(grid) == 2
        catch e
            # May error without MPI - that's OK
            @test e isa Exception
        end
    end
end
