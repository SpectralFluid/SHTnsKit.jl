# SHTnsKit.jl - Precomputed Legendre table tests
#
# Covers enable_plm_tables! / disable_plm_tables! (src/config.jl), previously
# untested. Tables are a performance optimization: transforms must be
# numerically identical whether or not the tables are active.

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

function _plm_rand_alm(rng, lmax)
    alm = randn(rng, ComplexF64, lmax + 1, lmax + 1)
    alm[:, 1] .= real.(alm[:, 1])
    for m in 0:lmax, l in 0:(m - 1)
        alm[l + 1, m + 1] = 0
    end
    return alm
end

@testset "Precomputed Legendre tables" begin
    lmax = 6
    nlat = lmax + 2
    nlon = 2 * lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(424)

    alm = _plm_rand_alm(rng, lmax)

    @testset "enable_plm_tables! activates and populates tables" begin
        @test cfg.use_plm_tables == false
        ret = enable_plm_tables!(cfg)        # alias of prepare_plm_tables!
        @test ret === cfg
        @test cfg.use_plm_tables == true
        @test length(cfg.plm_tables) == cfg.mmax + 1
        @test size(cfg.plm_tables[1]) == (lmax + 1, nlat)
        @test !isempty(cfg.NP_tables)
        @test !isempty(cfg.NdP_tables)
    end

    @testset "scalar transforms identical with tables on/off" begin
        # tables currently ON
        f_on = synthesis(cfg, alm; real_output=true)
        a_on = analysis(cfg, f_on)

        disable_plm_tables!(cfg)
        @test cfg.use_plm_tables == false
        @test isempty(cfg.plm_tables)
        @test isempty(cfg.dplm_tables)
        @test isempty(cfg.NP_tables)
        @test isempty(cfg.NdP_tables)

        f_off = synthesis(cfg, alm; real_output=true)
        a_off = analysis(cfg, f_off)

        @test isapprox(f_on, f_off; rtol=1e-10, atol=1e-12)
        @test isapprox(a_on, a_off; rtol=1e-10, atol=1e-12)
    end

    @testset "round-trip accuracy unaffected by table state" begin
        for use_tables in (false, true)
            use_tables ? enable_plm_tables!(cfg) : disable_plm_tables!(cfg)
            f = synthesis(cfg, alm; real_output=true)
            a = analysis(cfg, f)
            @test isapprox(a, alm; rtol=1e-9, atol=1e-11)
        end
    end

    @testset "disable is idempotent" begin
        disable_plm_tables!(cfg)
        ret = disable_plm_tables!(cfg)
        @test ret === cfg
        @test cfg.use_plm_tables == false
    end

    @testset "constructors skip tables above the memory limit" begin
        # The dense tables reached 32 GiB by default at lmax=1023; constructors
        # now compute rows on the fly past a limit, and the transforms agree.
        @test create_regular_config(8, 17).use_plm_tables
        previous = SHTnsKit._PLM_TABLE_AUTO_LIMIT[]
        try
            SHTnsKit._PLM_TABLE_AUTO_LIMIT[] = 0
            large = create_regular_config(8, 17)
            @test !large.use_plm_tables
            @test isempty(large.plm_tables)
            alm = zeros(ComplexF64, 9, 9)
            alm[4, 3] = 0.5 - 0.25im
            @test analysis(large, synthesis(large, alm)) ≈ alm
            @test prepare_plm_tables!(large).use_plm_tables  # explicit request still builds
        finally
            SHTnsKit._PLM_TABLE_AUTO_LIMIT[] = previous
        end
    end
end
