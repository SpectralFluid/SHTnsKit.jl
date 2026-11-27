# SHTnsKit.jl - Edge Case Tests
# Tests for boundary conditions, small configs, and special scenarios

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Edge Cases" begin
    @testset "lmax=0 transforms" begin
        lmax = 0
        nlat = 2
        nlon = 4
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        @test cfg.lmax == 0
        @test cfg.mmax == 0

        # Only l=0, m=0 mode: constant field
        alm = zeros(ComplexF64, 1, 1)
        alm[1, 1] = 3.0 + 0im

        f = synthesis(cfg, alm; real_output=true)
        @test size(f) == (nlat, nlon)

        alm_back = analysis(cfg, f)
        @test isapprox(alm_back[1, 1], alm[1, 1]; rtol=1e-10)
    end

    @testset "lmax=1 transforms" begin
        lmax = 1
        nlat = 3
        nlon = 4
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        alm = zeros(ComplexF64, 2, 2)
        alm[1, 1] = 1.0  # l=0, m=0
        alm[2, 1] = 0.5  # l=1, m=0
        alm[2, 2] = 0.3 + 0.1im  # l=1, m=1

        f = synthesis(cfg, alm; real_output=true)
        alm_back = analysis(cfg, f)
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "mmax < lmax" begin
        lmax = 10
        mmax = 5
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon)

        @test cfg.mmax == mmax
        @test cfg.lmax == lmax

        # Create coefficients with m <= mmax
        rng = MersenneTwister(77)
        alm = zeros(ComplexF64, lmax+1, mmax+1)
        for m in 0:mmax, l in m:lmax
            alm[l+1, m+1] = randn(rng, ComplexF64)
        end
        alm[:, 1] .= real.(alm[:, 1])

        f = synthesis(cfg, alm; real_output=true)
        alm_back = analysis(cfg, f)
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "mres=2 configuration" begin
        lmax = 8
        mres = 2
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; mres=mres, nlon=nlon)

        @test cfg.mres == mres
        @test cfg.nlm == nlm_calc(lmax, lmax, mres)
    end

    @testset "on-the-fly vs table roundtrip" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        rng = MersenneTwister(88)

        # Create test signal
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0.0
        end

        # On-the-fly
        cfg_fly = create_gauss_fly_config(lmax, nlat; nlon=nlon)
        f_fly = synthesis(cfg_fly, alm; real_output=true)
        alm_fly = analysis(cfg_fly, f_fly)

        # Table mode
        cfg_tbl = create_gauss_config(lmax, nlat; nlon=nlon)
        prepare_plm_tables!(cfg_tbl)
        f_tbl = synthesis(cfg_tbl, alm; real_output=true)
        alm_tbl = analysis(cfg_tbl, f_tbl)

        # Both should give same results
        @test isapprox(f_fly, f_tbl; rtol=1e-12)
        @test isapprox(alm_fly, alm_tbl; rtol=1e-12)
    end

    @testset "create_gauss_fly_config properties" begin
        lmax = 8
        nlat = lmax + 2
        cfg = create_gauss_fly_config(lmax, nlat)

        @test cfg.on_the_fly == true
        @test cfg.use_plm_tables == false
        @test isempty(cfg.plm_tables)
        @test isempty(cfg.dplm_tables)
    end

    @testset "Base.copy for SHTConfig" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        cfg_copy = copy(cfg)

        # Should have same values
        @test cfg_copy.lmax == cfg.lmax
        @test cfg_copy.mmax == cfg.mmax
        @test cfg_copy.nlat == cfg.nlat
        @test cfg_copy.nlon == cfg.nlon
        @test cfg_copy.x ≈ cfg.x
        @test cfg_copy.w ≈ cfg.w
        @test cfg_copy.θ ≈ cfg.θ

        # Should be independent (modifying copy doesn't affect original)
        cfg_copy.lmax = nlat - 1
        @test cfg_copy.lmax == lmax + 1
        @test cfg.lmax == lmax
    end

    @testset "config validation errors" begin
        @test_throws ArgumentError create_gauss_config(-1, 2)
        @test_throws ArgumentError create_gauss_config(5, 5; mmax=-1)
        @test_throws ArgumentError create_gauss_config(5, 5; mmax=6)  # mmax > lmax
        @test_throws ArgumentError create_gauss_config(5, 3)  # nlat < lmax+1
    end

    @testset "lmax reassignment respects each grid's latitude minimum" begin
        # The constructors enforce these minimums; assigning cfg.lmax used to
        # check only Gauss grids, so the others silently lost accuracy.
        dh = create_regular_config(8, 18; nlon=40, include_poles=true, use_dh_weights=true)
        @test_throws ArgumentError (dh.lmax = 12)
        @test dh.lmax == 8
        @test_throws ArgumentError (create_regular_config(8, 10).lmax = 9)
        @test_throws ArgumentError (create_regular_config(8, 9; include_poles=true).lmax = 9)
        # Lowering the degree keeps the quadrature exact.
        dh.mmax = 4; dh.lmax = 4
        rng = MersenneTwister(457)
        alm = zeros(ComplexF64, dh.lmax + 1, dh.mmax + 1)
        for m in 0:dh.mmax, l in m:dh.lmax
            alm[l + 1, m + 1] = m == 0 ? randn(rng) : randn(rng, ComplexF64)
        end
        @test analysis(dh, synthesis(dh, alm)) ≈ alm rtol=1e-10
    end

    @testset "south pole first toggle" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        x_orig = copy(cfg.x)
        set_south_pole_first!(cfg)
        @test is_south_pole_first(cfg)

        set_north_pole_first!(cfg)
        @test !is_south_pole_first(cfg)
        @test cfg.x ≈ x_orig
    end

    @testset "large lmax configuration" begin
        lmax = 64
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        @test cfg.lmax == lmax
        @test isapprox(sum(cfg.w), 2.0; rtol=1e-12)
        @test length(cfg.x) == nlat

        # Should still roundtrip correctly
        rng = MersenneTwister(55)
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0.0
        end

        f = synthesis(cfg, alm; real_output=true)
        alm_back = analysis(cfg, f)
        @test isapprox(alm_back, alm; rtol=1e-9, atol=1e-11)
    end

    @testset "purely zonal field (m=0 only)" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        alm = zeros(ComplexF64, lmax+1, lmax+1)
        alm[1, 1] = 1.0  # l=0
        alm[2, 1] = 0.5  # l=1
        alm[5, 1] = 0.3  # l=4

        f = synthesis(cfg, alm; real_output=true)

        # Zonal field: all longitudes at same latitude should be equal
        for i in 1:nlat
            @test all(abs.(f[i, :] .- f[i, 1]) .< 1e-10)
        end

        alm_back = analysis(cfg, f)
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "purely sectoral field (l=m)" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        alm = zeros(ComplexF64, lmax+1, lmax+1)
        alm[4, 4] = 1.0 + 0.5im  # l=3, m=3 (sectoral mode)

        f = synthesis(cfg, alm; real_output=true)
        alm_back = analysis(cfg, f)
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end

    @testset "real coefficients for m=0" begin
        # m=0 coefficients should stay real through roundtrip
        lmax = 6
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        alm = zeros(ComplexF64, lmax+1, lmax+1)
        for l in 0:lmax
            alm[l+1, 1] = Float64(l + 1)  # Real m=0 coefficients
        end

        f = synthesis(cfg, alm; real_output=true)
        alm_back = analysis(cfg, f)

        # m=0 should remain real
        for l in 0:lmax
            @test abs(imag(alm_back[l+1, 1])) < 1e-12
        end
        @test isapprox(alm_back, alm; rtol=1e-10, atol=1e-12)
    end
end
