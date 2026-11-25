# SHTnsKit.jl - Pointwise / latitude evaluation tests
# Verifies: synthesis_point matches synthesis at grid nodes, SH_to_lat matches
# a row of the full synthesis, and the complex pointwise variant is finite.

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

function _point_real_field_alm(rng, lmax)
    alm = randn(rng, ComplexF64, lmax + 1, lmax + 1)
    alm[:, 1] .= real.(alm[:, 1])
    for m in 0:lmax, l in 0:(m - 1)
        alm[l + 1, m + 1] = 0
    end
    return alm
end

@testset "Pointwise / latitude evaluation" begin
    @testset "synthesis_point agrees with synthesis at grid nodes" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(700)
        alm = _point_real_field_alm(rng, lmax)

        f = synthesis(cfg, alm; real_output=true)
        # Sample several (i, j) grid nodes
        for (i, j) in ((1, 1), (3, 4), (cfg.nlat, cfg.nlon), (2, cfg.nlon - 1))
            cost = cfg.x[i]
            phi = 2π * (j - 1) / cfg.nlon
            val = synthesis_point(cfg, alm, cost, phi)
            @test isapprox(val, f[i, j]; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "synthesis_point linearity" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(701)
        a = _point_real_field_alm(rng, lmax)
        b = _point_real_field_alm(rng, lmax)
        cost = 0.3; phi = 1.1

        va = synthesis_point(cfg, a, cost, phi)
        vb = synthesis_point(cfg, b, cost, phi)
        vab = synthesis_point(cfg, a .+ b, cost, phi)
        @test isapprox(vab, va + vb; rtol=1e-12, atol=1e-14)
    end

    @testset "SH_to_lat agrees with a row of full synthesis" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(702)
        alm_dense = _point_real_field_alm(rng, lmax)

        # Pack dense → cfg.nlm packed form
        Qlm = zeros(ComplexF64, cfg.nlm)
        for m in 0:cfg.mmax, l in m:lmax
            Qlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1] = alm_dense[l + 1, m + 1]
        end

        # Pick a latitude (use a grid point to compare exactly)
        i_pick = 2
        cost = cfg.x[i_pick]
        lat = SH_to_lat(cfg, Qlm, cost)
        @test length(lat) == cfg.nlon

        f = synthesis(cfg, alm_dense; real_output=true)
        @test isapprox(lat, f[i_pick, :]; rtol=1e-10, atol=1e-12)
    end

    @testset "SH_to_lat truncation (ltr/mtr) matches zero-padded full" begin
        lmax = 8
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(703)
        alm_dense = _point_real_field_alm(rng, lmax)

        Qlm = zeros(ComplexF64, cfg.nlm)
        for m in 0:cfg.mmax, l in m:lmax
            Qlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1] = alm_dense[l + 1, m + 1]
        end

        ltr = lmax - 2; mtr = cfg.mmax - 1
        cost = cfg.x[3]
        lat_tr = SH_to_lat(cfg, Qlm, cost; ltr=ltr, mtr=mtr)

        # Zero modes beyond truncation in dense → reference
        alm_tr = copy(alm_dense)
        for m in 0:cfg.mmax, l in 0:lmax
            if l > ltr || m > mtr
                alm_tr[l + 1, m + 1] = 0
            end
        end
        f_tr = synthesis(cfg, alm_tr; real_output=true)
        @test isapprox(lat_tr, f_tr[3, :]; rtol=1e-10, atol=1e-12)
    end

    @testset "synthesis_point dimension check" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        bad = zeros(ComplexF64, cfg.lmax, cfg.mmax + 1)
        @test_throws DimensionMismatch synthesis_point(cfg, bad, 0.1, 0.2)
    end

    @testset "Scalar local evaluation normalizes coefficients as needed" begin
        lmax = 47
        cfg = create_gauss_config(lmax, lmax+2; norm=:schmidt, cs_phase=false, real_norm=true)
        canonical = create_gauss_config(lmax, lmax+2)
        rng = MersenneTwister(704)
        dense = randn(rng, ComplexF32, lmax+1, lmax+1)
        packed = SHTnsKit.pack_lm(cfg, dense)
        signed = randn(rng, ComplexF32, nlm_cplx_calc(lmax, lmax, 1))
        for (evaluate, coefficients) in (
            ((c, a) -> synthesis_point(c, a, 0.3, 0.7), dense),
            ((c, a) -> synthesis_point_cplx(c, a, 0.3, 0.7), signed),
            ((c, a) -> SH_to_lat(c, a, 0.3; nphi=8, ltr=5), packed),
            ((c, a) -> SH_to_lat_cplx(c, a, 0.3; nphi=8, ltr=5), signed),
        )
            internal = SHTnsKit.convert_alm_norm!(similar(coefficients), coefficients, cfg; to_internal=true)
            @test evaluate(cfg, coefficients) == evaluate(canonical, internal)
            # The result and Legendre row fit well below the size of a coefficient copy.
            @test @allocated(evaluate(cfg, coefficients)) < sizeof(coefficients) ÷ 2
        end
    end
end
