# SHTnsKit.jl - Extended operator tests
# Covers `mul_ct_matrix`, `st_dt_matrix`, `SH_mul_mx` behaviors beyond what
# `test_operators.jl` exercises: identity by zero matrix, cos(θ) multiplication
# verified spatially, boundary-l handling, dimension checks.

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Operators (extended)" begin
    @testset "mul_ct_matrix / st_dt_matrix dimensions" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        mx = zeros(2 * cfg.nlm)
        @test mul_ct_matrix(cfg, mx) === mx
        @test length(mx) == 2 * cfg.nlm

        mx2 = zeros(2 * cfg.nlm)
        @test st_dt_matrix(cfg, mx2) === mx2

        # Both must error for wrong length
        @test_throws DimensionMismatch mul_ct_matrix(cfg, zeros(2 * cfg.nlm - 1))
        @test_throws DimensionMismatch st_dt_matrix(cfg, zeros(2 * cfg.nlm + 1))
    end

    @testset "SH_mul_mx with zero matrix produces zero" begin
        cfg = create_gauss_config(5, 7; nlon=11)
        rng = MersenneTwister(600)
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Rlm = similar(Qlm)
        SH_mul_mx(cfg, zeros(2 * cfg.nlm), Qlm, Rlm)
        @test all(==(zero(ComplexF64)), Rlm)
    end

    @testset "cosθ multiplication via SH_mul_mx matches grid product" begin
        # Apply the cos(θ) coupling operator to random packed coefficients, then
        # compare against synthesize → multiply by cos(θ) → analyze.
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(601)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        # Ensure m=0 entries are real (real-field packing)
        for lm0 in 0:(cfg.nlm - 1)
            if cfg.mi[lm0 + 1] == 0
                Qlm[lm0 + 1] = real(Qlm[lm0 + 1])
            end
        end

        # Spectral-domain cos(θ) multiplication
        mx = zeros(2 * cfg.nlm)
        mul_ct_matrix(cfg, mx)
        Rlm_spec = similar(Qlm)
        SH_mul_mx(cfg, mx, Qlm, Rlm_spec)

        # Reference: spatial-domain multiplication by x = cos(θ)
        # Unpack Qlm to dense matrix
        Qdense = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        for lm0 in 0:(cfg.nlm - 1)
            l = cfg.li[lm0 + 1]; m = cfg.mi[lm0 + 1]
            Qdense[l + 1, m + 1] = Qlm[lm0 + 1]
        end
        f = synthesis(cfg, Qdense; real_output=true)
        f_ct = similar(f)
        for i in 1:cfg.nlat, j in 1:cfg.nlon
            f_ct[i, j] = cfg.x[i] * f[i, j]
        end
        Rdense = analysis(cfg, f_ct)

        # Pack reference back to packed form
        Rlm_ref = zeros(ComplexF64, cfg.nlm)
        for lm0 in 0:(cfg.nlm - 1)
            l = cfg.li[lm0 + 1]; m = cfg.mi[lm0 + 1]
            Rlm_ref[lm0 + 1] = Rdense[l + 1, m + 1]
        end

        @test isapprox(Rlm_spec, Rlm_ref; rtol=1e-9, atol=1e-11)
    end

    @testset "st_dt_matrix coefficients are antisymmetric in l" begin
        # For the sinθ ∂θ operator, the outgoing coefficients are
        # c_minus = -(l+1) * a_l^m,  c_plus = l * b_l^m.
        # Spot-check by reconstructing a and b, then confirming signs.
        cfg = create_gauss_config(5, 7; nlon=11)
        mx_ct = zeros(2 * cfg.nlm); mul_ct_matrix(cfg, mx_ct)
        mx_st = zeros(2 * cfg.nlm); st_dt_matrix(cfg, mx_st)

        for lm0 in 0:(cfg.nlm - 1)
            l = cfg.li[lm0 + 1]
            a = mx_ct[2 * lm0 + 1]          # cos-op lower coeff
            b = mx_ct[2 * lm0 + 2]          # cos-op upper coeff
            c_minus = mx_st[2 * lm0 + 1]    # sinθ∂θ lower coeff
            c_plus  = mx_st[2 * lm0 + 2]    # sinθ∂θ upper coeff
            @test isapprox(c_minus, -(l + 1) * a; atol=1e-14)
            @test isapprox(c_plus, l * b; atol=1e-14)
        end
    end

    @testset "neighbor operators honor configured coefficient conventions" begin
        lmax = 6
        canonical_cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1,
                                  norm=:schmidt, real_norm=true, cs_phase=false)
        rng = MersenneTwister(602)
        Qcanonical = randn(rng, ComplexF64, canonical_cfg.nlm)
        Qcanonical[1:lmax + 1] .= real.(Qcanonical[1:lmax + 1])
        Qconfigured = similar(Qcanonical)
        SHTnsKit.convert_alm_norm!(Qconfigured, Qcanonical, cfg; to_internal=false)

        for build_matrix! in (mul_ct_matrix, st_dt_matrix)
            canonical_mx = zeros(2 * canonical_cfg.nlm)
            build_matrix!(canonical_cfg, canonical_mx)
            expected_canonical = similar(Qcanonical)
            SH_mul_mx(canonical_cfg, canonical_mx, Qcanonical, expected_canonical)
            expected_configured = similar(Qconfigured)
            SHTnsKit.convert_alm_norm!(expected_configured, expected_canonical, cfg;
                                      to_internal=false)

            configured_mx = zeros(2 * cfg.nlm)
            build_matrix!(cfg, configured_mx)
            got = similar(Qconfigured)
            SH_mul_mx(cfg, configured_mx, Qconfigured, got)
            @test got ≈ expected_configured rtol=2e-12 atol=2e-12
        end
    end

    @testset "SH_mul_mx dimension checks" begin
        cfg = create_gauss_config(3, 5; nlon=7)
        mx = zeros(2 * cfg.nlm)
        Q = zeros(ComplexF64, cfg.nlm)
        R = zeros(ComplexF64, cfg.nlm)

        @test_throws DimensionMismatch SH_mul_mx(cfg, zeros(2 * cfg.nlm - 1), Q, R)
        @test_throws DimensionMismatch SH_mul_mx(cfg, mx, zeros(ComplexF64, cfg.nlm - 1), R)
        @test_throws DimensionMismatch SH_mul_mx(cfg, mx, Q, zeros(ComplexF64, cfg.nlm + 1))
    end
end
