# SHTnsKit.jl - Spectral Operator Tests
# Tests for differential operators in spectral space

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Spectral Operators" begin
    @testset "cos(θ) multiplication operator" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(80)

        # Build operator matrix
        mx = zeros(Float64, 2*cfg.nlm)
        mul_ct_matrix(cfg, mx)

        # Random spectral coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])  # m=0 real

        # Apply operator
        Rlm = zeros(ComplexF64, cfg.nlm)
        SH_mul_mx(cfg, mx, Qlm, Rlm)

        # Verify by spatial multiplication
        f = synthesis_packed(cfg, Qlm)
        f_mat = reshape(f, nlat, nlon)
        f_times_cost = f_mat .* cfg.x  # x = cos(θ)
        Rlm_ref = analysis_packed(cfg, vec(f_times_cost))

        @test isapprox(Rlm, Rlm_ref; rtol=1e-9, atol=1e-11)
    end

    @testset "sin(θ) ∂/∂θ operator structure" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Build operator matrix
        mx = zeros(Float64, 2*cfg.nlm)
        st_dt_matrix(cfg, mx)

        # Coefficients should be non-zero
        @test any(mx .!= 0)

        # Operator structure: couples l to l±1
        # Verify boundary behavior: l=lmax shouldn't couple upward
        for m in 0:cfg.mmax
            lm_max = LM_index(cfg.lmax, cfg.mres, cfg.lmax, m)
            @test mx[2*lm_max + 2] == 0.0  # No coupling to l+1 at lmax
        end

        # l=0 shouldn't couple downward
        lm_0 = LM_index(cfg.lmax, cfg.mres, 0, 0)
        @test mx[2*lm_0 + 1] == 0.0  # No coupling to l-1 at l=0
    end

    @testset "Operator linearity" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(81)

        mx = zeros(Float64, 2*cfg.nlm)
        mul_ct_matrix(cfg, mx)

        # Two random coefficient vectors
        Qlm1 = randn(rng, ComplexF64, cfg.nlm)
        Qlm2 = randn(rng, ComplexF64, cfg.nlm)
        Qlm1[1:lmax+1] .= real.(Qlm1[1:lmax+1])
        Qlm2[1:lmax+1] .= real.(Qlm2[1:lmax+1])

        α, β = 2.5 + 0.3im, -1.2 + 0.7im

        # Apply operator to linear combination
        Qlm_combo = α .* Qlm1 .+ β .* Qlm2
        Rlm_combo = zeros(ComplexF64, cfg.nlm)
        SH_mul_mx(cfg, mx, Qlm_combo, Rlm_combo)

        # Apply operator separately and combine
        Rlm1 = zeros(ComplexF64, cfg.nlm)
        Rlm2 = zeros(ComplexF64, cfg.nlm)
        SH_mul_mx(cfg, mx, Qlm1, Rlm1)
        SH_mul_mx(cfg, mx, Qlm2, Rlm2)
        Rlm_separate = α .* Rlm1 .+ β .* Rlm2

        @test isapprox(Rlm_combo, Rlm_separate; rtol=1e-10, atol=1e-12)
    end

    @testset "Operator on single mode" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        mx = zeros(Float64, 2*cfg.nlm)
        mul_ct_matrix(cfg, mx)

        # Single mode l=3, m=1
        Qlm = zeros(ComplexF64, cfg.nlm)
        idx = LM_index(lmax, cfg.mres, 3, 1) + 1
        Qlm[idx] = 1.0 + 0.5im

        Rlm = zeros(ComplexF64, cfg.nlm)
        SH_mul_mx(cfg, mx, Qlm, Rlm)

        # cos(θ) multiplication couples to l-1 and l+1 only
        # So only modes (2,1) and (4,1) should be non-zero
        for m in 0:cfg.mmax
            for l in m:cfg.lmax
                idx_test = LM_index(lmax, cfg.mres, l, m) + 1
                if (l == 2 && m == 1) || (l == 4 && m == 1)
                    @test abs(Rlm[idx_test]) > 1e-14
                elseif m == 1 && (l == 3)
                    # Original mode might have zero contribution
                else
                    @test abs(Rlm[idx_test]) < 1e-14
                end
            end
        end
    end

    @testset "Derivative operator antisymmetry" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        mx = zeros(Float64, 2*cfg.nlm)
        st_dt_matrix(cfg, mx)

        # The sin(θ) d/dθ operator has specific sign structure
        # c_minus = -(l+1) * a_l^m
        # c_plus = l * b_l^m
        # Check that signs are as expected
        for m in 0:cfg.mmax
            for l in max(1, m):cfg.lmax-1
                lm = LM_index(lmax, cfg.mres, l, m)
                c_minus = mx[2*lm + 1]
                c_plus = mx[2*lm + 2]

                # c_minus should be negative (or zero)
                @test c_minus <= 0
                # c_plus should be positive (or zero)
                @test c_plus >= 0
            end
        end
    end

    @testset "Dense Laplacian fallback" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        @test isdefined(SHTnsKit, :apply_laplacian!)

        if isdefined(SHTnsKit, :apply_laplacian!)
            coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
            coefficients[1, 1] = 2.0
            coefficients[3, 2] = 1.5 - 0.25im
            coefficients[5, 3] = -0.5 + 0.75im
            expected = copy(coefficients)
            expected[1, 1] = 0
            expected[3, 2] *= -6
            expected[5, 3] *= -20

            returned = SHTnsKit.apply_laplacian!(cfg, coefficients)
            @test returned === coefficients
            @test coefficients == expected
            @test_throws DimensionMismatch SHTnsKit.apply_laplacian!(
                cfg,
                zeros(ComplexF64, cfg.lmax, cfg.mmax + 1),
            )
        end
    end
end
