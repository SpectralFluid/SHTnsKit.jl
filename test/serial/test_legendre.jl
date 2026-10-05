# SHTnsKit.jl - Legendre Polynomial Tests
# Tests for Plm_row!, Plm_and_dPdx_row!, Plm_and_dPdtheta_row!,
# Plm_over_sinth_row!, Plm_dPdtheta_over_sinth_row!, Nlm_table, gausslegendre

using Test
using SHTnsKit
using ForwardDiff: Dual, value, partials

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Legendre Polynomials" begin
    @testset "Plm_row! basic values" begin
        # P_0^0(x) = 1 for all x
        P = zeros(11)
        SHTnsKit.Plm_row!(P, 0.5, 10, 0)
        @test P[1] ≈ 1.0

        # P_1^0(x) = x
        @test P[2] ≈ 0.5

        # P_2^0(x) = (3x²-1)/2
        @test P[3] ≈ (3 * 0.25 - 1) / 2

        # P_3^0(x) = (5x³ - 3x)/2
        x = 0.5
        @test P[4] ≈ (5 * x^3 - 3x) / 2
    end

    @testset "Plm_row! m=1" begin
        x = 0.6
        P = zeros(11)
        SHTnsKit.Plm_row!(P, x, 10, 1)

        # P_1^1(x) = -sqrt(1-x²) (Condon-Shortley)
        @test isapprox(P[2], -sqrt(1 - x^2); rtol=1e-12)

        # P_2^1(x) = -3x*sqrt(1-x²)
        @test isapprox(P[3], -3x * sqrt(1 - x^2); rtol=1e-12)
    end

    @testset "Plm_row! m > lmax returns zeros" begin
        P = zeros(5)
        SHTnsKit.Plm_row!(P, 0.5, 2, 5)
        @test all(P .== 0.0)
    end

    @testset "Plm_row! negative m throws" begin
        P = zeros(5)
        @test_throws ArgumentError SHTnsKit.Plm_row!(P, 0.5, 4, -1)
    end

    @testset "Plm_row! at poles (x=±1)" begin
        P = zeros(11)

        # At x = 1 (north pole)
        SHTnsKit.Plm_row!(P, 1.0, 10, 0)
        # P_l^0(1) = 1 for all l
        for l in 0:10
            @test isapprox(P[l+1], 1.0; atol=1e-12)
        end

        # At x = -1 (south pole)
        SHTnsKit.Plm_row!(P, -1.0, 10, 0)
        # P_l^0(-1) = (-1)^l
        for l in 0:10
            @test isapprox(P[l+1], (-1.0)^l; atol=1e-12)
        end

        # m > 0 at poles: P_l^m(±1) = 0
        SHTnsKit.Plm_row!(P, 1.0, 10, 1)
        for l in 1:10
            @test abs(P[l+1]) < 1e-10
        end
    end

    @testset "Plm_row! orthogonality" begin
        # ∫₋₁¹ P_l^m(x) P_{l'}^m(x) dx = 2/(2l+1) * (l+m)!/(l-m)! * δ_{ll'}
        # Test using Gauss-Legendre quadrature
        lmax = 15
        nlat = lmax + 5
        x_nodes, w_nodes = SHTnsKit.gausslegendre(nlat)

        for m in 0:3
            P1 = zeros(lmax + 1)
            P2 = zeros(lmax + 1)
            for l1 in m:min(lmax, m+4), l2 in m:min(lmax, m+4)
                integral = 0.0
                for (xi, wi) in zip(x_nodes, w_nodes)
                    SHTnsKit.Plm_row!(P1, xi, lmax, m)
                    integral += wi * P1[l1+1] * P1[l2+1]
                end
                if l1 == l2
                    # Expected: 2/(2l+1) * (l+m)!/(l-m)!
                    expected = 2.0 / (2l1 + 1)
                    for k in (l1-m+1):(l1+m)
                        expected *= k
                    end
                    @test isapprox(integral, expected; rtol=1e-10)
                else
                    @test abs(integral) < 1e-10
                end
            end
        end
    end

    @testset "Plm_row! large m (log-space)" begin
        # Test that log-space computation (m > 140) gives same results as direct
        # for intermediate m values where both should work
        x = 0.7
        lmax = 200
        P_large = zeros(lmax + 1)

        # m = 150 (uses log-space path)
        SHTnsKit.Plm_row!(P_large, x, lmax, 150)
        # P_m^m should be nonzero and finite
        @test isfinite(P_large[151])
        @test P_large[151] != 0.0

        # m = 140 (uses direct path, boundary)
        P_direct = zeros(lmax + 1)
        SHTnsKit.Plm_row!(P_direct, x, lmax, 140)
        @test isfinite(P_direct[141])
        @test P_direct[141] != 0.0
    end

    @testset "Plm_row! large m at pole" begin
        P = zeros(201)
        # At pole, P_l^m = 0 for m > 0, even with log-space path
        SHTnsKit.Plm_row!(P, 1.0, 200, 150)
        @test all(abs.(P) .< 1e-10)
    end

    @testset "Plm_and_dPdx_row!" begin
        x = 0.6
        lmax = 10
        P = zeros(lmax + 1)
        dPdx = zeros(lmax + 1)

        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, x, lmax, 0)

        # dP_0^0/dx = 0
        @test dPdx[1] ≈ 0.0

        # dP_1^0/dx = 1
        @test isapprox(dPdx[2], 1.0; atol=1e-12)

        # dP_2^0/dx = 3x
        @test isapprox(dPdx[3], 3x; rtol=1e-12)

        # Verify P values are computed correctly too
        @test P[1] ≈ 1.0
        @test P[2] ≈ x
    end

    @testset "Plm_and_dPdx_row! at poles" begin
        lmax = 8
        P = zeros(lmax + 1)
        dPdx = zeros(lmax + 1)

        # North pole (x=1): dP_l^0/dx = l(l+1)/2
        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, 1.0, lmax, 0)
        for l in 1:lmax
            expected = l * (l + 1) / 2.0
            @test isapprox(dPdx[l+1], expected; rtol=1e-10)
        end

        # South pole (x=-1): dP_l^0/dx = (-1)^{l+1} * l(l+1)/2
        SHTnsKit.Plm_and_dPdx_row!(P, dPdx, -1.0, lmax, 0)
        for l in 1:lmax
            expected = (-1.0)^(l+1) * l * (l + 1) / 2.0
            @test isapprox(dPdx[l+1], expected; rtol=1e-10)
        end
    end

    @testset "Plm_and_dPdtheta_row!" begin
        x = 0.6
        lmax = 10
        P = zeros(lmax + 1)
        dPdtheta = zeros(lmax + 1)

        SHTnsKit.Plm_and_dPdtheta_row!(P, dPdtheta, x, lmax, 0)

        # dP_0^0/dθ = 0
        @test dPdtheta[1] ≈ 0.0

        # Relationship: dP/dθ = -sinθ * dP/dx
        P2 = zeros(lmax + 1)
        dPdx = zeros(lmax + 1)
        SHTnsKit.Plm_and_dPdx_row!(P2, dPdx, x, lmax, 0)
        sinth = sqrt(1 - x^2)
        for l in 0:lmax
            @test isapprox(dPdtheta[l+1], -sinth * dPdx[l+1]; rtol=1e-10)
        end
    end

    @testset "Plm_and_dPdtheta_row! at poles" begin
        lmax = 8
        P = zeros(lmax + 1)
        dPdtheta = zeros(lmax + 1)

        # m=0 at pole: dP/dθ = 0
        SHTnsKit.Plm_and_dPdtheta_row!(P, dPdtheta, 1.0, lmax, 0)
        for l in 0:lmax
            @test abs(dPdtheta[l+1]) < 1e-10
        end

        # m=1 at north pole: dP_l^1/dθ = -l(l+1)/2
        SHTnsKit.Plm_and_dPdtheta_row!(P, dPdtheta, 1.0, lmax, 1)
        for l in 1:lmax
            expected = -l * (l + 1) / 2.0
            @test isapprox(dPdtheta[l+1], expected; rtol=1e-10)
        end

        # m>1 at pole: dP/dθ = 0
        SHTnsKit.Plm_and_dPdtheta_row!(P, dPdtheta, 1.0, lmax, 2)
        for l in 0:lmax
            @test abs(dPdtheta[l+1]) < 1e-10
        end
    end

    @testset "Plm_over_sinth_row!" begin
        x = 0.6
        lmax = 10
        P = zeros(lmax + 1)
        P_over_sinth = zeros(lmax + 1)

        SHTnsKit.Plm_over_sinth_row!(P, P_over_sinth, x, lmax, 1)

        # Should equal P/sin(θ)
        sinth = sqrt(1 - x^2)
        for l in 1:lmax
            @test isapprox(P_over_sinth[l+1], P[l+1] / sinth; rtol=1e-12)
        end
    end

    @testset "Plm_over_sinth_row! at poles" begin
        lmax = 8
        P = zeros(lmax + 1)
        P_over_sinth = zeros(lmax + 1)

        # m=1 at north pole: P_l^1/sinθ = -l(l+1)/2
        SHTnsKit.Plm_over_sinth_row!(P, P_over_sinth, 1.0, lmax, 1)
        for l in 1:lmax
            expected = -l * (l + 1) / 2.0
            @test isapprox(P_over_sinth[l+1], expected; rtol=1e-10)
        end

        # m>1 at pole: P/sinθ = 0
        SHTnsKit.Plm_over_sinth_row!(P, P_over_sinth, 1.0, lmax, 2)
        for l in 0:lmax
            @test abs(P_over_sinth[l+1]) < 1e-10
        end
    end

    @testset "Plm_dPdtheta_over_sinth_row! combined" begin
        x = 0.6
        lmax = 10
        P = zeros(lmax + 1)
        dPdtheta = zeros(lmax + 1)
        P_over_sinth = zeros(lmax + 1)

        SHTnsKit.Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, 1)

        # Compare with separate calls
        P_ref = zeros(lmax + 1)
        dPdtheta_ref = zeros(lmax + 1)
        SHTnsKit.Plm_and_dPdtheta_row!(P_ref, dPdtheta_ref, x, lmax, 1)

        P_ref2 = zeros(lmax + 1)
        Pos_ref = zeros(lmax + 1)
        SHTnsKit.Plm_over_sinth_row!(P_ref2, Pos_ref, x, lmax, 1)

        @test isapprox(P, P_ref; rtol=1e-14)
        @test isapprox(dPdtheta, dPdtheta_ref; rtol=1e-14)
        @test isapprox(P_over_sinth, Pos_ref; rtol=1e-14)
    end

    @testset "Plm_dPdtheta_over_sinth_row! at poles" begin
        lmax = 8
        P = zeros(lmax + 1)
        dPdtheta = zeros(lmax + 1)
        P_over_sinth = zeros(lmax + 1)

        # m=1 at north pole
        SHTnsKit.Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, 1.0, lmax, 1)
        for l in 1:lmax
            @test isapprox(dPdtheta[l+1], -l * (l + 1) / 2.0; rtol=1e-10)
            @test isapprox(P_over_sinth[l+1], -l * (l + 1) / 2.0; rtol=1e-10)
        end

        # m=0 at pole: everything should be zero for derivatives
        SHTnsKit.Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, 1.0, lmax, 0)
        for l in 0:lmax
            @test abs(dPdtheta[l+1]) < 1e-10
        end
    end

    @testset "Plm_dPdtheta_over_sinth_row! lmax < m" begin
        P = zeros(5)
        dPdtheta = zeros(5)
        P_over_sinth = zeros(5)
        SHTnsKit.Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, 0.5, 2, 5)
        @test all(P .== 0.0)
        @test all(dPdtheta .== 0.0)
        @test all(P_over_sinth .== 0.0)
    end

    @testset "Nlm_table properties" begin
        lmax = 10
        mmax = 10
        N = SHTnsKit.Nlm_table(lmax, mmax)

        @test size(N) == (lmax + 1, mmax + 1)

        # N_{l,m} = 0 for l < m
        for m in 1:mmax, l in 0:(m-1)
            @test N[l+1, m+1] == 0.0
        end

        # N_{l,m} > 0 for l >= m
        for m in 0:mmax, l in m:lmax
            @test N[l+1, m+1] > 0.0
        end

        # N_{0,0} = 1/sqrt(4π)
        @test isapprox(N[1, 1], 1 / sqrt(4π); rtol=1e-12)

        # Verify orthonormality: N² * integral of P² = 1/(4π) ... simplified check
        # N_{l,0} = sqrt((2l+1)/(4π))
        for l in 0:lmax
            @test isapprox(N[l+1, 1], sqrt((2l + 1) / (4π)); rtol=1e-12)
        end
    end

    @testset "gausslegendre properties" begin
        for n in [4, 8, 16, 32, 64]
            x, w = SHTnsKit.gausslegendre(n)

            @test length(x) == n
            @test length(w) == n

            # Weights sum to 2
            @test isapprox(sum(w), 2.0; rtol=1e-12)

            # Nodes in [-1, 1]
            @test all(-1.0 .<= x .<= 1.0)

            # Symmetry: x[k] ≈ -x[n-k+1]
            for k in 1:(n ÷ 2)
                @test isapprox(x[k], -x[n - k + 1]; atol=1e-14)
                @test isapprox(w[k], w[n - k + 1]; atol=1e-14)
            end

            # Nodes are sorted
            @test issorted(x)

            # All weights positive
            @test all(w .> 0)
        end

        # n=1 is valid
        x, w = SHTnsKit.gausslegendre(1)
        @test x[1] ≈ 0.0
        @test w[1] ≈ 2.0

        # n=0 throws
        @test_throws ArgumentError SHTnsKit.gausslegendre(0)
    end

    @testset "gausslegendre exactness" begin
        # n-point Gauss-Legendre should be exact for polynomials up to degree 2n-1
        n = 8
        x, w = SHTnsKit.gausslegendre(n)

        # ∫₋₁¹ x^k dx = 2/(k+1) if k even, 0 if k odd
        for k in 0:(2n - 1)
            integral = sum(w .* x .^ k)
            if iseven(k)
                expected = 2.0 / (k + 1)
            else
                expected = 0.0
            end
            @test isapprox(integral, expected; atol=1e-12)
        end
    end

    @testset "Plm_norm_row! bounded + matches Nlm*P (lmax<=128)" begin
        for lmax in (16, 64, 128), x in (-0.9, 0.13, 0.77)
            cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1); N = cfg.Nlm
            P = zeros(lmax+1); g = zeros(lmax+1)
            for m in 0:lmax
                SHTnsKit.Plm_row!(P, x, lmax, m)
                SHTnsKit.Plm_norm_row!(g, x, lmax, m)
                @test all(isfinite, @view g[m+1:end])
                for l in m:lmax
                    ref = N[l+1,m+1]*P[l+1]
                    abs(ref) > 1e-9 && @test isapprox(g[l+1], ref; rtol=1e-9)
                end
            end
        end
        cfg = create_gauss_config(512, 514; nlon=1025); g = zeros(513)
        for m in 0:512
            SHTnsKit.Plm_norm_row!(g, 0.31, 512, m)
            @test all(isfinite, @view g[m+1:end]) && maximum(abs, @view g[m+1:end]) < 2.0
        end
    end

    @testset "Plm_norm_and_dPdtheta_row! matches N*dP (lmax 64, 128)" begin
        x = 0.4273
        for lmax in (64, 128)
            cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1); N = cfg.Nlm
            P = zeros(lmax+1); dP = zeros(lmax+1)
            g = zeros(lmax+1); dg = zeros(lmax+1)
            for m in 0:lmax
                SHTnsKit.Plm_and_dPdtheta_row!(P, dP, x, lmax, m)
                SHTnsKit.Plm_norm_and_dPdtheta_row!(g, dg, x, lmax, m)
                @test all(isfinite, @view dg[m+1:end])
                for l in max(1,m):lmax
                    ref = N[l+1,m+1] * dP[l+1]
                    isfinite(ref) && abs(ref) > 1e-9 &&
                        @test isapprox(dg[l+1], ref; rtol=1e-7)
                end
            end
        end
    end

    @testset "Plm_norm_dPdtheta_over_sinth_row! matches N*dP + N*P/sinθ (lmax 64, 128)" begin
        x = 0.4273
        for lmax in (64, 128)
            cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1); N = cfg.Nlm
            P = zeros(lmax+1); dP = zeros(lmax+1); Pos = zeros(lmax+1)
            g = zeros(lmax+1); dg = zeros(lmax+1); gos = zeros(lmax+1)
            for m in 0:lmax
                SHTnsKit.Plm_dPdtheta_over_sinth_row!(P, dP, Pos, x, lmax, m)
                SHTnsKit.Plm_norm_dPdtheta_over_sinth_row!(g, dg, gos, x, lmax, m)
                @test all(isfinite, @view dg[m+1:end])
                @test all(isfinite, @view gos[m+1:end])
                for l in max(1,m):lmax
                    rd  = N[l+1,m+1] * dP[l+1]
                    ros = N[l+1,m+1] * Pos[l+1]
                    isfinite(rd)  && abs(rd)  > 1e-9 && @test isapprox(dg[l+1],  rd;  rtol=1e-7)
                    isfinite(ros) && abs(ros) > 1e-9 && @test isapprox(gos[l+1], ros; rtol=1e-7)
                end
            end
        end
    end

    @testset "Plm_norm_dPdtheta_over_sinth_row! finite at lmax=512" begin
        lmax = 512
        g = zeros(lmax+1); dg = zeros(lmax+1); gos = zeros(lmax+1)
        for m in 0:lmax
            SHTnsKit.Plm_norm_dPdtheta_over_sinth_row!(g, dg, gos, 0.31, lmax, m)
            @test all(isfinite, @view dg[m+1:end])
            @test all(isfinite, @view gos[m+1:end])
        end
    end

    # The sectoral seed P̄_m^m ∝ sin^m θ leaves the Float64 range (≈1e-308) before
    # the values it seeds come back to O(1); from lmax ≈ 1900 the unscaled
    # recurrence returned values off by up to ~1e150. The reference is the plain
    # recurrence in BigFloat, whose exponent range never underflows.
    function _plm_norm_reference(lmax::Int, m::Int, x::BigFloat)
        P = zeros(BigFloat, lmax + 1)
        s = sqrt(1 - x^2)
        pmm = 1 / sqrt(4 * BigFloat(π))
        for k in 1:m
            pmm = -sqrt(BigFloat(2k + 1) / (2k)) * s * pmm
        end
        P[m+1] = pmm
        m < lmax && (P[m+2] = sqrt(BigFloat(2m + 3)) * x * pmm)
        for l in (m+2):lmax
            a = sqrt(BigFloat((2l - 1) * (2l + 1)) / ((l - m) * (l + m)))
            b = sqrt(BigFloat((2l + 1) * (l - 1 - m) * (l - 1 + m)) /
                     ((2l - 3) * (l - m) * (l + m)))
            P[l+1] = a * x * P[l] - b * P[l-1]
        end
        return P
    end

    @testset "Plm_norm_row! keeps the sectoral seed in range at high lmax" begin
        setprecision(BigFloat, 256) do
            for (lmax, m, s) in ((2047, 1190, 0.5), (2500, 1500, 0.6),
                                 (3000, 1743, 0.5), (4000, 3000, 0.05))
                x = sqrt(1 - s^2)
                ref = _plm_norm_reference(lmax, m, BigFloat(x))
                scale = max(1.0, Float64(maximum(abs, ref)))
                P = zeros(lmax + 1)
                SHTnsKit.Plm_norm_row!(P, x, lmax, m)
                @test maximum(abs.(P .- Float64.(ref))) <= 1e-11 * scale

                # The derivative rows are built from the same extended row.
                dP = zeros(lmax + 1); Pos = zeros(lmax + 1); Q = zeros(lmax + 1)
                SHTnsKit.Plm_norm_and_dPdtheta_row!(Q, dP, x, lmax, m)
                @test maximum(abs.(Q .- Float64.(ref))) <= 1e-11 * scale
                @test all(isfinite, dP)
                SHTnsKit.Plm_norm_dPdtheta_over_sinth_row!(Q, dP, Pos, x, lmax, m)
                @test maximum(abs.(Q .- Float64.(ref))) <= 1e-11 * scale
                @test maximum(abs.(Pos .- Float64.(ref) ./ s)) <= 1e-11 * scale / s
            end
        end
    end

    @testset "Plm_norm_row! extended range in Float32 and for dual numbers" begin
        setprecision(BigFloat, 256) do
            # Float32 seeds underflow from lmax ≈ 240: here sin^221 θ ≈ 1e-96.
            lmax, m, s = 600, 221, 0.368
            x = sqrt(1 - s^2)
            ref = Float64.(_plm_norm_reference(lmax, m, BigFloat(x)))
            P32 = zeros(Float32, lmax + 1)
            SHTnsKit.Plm_norm_row!(P32, Float32(x), lmax, m)
            @test maximum(abs, ref) > 0.05          # a non-negligible row
            @test maximum(abs.(P32 .- ref)) <= 1e-4 * maximum(abs, ref)

            # Dual numbers take the generic 2^-32 thresholds through the scaled
            # path; values and x-derivatives must both survive the rescaling.
            lmax, m, s = 600, 500, 0.4
            x = sqrt(1 - s^2)
            Pd = Vector{Dual{Nothing,Float64,1}}(undef, lmax + 1)
            SHTnsKit.Plm_norm_row!(Pd, Dual{Nothing}(x, 1.0), lmax, m)
            h = BigFloat(1) / BigFloat(10)^30
            up = _plm_norm_reference(lmax, m, BigFloat(x) + h)
            dn = _plm_norm_reference(lmax, m, BigFloat(x) - h)
            mid = _plm_norm_reference(lmax, m, BigFloat(x))
            dref = Float64.((up .- dn) ./ (2h))
            @test maximum(abs.(value.(Pd) .- Float64.(mid))) <=
                  1e-11 * max(1.0, Float64(maximum(abs, mid)))
            @test maximum(abs.(map(d -> partials(d)[1], Pd) .- dref)) <=
                  1e-9 * max(1.0, maximum(abs, dref))
        end
    end
end
