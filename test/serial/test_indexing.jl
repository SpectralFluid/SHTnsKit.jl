# SHTnsKit.jl - Indexing Utility Tests
# Tests for nlm calculations and LM index functions

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Indexing Utilities" begin
    @testset "nlm_calc triangular number" begin
        # Test for various lmax values
        for lmax in 1:20
            mmax = lmax
            mres = 1

            # Standard (real) packed size
            nlm = nlm_calc(lmax, mmax, mres)
            expected = (lmax + 1) * (lmax + 2) ÷ 2
            @test nlm == expected
        end
    end

    @testset "nlm_cplx_calc square number" begin
        for lmax in 1:15
            mmax = lmax
            mres = 1

            nlm_c = nlm_cplx_calc(lmax, mmax, mres)
            expected_c = (lmax + 1)^2
            @test nlm_c == expected_c
        end
    end

    @testset "LM_index bounds" begin
        lmax = 10
        mres = 1
        nlm = nlm_calc(lmax, lmax, mres)

        # All indices should be in [0, nlm)
        for m in 0:lmax
            for l in m:lmax
                idx = LM_index(lmax, mres, l, m)
                @test 0 <= idx < nlm
            end
        end
    end

    @testset "LM_index uniqueness" begin
        lmax = 8
        mres = 1
        nlm = nlm_calc(lmax, lmax, mres)

        # No duplicate indices
        indices = Set{Int}()
        count = 0
        for m in 0:lmax
            for l in m:lmax
                idx = LM_index(lmax, mres, l, m)
                @test !(idx in indices)
                push!(indices, idx)
                count += 1
            end
        end
        @test length(indices) == nlm
        @test count == nlm
    end

    @testset "LM_index coverage" begin
        lmax = 6
        mres = 1
        nlm = nlm_calc(lmax, lmax, mres)

        # Every index 0 to nlm-1 should be used exactly once
        index_count = zeros(Int, nlm)
        for m in 0:lmax
            for l in m:lmax
                idx = LM_index(lmax, mres, l, m) + 1  # Julia 1-based
                index_count[idx] += 1
            end
        end
        @test all(index_count .== 1)
    end

    @testset "im_from_lm" begin
        lmax = 8
        mres = 1

        # Verify im_from_lm returns correct m for each lm index
        # Note: im_from_lm signature is (lm, lmax, mres)
        for m in 0:lmax
            for l in m:lmax
                lm = LM_index(lmax, mres, l, m)
                m_recovered = im_from_lm(lm, lmax, mres)
                @test m_recovered == m
            end
        end
    end

    @testset "LiM_index" begin
        lmax = 6
        mres = 1

        # LiM_index should give valid indices
        # Note: LiM_index(lmax, mres, l, im) where l is actual degree and im is reduced m-index
        # Constraint: im*mres ≤ l ≤ lmax
        for im in 0:lmax
            m = im * mres
            for l in m:lmax
                idx = LiM_index(lmax, mres, l, im)
                @test idx >= 0
                @test idx < nlm_calc(lmax, lmax, mres)
            end
        end
    end

    @testset "mres > 1 spacing" begin
        lmax = 12
        mres = 2  # Only even m values

        nlm = nlm_calc(lmax, lmax, mres)

        # Count valid modes
        count = 0
        for m in 0:mres:lmax
            for l in m:lmax
                count += 1
            end
        end

        # Index range should accommodate
        indices = Set{Int}()
        for m in 0:mres:lmax
            for l in m:lmax
                idx = LM_index(lmax, mres, l, m)
                @test 0 <= idx < nlm
                @test !(idx in indices)
                push!(indices, idx)
            end
        end
    end

    @testset "LM_cplx_index negative m" begin
        lmax = 5
        mmax = 5

        # Test that negative m indices are valid and unique
        indices = Set{Int}()
        for l in 0:lmax
            for m in -min(l, mmax):min(l, mmax)
                idx = LM_cplx_index(lmax, mmax, l, m)
                @test 0 <= idx < nlm_cplx_calc(lmax, mmax, 1)
                @test !(idx in indices)
                push!(indices, idx)
            end
        end
    end

    @testset "LM_cplx symmetry" begin
        lmax = 4
        mmax = 4

        # For complex indexing, m and -m should have different indices
        for l in 1:lmax
            for m in 1:min(l, mmax)
                idx_pos = LM_cplx_index(lmax, mmax, l, m)
                idx_neg = LM_cplx_index(lmax, mmax, l, -m)
                @test idx_pos != idx_neg
            end
        end

        # m=0 should give unique index
        for l in 0:lmax
            idx_0 = LM_cplx_index(lmax, mmax, l, 0)
            @test 0 <= idx_0 < nlm_cplx_calc(lmax, mmax, 1)
        end
    end

    @testset "Index consistency across lmax" begin
        # Indices for smaller lmax should be consistent with larger
        for lmax_small in 2:6
            lmax_large = lmax_small + 2
            mres = 1

            for m in 0:lmax_small
                for l in m:lmax_small
                    # Same relative position in triangular array
                    idx_s = LM_index(lmax_small, mres, l, m)
                    @test idx_s >= 0
                    @test idx_s < nlm_calc(lmax_small, lmax_small, mres)
                end
            end
        end
    end

    @testset "Edge cases" begin
        mres = 1

        # lmax = 0
        @test nlm_calc(0, 0, mres) == 1
        @test LM_index(0, mres, 0, 0) == 0

        # lmax = 1
        @test nlm_calc(1, 1, mres) == 3
        @test LM_index(1, mres, 0, 0) == 0
        @test LM_index(1, mres, 1, 0) == 1
        @test LM_index(1, mres, 1, 1) == 2

        # Single mode access
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2)
        @test cfg.nlm == nlm_calc(lmax, lmax, mres)
    end

    @testset "im_from_lm is bounded by mmax, not lmax" begin
        # The bound was `lmax ÷ mres`, so for an mmax < lmax layout a
        # past-the-end index resolved to an order the configuration does not
        # store instead of raising.
        lmax, mmax = 8, 3
        n = nlm_calc(lmax, mmax, 1)
        # every in-range index still maps to its own order
        li, mi = build_li_mi(lmax, mmax, 1)
        for k in 1:n
            @test SHTnsKit.im_from_lm(k - 1, lmax, 1; mmax=mmax) == mi[k]
        end
        # one past the end now raises rather than inventing m = 4
        @test_throws ArgumentError SHTnsKit.im_from_lm(n, lmax, 1; mmax=mmax)
        # default mmax=lmax keeps the old, correct behaviour for full layouts
        @test SHTnsKit.im_from_lm(nlm_calc(lmax, lmax, 1) - 1, lmax, 1) == lmax
    end

end
