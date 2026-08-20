# SHTnsKit.jl - LoopVectorization (turbo) transform tests
#
# Covers the exported SIMD-vectorized helpers from the SHTnsKitLoopVecExt
# extension, previously untested:
#   analysis_turbo, synthesis_turbo, turbo_apply_laplacian!, benchmark_turbo_vs_simd
#
# These are pure-CPU SIMD variants of the standard transforms and must produce
# numerically identical results to the non-turbo baselines.

using Test
using Random
using LinearAlgebra
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

_has_loopvec = try; @eval using LoopVectorization; true; catch; false; end

if !_has_loopvec
    @info "Skipping turbo (LoopVectorization) tests (package not available)"
else
    @testset "Turbo (LoopVectorization) transforms" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2 * lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(717)

        alm = randn(rng, ComplexF64, lmax + 1, lmax + 1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m - 1)
            alm[l + 1, m + 1] = 0
        end
        f = synthesis(cfg, alm; real_output=true)

        @testset "analysis_turbo matches analysis" begin
            @test isapprox(analysis_turbo(cfg, f), analysis(cfg, f); rtol=1e-9, atol=1e-11)
        end

        @testset "synthesis_turbo matches synthesis" begin
            @test isapprox(synthesis_turbo(cfg, alm),
                           synthesis(cfg, alm; real_output=true); rtol=1e-9, atol=1e-11)
        end

        @testset "turbo round-trip" begin
            a_rt = analysis_turbo(cfg, synthesis_turbo(cfg, alm))
            @test isapprox(a_rt, alm; rtol=1e-9, atol=1e-11)
        end

        @testset "configured convention parity" begin
            convention_cases = (
                (:fourpi, false, true),
                (:schmidt, true, false),
                (:orthonormal, true, true),
            )
            for T in (Float32, Float64),
                (norm, real_norm, cs_phase) in convention_cases
                cfgc = create_gauss_config(3, 6; nlon=9, norm, real_norm, cs_phase)
                field = Matrix{T}(undef, cfgc.nlat, cfgc.nlon)
                for j in 1:cfgc.nlon, i in 1:cfgc.nlat
                    phi = T(2pi * (j - 1) / cfgc.nlon)
                    field[i, j] = sqrt(max(zero(T), one(T) - T(cfgc.x[i])^2)) * cos(phi)
                end

                norm_scale = norm === :orthonormal ? one(T) :
                             norm === :fourpi ? sqrt(T(4pi)) : sqrt(T(4pi / 3))
                real_scale = real_norm ? inv(sqrt(T(2))) : one(T)
                phase_scale = cs_phase ? one(T) : -one(T)
                expected_a11 = -sqrt(T(2pi / 3)) /
                               (norm_scale * real_scale * phase_scale)
                atol = T === Float32 ? 3f-5 : 2e-11
                rtol = T === Float32 ? 3f-5 : 2e-10

                got = analysis_turbo(cfgc, field)
                @test got ≈ analysis(cfgc, field) rtol=rtol atol=atol
                @test got[2, 2] ≈ expected_a11 rtol=rtol atol=atol

                requested = zeros(Complex{T}, cfgc.lmax + 1, cfgc.mmax + 1)
                requested[2, 2] = expected_a11
                @test synthesis_turbo(cfgc, requested) ≈ field rtol=rtol atol=atol
                @test analysis_turbo(cfgc, synthesis_turbo(cfgc, requested)) ≈
                      requested rtol=rtol atol=atol
            end
        end

        @testset "turbo_apply_laplacian! (matrix form)" begin
            A = copy(alm)
            turbo_apply_laplacian!(cfg, A)
            ref = similar(alm)
            for m in 0:lmax, l in 0:lmax
                ref[l + 1, m + 1] = -(l * (l + 1)) * alm[l + 1, m + 1]
            end
            @test isapprox(A, ref; rtol=1e-12, atol=1e-14)
        end

        @testset "turbo_apply_laplacian! (packed vector form)" begin
            Qlm = zeros(ComplexF64, cfg.nlm)
            for m in 0:cfg.mmax, l in m:cfg.lmax
                Qlm[LM_index(cfg.lmax, cfg.mres, l, m) + 1] = alm[l + 1, m + 1]
            end
            Q = copy(Qlm)
            turbo_apply_laplacian!(cfg, Q)
            for m in 0:cfg.mmax, l in m:cfg.lmax
                i = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                @test isapprox(Q[i], -(l * (l + 1)) * Qlm[i]; rtol=1e-12, atol=1e-14)
            end
        end

        @testset "benchmark_turbo_vs_simd returns timing report" begin
            b = benchmark_turbo_vs_simd(cfg; trials=1)
            @test haskey(b, :speedup)
            @test haskey(b, :analysis_turbo)
            @test haskey(b, :synthesis_turbo)
            @test b.analysis_turbo > 0
        end
    end
end
