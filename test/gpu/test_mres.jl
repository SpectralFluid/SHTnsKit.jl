using Test
using SHTnsKit
using CUDA

@testset "GPU transforms honor mres" begin
    if !CUDA.functional()
        @test_skip CUDA.functional()
    else
        for options in ((;), (; norm=:schmidt, cs_phase=false, real_norm=true)),
            mres in (2, 3)
            @testset "mres=$mres, normalization=$options" begin
                cfg = create_gauss_config(6, 9; nlon=15, mres, options...)
                full_cfg = create_gauss_config(6, 9; nlon=15, options...)
                shape = (cfg.lmax + 1, cfg.mmax + 1)
                Q = zeros(ComplexF64, shape)
                S = zeros(ComplexF64, shape)
                T = zeros(ComplexF64, shape)
                # Keep a zonal and a nonzero allowed order, plus excluded orders.
                Q[2, 1], Q[mres + 2, mres + 1] = 0.3, 0.4 - 0.2im
                S[3, 1], S[mres + 2, mres + 1] = -0.2, 0.3 + 0.1im
                T[2, 1], T[mres + 1, mres + 1] = 0.1, -0.2 + 0.4im
                Q[3, 2], Q[6, 6] = 0.7 + 0.1im, -0.2 + 0.3im
                S[3, 2], S[6, 6] = 0.2 - 0.5im, 0.1 + 0.2im
                T[2, 2], T[6, 6] = 0.3 + 0.2im, -0.4 + 0.1im
                excluded = [m + 1 for m in 0:cfg.mmax if m % mres != 0]
                allowed = 1:mres:(cfg.mmax + 1)

                for real_output in (true, false)
                    @testset "real_output=$real_output" begin
                        expected = synthesis(cfg, Q; real_output)
                        @test maximum(abs, expected) > 0.01
                        @test gpu_synthesis(cfg, Q; device=SHTnsKit.GPU(), real_output) ≈ expected atol=1e-11 rtol=1e-11

                        expected_θ, expected_φ = synthesis_sphtor(cfg, S, T; real_output)
                        actual_θ, actual_φ = gpu_synthesis_sphtor(cfg, S, T;
                            device=SHTnsKit.GPU(), real_output)
                        @test maximum(abs, expected_θ) > 0.01
                        @test maximum(abs, expected_φ) > 0.01
                        @test actual_θ ≈ expected_θ atol=1e-11 rtol=1e-11
                        @test actual_φ ≈ expected_φ atol=1e-11 rtol=1e-11
                    end
                end

                # Build fields with mres=1 so excluded modes reach analysis.
                spatial = synthesis(full_cfg, Q)
                actual_Q = gpu_analysis(cfg, spatial; device=SHTnsKit.GPU())
                expected_Q = analysis(cfg, spatial)
                @test actual_Q ≈ expected_Q atol=1e-11 rtol=1e-11
                @test all(iszero, actual_Q[:, excluded])
                @test actual_Q[:, allowed] ≈ Q[:, allowed] atol=1e-11 rtol=1e-11

                vθ, vφ = synthesis_sphtor(full_cfg, S, T)
                actual_S, actual_T = gpu_analysis_sphtor(cfg, vθ, vφ; device=SHTnsKit.GPU())
                expected_S, expected_T = analysis_sphtor(cfg, vθ, vφ)
                @test actual_S ≈ expected_S atol=1e-11 rtol=1e-11
                @test actual_T ≈ expected_T atol=1e-11 rtol=1e-11
                @test all(iszero, actual_S[:, excluded])
                @test all(iszero, actual_T[:, excluded])
                @test actual_S[:, allowed] ≈ S[:, allowed] atol=1e-11 rtol=1e-11
                @test actual_T[:, allowed] ≈ T[:, allowed] atol=1e-11 rtol=1e-11
            end
        end
    end
end
