# SHTnsKit.jl - FFT Utilities Tests
# Tests for fft_phi, ifft_phi, fft_phi!, ifft_phi!, DFT fallback

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "FFT Utilities" begin
    @testset "Integer real FFT input" begin
        for nlon in (5, 6)
            cfg = create_gauss_config(2, 4; nlon)
            f = [i*j for i in 1:4, j in 1:nlon]
            expected = analysis(cfg, Float64.(f))
            scratch = zeros(ComplexF64, 4, nlon ÷ 2 + 1)
            @test analysis(cfg, f; use_rfft=true) ≈ expected
            @test analysis(cfg, f; use_rfft=true, fft_scratch=scratch) ≈ expected
            @test SHTnsKit._dft_phi(f, -1) ≈ SHTnsKit.fft_phi(Float64.(f))
        end
    end

    @testset "FFT cache settings at process startup" begin
        script = "using SHTnsKit; print(fft_plan_cache_enabled(), ' ', SHTnsKit.fft_plan_cache_max!(3))"
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) -e $script`
        @test read(addenv(cmd, "SHTNSKIT_FFT_PLAN_CACHE" => "0",
                          "SHTNSKIT_CACHE_PENCILFFTS" => "1", "SHTNSKIT_FFT_PLAN_CACHE_MAX" => "3"), String) == "false 3"
        # A malformed cap warns and keeps the default instead of failing to load.
        for value in ("", "unlimited", "1e3")
            log = tempname()
            @test read(pipeline(addenv(cmd, "SHTNSKIT_FFT_PLAN_CACHE" => "1",
                                       "SHTNSKIT_FFT_PLAN_CACHE_MAX" => value); stderr=log),
                       String) == "true 64"
            @test occursin("SHTNSKIT_FFT_PLAN_CACHE_MAX", read(log, String))
            rm(log; force=true)
        end
    end

    @testset "fft_phi/ifft_phi roundtrip" begin
        A = randn(8, 16)
        B = SHTnsKit.fft_phi(A)
        C = SHTnsKit.ifft_phi(B)
        @test isapprox(real.(C), A; rtol=1e-12)
    end

    @testset "fft_phi output dimensions" begin
        nlat, nlon = 10, 20
        A = randn(nlat, nlon)
        B = SHTnsKit.fft_phi(A)
        @test size(B) == (nlat, nlon)
        @test eltype(B) <: Complex
    end

    @testset "fft_phi known signal" begin
        # Single cosine wave: cos(2πk*j/N) has FFT peaks at k and N-k
        nlon = 16
        nlat = 1
        k = 3
        A = zeros(nlat, nlon)
        for j in 0:(nlon-1)
            A[1, j+1] = cos(2π * k * j / nlon)
        end
        B = SHTnsKit.fft_phi(A)
        # FFT of cos gives N/2 at frequency k and N-k
        @test abs(B[1, k+1]) ≈ nlon / 2 atol=1e-10
        @test abs(B[1, nlon - k + 1]) ≈ nlon / 2 atol=1e-10
        # Other frequencies should be near zero
        for m in [1, 2, 5, 6, 7, 8]
            @test abs(B[1, m+1]) < 1e-10
        end
    end

    @testset "fft_phi! in-place" begin
        A = randn(8, 16)
        dest = zeros(ComplexF64, 8, 16)
        SHTnsKit.fft_phi!(dest, A)
        ref = SHTnsKit.fft_phi(A)
        @test isapprox(dest, ref; rtol=1e-12)
    end

    @testset "fft_phi! dimension mismatch" begin
        A = randn(8, 16)
        dest = zeros(ComplexF64, 8, 12)  # wrong size
        @test_throws DimensionMismatch SHTnsKit.fft_phi!(dest, A)
    end

    @testset "ifft_phi! in-place" begin
        A = randn(ComplexF64, 8, 16)
        dest = zeros(ComplexF64, 8, 16)
        SHTnsKit.ifft_phi!(dest, A)
        ref = SHTnsKit.ifft_phi(A)
        @test isapprox(dest, ref; rtol=1e-12)
    end

    @testset "ifft_phi! dimension mismatch" begin
        A = randn(ComplexF64, 8, 16)
        dest = zeros(ComplexF64, 8, 12)
        @test_throws DimensionMismatch SHTnsKit.ifft_phi!(dest, A)
    end

    @testset "fft_phi_backend tracking" begin
        A = randn(4, 8)
        SHTnsKit.fft_phi(A)
        backend = SHTnsKit.fft_phi_backend()
        @test backend in (:fftw, :dft)
    end

    @testset "DFT fallback consistency" begin
        # Direct DFT should match FFT results
        A = randn(4, 8)
        fft_result = SHTnsKit.fft_phi(A)
        dft_result = SHTnsKit._dft_phi(A, -1)
        @test isapprox(fft_result, dft_result; rtol=1e-10)
    end

    @testset "DFT inverse consistency" begin
        A = randn(ComplexF64, 4, 8)
        nlon = size(A, 2)
        forward = SHTnsKit._dft_phi(A, -1)
        inverse = (1 / nlon) * SHTnsKit._dft_phi(forward, +1)
        @test isapprox(inverse, A; rtol=1e-10)
    end

    @testset "fft_phi with complex input" begin
        A = randn(ComplexF64, 6, 12)
        B = SHTnsKit.fft_phi(A)
        C = SHTnsKit.ifft_phi(B)
        @test isapprox(C, A; rtol=1e-12)
    end

    @testset "fft_phi preserves Parseval" begin
        # Parseval's theorem: sum(|x|²) = (1/N) sum(|X|²)
        A = randn(4, 16)
        B = SHTnsKit.fft_phi(A)
        nlon = size(A, 2)
        for i in 1:size(A, 1)
            energy_spatial = sum(abs2, A[i, :])
            energy_spectral = sum(abs2, B[i, :]) / nlon
            @test isapprox(energy_spatial, energy_spectral; rtol=1e-10)
        end
    end
end
