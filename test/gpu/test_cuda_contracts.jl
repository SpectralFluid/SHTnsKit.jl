using Test
using SHTnsKit
using CUDA
using GPUArrays
using GPUArraysCore
using KernelAbstractions

const GPUExt = Base.get_extension(SHTnsKit, :SHTnsKitGPUExt)

struct KATestCuArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.size(a::KATestCuArray) = size(a.data)
Base.getindex(a::KATestCuArray, I...) = getindex(a.data, I...)
Base.setindex!(a::KATestCuArray, value, I...) = setindex!(a.data, value, I...)
KernelAbstractions.get_backend(::KATestCuArray) = KernelAbstractions.CPU()

@testset "CUDA extension contracts" begin
    @test GPUExt !== nothing
    @test SHTnsKit._GPU_LOOP_AVAILABLE[]
    @test SHTnsKit._GPU_KERNEL_LAUNCHER[] === GPUExt._launch_sht_loop!

    @testset "KernelAbstractions loop launcher" begin
        old_available = SHTnsKit._GPU_LOOP_AVAILABLE[]
        old_launcher = SHTnsKit._GPU_KERNEL_LAUNCHER[]
        old_backend = loop_backend()

        try
            set_loop_backend("auto")
            SHTnsKit._enable_gpu_loops!(GPUExt._launch_sht_loop!)

            src = KATestCuArray(reshape(collect(1.0:12.0), 3, 4))
            dest = KATestCuArray(zeros(3, 4))
            holder = (scale = 1.75,)
            offset = -0.5

            SHTnsKit.@sht_loop dest[i, j] = holder.scale * src[i, j] + offset over (i, j) ∈ CartesianIndices(dest)
            @test dest.data ≈ holder.scale .* src.data .+ offset
        finally
            SHTnsKit._GPU_LOOP_AVAILABLE[] = old_available
            SHTnsKit._GPU_KERNEL_LAUNCHER[] = old_launcher
            set_loop_backend(old_backend)
        end
    end

    @testset "Vector transform shape validation precedes CUDA work" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        spatial = zeros(cfg.nlat, cfg.nlon)
        coeffs = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)

        @test_throws DimensionMismatch gpu_analysis_sphtor(
            cfg, zeros(cfg.nlat - 1, cfg.nlon), spatial; device=SHTnsKit.GPU())
        @test_throws DimensionMismatch gpu_analysis_sphtor(
            cfg, spatial, zeros(cfg.nlat, cfg.nlon + 1); device=SHTnsKit.GPU())
        @test_throws DimensionMismatch gpu_synthesis_sphtor(
            cfg, zeros(ComplexF64, cfg.lmax, cfg.mmax + 1), coeffs; device=SHTnsKit.GPU())
        @test_throws DimensionMismatch gpu_synthesis_sphtor(
            cfg, coeffs, zeros(ComplexF64, cfg.lmax + 1, cfg.mmax); device=SHTnsKit.GPU())
    end

    @testset "Vector memory estimate includes all Legendre temporaries" begin
        cfg = create_gauss_config(8, 10; nlon=17)
        spatial_size = cfg.nlat * cfg.nlon * sizeof(ComplexF64)
        coeff_size = (cfg.lmax + 1) * (cfg.mmax + 1) * sizeof(ComplexF64)
        legendre_size = cfg.nlat * (cfg.lmax + 1) * (cfg.mmax + 1) * sizeof(Float64)

        expected = 4 * spatial_size + 2 * coeff_size + 6 * legendre_size
        @test estimate_memory_usage(cfg, :vector) == expected
    end
end
