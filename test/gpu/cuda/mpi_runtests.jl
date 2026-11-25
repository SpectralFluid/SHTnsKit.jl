using Test
using MPI
using SHTnsKit
using CUDA
using GPUArrays
using GPUArraysCore
import KernelAbstractions
using PencilArrays
using PencilFFTs

MPI.Initialized() || MPI.Init()
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "test", "parity", "mpi_gpu.jl"))

@testset "MPI + CUDA strict storage" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    @test extension !== nothing
    test_mpi_gpu_policy(extension)
    compound = Base.get_extension(SHTnsKit, :SHTnsKitParallelCUDAExt)
    test_mpi_gpu_source_contract(ROOT, :cuda, compound)
    test_mpi_gpu_ordinary_early_errors(extension, compound)
    cuda_functional = CUDA.functional()
    cuda_devices = cuda_functional ? collect(CUDA.devices()) : Any[]
    run_mpi_gpu_full_parity(
        :cuda, CUDA.CuArray, value -> value isa CUDA.AnyCuArray,
        cuda_functional, cuda_devices, CUDA.device!, CUDA.device,
    )
end

MPI.Barrier(MPI.COMM_WORLD)
