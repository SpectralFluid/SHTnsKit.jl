using Test
using MPI
using SHTnsKit
using AMDGPU
using GPUArrays
using GPUArraysCore
import KernelAbstractions
using PencilArrays
using PencilFFTs

MPI.Initialized() || MPI.Init()
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "test", "parity", "mpi_gpu.jl"))

@testset "MPI + AMDGPU strict storage" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    @test extension !== nothing
    test_mpi_gpu_policy(extension)
    compound = Base.get_extension(SHTnsKit, :SHTnsKitParallelAMDGPUExt)
    test_mpi_gpu_source_contract(ROOT, :amdgpu, compound)
    test_mpi_gpu_ordinary_early_errors(extension, compound)
    amdgpu_functional = AMDGPU.functional()
    amdgpu_devices = amdgpu_functional ? AMDGPU.devices() : Any[]
    run_mpi_gpu_full_parity(
        :amdgpu, AMDGPU.ROCArray, value -> value isa AMDGPU.AnyROCArray,
        amdgpu_functional, amdgpu_devices, AMDGPU.device!, AMDGPU.device,
    )
end

MPI.Barrier(MPI.COMM_WORLD)
