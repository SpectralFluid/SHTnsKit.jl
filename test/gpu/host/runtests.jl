# GPU wrappers and shared kernels on the KernelAbstractions CPU backend. This
# needs no vendor package or hardware, so CI runs it on every push; the vendor
# runners in test/gpu/cuda and test/gpu/amdgpu add the on-device parity.
using Test
using SHTnsKit
using KernelAbstractions

include("../wrapper_reference.jl")
include("../../parity/scalar_full.jl")
include("../../parity/scalar_variants.jl")
include("../../parity/sphtor_full.jl")
include("../../parity/operators.jl")
include("../../parity/rotations.jl")

@testset "Shared GPU kernels on the CPU backend" begin
    common = GPUWrapperReference.GPUCommon
    backend = KernelAbstractions.CPU()
    run_shared_scalar_kernel_reference(common, backend)
    run_shared_scalar_variant_kernel_reference(common, backend)
    run_shared_vector_kernel_reference(common, backend)
    run_shared_operator_kernel_reference(common, backend)
    run_shared_rotation_kernel_reference(common, backend)
end
