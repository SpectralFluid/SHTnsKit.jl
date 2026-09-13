module SHTnsKitParallelCUDAExt

import SHTnsKit
import MPI
import PencilArrays
import PencilFFTs
import CUDA
import GPUArrays
import GPUArraysCore
import KernelAbstractions
import LinearAlgebra

const CUDAExt = Base.get_extension(SHTnsKit, :SHTnsKitGPUExt)
CUDAExt === nothing && error("SHTnsKitGPUExt must load before its MPI composition extension")

const ParallelExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
ParallelExt === nothing && error("SHTnsKitParallelExt must load before its CUDA composition extension")

const VendorArray = CUDA.AnyCuArray
const VendorArrayType = CUDA.CuArray
const VENDOR_NAME = Val(:cuda)
include("ParallelGPUVendorFirewall.jl")

function _cuda_pinned(::Type{T}, n::Integer) where {T}
    n == 0 && return Vector{T}(undef, 0)
    return CUDA.pin(Vector{T}(undef, n))
end

const CUDA_PARALLEL_ADAPTER = ParallelExt.ParallelGPUAdapter(
    :cuda,
    value -> ParallelExt._parallel_root_buffer(value) isa CUDA.AnyCuArray,
    _ -> CUDA.CuArray,
    value -> CUDA.device(ParallelExt._parallel_root_buffer(value)),
    (f, device) -> CUDA.device!(f, device),
    _ -> MPI.has_cuda(),
    _ -> CUDA.synchronize(),
    _cuda_pinned,
    (host, device) -> copyto!(host, device),
    (device, host) -> copyto!(device, host),
)

function __init__()
    ParallelExt._register_parallel_gpu_adapter!(CUDA_PARALLEL_ADAPTER)
end

@inline _first_m(plan) =
    first(PencilArrays.range_local(PencilArrays.pencil(plan.F_buf))[1]) - 1

function ParallelExt._dist_transpose_gpu_analysis!(
        ::Val{:cuda}, adapter, plan, output, input)
    ParallelExt._gpu_transpose_forward!(
        adapter, plan, plan.F_buf, input,
    )
    ParallelExt._with_owner_device(adapter, output) do
        RT = typeof(real(zero(eltype(output))))
        tables = CUDAExt._cuda_scalar_tables(plan.cfg, RT)
        kernel! = CUDAExt.GPUCommon.distributed_scalar_analysis_kernel!(
            CUDA.CUDABackend(),
        )
        kernel!(parent(output), parent(plan.F_buf), tables.Plm, tables.weights,
                tables.scales, RT(SHTnsKit._analysis_phi_scale(plan.cfg)), _first_m(plan), plan.lmax,
                plan.mmax, plan.cfg.mres, plan.lmax;
                ndrange=size(parent(output)))
        CUDA.synchronize()
    end
    return output
end

function ParallelExt._dist_transpose_gpu_synthesis!(
        ::Val{:cuda}, adapter, plan, output, input)
    ParallelExt._with_owner_device(adapter, output) do
        fill!(parent(plan.F_buf), zero(eltype(plan.F_buf)))
        RT = typeof(real(zero(eltype(input))))
        tables = CUDAExt._cuda_scalar_tables(plan.cfg, RT)
        kernel! = CUDAExt.GPUCommon.distributed_scalar_synthesis_kernel!(
            CUDA.CUDABackend(),
        )
        kernel!(parent(plan.F_buf), parent(input), tables.Plm, tables.scales,
                RT(SHTnsKit.phi_inv_scale(plan.cfg)), _first_m(plan),
                plan.lmax, plan.mmax, plan.cfg.mres;
                ndrange=size(parent(plan.F_buf)))
        CUDA.synchronize()
    end
    ParallelExt._gpu_transpose_inverse!(
        adapter, plan, output, plan.F_buf,
    )
    return output
end

function ParallelExt._dist_transpose_gpu_vector_analysis!(
        ::Val{:cuda}, adapter, plan, Sout, Tout, Vt, Vp)
    ParallelExt._gpu_transpose_forward!(
        adapter, plan, plan.F_buf, Vt,
    )
    ParallelExt._gpu_transpose_forward!(
        adapter, plan, plan.F_buf2, Vp,
    )
    ParallelExt._with_owner_device(adapter, Sout) do
        RT = typeof(real(zero(eltype(Sout))))
        tables = CUDAExt._cuda_vector_tables(plan.cfg, RT)
        kernel! = CUDAExt.GPUCommon.distributed_vector_analysis_kernel!(
            CUDA.CUDABackend(),
        )
        kernel!(parent(Sout), parent(Tout), parent(plan.F_buf),
                parent(plan.F_buf2), tables.dtheta, tables.over_sin,
                tables.weights, tables.scales, tables.x, RT(SHTnsKit._analysis_phi_scale(plan.cfg)),
                _first_m(plan), plan.lmax, plan.mmax, plan.cfg.mres,
                plan.cfg.robert_form;
                ndrange=size(parent(Sout)))
        CUDA.synchronize()
    end
    return Sout, Tout
end

function ParallelExt._dist_transpose_gpu_vector_synthesis!(
        ::Val{:cuda}, adapter, plan, Vt, Vp, Sin, Tin)
    ParallelExt._with_owner_device(adapter, Vt) do
        fill!(parent(plan.F_buf), zero(eltype(plan.F_buf)))
        fill!(parent(plan.F_buf2), zero(eltype(plan.F_buf2)))
        RT = typeof(real(zero(eltype(Sin))))
        tables = CUDAExt._cuda_vector_tables(plan.cfg, RT)
        kernel! = CUDAExt.GPUCommon.distributed_vector_synthesis_kernel!(
            CUDA.CUDABackend(),
        )
        kernel!(parent(plan.F_buf), parent(plan.F_buf2), parent(Sin), parent(Tin),
                tables.dtheta, tables.over_sin, tables.scales, tables.x,
                RT(SHTnsKit.phi_inv_scale(plan.cfg)), _first_m(plan),
                plan.lmax, plan.mmax, plan.cfg.mres, plan.cfg.robert_form;
                ndrange=size(parent(plan.F_buf)))
        CUDA.synchronize()
    end
    ParallelExt._gpu_transpose_inverse!(
        adapter, plan, Vt, plan.F_buf,
    )
    ParallelExt._gpu_transpose_inverse!(
        adapter, plan, Vp, plan.F_buf2,
    )
    return Vt, Vp
end

end
