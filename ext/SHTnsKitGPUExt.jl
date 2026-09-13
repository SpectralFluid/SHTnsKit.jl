module SHTnsKitGPUExt

using SHTnsKit
using KernelAbstractions, GPUArrays, GPUArraysCore
using LinearAlgebra, FFTW

# Import CUDA
using CUDA
using CUDA.CUFFT

include("GPUCommon.jl")
using .GPUCommon: laplacian_kernel!, operator_matrix_kernel!,
                  packed_operator_kernel!, legendre_table_kernel!,
                  scalar_analysis_kernel!, scalar_synthesis_kernel!,
                  coefficient_conversion_kernel!,
                  coefficient_batch_conversion_kernel!, scalar_config_signature,
                  vector_config_signature,
                  real_pack_kernel!, real_unpack_kernel!,
                  mode_analysis_kernel!, mode_synthesis_kernel!,
                  scalar_batch_analysis_kernel!, scalar_batch_synthesis_kernel!,
                  complex_packed_analysis_kernel!, complex_packed_synthesis_kernel!,
                  scalar_host_tables, ScalarTableCache, scalar_cache_lookup,
                  scalar_cache_publish!,
                  scalar_cache_clear!, scalar_cache_size
using .GPUCommon: ScalarWorkspaceCache, scalar_workspace_use!,
                  scalar_workspace_clear!, scalar_workspace_size
using .GPUCommon: vector_derivative_table_kernel!, vector_analysis_kernel!,
                  vector_synthesis_kernel!, vector_diagonal_kernel!,
                  vector_mode_analysis_kernel!, vector_mode_synthesis_kernel!,
                  vector_batch_analysis_kernel!, vector_batch_synthesis_kernel!
using .GPUCommon: local_scalar_kernel!, local_complex_kernel!, local_qst_kernel!
using .GPUCommon: RotationBlockCache, rotation_cache_lookup,
                  rotation_cache_publish!, rotation_cache_clear!,
                  rotation_z_real_kernel!, rotation_real_kernel!,
                  rotation_cplx_kernel!, launch_sht_loop!

# Import functions from SHTnsKit to extend them
import SHTnsKit: gpu_analysis, gpu_synthesis, gpu_analysis_safe, gpu_synthesis_safe,
                 gpu_analysis_sphtor, gpu_synthesis_sphtor,
                 gpu_apply_laplacian!,
                 gpu_memory_info, check_gpu_memory, gpu_clear_cache!,
                 estimate_memory_usage, get_available_gpus, set_gpu_device

# Import device routing functions to extend.
import SHTnsKit: analysis, synthesis, synthesis_cplx, on_device,
                 analysis_packed, synthesis_packed,
                 analysis_packed_l, synthesis_packed_l,
                 analysis_axisym, synthesis_axisym,
                 analysis_axisym_l, synthesis_axisym_l,
                 analysis_packed_ml, synthesis_packed_ml,
                 analysis_packed_cplx, synthesis_packed_cplx,
                 analysis_packed_cplx_l, synthesis_packed_cplx_l,
                 analysis_batch, analysis_batch!,
                 synthesis_batch, synthesis_batch!, synthesis_batch_cplx,
                 analysis_sphtor, analysis_sphtor_cplx,
                 synthesis_sphtor, synthesis_sphtor_cplx,
                 analysis_sphtor_l, synthesis_sphtor_l,
                 synthesis_sphtor_l_cplx, synthesis_sph_l,
                 synthesis_sph_l_cplx, synthesis_tor_l,
                 synthesis_tor_l_cplx, analysis_sphtor_ml,
                 synthesis_sphtor_ml, synthesis_sph_ml, synthesis_tor_ml,
                 synthesis_grad, synthesis_grad_l, synthesis_grad_ml,
                 analysis_qst, analysis_qst_cplx,
                 synthesis_qst, synthesis_qst_cplx,
                 analysis_qst_l, synthesis_qst_l, synthesis_qst_l_cplx,
                 analysis_qst_ml, synthesis_qst_ml,
                 analysis_sphtor_batch, synthesis_sphtor_batch,
                 synthesis_sphtor_batch_cplx,
                 analysis_qst_batch, synthesis_qst_batch,
                 synthesis_qst_batch_cplx,
                 synthesis_sph, synthesis_sph_cplx,
                 synthesis_tor, synthesis_tor_cplx,
                 divergence_from_spheroidal, divergence_from_spheroidal!,
                 spheroidal_from_divergence, spheroidal_from_divergence!,
                 vorticity_from_toroidal, vorticity_from_toroidal!,
                 toroidal_from_vorticity, toroidal_from_vorticity!,
                 mul_ct_matrix, st_dt_matrix, SH_mul_mx,
                 analysis!, synthesis!, analysis_sphtor!, synthesis_sphtor!,
                 synthesis_point, synthesis_point_cplx,
                 SH_to_lat, SH_to_lat_cplx, SHqst_to_point, SHqst_to_lat,
                 SH_to_grad_point,
                 SH_Zrotate, SH_Yrotate, SH_Yrotate90, SH_Xrotate90,
                 shtns_rotation_apply_real, shtns_rotation_apply_cplx,
                 _register_gpu_adapter!, _gpu_adapter_functional,
                 _gpu_adapter_matches, _gpu_adapter_adapt,
                 _gpu_adapter_analysis, _gpu_adapter_synthesis,
                 _gpu_adapter_analysis_sphtor,
                 _gpu_adapter_synthesis_sphtor, _gpu_adapter_clear_cache!
import SHTnsKit: _enable_gpu_loops!

# ============================================================================
# CUDA Backend Integration with device_utils.jl
# ============================================================================

mutable struct CUDAAdapter end
const CUDA_ADAPTER = CUDAAdapter()

function __init__()
    _register_gpu_adapter!(:cuda, CUDA_ADAPTER)
    _enable_gpu_loops!(launch_sht_loop!)
    return nothing
end

_gpu_adapter_functional(::CUDAAdapter) = CUDA.functional()
_gpu_adapter_matches(::CUDAAdapter, ::CUDA.AnyCuArray) = true

function _require_cuda(operation::Symbol, device)
    CUDA.functional() || throw(SHTnsKit.BackendUnavailableError(
        operation,
        "CUDA.jl is loaded but CUDA.functional() is false; configure a working CUDA runtime (CPU fallback is available only through gpu_analysis_safe/gpu_synthesis_safe)",
    ))
    device isa SHTnsKit.GPU || throw(ArgumentError(
        "`$operation` is a strict GPU operation and does not accept CPU(); use a gpu_*_safe wrapper for explicit fallback",
    ))
    return nothing
end

function _gpu_adapter_adapt(::CUDAAdapter, value)
    CUDA.functional() || throw(SHTnsKit.BackendUnavailableError(
        :to_device,
        "CUDA.jl is loaded but CUDA.functional() is false",
    ))
    return CuArray(value)
end

struct CUDAScalarTables{TX,TW,TP,TS}
    x::TX
    weights::TW
    Plm::TP
    scales::TS
end

struct CUDAVectorTables{TX,TW,TS,TD,TO}
    x::TX
    weights::TW
    scales::TS
    dtheta::TD
    over_sin::TO
end

const _CUDA_SCALAR_CACHE = ScalarTableCache(8)
const _CUDA_VECTOR_CACHE = ScalarTableCache(8)
const _CUDA_LOCAL_CACHE = ScalarTableCache(8)
const _CUDA_WORKSPACE_CACHE = ScalarWorkspaceCache(8)
const _rotation_cache = RotationBlockCache(8)

struct CUDARotationBlocks{TO,TV,TS,T}
    offsets::TO
    values::TV
    input_scales::TS
    output_scales::TS
    alpha::T
    gamma::T
end

function _cuda_rotation_blocks(r::SHTRotation,
                               ::Type{T}) where {T<:AbstractFloat}
    device = CUDA.deviceid(CUDA.device())
    key = (device, T, r.lmax, r.mmax, r.α, r.β, r.γ, r.conv,
           r.norm, r.cs_phase, r.real_norm)
    cached = rotation_cache_lookup(_rotation_cache, key)
    cached === nothing || return cached::CUDARotationBlocks
    host = SHTnsKit._rotation_host_blocks(r, T)
    built = CUDARotationBlocks(
        CuArray(host.offsets), CuArray(host.values),
        CuArray(host.input_scales), CuArray(host.output_scales),
        host.alpha, host.gamma,
    )
    return rotation_cache_publish!(CUDA.synchronize, _rotation_cache, key, built)
end

function _cuda_validate_rotation(operation::Symbol,
                                 input::CUDA.AnyCuArray{<:Complex,1},
                                 output::CUDA.AnyCuArray{<:Complex,1},
                                 expected::Int)
    _require_cuda(operation, SHTnsKit.GPU())
    length(input) == expected || throw(DimensionMismatch(
        "rotation input length must be $expected",
    ))
    length(output) == expected || throw(DimensionMismatch(
        "rotation output length must be $expected",
    ))
    eltype(input) === eltype(output) || throw(ArgumentError(
        "rotation input and output element types must match",
    ))
    RT = typeof(real(zero(eltype(input))))
    RT in (Float32, Float64) || throw(ArgumentError(
        "GPU rotations support ComplexF32 and ComplexF64 coefficients",
    ))
    return RT
end

function SH_Zrotate(::SHTnsKit.GPU, cfg::SHTConfig,
                    input::CUDA.AnyCuArray{<:Complex,1}, angle::Real,
                    output::CUDA.AnyCuArray{<:Complex,1})
    RT = _cuda_validate_rotation(:SH_Zrotate, input, output, cfg.nlm)
    isfinite(angle) || throw(ArgumentError("rotation angle must be finite"))
    source = Base.mightalias(input, output) ? copy(input) : input
    orders = CuArray(Int32.(cfg.mi))
    rotation_z_real_kernel!(CUDABackend())(
        output, source, RT(angle), orders; ndrange=cfg.nlm,
    )
    CUDA.synchronize()
    return output
end

SH_Zrotate(cfg::SHTConfig, input::CUDA.AnyCuArray{<:Complex,1}, angle::Real,
           output::CUDA.AnyCuArray{<:Complex,1}) =
    SH_Zrotate(SHTnsKit.GPU(), cfg, input, angle, output)

function shtns_rotation_apply_real(::SHTnsKit.GPU, r::SHTRotation,
                                   input::CUDA.AnyCuArray{<:Complex,1},
                                   output::CUDA.AnyCuArray{<:Complex,1})
    expected = SHTnsKit.nlm_calc(r.lmax, r.mmax, 1)
    RT = _cuda_validate_rotation(
        :shtns_rotation_apply_real, input, output, expected,
    )
    source = Base.mightalias(input, output) ? copy(input) : input
    blocks = _cuda_rotation_blocks(r, RT)
    rotation_real_kernel!(CUDABackend())(
        output, source, blocks.offsets, blocks.values,
        blocks.input_scales, blocks.output_scales,
        blocks.alpha, blocks.gamma, r.lmax, r.mmax; ndrange=expected,
    )
    CUDA.synchronize()
    return output
end

shtns_rotation_apply_real(r::SHTRotation,
                          input::CUDA.AnyCuArray{<:Complex,1},
                          output::CUDA.AnyCuArray{<:Complex,1}) =
    shtns_rotation_apply_real(SHTnsKit.GPU(), r, input, output)

function shtns_rotation_apply_cplx(::SHTnsKit.GPU, r::SHTRotation,
                                   input::CUDA.AnyCuArray{<:Complex,1},
                                   output::CUDA.AnyCuArray{<:Complex,1})
    expected = SHTnsKit.nlm_cplx_calc(r.lmax, r.mmax, 1)
    RT = _cuda_validate_rotation(
        :shtns_rotation_apply_cplx, input, output, expected,
    )
    source = Base.mightalias(input, output) ? copy(input) : input
    blocks = _cuda_rotation_blocks(r, RT)
    rotation_cplx_kernel!(CUDABackend())(
        output, source, blocks.offsets, blocks.values,
        blocks.input_scales, blocks.output_scales,
        blocks.alpha, blocks.gamma, r.lmax, r.mmax; ndrange=expected,
    )
    CUDA.synchronize()
    return output
end

shtns_rotation_apply_cplx(r::SHTRotation,
                          input::CUDA.AnyCuArray{<:Complex,1},
                          output::CUDA.AnyCuArray{<:Complex,1}) =
    shtns_rotation_apply_cplx(SHTnsKit.GPU(), r, input, output)

function SH_Yrotate(::SHTnsKit.GPU, cfg::SHTConfig,
                    input::CUDA.AnyCuArray{<:Complex,1}, angle::Real,
                    output::CUDA.AnyCuArray{<:Complex,1})
    cfg.mres == 1 || throw(ArgumentError(
        "SH_Yrotate requires mres==1 (got mres=$(cfg.mres)); a Y-rotation mixes orders and cannot be represented in an mres-strided layout",
    ))
    r = SHTRotation(
        cfg.lmax, cfg.mmax; β=angle, norm=cfg.norm,
        cs_phase=cfg.cs_phase, real_norm=cfg.real_norm,
    )
    return shtns_rotation_apply_real(SHTnsKit.GPU(), r, input, output)
end

SH_Yrotate(cfg::SHTConfig, input::CUDA.AnyCuArray{<:Complex,1}, angle::Real,
           output::CUDA.AnyCuArray{<:Complex,1}) =
    SH_Yrotate(SHTnsKit.GPU(), cfg, input, angle, output)
SH_Yrotate90(::SHTnsKit.GPU, cfg::SHTConfig,
             input::CUDA.AnyCuArray{<:Complex,1},
             output::CUDA.AnyCuArray{<:Complex,1}) =
    SH_Yrotate(SHTnsKit.GPU(), cfg, input, pi / 2, output)
SH_Yrotate90(cfg::SHTConfig, input::CUDA.AnyCuArray{<:Complex,1},
             output::CUDA.AnyCuArray{<:Complex,1}) =
    SH_Yrotate90(SHTnsKit.GPU(), cfg, input, output)

function SH_Xrotate90(::SHTnsKit.GPU, cfg::SHTConfig,
                      input::CUDA.AnyCuArray{<:Complex,1},
                      output::CUDA.AnyCuArray{<:Complex,1})
    cfg.mres == 1 || throw(ArgumentError(
        "SH_Xrotate90 requires mres==1 (got mres=$(cfg.mres)); an X-rotation mixes orders and cannot be represented in an mres-strided layout",
    ))
    r = SHTRotation(
        cfg.lmax, cfg.mmax;
        norm=cfg.norm, cs_phase=cfg.cs_phase, real_norm=cfg.real_norm,
    )
    SHTnsKit.shtns_rotation_set_angles_ZYZ(r, pi / 2, pi / 2, -pi / 2)
    return shtns_rotation_apply_real(SHTnsKit.GPU(), r, input, output)
end

SH_Xrotate90(cfg::SHTConfig, input::CUDA.AnyCuArray{<:Complex,1},
             output::CUDA.AnyCuArray{<:Complex,1}) =
    SH_Xrotate90(SHTnsKit.GPU(), cfg, input, output)

struct CUDALocalTables{TP,TD,TO,TS}
    Plm::TP
    dtheta::TD
    over_sin::TO
    scales::TS
end

function _cuda_local_tables(cfg::SHTConfig, ::Type{T}, cost::Real) where {T<:AbstractFloat}
    SHTnsKit._validate_local_cost(cost, :local_evaluation)
    x = T(cost)
    device = CUDA.deviceid(CUDA.device())
    identity = objectid(cfg)
    signature = hash((vector_config_signature(cfg), x))
    cached = scalar_cache_lookup(
        _CUDA_LOCAL_CACHE, device, identity, T, signature,
    )
    cached === nothing || return cached
    x_device = CuArray(T[x])
    Nlm = CuArray(T.(cfg.Nlm))
    Plm = CUDA.zeros(T, 1, cfg.lmax + 1, cfg.mmax + 1)
    dtheta = similar(Plm)
    over_sin = similar(Plm)
    vector_derivative_table_kernel!(CUDABackend())(
        Plm, dtheta, over_sin, x_device, Nlm, cfg.lmax, cfg.mmax;
        ndrange=(1, cfg.mmax + 1),
    )
    scales = _cuda_scalar_tables(cfg, T).scales
    built = CUDALocalTables(Plm, dtheta, over_sin, scales)
    return scalar_cache_publish!(CUDA.synchronize,
        _CUDA_LOCAL_CACHE, device, identity, T, signature, built,
    )
end

@inline function _cuda_local_precision(array)
    T = typeof(real(zero(eltype(array))))
    T in (Float32, Float64) || throw(ArgumentError(
        "GPU local evaluation supports ComplexF32 and ComplexF64 coefficients",
    ))
    return T
end

function _cuda_validate_local_arrays(operation::Symbol, arrays...)
    _require_cuda(operation, SHTnsKit.GPU())
    for array in arrays
        array isa CUDA.AnyCuArray || throw(ArgumentError(
            "$operation requires CUDA-owned coefficient storage",
        ))
    end
    first_type = eltype(first(arrays))
    all(array -> eltype(array) === first_type, arrays) || throw(ArgumentError(
        "$operation requires matching coefficient element types",
    ))
    return _cuda_local_precision(first(arrays))
end

function _cuda_local_scalar(cfg, coefficients, cost, phi;
                            nphi::Int=1, ltr::Int=cfg.lmax,
                            mtr::Int=cfg.mmax, complex_layout::Bool=false)
    T = _cuda_validate_local_arrays(
        complex_layout ? :synthesis_point_cplx : :synthesis_point, coefficients,
    )
    SHTnsKit._validate_local_coordinates(cost, phi, :local_evaluation)
    SHTnsKit._validate_local_nphi(nphi, :local_evaluation)
    0 <= ltr <= cfg.lmax || throw(ArgumentError("ltr must be within [0, lmax]"))
    0 <= mtr <= cfg.mmax || throw(ArgumentError("mtr must be within [0, mmax]"))
    complex_layout && cfg.mres != 1 && throw(ArgumentError(
        "complex local evaluation supports mres==1 only",
    ))
    expected = complex_layout ? SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1) : nothing
    if complex_layout
        length(coefficients) == expected || throw(DimensionMismatch("alm length mismatch"))
    else
        size(coefficients) == (cfg.lmax + 1, cfg.mmax + 1) ||
            throw(DimensionMismatch("Qlm must be (lmax+1, mmax+1)"))
    end
    tables = _cuda_local_tables(cfg, T, cost)
    step = nphi == 1 ? zero(T) : T(2pi / nphi)
    if complex_layout
        output = similar(coefficients, Complex{T}, (nphi,))
        local_complex_kernel!(CUDABackend())(
            output, coefficients, tables.Plm, tables.scales,
            T(phi), step, cfg.lmax, cfg.mmax, ltr; ndrange=nphi,
        )
    else
        output = similar(coefficients, T, (nphi,))
        local_scalar_kernel!(CUDABackend())(
            output, coefficients, tables.Plm, tables.scales,
            T(phi), step, cfg.lmax, cfg.mmax, cfg.mres, ltr, mtr;
            ndrange=nphi,
        )
    end
    return nphi == 1 ? reshape(output, ()) : output
end

function _cuda_local_qst(cfg, Q, S, Tlm, cost, phi;
                         nphi::Int=1, ltr::Int=cfg.lmax,
                         mtr::Int=cfg.mmax,
                         has_q::Bool=true, has_s::Bool=true, has_t::Bool=true)
    arrays = has_t ? (Q, S, Tlm) : (Q, S)
    T = _cuda_validate_local_arrays(:SHqst_to_point, arrays...)
    SHTnsKit._validate_local_coordinates(cost, phi, :local_evaluation)
    SHTnsKit._validate_local_nphi(nphi, :local_evaluation)
    0 <= ltr <= cfg.lmax || throw(ArgumentError("ltr must be within [0, lmax]"))
    0 <= mtr <= cfg.mmax || throw(ArgumentError("mtr must be within [0, mmax]"))
    for (name, array) in zip(("Qlm", "Slm", "Tlm"), (Q, S, Tlm))
        ((name == "Tlm" && !has_t) || length(array) == cfg.nlm) ||
            throw(DimensionMismatch("$name length"))
    end
    tables = _cuda_local_tables(cfg, T, cost)
    Vr = similar(Q, T, (nphi,)); Vt = similar(Q, T, (nphi,)); Vp = similar(Q, T, (nphi,))
    step = nphi == 1 ? zero(T) : T(2pi / nphi)
    sinth = sqrt(max(zero(T), one(T) - T(cost)^2))
    local_qst_kernel!(CUDABackend())(
        Vr, Vt, Vp, Q, S, Tlm, tables.Plm, tables.dtheta,
        tables.over_sin, tables.scales, T(phi), step,
        cfg.lmax, cfg.mmax, cfg.mres, ltr, mtr,
        has_q, has_s, has_t, cfg.robert_form, sinth; ndrange=nphi,
    )
    return nphi == 1 ? (reshape(Vr, ()), reshape(Vt, ()), reshape(Vp, ())) :
                        (Vr, Vt, Vp)
end

synthesis_point(::SHTnsKit.GPU, cfg::SHTConfig, coefficients::CUDA.AnyCuArray{<:Complex,2}, cost::Real, phi::Real) =
    _cuda_local_scalar(cfg, coefficients, cost, phi)
synthesis_point(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{<:Complex,2}, cost::Real, phi::Real) =
    synthesis_point(SHTnsKit.GPU(), cfg, coefficients, cost, phi)
synthesis_point_cplx(::SHTnsKit.GPU, cfg::SHTConfig, coefficients::CUDA.AnyCuArray{<:Complex,1}, cost::Real, phi::Real) =
    _cuda_local_scalar(cfg, coefficients, cost, phi; complex_layout=true)
synthesis_point_cplx(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{<:Complex,1}, cost::Real, phi::Real) =
    synthesis_point_cplx(SHTnsKit.GPU(), cfg, coefficients, cost, phi)

SH_to_lat(::SHTnsKit.GPU, cfg::SHTConfig, coefficients::CUDA.AnyCuArray{<:Complex,1}, cost::Real;
          nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) =
    _cuda_local_qst(cfg, coefficients, coefficients, coefficients, cost, zero(cost);
                    nphi, ltr, mtr, has_s=false, has_t=false)[1]
SH_to_lat(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{<:Complex,1}, cost::Real; kwargs...) =
    SH_to_lat(SHTnsKit.GPU(), cfg, coefficients, cost; kwargs...)
SH_to_lat_cplx(::SHTnsKit.GPU, cfg::SHTConfig, coefficients::CUDA.AnyCuArray{<:Complex,1}, cost::Real;
               nphi::Int=cfg.nlon, ltr::Int=cfg.lmax) =
    _cuda_local_scalar(cfg, coefficients, cost, zero(cost); nphi, ltr, complex_layout=true)
SH_to_lat_cplx(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{<:Complex,1}, cost::Real; kwargs...) =
    SH_to_lat_cplx(SHTnsKit.GPU(), cfg, coefficients, cost; kwargs...)
SHqst_to_point(::SHTnsKit.GPU, cfg::SHTConfig, Q::CUDA.AnyCuArray{<:Complex,1}, S::CUDA.AnyCuArray{<:Complex,1}, Tlm::CUDA.AnyCuArray{<:Complex,1}, cost::Real, phi::Real) =
    _cuda_local_qst(cfg, Q, S, Tlm, cost, phi)
SHqst_to_point(cfg::SHTConfig, Q::CUDA.AnyCuArray{<:Complex,1}, S::CUDA.AnyCuArray{<:Complex,1}, Tlm::CUDA.AnyCuArray{<:Complex,1}, cost::Real, phi::Real) =
    SHqst_to_point(SHTnsKit.GPU(), cfg, Q, S, Tlm, cost, phi)
SHqst_to_lat(::SHTnsKit.GPU, cfg::SHTConfig, Q::CUDA.AnyCuArray{<:Complex,1}, S::CUDA.AnyCuArray{<:Complex,1}, Tlm::CUDA.AnyCuArray{<:Complex,1}, cost::Real;
             nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) =
    _cuda_local_qst(cfg, Q, S, Tlm, cost, zero(cost); nphi, ltr, mtr)
SHqst_to_lat(cfg::SHTConfig, Q::CUDA.AnyCuArray{<:Complex,1}, S::CUDA.AnyCuArray{<:Complex,1}, Tlm::CUDA.AnyCuArray{<:Complex,1}, cost::Real; kwargs...) =
    SHqst_to_lat(SHTnsKit.GPU(), cfg, Q, S, Tlm, cost; kwargs...)
SH_to_grad_point(::SHTnsKit.GPU, cfg::SHTConfig, Dr::CUDA.AnyCuArray{<:Complex,1}, S::CUDA.AnyCuArray{<:Complex,1}, cost::Real, phi::Real) =
    _cuda_local_qst(cfg, Dr, S, S, cost, phi; has_t=false)
SH_to_grad_point(cfg::SHTConfig, Dr::CUDA.AnyCuArray{<:Complex,1}, S::CUDA.AnyCuArray{<:Complex,1}, cost::Real, phi::Real) =
    SH_to_grad_point(SHTnsKit.GPU(), cfg, Dr, S, cost, phi)

function _cuda_scalar_tables(cfg::SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    device = CUDA.deviceid(CUDA.device())
    identity = objectid(cfg)
    signature = scalar_config_signature(cfg)
    cached = scalar_cache_lookup(
        _CUDA_SCALAR_CACHE, device, identity, T, signature,
    )
    cached === nothing || return cached

    x_host, weights_host, scales_host = scalar_host_tables(cfg, T)
    x = CuArray(x_host)
    weights = CuArray(weights_host)
    scales = CuArray(scales_host)
    Plm = CUDA.zeros(T, cfg.nlat, cfg.lmax + 1, cfg.mmax + 1)
    backend = CUDABackend()
    kernel! = legendre_table_kernel!(backend)
    kernel!(Plm, x, cfg.lmax, cfg.mmax;
            ndrange=(cfg.nlat, cfg.mmax + 1))
    built = CUDAScalarTables(x, weights, Plm, scales)

    return scalar_cache_publish!(CUDA.synchronize,
        _CUDA_SCALAR_CACHE, device, identity, T, signature, built,
    )
end

function _cuda_vector_tables(cfg::SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    device = CUDA.deviceid(CUDA.device())
    identity = objectid(cfg)
    signature = vector_config_signature(cfg)
    cached = scalar_cache_lookup(
        _CUDA_VECTOR_CACHE, device, identity, T, signature,
    )
    cached === nothing || return cached

    scalar = _cuda_scalar_tables(cfg, T)
    Nlm = CuArray(T.(cfg.Nlm))
    Plm = similar(scalar.Plm)
    dtheta = similar(Plm)
    over_sin = similar(Plm)
    kernel! = vector_derivative_table_kernel!(CUDABackend())
    kernel!(Plm, dtheta, over_sin, scalar.x, Nlm, cfg.lmax, cfg.mmax;
            ndrange=(cfg.nlat, cfg.mmax + 1))
    built = CUDAVectorTables(
        scalar.x, scalar.weights, scalar.scales, dtheta, over_sin,
    )
    return scalar_cache_publish!(CUDA.synchronize,
        _CUDA_VECTOR_CACHE, device, identity, T, signature, built,
    )
end

function _cuda_clear_scalar_cache!(; device=nothing)
    scalar_cache_clear!(_CUDA_SCALAR_CACHE; device)
    scalar_cache_clear!(_CUDA_VECTOR_CACHE; device)
    scalar_cache_clear!(_CUDA_LOCAL_CACHE; device)
    scalar_workspace_clear!(_CUDA_WORKSPACE_CACHE; device)
    rotation_cache_clear!(_rotation_cache; device)
    return nothing
end

function _gpu_adapter_clear_cache!(::CUDAAdapter)
    _cuda_clear_scalar_cache!()
    return nothing
end

function _cuda_workspace_builder(cfg::SHTConfig, ::Type{RT}, nfields::Int,
                                 use_rfft::Bool) where {RT<:AbstractFloat}
    CT = Complex{RT}
    spatial_shape = nfields == 0 ? (cfg.nlat, cfg.nlon) :
                                   (cfg.nlat, cfg.nlon, nfields)
    spectral_shape = nfields == 0 ? (cfg.lmax + 1, cfg.mmax + 1) :
                                    (cfg.lmax + 1, cfg.mmax + 1, nfields)
    canonical = CUDA.zeros(CT, spectral_shape)
    if use_rfft
        real_buffer = CUDA.zeros(RT, spatial_shape)
        half_shape = Base.setindex(spatial_shape, cfg.nlon ÷ 2 + 1, 2)
        fourier = CUDA.zeros(CT, half_shape)
        forward = CUFFT.plan_rfft(real_buffer, 2)
        inverse = CUFFT.plan_irfft(fourier, cfg.nlon, 2)
        return (; canonical, fourier, real_buffer, forward, inverse)
    end
    fourier = CUDA.zeros(CT, spatial_shape)
    forward = CUFFT.plan_fft!(fourier, 2)
    inverse = CUFFT.plan_ifft!(fourier, 2)
    return (; canonical, fourier, real_buffer=nothing, forward, inverse)
end

function _with_cuda_workspace(f, owner, cfg::SHTConfig, ::Type{RT},
                              nfields::Int, use_rfft::Bool) where {RT<:AbstractFloat}
    device = CUDA.deviceid(CUDA.device())
    kind = nfields == 0 ? :scalar : :batch
    shape = (cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres,
             nfields, use_rfft)
    signature = hash((scalar_config_signature(cfg), use_rfft, nfields))
    builder = () -> _cuda_workspace_builder(cfg, RT, nfields, use_rfft)
    return scalar_workspace_use!(
        f, builder, _CUDA_WORKSPACE_CACHE, device, owner, RT,
        kind, shape, signature,
    )
end

function _cuda_vector_workspace_builder(cfg::SHTConfig,
                                        ::Type{RT}) where {RT<:AbstractFloat}
    CT = Complex{RT}
    Ftheta = CUDA.zeros(CT, cfg.nlat, cfg.nlon)
    Fphi = similar(Ftheta)
    return (;
        Ftheta,
        Fphi,
        forward_theta=CUFFT.plan_fft!(Ftheta, 2),
        forward_phi=CUFFT.plan_fft!(Fphi, 2),
        inverse_theta=CUFFT.plan_ifft!(Ftheta, 2),
        inverse_phi=CUFFT.plan_ifft!(Fphi, 2),
    )
end

function _with_cuda_vector_workspace(f, owner, cfg::SHTConfig,
                                     ::Type{RT}) where {RT<:AbstractFloat}
    device = CUDA.deviceid(CUDA.device())
    shape = (cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres)
    signature = hash((scalar_config_signature(cfg), :vector))
    builder = () -> _cuda_vector_workspace_builder(cfg, RT)
    return scalar_workspace_use!(
        f, builder, _CUDA_WORKSPACE_CACHE, device, owner, RT,
        :vector, shape, signature,
    )
end

function _cuda_batch_scratch(cfg::SHTConfig, fft_batch, ::Type{CT},
                             nfields::Int, use_rfft::Bool, operands...) where {CT<:Complex}
    fft_batch === nothing && return nothing
    fft_batch isa CUDA.AnyCuArray || throw(ArgumentError(
        "fft_batch must use CUDA storage",
    ))
    nbins = use_rfft ? cfg.nlon ÷ 2 + 1 : cfg.nlon
    size(fft_batch) == (cfg.nlat, nbins, nfields) || throw(DimensionMismatch(
        "fft_batch size must be ($(cfg.nlat), $nbins, $nfields)",
    ))
    eltype(fft_batch) === CT || throw(ArgumentError(
        "fft_batch must have element type $CT",
    ))
    any(value -> Base.mightalias(fft_batch, value), operands) &&
        throw(ArgumentError("fft_batch must not alias transform input or output"))
    return fft_batch
end

function _cuda_scalar_analysis_direct!(owner, cfg::SHTConfig,
                                       output::CUDA.AnyCuArray,
                                       field::CUDA.AnyCuArray;
                                       use_rfft::Bool=false)
    _require_cuda(:analysis!, SHTnsKit.GPU())
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon))",
    ))
    size(output) == (cfg.lmax + 1, cfg.mmax + 1) ||
        throw(DimensionMismatch("plan analysis output shape mismatch"))
    use_rfft && !(eltype(field) <: Real) && throw(ArgumentError(
        "use_rfft=true requires a real-valued input",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2",
    ))
    RT = typeof(float(real(zero(eltype(field)))))
    CT = Complex{RT}
    eltype(output) <: Complex || throw(ArgumentError("analysis output must be complex"))
    tables = _cuda_scalar_tables(cfg, RT)
    return _with_cuda_workspace(owner, cfg, RT, 0, use_rfft) do workspace
        fourier = workspace.fourier
        if use_rfft
            copyto!(workspace.real_buffer, field)
            mul!(fourier, workspace.forward, workspace.real_buffer)
        else
            copyto!(fourier, field)
            mul!(fourier, workspace.forward, fourier)
        end
        backend = CUDABackend()
        scalar_analysis_kernel!(backend)(
            output, fourier, tables.Plm, tables.weights, RT(SHTnsKit._analysis_phi_scale(cfg)),
            cfg.lmax, cfg.mmax, cfg.mres, cfg.lmax;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        coefficient_conversion_kernel!(backend)(
            output, output, tables.scales, cfg.lmax, cfg.mmax, false;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        CUDA.synchronize()
        output
    end
end

function _cuda_scalar_synthesis_direct!(owner, cfg::SHTConfig,
                                        output::CUDA.AnyCuArray,
                                        coefficients::CUDA.AnyCuArray;
                                        real_output::Bool=true,
                                        use_rfft::Bool=false)
    _require_cuda(:synthesis!, SHTnsKit.GPU())
    size(coefficients) == (cfg.lmax + 1, cfg.mmax + 1) ||
        throw(DimensionMismatch("plan coefficient shape mismatch"))
    size(output) == (cfg.nlat, cfg.nlon) ||
        throw(DimensionMismatch("plan synthesis output shape mismatch"))
    !real_output && eltype(output) <: Real && throw(ArgumentError(
        "real plan output storage requires real_output=true",
    ))
    use_rfft && (!real_output || !(eltype(output) <: Real)) && throw(ArgumentError(
        "use_rfft=true requires real-valued output",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2",
    ))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    tables = _cuda_scalar_tables(cfg, RT)
    return _with_cuda_workspace(owner, cfg, RT, 0, use_rfft) do workspace
        backend = CUDABackend()
        coefficient_conversion_kernel!(backend)(
            workspace.canonical, coefficients, tables.scales,
            cfg.lmax, cfg.mmax, true;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        fill!(workspace.fourier, zero(eltype(workspace.fourier)))
        scalar_synthesis_kernel!(backend)(
            workspace.fourier, workspace.canonical, tables.Plm,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon,
            cfg.lmax, cfg.mmax, cfg.mres, real_output && !use_rfft;
            ndrange=(cfg.nlat, cfg.mmax + 1),
        )
        CUDA.synchronize()
        if use_rfft
            mul!(workspace.real_buffer, workspace.inverse, workspace.fourier)
            copyto!(output, workspace.real_buffer)
        else
            mul!(workspace.fourier, workspace.inverse, workspace.fourier)
            real_output ? (output .= real.(workspace.fourier)) :
                          copyto!(output, workspace.fourier)
        end
        CUDA.synchronize()
        output
    end
end

function _cuda_scalar_analysis(cfg::SHTConfig, field::CUDA.AnyCuArray;
                               use_rfft::Bool=false, fft_scratch=nothing,
                               lcap::Int=cfg.lmax)
    0 ≤ lcap ≤ cfg.lmax || throw(ArgumentError(
        "lcap must satisfy 0 ≤ lcap ≤ lmax=$(cfg.lmax)",
    ))
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon)), got $(size(field))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "CUDA scalar transforms do not accept a host fft_scratch",
    ))
    use_rfft && !(eltype(field) <: Real) && throw(ArgumentError(
        "use_rfft=true requires a real-valued input",
    ))
    RT = typeof(float(real(zero(eltype(field)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    # `use_rfft` is a performance hint. CUDA currently uses the complex CUFFT
    # pipeline for valid real transforms; the mathematical result is identical.
    fourier = CT.(field)
    gpu_fft!(fourier, 2)

    backend = CUDABackend()
    canonical = CUDA.zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    analyze! = scalar_analysis_kernel!(backend)
    analyze!(canonical, fourier, tables.Plm, tables.weights, RT(SHTnsKit._analysis_phi_scale(cfg)),
             cfg.lmax, cfg.mmax, cfg.mres, lcap;
             ndrange=(lcap + 1, min(cfg.mmax, lcap) + 1))

    configured = similar(canonical)
    convert! = coefficient_conversion_kernel!(backend)
    convert!(configured, canonical, tables.scales, cfg.lmax, cfg.mmax, false;
             ndrange=(cfg.lmax + 1, cfg.mmax + 1))
    CUDA.synchronize()
    return configured
end

function _cuda_scalar_synthesis(cfg::SHTConfig, coefficients::CUDA.AnyCuArray;
                                real_output::Bool=true, use_rfft::Bool=false,
                                fft_scratch=nothing, lcap::Int=cfg.lmax)
    0 ≤ lcap ≤ cfg.lmax || throw(ArgumentError(
        "lcap must satisfy 0 ≤ lcap ≤ lmax=$(cfg.lmax)",
    ))
    size(coefficients) == (cfg.lmax + 1, cfg.mmax + 1) || throw(DimensionMismatch(
        "coefficients must have size ($(cfg.lmax + 1), $(cfg.mmax + 1)), got $(size(coefficients))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "CUDA scalar transforms do not accept a host fft_scratch",
    ))
    use_rfft && !real_output && throw(ArgumentError(
        "use_rfft=true implies real_output",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2, got mmax=$(cfg.mmax), nlon=$(cfg.nlon)",
    ))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    configured = CT.(coefficients)
    canonical = similar(configured)
    backend = CUDABackend()
    convert! = coefficient_conversion_kernel!(backend)
    convert!(canonical, configured, tables.scales, cfg.lmax, cfg.mmax, true;
             ndrange=(cfg.lmax + 1, cfg.mmax + 1))

    # Keep one complex CUFFT implementation for both valid modes until a vendor
    # rFFT plan is measurably preferable; `use_rfft` does not alter semantics.
    fourier = CUDA.zeros(CT, cfg.nlat, cfg.nlon)
    synthesize! = scalar_synthesis_kernel!(backend)
    synthesize!(fourier, canonical, tables.Plm, RT(SHTnsKit.phi_inv_scale(cfg)),
                cfg.nlon, lcap, min(cfg.mmax, lcap), cfg.mres, real_output;
                ndrange=(cfg.nlat, min(cfg.mmax, lcap) + 1))
    CUDA.synchronize()
    gpu_ifft!(fourier, 2)
    return real_output ? real.(fourier) : fourier
end

function _gpu_adapter_analysis(::CUDAAdapter, cfg::SHTConfig, field::CUDA.AnyCuArray; kwargs...)
    _require_cuda(:analysis, SHTnsKit.GPU())
    return _cuda_scalar_analysis(cfg, field; kwargs...)
end

function _gpu_adapter_synthesis(::CUDAAdapter, cfg::SHTConfig, coefficients::CUDA.AnyCuArray; kwargs...)
    _require_cuda(:synthesis, SHTnsKit.GPU())
    return _cuda_scalar_synthesis(cfg, coefficients; kwargs...)
end

analysis(cfg::SHTConfig, field::CUDA.AnyCuArray{T,2}; kwargs...) where {T} =
    analysis(SHTnsKit.GPU(), cfg, field; kwargs...)
synthesis(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,2}; kwargs...) where {T} =
    synthesis(SHTnsKit.GPU(), cfg, coefficients; kwargs...)
synthesis_cplx(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,2}) where {T} =
    synthesis_cplx(SHTnsKit.GPU(), cfg, coefficients)

function _cuda_vector_analysis(cfg::SHTConfig,
                               Vt::CUDA.AnyCuArray{T,2},
                               Vp::CUDA.AnyCuArray{R,2};
                               use_rfft::Bool=false,
                               lcap::Int=cfg.lmax) where {T<:Number,R<:Number}
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    Sout = CUDA.zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    Tout = similar(Sout)
    return _cuda_vector_analysis_direct!(
        cfg, cfg, Sout, Tout, Vt, Vp; use_rfft, lcap,
    )
end

function _cuda_vector_analysis_direct!(owner, cfg::SHTConfig,
                                       Sout::CUDA.AnyCuArray,
                                       Tout::CUDA.AnyCuArray,
                                       Vt::CUDA.AnyCuArray{T,2},
                                       Vp::CUDA.AnyCuArray{R,2};
                                       use_rfft::Bool=false,
                                       lcap::Int=cfg.lmax) where {T<:Number,R<:Number}
    _require_cuda(:analysis_sphtor!, SHTnsKit.GPU())
    size(Vt) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "Vt must have size ($(cfg.nlat), $(cfg.nlon))",
    ))
    size(Vp) == size(Vt) || throw(DimensionMismatch("Vp must match Vt"))
    0 <= lcap <= cfg.lmax || throw(ArgumentError("invalid vector degree cap"))
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    size(Sout) == expected || throw(DimensionMismatch("Sout must have size $expected"))
    size(Tout) == expected || throw(DimensionMismatch("Tout must have size $expected"))
    use_rfft && (!(T <: Real) || !(R <: Real)) && throw(ArgumentError(
        "use_rfft=true requires real-valued vector components",
    ))
    RTt = typeof(float(real(zero(T))))
    RTp = typeof(float(real(zero(R))))
    RTt === RTp || throw(ArgumentError(
        "Vt and Vp must use the same Float32/Float64 precision",
    ))
    RT = RTt
    CT = Complex{RT}
    RT in (Float32, Float64) || throw(ArgumentError(
        "vector analysis supports Float32 and Float64 precision",
    ))
    eltype(Sout) === CT && eltype(Tout) === CT || throw(ArgumentError(
        "Sout and Tout must have element type $CT",
    ))
    any(Base.mightalias(a, b) for a in (Sout, Tout), b in (Vt, Vp)) &&
        throw(ArgumentError("vector analysis outputs must not alias inputs"))
    Base.mightalias(Sout, Tout) && throw(ArgumentError(
        "Sout and Tout must not alias each other",
    ))
    tables = _cuda_vector_tables(cfg, RT)
    return _with_cuda_vector_workspace(owner, cfg, RT) do workspace
        copyto!(workspace.Ftheta, Vt)
        copyto!(workspace.Fphi, Vp)
        mul!(workspace.Ftheta, workspace.forward_theta, workspace.Ftheta)
        mul!(workspace.Fphi, workspace.forward_phi, workspace.Fphi)
        fill!(Sout, zero(CT)); fill!(Tout, zero(CT))
        vector_analysis_kernel!(CUDABackend())(
            Sout, Tout, workspace.Ftheta, workspace.Fphi,
            tables.dtheta, tables.over_sin, tables.weights, tables.scales,
            tables.x, RT(SHTnsKit._analysis_phi_scale(cfg)), lcap, min(cfg.mmax, lcap), cfg.mres,
            cfg.robert_form; ndrange=(lcap + 1, min(cfg.mmax, lcap) + 1),
        )
        CUDA.synchronize()
        Sout, Tout
    end
end

function _cuda_vector_synthesis(cfg::SHTConfig,
                                Slm::CUDA.AnyCuArray{T,2},
                                Tlm::CUDA.AnyCuArray{R,2};
                                real_output::Bool=true,
                                use_rfft::Bool=false,
                                lcap::Int=cfg.lmax) where {T<:Number,R<:Number}
    RT = typeof(float(real(zero(T))))
    output_type = real_output ? RT : Complex{RT}
    Vt = CUDA.zeros(output_type, cfg.nlat, cfg.nlon)
    Vp = similar(Vt)
    return _cuda_vector_synthesis_direct!(
        cfg, cfg, Vt, Vp, Slm, Tlm; real_output, use_rfft, lcap,
    )
end

function _cuda_vector_synthesis_direct!(owner, cfg::SHTConfig,
                                        Vt::CUDA.AnyCuArray,
                                        Vp::CUDA.AnyCuArray,
                                        Slm::CUDA.AnyCuArray{T,2},
                                        Tlm::CUDA.AnyCuArray{R,2};
                                        real_output::Bool=true,
                                        use_rfft::Bool=false,
                                        lcap::Int=cfg.lmax) where {T<:Number,R<:Number}
    _require_cuda(:synthesis_sphtor!, SHTnsKit.GPU())
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    size(Slm) == expected || throw(DimensionMismatch("Slm must have size $expected"))
    size(Tlm) == expected || throw(DimensionMismatch("Tlm must have size $expected"))
    0 <= lcap <= cfg.lmax || throw(ArgumentError("invalid vector degree cap"))
    spatial = (cfg.nlat, cfg.nlon)
    size(Vt) == spatial || throw(DimensionMismatch("Vt must have size $spatial"))
    size(Vp) == spatial || throw(DimensionMismatch("Vp must have size $spatial"))
    use_rfft && !real_output && throw(ArgumentError("use_rfft=true implies real_output"))
    RTs = typeof(float(real(zero(T))))
    RTt = typeof(float(real(zero(R))))
    RTs === RTt || throw(ArgumentError(
        "Slm and Tlm must use the same Float32/Float64 precision",
    ))
    RT = RTs
    CT = Complex{RT}
    RT in (Float32, Float64) || throw(ArgumentError(
        "vector synthesis supports Float32 and Float64 precision",
    ))
    output_type = real_output ? RT : CT
    eltype(Vt) === output_type && eltype(Vp) === output_type || throw(ArgumentError(
        "Vt and Vp must have element type $output_type",
    ))
    any(Base.mightalias(a, b) for a in (Vt, Vp), b in (Slm, Tlm)) &&
        throw(ArgumentError("vector synthesis outputs must not alias inputs"))
    Base.mightalias(Vt, Vp) && throw(ArgumentError(
        "Vt and Vp must not alias each other",
    ))
    tables = _cuda_vector_tables(cfg, RT)
    return _with_cuda_vector_workspace(owner, cfg, RT) do workspace
        fill!(workspace.Ftheta, zero(CT))
        fill!(workspace.Fphi, zero(CT))
        vector_synthesis_kernel!(CUDABackend())(
            workspace.Ftheta, workspace.Fphi, Slm, Tlm,
            tables.dtheta, tables.over_sin, tables.scales, tables.x,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, lcap, min(cfg.mmax, lcap),
            cfg.mres, real_output, cfg.robert_form;
            ndrange=(cfg.nlat, min(cfg.mmax, lcap) + 1),
        )
        CUDA.synchronize()
        mul!(workspace.Ftheta, workspace.inverse_theta, workspace.Ftheta)
        mul!(workspace.Fphi, workspace.inverse_phi, workspace.Fphi)
        if real_output
            Vt .= real.(workspace.Ftheta)
            Vp .= real.(workspace.Fphi)
        else
            copyto!(Vt, workspace.Ftheta)
            copyto!(Vp, workspace.Fphi)
        end
        CUDA.synchronize()
        Vt, Vp
    end
end

_gpu_adapter_analysis_sphtor(::CUDAAdapter, cfg::SHTConfig,
                             Vt::CUDA.AnyCuArray, Vp::CUDA.AnyCuArray; kwargs...) =
    _cuda_vector_analysis(cfg, Vt, Vp; kwargs...)
_gpu_adapter_synthesis_sphtor(::CUDAAdapter, cfg::SHTConfig,
                              Slm::CUDA.AnyCuArray, Tlm::CUDA.AnyCuArray; kwargs...) =
    _cuda_vector_synthesis(cfg, Slm, Tlm; kwargs...)

analysis_sphtor(cfg::SHTConfig, Vt::CUDA.AnyCuArray{T,2},
                 Vp::CUDA.AnyCuArray{R,2}; kwargs...) where {T,R} =
    analysis_sphtor(SHTnsKit.GPU(), cfg, Vt, Vp; kwargs...)
analysis_sphtor_cplx(cfg::SHTConfig, Vt::CUDA.AnyCuArray{T,2},
                      Vp::CUDA.AnyCuArray{R,2}) where {T<:Complex,R<:Complex} =
    analysis_sphtor_cplx(SHTnsKit.GPU(), cfg, Vt, Vp)
synthesis_sphtor(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2},
                  Tlm::CUDA.AnyCuArray{R,2}; kwargs...) where {T,R} =
    synthesis_sphtor(SHTnsKit.GPU(), cfg, Slm, Tlm; kwargs...)
synthesis_sphtor_cplx(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2},
                       Tlm::CUDA.AnyCuArray{R,2}) where {T,R} =
    synthesis_sphtor_cplx(SHTnsKit.GPU(), cfg, Slm, Tlm)
synthesis_sph(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2}; kwargs...) where {T} =
    synthesis_sph(SHTnsKit.GPU(), cfg, Slm; kwargs...)
synthesis_sph_cplx(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2}) where {T} =
    synthesis_sph_cplx(SHTnsKit.GPU(), cfg, Slm)
synthesis_tor(cfg::SHTConfig, Tlm::CUDA.AnyCuArray{T,2}; kwargs...) where {T} =
    synthesis_tor(SHTnsKit.GPU(), cfg, Tlm; kwargs...)
synthesis_tor_cplx(cfg::SHTConfig, Tlm::CUDA.AnyCuArray{T,2}) where {T} =
    synthesis_tor_cplx(SHTnsKit.GPU(), cfg, Tlm)
analysis_qst(cfg::SHTConfig, Vr::CUDA.AnyCuArray{T,2},
             Vt::CUDA.AnyCuArray{R,2}, Vp::CUDA.AnyCuArray{S,2};
             kwargs...) where {T,R,S} =
    analysis_qst(SHTnsKit.GPU(), cfg, Vr, Vt, Vp; kwargs...)
analysis_qst_cplx(cfg::SHTConfig, Vr::CUDA.AnyCuArray{T,2},
                  Vt::CUDA.AnyCuArray{R,2},
                  Vp::CUDA.AnyCuArray{S,2}) where {T<:Complex,R<:Complex,S<:Complex} =
    analysis_qst_cplx(SHTnsKit.GPU(), cfg, Vr, Vt, Vp)
synthesis_qst(cfg::SHTConfig, Qlm::CUDA.AnyCuArray{T,2},
              Slm::CUDA.AnyCuArray{R,2}, Tlm::CUDA.AnyCuArray{S,2};
              kwargs...) where {T,R,S} =
    synthesis_qst(SHTnsKit.GPU(), cfg, Qlm, Slm, Tlm; kwargs...)
synthesis_qst_cplx(cfg::SHTConfig, Qlm::CUDA.AnyCuArray{T,2},
                   Slm::CUDA.AnyCuArray{R,2},
                   Tlm::CUDA.AnyCuArray{S,2}) where {T,R,S} =
    synthesis_qst_cplx(SHTnsKit.GPU(), cfg, Qlm, Slm, Tlm)

# --------------------------------------------------------------------------
# Degree/fixed-order/vector/QST variants.  These methods deliberately live in
# the vendor extension so an inferred call can never enter a host implementation.

function analysis_sphtor_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            Vt::CUDA.AnyCuArray{T,2},
                            Vp::CUDA.AnyCuArray{R,2}, ltr::Integer) where {T,R}
    lcap = SHTnsKit._validate_degree_limit(cfg, ltr)
    return _cuda_vector_analysis(cfg, Vt, Vp; lcap)
end
analysis_sphtor_l(cfg::SHTConfig, Vt::CUDA.AnyCuArray{T,2},
                   Vp::CUDA.AnyCuArray{R,2}, ltr::Integer) where {T,R} =
    analysis_sphtor_l(SHTnsKit.GPU(), cfg, Vt, Vp, ltr)

function synthesis_sphtor_l(::SHTnsKit.GPU, cfg::SHTConfig,
                             Slm::CUDA.AnyCuArray{T,2},
                             Tlm::CUDA.AnyCuArray{R,2}, ltr::Integer;
                             real_output::Bool=true) where {T,R}
    lcap = SHTnsKit._validate_degree_limit(cfg, ltr)
    return _cuda_vector_synthesis(cfg, Slm, Tlm; real_output, lcap)
end
synthesis_sphtor_l(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2},
                    Tlm::CUDA.AnyCuArray{R,2}, ltr::Integer; kwargs...) where {T,R} =
    synthesis_sphtor_l(SHTnsKit.GPU(), cfg, Slm, Tlm, ltr; kwargs...)
synthesis_sphtor_l_cplx(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2},
                         Tlm::CUDA.AnyCuArray{R,2}, ltr::Integer) where {T,R} =
    synthesis_sphtor_l(SHTnsKit.GPU(), cfg, Slm, Tlm, ltr; real_output=false)
synthesis_sphtor_l_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                        Slm::CUDA.AnyCuArray{T,2},
                        Tlm::CUDA.AnyCuArray{R,2}, ltr::Integer) where {T,R} =
    synthesis_sphtor_l(SHTnsKit.GPU(), cfg, Slm, Tlm, ltr; real_output=false)

function _cuda_zero_spectrum(reference::CUDA.AnyCuArray, cfg::SHTConfig)
    return CUDA.zeros(eltype(reference), cfg.lmax + 1, cfg.mmax + 1)
end
synthesis_sph_l(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2}, ltr::Integer;
                real_output::Bool=true) where {T} =
    synthesis_sphtor_l(SHTnsKit.GPU(), cfg, Slm, _cuda_zero_spectrum(Slm, cfg),
                       ltr; real_output)
synthesis_sph_l(::SHTnsKit.GPU, cfg::SHTConfig,
                Slm::CUDA.AnyCuArray{T,2}, ltr::Integer;
                kwargs...) where {T} = synthesis_sph_l(cfg, Slm, ltr; kwargs...)
synthesis_sph_l_cplx(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2},
                     ltr::Integer) where {T} =
    synthesis_sph_l(cfg, Slm, ltr; real_output=false)
synthesis_sph_l_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     Slm::CUDA.AnyCuArray{T,2}, ltr::Integer) where {T} =
    synthesis_sph_l(cfg, Slm, ltr; real_output=false)
synthesis_tor_l(cfg::SHTConfig, Tlm::CUDA.AnyCuArray{T,2}, ltr::Integer;
                real_output::Bool=true) where {T} =
    synthesis_sphtor_l(SHTnsKit.GPU(), cfg, _cuda_zero_spectrum(Tlm, cfg), Tlm,
                       ltr; real_output)
synthesis_tor_l(::SHTnsKit.GPU, cfg::SHTConfig,
                Tlm::CUDA.AnyCuArray{T,2}, ltr::Integer;
                kwargs...) where {T} = synthesis_tor_l(cfg, Tlm, ltr; kwargs...)
synthesis_tor_l_cplx(cfg::SHTConfig, Tlm::CUDA.AnyCuArray{T,2},
                     ltr::Integer) where {T} =
    synthesis_tor_l(cfg, Tlm, ltr; real_output=false)
synthesis_tor_l_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     Tlm::CUDA.AnyCuArray{T,2}, ltr::Integer) where {T} =
    synthesis_tor_l(cfg, Tlm, ltr; real_output=false)

function _cuda_vector_mode_analysis(cfg::SHTConfig, stored_im::Integer,
                                    Vt::CUDA.AnyCuArray{T,1},
                                    Vp::CUDA.AnyCuArray{R,1},
                                    ltr::Integer) where {T<:Complex,R<:Complex}
    physical_m, lcap = SHTnsKit._validate_vector_fixed_order(cfg, stored_im, ltr)
    length(Vt) == cfg.nlat || throw(DimensionMismatch("Vt mode length mismatch"))
    length(Vp) == cfg.nlat || throw(DimensionMismatch("Vp mode length mismatch"))
    RTt = typeof(float(real(zero(T)))); RTp = typeof(float(real(zero(R))))
    RTt === RTp || throw(ArgumentError("mode components must have the same precision"))
    RT = RTt; CT = Complex{RT}
    tables = _cuda_vector_tables(cfg, RT)
    Sout = CUDA.zeros(CT, lcap - physical_m + 1)
    Tout = similar(Sout)
    vector_mode_analysis_kernel!(CUDABackend())(
        Sout, Tout, Vt, Vp, tables.dtheta, tables.over_sin,
        tables.weights, tables.scales, tables.x, RT(SHTnsKit._analysis_phi_scale(cfg)), physical_m,
        lcap, cfg.robert_form; ndrange=length(Sout),
    )
    CUDA.synchronize()
    return Sout, Tout
end

function _cuda_vector_mode_synthesis(cfg::SHTConfig, stored_im::Integer,
                                     Sl::CUDA.AnyCuArray{T,1},
                                     Tl::CUDA.AnyCuArray{R,1},
                                     ltr::Integer) where {T<:Complex,R<:Complex}
    physical_m, lcap = SHTnsKit._validate_vector_fixed_order(cfg, stored_im, ltr)
    expected = lcap - physical_m + 1
    length(Sl) == expected || throw(DimensionMismatch("Sl mode length mismatch"))
    length(Tl) == expected || throw(DimensionMismatch("Tl mode length mismatch"))
    RTs = typeof(float(real(zero(T)))); RTt = typeof(float(real(zero(R))))
    RTs === RTt || throw(ArgumentError("mode coefficients must have the same precision"))
    RT = RTs; CT = Complex{RT}
    tables = _cuda_vector_tables(cfg, RT)
    Vt = CUDA.zeros(CT, cfg.nlat); Vp = similar(Vt)
    vector_mode_synthesis_kernel!(CUDABackend())(
        Vt, Vp, Sl, Tl, tables.dtheta, tables.over_sin, tables.scales,
        tables.x, RT(SHTnsKit.phi_inv_scale(cfg)), physical_m, lcap,
        cfg.robert_form; ndrange=cfg.nlat,
    )
    CUDA.synchronize()
    return Vt, Vp
end

analysis_sphtor_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                   Vt::CUDA.AnyCuArray{T,1}, Vp::CUDA.AnyCuArray{R,1},
                   ltr::Integer) where {T<:Complex,R<:Complex} =
    _cuda_vector_mode_analysis(cfg, im, Vt, Vp, ltr)
analysis_sphtor_ml(cfg::SHTConfig, im::Integer, Vt::CUDA.AnyCuArray{T,1},
                   Vp::CUDA.AnyCuArray{R,1}, ltr::Integer) where {T<:Complex,R<:Complex} =
    analysis_sphtor_ml(SHTnsKit.GPU(), cfg, im, Vt, Vp, ltr)
synthesis_sphtor_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                    Sl::CUDA.AnyCuArray{T,1}, Tl::CUDA.AnyCuArray{R,1},
                    ltr::Integer) where {T<:Complex,R<:Complex} =
    _cuda_vector_mode_synthesis(cfg, im, Sl, Tl, ltr)
synthesis_sphtor_ml(cfg::SHTConfig, im::Integer, Sl::CUDA.AnyCuArray{T,1},
                    Tl::CUDA.AnyCuArray{R,1}, ltr::Integer) where {T<:Complex,R<:Complex} =
    synthesis_sphtor_ml(SHTnsKit.GPU(), cfg, im, Sl, Tl, ltr)
synthesis_sph_ml(cfg::SHTConfig, im::Integer, Sl::CUDA.AnyCuArray{T,1},
                 ltr::Integer) where {T<:Complex} =
    synthesis_sphtor_ml(cfg, im, Sl, CUDA.zeros(T, length(Sl)), ltr)
synthesis_sph_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                 Sl::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Complex} =
    synthesis_sph_ml(cfg, im, Sl, ltr)
synthesis_tor_ml(cfg::SHTConfig, im::Integer, Tl::CUDA.AnyCuArray{T,1},
                 ltr::Integer) where {T<:Complex} =
    synthesis_sphtor_ml(cfg, im, CUDA.zeros(T, length(Tl)), Tl, ltr)
synthesis_tor_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                 Tl::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Complex} =
    synthesis_tor_ml(cfg, im, Tl, ltr)
synthesis_grad(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2}; kwargs...) where {T} =
    synthesis_sph(cfg, Slm; kwargs...)
synthesis_grad(::SHTnsKit.GPU, cfg::SHTConfig,
               Slm::CUDA.AnyCuArray{T,2}; kwargs...) where {T} =
    synthesis_grad(cfg, Slm; kwargs...)
synthesis_grad_l(cfg::SHTConfig, Slm::CUDA.AnyCuArray{T,2}, ltr::Integer;
                 kwargs...) where {T} = synthesis_sph_l(cfg, Slm, ltr; kwargs...)
synthesis_grad_l(::SHTnsKit.GPU, cfg::SHTConfig,
                 Slm::CUDA.AnyCuArray{T,2}, ltr::Integer;
                 kwargs...) where {T} = synthesis_grad_l(cfg, Slm, ltr; kwargs...)
synthesis_grad_ml(cfg::SHTConfig, im::Integer, Sl::CUDA.AnyCuArray{T,1},
                  ltr::Integer) where {T<:Complex} =
    synthesis_sph_ml(cfg, im, Sl, ltr)
synthesis_grad_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                  Sl::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Complex} =
    synthesis_grad_ml(cfg, im, Sl, ltr)

analysis_qst_l(::SHTnsKit.GPU, cfg::SHTConfig,
               Vr::CUDA.AnyCuArray{T,2}, Vt::CUDA.AnyCuArray{R,2},
               Vp::CUDA.AnyCuArray{S,2}, ltr::Integer) where {T,R,S} = begin
    lcap = SHTnsKit._validate_degree_limit(cfg, ltr)
    (_cuda_scalar_analysis(cfg, Vr; lcap),
     _cuda_vector_analysis(cfg, Vt, Vp; lcap)...)
end
analysis_qst_l(cfg::SHTConfig, Vr::CUDA.AnyCuArray{T,2},
               Vt::CUDA.AnyCuArray{R,2}, Vp::CUDA.AnyCuArray{S,2},
               ltr::Integer) where {T,R,S} =
    analysis_qst_l(SHTnsKit.GPU(), cfg, Vr, Vt, Vp, ltr)
synthesis_qst_l(::SHTnsKit.GPU, cfg::SHTConfig,
                Q::CUDA.AnyCuArray{T,2}, S::CUDA.AnyCuArray{R,2},
                Tlm::CUDA.AnyCuArray{U,2}, ltr::Integer;
                real_output::Bool=true) where {T,R,U} = begin
    lcap = SHTnsKit._validate_degree_limit(cfg, ltr)
    (_cuda_scalar_synthesis(cfg, Q; real_output, lcap),
     _cuda_vector_synthesis(cfg, S, Tlm; real_output, lcap)...)
end
synthesis_qst_l(cfg::SHTConfig, Q::CUDA.AnyCuArray{T,2},
                S::CUDA.AnyCuArray{R,2}, Tlm::CUDA.AnyCuArray{U,2},
                ltr::Integer; kwargs...) where {T,R,U} =
    synthesis_qst_l(SHTnsKit.GPU(), cfg, Q, S, Tlm, ltr; kwargs...)
synthesis_qst_l_cplx(cfg::SHTConfig, Q::CUDA.AnyCuArray{T,2},
                     S::CUDA.AnyCuArray{R,2}, Tlm::CUDA.AnyCuArray{U,2},
                     ltr::Integer) where {T,R,U} =
    synthesis_qst_l(cfg, Q, S, Tlm, ltr; real_output=false)
synthesis_qst_l_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     Q::CUDA.AnyCuArray{T,2}, S::CUDA.AnyCuArray{R,2},
                     Tlm::CUDA.AnyCuArray{U,2}, ltr::Integer) where {T,R,U} =
    synthesis_qst_l(SHTnsKit.GPU(), cfg, Q, S, Tlm, ltr; real_output=false)
analysis_qst_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                Vr::CUDA.AnyCuArray{T,1}, Vt::CUDA.AnyCuArray{R,1},
                Vp::CUDA.AnyCuArray{U,1}, ltr::Integer) where {T<:Complex,R<:Complex,U<:Complex} = begin
    stored_im, _, lcap = SHTnsKit._validate_stored_order(cfg, im, ltr)
    (analysis_packed_ml(SHTnsKit.GPU(), cfg, stored_im, Vr, lcap),
     _cuda_vector_mode_analysis(cfg, stored_im, Vt, Vp, lcap)...)
end
analysis_qst_ml(cfg::SHTConfig, im::Integer, Vr::CUDA.AnyCuArray{T,1},
                Vt::CUDA.AnyCuArray{R,1}, Vp::CUDA.AnyCuArray{U,1},
                ltr::Integer) where {T<:Complex,R<:Complex,U<:Complex} =
    analysis_qst_ml(SHTnsKit.GPU(), cfg, im, Vr, Vt, Vp, ltr)
synthesis_qst_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                 Q::CUDA.AnyCuArray{T,1}, S::CUDA.AnyCuArray{R,1},
                 Tlm::CUDA.AnyCuArray{U,1}, ltr::Integer) where {T<:Complex,R<:Complex,U<:Complex} = begin
    stored_im, _, lcap = SHTnsKit._validate_stored_order(cfg, im, ltr)
    (synthesis_packed_ml(SHTnsKit.GPU(), cfg, stored_im, Q, lcap),
     _cuda_vector_mode_synthesis(cfg, stored_im, S, Tlm, lcap)...)
end
synthesis_qst_ml(cfg::SHTConfig, im::Integer, Q::CUDA.AnyCuArray{T,1},
                 S::CUDA.AnyCuArray{R,1}, Tlm::CUDA.AnyCuArray{U,1},
                 ltr::Integer) where {T<:Complex,R<:Complex,U<:Complex} =
    synthesis_qst_ml(SHTnsKit.GPU(), cfg, im, Q, S, Tlm, ltr)

function _cuda_vector_batch_analysis(cfg::SHTConfig,
                                     Vt::CUDA.AnyCuArray{T,3},
                                     Vp::CUDA.AnyCuArray{R,3}) where {T<:Real,R<:Real}
    size(Vt, 1) == cfg.nlat && size(Vt, 2) == cfg.nlon ||
        throw(DimensionMismatch("vector batch must start with (nlat, nlon)"))
    size(Vp) == size(Vt) || throw(DimensionMismatch("Vt/Vp batch shape mismatch"))
    nfields = size(Vt, 3)
    nfields > 0 || throw(ArgumentError("analysis_sphtor_batch requires at least one field"))
    RTt = float(T); RTp = float(R)
    RTt === RTp || throw(ArgumentError("vector batches must use the same precision"))
    RT = RTt; CT = Complex{RT}
    tables = _cuda_vector_tables(cfg, RT)
    Ft = CT.(Vt); Fp = CT.(Vp)
    gpu_fft!(Ft, 2); gpu_fft!(Fp, 2)
    Sout = CUDA.zeros(CT, cfg.lmax + 1, cfg.mmax + 1, nfields)
    Tout = similar(Sout)
    vector_batch_analysis_kernel!(CUDABackend())(
        Sout, Tout, Ft, Fp, tables.dtheta, tables.over_sin,
        tables.weights, tables.scales, tables.x, RT(SHTnsKit._analysis_phi_scale(cfg)), cfg.lmax,
        cfg.mmax, cfg.mres, cfg.robert_form; ndrange=size(Sout),
    )
    CUDA.synchronize()
    return Sout, Tout
end

function _cuda_vector_batch_synthesis(cfg::SHTConfig,
                                      S::CUDA.AnyCuArray{T,3},
                                      Tlm::CUDA.AnyCuArray{R,3};
                                      real_output::Bool=true) where {T<:Complex,R<:Complex}
    size(S, 1) == cfg.lmax + 1 && size(S, 2) == cfg.mmax + 1 ||
        throw(DimensionMismatch("vector coefficient batch has wrong spectral shape"))
    size(Tlm) == size(S) || throw(DimensionMismatch("S/T batch shape mismatch"))
    nfields = size(S, 3)
    nfields > 0 || throw(ArgumentError("synthesis_sphtor_batch requires at least one field"))
    RTs = typeof(float(real(zero(T)))); RTt = typeof(float(real(zero(R))))
    RTs === RTt || throw(ArgumentError("vector batches must use the same precision"))
    RT = RTs; CT = Complex{RT}
    tables = _cuda_vector_tables(cfg, RT)
    Ft = CUDA.zeros(CT, cfg.nlat, cfg.nlon, nfields)
    Fp = CUDA.zeros(CT, cfg.nlat, cfg.nlon, nfields)
    vector_batch_synthesis_kernel!(CUDABackend())(
        Ft, Fp, S, Tlm, tables.dtheta, tables.over_sin, tables.scales,
        tables.x, RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, cfg.lmax,
        cfg.mmax, cfg.mres, real_output, cfg.robert_form;
        ndrange=(cfg.nlat, cfg.mmax + 1, nfields),
    )
    CUDA.synchronize()
    gpu_ifft!(Ft, 2); gpu_ifft!(Fp, 2)
    return real_output ? (real.(Ft), real.(Fp)) : (Ft, Fp)
end

analysis_sphtor_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                      Vt::CUDA.AnyCuArray{T,3},
                      Vp::CUDA.AnyCuArray{R,3}) where {T<:Real,R<:Real} =
    _cuda_vector_batch_analysis(cfg, Vt, Vp)
analysis_sphtor_batch(cfg::SHTConfig, Vt::CUDA.AnyCuArray{T,3},
                      Vp::CUDA.AnyCuArray{R,3}) where {T<:Real,R<:Real} =
    analysis_sphtor_batch(SHTnsKit.GPU(), cfg, Vt, Vp)
synthesis_sphtor_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                       S::CUDA.AnyCuArray{T,3},
                       Tlm::CUDA.AnyCuArray{R,3};
                       real_output::Bool=true) where {T<:Complex,R<:Complex} =
    _cuda_vector_batch_synthesis(cfg, S, Tlm; real_output)
synthesis_sphtor_batch(cfg::SHTConfig, S::CUDA.AnyCuArray{T,3},
                       Tlm::CUDA.AnyCuArray{R,3}; kwargs...) where {T<:Complex,R<:Complex} =
    synthesis_sphtor_batch(SHTnsKit.GPU(), cfg, S, Tlm; kwargs...)
synthesis_sphtor_batch_cplx(cfg::SHTConfig, S::CUDA.AnyCuArray{T,3},
                            Tlm::CUDA.AnyCuArray{R,3}) where {T<:Complex,R<:Complex} =
    synthesis_sphtor_batch(SHTnsKit.GPU(), cfg, S, Tlm; real_output=false)
synthesis_sphtor_batch_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                            S::CUDA.AnyCuArray{T,3},
                            Tlm::CUDA.AnyCuArray{R,3}) where {T<:Complex,R<:Complex} =
    synthesis_sphtor_batch(SHTnsKit.GPU(), cfg, S, Tlm; real_output=false)

analysis_qst_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                   Vr::CUDA.AnyCuArray{T,3}, Vt::CUDA.AnyCuArray{R,3},
                   Vp::CUDA.AnyCuArray{U,3}) where {T<:Real,R<:Real,U<:Real} =
    (_cuda_batch_analysis(cfg, Vr), _cuda_vector_batch_analysis(cfg, Vt, Vp)...)
analysis_qst_batch(cfg::SHTConfig, Vr::CUDA.AnyCuArray{T,3},
                   Vt::CUDA.AnyCuArray{R,3}, Vp::CUDA.AnyCuArray{U,3}) where {T<:Real,R<:Real,U<:Real} =
    analysis_qst_batch(SHTnsKit.GPU(), cfg, Vr, Vt, Vp)
synthesis_qst_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                    Q::CUDA.AnyCuArray{T,3}, S::CUDA.AnyCuArray{R,3},
                    Tlm::CUDA.AnyCuArray{U,3};
                    real_output::Bool=true) where {T<:Complex,R<:Complex,U<:Complex} =
    (_cuda_batch_synthesis(cfg, Q; real_output),
     _cuda_vector_batch_synthesis(cfg, S, Tlm; real_output)...)
synthesis_qst_batch(cfg::SHTConfig, Q::CUDA.AnyCuArray{T,3},
                    S::CUDA.AnyCuArray{R,3}, Tlm::CUDA.AnyCuArray{U,3};
                    kwargs...) where {T<:Complex,R<:Complex,U<:Complex} =
    synthesis_qst_batch(SHTnsKit.GPU(), cfg, Q, S, Tlm; kwargs...)
synthesis_qst_batch_cplx(cfg::SHTConfig, Q::CUDA.AnyCuArray{T,3},
                         S::CUDA.AnyCuArray{R,3},
                         Tlm::CUDA.AnyCuArray{U,3}) where {T<:Complex,R<:Complex,U<:Complex} =
    synthesis_qst_batch(SHTnsKit.GPU(), cfg, Q, S, Tlm; real_output=false)
synthesis_qst_batch_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                         Q::CUDA.AnyCuArray{T,3}, S::CUDA.AnyCuArray{R,3},
                         Tlm::CUDA.AnyCuArray{U,3}) where {T<:Complex,R<:Complex,U<:Complex} =
    synthesis_qst_batch(SHTnsKit.GPU(), cfg, Q, S, Tlm; real_output=false)

function analysis_sphtor!(plan::SHTPlan, Sout::CUDA.AnyCuArray,
                           Tout::CUDA.AnyCuArray, Vt::CUDA.AnyCuArray,
                           Vp::CUDA.AnyCuArray)
    return _cuda_vector_analysis_direct!(
        plan, plan.cfg, Sout, Tout, Vt, Vp; use_rfft=plan.use_rfft,
    )
end

function synthesis_sphtor!(plan::SHTPlan, Vt::CUDA.AnyCuArray,
                            Vp::CUDA.AnyCuArray, Slm::CUDA.AnyCuArray,
                            Tlm::CUDA.AnyCuArray; real_output::Bool=true)
    return _cuda_vector_synthesis_direct!(
        plan, plan.cfg, Vt, Vp, Slm, Tlm;
        real_output, use_rfft=plan.use_rfft,
    )
end

function _cuda_vector_diagonal!(output::CUDA.AnyCuArray,
                                cfg::SHTConfig,
                                input::CUDA.AnyCuArray;
                                inverse::Bool)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    size(input) == expected || throw(DimensionMismatch("input must have size $expected"))
    size(output) == expected || throw(DimensionMismatch("output must have size $expected"))
    eltype(output) === eltype(input) || throw(ArgumentError(
        "vector operator input and output element types must match",
    ))
    source = output === input || !Base.mightalias(output, input) ? input : copy(input)
    vector_diagonal_kernel!(CUDABackend())(
        output, source, cfg.lmax, cfg.mmax, cfg.mres, inverse;
        ndrange=expected,
    )
    CUDA.synchronize()
    return output
end

function _cuda_operator_matrix!(cfg::SHTConfig, mx::CUDA.AnyCuArray{T,1},
                                derivative::Bool) where {T<:Real}
    _require_cuda(derivative ? :st_dt_matrix : :mul_ct_matrix, SHTnsKit.GPU())
    length(mx) == 2cfg.nlm || throw(DimensionMismatch(
        "mx length must be 2*nlm=$(2cfg.nlm)",
    ))
    metadata = SHTnsKit._gpu_operator_metadata(cfg, T)
    li = CuArray(metadata.li); mi = CuArray(metadata.mi)
    down = CuArray(metadata.down_ratios); up = CuArray(metadata.up_ratios)
    operator_matrix_kernel!(CUDABackend())(
        mx, li, mi, down, up, cfg.lmax, derivative; ndrange=cfg.nlm,
    )
    CUDA.synchronize()
    return mx
end

mul_ct_matrix(::SHTnsKit.GPU, cfg::SHTConfig,
              mx::CUDA.AnyCuArray{T,1}) where {T<:Real} =
    _cuda_operator_matrix!(cfg, mx, false)
mul_ct_matrix(cfg::SHTConfig, mx::CUDA.AnyCuArray{T,1}) where {T<:Real} =
    mul_ct_matrix(SHTnsKit.GPU(), cfg, mx)
st_dt_matrix(::SHTnsKit.GPU, cfg::SHTConfig,
             mx::CUDA.AnyCuArray{T,1}) where {T<:Real} =
    _cuda_operator_matrix!(cfg, mx, true)
st_dt_matrix(cfg::SHTConfig, mx::CUDA.AnyCuArray{T,1}) where {T<:Real} =
    st_dt_matrix(SHTnsKit.GPU(), cfg, mx)

function SH_mul_mx(::SHTnsKit.GPU, cfg::SHTConfig,
                   mx::CUDA.AnyCuArray{<:Real,1},
                   input::CUDA.AnyCuArray{<:Complex,1},
                   output::CUDA.AnyCuArray{<:Complex,1})
    _require_cuda(:SH_mul_mx, SHTnsKit.GPU())
    length(mx) == 2cfg.nlm || throw(DimensionMismatch(
        "mx length must be 2*nlm=$(2cfg.nlm)",
    ))
    length(input) == cfg.nlm || throw(DimensionMismatch(
        "Qlm length must be nlm=$(cfg.nlm)",
    ))
    length(output) == cfg.nlm || throw(DimensionMismatch(
        "Rlm length must be nlm=$(cfg.nlm)",
    ))
    Base.mightalias(input, output) && throw(ArgumentError(
        "SH_mul_mx input and output must not alias",
    ))
    T = typeof(real(zero(eltype(mx))))
    metadata = SHTnsKit._gpu_operator_metadata(cfg, T)
    lower = CuArray(metadata.lower); upper = CuArray(metadata.upper)
    packed_operator_kernel!(CUDABackend())(
        output, input, mx, lower, upper; ndrange=cfg.nlm,
    )
    CUDA.synchronize()
    return output
end

SH_mul_mx(cfg::SHTConfig, mx::CUDA.AnyCuArray{<:Real,1},
          input::CUDA.AnyCuArray{<:Complex,1},
          output::CUDA.AnyCuArray{<:Complex,1}) =
    SH_mul_mx(SHTnsKit.GPU(), cfg, mx, input, output)

divergence_from_spheroidal(cfg::SHTConfig, input::CUDA.AnyCuArray) =
    divergence_from_spheroidal!(cfg, similar(input), input)
divergence_from_spheroidal(::SHTnsKit.GPU, cfg::SHTConfig,
                           input::CUDA.AnyCuArray) =
    divergence_from_spheroidal(cfg, input)
divergence_from_spheroidal!(cfg::SHTConfig, output::CUDA.AnyCuArray,
                            input::CUDA.AnyCuArray) =
    _cuda_vector_diagonal!(output, cfg, input; inverse=false)
divergence_from_spheroidal!(::SHTnsKit.GPU, cfg::SHTConfig,
                            output::CUDA.AnyCuArray,
                            input::CUDA.AnyCuArray) =
    divergence_from_spheroidal!(cfg, output, input)
spheroidal_from_divergence(cfg::SHTConfig, input::CUDA.AnyCuArray) =
    spheroidal_from_divergence!(cfg, similar(input), input)
spheroidal_from_divergence(::SHTnsKit.GPU, cfg::SHTConfig,
                           input::CUDA.AnyCuArray) =
    spheroidal_from_divergence(cfg, input)
spheroidal_from_divergence!(cfg::SHTConfig, output::CUDA.AnyCuArray,
                            input::CUDA.AnyCuArray) =
    _cuda_vector_diagonal!(output, cfg, input; inverse=true)
spheroidal_from_divergence!(::SHTnsKit.GPU, cfg::SHTConfig,
                            output::CUDA.AnyCuArray,
                            input::CUDA.AnyCuArray) =
    spheroidal_from_divergence!(cfg, output, input)
vorticity_from_toroidal(cfg::SHTConfig, input::CUDA.AnyCuArray) =
    vorticity_from_toroidal!(cfg, similar(input), input)
vorticity_from_toroidal(::SHTnsKit.GPU, cfg::SHTConfig,
                        input::CUDA.AnyCuArray) =
    vorticity_from_toroidal(cfg, input)
vorticity_from_toroidal!(cfg::SHTConfig, output::CUDA.AnyCuArray,
                         input::CUDA.AnyCuArray) =
    _cuda_vector_diagonal!(output, cfg, input; inverse=false)
vorticity_from_toroidal!(::SHTnsKit.GPU, cfg::SHTConfig,
                         output::CUDA.AnyCuArray,
                         input::CUDA.AnyCuArray) =
    vorticity_from_toroidal!(cfg, output, input)
toroidal_from_vorticity(cfg::SHTConfig, input::CUDA.AnyCuArray) =
    toroidal_from_vorticity!(cfg, similar(input), input)
toroidal_from_vorticity(::SHTnsKit.GPU, cfg::SHTConfig,
                        input::CUDA.AnyCuArray) =
    toroidal_from_vorticity(cfg, input)
toroidal_from_vorticity!(cfg::SHTConfig, output::CUDA.AnyCuArray,
                         input::CUDA.AnyCuArray) =
    _cuda_vector_diagonal!(output, cfg, input; inverse=true)
toroidal_from_vorticity!(::SHTnsKit.GPU, cfg::SHTConfig,
                         output::CUDA.AnyCuArray,
                         input::CUDA.AnyCuArray) =
    toroidal_from_vorticity!(cfg, output, input)

@inline function _cuda_lcap(cfg::SHTConfig, ltr::Integer)
    return SHTnsKit._validate_degree_limit(cfg, ltr)
end

function _cuda_pack_lm(cfg::SHTConfig, dense::CUDA.AnyCuArray, lcap::Int)
    packed = CUDA.zeros(eltype(dense), cfg.nlm)
    kernel! = real_pack_kernel!(CUDABackend())
    kernel!(packed, dense, cfg.lmax, cfg.mmax, cfg.mres, lcap;
            ndrange=(cfg.lmax + 1, cfg.mmax ÷ cfg.mres + 1))
    CUDA.synchronize()
    return packed
end

function _cuda_unpack_lm(cfg::SHTConfig, packed::CUDA.AnyCuArray, lcap::Int)
    length(packed) == cfg.nlm || throw(DimensionMismatch(
        "Qlm must have length $(cfg.nlm)",
    ))
    dense = CUDA.zeros(eltype(packed), cfg.lmax + 1, cfg.mmax + 1)
    kernel! = real_unpack_kernel!(CUDABackend())
    kernel!(dense, packed, cfg.lmax, cfg.mmax, cfg.mres, lcap;
            ndrange=size(dense))
    CUDA.synchronize()
    return dense
end

function analysis_packed(::SHTnsKit.GPU, cfg::SHTConfig,
                         field::CUDA.AnyCuArray{T,1}) where {T<:Real}
    _require_cuda(:analysis_packed, SHTnsKit.GPU())
    length(field) == cfg.nspat || throw(DimensionMismatch(
        "field must have length $(cfg.nspat)",
    ))
    return _cuda_pack_lm(
        cfg, _cuda_scalar_analysis(cfg, reshape(field, cfg.nlat, cfg.nlon)), cfg.lmax,
    )
end
analysis_packed(cfg::SHTConfig, field::CUDA.AnyCuArray{T,1}) where {T<:Real} =
    analysis_packed(SHTnsKit.GPU(), cfg, field)

function synthesis_packed(::SHTnsKit.GPU, cfg::SHTConfig,
                          coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex}
    _require_cuda(:synthesis_packed, SHTnsKit.GPU())
    return vec(_cuda_scalar_synthesis(
        cfg, _cuda_unpack_lm(cfg, coefficients, cfg.lmax); real_output=true,
    ))
end
synthesis_packed(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex} =
    synthesis_packed(SHTnsKit.GPU(), cfg, coefficients)

function analysis_packed_l(::SHTnsKit.GPU, cfg::SHTConfig,
                           field::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Real}
    lcap = _cuda_lcap(cfg, ltr)
    length(field) == cfg.nspat || throw(DimensionMismatch(
        "field must have length $(cfg.nspat)",
    ))
    return _cuda_pack_lm(
        cfg, _cuda_scalar_analysis(
            cfg, reshape(field, cfg.nlat, cfg.nlon); lcap,
        ), lcap,
    )
end
analysis_packed_l(cfg::SHTConfig, field::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Real} =
    analysis_packed_l(SHTnsKit.GPU(), cfg, field, ltr)

function synthesis_packed_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            coefficients::CUDA.AnyCuArray{T,1},
                            ltr::Integer) where {T<:Complex}
    lcap = _cuda_lcap(cfg, ltr)
    return vec(_cuda_scalar_synthesis(
        cfg, _cuda_unpack_lm(cfg, coefficients, lcap);
        real_output=true, lcap,
    ))
end
synthesis_packed_l(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    synthesis_packed_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _cuda_mode_analysis(cfg::SHTConfig, physical_m::Int,
                             mode::CUDA.AnyCuArray, lcap::Int, scale)
    RT = typeof(float(real(zero(eltype(mode)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    canonical = CUDA.zeros(CT, lcap - physical_m + 1)
    kernel! = mode_analysis_kernel!(CUDABackend())
    kernel!(canonical, mode, tables.Plm, tables.weights, RT(scale), physical_m, lcap;
            ndrange=length(canonical))
    configured = canonical ./ @view(tables.scales[(physical_m + 1):(lcap + 1), physical_m + 1])
    CUDA.synchronize()
    return configured
end

function _cuda_mode_synthesis(cfg::SHTConfig, physical_m::Int,
                              coefficients::CUDA.AnyCuArray, lcap::Int, scale)
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    canonical = CT.(coefficients) .*
        @view(tables.scales[(physical_m + 1):(lcap + 1), physical_m + 1])
    mode = CUDA.zeros(CT, cfg.nlat)
    kernel! = mode_synthesis_kernel!(CUDABackend())
    kernel!(mode, canonical, tables.Plm, RT(scale), physical_m, lcap;
            ndrange=cfg.nlat)
    CUDA.synchronize()
    return mode
end

function analysis_axisym(::SHTnsKit.GPU, cfg::SHTConfig,
                         field::CUDA.AnyCuArray{T,1}) where {T<:Real}
    length(field) == cfg.nlat || throw(DimensionMismatch(
        "field must have length nlat=$(cfg.nlat)",
    ))
    return _cuda_mode_analysis(cfg, 0, field, cfg.lmax, SHTnsKit._analysis_phi_scale(cfg) * cfg.nlon)
end
analysis_axisym(cfg::SHTConfig, field::CUDA.AnyCuArray{T,1}) where {T<:Real} =
    analysis_axisym(SHTnsKit.GPU(), cfg, field)

function synthesis_axisym(::SHTnsKit.GPU, cfg::SHTConfig,
                          coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex}
    length(coefficients) == cfg.lmax + 1 || throw(DimensionMismatch(
        "coefficients must have length lmax+1=$(cfg.lmax + 1)",
    ))
    return real.(_cuda_mode_synthesis(cfg, 0, coefficients, cfg.lmax, 1))
end
synthesis_axisym(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex} =
    synthesis_axisym(SHTnsKit.GPU(), cfg, coefficients)

function analysis_axisym_l(::SHTnsKit.GPU, cfg::SHTConfig,
                           field::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Real}
    lcap = _cuda_lcap(cfg, ltr)
    length(field) == cfg.nlat || throw(DimensionMismatch(
        "field must have length nlat=$(cfg.nlat)",
    ))
    return _cuda_mode_analysis(cfg, 0, field, lcap, SHTnsKit._analysis_phi_scale(cfg) * cfg.nlon)
end
analysis_axisym_l(cfg::SHTConfig, field::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Real} =
    analysis_axisym_l(SHTnsKit.GPU(), cfg, field, ltr)

function synthesis_axisym_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            coefficients::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Complex}
    lcap = _cuda_lcap(cfg, ltr)
    length(coefficients) >= lcap + 1 || throw(DimensionMismatch(
        "coefficients must contain degrees 0:ltr",
    ))
    return real.(_cuda_mode_synthesis(
        cfg, 0, @view(coefficients[1:(lcap + 1)]), lcap, 1,
    ))
end
synthesis_axisym_l(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    synthesis_axisym_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _cuda_fixed_order(cfg::SHTConfig, im::Int, ltr::Integer)
    im >= 0 || throw(ArgumentError("im must be >= 0"))
    im <= cfg.mmax ÷ cfg.mres || throw(ArgumentError(
        "im must be <= mmax/mres=$(cfg.mmax ÷ cfg.mres)",
    ))
    lcap = _cuda_lcap(cfg, ltr)
    physical_m = im * cfg.mres
    lcap >= physical_m || throw(ArgumentError(
        "ltr must be >= im*mres=$(physical_m)",
    ))
    return physical_m, lcap
end

function analysis_packed_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Int,
                            mode::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Complex}
    physical_m, lcap = _cuda_fixed_order(cfg, im, ltr)
    length(mode) == cfg.nlat || throw(DimensionMismatch(
        "mode must have length nlat=$(cfg.nlat)",
    ))
    return _cuda_mode_analysis(cfg, physical_m, mode, lcap, SHTnsKit._analysis_phi_scale(cfg))
end
analysis_packed_ml(cfg::SHTConfig, im::Int, mode::CUDA.AnyCuArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    analysis_packed_ml(SHTnsKit.GPU(), cfg, im, mode, ltr)

function synthesis_packed_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Int,
                             coefficients::CUDA.AnyCuArray{T,1},
                             ltr::Integer) where {T<:Complex}
    physical_m, lcap = _cuda_fixed_order(cfg, im, ltr)
    length(coefficients) == lcap - physical_m + 1 || throw(DimensionMismatch(
        "coefficients have the wrong fixed-order length",
    ))
    return _cuda_mode_synthesis(
        cfg, physical_m, coefficients, lcap, SHTnsKit.phi_inv_scale(cfg),
    )
end
synthesis_packed_ml(cfg::SHTConfig, im::Int,
                    coefficients::CUDA.AnyCuArray{T,1}, ltr::Integer) where {T<:Complex} =
    synthesis_packed_ml(SHTnsKit.GPU(), cfg, im, coefficients, ltr)

function _cuda_analysis_packed_cplx(cfg::SHTConfig,
                                    field::CUDA.AnyCuArray{T,2},
                                    lcap::Int) where {T<:Complex}
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon))",
    ))
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    fourier = CT.(field)
    gpu_fft!(fourier, 2)
    packed = CUDA.zeros(CT, SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1))
    mcap = min(cfg.mmax, lcap)
    kernel! = complex_packed_analysis_kernel!(CUDABackend())
    kernel!(packed, fourier, tables.Plm, tables.weights, tables.scales,
            RT(SHTnsKit._analysis_phi_scale(cfg)), cfg.nlon, lcap, cfg.mmax, mcap;
            ndrange=(lcap + 1, 2mcap + 1))
    CUDA.synchronize()
    return packed
end
function analysis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                              field::CUDA.AnyCuArray{T,2}) where {T<:Complex}
    return _cuda_analysis_packed_cplx(cfg, field, cfg.lmax)
end
analysis_packed_cplx(cfg::SHTConfig, field::CUDA.AnyCuArray{T,2}) where {T<:Complex} =
    analysis_packed_cplx(SHTnsKit.GPU(), cfg, field)

function analysis_packed_cplx_l(::SHTnsKit.GPU, cfg::SHTConfig,
                                field::CUDA.AnyCuArray{T,2},
                                ltr::Integer) where {T<:Complex}
    return _cuda_analysis_packed_cplx(cfg, field, _cuda_lcap(cfg, ltr))
end
analysis_packed_cplx_l(cfg::SHTConfig, field::CUDA.AnyCuArray{T,2},
                       ltr::Integer) where {T<:Complex} =
    analysis_packed_cplx_l(SHTnsKit.GPU(), cfg, field, ltr)

function _cuda_synthesis_packed_cplx(cfg::SHTConfig,
                                     coefficients::CUDA.AnyCuArray{T,1},
                                     lcap::Int) where {T<:Complex}
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    expected = SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(coefficients) == expected || throw(DimensionMismatch(
        "coefficients must have length $expected",
    ))
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    fourier = CUDA.zeros(CT, cfg.nlat, cfg.nlon)
    mcap = min(cfg.mmax, lcap)
    kernel! = complex_packed_synthesis_kernel!(CUDABackend())
    kernel!(fourier, coefficients, tables.Plm, tables.scales,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, lcap, cfg.mmax, mcap;
            ndrange=(cfg.nlat, 2mcap + 1))
    CUDA.synchronize()
    gpu_ifft!(fourier, 2)
    return fourier
end
function synthesis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                               coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex}
    return _cuda_synthesis_packed_cplx(cfg, coefficients, cfg.lmax)
end
synthesis_packed_cplx(cfg::SHTConfig,
                      coefficients::CUDA.AnyCuArray{T,1}) where {T<:Complex} =
    synthesis_packed_cplx(SHTnsKit.GPU(), cfg, coefficients)

function synthesis_packed_cplx_l(::SHTnsKit.GPU, cfg::SHTConfig,
                                 coefficients::CUDA.AnyCuArray{T,1},
                                 ltr::Integer) where {T<:Complex}
    return _cuda_synthesis_packed_cplx(
        cfg, coefficients, _cuda_lcap(cfg, ltr),
    )
end
synthesis_packed_cplx_l(cfg::SHTConfig,
                        coefficients::CUDA.AnyCuArray{T,1},
                        ltr::Integer) where {T<:Complex} =
    synthesis_packed_cplx_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _cuda_batch_analysis(cfg::SHTConfig, fields::CUDA.AnyCuArray;
                              use_rfft::Bool=false, fft_batch=nothing)
    fft_batch === nothing || throw(ArgumentError(
        "CUDA scalar batches do not accept caller-provided fft_batch scratch",
    ))
    size(fields, 1) == cfg.nlat && size(fields, 2) == cfg.nlon ||
        throw(DimensionMismatch("fields must start with (nlat, nlon)"))
    size(fields, 3) > 0 || throw(ArgumentError(
        "analysis_batch requires at least one field",
    ))
    use_rfft && !(eltype(fields) <: Real) && throw(ArgumentError(
        "use_rfft=true requires real-valued fields",
    ))
    RT = typeof(float(real(zero(eltype(fields)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    fourier = CT.(fields)
    gpu_fft!(fourier, 2)
    canonical = CUDA.zeros(CT, cfg.lmax + 1, cfg.mmax + 1, size(fields, 3))
    kernel! = scalar_batch_analysis_kernel!(CUDABackend())
    kernel!(canonical, fourier, tables.Plm, tables.weights, RT(SHTnsKit._analysis_phi_scale(cfg)),
            cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(canonical))
    configured = canonical ./ reshape(
        tables.scales, cfg.lmax + 1, cfg.mmax + 1, 1,
    )
    CUDA.synchronize()
    return configured
end

function _cuda_batch_analysis_direct!(cfg::SHTConfig,
                                      output::CUDA.AnyCuArray{<:Complex,3},
                                      fields::CUDA.AnyCuArray{<:Real,3};
                                      use_rfft::Bool=false, fft_batch=nothing)
    _require_cuda(:analysis_batch!, SHTnsKit.GPU())
    size(fields, 1) == cfg.nlat && size(fields, 2) == cfg.nlon ||
        throw(DimensionMismatch("fields must start with (nlat, nlon)"))
    nfields = size(fields, 3)
    nfields > 0 || throw(ArgumentError(
        "analysis_batch! requires at least one field",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2",
    ))
    size(output) == (cfg.lmax + 1, cfg.mmax + 1, nfields) ||
        throw(DimensionMismatch("output batch shape mismatch"))
    RT = float(eltype(fields))
    CT = Complex{RT}
    scratch = _cuda_batch_scratch(
        cfg, fft_batch, CT, nfields, use_rfft, output, fields,
    )
    tables = _cuda_scalar_tables(cfg, RT)
    return _with_cuda_workspace(cfg, cfg, RT, nfields, use_rfft) do workspace
        fourier = scratch === nothing ? workspace.fourier : scratch
        if use_rfft
            copyto!(workspace.real_buffer, fields)
            mul!(fourier, workspace.forward, workspace.real_buffer)
        else
            copyto!(fourier, fields)
            mul!(fourier, workspace.forward, fourier)
        end
        backend = CUDABackend()
        scalar_batch_analysis_kernel!(backend)(
            output, fourier, tables.Plm, tables.weights, RT(SHTnsKit._analysis_phi_scale(cfg)),
            cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(output),
        )
        coefficient_batch_conversion_kernel!(backend)(
            output, output, tables.scales, cfg.lmax, cfg.mmax, false;
            ndrange=size(output),
        )
        CUDA.synchronize()
        output
    end
end

function analysis_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                        fields::CUDA.AnyCuArray{T,3}; use_rfft::Bool=false) where {T<:Real}
    return _cuda_batch_analysis(cfg, fields; use_rfft)
end
analysis_batch(cfg::SHTConfig, fields::CUDA.AnyCuArray{T,3}; kwargs...) where {T<:Real} =
    analysis_batch(SHTnsKit.GPU(), cfg, fields; kwargs...)

function analysis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                         output::CUDA.AnyCuArray{T,3},
                         fields::CUDA.AnyCuArray{R,3}; kwargs...) where {T<:Complex,R<:Real}
    return _cuda_batch_analysis_direct!(cfg, output, fields; kwargs...)
end
analysis_batch!(cfg::SHTConfig, output::CUDA.AnyCuArray{T,3},
                fields::CUDA.AnyCuArray{R,3}; kwargs...) where {T<:Complex,R<:Real} =
    analysis_batch!(SHTnsKit.GPU(), cfg, output, fields; kwargs...)

function _cuda_batch_synthesis(cfg::SHTConfig,
                               coefficients::CUDA.AnyCuArray;
                               real_output::Bool=true, use_rfft::Bool=false,
                               fft_batch=nothing)
    fft_batch === nothing || throw(ArgumentError(
        "CUDA scalar batches do not accept caller-provided fft_batch scratch",
    ))
    size(coefficients, 1) == cfg.lmax + 1 &&
        size(coefficients, 2) == cfg.mmax + 1 ||
        throw(DimensionMismatch("coefficient batch has the wrong spectral shape"))
    size(coefficients, 3) > 0 || throw(ArgumentError(
        "synthesis_batch requires at least one coefficient field",
    ))
    use_rfft && !real_output && throw(ArgumentError("use_rfft=true implies real_output"))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _cuda_scalar_tables(cfg, RT)
    canonical = CT.(coefficients) .* reshape(
        tables.scales, cfg.lmax + 1, cfg.mmax + 1, 1,
    )
    fourier = CUDA.zeros(CT, cfg.nlat, cfg.nlon, size(coefficients, 3))
    kernel! = scalar_batch_synthesis_kernel!(CUDABackend())
    kernel!(fourier, canonical, tables.Plm, RT(SHTnsKit.phi_inv_scale(cfg)),
            cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres, real_output;
            ndrange=(cfg.nlat, cfg.mmax + 1, size(coefficients, 3)))
    CUDA.synchronize()
    gpu_ifft!(fourier, 2)
    return real_output ? real.(fourier) : fourier
end

function _cuda_batch_synthesis_direct!(cfg::SHTConfig,
                                       output::CUDA.AnyCuArray{<:Number,3},
                                       coefficients::CUDA.AnyCuArray{<:Complex,3};
                                       real_output::Bool=true,
                                       use_rfft::Bool=false, fft_batch=nothing)
    _require_cuda(:synthesis_batch!, SHTnsKit.GPU())
    size(coefficients, 1) == cfg.lmax + 1 &&
        size(coefficients, 2) == cfg.mmax + 1 ||
        throw(DimensionMismatch("coefficient batch has the wrong spectral shape"))
    nfields = size(coefficients, 3)
    nfields > 0 || throw(ArgumentError(
        "synthesis_batch! requires at least one coefficient field",
    ))
    size(output) == (cfg.nlat, cfg.nlon, nfields) ||
        throw(DimensionMismatch("output batch shape mismatch"))
    !real_output && eltype(output) <: Real && throw(ArgumentError(
        "real batch output storage requires real_output=true",
    ))
    use_rfft && (!real_output || !(eltype(output) <: Real)) && throw(ArgumentError(
        "use_rfft=true requires real-valued output",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2",
    ))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    scratch = _cuda_batch_scratch(
        cfg, fft_batch, CT, nfields, use_rfft, output, coefficients,
    )
    tables = _cuda_scalar_tables(cfg, RT)
    return _with_cuda_workspace(cfg, cfg, RT, nfields, use_rfft) do workspace
        fourier = scratch === nothing ? workspace.fourier : scratch
        backend = CUDABackend()
        coefficient_batch_conversion_kernel!(backend)(
            workspace.canonical, coefficients, tables.scales,
            cfg.lmax, cfg.mmax, true; ndrange=size(coefficients),
        )
        fill!(fourier, zero(eltype(fourier)))
        scalar_batch_synthesis_kernel!(backend)(
            fourier, workspace.canonical, tables.Plm,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon,
            cfg.lmax, cfg.mmax, cfg.mres, real_output && !use_rfft;
            ndrange=(cfg.nlat, cfg.mmax + 1, nfields),
        )
        CUDA.synchronize()
        if use_rfft
            mul!(workspace.real_buffer, workspace.inverse, fourier)
            copyto!(output, workspace.real_buffer)
        else
            mul!(fourier, workspace.inverse, fourier)
            real_output ? (output .= real.(fourier)) : copyto!(output, fourier)
        end
        CUDA.synchronize()
        output
    end
end

function synthesis_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                         coefficients::CUDA.AnyCuArray{T,3};
                         real_output::Bool=true, use_rfft::Bool=false) where {T<:Complex}
    return _cuda_batch_synthesis(cfg, coefficients; real_output, use_rfft)
end
synthesis_batch(cfg::SHTConfig, coefficients::CUDA.AnyCuArray{T,3}; kwargs...) where {T<:Complex} =
    synthesis_batch(SHTnsKit.GPU(), cfg, coefficients; kwargs...)
synthesis_batch_cplx(cfg::SHTConfig,
                     coefficients::CUDA.AnyCuArray{T,3}) where {T<:Complex} =
    _cuda_batch_synthesis(cfg, coefficients; real_output=false)
synthesis_batch_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     coefficients::CUDA.AnyCuArray{T,3}) where {T<:Complex} =
    _cuda_batch_synthesis(cfg, coefficients; real_output=false)

function synthesis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                          output::CUDA.AnyCuArray{T,3},
                          coefficients::CUDA.AnyCuArray{R,3}; kwargs...) where {T,R<:Complex}
    return _cuda_batch_synthesis_direct!(cfg, output, coefficients; kwargs...)
end
function synthesis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                          output::CUDA.AnyCuArray{T,3},
                          coefficients::CUDA.AnyCuArray{R,3}; kwargs...) where {T<:Real,R<:Complex}
    return _cuda_batch_synthesis_direct!(cfg, output, coefficients; kwargs...)
end
synthesis_batch!(cfg::SHTConfig, output::CUDA.AnyCuArray{T,3},
                 coefficients::CUDA.AnyCuArray{R,3}; kwargs...) where {T,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)

function analysis!(::SHTnsKit.GPU, plan::SHTPlan,
                   output::CUDA.AnyCuArray{T,2},
                   field::CUDA.AnyCuArray{R,2}) where {T<:Complex,R<:Number}
    return _cuda_scalar_analysis_direct!(
        plan, plan.cfg, output, field; use_rfft=plan.use_rfft,
    )
end
analysis!(plan::SHTPlan, output::CUDA.AnyCuArray{T,2},
          field::CUDA.AnyCuArray{R,2}) where {T<:Complex,R<:Number} =
    analysis!(SHTnsKit.GPU(), plan, output, field)

function synthesis!(::SHTnsKit.GPU, plan::SHTPlan,
                    output::CUDA.AnyCuArray{T,2},
                    coefficients::CUDA.AnyCuArray{R,2};
                    real_output::Bool=true) where {T<:Number,R<:Complex}
    return _cuda_scalar_synthesis_direct!(
        plan, plan.cfg, output, coefficients;
        real_output, use_rfft=plan.use_rfft,
    )
end
synthesis!(plan::SHTPlan, output::CUDA.AnyCuArray{T,2},
           coefficients::CUDA.AnyCuArray{R,2};
           real_output::Bool=true) where {T<:Number,R<:Complex} =
    synthesis!(SHTnsKit.GPU(), plan, output, coefficients; real_output)
synthesis_batch!(cfg::SHTConfig, output::CUDA.AnyCuArray{T,3},
                 coefficients::CUDA.AnyCuArray{R,3}; kwargs...) where {T<:Real,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)

"""
    _to_gpu_impl(arr::AbstractArray)

Transfer array to CUDA GPU. Overrides the stub in device_utils.jl.
"""
function _to_gpu_impl(arr::AbstractArray)
    if !CUDA.functional()
        error("CUDA is not functional")
    end
    return CuArray(arr)
end

# Avoid double conversion
_to_gpu_impl(arr::CuArray) = arr

"""
    on_device(arr::CuArray) -> ComputeDevice

Returns `GPU()` for CUDA arrays.
"""
on_device(::CUDA.AnyCuArray) = SHTnsKit.GPU()

@inline _is_cpu_device(::SHTnsKit.CPU) = true
@inline _is_cpu_device(::SHTnsKit.GPU) = false

"""
    get_available_gpus()

Returns a list of available CUDA GPU devices with their IDs and names.
"""
function get_available_gpus()
    gpus = []
    if CUDA.functional()
        for i = 0:(CUDA.ndevices()-1)
            push!(gpus, (device=SHTnsKit.GPU(), id=i, name=CUDA.name(CUDA.CuDevice(i))))
        end
    end
    return gpus
end

"""
    set_gpu_device(device_id::Int)

Set the active CUDA GPU device by ID.
"""
function set_gpu_device(device_id::Int)
    if CUDA.functional()
        CUDA.device!(device_id)
        return true
    end
    return false
end

set_gpu_device(::SHTnsKit.GPU, device_id::Int) = set_gpu_device(device_id)

# ============================================================================
# cuFFT-based FFT operations with pre-planned transforms
# ============================================================================

"""
    CuFFTPlan

Pre-planned cuFFT operations for efficient repeated transforms.
"""
struct CuFFTPlan{ForwardPlan,InversePlan,Buffer}
    forward_plan::ForwardPlan
    inverse_plan::InversePlan
    buffer::Buffer
    nlat::Int
    nlon::Int
end

"""
    create_cufft_plan(nlat::Int, nlon::Int)

Create pre-planned cuFFT operations for a grid of size (nlat, nlon).
Forward and inverse plans are created for transforms along the longitude dimension.
"""
function create_cufft_plan(nlat::Int, nlon::Int)
    # Allocate buffer for FFT operations
    buffer = CUDA.zeros(ComplexF64, nlat, nlon)

    # Create forward FFT plan along dimension 2 (longitude)
    forward_plan = CUFFT.plan_fft!(buffer, 2)

    # Create inverse FFT plan along dimension 2 (longitude)
    inverse_plan = CUFFT.plan_ifft!(buffer, 2)

    return CuFFTPlan(forward_plan, inverse_plan, buffer, nlat, nlon)
end

"""
    gpu_fft!(plan::CuFFTPlan, data::CuArray)

Perform in-place forward FFT using pre-planned cuFFT.
"""
function gpu_fft!(plan::CuFFTPlan, data::CuArray)
    mul!(data, plan.forward_plan, data)
    return data
end

"""
    gpu_ifft!(plan::CuFFTPlan, data::CuArray)

Perform in-place inverse FFT using pre-planned cuFFT.
"""
function gpu_ifft!(plan::CuFFTPlan, data::CuArray)
    mul!(data, plan.inverse_plan, data)
    return data
end

"""
    gpu_fft!(data::CuArray, dims)

Perform FFT on CUDA array along specified dimensions (without pre-planning).
"""
function gpu_fft!(data::CuArray, dims)
    plan = CUFFT.plan_fft!(data, dims)
    plan * data  # In-place plan: mul! is called internally, modifying data
    return data
end

"""
    gpu_ifft!(data::CuArray, dims)

Perform inverse FFT on CUDA array along specified dimensions (without pre-planning).
"""
function gpu_ifft!(data::CuArray, dims)
    plan = CUFFT.plan_ifft!(data, dims)
    plan * data
    return data
end

"""
    gpu_rfft(data::CuArray, dims)

Perform real-to-complex FFT on CUDA array.
"""
function gpu_rfft(data::CuArray{<:Real}, dims)
    return CUFFT.rfft(data, dims)
end

"""
    gpu_irfft(data::CuArray, n, dims)

Perform complex-to-real inverse FFT on CUDA array.
"""
function gpu_irfft(data::CuArray, n, dims)
    return CUFFT.irfft(data, n, dims)
end

# ============================================================================
# GPU-accelerated core operations using KernelAbstractions
# ============================================================================

@kernel function legendre_associated_kernel!(Plm, x, lmax, mmax)
    """
    GPU kernel for computing ORTHONORMAL normalized associated Legendre functions P̄_l^m(x).
    Parallelized over (latitude, m) pairs for maximum GPU utilization.

    Computes P̄_l^m = Nlm·P_l^m directly via a bounded recurrence (|P̄| ≲ 1 at all l,m),
    fixing overflow for lmax ≥ ~151 that afflicted the old unnormalized approach.
    The downstream analysis/synthesis kernels must NOT multiply by Nlm (it is folded in).

    Recurrence (Condon–Shortley phase included, orthonormal convention):
      P̄_0^0 = 1/sqrt(4π)   (INV_SQRT_4PI)
      P̄_m^m = −sqrt((2m+1)/(2m)) · sinθ · P̄_{m−1}^{m−1}   (sectoral step)
      P̄_{m+1}^m = sqrt(2m+3) · x · P̄_m^m
      P̄_l^m = a·x·P̄_{l−1}^m − b·P̄_{l−2}^m   (l ≥ m+2)
        a = sqrt(((2l−1)(2l+1)) / ((l−m)(l+m)))
        b = sqrt(((2l+1)(l−1−m)(l−1+m)) / ((2l−3)(l−m)(l+m)))
    """
    i, m_idx = @index(Global, NTuple)
    nlat = length(x)
    if i <= nlat && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        sint = sqrt(max(0.0, 1.0 - xi*xi))  # sin(θ)

        # ── Sectoral term P̄_m^m ────────────────────────────────────────────────
        # Start from P̄_0^0 = 1/sqrt(4π) and apply the sectoral recurrence m times.
        # Each step: P̄_k^k = −sqrt((2k+1)/(2k)) · sinθ · P̄_{k−1}^{k−1}
        const_inv_sqrt4pi = 0.28209479177387814  # 1/sqrt(4π)
        pmm = const_inv_sqrt4pi
        @inbounds for k = 1:m
            pmm = -sqrt((2.0*k + 1.0) / (2.0*k)) * sint * pmm
        end
        Plm[i, m+1, m_idx] = pmm

        # ── P̄_{m+1}^m ──────────────────────────────────────────────────────────
        if m < lmax
            pm1m = sqrt(2.0*m + 3.0) * xi * pmm
            Plm[i, m+2, m_idx] = pm1m

            # ── Three-term recurrence for l ≥ m+2 ──────────────────────────────
            plm_prev2 = pmm    # P̄_{l−2}^m
            plm_prev1 = pm1m   # P̄_{l−1}^m
            @inbounds for l = m+2:lmax
                fl  = Float64(l)
                fm  = Float64(m)
                a = sqrt(((2.0*fl - 1.0) * (2.0*fl + 1.0)) /
                         ((fl - fm) * (fl + fm)))
                b = sqrt(((2.0*fl + 1.0) * (fl - 1.0 - fm) * (fl - 1.0 + fm)) /
                         ((2.0*fl - 3.0) * (fl - fm) * (fl + fm)))
                plm = a * xi * plm_prev1 - b * plm_prev2
                Plm[i, l+1, m_idx] = plm
                plm_prev2 = plm_prev1
                plm_prev1 = plm
            end
        end
    end
end

@kernel function legendre_and_derivative_kernel!(Plm, dPlm, x, lmax, mmax)
    """
    GPU kernel for computing ORTHONORMAL normalized P̄_l^m(x) AND their
    derivatives dP̄_l^m/dx. Required for vector spherical harmonic transforms.

    Computes P̄_l^m via the same bounded normalized sectoral recurrence as
    legendre_associated_kernel! (no overflow at high lmax).

    For dP̄_l^m/dx the standard formula adapted to the normalized functions is:
      (x²−1) · dP̄_l^m/dx = l·x·P̄_l^m − sqrt((l²−m²)(2l+1)/(2l−1)) · P̄_{l−1}^m
    At poles (x²−1 ≈ 0) the derivative is set to zero (pole handling is done by
    the pole closed-form in the downstream synthesis kernel via Nlm separately).

    The downstream kernels must NOT multiply by Nlm (it is folded into P̄ and dP̄).
    """
    i, m_idx = @index(Global, NTuple)
    nlat = length(x)
    if i <= nlat && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        sint = sqrt(max(0.0, 1.0 - xi*xi))
        x2m1 = xi*xi - 1.0
        # Guard against x = ±1 (poles)
        inv_x2m1 = abs(x2m1) < 1e-14 ? 0.0 : 1.0 / x2m1

        # ── Sectoral P̄_m^m (same as legendre_associated_kernel!) ────────────
        const_inv_sqrt4pi = 0.28209479177387814
        pmm = const_inv_sqrt4pi
        @inbounds for k = 1:m
            pmm = -sqrt((2.0*k + 1.0) / (2.0*k)) * sint * pmm
        end
        Plm[i, m+1, m_idx] = pmm

        # dP̄_m^m/dx = m·x·P̄_m^m / (x²−1)   [0 for m=0]
        if m == 0
            dPlm[i, 1, 1] = 0.0
        else
            dPlm[i, m+1, m_idx] = Float64(m) * xi * pmm * inv_x2m1
        end

        if m < lmax
            # ── P̄_{m+1}^m ───────────────────────────────────────────────────
            pm1m = sqrt(2.0*m + 3.0) * xi * pmm
            Plm[i, m+2, m_idx] = pm1m

            # dP̄_{m+1}^m/dx: l=m+1 case.
            # Numerator: (m+1)·x·P̄_{m+1}^m − sqrt(((m+1)²−m²)(2(m+1)+1)/(2(m+1)−1)) · P̄_m^m
            # = (m+1)·x·P̄_{m+1}^m − sqrt((2m+1)(2m+3)/(2m+1)) · P̄_m^m
            # = (m+1)·x·P̄_{m+1}^m − sqrt(2m+3) · P̄_m^m
            dPlm[i, m+2, m_idx] = (Float64(m+1) * xi * pm1m -
                                    sqrt(2.0*m + 3.0) * pmm) * inv_x2m1

            # ── Three-term recurrence for l ≥ m+2 ──────────────────────────
            plm_prev2 = pmm    # P̄_{l−2}^m = P̄_m^m
            plm_prev1 = pm1m   # P̄_{l−1}^m = P̄_{m+1}^m
            @inbounds for l = m+2:lmax
                fl = Float64(l)
                fm = Float64(m)
                a = sqrt(((2.0*fl - 1.0) * (2.0*fl + 1.0)) /
                         ((fl - fm) * (fl + fm)))
                b = sqrt(((2.0*fl + 1.0) * (fl - 1.0 - fm) * (fl - 1.0 + fm)) /
                         ((2.0*fl - 3.0) * (fl - fm) * (fl + fm)))
                plm = a * xi * plm_prev1 - b * plm_prev2
                Plm[i, l+1, m_idx] = plm
                # dP̄_l^m/dx = [l·x·P̄_l^m − sqrt((l²−m²)(2l+1)/(2l−1))·P̄_{l−1}^m] / (x²−1)
                scale_lm = sqrt((fl*fl - fm*fm) * (2.0*fl + 1.0) / (2.0*fl - 1.0))
                dPlm[i, l+1, m_idx] = (fl * xi * plm - scale_lm * plm_prev1) * inv_x2m1
                plm_prev2 = plm_prev1
                plm_prev1 = plm
            end
        end
    end
end

"""
    gpu_analysis(cfg::SHTConfig, spatial_data; device=GPU())

GPU-accelerated spherical harmonic analysis transform using cuFFT.

Implements: a_lm = ∫∫ f(θ,φ) Y_l^m*(θ,φ) sin(θ) dθ dφ
1. FFT along φ (dimension 2) to extract Fourier modes
2. Gauss-Legendre integration along θ (dimension 1) with P_l^m weights

Fully parallelized: all (l,m) coefficients computed in a single kernel launch.

"""
function gpu_analysis(cfg::SHTConfig, spatial_data; device=SHTnsKit.GPU(), kwargs...)
    _require_cuda(:gpu_analysis, device)
    device_data = spatial_data isa CUDA.AnyCuArray ? spatial_data : CuArray(spatial_data)
    return Array(_cuda_scalar_analysis(cfg, device_data; kwargs...))
end

"""
    gpu_synthesis(cfg::SHTConfig, coeffs; device=GPU(), real_output=true)

GPU-accelerated spherical harmonic synthesis transform using cuFFT.

Implements: f(θ,φ) = Σ_l Σ_m a_lm Y_l^m(θ,φ)
1. Legendre summation along θ: F_m(θ) = Σ_l a_lm * P_l^m(cos θ) * N_lm
2. Inverse FFT along φ (dimension 2) to reconstruct spatial field

Fully parallelized: all (θ,m) Fourier modes computed in a single kernel launch.
"""
function gpu_synthesis(cfg::SHTConfig, coeffs; device=SHTnsKit.GPU(),
                       real_output=true, kwargs...)
    _require_cuda(:gpu_synthesis, device)
    device_coefficients = coeffs isa CUDA.AnyCuArray ? coeffs : CuArray(coeffs)
    return Array(_cuda_scalar_synthesis(
        cfg, device_coefficients; real_output, kwargs...,
    ))
end

# ============================================================================
# Laplacian operator
# ============================================================================

"""
    gpu_apply_laplacian!(cfg::SHTConfig, coeffs; device=GPU())

GPU-accelerated Laplacian operator in spectral space.
"""
function gpu_apply_laplacian!(cfg::SHTConfig, coeffs::CUDA.AnyCuArray;
                              device=SHTnsKit.GPU())
    _require_cuda(:gpu_apply_laplacian!, device)

    # Validate input dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(coeffs, 1) == lmax + 1 || throw(DimensionMismatch("coeffs must have $(lmax+1) rows (lmax+1), got $(size(coeffs, 1))"))
    size(coeffs, 2) == mmax + 1 || throw(DimensionMismatch("coeffs must have $(mmax+1) columns (mmax+1), got $(size(coeffs, 2))"))

    laplacian_kernel!(CUDABackend())(
        coeffs, coeffs, lmax, mmax, cfg.mres; ndrange=(lmax+1, mmax+1),
    )
    CUDA.synchronize()
    return coeffs
end

# Historical host-buffer compatibility stays explicit and isolated.  Typed and
# inferred device dispatch never enter this bridge.
function gpu_apply_laplacian!(cfg::SHTConfig, coeffs::AbstractMatrix;
                              device=SHTnsKit.GPU())
    _require_cuda(:gpu_apply_laplacian!, device)
    device_coeffs = CuArray(coeffs)
    gpu_apply_laplacian!(cfg, device_coeffs; device)
    copyto!(coeffs, device_coeffs)
    return coeffs
end

# ============================================================================
# Memory utilities
# ============================================================================

"""
    gpu_memory_info()

Get memory information for the active CUDA device.
Returns a named tuple with `free` and `total` fields (in bytes).
"""
function gpu_memory_info()
    _require_cuda(:gpu_memory_info, SHTnsKit.GPU())
    mem = CUDA.MemoryInfo()
    return (free=mem.free_bytes, total=mem.total_bytes)
end

"""
    check_gpu_memory(required_bytes::Int)

Check if sufficient GPU memory is available.
"""
function check_gpu_memory(required_bytes::Int)
    CUDA.functional() || return false
    mem_info = gpu_memory_info()
    if mem_info.free < required_bytes
        @warn "Insufficient memory: need $(required_bytes÷(1024^3)) GB, have $(mem_info.free÷(1024^3)) GB available"
        return false
    end
    return true
end

"""
    gpu_clear_cache!()

Clear the active CUDA device's memory cache.
"""
function gpu_clear_cache!()
    _require_cuda(:gpu_clear_cache!, SHTnsKit.GPU())
    # Preserve the historical API's process-wide scalar-cache clear while
    # `CUDA.reclaim()` releases the active device's allocator cache.
    _cuda_clear_scalar_cache!()
    try
        CUDA.reclaim()
        @info "CUDA memory cache cleared"
    catch e
        @warn "Failed to clear CUDA cache: $e"
    end
    return nothing
end

"""
    estimate_memory_usage(cfg::SHTConfig, operation::Symbol)

Estimate memory usage for GPU operations.
"""
function estimate_memory_usage(cfg::SHTConfig, operation::Symbol)
    spatial_size = cfg.nlat * cfg.nlon * 16  # ComplexF64 = 16 bytes
    coeff_size = (cfg.lmax + 1) * (cfg.mmax + 1) * 16
    legendre_size = cfg.nlat * (cfg.lmax + 1) * (cfg.mmax + 1) * 8

    if operation == :analysis
        return spatial_size + coeff_size + legendre_size + spatial_size
    elseif operation == :synthesis
        return coeff_size + spatial_size + legendre_size + spatial_size
    elseif operation == :vector
        # Plm and dPlm are real (2×), while S/T contribution tensors are
        # ComplexF64 (4× the real tensor's bytes): 6× in total.
        return 2 * spatial_size + 2 * coeff_size + 6 * legendre_size + 2 * spatial_size
    else
        return spatial_size + coeff_size
    end
end

function _cpu_analysis_fallback(cfg::SHTConfig, spatial_data)
    cpu_data = SHTnsKit.to_device(SHTnsKit.CPU(), spatial_data)
    return SHTnsKit.analysis(SHTnsKit.CPU(), cfg, cpu_data)
end

function _cpu_synthesis_fallback(cfg::SHTConfig, coefficients; real_output::Bool)
    cpu_coefficients = SHTnsKit.to_device(SHTnsKit.CPU(), coefficients)
    return SHTnsKit.synthesis(
        SHTnsKit.CPU(),
        cfg,
        cpu_coefficients;
        real_output=real_output,
    )
end

function _with_cuda_oom_fallback(gpu_call, cpu_call)
    try
        return gpu_call()
    catch err
        err isa CUDA.OutOfGPUMemoryError || rethrow()
        @warn "GPU out of memory, falling back to CPU" exception=(err, catch_backtrace())
        return cpu_call()
    end
end

"""
    gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device())

Memory-safe GPU analysis with automatic fallback to CPU.
"""
function gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device())
    if _is_cpu_device(device) || !CUDA.functional()
        return _cpu_analysis_fallback(cfg, spatial_data)
    end

    required_memory = estimate_memory_usage(cfg, :analysis)
    if !check_gpu_memory(required_memory)
        @info "Falling back to CPU due to memory constraints"
        return _cpu_analysis_fallback(cfg, spatial_data)
    end

    return _with_cuda_oom_fallback(
        () -> gpu_analysis(cfg, spatial_data; device=device),
        () -> _cpu_analysis_fallback(cfg, spatial_data),
    )
end

"""
    gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)

Memory-safe GPU synthesis with automatic fallback to CPU.
"""
function gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
    if _is_cpu_device(device) || !CUDA.functional()
        return _cpu_synthesis_fallback(cfg, coeffs; real_output=real_output)
    end

    required_memory = estimate_memory_usage(cfg, :synthesis)
    if !check_gpu_memory(required_memory)
        @info "Falling back to CPU due to memory constraints"
        return _cpu_synthesis_fallback(cfg, coeffs; real_output=real_output)
    end

    return _with_cuda_oom_fallback(
        () -> gpu_synthesis(cfg, coeffs; device=device, real_output=real_output),
        () -> _cpu_synthesis_fallback(cfg, coeffs; real_output=real_output),
    )
end

# Types defined by this extension.
export CuFFTPlan, create_cufft_plan, gpu_fft!, gpu_ifft!

end # module SHTnsKitGPUExt
