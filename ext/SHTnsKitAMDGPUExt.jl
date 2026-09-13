module SHTnsKitAMDGPUExt

using SHTnsKit
using AMDGPU
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using FFTW
using LinearAlgebra: mul!

include("GPUCommon.jl")
using .GPUCommon: laplacian_kernel!, operator_matrix_kernel!,
                  packed_operator_kernel!, legendre_table_kernel!,
                  scalar_analysis_kernel!,
                  scalar_synthesis_kernel!, coefficient_conversion_kernel!,
                  coefficient_batch_conversion_kernel!,
                  real_pack_kernel!, real_unpack_kernel!,
                  mode_analysis_kernel!, mode_synthesis_kernel!,
                  scalar_batch_analysis_kernel!, scalar_batch_synthesis_kernel!,
                  complex_packed_analysis_kernel!, complex_packed_synthesis_kernel!,
                  scalar_config_signature, vector_config_signature,
                  scalar_host_tables,
                  ScalarTableCache, scalar_cache_lookup,
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

import SHTnsKit: gpu_apply_laplacian!

mutable struct AMDGPUAdapter end
const AMDGPU_ADAPTER = AMDGPUAdapter()

function __init__()
    _register_gpu_adapter!(:amdgpu, AMDGPU_ADAPTER)
    _enable_gpu_loops!(launch_sht_loop!)
    return nothing
end

_gpu_adapter_functional(::AMDGPUAdapter) = AMDGPU.functional()
_gpu_adapter_matches(::AMDGPUAdapter, ::AMDGPU.AnyROCArray) = true

function _gpu_adapter_adapt(::AMDGPUAdapter, value)
    AMDGPU.functional() || throw(SHTnsKit.BackendUnavailableError(
        :to_device,
        "AMDGPU.jl is loaded but AMDGPU.functional() is false",
    ))
    return ROCArray(value)
end

function _require_amdgpu(operation::Symbol)
    AMDGPU.functional() || throw(SHTnsKit.BackendUnavailableError(
        operation,
        "AMDGPU.jl is loaded but AMDGPU.functional() is false",
    ))
    AMDGPU.functional(:rocfft) || throw(SHTnsKit.BackendUnavailableError(
        operation,
        "AMDGPU is functional but rocFFT is unavailable",
    ))
    return nothing
end

struct AMDGPUScalarTables{TX,TW,TP,TS}
    x::TX
    weights::TW
    Plm::TP
    scales::TS
end

struct AMDGPUVectorTables{TX,TW,TS,TD,TO}
    x::TX
    weights::TW
    scales::TS
    dtheta::TD
    over_sin::TO
end

const _AMDGPU_SCALAR_CACHE = ScalarTableCache(8)
const _AMDGPU_VECTOR_CACHE = ScalarTableCache(8)
const _AMDGPU_LOCAL_CACHE = ScalarTableCache(8)
const _AMDGPU_WORKSPACE_CACHE = ScalarWorkspaceCache(8)
const _rotation_cache = RotationBlockCache(8)

struct AMDGPURotationBlocks{TO,TV,TS,T}
    offsets::TO
    values::TV
    input_scales::TS
    output_scales::TS
    alpha::T
    gamma::T
end

function _require_amdgpu_rotation(operation::Symbol)
    AMDGPU.functional() || throw(SHTnsKit.BackendUnavailableError(
        operation, "AMDGPU.jl is loaded but AMDGPU.functional() is false",
    ))
    return nothing
end

function _amdgpu_rotation_blocks(r::SHTRotation,
                                 ::Type{T}) where {T<:AbstractFloat}
    device = AMDGPU.device_id()
    key = (device, T, r.lmax, r.mmax, r.α, r.β, r.γ, r.conv,
           r.norm, r.cs_phase, r.real_norm)
    cached = rotation_cache_lookup(_rotation_cache, key)
    cached === nothing || return cached::AMDGPURotationBlocks
    host = SHTnsKit._rotation_host_blocks(r, T)
    built = AMDGPURotationBlocks(
        ROCArray(host.offsets), ROCArray(host.values),
        ROCArray(host.input_scales), ROCArray(host.output_scales),
        host.alpha, host.gamma,
    )
    return rotation_cache_publish!(AMDGPU.synchronize, _rotation_cache, key, built)
end

function _amdgpu_validate_rotation(operation::Symbol,
                                   input::AMDGPU.AnyROCArray{<:Complex,1},
                                   output::AMDGPU.AnyROCArray{<:Complex,1},
                                   expected::Int)
    _require_amdgpu_rotation(operation)
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
                    input::AMDGPU.AnyROCArray{<:Complex,1}, angle::Real,
                    output::AMDGPU.AnyROCArray{<:Complex,1})
    RT = _amdgpu_validate_rotation(:SH_Zrotate, input, output, cfg.nlm)
    isfinite(angle) || throw(ArgumentError("rotation angle must be finite"))
    source = Base.mightalias(input, output) ? copy(input) : input
    orders = ROCArray(Int32.(cfg.mi))
    rotation_z_real_kernel!(ROCBackend())(
        output, source, RT(angle), orders; ndrange=cfg.nlm,
    )
    AMDGPU.synchronize()
    return output
end

SH_Zrotate(cfg::SHTConfig, input::AMDGPU.AnyROCArray{<:Complex,1}, angle::Real,
           output::AMDGPU.AnyROCArray{<:Complex,1}) =
    SH_Zrotate(SHTnsKit.GPU(), cfg, input, angle, output)

function shtns_rotation_apply_real(::SHTnsKit.GPU, r::SHTRotation,
                                   input::AMDGPU.AnyROCArray{<:Complex,1},
                                   output::AMDGPU.AnyROCArray{<:Complex,1})
    expected = SHTnsKit.nlm_calc(r.lmax, r.mmax, 1)
    RT = _amdgpu_validate_rotation(
        :shtns_rotation_apply_real, input, output, expected,
    )
    source = Base.mightalias(input, output) ? copy(input) : input
    blocks = _amdgpu_rotation_blocks(r, RT)
    rotation_real_kernel!(ROCBackend())(
        output, source, blocks.offsets, blocks.values,
        blocks.input_scales, blocks.output_scales,
        blocks.alpha, blocks.gamma, r.lmax, r.mmax; ndrange=expected,
    )
    AMDGPU.synchronize()
    return output
end

shtns_rotation_apply_real(r::SHTRotation,
                          input::AMDGPU.AnyROCArray{<:Complex,1},
                          output::AMDGPU.AnyROCArray{<:Complex,1}) =
    shtns_rotation_apply_real(SHTnsKit.GPU(), r, input, output)

function shtns_rotation_apply_cplx(::SHTnsKit.GPU, r::SHTRotation,
                                   input::AMDGPU.AnyROCArray{<:Complex,1},
                                   output::AMDGPU.AnyROCArray{<:Complex,1})
    expected = SHTnsKit.nlm_cplx_calc(r.lmax, r.mmax, 1)
    RT = _amdgpu_validate_rotation(
        :shtns_rotation_apply_cplx, input, output, expected,
    )
    source = Base.mightalias(input, output) ? copy(input) : input
    blocks = _amdgpu_rotation_blocks(r, RT)
    rotation_cplx_kernel!(ROCBackend())(
        output, source, blocks.offsets, blocks.values,
        blocks.input_scales, blocks.output_scales,
        blocks.alpha, blocks.gamma, r.lmax, r.mmax; ndrange=expected,
    )
    AMDGPU.synchronize()
    return output
end

shtns_rotation_apply_cplx(r::SHTRotation,
                          input::AMDGPU.AnyROCArray{<:Complex,1},
                          output::AMDGPU.AnyROCArray{<:Complex,1}) =
    shtns_rotation_apply_cplx(SHTnsKit.GPU(), r, input, output)

function SH_Yrotate(::SHTnsKit.GPU, cfg::SHTConfig,
                    input::AMDGPU.AnyROCArray{<:Complex,1}, angle::Real,
                    output::AMDGPU.AnyROCArray{<:Complex,1})
    cfg.mres == 1 || throw(ArgumentError(
        "SH_Yrotate requires mres==1 (got mres=$(cfg.mres)); a Y-rotation mixes orders and cannot be represented in an mres-strided layout",
    ))
    r = SHTRotation(
        cfg.lmax, cfg.mmax; β=angle, norm=cfg.norm,
        cs_phase=cfg.cs_phase, real_norm=cfg.real_norm,
    )
    return shtns_rotation_apply_real(SHTnsKit.GPU(), r, input, output)
end

SH_Yrotate(cfg::SHTConfig, input::AMDGPU.AnyROCArray{<:Complex,1}, angle::Real,
           output::AMDGPU.AnyROCArray{<:Complex,1}) =
    SH_Yrotate(SHTnsKit.GPU(), cfg, input, angle, output)
SH_Yrotate90(::SHTnsKit.GPU, cfg::SHTConfig,
             input::AMDGPU.AnyROCArray{<:Complex,1},
             output::AMDGPU.AnyROCArray{<:Complex,1}) =
    SH_Yrotate(SHTnsKit.GPU(), cfg, input, pi / 2, output)
SH_Yrotate90(cfg::SHTConfig, input::AMDGPU.AnyROCArray{<:Complex,1},
             output::AMDGPU.AnyROCArray{<:Complex,1}) =
    SH_Yrotate90(SHTnsKit.GPU(), cfg, input, output)

function SH_Xrotate90(::SHTnsKit.GPU, cfg::SHTConfig,
                      input::AMDGPU.AnyROCArray{<:Complex,1},
                      output::AMDGPU.AnyROCArray{<:Complex,1})
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

SH_Xrotate90(cfg::SHTConfig, input::AMDGPU.AnyROCArray{<:Complex,1},
             output::AMDGPU.AnyROCArray{<:Complex,1}) =
    SH_Xrotate90(SHTnsKit.GPU(), cfg, input, output)

struct AMDGPULocalTables{TP,TD,TO,TS}
    Plm::TP
    dtheta::TD
    over_sin::TO
    scales::TS
end

function _amdgpu_local_tables(cfg::SHTConfig, ::Type{T}, cost::Real) where {T<:AbstractFloat}
    SHTnsKit._validate_local_cost(cost, :local_evaluation)
    x = T(cost)
    device = AMDGPU.device_id()
    identity = objectid(cfg)
    signature = hash((vector_config_signature(cfg), x))
    cached = scalar_cache_lookup(
        _AMDGPU_LOCAL_CACHE, device, identity, T, signature,
    )
    cached === nothing || return cached
    x_device = ROCArray(T[x])
    Nlm = ROCArray(T.(cfg.Nlm))
    Plm = AMDGPU.zeros(T, 1, cfg.lmax + 1, cfg.mmax + 1)
    dtheta = similar(Plm)
    over_sin = similar(Plm)
    vector_derivative_table_kernel!(ROCBackend())(
        Plm, dtheta, over_sin, x_device, Nlm, cfg.lmax, cfg.mmax;
        ndrange=(1, cfg.mmax + 1),
    )
    scales = _amdgpu_scalar_tables(cfg, T).scales
    built = AMDGPULocalTables(Plm, dtheta, over_sin, scales)
    return scalar_cache_publish!(AMDGPU.synchronize,
        _AMDGPU_LOCAL_CACHE, device, identity, T, signature, built,
    )
end

@inline function _amdgpu_local_precision(array)
    T = typeof(real(zero(eltype(array))))
    T in (Float32, Float64) || throw(ArgumentError(
        "GPU local evaluation supports ComplexF32 and ComplexF64 coefficients",
    ))
    return T
end

function _amdgpu_validate_local_arrays(operation::Symbol, arrays...)
    _require_amdgpu(operation)
    for array in arrays
        array isa AMDGPU.AnyROCArray || throw(ArgumentError(
            "$operation requires AMDGPU-owned coefficient storage",
        ))
    end
    first_type = eltype(first(arrays))
    all(array -> eltype(array) === first_type, arrays) || throw(ArgumentError(
        "$operation requires matching coefficient element types",
    ))
    return _amdgpu_local_precision(first(arrays))
end

function _amdgpu_local_scalar(cfg, coefficients, cost, phi;
                              nphi::Int=1, ltr::Int=cfg.lmax,
                              mtr::Int=cfg.mmax, complex_layout::Bool=false)
    T = _amdgpu_validate_local_arrays(
        complex_layout ? :synthesis_point_cplx : :synthesis_point, coefficients,
    )
    SHTnsKit._validate_local_coordinates(cost, phi, :local_evaluation)
    SHTnsKit._validate_local_nphi(nphi, :local_evaluation)
    0 <= ltr <= cfg.lmax || throw(ArgumentError("ltr must be within [0, lmax]"))
    0 <= mtr <= cfg.mmax || throw(ArgumentError("mtr must be within [0, mmax]"))
    complex_layout && cfg.mres != 1 && throw(ArgumentError(
        "complex local evaluation supports mres==1 only",
    ))
    if complex_layout
        length(coefficients) == SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1) ||
            throw(DimensionMismatch("alm length mismatch"))
    else
        size(coefficients) == (cfg.lmax + 1, cfg.mmax + 1) ||
            throw(DimensionMismatch("Qlm must be (lmax+1, mmax+1)"))
    end
    tables = _amdgpu_local_tables(cfg, T, cost)
    step = nphi == 1 ? zero(T) : T(2pi / nphi)
    if complex_layout
        output = similar(coefficients, Complex{T}, (nphi,))
        local_complex_kernel!(ROCBackend())(
            output, coefficients, tables.Plm, tables.scales,
            T(phi), step, cfg.lmax, cfg.mmax, ltr; ndrange=nphi,
        )
    else
        output = similar(coefficients, T, (nphi,))
        local_scalar_kernel!(ROCBackend())(
            output, coefficients, tables.Plm, tables.scales,
            T(phi), step, cfg.lmax, cfg.mmax, cfg.mres, ltr, mtr;
            ndrange=nphi,
        )
    end
    return nphi == 1 ? reshape(output, ()) : output
end

function _amdgpu_local_qst(cfg, Q, S, Tlm, cost, phi;
                           nphi::Int=1, ltr::Int=cfg.lmax,
                           mtr::Int=cfg.mmax,
                           has_q::Bool=true, has_s::Bool=true, has_t::Bool=true)
    arrays = has_t ? (Q, S, Tlm) : (Q, S)
    T = _amdgpu_validate_local_arrays(:SHqst_to_point, arrays...)
    SHTnsKit._validate_local_coordinates(cost, phi, :local_evaluation)
    SHTnsKit._validate_local_nphi(nphi, :local_evaluation)
    0 <= ltr <= cfg.lmax || throw(ArgumentError("ltr must be within [0, lmax]"))
    0 <= mtr <= cfg.mmax || throw(ArgumentError("mtr must be within [0, mmax]"))
    for (name, array) in zip(("Qlm", "Slm", "Tlm"), (Q, S, Tlm))
        ((name == "Tlm" && !has_t) || length(array) == cfg.nlm) ||
            throw(DimensionMismatch("$name length"))
    end
    tables = _amdgpu_local_tables(cfg, T, cost)
    Vr = similar(Q, T, (nphi,)); Vt = similar(Q, T, (nphi,)); Vp = similar(Q, T, (nphi,))
    step = nphi == 1 ? zero(T) : T(2pi / nphi)
    sinth = sqrt(max(zero(T), one(T) - T(cost)^2))
    local_qst_kernel!(ROCBackend())(
        Vr, Vt, Vp, Q, S, Tlm, tables.Plm, tables.dtheta,
        tables.over_sin, tables.scales, T(phi), step,
        cfg.lmax, cfg.mmax, cfg.mres, ltr, mtr,
        has_q, has_s, has_t, cfg.robert_form, sinth; ndrange=nphi,
    )
    return nphi == 1 ? (reshape(Vr, ()), reshape(Vt, ()), reshape(Vp, ())) :
                        (Vr, Vt, Vp)
end

synthesis_point(::SHTnsKit.GPU, cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{<:Complex,2}, cost::Real, phi::Real) =
    _amdgpu_local_scalar(cfg, coefficients, cost, phi)
synthesis_point(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{<:Complex,2}, cost::Real, phi::Real) =
    synthesis_point(SHTnsKit.GPU(), cfg, coefficients, cost, phi)
synthesis_point_cplx(::SHTnsKit.GPU, cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real, phi::Real) =
    _amdgpu_local_scalar(cfg, coefficients, cost, phi; complex_layout=true)
synthesis_point_cplx(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real, phi::Real) =
    synthesis_point_cplx(SHTnsKit.GPU(), cfg, coefficients, cost, phi)
SH_to_lat(::SHTnsKit.GPU, cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real;
          nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) =
    _amdgpu_local_qst(cfg, coefficients, coefficients, coefficients, cost, zero(cost);
                      nphi, ltr, mtr, has_s=false, has_t=false)[1]
SH_to_lat(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real; kwargs...) =
    SH_to_lat(SHTnsKit.GPU(), cfg, coefficients, cost; kwargs...)
SH_to_lat_cplx(::SHTnsKit.GPU, cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real;
               nphi::Int=cfg.nlon, ltr::Int=cfg.lmax) =
    _amdgpu_local_scalar(cfg, coefficients, cost, zero(cost); nphi, ltr, complex_layout=true)
SH_to_lat_cplx(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real; kwargs...) =
    SH_to_lat_cplx(SHTnsKit.GPU(), cfg, coefficients, cost; kwargs...)
SHqst_to_point(::SHTnsKit.GPU, cfg::SHTConfig, Q::AMDGPU.AnyROCArray{<:Complex,1}, S::AMDGPU.AnyROCArray{<:Complex,1}, Tlm::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real, phi::Real) =
    _amdgpu_local_qst(cfg, Q, S, Tlm, cost, phi)
SHqst_to_point(cfg::SHTConfig, Q::AMDGPU.AnyROCArray{<:Complex,1}, S::AMDGPU.AnyROCArray{<:Complex,1}, Tlm::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real, phi::Real) =
    SHqst_to_point(SHTnsKit.GPU(), cfg, Q, S, Tlm, cost, phi)
SHqst_to_lat(::SHTnsKit.GPU, cfg::SHTConfig, Q::AMDGPU.AnyROCArray{<:Complex,1}, S::AMDGPU.AnyROCArray{<:Complex,1}, Tlm::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real;
             nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) =
    _amdgpu_local_qst(cfg, Q, S, Tlm, cost, zero(cost); nphi, ltr, mtr)
SHqst_to_lat(cfg::SHTConfig, Q::AMDGPU.AnyROCArray{<:Complex,1}, S::AMDGPU.AnyROCArray{<:Complex,1}, Tlm::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real; kwargs...) =
    SHqst_to_lat(SHTnsKit.GPU(), cfg, Q, S, Tlm, cost; kwargs...)
SH_to_grad_point(::SHTnsKit.GPU, cfg::SHTConfig, Dr::AMDGPU.AnyROCArray{<:Complex,1}, S::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real, phi::Real) =
    _amdgpu_local_qst(cfg, Dr, S, S, cost, phi; has_t=false)
SH_to_grad_point(cfg::SHTConfig, Dr::AMDGPU.AnyROCArray{<:Complex,1}, S::AMDGPU.AnyROCArray{<:Complex,1}, cost::Real, phi::Real) =
    SH_to_grad_point(SHTnsKit.GPU(), cfg, Dr, S, cost, phi)

function _amdgpu_scalar_tables(cfg::SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    device = AMDGPU.device_id()
    identity = objectid(cfg)
    signature = scalar_config_signature(cfg)
    cached = scalar_cache_lookup(
        _AMDGPU_SCALAR_CACHE, device, identity, T, signature,
    )
    cached === nothing || return cached

    x_host, weights_host, scales_host = scalar_host_tables(cfg, T)
    x = ROCArray(x_host)
    weights = ROCArray(weights_host)
    scales = ROCArray(scales_host)
    Plm = AMDGPU.zeros(T, cfg.nlat, cfg.lmax + 1, cfg.mmax + 1)
    backend = ROCBackend()
    kernel! = legendre_table_kernel!(backend)
    kernel!(Plm, x, cfg.lmax, cfg.mmax;
            ndrange=(cfg.nlat, cfg.mmax + 1))
    built = AMDGPUScalarTables(x, weights, Plm, scales)

    return scalar_cache_publish!(AMDGPU.synchronize,
        _AMDGPU_SCALAR_CACHE, device, identity, T, signature, built,
    )
end

function _amdgpu_vector_tables(cfg::SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    device = AMDGPU.device_id()
    identity = objectid(cfg)
    signature = vector_config_signature(cfg)
    cached = scalar_cache_lookup(
        _AMDGPU_VECTOR_CACHE, device, identity, T, signature,
    )
    cached === nothing || return cached

    scalar = _amdgpu_scalar_tables(cfg, T)
    Nlm = ROCArray(T.(cfg.Nlm))
    Plm = similar(scalar.Plm)
    dtheta = similar(Plm)
    over_sin = similar(Plm)
    kernel! = vector_derivative_table_kernel!(ROCBackend())
    kernel!(Plm, dtheta, over_sin, scalar.x, Nlm, cfg.lmax, cfg.mmax;
            ndrange=(cfg.nlat, cfg.mmax + 1))
    built = AMDGPUVectorTables(
        scalar.x, scalar.weights, scalar.scales, dtheta, over_sin,
    )
    return scalar_cache_publish!(AMDGPU.synchronize,
        _AMDGPU_VECTOR_CACHE, device, identity, T, signature, built,
    )
end

function _amdgpu_clear_scalar_cache!(; device=nothing)
    scalar_cache_clear!(_AMDGPU_SCALAR_CACHE; device)
    scalar_cache_clear!(_AMDGPU_VECTOR_CACHE; device)
    scalar_cache_clear!(_AMDGPU_LOCAL_CACHE; device)
    scalar_workspace_clear!(_AMDGPU_WORKSPACE_CACHE; device)
    rotation_cache_clear!(_rotation_cache; device)
    return nothing
end

function _gpu_adapter_clear_cache!(::AMDGPUAdapter)
    _amdgpu_clear_scalar_cache!()
    return nothing
end

function _amdgpu_workspace_builder(cfg::SHTConfig, ::Type{RT}, nfields::Int,
                                   use_rfft::Bool) where {RT<:AbstractFloat}
    CT = Complex{RT}
    spatial_shape = nfields == 0 ? (cfg.nlat, cfg.nlon) :
                                   (cfg.nlat, cfg.nlon, nfields)
    spectral_shape = nfields == 0 ? (cfg.lmax + 1, cfg.mmax + 1) :
                                    (cfg.lmax + 1, cfg.mmax + 1, nfields)
    canonical = AMDGPU.zeros(CT, spectral_shape)
    if use_rfft
        real_buffer = AMDGPU.zeros(RT, spatial_shape)
        half_shape = Base.setindex(spatial_shape, cfg.nlon ÷ 2 + 1, 2)
        fourier = AMDGPU.zeros(CT, half_shape)
        forward = FFTW.plan_rfft(real_buffer, 2)
        inverse = FFTW.plan_irfft(fourier, cfg.nlon, 2)
        return (; canonical, fourier, real_buffer, forward, inverse)
    end
    fourier = AMDGPU.zeros(CT, spatial_shape)
    # rocFFT executes in-place plans with `plan * buffer`; its mul! API
    # is restricted to out-of-place (including real) transforms.
    forward = FFTW.plan_fft!(fourier, 2)
    inverse = FFTW.plan_ifft!(fourier, 2)
    return (; canonical, fourier, real_buffer=nothing, forward, inverse)
end

function _with_amdgpu_workspace(f, owner, cfg::SHTConfig, ::Type{RT},
                                nfields::Int, use_rfft::Bool) where {RT<:AbstractFloat}
    device = AMDGPU.device_id()
    kind = nfields == 0 ? :scalar : :batch
    shape = (cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres,
             nfields, use_rfft)
    signature = hash((scalar_config_signature(cfg), use_rfft, nfields))
    builder = () -> _amdgpu_workspace_builder(cfg, RT, nfields, use_rfft)
    return scalar_workspace_use!(
        f, builder, _AMDGPU_WORKSPACE_CACHE, device, owner, RT,
        kind, shape, signature,
    )
end

function _amdgpu_vector_workspace_builder(cfg::SHTConfig,
                                          ::Type{RT}) where {RT<:AbstractFloat}
    CT = Complex{RT}
    Ftheta = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon)
    Fphi = similar(Ftheta)
    return (;
        Ftheta,
        Fphi,
        forward_theta=FFTW.plan_fft!(Ftheta, 2),
        forward_phi=FFTW.plan_fft!(Fphi, 2),
        inverse_theta=FFTW.plan_ifft!(Ftheta, 2),
        inverse_phi=FFTW.plan_ifft!(Fphi, 2),
    )
end

function _with_amdgpu_vector_workspace(f, owner, cfg::SHTConfig,
                                       ::Type{RT}) where {RT<:AbstractFloat}
    device = AMDGPU.device_id()
    shape = (cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres)
    signature = hash((scalar_config_signature(cfg), :vector))
    builder = () -> _amdgpu_vector_workspace_builder(cfg, RT)
    return scalar_workspace_use!(
        f, builder, _AMDGPU_WORKSPACE_CACHE, device, owner, RT,
        :vector, shape, signature,
    )
end

function _amdgpu_batch_scratch(cfg::SHTConfig, fft_batch, ::Type{CT},
                               nfields::Int, use_rfft::Bool, operands...) where {CT<:Complex}
    fft_batch === nothing && return nothing
    fft_batch isa AMDGPU.AnyROCArray || throw(ArgumentError(
        "fft_batch must use AMDGPU storage",
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

function _amdgpu_scalar_analysis_direct!(owner, cfg::SHTConfig,
                                         output::AMDGPU.AnyROCArray,
                                         field::AMDGPU.AnyROCArray;
                                         use_rfft::Bool=false)
    _require_amdgpu(:analysis!)
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
    eltype(output) <: Complex || throw(ArgumentError("analysis output must be complex"))
    tables = _amdgpu_scalar_tables(cfg, RT)
    return _with_amdgpu_workspace(owner, cfg, RT, 0, use_rfft) do workspace
        fourier = workspace.fourier
        if use_rfft
            copyto!(workspace.real_buffer, field)
            mul!(fourier, workspace.forward, workspace.real_buffer)
        else
            copyto!(fourier, field)
            workspace.forward * fourier
        end
        backend = ROCBackend()
        scalar_analysis_kernel!(backend)(
            output, fourier, tables.Plm, tables.weights, RT(SHTnsKit._analysis_phi_scale(cfg)),
            cfg.lmax, cfg.mmax, cfg.mres, cfg.lmax;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        coefficient_conversion_kernel!(backend)(
            output, output, tables.scales, cfg.lmax, cfg.mmax, false;
            ndrange=(cfg.lmax + 1, cfg.mmax + 1),
        )
        AMDGPU.synchronize()
        output
    end
end

function _amdgpu_scalar_synthesis_direct!(owner, cfg::SHTConfig,
                                          output::AMDGPU.AnyROCArray,
                                          coefficients::AMDGPU.AnyROCArray;
                                          real_output::Bool=true,
                                          use_rfft::Bool=false)
    _require_amdgpu(:synthesis!)
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
    tables = _amdgpu_scalar_tables(cfg, RT)
    return _with_amdgpu_workspace(owner, cfg, RT, 0, use_rfft) do workspace
        backend = ROCBackend()
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
        AMDGPU.synchronize()
        if use_rfft
            mul!(workspace.real_buffer, workspace.inverse, workspace.fourier)
            copyto!(output, workspace.real_buffer)
        else
            workspace.inverse * workspace.fourier
            real_output ? (output .= real.(workspace.fourier)) :
                          copyto!(output, workspace.fourier)
        end
        AMDGPU.synchronize()
        output
    end
end

function _amdgpu_scalar_analysis(cfg::SHTConfig, field::AMDGPU.AnyROCArray;
                                 use_rfft::Bool=false, fft_scratch=nothing,
                                 lcap::Int=cfg.lmax)
    0 ≤ lcap ≤ cfg.lmax || throw(ArgumentError(
        "lcap must satisfy 0 ≤ lcap ≤ lmax=$(cfg.lmax)",
    ))
    _require_amdgpu(:analysis)
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon)), got $(size(field))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "AMDGPU scalar transforms do not accept a host fft_scratch",
    ))
    use_rfft && !(eltype(field) <: Real) && throw(ArgumentError(
        "use_rfft=true requires a real-valued input",
    ))
    RT = typeof(float(real(zero(eltype(field)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    # `use_rfft` is a performance hint. The current rocFFT adapter deliberately
    # shares the complex pipeline for valid real transforms.
    fourier = CT.(field)
    FFTW.fft!(fourier, 2)

    backend = ROCBackend()
    canonical = AMDGPU.zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    analyze! = scalar_analysis_kernel!(backend)
    analyze!(canonical, fourier, tables.Plm, tables.weights, RT(SHTnsKit._analysis_phi_scale(cfg)),
             cfg.lmax, cfg.mmax, cfg.mres, lcap;
             ndrange=(lcap + 1, min(cfg.mmax, lcap) + 1))

    configured = similar(canonical)
    convert! = coefficient_conversion_kernel!(backend)
    convert!(configured, canonical, tables.scales, cfg.lmax, cfg.mmax, false;
             ndrange=(cfg.lmax + 1, cfg.mmax + 1))
    AMDGPU.synchronize()
    return configured
end

function _amdgpu_scalar_synthesis(cfg::SHTConfig,
                                  coefficients::AMDGPU.AnyROCArray;
                                  real_output::Bool=true, use_rfft::Bool=false,
                                  fft_scratch=nothing, lcap::Int=cfg.lmax)
    0 ≤ lcap ≤ cfg.lmax || throw(ArgumentError(
        "lcap must satisfy 0 ≤ lcap ≤ lmax=$(cfg.lmax)",
    ))
    _require_amdgpu(:synthesis)
    size(coefficients) == (cfg.lmax + 1, cfg.mmax + 1) || throw(DimensionMismatch(
        "coefficients must have size ($(cfg.lmax + 1), $(cfg.mmax + 1)), got $(size(coefficients))",
    ))
    fft_scratch === nothing || throw(ArgumentError(
        "AMDGPU scalar transforms do not accept a host fft_scratch",
    ))
    use_rfft && !real_output && throw(ArgumentError(
        "use_rfft=true implies real_output",
    ))
    use_rfft && cfg.mmax > cfg.nlon ÷ 2 && throw(ArgumentError(
        "use_rfft=true requires mmax ≤ nlon÷2, got mmax=$(cfg.mmax), nlon=$(cfg.nlon)",
    ))
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    configured = CT.(coefficients)
    canonical = similar(configured)
    backend = ROCBackend()
    convert! = coefficient_conversion_kernel!(backend)
    convert!(canonical, configured, tables.scales, cfg.lmax, cfg.mmax, true;
             ndrange=(cfg.lmax + 1, cfg.mmax + 1))

    # The complex rocFFT route is mathematically identical for this valid hint.
    fourier = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon)
    synthesize! = scalar_synthesis_kernel!(backend)
    synthesize!(fourier, canonical, tables.Plm, RT(SHTnsKit.phi_inv_scale(cfg)),
                cfg.nlon, lcap, min(cfg.mmax, lcap), cfg.mres, real_output;
                ndrange=(cfg.nlat, min(cfg.mmax, lcap) + 1))
    AMDGPU.synchronize()
    FFTW.ifft!(fourier, 2)
    return real_output ? real.(fourier) : fourier
end

function _gpu_adapter_analysis(::AMDGPUAdapter, cfg::SHTConfig,
                               field::AMDGPU.AnyROCArray; kwargs...)
    return _amdgpu_scalar_analysis(cfg, field; kwargs...)
end

function _gpu_adapter_synthesis(::AMDGPUAdapter, cfg::SHTConfig,
                                coefficients::AMDGPU.AnyROCArray; kwargs...)
    return _amdgpu_scalar_synthesis(cfg, coefficients; kwargs...)
end

on_device(::AMDGPU.AnyROCArray) = SHTnsKit.GPU()

analysis(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,2}; kwargs...) where {T} =
    analysis(SHTnsKit.GPU(), cfg, field; kwargs...)
synthesis(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{T,2}; kwargs...) where {T} =
    synthesis(SHTnsKit.GPU(), cfg, coefficients; kwargs...)
synthesis_cplx(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{T,2}) where {T} =
    synthesis_cplx(SHTnsKit.GPU(), cfg, coefficients)

function _amdgpu_vector_analysis(cfg::SHTConfig,
                                 Vt::AMDGPU.AnyROCArray{T,2},
                                 Vp::AMDGPU.AnyROCArray{R,2};
                                 use_rfft::Bool=false,
                                 lcap::Int=cfg.lmax) where {T<:Number,R<:Number}
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    Sout = AMDGPU.zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    Tout = similar(Sout)
    return _amdgpu_vector_analysis_direct!(
        cfg, cfg, Sout, Tout, Vt, Vp; use_rfft, lcap,
    )
end

function _amdgpu_vector_analysis_direct!(owner, cfg::SHTConfig,
                                         Sout::AMDGPU.AnyROCArray,
                                         Tout::AMDGPU.AnyROCArray,
                                         Vt::AMDGPU.AnyROCArray{T,2},
                                         Vp::AMDGPU.AnyROCArray{R,2};
                                         use_rfft::Bool=false,
                                         lcap::Int=cfg.lmax) where {T<:Number,R<:Number}
    _require_amdgpu(:analysis_sphtor!)
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
    tables = _amdgpu_vector_tables(cfg, RT)
    return _with_amdgpu_vector_workspace(owner, cfg, RT) do workspace
        copyto!(workspace.Ftheta, Vt)
        copyto!(workspace.Fphi, Vp)
        workspace.forward_theta * workspace.Ftheta
        workspace.forward_phi * workspace.Fphi
        fill!(Sout, zero(CT)); fill!(Tout, zero(CT))
        vector_analysis_kernel!(ROCBackend())(
            Sout, Tout, workspace.Ftheta, workspace.Fphi,
            tables.dtheta, tables.over_sin, tables.weights, tables.scales,
            tables.x, RT(SHTnsKit._analysis_phi_scale(cfg)), lcap, min(cfg.mmax, lcap), cfg.mres,
            cfg.robert_form; ndrange=(lcap + 1, min(cfg.mmax, lcap) + 1),
        )
        AMDGPU.synchronize()
        Sout, Tout
    end
end

function _amdgpu_vector_synthesis(cfg::SHTConfig,
                                  Slm::AMDGPU.AnyROCArray{T,2},
                                  Tlm::AMDGPU.AnyROCArray{R,2};
                                  real_output::Bool=true,
                                  use_rfft::Bool=false,
                                  lcap::Int=cfg.lmax) where {T<:Number,R<:Number}
    RT = typeof(float(real(zero(T))))
    output_type = real_output ? RT : Complex{RT}
    Vt = AMDGPU.zeros(output_type, cfg.nlat, cfg.nlon)
    Vp = similar(Vt)
    return _amdgpu_vector_synthesis_direct!(
        cfg, cfg, Vt, Vp, Slm, Tlm; real_output, use_rfft, lcap,
    )
end

function _amdgpu_vector_synthesis_direct!(owner, cfg::SHTConfig,
                                          Vt::AMDGPU.AnyROCArray,
                                          Vp::AMDGPU.AnyROCArray,
                                          Slm::AMDGPU.AnyROCArray{T,2},
                                          Tlm::AMDGPU.AnyROCArray{R,2};
                                          real_output::Bool=true,
                                          use_rfft::Bool=false,
                                          lcap::Int=cfg.lmax) where {T<:Number,R<:Number}
    _require_amdgpu(:synthesis_sphtor!)
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
    tables = _amdgpu_vector_tables(cfg, RT)
    return _with_amdgpu_vector_workspace(owner, cfg, RT) do workspace
        fill!(workspace.Ftheta, zero(CT))
        fill!(workspace.Fphi, zero(CT))
        vector_synthesis_kernel!(ROCBackend())(
            workspace.Ftheta, workspace.Fphi, Slm, Tlm,
            tables.dtheta, tables.over_sin, tables.scales, tables.x,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, lcap, min(cfg.mmax, lcap),
            cfg.mres, real_output, cfg.robert_form;
            ndrange=(cfg.nlat, min(cfg.mmax, lcap) + 1),
        )
        AMDGPU.synchronize()
        workspace.inverse_theta * workspace.Ftheta
        workspace.inverse_phi * workspace.Fphi
        if real_output
            Vt .= real.(workspace.Ftheta)
            Vp .= real.(workspace.Fphi)
        else
            copyto!(Vt, workspace.Ftheta)
            copyto!(Vp, workspace.Fphi)
        end
        AMDGPU.synchronize()
        Vt, Vp
    end
end

_gpu_adapter_analysis_sphtor(::AMDGPUAdapter, cfg::SHTConfig,
                             Vt::AMDGPU.AnyROCArray,
                             Vp::AMDGPU.AnyROCArray; kwargs...) =
    _amdgpu_vector_analysis(cfg, Vt, Vp; kwargs...)
_gpu_adapter_synthesis_sphtor(::AMDGPUAdapter, cfg::SHTConfig,
                              Slm::AMDGPU.AnyROCArray,
                              Tlm::AMDGPU.AnyROCArray; kwargs...) =
    _amdgpu_vector_synthesis(cfg, Slm, Tlm; kwargs...)

analysis_sphtor(cfg::SHTConfig, Vt::AMDGPU.AnyROCArray{T,2},
                 Vp::AMDGPU.AnyROCArray{R,2}; kwargs...) where {T,R} =
    analysis_sphtor(SHTnsKit.GPU(), cfg, Vt, Vp; kwargs...)
analysis_sphtor_cplx(cfg::SHTConfig, Vt::AMDGPU.AnyROCArray{T,2},
                      Vp::AMDGPU.AnyROCArray{R,2}) where {T<:Complex,R<:Complex} =
    analysis_sphtor_cplx(SHTnsKit.GPU(), cfg, Vt, Vp)
synthesis_sphtor(cfg::SHTConfig, Slm::AMDGPU.AnyROCArray{T,2},
                  Tlm::AMDGPU.AnyROCArray{R,2}; kwargs...) where {T,R} =
    synthesis_sphtor(SHTnsKit.GPU(), cfg, Slm, Tlm; kwargs...)
synthesis_sphtor_cplx(cfg::SHTConfig, Slm::AMDGPU.AnyROCArray{T,2},
                       Tlm::AMDGPU.AnyROCArray{R,2}) where {T,R} =
    synthesis_sphtor_cplx(SHTnsKit.GPU(), cfg, Slm, Tlm)
synthesis_sph(cfg::SHTConfig, Slm::AMDGPU.AnyROCArray{T,2}; kwargs...) where {T} =
    synthesis_sph(SHTnsKit.GPU(), cfg, Slm; kwargs...)
synthesis_sph_cplx(cfg::SHTConfig, Slm::AMDGPU.AnyROCArray{T,2}) where {T} =
    synthesis_sph_cplx(SHTnsKit.GPU(), cfg, Slm)
synthesis_tor(cfg::SHTConfig, Tlm::AMDGPU.AnyROCArray{T,2}; kwargs...) where {T} =
    synthesis_tor(SHTnsKit.GPU(), cfg, Tlm; kwargs...)
synthesis_tor_cplx(cfg::SHTConfig, Tlm::AMDGPU.AnyROCArray{T,2}) where {T} =
    synthesis_tor_cplx(SHTnsKit.GPU(), cfg, Tlm)
analysis_qst(cfg::SHTConfig, Vr::AMDGPU.AnyROCArray{T,2},
             Vt::AMDGPU.AnyROCArray{R,2}, Vp::AMDGPU.AnyROCArray{S,2};
             kwargs...) where {T,R,S} =
    analysis_qst(SHTnsKit.GPU(), cfg, Vr, Vt, Vp; kwargs...)
analysis_qst_cplx(cfg::SHTConfig, Vr::AMDGPU.AnyROCArray{T,2},
                  Vt::AMDGPU.AnyROCArray{R,2},
                  Vp::AMDGPU.AnyROCArray{S,2}) where {T<:Complex,R<:Complex,S<:Complex} =
    analysis_qst_cplx(SHTnsKit.GPU(), cfg, Vr, Vt, Vp)
synthesis_qst(cfg::SHTConfig, Qlm::AMDGPU.AnyROCArray{T,2},
              Slm::AMDGPU.AnyROCArray{R,2}, Tlm::AMDGPU.AnyROCArray{S,2};
              kwargs...) where {T,R,S} =
    synthesis_qst(SHTnsKit.GPU(), cfg, Qlm, Slm, Tlm; kwargs...)
synthesis_qst_cplx(cfg::SHTConfig, Qlm::AMDGPU.AnyROCArray{T,2},
                   Slm::AMDGPU.AnyROCArray{R,2},
                   Tlm::AMDGPU.AnyROCArray{S,2}) where {T,R,S} =
    synthesis_qst_cplx(SHTnsKit.GPU(), cfg, Qlm, Slm, Tlm)

# Degree/fixed-order/vector/QST variants stay wholly in ROCArray storage.
analysis_sphtor_l(::SHTnsKit.GPU, cfg::SHTConfig,
                  Vt::AMDGPU.AnyROCArray{T,2}, Vp::AMDGPU.AnyROCArray{R,2},
                  ltr::Integer) where {T,R} =
    _amdgpu_vector_analysis(cfg, Vt, Vp;
                            lcap=SHTnsKit._validate_degree_limit(cfg, ltr))
analysis_sphtor_l(cfg::SHTConfig, Vt::AMDGPU.AnyROCArray{T,2},
                  Vp::AMDGPU.AnyROCArray{R,2}, ltr::Integer) where {T,R} =
    analysis_sphtor_l(SHTnsKit.GPU(), cfg, Vt, Vp, ltr)
synthesis_sphtor_l(::SHTnsKit.GPU, cfg::SHTConfig,
                   S::AMDGPU.AnyROCArray{T,2}, Tlm::AMDGPU.AnyROCArray{R,2},
                   ltr::Integer; real_output::Bool=true) where {T,R} =
    _amdgpu_vector_synthesis(cfg, S, Tlm; real_output,
                             lcap=SHTnsKit._validate_degree_limit(cfg, ltr))
synthesis_sphtor_l(cfg::SHTConfig, S::AMDGPU.AnyROCArray{T,2},
                   Tlm::AMDGPU.AnyROCArray{R,2}, ltr::Integer;
                   kwargs...) where {T,R} =
    synthesis_sphtor_l(SHTnsKit.GPU(), cfg, S, Tlm, ltr; kwargs...)
synthesis_sphtor_l_cplx(cfg::SHTConfig, S::AMDGPU.AnyROCArray{T,2},
                        Tlm::AMDGPU.AnyROCArray{R,2}, ltr::Integer) where {T,R} =
    synthesis_sphtor_l(cfg, S, Tlm, ltr; real_output=false)
synthesis_sphtor_l_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                        S::AMDGPU.AnyROCArray{T,2},
                        Tlm::AMDGPU.AnyROCArray{R,2}, ltr::Integer) where {T,R} =
    synthesis_sphtor_l(SHTnsKit.GPU(), cfg, S, Tlm, ltr; real_output=false)

_amdgpu_zero_spectrum(reference::AMDGPU.AnyROCArray, cfg::SHTConfig) =
    AMDGPU.zeros(eltype(reference), cfg.lmax + 1, cfg.mmax + 1)
synthesis_sph_l(cfg::SHTConfig, S::AMDGPU.AnyROCArray{T,2}, ltr::Integer;
                real_output::Bool=true) where {T} =
    synthesis_sphtor_l(cfg, S, _amdgpu_zero_spectrum(S, cfg), ltr; real_output)
synthesis_sph_l(::SHTnsKit.GPU, cfg::SHTConfig,
                S::AMDGPU.AnyROCArray{T,2}, ltr::Integer;
                kwargs...) where {T} = synthesis_sph_l(cfg, S, ltr; kwargs...)
synthesis_sph_l_cplx(cfg::SHTConfig, S::AMDGPU.AnyROCArray{T,2},
                     ltr::Integer) where {T} = synthesis_sph_l(cfg, S, ltr; real_output=false)
synthesis_sph_l_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     S::AMDGPU.AnyROCArray{T,2}, ltr::Integer) where {T} =
    synthesis_sph_l(cfg, S, ltr; real_output=false)
synthesis_tor_l(cfg::SHTConfig, Tlm::AMDGPU.AnyROCArray{T,2}, ltr::Integer;
                real_output::Bool=true) where {T} =
    synthesis_sphtor_l(cfg, _amdgpu_zero_spectrum(Tlm, cfg), Tlm, ltr; real_output)
synthesis_tor_l(::SHTnsKit.GPU, cfg::SHTConfig,
                Tlm::AMDGPU.AnyROCArray{T,2}, ltr::Integer;
                kwargs...) where {T} = synthesis_tor_l(cfg, Tlm, ltr; kwargs...)
synthesis_tor_l_cplx(cfg::SHTConfig, Tlm::AMDGPU.AnyROCArray{T,2},
                     ltr::Integer) where {T} = synthesis_tor_l(cfg, Tlm, ltr; real_output=false)
synthesis_tor_l_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     Tlm::AMDGPU.AnyROCArray{T,2}, ltr::Integer) where {T} =
    synthesis_tor_l(cfg, Tlm, ltr; real_output=false)

function _amdgpu_vector_mode_analysis(cfg::SHTConfig, stored_im::Integer,
                                      Vt::AMDGPU.AnyROCArray{T,1},
                                      Vp::AMDGPU.AnyROCArray{R,1},
                                      ltr::Integer) where {T<:Complex,R<:Complex}
    physical_m, lcap = SHTnsKit._validate_vector_fixed_order(cfg, stored_im, ltr)
    length(Vt) == cfg.nlat || throw(DimensionMismatch("Vt mode length mismatch"))
    length(Vp) == cfg.nlat || throw(DimensionMismatch("Vp mode length mismatch"))
    RTt = typeof(float(real(zero(T)))); RTp = typeof(float(real(zero(R))))
    RTt === RTp || throw(ArgumentError("mode components must have the same precision"))
    RT = RTt; CT = Complex{RT}; tables = _amdgpu_vector_tables(cfg, RT)
    S = AMDGPU.zeros(CT, lcap - physical_m + 1); Tlm = similar(S)
    vector_mode_analysis_kernel!(ROCBackend())(
        S, Tlm, Vt, Vp, tables.dtheta, tables.over_sin, tables.weights,
        tables.scales, tables.x, RT(SHTnsKit._analysis_phi_scale(cfg)), physical_m, lcap,
        cfg.robert_form; ndrange=length(S),
    )
    AMDGPU.synchronize()
    return S, Tlm
end

function _amdgpu_vector_mode_synthesis(cfg::SHTConfig, stored_im::Integer,
                                       S::AMDGPU.AnyROCArray{T,1},
                                       Tlm::AMDGPU.AnyROCArray{R,1},
                                       ltr::Integer) where {T<:Complex,R<:Complex}
    physical_m, lcap = SHTnsKit._validate_vector_fixed_order(cfg, stored_im, ltr)
    expected = lcap - physical_m + 1
    length(S) == expected || throw(DimensionMismatch("S mode length mismatch"))
    length(Tlm) == expected || throw(DimensionMismatch("T mode length mismatch"))
    RTs = typeof(float(real(zero(T)))); RTt = typeof(float(real(zero(R))))
    RTs === RTt || throw(ArgumentError("mode coefficients must have the same precision"))
    RT = RTs; CT = Complex{RT}; tables = _amdgpu_vector_tables(cfg, RT)
    Vt = AMDGPU.zeros(CT, cfg.nlat); Vp = similar(Vt)
    vector_mode_synthesis_kernel!(ROCBackend())(
        Vt, Vp, S, Tlm, tables.dtheta, tables.over_sin, tables.scales,
        tables.x, RT(SHTnsKit.phi_inv_scale(cfg)), physical_m, lcap,
        cfg.robert_form; ndrange=cfg.nlat,
    )
    AMDGPU.synchronize()
    return Vt, Vp
end

analysis_sphtor_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                   Vt::AMDGPU.AnyROCArray{T,1}, Vp::AMDGPU.AnyROCArray{R,1},
                   ltr::Integer) where {T<:Complex,R<:Complex} =
    _amdgpu_vector_mode_analysis(cfg, im, Vt, Vp, ltr)
analysis_sphtor_ml(cfg::SHTConfig, im::Integer, Vt::AMDGPU.AnyROCArray{T,1},
                   Vp::AMDGPU.AnyROCArray{R,1}, ltr::Integer) where {T<:Complex,R<:Complex} =
    analysis_sphtor_ml(SHTnsKit.GPU(), cfg, im, Vt, Vp, ltr)
synthesis_sphtor_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                    S::AMDGPU.AnyROCArray{T,1}, Tlm::AMDGPU.AnyROCArray{R,1},
                    ltr::Integer) where {T<:Complex,R<:Complex} =
    _amdgpu_vector_mode_synthesis(cfg, im, S, Tlm, ltr)
synthesis_sphtor_ml(cfg::SHTConfig, im::Integer, S::AMDGPU.AnyROCArray{T,1},
                    Tlm::AMDGPU.AnyROCArray{R,1}, ltr::Integer) where {T<:Complex,R<:Complex} =
    synthesis_sphtor_ml(SHTnsKit.GPU(), cfg, im, S, Tlm, ltr)
synthesis_sph_ml(cfg::SHTConfig, im::Integer, S::AMDGPU.AnyROCArray{T,1},
                 ltr::Integer) where {T<:Complex} =
    synthesis_sphtor_ml(cfg, im, S, AMDGPU.zeros(T, length(S)), ltr)
synthesis_sph_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                 S::AMDGPU.AnyROCArray{T,1}, ltr::Integer) where {T<:Complex} =
    synthesis_sph_ml(cfg, im, S, ltr)
synthesis_tor_ml(cfg::SHTConfig, im::Integer, Tlm::AMDGPU.AnyROCArray{T,1},
                 ltr::Integer) where {T<:Complex} =
    synthesis_sphtor_ml(cfg, im, AMDGPU.zeros(T, length(Tlm)), Tlm, ltr)
synthesis_tor_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                 Tlm::AMDGPU.AnyROCArray{T,1}, ltr::Integer) where {T<:Complex} =
    synthesis_tor_ml(cfg, im, Tlm, ltr)
synthesis_grad(cfg::SHTConfig, S::AMDGPU.AnyROCArray{T,2}; kwargs...) where {T} =
    synthesis_sph(cfg, S; kwargs...)
synthesis_grad(::SHTnsKit.GPU, cfg::SHTConfig,
               S::AMDGPU.AnyROCArray{T,2}; kwargs...) where {T} =
    synthesis_grad(cfg, S; kwargs...)
synthesis_grad_l(cfg::SHTConfig, S::AMDGPU.AnyROCArray{T,2}, ltr::Integer;
                 kwargs...) where {T} = synthesis_sph_l(cfg, S, ltr; kwargs...)
synthesis_grad_l(::SHTnsKit.GPU, cfg::SHTConfig,
                 S::AMDGPU.AnyROCArray{T,2}, ltr::Integer;
                 kwargs...) where {T} = synthesis_grad_l(cfg, S, ltr; kwargs...)
synthesis_grad_ml(cfg::SHTConfig, im::Integer, S::AMDGPU.AnyROCArray{T,1},
                  ltr::Integer) where {T<:Complex} = synthesis_sph_ml(cfg, im, S, ltr)
synthesis_grad_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                  S::AMDGPU.AnyROCArray{T,1}, ltr::Integer) where {T<:Complex} =
    synthesis_grad_ml(cfg, im, S, ltr)

analysis_qst_l(::SHTnsKit.GPU, cfg::SHTConfig, Vr::AMDGPU.AnyROCArray{T,2},
               Vt::AMDGPU.AnyROCArray{R,2}, Vp::AMDGPU.AnyROCArray{U,2},
               ltr::Integer) where {T,R,U} = begin
    lcap = SHTnsKit._validate_degree_limit(cfg, ltr)
    (_amdgpu_scalar_analysis(cfg, Vr; lcap),
     _amdgpu_vector_analysis(cfg, Vt, Vp; lcap)...)
end
analysis_qst_l(cfg::SHTConfig, Vr::AMDGPU.AnyROCArray{T,2},
               Vt::AMDGPU.AnyROCArray{R,2}, Vp::AMDGPU.AnyROCArray{U,2},
               ltr::Integer) where {T,R,U} = analysis_qst_l(SHTnsKit.GPU(), cfg, Vr, Vt, Vp, ltr)
synthesis_qst_l(::SHTnsKit.GPU, cfg::SHTConfig, Q::AMDGPU.AnyROCArray{T,2},
                S::AMDGPU.AnyROCArray{R,2}, Tlm::AMDGPU.AnyROCArray{U,2},
                ltr::Integer; real_output::Bool=true) where {T,R,U} = begin
    lcap = SHTnsKit._validate_degree_limit(cfg, ltr)
    (_amdgpu_scalar_synthesis(cfg, Q; real_output, lcap),
     _amdgpu_vector_synthesis(cfg, S, Tlm; real_output, lcap)...)
end
synthesis_qst_l(cfg::SHTConfig, Q::AMDGPU.AnyROCArray{T,2},
                S::AMDGPU.AnyROCArray{R,2}, Tlm::AMDGPU.AnyROCArray{U,2},
                ltr::Integer; kwargs...) where {T,R,U} =
    synthesis_qst_l(SHTnsKit.GPU(), cfg, Q, S, Tlm, ltr; kwargs...)
synthesis_qst_l_cplx(cfg::SHTConfig, Q::AMDGPU.AnyROCArray{T,2},
                     S::AMDGPU.AnyROCArray{R,2}, Tlm::AMDGPU.AnyROCArray{U,2},
                     ltr::Integer) where {T,R,U} = synthesis_qst_l(cfg, Q, S, Tlm, ltr; real_output=false)
synthesis_qst_l_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     Q::AMDGPU.AnyROCArray{T,2}, S::AMDGPU.AnyROCArray{R,2},
                     Tlm::AMDGPU.AnyROCArray{U,2}, ltr::Integer) where {T,R,U} =
    synthesis_qst_l(SHTnsKit.GPU(), cfg, Q, S, Tlm, ltr; real_output=false)
analysis_qst_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                Vr::AMDGPU.AnyROCArray{T,1}, Vt::AMDGPU.AnyROCArray{R,1},
                Vp::AMDGPU.AnyROCArray{U,1}, ltr::Integer) where {T<:Complex,R<:Complex,U<:Complex} = begin
    stored_im, _, lcap = SHTnsKit._validate_stored_order(cfg, im, ltr)
    (analysis_packed_ml(SHTnsKit.GPU(), cfg, stored_im, Vr, lcap),
     _amdgpu_vector_mode_analysis(cfg, stored_im, Vt, Vp, lcap)...)
end
analysis_qst_ml(cfg::SHTConfig, im::Integer, Vr::AMDGPU.AnyROCArray{T,1},
                Vt::AMDGPU.AnyROCArray{R,1}, Vp::AMDGPU.AnyROCArray{U,1},
                ltr::Integer) where {T<:Complex,R<:Complex,U<:Complex} =
    analysis_qst_ml(SHTnsKit.GPU(), cfg, im, Vr, Vt, Vp, ltr)
synthesis_qst_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Integer,
                 Q::AMDGPU.AnyROCArray{T,1}, S::AMDGPU.AnyROCArray{R,1},
                 Tlm::AMDGPU.AnyROCArray{U,1}, ltr::Integer) where {T<:Complex,R<:Complex,U<:Complex} = begin
    stored_im, _, lcap = SHTnsKit._validate_stored_order(cfg, im, ltr)
    (synthesis_packed_ml(SHTnsKit.GPU(), cfg, stored_im, Q, lcap),
     _amdgpu_vector_mode_synthesis(cfg, stored_im, S, Tlm, lcap)...)
end
synthesis_qst_ml(cfg::SHTConfig, im::Integer, Q::AMDGPU.AnyROCArray{T,1},
                 S::AMDGPU.AnyROCArray{R,1}, Tlm::AMDGPU.AnyROCArray{U,1},
                 ltr::Integer) where {T<:Complex,R<:Complex,U<:Complex} =
    synthesis_qst_ml(SHTnsKit.GPU(), cfg, im, Q, S, Tlm, ltr)

function _amdgpu_vector_batch_analysis(cfg::SHTConfig,
                                       Vt::AMDGPU.AnyROCArray{T,3},
                                       Vp::AMDGPU.AnyROCArray{R,3}) where {T<:Real,R<:Real}
    size(Vt, 1) == cfg.nlat && size(Vt, 2) == cfg.nlon ||
        throw(DimensionMismatch("vector batch must start with (nlat, nlon)"))
    size(Vp) == size(Vt) || throw(DimensionMismatch("Vt/Vp batch shape mismatch"))
    nfields = size(Vt, 3)
    nfields > 0 || throw(ArgumentError("analysis_sphtor_batch requires at least one field"))
    RTt = float(T); RTp = float(R)
    RTt === RTp || throw(ArgumentError("vector batches must use the same precision"))
    RT = RTt; CT = Complex{RT}; tables = _amdgpu_vector_tables(cfg, RT)
    Ft = CT.(Vt); Fp = CT.(Vp)
    FFTW.fft!(Ft, 2); FFTW.fft!(Fp, 2)
    S = AMDGPU.zeros(CT, cfg.lmax + 1, cfg.mmax + 1, nfields); Tlm = similar(S)
    vector_batch_analysis_kernel!(ROCBackend())(
        S, Tlm, Ft, Fp, tables.dtheta, tables.over_sin, tables.weights,
        tables.scales, tables.x, RT(SHTnsKit._analysis_phi_scale(cfg)), cfg.lmax, cfg.mmax,
        cfg.mres, cfg.robert_form; ndrange=size(S),
    )
    AMDGPU.synchronize()
    return S, Tlm
end

function _amdgpu_vector_batch_synthesis(cfg::SHTConfig,
                                        S::AMDGPU.AnyROCArray{T,3},
                                        Tlm::AMDGPU.AnyROCArray{R,3};
                                        real_output::Bool=true) where {T<:Complex,R<:Complex}
    size(S, 1) == cfg.lmax + 1 && size(S, 2) == cfg.mmax + 1 ||
        throw(DimensionMismatch("vector coefficient batch has wrong spectral shape"))
    size(Tlm) == size(S) || throw(DimensionMismatch("S/T batch shape mismatch"))
    nfields = size(S, 3)
    nfields > 0 || throw(ArgumentError("synthesis_sphtor_batch requires at least one field"))
    RTs = typeof(float(real(zero(T)))); RTt = typeof(float(real(zero(R))))
    RTs === RTt || throw(ArgumentError("vector batches must use the same precision"))
    RT = RTs; CT = Complex{RT}; tables = _amdgpu_vector_tables(cfg, RT)
    Ft = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon, nfields)
    Fp = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon, nfields)
    vector_batch_synthesis_kernel!(ROCBackend())(
        Ft, Fp, S, Tlm, tables.dtheta, tables.over_sin, tables.scales,
        tables.x, RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, cfg.lmax,
        cfg.mmax, cfg.mres, real_output, cfg.robert_form;
        ndrange=(cfg.nlat, cfg.mmax + 1, nfields),
    )
    AMDGPU.synchronize()
    FFTW.ifft!(Ft, 2); FFTW.ifft!(Fp, 2)
    return real_output ? (real.(Ft), real.(Fp)) : (Ft, Fp)
end

analysis_sphtor_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                      Vt::AMDGPU.AnyROCArray{T,3},
                      Vp::AMDGPU.AnyROCArray{R,3}) where {T<:Real,R<:Real} =
    _amdgpu_vector_batch_analysis(cfg, Vt, Vp)
analysis_sphtor_batch(cfg::SHTConfig, Vt::AMDGPU.AnyROCArray{T,3},
                      Vp::AMDGPU.AnyROCArray{R,3}) where {T<:Real,R<:Real} =
    analysis_sphtor_batch(SHTnsKit.GPU(), cfg, Vt, Vp)
synthesis_sphtor_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                       S::AMDGPU.AnyROCArray{T,3},
                       Tlm::AMDGPU.AnyROCArray{R,3};
                       real_output::Bool=true) where {T<:Complex,R<:Complex} =
    _amdgpu_vector_batch_synthesis(cfg, S, Tlm; real_output)
synthesis_sphtor_batch(cfg::SHTConfig, S::AMDGPU.AnyROCArray{T,3},
                       Tlm::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Complex,R<:Complex} =
    synthesis_sphtor_batch(SHTnsKit.GPU(), cfg, S, Tlm; kwargs...)
synthesis_sphtor_batch_cplx(cfg::SHTConfig, S::AMDGPU.AnyROCArray{T,3},
                            Tlm::AMDGPU.AnyROCArray{R,3}) where {T<:Complex,R<:Complex} =
    synthesis_sphtor_batch(SHTnsKit.GPU(), cfg, S, Tlm; real_output=false)
synthesis_sphtor_batch_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                            S::AMDGPU.AnyROCArray{T,3},
                            Tlm::AMDGPU.AnyROCArray{R,3}) where {T<:Complex,R<:Complex} =
    synthesis_sphtor_batch(SHTnsKit.GPU(), cfg, S, Tlm; real_output=false)
analysis_qst_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                   Vr::AMDGPU.AnyROCArray{T,3}, Vt::AMDGPU.AnyROCArray{R,3},
                   Vp::AMDGPU.AnyROCArray{U,3}) where {T<:Real,R<:Real,U<:Real} =
    (_amdgpu_batch_analysis(cfg, Vr), _amdgpu_vector_batch_analysis(cfg, Vt, Vp)...)
analysis_qst_batch(cfg::SHTConfig, Vr::AMDGPU.AnyROCArray{T,3},
                   Vt::AMDGPU.AnyROCArray{R,3}, Vp::AMDGPU.AnyROCArray{U,3}) where {T<:Real,R<:Real,U<:Real} =
    analysis_qst_batch(SHTnsKit.GPU(), cfg, Vr, Vt, Vp)
synthesis_qst_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                    Q::AMDGPU.AnyROCArray{T,3}, S::AMDGPU.AnyROCArray{R,3},
                    Tlm::AMDGPU.AnyROCArray{U,3};
                    real_output::Bool=true) where {T<:Complex,R<:Complex,U<:Complex} =
    (_amdgpu_batch_synthesis(cfg, Q; real_output),
     _amdgpu_vector_batch_synthesis(cfg, S, Tlm; real_output)...)
synthesis_qst_batch(cfg::SHTConfig, Q::AMDGPU.AnyROCArray{T,3},
                    S::AMDGPU.AnyROCArray{R,3}, Tlm::AMDGPU.AnyROCArray{U,3};
                    kwargs...) where {T<:Complex,R<:Complex,U<:Complex} =
    synthesis_qst_batch(SHTnsKit.GPU(), cfg, Q, S, Tlm; kwargs...)
synthesis_qst_batch_cplx(cfg::SHTConfig, Q::AMDGPU.AnyROCArray{T,3},
                         S::AMDGPU.AnyROCArray{R,3},
                         Tlm::AMDGPU.AnyROCArray{U,3}) where {T<:Complex,R<:Complex,U<:Complex} =
    synthesis_qst_batch(SHTnsKit.GPU(), cfg, Q, S, Tlm; real_output=false)
synthesis_qst_batch_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                         Q::AMDGPU.AnyROCArray{T,3}, S::AMDGPU.AnyROCArray{R,3},
                         Tlm::AMDGPU.AnyROCArray{U,3}) where {T<:Complex,R<:Complex,U<:Complex} =
    synthesis_qst_batch(SHTnsKit.GPU(), cfg, Q, S, Tlm; real_output=false)

function analysis_sphtor!(plan::SHTPlan, Sout::AMDGPU.AnyROCArray,
                           Tout::AMDGPU.AnyROCArray,
                           Vt::AMDGPU.AnyROCArray, Vp::AMDGPU.AnyROCArray)
    return _amdgpu_vector_analysis_direct!(
        plan, plan.cfg, Sout, Tout, Vt, Vp; use_rfft=plan.use_rfft,
    )
end

function synthesis_sphtor!(plan::SHTPlan, Vt::AMDGPU.AnyROCArray,
                            Vp::AMDGPU.AnyROCArray,
                            Slm::AMDGPU.AnyROCArray,
                            Tlm::AMDGPU.AnyROCArray; real_output::Bool=true)
    return _amdgpu_vector_synthesis_direct!(
        plan, plan.cfg, Vt, Vp, Slm, Tlm;
        real_output, use_rfft=plan.use_rfft,
    )
end

function _amdgpu_vector_diagonal!(output::AMDGPU.AnyROCArray,
                                  cfg::SHTConfig,
                                  input::AMDGPU.AnyROCArray;
                                  inverse::Bool)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    size(input) == expected || throw(DimensionMismatch("input must have size $expected"))
    size(output) == expected || throw(DimensionMismatch("output must have size $expected"))
    eltype(output) === eltype(input) || throw(ArgumentError(
        "vector operator input and output element types must match",
    ))
    source = output === input || !Base.mightalias(output, input) ? input : copy(input)
    vector_diagonal_kernel!(ROCBackend())(
        output, source, cfg.lmax, cfg.mmax, cfg.mres, inverse;
        ndrange=expected,
    )
    AMDGPU.synchronize()
    return output
end

function _amdgpu_operator_matrix!(cfg::SHTConfig,
                                  mx::AMDGPU.AnyROCArray{T,1},
                                  derivative::Bool) where {T<:Real}
    _require_amdgpu(derivative ? :st_dt_matrix : :mul_ct_matrix)
    length(mx) == 2cfg.nlm || throw(DimensionMismatch(
        "mx length must be 2*nlm=$(2cfg.nlm)",
    ))
    metadata = SHTnsKit._gpu_operator_metadata(cfg, T)
    li = ROCArray(metadata.li); mi = ROCArray(metadata.mi)
    down = ROCArray(metadata.down_ratios); up = ROCArray(metadata.up_ratios)
    operator_matrix_kernel!(ROCBackend())(
        mx, li, mi, down, up, cfg.lmax, derivative; ndrange=cfg.nlm,
    )
    AMDGPU.synchronize()
    return mx
end

mul_ct_matrix(::SHTnsKit.GPU, cfg::SHTConfig,
              mx::AMDGPU.AnyROCArray{T,1}) where {T<:Real} =
    _amdgpu_operator_matrix!(cfg, mx, false)
mul_ct_matrix(cfg::SHTConfig, mx::AMDGPU.AnyROCArray{T,1}) where {T<:Real} =
    mul_ct_matrix(SHTnsKit.GPU(), cfg, mx)
st_dt_matrix(::SHTnsKit.GPU, cfg::SHTConfig,
             mx::AMDGPU.AnyROCArray{T,1}) where {T<:Real} =
    _amdgpu_operator_matrix!(cfg, mx, true)
st_dt_matrix(cfg::SHTConfig, mx::AMDGPU.AnyROCArray{T,1}) where {T<:Real} =
    st_dt_matrix(SHTnsKit.GPU(), cfg, mx)

function SH_mul_mx(::SHTnsKit.GPU, cfg::SHTConfig,
                   mx::AMDGPU.AnyROCArray{<:Real,1},
                   input::AMDGPU.AnyROCArray{<:Complex,1},
                   output::AMDGPU.AnyROCArray{<:Complex,1})
    _require_amdgpu(:SH_mul_mx)
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
    lower = ROCArray(metadata.lower); upper = ROCArray(metadata.upper)
    packed_operator_kernel!(ROCBackend())(
        output, input, mx, lower, upper; ndrange=cfg.nlm,
    )
    AMDGPU.synchronize()
    return output
end

SH_mul_mx(cfg::SHTConfig, mx::AMDGPU.AnyROCArray{<:Real,1},
          input::AMDGPU.AnyROCArray{<:Complex,1},
          output::AMDGPU.AnyROCArray{<:Complex,1}) =
    SH_mul_mx(SHTnsKit.GPU(), cfg, mx, input, output)

function gpu_apply_laplacian!(cfg::SHTConfig, coeffs::AMDGPU.AnyROCArray;
                              device=SHTnsKit.GPU())
    device isa SHTnsKit.GPU || throw(ArgumentError(
        "gpu_apply_laplacian! requires GPU(), got $(typeof(device))",
    ))
    _require_amdgpu(:gpu_apply_laplacian!)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    size(coeffs) == expected || throw(DimensionMismatch(
        "coeffs must have size $expected",
    ))
    laplacian_kernel!(ROCBackend())(
        coeffs, coeffs, cfg.lmax, cfg.mmax, cfg.mres; ndrange=expected,
    )
    AMDGPU.synchronize()
    return coeffs
end

divergence_from_spheroidal(cfg::SHTConfig, input::AMDGPU.AnyROCArray) =
    divergence_from_spheroidal!(cfg, similar(input), input)
divergence_from_spheroidal(::SHTnsKit.GPU, cfg::SHTConfig,
                           input::AMDGPU.AnyROCArray) =
    divergence_from_spheroidal(cfg, input)
divergence_from_spheroidal!(cfg::SHTConfig, output::AMDGPU.AnyROCArray,
                            input::AMDGPU.AnyROCArray) =
    _amdgpu_vector_diagonal!(output, cfg, input; inverse=false)
divergence_from_spheroidal!(::SHTnsKit.GPU, cfg::SHTConfig,
                            output::AMDGPU.AnyROCArray,
                            input::AMDGPU.AnyROCArray) =
    divergence_from_spheroidal!(cfg, output, input)
spheroidal_from_divergence(cfg::SHTConfig, input::AMDGPU.AnyROCArray) =
    spheroidal_from_divergence!(cfg, similar(input), input)
spheroidal_from_divergence(::SHTnsKit.GPU, cfg::SHTConfig,
                           input::AMDGPU.AnyROCArray) =
    spheroidal_from_divergence(cfg, input)
spheroidal_from_divergence!(cfg::SHTConfig, output::AMDGPU.AnyROCArray,
                            input::AMDGPU.AnyROCArray) =
    _amdgpu_vector_diagonal!(output, cfg, input; inverse=true)
spheroidal_from_divergence!(::SHTnsKit.GPU, cfg::SHTConfig,
                            output::AMDGPU.AnyROCArray,
                            input::AMDGPU.AnyROCArray) =
    spheroidal_from_divergence!(cfg, output, input)
vorticity_from_toroidal(cfg::SHTConfig, input::AMDGPU.AnyROCArray) =
    vorticity_from_toroidal!(cfg, similar(input), input)
vorticity_from_toroidal(::SHTnsKit.GPU, cfg::SHTConfig,
                        input::AMDGPU.AnyROCArray) =
    vorticity_from_toroidal(cfg, input)
vorticity_from_toroidal!(cfg::SHTConfig, output::AMDGPU.AnyROCArray,
                         input::AMDGPU.AnyROCArray) =
    _amdgpu_vector_diagonal!(output, cfg, input; inverse=false)
vorticity_from_toroidal!(::SHTnsKit.GPU, cfg::SHTConfig,
                         output::AMDGPU.AnyROCArray,
                         input::AMDGPU.AnyROCArray) =
    vorticity_from_toroidal!(cfg, output, input)
toroidal_from_vorticity(cfg::SHTConfig, input::AMDGPU.AnyROCArray) =
    toroidal_from_vorticity!(cfg, similar(input), input)
toroidal_from_vorticity(::SHTnsKit.GPU, cfg::SHTConfig,
                        input::AMDGPU.AnyROCArray) =
    toroidal_from_vorticity(cfg, input)
toroidal_from_vorticity!(cfg::SHTConfig, output::AMDGPU.AnyROCArray,
                         input::AMDGPU.AnyROCArray) =
    _amdgpu_vector_diagonal!(output, cfg, input; inverse=true)
toroidal_from_vorticity!(::SHTnsKit.GPU, cfg::SHTConfig,
                         output::AMDGPU.AnyROCArray,
                         input::AMDGPU.AnyROCArray) =
    toroidal_from_vorticity!(cfg, output, input)

@inline function _amdgpu_lcap(cfg::SHTConfig, ltr::Integer)
    return SHTnsKit._validate_degree_limit(cfg, ltr)
end

function _amdgpu_pack_lm(cfg::SHTConfig, dense::AMDGPU.AnyROCArray, lcap::Int)
    packed = AMDGPU.zeros(eltype(dense), cfg.nlm)
    kernel! = real_pack_kernel!(ROCBackend())
    kernel!(packed, dense, cfg.lmax, cfg.mmax, cfg.mres, lcap;
            ndrange=(cfg.lmax + 1, cfg.mmax ÷ cfg.mres + 1))
    AMDGPU.synchronize()
    return packed
end


function _amdgpu_unpack_lm(cfg::SHTConfig, packed::AMDGPU.AnyROCArray, lcap::Int)
    length(packed) == cfg.nlm || throw(DimensionMismatch(
        "Qlm must have length $(cfg.nlm)",
    ))
    dense = AMDGPU.zeros(eltype(packed), cfg.lmax + 1, cfg.mmax + 1)
    kernel! = real_unpack_kernel!(ROCBackend())
    kernel!(dense, packed, cfg.lmax, cfg.mmax, cfg.mres, lcap;
            ndrange=size(dense))
    AMDGPU.synchronize()
    return dense
end

function analysis_packed(::SHTnsKit.GPU, cfg::SHTConfig,
                         field::AMDGPU.AnyROCArray{T,1}) where {T<:Real}
    _require_amdgpu(:analysis_packed)
    length(field) == cfg.nspat || throw(DimensionMismatch(
        "field must have length $(cfg.nspat)",
    ))
    return _amdgpu_pack_lm(
        cfg, _amdgpu_scalar_analysis(cfg, reshape(field, cfg.nlat, cfg.nlon)), cfg.lmax,
    )
end
analysis_packed(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,1}) where {T<:Real} =
    analysis_packed(SHTnsKit.GPU(), cfg, field)

function synthesis_packed(::SHTnsKit.GPU, cfg::SHTConfig,
                          coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex}
    _require_amdgpu(:synthesis_packed)
    return vec(_amdgpu_scalar_synthesis(
        cfg, _amdgpu_unpack_lm(cfg, coefficients, cfg.lmax); real_output=true,
    ))
end
synthesis_packed(cfg::SHTConfig,
                 coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex} =
    synthesis_packed(SHTnsKit.GPU(), cfg, coefficients)

function analysis_packed_l(::SHTnsKit.GPU, cfg::SHTConfig,
                           field::AMDGPU.AnyROCArray{T,1}, ltr::Integer) where {T<:Real}
    lcap = _amdgpu_lcap(cfg, ltr)
    length(field) == cfg.nspat || throw(DimensionMismatch(
        "field must have length $(cfg.nspat)",
    ))
    return _amdgpu_pack_lm(
        cfg, _amdgpu_scalar_analysis(
            cfg, reshape(field, cfg.nlat, cfg.nlon); lcap,
        ), lcap,
    )
end
analysis_packed_l(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,1},
                  ltr::Integer) where {T<:Real} =
    analysis_packed_l(SHTnsKit.GPU(), cfg, field, ltr)

function synthesis_packed_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            coefficients::AMDGPU.AnyROCArray{T,1},
                            ltr::Integer) where {T<:Complex}
    lcap = _amdgpu_lcap(cfg, ltr)
    return vec(_amdgpu_scalar_synthesis(
        cfg, _amdgpu_unpack_lm(cfg, coefficients, lcap);
        real_output=true, lcap,
    ))
end
synthesis_packed_l(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    synthesis_packed_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _amdgpu_mode_analysis(cfg::SHTConfig, physical_m::Int,
                               mode::AMDGPU.AnyROCArray, lcap::Int, scale)
    RT = typeof(float(real(zero(eltype(mode)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    canonical = AMDGPU.zeros(CT, lcap - physical_m + 1)
    kernel! = mode_analysis_kernel!(ROCBackend())
    kernel!(canonical, mode, tables.Plm, tables.weights, RT(scale), physical_m, lcap;
            ndrange=length(canonical))
    configured = canonical ./ @view(tables.scales[(physical_m + 1):(lcap + 1), physical_m + 1])
    AMDGPU.synchronize()
    return configured
end

function _amdgpu_mode_synthesis(cfg::SHTConfig, physical_m::Int,
                                coefficients::AMDGPU.AnyROCArray, lcap::Int, scale)
    RT = typeof(float(real(zero(eltype(coefficients)))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    canonical = CT.(coefficients) .*
        @view(tables.scales[(physical_m + 1):(lcap + 1), physical_m + 1])
    mode = AMDGPU.zeros(CT, cfg.nlat)
    kernel! = mode_synthesis_kernel!(ROCBackend())
    kernel!(mode, canonical, tables.Plm, RT(scale), physical_m, lcap;
            ndrange=cfg.nlat)
    AMDGPU.synchronize()
    return mode
end

function analysis_axisym(::SHTnsKit.GPU, cfg::SHTConfig,
                         field::AMDGPU.AnyROCArray{T,1}) where {T<:Real}
    length(field) == cfg.nlat || throw(DimensionMismatch(
        "field must have length nlat=$(cfg.nlat)",
    ))
    return _amdgpu_mode_analysis(cfg, 0, field, cfg.lmax, SHTnsKit._analysis_phi_scale(cfg) * cfg.nlon)
end
analysis_axisym(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,1}) where {T<:Real} =
    analysis_axisym(SHTnsKit.GPU(), cfg, field)

function synthesis_axisym(::SHTnsKit.GPU, cfg::SHTConfig,
                          coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex}
    length(coefficients) == cfg.lmax + 1 || throw(DimensionMismatch(
        "coefficients must have length lmax+1=$(cfg.lmax + 1)",
    ))
    return real.(_amdgpu_mode_synthesis(cfg, 0, coefficients, cfg.lmax, 1))
end
synthesis_axisym(cfg::SHTConfig,
                 coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex} =
    synthesis_axisym(SHTnsKit.GPU(), cfg, coefficients)

function analysis_axisym_l(::SHTnsKit.GPU, cfg::SHTConfig,
                           field::AMDGPU.AnyROCArray{T,1}, ltr::Integer) where {T<:Real}
    lcap = _amdgpu_lcap(cfg, ltr)
    length(field) == cfg.nlat || throw(DimensionMismatch(
        "field must have length nlat=$(cfg.nlat)",
    ))
    return _amdgpu_mode_analysis(cfg, 0, field, lcap, SHTnsKit._analysis_phi_scale(cfg) * cfg.nlon)
end
analysis_axisym_l(cfg::SHTConfig, field::AMDGPU.AnyROCArray{T,1},
                  ltr::Integer) where {T<:Real} =
    analysis_axisym_l(SHTnsKit.GPU(), cfg, field, ltr)

function synthesis_axisym_l(::SHTnsKit.GPU, cfg::SHTConfig,
                            coefficients::AMDGPU.AnyROCArray{T,1},
                            ltr::Integer) where {T<:Complex}
    lcap = _amdgpu_lcap(cfg, ltr)
    length(coefficients) >= lcap + 1 || throw(DimensionMismatch(
        "coefficients must contain degrees 0:ltr",
    ))
    return real.(_amdgpu_mode_synthesis(
        cfg, 0, @view(coefficients[1:(lcap + 1)]), lcap, 1,
    ))
end
synthesis_axisym_l(cfg::SHTConfig, coefficients::AMDGPU.AnyROCArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    synthesis_axisym_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _amdgpu_fixed_order(cfg::SHTConfig, im::Int, ltr::Integer)
    im >= 0 || throw(ArgumentError("im must be >= 0"))
    im <= cfg.mmax ÷ cfg.mres || throw(ArgumentError(
        "im must be <= mmax/mres=$(cfg.mmax ÷ cfg.mres)",
    ))
    lcap = _amdgpu_lcap(cfg, ltr)
    physical_m = im * cfg.mres
    lcap >= physical_m || throw(ArgumentError(
        "ltr must be >= im*mres=$(physical_m)",
    ))
    return physical_m, lcap
end

function analysis_packed_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Int,
                            mode::AMDGPU.AnyROCArray{T,1}, ltr::Integer) where {T<:Complex}
    physical_m, lcap = _amdgpu_fixed_order(cfg, im, ltr)
    length(mode) == cfg.nlat || throw(DimensionMismatch(
        "mode must have length nlat=$(cfg.nlat)",
    ))
    return _amdgpu_mode_analysis(cfg, physical_m, mode, lcap, SHTnsKit._analysis_phi_scale(cfg))
end
analysis_packed_ml(cfg::SHTConfig, im::Int, mode::AMDGPU.AnyROCArray{T,1},
                   ltr::Integer) where {T<:Complex} =
    analysis_packed_ml(SHTnsKit.GPU(), cfg, im, mode, ltr)

function synthesis_packed_ml(::SHTnsKit.GPU, cfg::SHTConfig, im::Int,
                             coefficients::AMDGPU.AnyROCArray{T,1},
                             ltr::Integer) where {T<:Complex}
    physical_m, lcap = _amdgpu_fixed_order(cfg, im, ltr)
    length(coefficients) == lcap - physical_m + 1 || throw(DimensionMismatch(
        "coefficients have the wrong fixed-order length",
    ))
    return _amdgpu_mode_synthesis(
        cfg, physical_m, coefficients, lcap, SHTnsKit.phi_inv_scale(cfg),
    )
end
synthesis_packed_ml(cfg::SHTConfig, im::Int,
                    coefficients::AMDGPU.AnyROCArray{T,1},
                    ltr::Integer) where {T<:Complex} =
    synthesis_packed_ml(SHTnsKit.GPU(), cfg, im, coefficients, ltr)

function _amdgpu_analysis_packed_cplx(cfg::SHTConfig,
                                      field::AMDGPU.AnyROCArray{T,2},
                                      lcap::Int) where {T<:Complex}
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    size(field) == (cfg.nlat, cfg.nlon) || throw(DimensionMismatch(
        "field must have size ($(cfg.nlat), $(cfg.nlon))",
    ))
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    fourier = CT.(field)
    FFTW.fft!(fourier, 2)
    packed = AMDGPU.zeros(CT, SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1))
    mcap = min(cfg.mmax, lcap)
    kernel! = complex_packed_analysis_kernel!(ROCBackend())
    kernel!(packed, fourier, tables.Plm, tables.weights, tables.scales,
            RT(SHTnsKit._analysis_phi_scale(cfg)), cfg.nlon, lcap, cfg.mmax, mcap;
            ndrange=(lcap + 1, 2mcap + 1))
    AMDGPU.synchronize()
    return packed
end
function analysis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                              field::AMDGPU.AnyROCArray{T,2}) where {T<:Complex}
    return _amdgpu_analysis_packed_cplx(cfg, field, cfg.lmax)
end
analysis_packed_cplx(cfg::SHTConfig,
                     field::AMDGPU.AnyROCArray{T,2}) where {T<:Complex} =
    analysis_packed_cplx(SHTnsKit.GPU(), cfg, field)

function analysis_packed_cplx_l(::SHTnsKit.GPU, cfg::SHTConfig,
                                field::AMDGPU.AnyROCArray{T,2},
                                ltr::Integer) where {T<:Complex}
    return _amdgpu_analysis_packed_cplx(
        cfg, field, _amdgpu_lcap(cfg, ltr),
    )
end
analysis_packed_cplx_l(cfg::SHTConfig,
                       field::AMDGPU.AnyROCArray{T,2},
                       ltr::Integer) where {T<:Complex} =
    analysis_packed_cplx_l(SHTnsKit.GPU(), cfg, field, ltr)

function _amdgpu_synthesis_packed_cplx(cfg::SHTConfig,
                                       coefficients::AMDGPU.AnyROCArray{T,1},
                                       lcap::Int) where {T<:Complex}
    cfg.mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    expected = SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(coefficients) == expected || throw(DimensionMismatch(
        "coefficients must have length $expected",
    ))
    RT = typeof(float(real(zero(T))))
    CT = Complex{RT}
    tables = _amdgpu_scalar_tables(cfg, RT)
    fourier = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon)
    mcap = min(cfg.mmax, lcap)
    kernel! = complex_packed_synthesis_kernel!(ROCBackend())
    kernel!(fourier, coefficients, tables.Plm, tables.scales,
            RT(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, lcap, cfg.mmax, mcap;
            ndrange=(cfg.nlat, 2mcap + 1))
    AMDGPU.synchronize()
    FFTW.ifft!(fourier, 2)
    return fourier
end
function synthesis_packed_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                               coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex}
    return _amdgpu_synthesis_packed_cplx(cfg, coefficients, cfg.lmax)
end
synthesis_packed_cplx(cfg::SHTConfig,
                      coefficients::AMDGPU.AnyROCArray{T,1}) where {T<:Complex} =
    synthesis_packed_cplx(SHTnsKit.GPU(), cfg, coefficients)

function synthesis_packed_cplx_l(::SHTnsKit.GPU, cfg::SHTConfig,
                                 coefficients::AMDGPU.AnyROCArray{T,1},
                                 ltr::Integer) where {T<:Complex}
    return _amdgpu_synthesis_packed_cplx(
        cfg, coefficients, _amdgpu_lcap(cfg, ltr),
    )
end
synthesis_packed_cplx_l(cfg::SHTConfig,
                        coefficients::AMDGPU.AnyROCArray{T,1},
                        ltr::Integer) where {T<:Complex} =
    synthesis_packed_cplx_l(SHTnsKit.GPU(), cfg, coefficients, ltr)

function _amdgpu_batch_analysis(cfg::SHTConfig, fields::AMDGPU.AnyROCArray;
                                use_rfft::Bool=false, fft_batch=nothing)
    fft_batch === nothing || throw(ArgumentError(
        "AMDGPU scalar batches do not accept caller-provided fft_batch scratch",
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
    tables = _amdgpu_scalar_tables(cfg, RT)
    fourier = CT.(fields)
    FFTW.fft!(fourier, 2)
    canonical = AMDGPU.zeros(CT, cfg.lmax + 1, cfg.mmax + 1, size(fields, 3))
    kernel! = scalar_batch_analysis_kernel!(ROCBackend())
    kernel!(canonical, fourier, tables.Plm, tables.weights, RT(SHTnsKit._analysis_phi_scale(cfg)),
            cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(canonical))
    configured = canonical ./ reshape(
        tables.scales, cfg.lmax + 1, cfg.mmax + 1, 1,
    )
    AMDGPU.synchronize()
    return configured
end

function _amdgpu_batch_analysis_direct!(cfg::SHTConfig,
                                        output::AMDGPU.AnyROCArray{<:Complex,3},
                                        fields::AMDGPU.AnyROCArray{<:Real,3};
                                        use_rfft::Bool=false, fft_batch=nothing)
    _require_amdgpu(:analysis_batch!)
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
    scratch = _amdgpu_batch_scratch(
        cfg, fft_batch, CT, nfields, use_rfft, output, fields,
    )
    tables = _amdgpu_scalar_tables(cfg, RT)
    return _with_amdgpu_workspace(cfg, cfg, RT, nfields, use_rfft) do workspace
        fourier = scratch === nothing ? workspace.fourier : scratch
        if use_rfft
            copyto!(workspace.real_buffer, fields)
            mul!(fourier, workspace.forward, workspace.real_buffer)
        else
            copyto!(fourier, fields)
            workspace.forward * fourier
        end
        backend = ROCBackend()
        scalar_batch_analysis_kernel!(backend)(
            output, fourier, tables.Plm, tables.weights, RT(SHTnsKit._analysis_phi_scale(cfg)),
            cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(output),
        )
        coefficient_batch_conversion_kernel!(backend)(
            output, output, tables.scales, cfg.lmax, cfg.mmax, false;
            ndrange=size(output),
        )
        AMDGPU.synchronize()
        output
    end
end

analysis_batch(::SHTnsKit.GPU, cfg::SHTConfig,
               fields::AMDGPU.AnyROCArray{T,3}; use_rfft::Bool=false) where {T<:Real} =
    _amdgpu_batch_analysis(cfg, fields; use_rfft)
analysis_batch(cfg::SHTConfig, fields::AMDGPU.AnyROCArray{T,3}; kwargs...) where {T<:Real} =
    analysis_batch(SHTnsKit.GPU(), cfg, fields; kwargs...)

function analysis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                         output::AMDGPU.AnyROCArray{T,3},
                         fields::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Complex,R<:Real}
    return _amdgpu_batch_analysis_direct!(cfg, output, fields; kwargs...)
end
analysis_batch!(cfg::SHTConfig, output::AMDGPU.AnyROCArray{T,3},
                fields::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Complex,R<:Real} =
    analysis_batch!(SHTnsKit.GPU(), cfg, output, fields; kwargs...)

function _amdgpu_batch_synthesis(cfg::SHTConfig,
                                 coefficients::AMDGPU.AnyROCArray;
                                 real_output::Bool=true, use_rfft::Bool=false,
                                 fft_batch=nothing)
    fft_batch === nothing || throw(ArgumentError(
        "AMDGPU scalar batches do not accept caller-provided fft_batch scratch",
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
    tables = _amdgpu_scalar_tables(cfg, RT)
    canonical = CT.(coefficients) .* reshape(
        tables.scales, cfg.lmax + 1, cfg.mmax + 1, 1,
    )
    fourier = AMDGPU.zeros(CT, cfg.nlat, cfg.nlon, size(coefficients, 3))
    kernel! = scalar_batch_synthesis_kernel!(ROCBackend())
    kernel!(fourier, canonical, tables.Plm, RT(SHTnsKit.phi_inv_scale(cfg)),
            cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres, real_output;
            ndrange=(cfg.nlat, cfg.mmax + 1, size(coefficients, 3)))
    AMDGPU.synchronize()
    FFTW.ifft!(fourier, 2)
    return real_output ? real.(fourier) : fourier
end

function _amdgpu_batch_synthesis_direct!(cfg::SHTConfig,
                                         output::AMDGPU.AnyROCArray{<:Number,3},
                                         coefficients::AMDGPU.AnyROCArray{<:Complex,3};
                                         real_output::Bool=true,
                                         use_rfft::Bool=false, fft_batch=nothing)
    _require_amdgpu(:synthesis_batch!)
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
    scratch = _amdgpu_batch_scratch(
        cfg, fft_batch, CT, nfields, use_rfft, output, coefficients,
    )
    tables = _amdgpu_scalar_tables(cfg, RT)
    return _with_amdgpu_workspace(cfg, cfg, RT, nfields, use_rfft) do workspace
        fourier = scratch === nothing ? workspace.fourier : scratch
        backend = ROCBackend()
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
        AMDGPU.synchronize()
        if use_rfft
            mul!(workspace.real_buffer, workspace.inverse, fourier)
            copyto!(output, workspace.real_buffer)
        else
            workspace.inverse * fourier
            real_output ? (output .= real.(fourier)) : copyto!(output, fourier)
        end
        AMDGPU.synchronize()
        output
    end
end

synthesis_batch(::SHTnsKit.GPU, cfg::SHTConfig,
                coefficients::AMDGPU.AnyROCArray{T,3};
                real_output::Bool=true, use_rfft::Bool=false) where {T<:Complex} =
    _amdgpu_batch_synthesis(cfg, coefficients; real_output, use_rfft)
synthesis_batch(cfg::SHTConfig,
                coefficients::AMDGPU.AnyROCArray{T,3}; kwargs...) where {T<:Complex} =
    synthesis_batch(SHTnsKit.GPU(), cfg, coefficients; kwargs...)
synthesis_batch_cplx(cfg::SHTConfig,
                     coefficients::AMDGPU.AnyROCArray{T,3}) where {T<:Complex} =
    _amdgpu_batch_synthesis(cfg, coefficients; real_output=false)
synthesis_batch_cplx(::SHTnsKit.GPU, cfg::SHTConfig,
                     coefficients::AMDGPU.AnyROCArray{T,3}) where {T<:Complex} =
    _amdgpu_batch_synthesis(cfg, coefficients; real_output=false)

function synthesis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                          output::AMDGPU.AnyROCArray{T,3},
                          coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T,R<:Complex}
    return _amdgpu_batch_synthesis_direct!(cfg, output, coefficients; kwargs...)
end
function synthesis_batch!(::SHTnsKit.GPU, cfg::SHTConfig,
                          output::AMDGPU.AnyROCArray{T,3},
                          coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Real,R<:Complex}
    return _amdgpu_batch_synthesis_direct!(cfg, output, coefficients; kwargs...)
end
synthesis_batch!(cfg::SHTConfig, output::AMDGPU.AnyROCArray{T,3},
                 coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)

function analysis!(::SHTnsKit.GPU, plan::SHTPlan,
                   output::AMDGPU.AnyROCArray{T,2},
                   field::AMDGPU.AnyROCArray{R,2}) where {T<:Complex,R<:Number}
    return _amdgpu_scalar_analysis_direct!(
        plan, plan.cfg, output, field; use_rfft=plan.use_rfft,
    )
end
analysis!(plan::SHTPlan, output::AMDGPU.AnyROCArray{T,2},
          field::AMDGPU.AnyROCArray{R,2}) where {T<:Complex,R<:Number} =
    analysis!(SHTnsKit.GPU(), plan, output, field)

function synthesis!(::SHTnsKit.GPU, plan::SHTPlan,
                    output::AMDGPU.AnyROCArray{T,2},
                    coefficients::AMDGPU.AnyROCArray{R,2};
                    real_output::Bool=true) where {T<:Number,R<:Complex}
    return _amdgpu_scalar_synthesis_direct!(
        plan, plan.cfg, output, coefficients;
        real_output, use_rfft=plan.use_rfft,
    )
end
synthesis!(plan::SHTPlan, output::AMDGPU.AnyROCArray{T,2},
           coefficients::AMDGPU.AnyROCArray{R,2};
           real_output::Bool=true) where {T<:Number,R<:Complex} =
    synthesis!(SHTnsKit.GPU(), plan, output, coefficients; real_output)
synthesis_batch!(cfg::SHTConfig, output::AMDGPU.AnyROCArray{T,3},
                 coefficients::AMDGPU.AnyROCArray{R,3}; kwargs...) where {T<:Real,R<:Complex} =
    synthesis_batch!(SHTnsKit.GPU(), cfg, output, coefficients; kwargs...)

end # module SHTnsKitAMDGPUExt
