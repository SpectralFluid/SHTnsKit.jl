module SHTnsKitMetalExt

import SHTnsKit as SHT
using SHTnsKit: GPU, SHTConfig, SHTPlan
import Metal
using Metal: MtlArray, WrappedMtlArray, WrappedMtlMatrix, WrappedMtlVector, MetalBackend
using KernelAbstractions: @kernel, @index
using LinearAlgebra: mul!

include("GPUCommon.jl")
using .GPUCommon

mutable struct MetalAdapter end
const ADAPTER = MetalAdapter()
const SCALAR_TABLES = ScalarTableCache()
const VECTOR_TABLES = ScalarTableCache()
const PLAN_BUFFERS = ScalarWorkspaceCache()

function __init__()
    SHT._register_gpu_adapter!(:metal, ADAPTER)
    SHT._enable_gpu_loops!(launch_sht_loop!)
end

SHT._gpu_adapter_functional(::MetalAdapter) = Metal.functional()
SHT._gpu_adapter_matches(::MetalAdapter, ::WrappedMtlArray) = true
# A wrapper of a wrapper (a view of a PermutedDimsArray, say) is Metal storage too.
SHT._gpu_adapter_matches(adapter::MetalAdapter, a::AbstractArray) =
    (p = parent(a)) !== a && SHT._gpu_adapter_matches(adapter, p)
SHT.on_device(::WrappedMtlArray) = GPU()

# Broadcasting gives nested wrappers the CPU's style, so inputs that reach the
# adapter hooks in that form are first copied into a dense device array.
_device_array(a::WrappedMtlArray) = a
_device_array(a::AbstractArray) = (dense = similar(_storage_root(a), eltype(a), size(a))) .= a
_storage_root(a::MtlArray) = a
_storage_root(a::AbstractArray) = _storage_root(parent(a))
SHT._gpu_adapter_dense(::MetalAdapter, value::AbstractArray) = _device_array(value)

function SHT._gpu_adapter_adapt(::MetalAdapter, value)
    # MtlArray preserves precision. Metal.mtl would silently downcast doubles.
    eltype(value) in (Float64, ComplexF64) && throw(ArgumentError(
        "Metal cannot store $(eltype(value)); convert explicitly to Float32 or ComplexF32",
    ))
    return MtlArray(value)
end

function SHT._gpu_adapter_clear_cache!(::MetalAdapter)
    scalar_cache_clear!(SCALAR_TABLES)
    scalar_cache_clear!(VECTOR_TABLES)
    scalar_workspace_clear!(PLAN_BUFFERS)
    return nothing
end

# Apple GPUs have no Float64 arithmetic. Evaluate the latitude recurrence once
# from the original grid nodes and round each entry directly into shared Metal
# memory. Only immutable configuration data is touched by the CPU.
function _basis_tables(cfg, vector)
    dims = (cfg.nlat, cfg.lmax + 1, cfg.mmax + 1)
    shared_table() = MtlArray{Float32,3,Metal.SharedStorage}(undef, dims)
    tables = vector ? (dtheta=shared_table(), over_sin=shared_table()) : (Plm=shared_table(),)
    GC.@preserve tables begin
        host = map(a -> unsafe_wrap(Array, a), tables)
        p = zeros(cfg.lmax + 1)
        d = vector ? similar(p) : nothing
        p_over_s = vector ? similar(p) : nothing
        scratch = vector ? zeros(cfg.lmax + 2) : nothing
        for m in 0:cfg.mmax, i in eachindex(cfg.x)
            if vector
                SHT.Plm_norm_dPdtheta_over_sinth_row!(p, d, p_over_s, cfg.x[i], cfg.lmax, m, scratch)
                host.dtheta[i, :, m+1] .= d
                host.over_sin[i, :, m+1] .= p_over_s
            else
                SHT.Plm_norm_row!(p, cfg.x[i], cfg.lmax, m)
                host.Plm[i, :, m+1] .= p
            end
        end
    end
    return tables
end

function _tables(cfg, vector)
    cache = vector ? VECTOR_TABLES : SCALAR_TABLES
    signature = vector ? vector_config_signature(cfg) : scalar_config_signature(cfg)
    device = Metal.device().registryID
    existing = scalar_cache_lookup(cache, device, objectid(cfg), Float32, signature; owner=cfg)
    existing === nothing || return existing
    _, weights, scales, sint = scalar_host_tables(cfg, Float32)
    basis = _basis_tables(cfg, vector)
    tables = (; basis..., weights=MtlArray(weights), scales=MtlArray(scales), sint=MtlArray(sint))
    return scalar_cache_publish!(Metal.synchronize, cache, device, objectid(cfg), Float32,
                                 signature, tables; owner=cfg)
end

_spatial(cfg) = (cfg.nlat, cfg.nlon)
_spectral(cfg) = (cfg.lmax + 1, cfg.mmax + 1)
_shape(dims, a) = (dims..., size(a)[3:end]...)
_range(a) = (size(a, 1), size(a, 2), size(a, 3))

function _check_field(a, dims)
    (size(a, 1), size(a, 2)) == dims || throw(DimensionMismatch(
        "expected leading dimensions $dims, got $(size(a))",
    ))
    size(a, 3) > 0 || throw(ArgumentError("a transform batch must contain at least one field"))
    eltype(a) in (Float32, ComplexF32) || throw(ArgumentError(
        "Metal transforms require Float32 or ComplexF32, got $(eltype(a))",
    ))
end

# Kernels and MPSGraph FFTs run in submission order on the task's command queue,
# so the helpers below only submit work. Each public method waits once, before
# it returns device arrays or releases a cached workspace.
_completed(result) = (Metal.synchronize(); result)

# MPSGraph encodes the entire underlying buffer starting at byte zero.
_graph_buffer(a) = a isa MtlArray && iszero(a.offset)
_nbins(nlon, half) = half ? nlon ÷ 2 + 1 : nlon

# The buffer and byte range an array can touch: exact for dense arrays and the
# contiguous views Metal stores at an offset, the whole parent for other views.
# Under GPUArrays 11, Base.mightalias compares whole buffers, so disjoint parts
# of one buffer looked like the same memory.
function _byte_span(a::MtlArray)
    p = pointer(a)
    return p.buffer, Int(p.offset):(Int(p.offset) + sizeof(a) - 1)
end
_byte_span(a::AbstractArray) = (p = parent(a)) === a ? nothing : _byte_span(p)

function _overlaps(a, b)
    # Views of one parent whose index ranges are disjoint, such as interleaved
    # columns or separate row blocks, cannot overlap though their spans do.
    a isa SubArray && b isa SubArray && parent(a) === parent(b) &&
        !Base.mightalias(a, b) && return false
    sa, sb = _byte_span(a), _byte_span(b)
    (sa === nothing || sb === nothing) && return Base.mightalias(a, b)
    return sa[1] == sb[1] && !isdisjoint(sa[2], sb[2])
end

function _fourier_buffer(cfg, source, half, scratch; reuse=false)
    dims = _shape((cfg.nlat, _nbins(cfg.nlon, half)), source)
    scratch === nothing && return similar(source, ComplexF32, dims)
    _graph_buffer(scratch) && eltype(scratch) === ComplexF32 || throw(ArgumentError(
        "FFT scratch must be a dense ComplexF32 MtlArray starting at offset zero",
    ))
    size(scratch) == dims || throw(DimensionMismatch("FFT scratch must have size $dims"))
    # A complex input may serve as its own forward FFT scratch, as on the CPU.
    reuse && scratch === source && return scratch
    _overlaps(scratch, source) && throw(ArgumentError("FFT scratch overlaps the input"))
    return scratch
end

function _forward(cfg, field, half, scratch=nothing)
    half && eltype(field) !== Float32 && throw(ArgumentError("use_rfft=true requires real input"))
    bins = _fourier_buffer(cfg, field, half, scratch; reuse=!half)
    if half && cfg.nlon > 1
        dense = _graph_buffer(field) ? field : Float32.(field)
        mul!(bins, Metal.plan_rfft(dense, (2,)), dense)
    else
        bins === field || (bins .= field)
        cfg.nlon > 1 && Metal.fft!(bins, (2,))
    end
    return bins
end

function _inverse(bins, nlon, real_output, half, out=nothing)
    if half && iseven(nlon)
        if _graph_buffer(out)
            return mul!(out, Metal.plan_irfft(bins, nlon, (2,)), bins)
        end
        values = Metal.irfft(bins, nlon, (2,))
        out === nothing && return values
        return out .= values
    end
    if half && nlon > 1
        # MPSGraph infers an even real inverse length from a half spectrum.
        # Expand the positive frequencies to recover an odd-length field.
        width = size(bins, 2)
        full = similar(bins, _shape((size(bins, 1), nlon), bins))
        selectdim(full, 2, 1:width) .= bins
        selectdim(full, 2, width+1:nlon) .= conj.(selectdim(bins, 2, width:-1:2))
        bins = full
    end
    nlon > 1 && Metal.ifft!(bins, (2,))
    out === nothing && return real_output ? real.(bins) : bins
    return real_output ? (out .= real.(bins)) : (out .= bins)
end

function _scalar_analysis!(out, cfg, field; use_rfft=false, fft_scratch=nothing, lcap=cfg.lmax)
    _check_field(field, _spatial(cfg))
    size(out) == _shape(_spectral(cfg), field) || throw(DimensionMismatch("incorrect spectral output shape"))
    eltype(out) === ComplexF32 || throw(ArgumentError("spectral output must use ComplexF32"))
    fft_scratch !== nothing && _overlaps(fft_scratch, out) &&
        throw(ArgumentError("FFT scratch overlaps the spectral output"))
    bins = _forward(cfg, field, use_rfft, fft_scratch)
    tables = _tables(cfg, false)
    # The full grid is one contiguous band of orders starting at m=0. The
    # shared band kernel writes configured coefficients directly into out.
    GPUCommon.distributed_scalar_analysis_kernel!(MetalBackend())(
        out, bins, tables.Plm, tables.weights, tables.scales, Float32(SHT._analysis_phi_scale(cfg)),
        0, cfg.lmax, cfg.mmax, cfg.mres, lcap; ndrange=_range(out),
    )
    return out
end

function _scalar_analysis(cfg, field; kwargs...)
    out = similar(field, ComplexF32, _shape(_spectral(cfg), field))
    return _scalar_analysis!(out, cfg, field; kwargs...)
end

function _check_output(out, cfg, coeff, real_output, use_rfft)
    size(out) == _shape(_spatial(cfg), coeff) || throw(DimensionMismatch("incorrect spatial output shape"))
    _check_field(out, _spatial(cfg))
    !real_output && eltype(out) !== ComplexF32 && throw(ArgumentError("complex output requires ComplexF32 storage"))
    use_rfft && eltype(out) !== Float32 && throw(ArgumentError("use_rfft=true requires Float32 output storage"))
end

@kernel function _mirror_orders!(bins, nlon)
    i, m, batch = @index(Global, NTuple)
    @inbounds bins[i, nlon - m + 1, batch] = conj(bins[i, m + 1, batch])
end

function _scalar_synthesis(cfg, coeff; real_output=true, use_rfft=false,
                           fft_scratch=nothing, out=nothing, lcap=cfg.lmax,
                           half=use_rfft && (iseven(cfg.nlon) || fft_scratch !== nothing))
    _check_field(coeff, _spectral(cfg))
    use_rfft && !real_output && throw(ArgumentError("use_rfft=true requires real_output=true"))
    out === nothing || _check_output(out, cfg, coeff, real_output, use_rfft)
    fft_scratch !== nothing && out !== nothing && _overlaps(fft_scratch, out) &&
        throw(ArgumentError("FFT scratch overlaps the spatial output"))
    # An odd-length Metal real inverse needs a full complex grid. Write into
    # that grid directly unless the caller supplied half-width FFT scratch.
    bins = _fourier_buffer(cfg, coeff, half, fft_scratch)
    tables = _tables(cfg, false)
    mcap = min(lcap, cfg.mmax)
    GPUCommon.distributed_scalar_synthesis_kernel!(MetalBackend())(
        bins, coeff, tables.Plm, tables.scales, Float32(SHT.phi_inv_scale(cfg)),
        0, lcap, mcap, cfg.mres; ndrange=_range(bins),
    )
    if real_output && !half && mcap > 0
        _mirror_orders!(MetalBackend())(bins, cfg.nlon; ndrange=(cfg.nlat, mcap, size(coeff, 3)))
    end
    return _inverse(bins, cfg.nlon, real_output, half, out)
end

SHT._gpu_adapter_analysis(::MetalAdapter, cfg::SHTConfig, f; use_rfft::Bool=false, fft_scratch=nothing) =
    _completed(_scalar_analysis(cfg, _device_array(f); use_rfft, fft_scratch))
SHT._gpu_adapter_synthesis(::MetalAdapter, cfg::SHTConfig, a; real_output::Bool=true,
                           use_rfft::Bool=false, fft_scratch=nothing) =
    _completed(_scalar_synthesis(cfg, _device_array(a); real_output, use_rfft, fft_scratch))
SHT.analysis(cfg::SHTConfig, f::WrappedMtlMatrix; kwargs...) =
    SHT.analysis(GPU(), cfg, f; kwargs...)
SHT.synthesis(cfg::SHTConfig, a::WrappedMtlMatrix; kwargs...) =
    SHT.synthesis(GPU(), cfg, a; kwargs...)
SHT.synthesis_cplx(cfg::SHTConfig, a::WrappedMtlMatrix) =
    SHT.synthesis_cplx(GPU(), cfg, a)

function _check_layout(cfg, dense, packed)
    size(dense) == _spectral(cfg) || throw(DimensionMismatch("incorrect dense coefficient shape"))
    length(packed) == cfg.nlm || throw(DimensionMismatch("expected $(cfg.nlm) packed coefficients"))
end

function _pack!(packed, cfg, dense, lcap)
    _check_layout(cfg, dense, packed)
    source = _overlaps(dense, packed) ? copy(dense) : dense
    real_pack_kernel!(MetalBackend())(packed, source, cfg.lmax, cfg.mmax, cfg.mres, lcap;
                                      ndrange=(cfg.lmax + 1, cfg.mmax ÷ cfg.mres + 1))
    return packed
end

function _pack(cfg, dense, lcap)
    packed = similar(dense, eltype(dense), (cfg.nlm,))
    return _pack!(packed, cfg, dense, lcap)
end

function _unpack(cfg, packed, lcap)
    length(packed) == cfg.nlm || throw(DimensionMismatch("expected $(cfg.nlm) packed coefficients"))
    dense = similar(packed, eltype(packed), _spectral(cfg))
    real_unpack_kernel!(MetalBackend())(dense, packed, cfg.lmax, cfg.mmax, cfg.mres, lcap;
                                        ndrange=size(dense))
    return dense
end

SHT.pack_lm(cfg::SHTConfig, a::WrappedMtlMatrix) = _completed(_pack(cfg, a, cfg.lmax))
SHT.unpack_lm(cfg::SHTConfig, a::WrappedMtlVector) = _completed(_unpack(cfg, a, cfg.lmax))
SHT.pack_lm!(packed::WrappedMtlVector, cfg::SHTConfig, a::WrappedMtlMatrix) =
    _completed(_pack!(packed, cfg, a, cfg.lmax))

# The public unpack_lm! preserves entries with no packed counterpart. The
# allocating transform path above initializes those entries to zero.
@kernel function _scatter_lm!(dense, packed, lmax, mres)
    lidx, imidx = @index(Global, NTuple)
    l, m = lidx - 1, (imidx - 1) * mres
    if l >= m
        k = GPUCommon._real_packed_device_index(l, m, lmax, mres)
        @inbounds dense[lidx, m + 1] = packed[k + 1]
    end
end

function SHT.unpack_lm!(dense::WrappedMtlMatrix, cfg::SHTConfig, packed::WrappedMtlVector)
    _check_layout(cfg, dense, packed)
    source = _overlaps(dense, packed) ? copy(packed) : packed
    _scatter_lm!(MetalBackend())(dense, source, cfg.lmax, cfg.mres;
                                 ndrange=(cfg.lmax + 1, cfg.mmax ÷ cfg.mres + 1))
    Metal.synchronize()
    return dense
end

function SHT.analysis_packed_l(::GPU, cfg::SHTConfig, field::WrappedMtlVector{<:Real}, ltr::Integer)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    length(field) == cfg.nspat || throw(DimensionMismatch("expected $(cfg.nspat) spatial samples"))
    grid = reshape(field, _spatial(cfg))
    dense = _scalar_analysis(cfg, grid; lcap, use_rfft=_graph_buffer(grid))
    return _completed(_pack(cfg, dense, lcap))
end
function SHT.synthesis_packed_l(::GPU, cfg::SHTConfig, packed::WrappedMtlVector{<:Complex}, ltr::Integer)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    return _completed(vec(_scalar_synthesis(cfg, _unpack(cfg, packed, lcap); lcap, use_rfft=true)))
end
SHT.analysis_packed(::GPU, cfg::SHTConfig, f::WrappedMtlVector{<:Real}) =
    SHT.analysis_packed_l(GPU(), cfg, f, cfg.lmax)
SHT.synthesis_packed(::GPU, cfg::SHTConfig, a::WrappedMtlVector{<:Complex}) =
    SHT.synthesis_packed_l(GPU(), cfg, a, cfg.lmax)
SHT.analysis_packed(cfg::SHTConfig, f::WrappedMtlVector{<:Real}) = SHT.analysis_packed(GPU(), cfg, f)
SHT.synthesis_packed(cfg::SHTConfig, a::WrappedMtlVector{<:Complex}) = SHT.synthesis_packed(GPU(), cfg, a)
SHT.analysis_packed_l(cfg::SHTConfig, f::WrappedMtlVector{<:Real}, ltr::Integer) =
    SHT.analysis_packed_l(GPU(), cfg, f, ltr)
SHT.synthesis_packed_l(cfg::SHTConfig, a::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_packed_l(GPU(), cfg, a, ltr)

function _complex_layout(cfg)
    cfg.mres == 1 || throw(ArgumentError("LM_cplx requires mres=1"))
    return SHT.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
end

function SHT.analysis_packed_cplx_l(::GPU, cfg::SHTConfig, field::WrappedMtlMatrix{<:Complex}, ltr::Integer)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    mcap = min(lcap, cfg.mmax)
    _check_field(field, _spatial(cfg))
    packed = fill!(similar(field, ComplexF32, (_complex_layout(cfg),)), 0f0)
    bins = _forward(cfg, field, false)
    tables = _tables(cfg, false)
    complex_packed_analysis_kernel!(MetalBackend())(
        packed, bins, tables.Plm, tables.weights, tables.scales, Float32(SHT._analysis_phi_scale(cfg)),
        cfg.nlon, lcap, cfg.mmax, mcap; ndrange=(lcap + 1, 2mcap + 1),
    )
    Metal.synchronize()
    return packed
end
function SHT.synthesis_packed_cplx_l(::GPU, cfg::SHTConfig, packed::WrappedMtlVector{<:Complex}, ltr::Integer)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    mcap = min(lcap, cfg.mmax)
    length(packed) == _complex_layout(cfg) || throw(DimensionMismatch("incorrect LM_cplx coefficient count"))
    eltype(packed) === ComplexF32 || throw(ArgumentError("Metal LM_cplx coefficients must use ComplexF32"))
    bins = fill!(similar(packed, ComplexF32, _spatial(cfg)), 0f0)
    tables = _tables(cfg, false)
    complex_packed_synthesis_kernel!(MetalBackend())(
        bins, packed, tables.Plm, tables.scales, Float32(SHT.phi_inv_scale(cfg)),
        cfg.nlon, lcap, cfg.mmax, mcap; ndrange=(cfg.nlat, 2mcap + 1),
    )
    result = _inverse(bins, cfg.nlon, false, false)
    Metal.synchronize()
    return result
end
SHT.analysis_packed_cplx(::GPU, cfg::SHTConfig, f::WrappedMtlMatrix{<:Complex}) =
    SHT.analysis_packed_cplx_l(GPU(), cfg, f, cfg.lmax)
SHT.synthesis_packed_cplx(::GPU, cfg::SHTConfig, a::WrappedMtlVector{<:Complex}) =
    SHT.synthesis_packed_cplx_l(GPU(), cfg, a, cfg.lmax)
SHT.analysis_packed_cplx(cfg::SHTConfig, f::WrappedMtlMatrix{<:Complex}) = SHT.analysis_packed_cplx(GPU(), cfg, f)
SHT.synthesis_packed_cplx(cfg::SHTConfig, a::WrappedMtlVector{<:Complex}) = SHT.synthesis_packed_cplx(GPU(), cfg, a)
SHT.analysis_packed_cplx_l(cfg::SHTConfig, f::WrappedMtlMatrix{<:Complex}, ltr::Integer) =
    SHT.analysis_packed_cplx_l(GPU(), cfg, f, ltr)
SHT.synthesis_packed_cplx_l(cfg::SHTConfig, a::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_packed_cplx_l(GPU(), cfg, a, ltr)

# Mixed components promote as on the CPU. Metal's only mix is Float16 with
# Float32, so the Float16 component is widened before the kernels run.
function _promoted_pair(a, b)
    R = real(float(promote_type(eltype(a), eltype(b))))
    return SHT._with_precision(a, R), SHT._with_precision(b, R)
end

function _vector_analysis(cfg, vt, vp; use_rfft=false, buffers=(nothing, nothing), out=nothing,
                          lcap=cfg.lmax)
    vt, vp = _promoted_pair(vt, vp)
    _check_field(vt, _spatial(cfg))
    _check_field(vp, _spatial(cfg))
    size(vt) == size(vp) || throw(DimensionMismatch("vector components must have the same shape"))
    SHT._validate_robert_analysis(cfg; ltr=lcap)
    shape = _shape(_spectral(cfg), vt)
    if out === nothing
        s = similar(vt, ComplexF32, shape)
        t = similar(s)
    else
        s, t = out
        size(s) == size(t) == shape || throw(DimensionMismatch("incorrect vector spectrum shapes"))
        eltype(s) === eltype(t) === ComplexF32 || throw(ArgumentError("vector spectra must use ComplexF32"))
        _overlaps(s, t) && throw(ArgumentError("output spectra must be distinct"))
    end
    ft = _forward(cfg, vt, use_rfft, buffers[1])
    fp = _forward(cfg, vp, use_rfft, buffers[2])
    tables = _tables(cfg, true)
    mcap = min(lcap, cfg.mmax)
    # Below the full degree the kernel covers only l, m ≤ lcap.
    lcap < cfg.lmax && (fill!(s, 0f0); fill!(t, 0f0))
    vector_batch_analysis_kernel!(MetalBackend())(
        s, t, ft, fp, tables.dtheta, tables.over_sin, tables.weights, tables.scales,
        tables.sint, Float32(SHT._analysis_phi_scale(cfg)), lcap, mcap,
        cfg.mres, cfg.robert_form; ndrange=(lcap + 1, mcap + 1, size(s, 3)),
    )
    return s, t
end

function _vector_synthesis(cfg, s, t; real_output=true, use_rfft=false,
                           buffers=(nothing, nothing), out=(nothing, nothing),
                           half=use_rfft && iseven(cfg.nlon), lcap=cfg.lmax)
    s, t = _promoted_pair(s, t)
    _check_field(s, _spectral(cfg))
    _check_field(t, _spectral(cfg))
    size(s) == size(t) || throw(DimensionMismatch("vector spectra must have the same shape"))
    use_rfft && !real_output && throw(ArgumentError("use_rfft=true requires real_output=true"))
    if out[1] !== nothing
        _check_output(out[1], cfg, s, real_output, use_rfft)
        _check_output(out[2], cfg, t, real_output, use_rfft)
        _overlaps(out...) && throw(ArgumentError("vector outputs must be distinct"))
    end
    ft = fill!(_fourier_buffer(cfg, s, half, buffers[1]), 0f0)
    fp = fill!(_fourier_buffer(cfg, t, half, buffers[2]), 0f0)
    tables = _tables(cfg, true)
    mcap = min(lcap, cfg.mmax)
    vector_batch_synthesis_kernel!(MetalBackend())(
        ft, fp, s, t, tables.dtheta, tables.over_sin, tables.scales, tables.sint,
        Float32(SHT.phi_inv_scale(cfg)), cfg.nlon, lcap, mcap, cfg.mres,
        real_output && !half, cfg.robert_form; ndrange=(cfg.nlat, mcap + 1, size(s, 3)),
    )
    vt = _inverse(ft, cfg.nlon, real_output, half, out[1])
    vp = _inverse(fp, cfg.nlon, real_output, half, out[2])
    return vt, vp
end

SHT._gpu_adapter_analysis_sphtor(::MetalAdapter, cfg::SHTConfig, vt, vp; use_rfft::Bool=false) =
    _completed(_vector_analysis(cfg, _device_array(vt), _device_array(vp); use_rfft))
SHT._gpu_adapter_synthesis_sphtor(::MetalAdapter, cfg::SHTConfig, s, t;
                                  real_output::Bool=true, use_rfft::Bool=false) =
    _completed(_vector_synthesis(cfg, _device_array(s), _device_array(t); real_output, use_rfft))
SHT.analysis_sphtor(cfg::SHTConfig, vt::WrappedMtlMatrix, vp::WrappedMtlMatrix; kwargs...) =
    SHT.analysis_sphtor(GPU(), cfg, vt, vp; kwargs...)
SHT.synthesis_sphtor(cfg::SHTConfig, s::WrappedMtlMatrix, t::WrappedMtlMatrix; kwargs...) =
    SHT.synthesis_sphtor(GPU(), cfg, s, t; kwargs...)
SHT.synthesis_sphtor_cplx(cfg::SHTConfig, s::WrappedMtlMatrix, t::WrappedMtlMatrix) =
    SHT.synthesis_sphtor_cplx(GPU(), cfg, s, t)
SHT.analysis_sphtor_cplx(cfg::SHTConfig, vt::WrappedMtlMatrix{<:Complex}, vp::WrappedMtlMatrix{<:Complex}) =
    SHT.analysis_sphtor_cplx(GPU(), cfg, vt, vp)
SHT.analysis_qst(cfg::SHTConfig, vr::WrappedMtlMatrix, vt::WrappedMtlMatrix,
                 vp::WrappedMtlMatrix; kwargs...) = SHT.analysis_qst(GPU(), cfg, vr, vt, vp; kwargs...)
SHT.synthesis_qst(cfg::SHTConfig, q::WrappedMtlMatrix, s::WrappedMtlMatrix,
                  t::WrappedMtlMatrix; kwargs...) = SHT.synthesis_qst(GPU(), cfg, q, s, t; kwargs...)

SHT.synthesis_sph(cfg::SHTConfig, s::WrappedMtlMatrix; kwargs...) = SHT.synthesis_sph(GPU(), cfg, s; kwargs...)
SHT.synthesis_tor(cfg::SHTConfig, t::WrappedMtlMatrix; kwargs...) = SHT.synthesis_tor(GPU(), cfg, t; kwargs...)
SHT.synthesis_sph_cplx(cfg::SHTConfig, s::WrappedMtlMatrix) = SHT.synthesis_sph_cplx(GPU(), cfg, s)
SHT.synthesis_tor_cplx(cfg::SHTConfig, t::WrappedMtlMatrix) = SHT.synthesis_tor_cplx(GPU(), cfg, t)

function _with_plan_buffers(f, plan, count; inverse=false)
    # Metal never uses the plan's CPU Legendre buffers, so a plan remains valid
    # after cfg.lmax grows. Its FFT buffers depend only on the grid.
    cfg = plan.cfg
    half = plan.use_rfft && (!inverse || iseven(cfg.nlon))
    width = _nbins(cfg.nlon, half)
    shape = (cfg.nlat, cfg.nlon, count)
    build() = [MtlArray{ComplexF32}(undef, cfg.nlat, width) for _ in 1:count]
    return scalar_workspace_use!(build, PLAN_BUFFERS, Metal.device().registryID,
                                 plan, Float32, :fft, shape, hash(shape)) do buffers
        try
            # An analysis-only plan keeps half-width buffers. On the first odd
            # inverse, grow them under the plan lock. Later real forward FFTs
            # can write straight into the contiguous prefix of each buffer.
            active = ntuple(count) do i
                b = buffers[i]
                size(b, 2) == width && return b
                size(b, 2) > width && return view(b, :, 1:width)
                return buffers[i] = similar(b, (cfg.nlat, width))
            end
            _completed(f(cfg, active, half))
        catch
            # Drain any partially submitted work before another task reuses the buffers.
            Metal.synchronize()
            rethrow()
        end
    end
end

SHT.analysis!(cfg::SHTConfig, out::WrappedMtlMatrix, f::WrappedMtlMatrix;
              fft_scratch=nothing, use_rfft::Bool=false) =
    _completed(_scalar_analysis!(out, cfg, f; fft_scratch, use_rfft))
SHT.synthesis!(cfg::SHTConfig, out::WrappedMtlMatrix, a::WrappedMtlMatrix;
               real_output::Bool=true, fft_scratch=nothing, use_rfft::Bool=false) =
    _completed(_scalar_synthesis(cfg, a; out, real_output, fft_scratch, use_rfft))

function SHT.analysis!(plan::SHTPlan, out::WrappedMtlMatrix, f::WrappedMtlMatrix)
    _with_plan_buffers(plan, 1) do cfg, (scratch,), half
        _scalar_analysis!(out, cfg, f; use_rfft=half, fft_scratch=scratch)
    end
end
function SHT.synthesis!(plan::SHTPlan, out::WrappedMtlMatrix, a::WrappedMtlMatrix; real_output=true)
    _with_plan_buffers(plan, 1; inverse=true) do cfg, (scratch,), half
        _scalar_synthesis(cfg, a; out, real_output, use_rfft=plan.use_rfft, fft_scratch=scratch, half)
    end
end
SHT.analysis!(::GPU, plan::SHTPlan, out::WrappedMtlMatrix, f::WrappedMtlMatrix) =
    SHT.analysis!(plan, out, f)
SHT.synthesis!(::GPU, plan::SHTPlan, out::WrappedMtlMatrix, a::WrappedMtlMatrix; kwargs...) =
    SHT.synthesis!(plan, out, a; kwargs...)

function SHT.analysis_sphtor!(plan::SHTPlan, s::WrappedMtlMatrix, t::WrappedMtlMatrix,
                              vt::WrappedMtlMatrix, vp::WrappedMtlMatrix)
    _with_plan_buffers(plan, 2) do cfg, buffers, half
        _vector_analysis(cfg, vt, vp; out=(s,t), use_rfft=half, buffers)
    end
end
function SHT.synthesis_sphtor!(plan::SHTPlan, vt::WrappedMtlMatrix, vp::WrappedMtlMatrix,
                               s::WrappedMtlMatrix, t::WrappedMtlMatrix; real_output=true)
    _with_plan_buffers(plan, 2; inverse=true) do cfg, buffers, half
        _vector_synthesis(cfg, s, t; out=(vt,vp), real_output, use_rfft=plan.use_rfft, buffers, half)
    end
end

const RealBatch = WrappedMtlArray{<:Real,3}
const ComplexBatch = WrappedMtlArray{<:Complex,3}
const MetalBatch = WrappedMtlArray{T,3} where T

SHT.analysis_batch(::GPU, cfg::SHTConfig, f::RealBatch; use_rfft=false) =
    _completed(_scalar_analysis(cfg, f; use_rfft))
SHT.analysis_batch(cfg::SHTConfig, f::RealBatch; kwargs...) = SHT.analysis_batch(GPU(), cfg, f; kwargs...)
SHT.synthesis_batch(::GPU, cfg::SHTConfig, a::ComplexBatch; real_output=true, use_rfft=false) =
    _completed(_scalar_synthesis(cfg, a; real_output, use_rfft))
SHT.synthesis_batch(cfg::SHTConfig, a::ComplexBatch; kwargs...) = SHT.synthesis_batch(GPU(), cfg, a; kwargs...)
SHT.synthesis_batch_cplx(::GPU, cfg::SHTConfig, a::ComplexBatch) =
    _completed(_scalar_synthesis(cfg, a; real_output=false))
SHT.synthesis_batch_cplx(cfg::SHTConfig, a::ComplexBatch) = SHT.synthesis_batch_cplx(GPU(), cfg, a)

SHT.analysis_batch!(::GPU, cfg::SHTConfig, out::ComplexBatch, f::RealBatch; fft_batch=nothing, use_rfft=false) =
    _completed(_scalar_analysis!(out, cfg, f; fft_scratch=fft_batch, use_rfft))
SHT.analysis_batch!(cfg::SHTConfig, out::ComplexBatch, f::RealBatch; kwargs...) =
    SHT.analysis_batch!(GPU(), cfg, out, f; kwargs...)
SHT.synthesis_batch!(::GPU, cfg::SHTConfig, out::MetalBatch, a::ComplexBatch;
                     real_output=true, fft_batch=nothing, use_rfft=false) =
    _completed(_scalar_synthesis(cfg, a; out, real_output, fft_scratch=fft_batch, use_rfft))
SHT.synthesis_batch!(cfg::SHTConfig, out::MetalBatch, a::ComplexBatch; kwargs...) =
    SHT.synthesis_batch!(GPU(), cfg, out, a; kwargs...)

_batch_vector_analysis(cfg, vt, vp) =
    _vector_analysis(cfg, vt, vp; use_rfft=_graph_buffer(vt) && _graph_buffer(vp))
SHT.analysis_sphtor_batch(::GPU, cfg::SHTConfig, vt::RealBatch, vp::RealBatch) =
    _completed(_batch_vector_analysis(cfg, vt, vp))
SHT.analysis_sphtor_batch(cfg::SHTConfig, vt::RealBatch, vp::RealBatch) =
    SHT.analysis_sphtor_batch(GPU(), cfg, vt, vp)
SHT.synthesis_sphtor_batch(::GPU, cfg::SHTConfig, s::ComplexBatch, t::ComplexBatch; real_output=true) =
    _completed(_vector_synthesis(cfg, s, t; real_output, use_rfft=real_output))
SHT.synthesis_sphtor_batch(cfg::SHTConfig, s::ComplexBatch, t::ComplexBatch; kwargs...) =
    SHT.synthesis_sphtor_batch(GPU(), cfg, s, t; kwargs...)
SHT.synthesis_sphtor_batch_cplx(::GPU, cfg::SHTConfig, s::ComplexBatch, t::ComplexBatch) =
    _completed(_vector_synthesis(cfg, s, t; real_output=false))
SHT.synthesis_sphtor_batch_cplx(cfg::SHTConfig, s::ComplexBatch, t::ComplexBatch) =
    SHT.synthesis_sphtor_batch_cplx(GPU(), cfg, s, t)

function SHT.analysis_qst_batch(::GPU, cfg::SHTConfig, vr::RealBatch, vt::RealBatch, vp::RealBatch)
    GPUCommon.qst_batch_precision(vr, vt, vp)
    vr, vt, vp = SHT._qst_promoted((vr, vt, vp))
    q = _scalar_analysis(cfg, vr; use_rfft=_graph_buffer(vr))
    s, t = _batch_vector_analysis(cfg, vt, vp)
    return _completed((q, s, t))
end
SHT.analysis_qst_batch(cfg::SHTConfig, vr::RealBatch, vt::RealBatch, vp::RealBatch) =
    SHT.analysis_qst_batch(GPU(), cfg, vr, vt, vp)
function SHT.synthesis_qst_batch(::GPU, cfg::SHTConfig, q::ComplexBatch, s::ComplexBatch,
                                 t::ComplexBatch; real_output=true)
    GPUCommon.qst_batch_precision(q, s, t)
    q, s, t = SHT._qst_promoted((q, s, t))
    vr = _scalar_synthesis(cfg, q; real_output, use_rfft=real_output)
    vt, vp = _vector_synthesis(cfg, s, t; real_output, use_rfft=real_output)
    return _completed((vr, vt, vp))
end
SHT.synthesis_qst_batch(cfg::SHTConfig, q::ComplexBatch, s::ComplexBatch, t::ComplexBatch; kwargs...) =
    SHT.synthesis_qst_batch(GPU(), cfg, q, s, t; kwargs...)
SHT.synthesis_qst_batch_cplx(::GPU, cfg::SHTConfig, q::ComplexBatch, s::ComplexBatch, t::ComplexBatch) =
    SHT.synthesis_qst_batch(GPU(), cfg, q, s, t; real_output=false)
SHT.synthesis_qst_batch_cplx(cfg::SHTConfig, q::ComplexBatch, s::ComplexBatch, t::ComplexBatch) =
    SHT.synthesis_qst_batch_cplx(GPU(), cfg, q, s, t)

function SHT.gpu_apply_laplacian!(cfg::SHTConfig, a::WrappedMtlMatrix; device=GPU())
    device isa GPU || throw(ArgumentError("gpu_apply_laplacian! requires GPU()"))
    _check_field(a, _spectral(cfg))
    laplacian_kernel!(MetalBackend())(a, a, cfg.lmax, cfg.mmax, cfg.mres; ndrange=size(a))
    Metal.synchronize()
    return a
end

function _vector_diagonal!(cfg, out, a, inverse)
    _check_field(a, _spectral(cfg))
    size(out) == size(a) || throw(DimensionMismatch("incorrect vector operator output shape"))
    eltype(out) === eltype(a) || throw(ArgumentError("operator input and output precision must match"))
    source = out !== a && _overlaps(out, a) ? copy(a) : a
    vector_diagonal_kernel!(MetalBackend())(out, source, cfg.lmax, cfg.mmax, cfg.mres, inverse;
                                           ndrange=size(a))
    Metal.synchronize()
    return out
end

for (op, inverse) in ((:divergence_from_spheroidal, false), (:vorticity_from_toroidal, false),
                      (:spheroidal_from_divergence, true), (:toroidal_from_vorticity, true))
    inplace = Symbol(op, :!)
    @eval begin
        SHT.$op(cfg::SHTConfig, a::WrappedMtlMatrix) = _vector_diagonal!(cfg, similar(a), a, $inverse)
        SHT.$op(::GPU, cfg::SHTConfig, a::WrappedMtlMatrix) = SHT.$op(cfg, a)
        SHT.$inplace(cfg::SHTConfig, out::WrappedMtlMatrix, a::WrappedMtlMatrix) =
            _vector_diagonal!(cfg, out, a, $inverse)
        SHT.$inplace(::GPU, cfg::SHTConfig, out::WrappedMtlMatrix, a::WrappedMtlMatrix) =
            SHT.$inplace(cfg, out, a)
    end
end

# Degree-truncated, single-order and axisymmetric transforms. As in the CUDA
# and ROCm extensions, they reuse the full-grid kernels with a degree cap, or
# the shared single-order kernels with the cached tables.

function SHT.analysis_sphtor_l(::GPU, cfg::SHTConfig, vt::WrappedMtlMatrix, vp::WrappedMtlMatrix,
                               ltr::Integer)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    return _completed(_vector_analysis(cfg, vt, vp; lcap))
end
function SHT.synthesis_sphtor_l(::GPU, cfg::SHTConfig, s::WrappedMtlMatrix, t::WrappedMtlMatrix,
                                ltr::Integer; real_output::Bool=true)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    return _completed(_vector_synthesis(cfg, s, t; real_output, lcap))
end
SHT.synthesis_sphtor_l_cplx(::GPU, cfg::SHTConfig, s::WrappedMtlMatrix, t::WrappedMtlMatrix,
                            ltr::Integer) =
    SHT.synthesis_sphtor_l(GPU(), cfg, s, t, ltr; real_output=false)

_zeros_like(a) = fill!(similar(a), 0f0)
SHT.synthesis_sph_l(::GPU, cfg::SHTConfig, s::WrappedMtlMatrix, ltr::Integer; real_output::Bool=true) =
    SHT.synthesis_sphtor_l(GPU(), cfg, s, _zeros_like(s), ltr; real_output)
SHT.synthesis_tor_l(::GPU, cfg::SHTConfig, t::WrappedMtlMatrix, ltr::Integer; real_output::Bool=true) =
    SHT.synthesis_sphtor_l(GPU(), cfg, _zeros_like(t), t, ltr; real_output)
SHT.synthesis_sph_l_cplx(::GPU, cfg::SHTConfig, s::WrappedMtlMatrix, ltr::Integer) =
    SHT.synthesis_sph_l(GPU(), cfg, s, ltr; real_output=false)
SHT.synthesis_tor_l_cplx(::GPU, cfg::SHTConfig, t::WrappedMtlMatrix, ltr::Integer) =
    SHT.synthesis_tor_l(GPU(), cfg, t, ltr; real_output=false)
SHT.synthesis_grad(::GPU, cfg::SHTConfig, s::WrappedMtlMatrix; kwargs...) =
    SHT.synthesis_sph(GPU(), cfg, s; kwargs...)
SHT.synthesis_grad_l(::GPU, cfg::SHTConfig, s::WrappedMtlMatrix, ltr::Integer; kwargs...) =
    SHT.synthesis_sph_l(GPU(), cfg, s, ltr; kwargs...)

function SHT.analysis_qst_l(::GPU, cfg::SHTConfig, vr::WrappedMtlMatrix, vt::WrappedMtlMatrix,
                            vp::WrappedMtlMatrix, ltr::Integer)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    vr, vt, vp = SHT._qst_promoted((vr, vt, vp))
    q = _scalar_analysis(cfg, vr; lcap)
    s, t = _vector_analysis(cfg, vt, vp; lcap)
    return _completed((q, s, t))
end
function SHT.synthesis_qst_l(::GPU, cfg::SHTConfig, q::WrappedMtlMatrix, s::WrappedMtlMatrix,
                             t::WrappedMtlMatrix, ltr::Integer; real_output::Bool=true)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    q, s, t = SHT._qst_promoted((q, s, t))
    vr = _scalar_synthesis(cfg, q; real_output, lcap)
    vt, vp = _vector_synthesis(cfg, s, t; real_output, lcap)
    return _completed((vr, vt, vp))
end
SHT.synthesis_qst_l_cplx(::GPU, cfg::SHTConfig, q::WrappedMtlMatrix, s::WrappedMtlMatrix,
                         t::WrappedMtlMatrix, ltr::Integer) =
    SHT.synthesis_qst_l(GPU(), cfg, q, s, t, ltr; real_output=false)

function _check_mode(a, n)
    length(a) == n || throw(DimensionMismatch("expected $n entries, got $(length(a))"))
    eltype(a) in (Float32, ComplexF32) || throw(ArgumentError(
        "Metal transforms require Float32 or ComplexF32, got $(eltype(a))",
    ))
end

# Degrees m:lcap of one physical order m, in the configured normalization.
function _mode_analysis(cfg, m, mode, lcap, scale)
    tables = _tables(cfg, false)
    out = similar(mode, ComplexF32, lcap - m + 1)
    mode_analysis_kernel!(MetalBackend())(out, mode, tables.Plm, tables.weights, Float32(scale),
                                          m, lcap; ndrange=length(out))
    out ./= view(tables.scales, (m + 1):(lcap + 1), m + 1)
    return out
end

function _mode_synthesis(cfg, m, coefficients, lcap, scale)
    tables = _tables(cfg, false)
    canonical = coefficients .* view(tables.scales, (m + 1):(lcap + 1), m + 1)
    mode = similar(coefficients, ComplexF32, cfg.nlat)
    mode_synthesis_kernel!(MetalBackend())(mode, canonical, tables.Plm, Float32(scale), m, lcap;
                                           ndrange=cfg.nlat)
    return mode
end

function _vector_mode_analysis(cfg, stored_im, vt, vp, ltr)
    m, lcap = SHT._validate_vector_fixed_order(cfg, stored_im, ltr)
    SHT._validate_robert_analysis(cfg; m, ltr=lcap)
    vt, vp = _promoted_pair(vt, vp)
    _check_mode(vt, cfg.nlat)
    _check_mode(vp, cfg.nlat)
    tables = _tables(cfg, true)
    s = similar(vt, ComplexF32, lcap - m + 1)
    t = similar(s)
    vector_mode_analysis_kernel!(MetalBackend())(
        s, t, vt, vp, tables.dtheta, tables.over_sin, tables.weights, tables.scales,
        tables.sint, Float32(SHT._analysis_phi_scale(cfg)), m, lcap, cfg.robert_form;
        ndrange=length(s),
    )
    return s, t
end

function _vector_mode_synthesis(cfg, stored_im, s, t, ltr)
    m, lcap = SHT._validate_vector_fixed_order(cfg, stored_im, ltr)
    s, t = _promoted_pair(s, t)
    _check_mode(s, lcap - m + 1)
    _check_mode(t, lcap - m + 1)
    tables = _tables(cfg, true)
    vt = similar(s, ComplexF32, cfg.nlat)
    vp = similar(vt)
    vector_mode_synthesis_kernel!(MetalBackend())(
        vt, vp, s, t, tables.dtheta, tables.over_sin, tables.scales, tables.sint,
        Float32(SHT.phi_inv_scale(cfg)), m, lcap, cfg.robert_form; ndrange=cfg.nlat,
    )
    return vt, vp
end

function SHT.analysis_packed_ml(::GPU, cfg::SHTConfig, im::Int, mode::WrappedMtlVector{<:Complex},
                                ltr::Integer)
    _, m, lcap = SHT._validate_stored_order(cfg, im, ltr)
    _check_mode(mode, cfg.nlat)
    return _completed(_mode_analysis(cfg, m, mode, lcap, SHT._analysis_phi_scale(cfg)))
end
function SHT.synthesis_packed_ml(::GPU, cfg::SHTConfig, im::Int,
                                 coefficients::WrappedMtlVector{<:Complex}, ltr::Integer)
    _, m, lcap = SHT._validate_stored_order(cfg, im, ltr)
    _check_mode(coefficients, lcap - m + 1)
    return _completed(_mode_synthesis(cfg, m, coefficients, lcap, SHT.phi_inv_scale(cfg)))
end

SHT.analysis_sphtor_ml(::GPU, cfg::SHTConfig, im::Integer, vt::WrappedMtlVector{<:Complex},
                       vp::WrappedMtlVector{<:Complex}, ltr::Integer) =
    _completed(_vector_mode_analysis(cfg, im, vt, vp, ltr))
SHT.synthesis_sphtor_ml(::GPU, cfg::SHTConfig, im::Integer, s::WrappedMtlVector{<:Complex},
                        t::WrappedMtlVector{<:Complex}, ltr::Integer) =
    _completed(_vector_mode_synthesis(cfg, im, s, t, ltr))
SHT.synthesis_sph_ml(::GPU, cfg::SHTConfig, im::Integer, s::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_sphtor_ml(GPU(), cfg, im, s, _zeros_like(s), ltr)
SHT.synthesis_tor_ml(::GPU, cfg::SHTConfig, im::Integer, t::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_sphtor_ml(GPU(), cfg, im, _zeros_like(t), t, ltr)
SHT.synthesis_grad_ml(::GPU, cfg::SHTConfig, im::Integer, s::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_sph_ml(GPU(), cfg, im, s, ltr)

function SHT.analysis_qst_ml(::GPU, cfg::SHTConfig, im::Integer, vr::WrappedMtlVector{<:Complex},
                             vt::WrappedMtlVector{<:Complex}, vp::WrappedMtlVector{<:Complex},
                             ltr::Integer)
    stored_im, m, lcap = SHT._validate_stored_order(cfg, im, ltr)
    vr, vt, vp = SHT._qst_promoted((vr, vt, vp))
    _check_mode(vr, cfg.nlat)
    q = _mode_analysis(cfg, m, vr, lcap, SHT._analysis_phi_scale(cfg))
    s, t = _vector_mode_analysis(cfg, stored_im, vt, vp, lcap)
    return _completed((q, s, t))
end
function SHT.synthesis_qst_ml(::GPU, cfg::SHTConfig, im::Integer, q::WrappedMtlVector{<:Complex},
                              s::WrappedMtlVector{<:Complex}, t::WrappedMtlVector{<:Complex},
                              ltr::Integer)
    stored_im, m, lcap = SHT._validate_stored_order(cfg, im, ltr)
    q, s, t = SHT._qst_promoted((q, s, t))
    _check_mode(q, lcap - m + 1)
    vr = _mode_synthesis(cfg, m, q, lcap, SHT.phi_inv_scale(cfg))
    vt, vp = _vector_mode_synthesis(cfg, stored_im, s, t, lcap)
    return _completed((vr, vt, vp))
end

# The axisymmetric profile is constant in φ, so analysis applies the whole
# 2π = cphi*nlon quadrature factor the full transform splits with the FFT.
SHT.analysis_axisym(::GPU, cfg::SHTConfig, field::WrappedMtlVector{<:Real}) =
    SHT.analysis_axisym_l(GPU(), cfg, field, cfg.lmax)
function SHT.analysis_axisym_l(::GPU, cfg::SHTConfig, field::WrappedMtlVector{<:Real}, ltr::Integer)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    _check_mode(field, cfg.nlat)
    scale = SHT._analysis_phi_scale(cfg) * cfg.nlon
    return _completed(_mode_analysis(cfg, 0, field, lcap, scale))
end
function SHT.synthesis_axisym(::GPU, cfg::SHTConfig, coefficients::WrappedMtlVector{<:Complex})
    length(coefficients) == cfg.lmax + 1 || throw(DimensionMismatch(
        "coefficients must have length lmax+1=$(cfg.lmax + 1)",
    ))
    return SHT.synthesis_axisym_l(GPU(), cfg, coefficients, cfg.lmax)
end
function SHT.synthesis_axisym_l(::GPU, cfg::SHTConfig, coefficients::WrappedMtlVector{<:Complex},
                                ltr::Integer)
    lcap = SHT._validate_degree_limit(cfg, ltr)
    lcap < length(coefficients) || throw(ArgumentError(
        "ltr must be <= length(Qlm)-1=$(length(coefficients) - 1)",
    ))
    used = view(coefficients, 1:(lcap + 1))
    _check_mode(used, lcap + 1)
    mode = _mode_synthesis(cfg, 0, used, lcap, SHT._evaluator_phi_scale(cfg))
    return _completed(real.(mode))
end

# Storage-dispatched forms.
SHT.analysis_sphtor_l(cfg::SHTConfig, vt::WrappedMtlMatrix, vp::WrappedMtlMatrix, ltr::Integer) =
    SHT.analysis_sphtor_l(GPU(), cfg, vt, vp, ltr)
SHT.synthesis_sphtor_l(cfg::SHTConfig, s::WrappedMtlMatrix, t::WrappedMtlMatrix, ltr::Integer;
                       kwargs...) =
    SHT.synthesis_sphtor_l(GPU(), cfg, s, t, ltr; kwargs...)
SHT.synthesis_sphtor_l_cplx(cfg::SHTConfig, s::WrappedMtlMatrix, t::WrappedMtlMatrix, ltr::Integer) =
    SHT.synthesis_sphtor_l_cplx(GPU(), cfg, s, t, ltr)
SHT.synthesis_sph_l(cfg::SHTConfig, s::WrappedMtlMatrix, ltr::Integer; kwargs...) =
    SHT.synthesis_sph_l(GPU(), cfg, s, ltr; kwargs...)
SHT.synthesis_tor_l(cfg::SHTConfig, t::WrappedMtlMatrix, ltr::Integer; kwargs...) =
    SHT.synthesis_tor_l(GPU(), cfg, t, ltr; kwargs...)
SHT.synthesis_sph_l_cplx(cfg::SHTConfig, s::WrappedMtlMatrix, ltr::Integer) =
    SHT.synthesis_sph_l_cplx(GPU(), cfg, s, ltr)
SHT.synthesis_tor_l_cplx(cfg::SHTConfig, t::WrappedMtlMatrix, ltr::Integer) =
    SHT.synthesis_tor_l_cplx(GPU(), cfg, t, ltr)
SHT.analysis_qst_l(cfg::SHTConfig, vr::WrappedMtlMatrix, vt::WrappedMtlMatrix, vp::WrappedMtlMatrix,
                   ltr::Integer) =
    SHT.analysis_qst_l(GPU(), cfg, vr, vt, vp, ltr)
SHT.synthesis_qst_l(cfg::SHTConfig, q::WrappedMtlMatrix, s::WrappedMtlMatrix, t::WrappedMtlMatrix,
                    ltr::Integer; kwargs...) =
    SHT.synthesis_qst_l(GPU(), cfg, q, s, t, ltr; kwargs...)
SHT.synthesis_qst_l_cplx(cfg::SHTConfig, q::WrappedMtlMatrix, s::WrappedMtlMatrix,
                         t::WrappedMtlMatrix, ltr::Integer) =
    SHT.synthesis_qst_l_cplx(GPU(), cfg, q, s, t, ltr)
SHT.analysis_packed_ml(cfg::SHTConfig, im::Int, mode::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.analysis_packed_ml(GPU(), cfg, im, mode, ltr)
SHT.synthesis_packed_ml(cfg::SHTConfig, im::Int, a::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_packed_ml(GPU(), cfg, im, a, ltr)
SHT.analysis_sphtor_ml(cfg::SHTConfig, im::Integer, vt::WrappedMtlVector{<:Complex},
                       vp::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.analysis_sphtor_ml(GPU(), cfg, im, vt, vp, ltr)
SHT.synthesis_sphtor_ml(cfg::SHTConfig, im::Integer, s::WrappedMtlVector{<:Complex},
                        t::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_sphtor_ml(GPU(), cfg, im, s, t, ltr)
SHT.synthesis_sph_ml(cfg::SHTConfig, im::Integer, s::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_sph_ml(GPU(), cfg, im, s, ltr)
SHT.synthesis_tor_ml(cfg::SHTConfig, im::Integer, t::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_tor_ml(GPU(), cfg, im, t, ltr)
SHT.analysis_qst_ml(cfg::SHTConfig, im::Integer, vr::WrappedMtlVector{<:Complex},
                    vt::WrappedMtlVector{<:Complex}, vp::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.analysis_qst_ml(GPU(), cfg, im, vr, vt, vp, ltr)
SHT.synthesis_qst_ml(cfg::SHTConfig, im::Integer, q::WrappedMtlVector{<:Complex},
                     s::WrappedMtlVector{<:Complex}, t::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_qst_ml(GPU(), cfg, im, q, s, t, ltr)
SHT.analysis_axisym(cfg::SHTConfig, field::WrappedMtlVector{<:Real}) =
    SHT.analysis_axisym(GPU(), cfg, field)
SHT.analysis_axisym_l(cfg::SHTConfig, field::WrappedMtlVector{<:Real}, ltr::Integer) =
    SHT.analysis_axisym_l(GPU(), cfg, field, ltr)
SHT.synthesis_axisym(cfg::SHTConfig, a::WrappedMtlVector{<:Complex}) =
    SHT.synthesis_axisym(GPU(), cfg, a)
SHT.synthesis_axisym_l(cfg::SHTConfig, a::WrappedMtlVector{<:Complex}, ltr::Integer) =
    SHT.synthesis_axisym_l(GPU(), cfg, a, ltr)

end # module
