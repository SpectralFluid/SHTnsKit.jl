using Test
using LinearAlgebra

# Reuse the same mathematical oracle suites as CPU and standalone GPU parity.
isdefined(@__MODULE__, :ScalarParityAdapter) || include("scalar_full.jl")
isdefined(@__MODULE__, :VectorParityAdapter) || include("sphtor_full.jl")
isdefined(@__MODULE__, :QSTParityAdapter) || include("qst_full.jl")
isdefined(@__MODULE__, :SHTNS37_MANIFEST_PATH) || include("shtns37_fixtures.jl")
isdefined(@__MODULE__, :firewall_dispatch_probe) ||
    include(joinpath(@__DIR__, "..", "parallel", "mpi_gpu_firewall_probe.jl"))

"""A CPU-backed stand-in that exercises the MPI/GPU policy without GPU hardware."""
mutable struct MockMPIArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.IndexStyle(::Type{<:MockMPIArray}) = IndexLinear()
Base.size(array::MockMPIArray) = size(array.data)
Base.getindex(array::MockMPIArray, indices...) = array.data[indices...]
Base.setindex!(array::MockMPIArray, value, indices...) =
    (array.data[indices...] = value)
Base.copyto!(destination::MockMPIArray, source::AbstractArray) =
    (copyto!(destination.data, source); destination)
Base.copyto!(destination::AbstractArray, source::MockMPIArray) =
    copyto!(destination, source.data)
MockMPIArray{T}(::UndefInitializer, dims::Tuple) where {T} =
    MockMPIArray(Array{T}(undef, dims))
MockMPIArray{T}(::UndefInitializer, n::Integer) where {T} =
    MockMPIArray(Vector{T}(undef, n))
Base.similar(array::MockMPIArray, ::Type{T}, dims::Dims) where {T} =
    MockMPIArray(Array{T}(undef, dims))

"""A registry-isolated fake used only by mixed-storage dispatch tests."""
mutable struct MockMixedVendorArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.IndexStyle(::Type{<:MockMixedVendorArray}) = IndexLinear()
Base.size(array::MockMixedVendorArray) = size(array.data)
Base.getindex(array::MockMixedVendorArray, indices...) = array.data[indices...]
Base.setindex!(array::MockMixedVendorArray, value, indices...) =
    (array.data[indices...] = value)
Base.copyto!(destination::MockMixedVendorArray, source::AbstractArray) =
    (copyto!(destination.data, source); destination)
Base.copyto!(destination::AbstractArray, source::MockMixedVendorArray) =
    copyto!(destination, source.data)
Base.copyto!(destination::MockMixedVendorArray,
             source::MockMixedVendorArray) =
    (copyto!(destination.data, source.data); destination)
MockMixedVendorArray{T}(::UndefInitializer, dims::Tuple) where {T} =
    MockMixedVendorArray(Array{T}(undef, dims))
MockMixedVendorArray{T}(::UndefInitializer, n::Integer) where {T} =
    MockMixedVendorArray(Vector{T}(undef, n))
Base.similar(array::MockMixedVendorArray, ::Type{T}, dims::Dims) where {T} =
    MockMixedVendorArray(Array{T}(undef, dims))

"""A fake allocation whose device may differ from the task's current device."""
mutable struct MockMultiDeviceArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    device::Int
end

const MOCK_CURRENT_DEVICE = Ref(99)
MockMultiDeviceArray{T}(::UndefInitializer, dims::Dims) where {T} =
    MockMultiDeviceArray(Array{T}(undef, dims), MOCK_CURRENT_DEVICE[])

Base.IndexStyle(::Type{<:MockMultiDeviceArray}) = IndexLinear()
Base.size(array::MockMultiDeviceArray) = size(array.data)
Base.getindex(array::MockMultiDeviceArray, indices...) = array.data[indices...]
Base.setindex!(array::MockMultiDeviceArray, value, indices...) =
    (array.data[indices...] = value)
Base.copyto!(destination::MockMultiDeviceArray, source::AbstractArray) =
    (copyto!(destination.data, source); destination)
Base.copyto!(destination::AbstractArray, source::MockMultiDeviceArray) =
    copyto!(destination, source.data)
Base.copyto!(destination::MockMultiDeviceArray,
             source::MockMultiDeviceArray) =
    (copyto!(destination.data, source.data); destination)

let extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    Core.eval(extension, Meta.parse("""
        function _dist_transpose_gpu_analysis!(
                ::Val{:mock_native_scope}, adapter, callback::Function,
                output::Main.MockMultiDeviceArray,
                input::Main.MockMultiDeviceArray)
            return _with_owner_device(adapter, input) do
                callback()
                output
            end
        end
    """))
end

function _cache_temporary_view!(extension, adapter, comm)
    root = MockMultiDeviceArray(reshape(Float32.(1:4), 2, 2), 1)
    extension._staging_entry(adapter, comm, view(root, 1, :), 2)
    return WeakRef(root)
end

function _cache_temporary_logical_view!(extension, adapter, comm, root)
    logical = view(root, 1, :)
    extension._staging_entry(adapter, comm, logical, length(logical))
    return WeakRef(logical)
end

function _mpi_gpu_place(array_type, values::AbstractArray, decomposition, comm)
    pen = PencilArrays.Pencil(array_type, size(values), decomposition, comm)
    result = PencilArrays.PencilArray{eltype(values)}(undef, pen)
    ranges = PencilArrays.range_local(pen)
    local_values = Array{eltype(values)}(undef, size(parent(result)))
    @inbounds for index in CartesianIndices(local_values)
        global_index = ntuple(d -> ranges[d][index[d]], ndims(values))
        local_values[index] = values[global_index...]
    end
    copyto!(parent(result), local_values)
    return result
end

function _mpi_gpu_collect(value::PencilArrays.PencilArray, comm)
    local_values = Array(parent(value))
    global_value = zeros(eltype(value), PencilArrays.size_global(value))
    ranges = PencilArrays.range_local(PencilArrays.pencil(value))
    @inbounds for index in CartesianIndices(local_values)
        global_index = ntuple(d -> ranges[d][index[d]], ndims(local_values))
        global_value[global_index...] = local_values[index]
    end
    MPI.Allreduce!(global_value, +, comm)
    return global_value
end

_mpi_gpu_collect_any(value::PencilArrays.PencilArray, comm) =
    _mpi_gpu_collect(value, comm)
_mpi_gpu_collect_any(value::AbstractArray, _comm) = Array(value)

function _mpi_gpu_fill_native_spatial!(destination, fields)
    ranges = PencilArrays.range_local(PencilArrays.pencil(destination))
    host = Array{eltype(destination)}(undef, size(parent(destination)))
    @inbounds for lev in axes(host, 3), local_θ in axes(host, 2),
                  local_φ in axes(host, 1)
        host[local_φ, local_θ, lev] =
            fields[lev][ranges[2][local_θ], ranges[1][local_φ]]
    end
    copyto!(parent(destination), host)
    return destination
end

function _mpi_gpu_fill_native_spectral!(destination, coefficients, m_local)
    host = zeros(eltype(destination), size(parent(destination)))
    @inbounds for lev in axes(host, 3), (local_m, m) in enumerate(m_local),
                  l in m:(size(host, 1) - 1)
        host[l + 1, local_m, lev] = coefficients[lev][l + 1, m + 1]
    end
    copyto!(parent(destination), host)
    return destination
end

function _mpi_gpu_native_spectral_error(value, references, m_local)
    host = Array(parent(value))
    error = zero(typeof(abs(zero(eltype(host)))))
    @inbounds for lev in axes(host, 3), (local_m, m) in enumerate(m_local),
                  l in m:(size(host, 1) - 1)
        error = max(error, abs(
            host[l + 1, local_m, lev] - references[lev][l + 1, m + 1],
        ))
    end
    return error
end

function _mpi_gpu_native_spatial_error(value, references)
    ranges = PencilArrays.range_local(PencilArrays.pencil(value))
    host = Array(parent(value))
    error = zero(eltype(host))
    @inbounds for lev in axes(host, 3), local_θ in axes(host, 2),
                  local_φ in axes(host, 1)
        error = max(error, abs(
            host[local_φ, local_θ, lev] -
            references[lev][ranges[2][local_θ], ranges[1][local_φ]],
        ))
    end
    return error
end

function _mpi_gpu_native_references(cfg, ::Type{RT}, nlev) where {RT}
    CT = Complex{RT}
    Q = [zeros(CT, cfg.lmax + 1, cfg.mmax + 1) for _ in 1:nlev]
    S = [zeros(CT, cfg.lmax + 1, cfg.mmax + 1) for _ in 1:nlev]
    T = [zeros(CT, cfg.lmax + 1, cfg.mmax + 1) for _ in 1:nlev]
    for lev in 1:nlev, m in 0:cfg.mmax, l in m:cfg.lmax
        scale = RT(0.025 * (lev + 1) / (l + 1)^2)
        imag_scale = m == 0 ? zero(RT) : RT(0.35) * scale
        Q[lev][l + 1, m + 1] = CT(scale, imag_scale)
        if l > 0
            S[lev][l + 1, m + 1] = CT(RT(0.7) * scale, -imag_scale)
            T[lev][l + 1, m + 1] = CT(-RT(0.4) * scale, RT(0.2) * imag_scale)
        end
    end
    Vr = [SHTnsKit.synthesis(cfg, Q[lev]; real_output=true) for lev in 1:nlev]
    vector = [SHTnsKit.synthesis_sphtor(
        cfg, S[lev], T[lev]; real_output=true,
    ) for lev in 1:nlev]
    Vt = [vector[lev][1] for lev in 1:nlev]
    Vp = [vector[lev][2] for lev in 1:nlev]
    Qanalysis = [SHTnsKit.analysis(cfg, Vr[lev]) for lev in 1:nlev]
    STanalysis = [SHTnsKit.analysis_sphtor(
        cfg, Vt[lev], Vp[lev],
    ) for lev in 1:nlev]
    Sanalysis = [STanalysis[lev][1] for lev in 1:nlev]
    Tanalysis = [STanalysis[lev][2] for lev in 1:nlev]
    return (; Q, S, T, Vr, Vt, Vp, Qanalysis, Sanalysis, Tanalysis)
end

@inline function _mpi_gpu_assert_resident(value, is_vendor)
    if value isa PencilArrays.PencilArray
        @test is_vendor(parent(value))
    elseif value isa AbstractArray
        @test is_vendor(value)
    elseif value isa Tuple
        foreach(item -> _mpi_gpu_assert_resident(item, is_vendor), value)
    end
    return value
end

struct MPIGPUScalarAdapter{A,F} <: ScalarParityAdapter
    array_type::A
    is_vendor::F
    comm
end
struct MPIGPUVectorAdapter{A,F} <: VectorParityAdapter
    array_type::A
    is_vendor::F
    comm
end
struct MPIGPUQSTAdapter{A,F} <: QSTParityAdapter
    array_type::A
    is_vendor::F
    comm
end

place(adapter::MPIGPUScalarAdapter, cfg, value, kind::Symbol) =
    _mpi_gpu_place(adapter.array_type, value,
                   kind === :spectral ? (2,) : (1,), adapter.comm)
collect_result(adapter::MPIGPUScalarAdapter, value::PencilArrays.PencilArray,
               _cfg) = _mpi_gpu_collect(value, adapter.comm)
analysis_call(::MPIGPUScalarAdapter, cfg, field; use_rfft=false) =
    SHTnsKit.analysis(cfg, field; use_rfft)
synthesis_call(::MPIGPUScalarAdapter, cfg, coefficients, prototype;
               real_output, use_rfft=false) = SHTnsKit.synthesis(
    cfg, coefficients; prototype_θφ=prototype, real_output, use_rfft,
)
synthesis_cplx_call(::MPIGPUScalarAdapter, cfg, coefficients, prototype) =
    SHTnsKit.synthesis_cplx(cfg, coefficients; prototype_θφ=prototype)
assert_resident(adapter::MPIGPUScalarAdapter, value) =
    _mpi_gpu_assert_resident(value, adapter.is_vendor)

vector_place(adapter::MPIGPUVectorAdapter, cfg, value, kind::Symbol) =
    _mpi_gpu_place(adapter.array_type, value,
                   kind === :spectral ? (2,) : (1,), adapter.comm)
vector_collect(adapter::MPIGPUVectorAdapter,
               value::PencilArrays.PencilArray, _cfg) =
    _mpi_gpu_collect(value, adapter.comm)
vector_resident(adapter::MPIGPUVectorAdapter, value) =
    _mpi_gpu_assert_resident(value, adapter.is_vendor)
vector_analysis(::MPIGPUVectorAdapter, cfg, Vt, Vp; use_rfft=false) =
    SHTnsKit.analysis_sphtor(cfg, Vt, Vp; use_rfft)
vector_analysis_cplx(::MPIGPUVectorAdapter, cfg, Vt, Vp) =
    SHTnsKit.analysis_sphtor_cplx(cfg, Vt, Vp)
vector_synthesis(::MPIGPUVectorAdapter, cfg, S, T, prototype;
                 real_output=true, use_rfft=false) = SHTnsKit.synthesis_sphtor(
    cfg, S, T; prototype_θφ=prototype, real_output, use_rfft,
)
vector_synthesis_cplx(::MPIGPUVectorAdapter, cfg, S, T, prototype) =
    SHTnsKit.synthesis_sphtor_cplx(cfg, S, T; prototype_θφ=prototype)
vector_sph(::MPIGPUVectorAdapter, cfg, S, prototype; real_output=true) =
    SHTnsKit.synthesis_sph(cfg, S; prototype_θφ=prototype, real_output)
vector_sph_cplx(::MPIGPUVectorAdapter, cfg, S, prototype) =
    SHTnsKit.synthesis_sph_cplx(cfg, S; prototype_θφ=prototype)
vector_tor(::MPIGPUVectorAdapter, cfg, T, prototype; real_output=true) =
    SHTnsKit.synthesis_tor(cfg, T; prototype_θφ=prototype, real_output)
vector_tor_cplx(::MPIGPUVectorAdapter, cfg, T, prototype) =
    SHTnsKit.synthesis_tor_cplx(cfg, T; prototype_θφ=prototype)

qst_place(adapter::MPIGPUQSTAdapter, cfg, value, kind::Symbol) =
    _mpi_gpu_place(adapter.array_type, value,
                   kind === :spectral ? (2,) : (1,), adapter.comm)
qst_collect(adapter::MPIGPUQSTAdapter, value::PencilArrays.PencilArray, _cfg) =
    _mpi_gpu_collect(value, adapter.comm)
qst_resident(adapter::MPIGPUQSTAdapter, value) =
    _mpi_gpu_assert_resident(value, adapter.is_vendor)
qst_analysis(::MPIGPUQSTAdapter, cfg, Vr, Vt, Vp; use_rfft=false) =
    SHTnsKit.analysis_qst(cfg, Vr, Vt, Vp; use_rfft)
qst_analysis_cplx(::MPIGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    SHTnsKit.analysis_qst_cplx(cfg, Vr, Vt, Vp)
qst_analysis_inferred(::MPIGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    SHTnsKit.analysis_qst(cfg, Vr, Vt, Vp)
qst_analysis_cplx_inferred(::MPIGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    SHTnsKit.analysis_qst_cplx(cfg, Vr, Vt, Vp)
qst_synthesis(::MPIGPUQSTAdapter, cfg, Q, S, T, prototype;
              real_output=true, use_rfft=false) = SHTnsKit.synthesis_qst(
    cfg, Q, S, T; prototype_θφ=prototype, real_output, use_rfft,
)
qst_synthesis_cplx(::MPIGPUQSTAdapter, cfg, Q, S, T, prototype) =
    SHTnsKit.synthesis_qst_cplx(cfg, Q, S, T; prototype_θφ=prototype)
qst_synthesis_inferred(::MPIGPUQSTAdapter, cfg, Q, S, T, prototype) =
    SHTnsKit.synthesis_qst(cfg, Q, S, T; prototype_θφ=prototype)
qst_synthesis_cplx_inferred(::MPIGPUQSTAdapter, cfg, Q, S, T, prototype) =
    SHTnsKit.synthesis_qst_cplx(cfg, Q, S, T; prototype_θφ=prototype)

function _test_shtns37_analysis_fixture_mpi_gpu(f,p,cfg,place_values,check)
    cap=Symbol(f["capability"]);a=f["atol"];r=f["rtol"]
    if cap===:scalar_real_full
        got=analysis_batch(cfg,place_values(p["field"],:spatial));check(got,_shtns37_batch_dense(cfg,p,"coefficients"),a,r)
    elseif cap===:scalar_complex_full
        check(dist_analysis_packed_cplx(cfg,place_values(p["field"],:spatial)),reshape(vec(p["coefficients"]),:,1),a,r)
    elseif cap===:scalar_l
        check(analysis_packed_l(cfg,place_values(p["field"],:spatial),f["ltr"]),reshape(vec(p["coefficients"]),:,1),a,r)
    elseif cap===:scalar_ml
        sc=f["fixed_mode_scale"];check(analysis_packed_ml(cfg,f["stored_im"],place_values(reshape(sc.*vec(p["field"]),:,1),:spatial),f["ltr"]),reshape(vec(p["coefficients"]),:,1),a,r)
    elseif cap===:sphtor_full
        got=analysis_sphtor_batch(cfg,place_values(p["Vt"],:spatial),place_values(p["Vp"],:spatial));for(i,n)in enumerate(("S","T"));check(got[i],_shtns37_batch_dense(cfg,p,n),a,r);end
    elseif cap===:sphtor_l
        got=analysis_sphtor_l(cfg,place_values(p["Vt"],:spatial),place_values(p["Vp"],:spatial),f["ltr"]);for(i,n)in enumerate(("S","T"));check(got[i],_shtns37_dense(cfg,p[n]),a,r);end
    elseif cap===:sphtor_ml
        sc=f["fixed_mode_scale"];got=analysis_sphtor_ml(cfg,f["stored_im"],place_values(reshape(sc.*vec(p["Vt"]),:,1),:spatial),place_values(reshape(sc.*vec(p["Vp"]),:,1),:spatial),f["ltr"]);for(i,n)in enumerate(("S","T"));check(got[i],reshape(vec(p[n]),:,1),a,r);end
    elseif cap===:qst_full
        got=analysis_qst_batch(cfg,(place_values(p[n],:spatial) for n in ("Vr","Vt","Vp"))...);for(i,n)in enumerate(("Q","S","T"));check(got[i],_shtns37_batch_dense(cfg,p,n),a,r);end
    elseif cap===:qst_l
        got=analysis_qst_l(cfg,(place_values(p[n],:spatial) for n in ("Vr","Vt","Vp"))...,f["ltr"]);for(i,n)in enumerate(("Q","S","T"));check(got[i],_shtns37_dense(cfg,p[n]),a,r);end
    elseif cap===:qst_ml
        sc=f["fixed_mode_scale"];got=analysis_qst_ml(cfg,f["stored_im"],(place_values(reshape(sc.*vec(p[n]),:,1),:spatial) for n in ("Vr","Vt","Vp"))...,f["ltr"]);for(i,n)in enumerate(("Q","S","T"));check(got[i],reshape(vec(p[n]),:,1),a,r);end
    end
end

"""Compare every SHTns 3.7 payload on GPU-backed distributed storage."""
function test_shtns37_mpi_gpu_fixtures(array_type, is_vendor, comm)
    manifest = TOML.parsefile(SHTNS37_MANIFEST_PATH)
    place_values(values, kind) = _mpi_gpu_place(
        array_type, values, kind === :spectral ? (2,) : (1,), comm,
    )
    collect_values(value) = _mpi_gpu_collect_any(value, comm)
    function check(value, expected, atol, rtol)
        _mpi_gpu_assert_resident(value, is_vendor)
        @test collect_values(value) ≈ expected atol=atol rtol=rtol
    end
    @testset "SHTns 3.7 MPI GPU fixtures" begin
        for f in manifest["fixture"]
            cfg = _shtns37_config(f)
            p = _shtns37_payloads(f)
            cap = Symbol(f["capability"])
            atol, rtol = f["atol"], f["rtol"]
            RT = f["precision"] == "float32" ? Float32 : Float64
            prototype = place_values(zeros(RT, cfg.nlat, cfg.nlon), :spatial)
            id = f["id"]
            @testset "$id" begin
                if get(f,"direction","")=="analysis"
                    _test_shtns37_analysis_fixture_mpi_gpu(f,p,cfg,place_values,check)
                elseif cap === :scalar_real_full
                    Q = place_values(_shtns37_dense(cfg, p["coefficients"]), :spectral)
                    check(synthesis(cfg, Q; prototype_θφ=prototype), p["field"], atol, rtol)
                elseif cap === :scalar_complex_full
                    A = place_values(reshape(vec(p["coefficients"]), :, 1), :spatial)
                    cp = place_values(zeros(Complex{RT}, cfg.nlat, cfg.nlon), :spatial)
                    check(synthesis_packed_cplx(cfg, A; prototype_θφ=cp), p["field"], atol, rtol)
                elseif cap in (:scalar_l, :packed_storage)
                    Q = place_values(reshape(vec(p["coefficients"]), :, 1), :spatial)
                    got = cap === :scalar_l ? synthesis_packed_l(cfg, Q, f["ltr"]; prototype_θφ=prototype) : synthesis_packed(cfg, Q; prototype_θφ=prototype)
                    check(got, p["field"], atol, rtol)
                elseif cap === :scalar_ml
                    Q = place_values(reshape(vec(p["coefficients"]), :, 1), :spatial)
                    check(synthesis_packed_ml(cfg, f["stored_im"], Q, f["ltr"]), f["fixed_mode_scale"] .* reshape(vec(p["field"]), :, 1), atol, rtol)
                elseif cap === :scalar_batch
                    dense = cat((_shtns37_dense(cfg, p["coefficients"][:, k]) for k in axes(p["coefficients"], 2))...; dims=3)
                    Q = place_values(dense, :spectral)
                    proto = place_values(zeros(eltype(p["field"]), size(p["field"])), :spatial)
                    check(synthesis_batch(cfg, Q; prototype_θφ=proto), p["field"], atol, rtol)
                elseif cap in (:sphtor_full, :sphtor_l)
                    S = place_values(_shtns37_dense(cfg, p["S"]), :spectral)
                    T = place_values(_shtns37_dense(cfg, p["T"]), :spectral)
                    got = cap === :sphtor_full ? synthesis_sphtor(cfg, S, T; prototype_θφ=prototype) : synthesis_sphtor_l(cfg, S, T, f["ltr"]; prototype_θφ=prototype)
                    check(got[1], p["Vt"], atol, rtol); check(got[2], p["Vp"], atol, rtol)
                elseif cap === :sphtor_ml
                    S = place_values(reshape(vec(p["S"]), :, 1), :spatial); T = place_values(reshape(vec(p["T"]), :, 1), :spatial)
                    got = synthesis_sphtor_ml(cfg, f["stored_im"], S, T, f["ltr"]); scale = f["fixed_mode_scale"]
                    check(got[1], scale .* reshape(vec(p["Vt"]), :, 1), atol, rtol); check(got[2], scale .* reshape(vec(p["Vp"]), :, 1), atol, rtol)
                elseif cap === :sphtor_batch
                    S = place_values(_shtns37_batch_dense(cfg, p, "S"), :spectral); T = place_values(_shtns37_batch_dense(cfg, p, "T"), :spectral)
                    got = synthesis_sphtor_batch(cfg, S, T); check(got[1], p["Vt"], atol, rtol); check(got[2], p["Vp"], atol, rtol)
                elseif cap in (:qst_full, :qst_l)
                    Q = place_values(_shtns37_dense(cfg, p["Q"]), :spectral); S = place_values(_shtns37_dense(cfg, p["S"]), :spectral); T = place_values(_shtns37_dense(cfg, p["T"]), :spectral)
                    got = cap === :qst_full ? synthesis_qst(cfg, Q, S, T; prototype_θφ=prototype) : synthesis_qst_l(cfg, Q, S, T, f["ltr"]; prototype_θφ=prototype)
                    for (i, name) in enumerate(("Vr", "Vt", "Vp")); check(got[i], p[name], atol, rtol); end
                elseif cap === :qst_ml
                    Q = place_values(reshape(vec(p["Q"]), :, 1), :spatial); S = place_values(reshape(vec(p["S"]), :, 1), :spatial); T = place_values(reshape(vec(p["T"]), :, 1), :spatial)
                    got = synthesis_qst_ml(cfg, f["stored_im"], Q, S, T, f["ltr"]); scale = f["fixed_mode_scale"]
                    for (i, name) in enumerate(("Vr", "Vt", "Vp")); check(got[i], scale .* reshape(vec(p[name]), :, 1), atol, rtol); end
                elseif cap === :qst_batch
                    Q = place_values(_shtns37_batch_dense(cfg, p, "Q"), :spectral); S = place_values(_shtns37_batch_dense(cfg, p, "S"), :spectral); T = place_values(_shtns37_batch_dense(cfg, p, "T"), :spectral)
                    got = synthesis_qst_batch(cfg, Q, S, T); for (i, name) in enumerate(("Vr", "Vt", "Vp")); check(got[i], p[name], atol, rtol); end
                elseif cap === :point
                    Q = place_values(_shtns37_dense(cfg, p["Q"]), :spectral); @test synthesis_point(cfg, Q, f["cost"], f["phi"]) ≈ p["value"][1] atol=atol rtol=rtol
                elseif cap === :point_complex
                    A = place_values(reshape(vec(p["A"]), :, 1), :spatial); @test synthesis_point_cplx(cfg, A, f["cost"], f["phi"]) ≈ p["value"][1] atol=atol rtol=rtol
                elseif cap === :latitude
                    Q = place_values(_shtns37_dense(cfg, p["Q"]), :spectral); check(SH_to_lat(cfg, Q, f["cost"]; nphi=f["nphi"], ltr=f["ltr"], mtr=f["mmax"]), vec(p["values"]), atol, rtol)
                elseif cap === :latitude_complex
                    A = place_values(reshape(vec(p["A"]), :, 1), :spatial); check(SH_to_lat_cplx(cfg, A, f["cost"]; nphi=f["nphi"], ltr=f["ltr"]), vec(p["values"]), atol, rtol)
                elseif cap in (:qst_point, :qst_latitude, :gradient_point)
                    names = cap === :gradient_point ? ("Dr", "S") : ("Q", "S", "T")
                    arrays = map(name -> place_values(_shtns37_dense(cfg, p[name]), :spectral), names)
                    got = cap === :gradient_point ? SH_to_grad_point(cfg, arrays..., f["cost"], f["phi"]) : cap === :qst_point ? SHqst_to_point(cfg, arrays..., f["cost"], f["phi"]) : SHqst_to_lat(cfg, arrays..., f["cost"]; nphi=f["nphi"], ltr=f["ltr"], mtr=f["mmax"])
                    if cap === :qst_latitude; for (i, name) in enumerate(("Vr", "Vt", "Vp")); check(got[i], vec(p[name]), atol, rtol); end; else; @test collect(got) ≈ vec(p["value"]) atol=atol rtol=rtol; end
                elseif cap === :operators
                    Q = place_values(_shtns37_dense(cfg, p["Q"]), :spectral); rct = similar(Q); rdt = similar(Q)
                    SH_mul_mx(CPU(), cfg, vec(p["ct_matrix"]), Q, rct); SH_mul_mx(CPU(), cfg, vec(p["dt_matrix"]), Q, rdt)
                    check(rct, _shtns37_dense(cfg, p["ct_result"]), atol, rtol); check(rdt, _shtns37_dense(cfg, p["dt_result"]), atol, rtol)
                elseif cap === :rotations
                    Q = place_values(_shtns37_dense(cfg, p["Q"]), :spectral); z = similar(Q); y = similar(Q); y90=similar(Q);x90=similar(Q)
                    dist_SH_Zrotate(cfg, Q, f["z_angle"], z); dist_SH_Yrotate(cfg, Q, f["y_angle"], y)
                    dist_SH_Yrotate90(cfg,Q,y90);dist_SH_Xrotate90(cfg,Q,x90)
                    check(z, _shtns37_dense(cfg, p["Z"]), atol, rtol); check(y, _shtns37_dense(cfg, p["Y"]), atol, rtol);check(y90,_shtns37_dense(cfg,p["Y90"]),atol,rtol);check(x90,_shtns37_dense(cfg,p["X90"]),atol,rtol)
                end
            end
        end
    end
end

"""
Run the shared two-rank hardware matrix. This function is always parsed and its
invocation is source-checked; on machines without a matching device per rank it
emits exactly one honest skip and performs no device-math claim.
"""
function run_mpi_gpu_full_parity(vendor::Symbol, array_type::Type,
                                 is_vendor, functional::Bool, devices,
                                 activate_device!, device_of)
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    local_ready = functional && !isempty(devices)
    ready_ranks = MPI.Allreduce(local_ready ? 1 : 0, +, comm)
    if nranks != 2 || ready_ranks != nranks
        @test_skip nranks == 2 && ready_ranks == nranks
        return nothing
    end

    assigned = devices[mod(rank, length(devices)) + 1]
    activate_device!(assigned)
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    compound = vendor === :cuda ?
        Base.get_extension(SHTnsKit, :SHTnsKitParallelCUDAExt) :
        Base.get_extension(SHTnsKit, :SHTnsKitParallelAMDGPUExt)
    adapter = extension._parallel_gpu_adapter(array_type{Float32}(undef, 1))
    @test adapter !== nothing
    @test compound !== nothing

    @testset "actual allocation device cache key" begin
        if length(devices) >= 2
            activate_device!(devices[1])
            first_buffer = array_type{Float32}(undef, 1)
            activate_device!(devices[2])
            second_buffer = array_type{Float32}(undef, 1)
            @test device_of(first_buffer) == devices[1]
            @test device_of(second_buffer) == devices[2]
            extension.parallel_gpu_clear_caches!()
            extension._gpu_awareness(adapter, MPI.COMM_SELF, first_buffer)
            extension._gpu_awareness(adapter, MPI.COMM_SELF, second_buffer)
            @test extension.parallel_gpu_cache_sizes().awareness == 2
        else
            @test_skip length(devices) >= 2
        end
        activate_device!(assigned)
    end

    @testset "ordinary GPU APIs reject before whole-call staging" begin
        cfg = SHTnsKit.create_gauss_config(2, 4; nlon=6)
        field = _mpi_gpu_place(
            array_type, fill(Float32(0.25), cfg.nlat, cfg.nlon), (1,), comm,
        )
        coefficients = _mpi_gpu_place(
            array_type,
            zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1), (2,), comm,
        )
        batch_field = _mpi_gpu_place(
            array_type,
            fill(Float32(0.25), cfg.nlat, cfg.nlon, 2), (1,), comm,
        )
        batch_coefficients = _mpi_gpu_place(
            array_type,
            zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1, 2), (2,), comm,
        )
        dense_coefficients = array_type(
            zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1),
        )
        dense_reduction_output = similar(dense_coefficients)
        packed_coefficients = array_type(zeros(ComplexF32, cfg.nlm))
        distributed_plan = extension.create_distributed_spectral_plan(
            cfg.lmax, cfg.mmax, comm; mres=cfg.mres,
        )
        distributed_plan_2d = extension.create_distributed_spectral_plan_2d(
            cfg.lmax, cfg.mmax, comm;
            p_l=1, p_m=nranks, mres=cfg.mres,
        )
        public_calls = (
            cfg_scalar=(:analysis, () -> SHTnsKit.analysis(cfg, field)),
            cfg_vector=(:analysis_sphtor,
                () -> SHTnsKit.analysis_sphtor(cfg, field, field)),
            cfg_qst=(:analysis_qst,
                () -> SHTnsKit.analysis_qst(cfg, field, field, field)),
            degree_restricted=(:analysis_qst_l,
                () -> SHTnsKit.analysis_qst_l(
                    cfg, field, field, field, cfg.lmax,
                )),
            order_restricted=(:analysis_qst_ml,
                () -> SHTnsKit.analysis_qst_ml(
                    cfg, 0, coefficients, coefficients, coefficients, cfg.lmax,
                )),
            packed_axisymmetric=(:analysis_packed,
                () -> SHTnsKit.analysis_packed(cfg, field)),
            batch_bang=(:analysis_batch!,
                () -> SHTnsKit.analysis_batch!(
                    cfg, batch_coefficients, batch_field,
                )),
            local_evaluation_gradient=(:SH_to_grad_point,
                () -> SHTnsKit.SH_to_grad_point(
                    cfg, coefficients, coefficients, 0.25f0, 0.5f0,
                )),
            diagonal_operator=(:divergence_from_spheroidal!,
                () -> SHTnsKit.divergence_from_spheroidal!(
                    cfg, similar(coefficients), coefficients,
                )),
            distributed_operator=(:dist_scalar_laplacian!,
                () -> SHTnsKit.dist_scalar_laplacian!(
                    cfg, similar(field), field,
                )),
            rotation=(:dist_SH_rotate_euler,
                () -> SHTnsKit.dist_SH_rotate_euler(
                    cfg, coefficients, 0.1f0, 0.2f0, 0.3f0,
                    similar(coefficients),
                )),
            packed_rotation=(:dist_SH_Yrotate_packed,
                () -> SHTnsKit.dist_SH_Yrotate_packed(
                    cfg, packed_coefficients, 0.2f0;
                    prototype_lm=coefficients,
                )),
            diagnostics=(:energy_vector_l_spectrum,
                () -> SHTnsKit.energy_vector_l_spectrum(
                    cfg, coefficients, coefficients,
                )),
            diagnostics_scalar_total=(:enstrophy,
                () -> SHTnsKit.enstrophy(cfg, coefficients)),
            diagnostics_vector_total=(:energy_vector,
                () -> SHTnsKit.energy_vector(
                    cfg, coefficients, coefficients,
                )),
            distributed_storage_reduce=(:distributed_spectral_reduce!,
                () -> extension.distributed_spectral_reduce!(
                    distributed_plan, dense_coefficients,
                    dense_reduction_output,
                )),
            distributed_storage_1d=(:dist_analysis_distributed,
                () -> extension.dist_analysis_distributed(
                    cfg, field; plan=distributed_plan,
                )),
            distributed_storage_2d=(:dist_analysis_distributed_2d,
                () -> extension.dist_analysis_distributed_2d(
                    cfg, field; plan=distributed_plan_2d,
                )),
            distributed_compatibility=(:dist_analysis,
                () -> SHTnsKit.dist_analysis(cfg, field)),
            dense_compatibility=(:dist_synthesis_qst_dense,
                () -> SHTnsKit.dist_synthesis_qst(
                    cfg, dense_coefficients, dense_coefficients,
                    dense_coefficients; prototype_θφ=field,
                )),
        )
        @test keys(public_calls) == keys(MPI_GPU_ORDINARY_EARLY_ERROR_FAMILIES)
        for (family, (operation, call)) in pairs(public_calls)
            before_stats = extension.parallel_gpu_stats()
            before_payloads = map(
                value -> Array(parent(value)),
                (field, coefficients, batch_field, batch_coefficients),
            )
            caught = try
                call()
                nothing
            catch error
                error
            end
            @test caught isa SHTnsKit.BackendUnavailableError
            if caught isa SHTnsKit.BackendUnavailableError
                @test caught.operation === operation
            end
            @test extension.parallel_gpu_stats() == before_stats
            @test map(value -> Array(parent(value)),
                      (field, coefficients, batch_field, batch_coefficients)) ==
                  before_payloads
            MPI.Barrier(comm)
        end
        close(distributed_plan_2d)
    end

    @testset "staged collectives use vendor bulk copies" begin
        # Pinned host stages are views into reusable buffers. The vendor
        # transfers must copy through their parent Array rather than fall back
        # to element-wise copies that index device memory.
        staged_adapter = extension.ParallelGPUAdapter(
            Symbol(vendor, :_staged_collectives), adapter.matches,
            adapter.array_type, adapter.device, adapter.with_device,
            _ -> false, adapter.synchronize, adapter.allocate_pinned,
            adapter.device_to_host!, adapter.host_to_device!,
        )
        extension.parallel_gpu_clear_caches!()
        buffer = array_type(Float32[1, 2, 3])
        @test extension.allreduce!(
            buffer, +, comm; adapter=staged_adapter,
        ) === buffer
        @test Array(buffer) == nranks .* Float32[1, 2, 3]
        matrix = array_type(reshape(Float32.(1:6), 2, 3))
        result = compound._internal_staged_vendor_call_with_adapter(
            staged_adapter, comm, :staged_vendor_bulk_copies,
            host -> (host .*= 2; host .+ 1), matrix; mutated=(1,),
        )
        @test Array(matrix) == 2 .* reshape(Float32.(1:6), 2, 3)
        @test is_vendor(result)
        @test Array(result) == 2 .* reshape(Float32.(1:6), 2, 3) .+ 1
        # exchange! requires the registered vendor name; clearing the caches
        # keeps this adapter's awareness verdict from reaching later calls.
        extension.parallel_gpu_clear_caches!()
        exchange_adapter = extension.ParallelGPUAdapter(
            adapter.name, adapter.matches, adapter.array_type, adapter.device,
            adapter.with_device, _ -> false, adapter.synchronize,
            adapter.allocate_pinned, adapter.device_to_host!,
            adapter.host_to_device!,
        )
        send = array_type(Float32.(100rank .+ (1:nranks)))
        receive = array_type(zeros(Float32, nranks))
        extension.exchange!(
            send, receive, comm; adapter=exchange_adapter,
            collective=(send, receive, comm) -> MPI.Alltoall!(
                MPI.UBuffer(send, 1), MPI.UBuffer(receive, 1), comm,
            ),
        )
        @test Array(receive) == Float32.(100 .* (0:nranks-1) .+ rank .+ 1)
        extension.parallel_gpu_clear_caches!()
    end

    for RT in (Float32, Float64)
        CT = Complex{RT}
        tol = RT === Float32 ? RT(8e-4) : RT(8e-11)
        cfg = SHTnsKit.create_gauss_config(3, 8; nlon=8)

        @testset "native scalar/vector/QST transpose nonzero numerics" begin
            @test cfg.norm === :orthonormal
            @test cfg.cs_phase
            @test !cfg.real_norm
            staged_before = extension.parallel_gpu_stats().staged_calls
            forced_staged_adapter = extension.ParallelGPUAdapter(
                Symbol(vendor, :_native_forced_staged), adapter.matches,
                adapter.array_type, adapter.device, adapter.with_device,
                _ -> false, adapter.synchronize, adapter.allocate_pinned,
                adapter.device_to_host!, adapter.host_to_device!,
            )
            for nlev in (1, 2, 5)
                plan = SHTnsKit.DistTransposePlan(
                    cfg; comm, nlev, array_type, real_type=RT,
                    with_vector=true,
                )
                refs = _mpi_gpu_native_references(cfg, RT, nlev)
                Vr = SHTnsKit.allocate_spatial(plan)
                Vt = SHTnsKit.allocate_spatial(plan)
                Vp = SHTnsKit.allocate_spatial(plan)
                _mpi_gpu_fill_native_spatial!(Vr, refs.Vr)
                _mpi_gpu_fill_native_spatial!(Vt, refs.Vt)
                _mpi_gpu_fill_native_spatial!(Vp, refs.Vp)
                Q = SHTnsKit.allocate_spectral(plan)
                S = SHTnsKit.allocate_spectral(plan)
                T = SHTnsKit.allocate_spectral(plan)

                @test SHTnsKit.dist_analysis!(plan, Q, Vr) === Q
                scalar_error = _mpi_gpu_native_spectral_error(
                    Q, refs.Qanalysis, plan.m_local,
                )
                @test MPI.Allreduce(scalar_error, MPI.MAX, comm) <= tol
                @test SHTnsKit.dist_analysis_sphtor!(
                    plan, S, T, Vt, Vp,
                ) === (S, T)
                vector_error = max(
                    _mpi_gpu_native_spectral_error(
                        S, refs.Sanalysis, plan.m_local,
                    ),
                    _mpi_gpu_native_spectral_error(
                        T, refs.Tanalysis, plan.m_local,
                    ),
                )
                @test MPI.Allreduce(vector_error, MPI.MAX, comm) <= 4tol
                @test SHTnsKit.dist_analysis_qst!(
                    plan, Q, S, T, Vr, Vt, Vp,
                ) === (Q, S, T)
                qst_error = max(
                    _mpi_gpu_native_spectral_error(
                        Q, refs.Qanalysis, plan.m_local,
                    ),
                    _mpi_gpu_native_spectral_error(
                        S, refs.Sanalysis, plan.m_local,
                    ),
                    _mpi_gpu_native_spectral_error(
                        T, refs.Tanalysis, plan.m_local,
                    ),
                )
                @test MPI.Allreduce(qst_error, MPI.MAX, comm) <= 4tol

                _mpi_gpu_fill_native_spectral!(Q, refs.Q, plan.m_local)
                _mpi_gpu_fill_native_spectral!(S, refs.S, plan.m_local)
                _mpi_gpu_fill_native_spectral!(T, refs.T, plan.m_local)
                @test SHTnsKit.dist_synthesis!(plan, Vr, Q) === Vr
                scalar_error = _mpi_gpu_native_spatial_error(Vr, refs.Vr)
                @test MPI.Allreduce(scalar_error, MPI.MAX, comm) <= tol
                @test SHTnsKit.dist_synthesis_sphtor!(
                    plan, Vt, Vp, S, T,
                ) === (Vt, Vp)
                vector_error = max(
                    _mpi_gpu_native_spatial_error(Vt, refs.Vt),
                    _mpi_gpu_native_spatial_error(Vp, refs.Vp),
                )
                @test MPI.Allreduce(vector_error, MPI.MAX, comm) <= 4tol
                @test SHTnsKit.dist_synthesis_qst!(
                    plan, Vr, Vt, Vp, Q, S, T,
                ) === (Vr, Vt, Vp)
                qst_error = max(
                    _mpi_gpu_native_spatial_error(Vr, refs.Vr),
                    _mpi_gpu_native_spatial_error(Vt, refs.Vt),
                    _mpi_gpu_native_spatial_error(Vp, refs.Vp),
                )
                @test MPI.Allreduce(qst_error, MPI.MAX, comm) <= 4tol
                _mpi_gpu_assert_resident((Vr, Vt, Vp, Q, S, T), is_vendor)

                forced_before = extension.parallel_gpu_stats().staged_calls
                _mpi_gpu_fill_native_spatial!(Vr, refs.Vr)
                _mpi_gpu_fill_native_spatial!(Vt, refs.Vt)
                _mpi_gpu_fill_native_spatial!(Vp, refs.Vp)
                extension._dist_transpose_gpu_analysis!(
                    forced_staged_adapter, plan, Q, Vr,
                )
                extension._dist_transpose_gpu_vector_analysis!(
                    forced_staged_adapter, plan, S, T, Vt, Vp,
                )
                @test MPI.Allreduce(max(
                    _mpi_gpu_native_spectral_error(
                        Q, refs.Qanalysis, plan.m_local,
                    ),
                    _mpi_gpu_native_spectral_error(
                        S, refs.Sanalysis, plan.m_local,
                    ),
                    _mpi_gpu_native_spectral_error(
                        T, refs.Tanalysis, plan.m_local,
                    ),
                ), MPI.MAX, comm) <= 4tol
                _mpi_gpu_fill_native_spectral!(Q, refs.Q, plan.m_local)
                _mpi_gpu_fill_native_spectral!(S, refs.S, plan.m_local)
                _mpi_gpu_fill_native_spectral!(T, refs.T, plan.m_local)
                extension._dist_transpose_gpu_synthesis!(
                    forced_staged_adapter, plan, Vr, Q,
                )
                extension._dist_transpose_gpu_vector_synthesis!(
                    forced_staged_adapter, plan, Vt, Vp, S, T,
                )
                @test MPI.Allreduce(max(
                    _mpi_gpu_native_spatial_error(Vr, refs.Vr),
                    _mpi_gpu_native_spatial_error(Vt, refs.Vt),
                    _mpi_gpu_native_spatial_error(Vp, refs.Vp),
                ), MPI.MAX, comm) <= 4tol
                @test extension.parallel_gpu_stats().staged_calls ==
                      forced_before + 6
            end
            @test extension.parallel_gpu_stats().staged_calls >= staged_before + 18
        end

        @testset "repeated-plan cache and residency" begin
            plan = SHTnsKit.DistTransposePlan(
                cfg; comm, nlev=2, array_type, real_type=RT,
            )
            input = SHTnsKit.allocate_spatial(plan)
            output = SHTnsKit.allocate_spectral(plan)
            fill!(parent(input), RT(0.125))
            for _ in 1:3
                @test SHTnsKit.dist_analysis!(plan, output, input) === output
            end
            _mpi_gpu_assert_resident((input, output), is_vendor)
        end
    end
    return nothing
end
# Exercise the real distributed FFT staging boundary without requiring a GPU.
# Cache pressure on a subcommunicator must not change the MPI contexts used by
# a still-live plan on the parent communicator.
function test_mpi_gpu_native_cache_collectives(extension)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    adapter = extension.ParallelGPUAdapter(
        :mock_native_cache_collectives,
        value -> value isa Array, _ -> Array, _ -> 0,
        (f, _device) -> f(), _ -> false, _ -> nothing,
        (T, n) -> Vector{T}(undef, n),
        (host, device) -> copyto!(host, device),
        (device, host) -> copyto!(device, host),
    )
    cfg = SHTnsKit.create_gauss_config(2, 4; nlon=6)
    plan = SHTnsKit.DistTransposePlan(cfg; comm, nlev=2)
    input = SHTnsKit.allocate_spatial(plan)
    original = reshape(
        sin.(collect(1:length(parent(input))) .+ rank), size(parent(input)),
    )
    copyto!(parent(input), original)
    mul!(plan.F_buf, plan.fft_plan, input)
    expected = copy(parent(plan.F_buf))

    @testset "native host cache survives rank-local $disruption" for disruption in
            (:clear, :eviction)
        extension.parallel_gpu_clear_caches!()
        extension._gpu_transpose_forward!(adapter, plan, plan.F_buf, input)
        @test parent(plan.F_buf) ≈ expected
        if rank == 0
            if disruption === :clear
                extension.parallel_gpu_clear_caches!()
            else
                # Keep all local plans alive so this deterministically exceeds
                # the bounded cache capacity on rank zero only.
                local_plans = [SHTnsKit.DistTransposePlan(
                    cfg; comm=MPI.COMM_SELF, nlev=2,
                ) for _ in 1:8]
                for local_plan in local_plans
                    local_input = SHTnsKit.allocate_spatial(local_plan)
                    fill!(parent(local_input), 1)
                    extension._gpu_transpose_forward!(
                        adapter, local_plan, local_plan.F_buf, local_input,
                    )
                end
            end
        end
        MPI.Barrier(comm)
        # Rank zero rebuilds while peers retain their cached mirror. Both the
        # forward and inverse transforms must still use matching communicators.
        extension._gpu_transpose_forward!(adapter, plan, plan.F_buf, input)
        @test parent(plan.F_buf) ≈ expected
        extension._gpu_transpose_inverse!(adapter, plan, input, plan.F_buf)
        @test parent(input) ≈ original
    end
    extension.parallel_gpu_clear_caches!()
    return nothing
end

function test_mpi_gpu_policy(extension)
    @test isdefined(extension, :ParallelGPUAdapter)
    @test isdefined(extension, :exchange!)
    @test isdefined(extension, :allreduce!)
    @test isdefined(extension, :parallel_gpu_stats)
    @test isdefined(extension, :parallel_gpu_clear_caches!)
    @test isdefined(extension, :parallel_gpu_cache_sizes)

    aware_calls = Ref(0)
    sync_calls = Ref(0)
    host_allocations = Ref(0)
    host_to_device = Ref(0)
    device_to_host = Ref(0)
    direct_collectives = Ref(0)
    staged_collectives = Ref(0)

    adapter = extension.ParallelGPUAdapter(
        :mock,
        value -> value isa MockMPIArray,
        _ -> MockMPIArray,
        _ -> 7,
        _ -> (aware_calls[] += 1; false),
        _ -> (sync_calls[] += 1),
        (T, n) -> (host_allocations[] += 1; Vector{T}(undef, n)),
        (host, device) -> (device_to_host[] += 1; copyto!(host, device)),
        (device, host) -> (host_to_device[] += 1; copyto!(device, host)),
    )

    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(2)
    buffer = MockMPIArray(Float32[1, 2, 3])
    comm = Ref(:subgroup_a)
    collective = function(host, _op, _comm)
        staged_collectives[] += 1
        host .*= 2
        return host
    end
    extension.allreduce!(buffer, +, comm; adapter, collective)
    extension.allreduce!(buffer, +, comm; adapter, collective)
    @test buffer.data == Float32[4, 8, 12]
    @test aware_calls[] == 1
    @test host_allocations[] == 1
    @test device_to_host[] == 2
    @test host_to_device[] == 2
    @test sync_calls[] == 6
    @test staged_collectives[] == 2
    @test extension.parallel_gpu_cache_sizes().staging == 1

    # Every host-to-device result transfer uses a bounded pinned snapshot,
    # including zero-length results, and reuses it on repeated calls.
    extension.parallel_gpu_clear_caches!()
    pin_lengths = Int[]
    result_adapter = extension.ParallelGPUAdapter(
        :mock_pinned_results, adapter.matches, adapter.array_type,
        adapter.device, adapter.with_device, adapter.gpu_aware,
        adapter.synchronize,
        (T, n) -> (push!(pin_lengths, n); Vector{T}(undef, n)),
        adapter.device_to_host!, adapter.host_to_device!,
    )
    for _ in 1:2
        restored = extension._staged_gpu_call(
            result_adapter, :pinned_result, comm,
            host -> Float32[sum(host), length(host)], buffer;
            validate_storage=false,
        )
        @test restored.data == Float32[24, 3]
    end
    empty_device = MockMPIArray(Float32[])
    empty_restored = extension._staged_gpu_call(
        result_adapter, :pinned_empty_result, comm,
        _host -> Float32[], empty_device; validate_storage=false,
    )
    @test isempty(empty_restored)
    @test count(==(3), pin_lengths) == 1
    @test count(==(2), pin_lengths) == 1
    @test count(==(0), pin_lengths) <= 2

    # Out-of-place result restoration must not wait on staging reservations
    # still held by the same call.  This includes one-slot caches and a full
    # default-capacity wave of concurrent calls.
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(1)
    one_slot_task = Threads.@spawn extension._staged_gpu_call(
        result_adapter, :one_slot_result, comm,
        host -> Float64.(host), MockMPIArray(Float32[2, 4]);
        validate_storage=false,
    )
    one_slot_status = Base.timedwait(() -> istaskdone(one_slot_task), 2.0)
    if one_slot_status !== :ok
        extension.parallel_gpu_cache_limit!(2)
        notify(extension._GPU_STAGING_AVAILABLE)
    end
    one_slot_result = fetch(one_slot_task)
    @test one_slot_status === :ok
    @test one_slot_result.data == Float64[2, 4]

    # The cache limit bounds retained idle buffers, not a call's live working
    # set. Two distinct inputs and exchange send/receive must complete at limit
    # one; duplicate inputs share one reservation.
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(1)
    working_allocations = Threads.Atomic{Int}(0)
    working_adapter = extension.ParallelGPUAdapter(
        :mock_working_set, adapter.matches, adapter.array_type,
        adapter.device, adapter.with_device, adapter.gpu_aware,
        adapter.synchronize,
        (T, n) -> begin
            Threads.atomic_add!(working_allocations, 1)
            Vector{T}(undef, n)
        end,
        adapter.device_to_host!, adapter.host_to_device!,
    )
    working_first = MockMPIArray(Float32[1, 2])
    working_second = MockMPIArray(Float32[3, 4])
    two_input_task = Threads.@spawn try
        extension._staged_gpu_call(
            working_adapter, :limit_one_two_inputs, comm,
            (first, second) -> sum(first .+ second),
            working_first, working_second; validate_storage=false,
        )
    catch error
        error
    end
    two_input_status = Base.timedwait(() -> istaskdone(two_input_task), 2.0)
    if two_input_status !== :ok
        extension.parallel_gpu_cache_limit!(3)
        notify(extension._GPU_STAGING_AVAILABLE)
    end
    two_input_result = fetch(two_input_task)
    @test two_input_status === :ok
    @test two_input_result == 10
    @test working_allocations[] == 2
    @test length(extension._GPU_STAGING) <= 1

    extension.parallel_gpu_clear_caches!()
    working_allocations[] = 0
    extension.parallel_gpu_cache_limit!(1)
    @test extension._staged_gpu_call(
        working_adapter, :limit_one_duplicate_input, comm,
        (first, duplicate) -> sum(first .+ duplicate),
        working_first, working_first; validate_storage=false,
    ) == 6
    @test working_allocations[] == 1

    extension.parallel_gpu_clear_caches!()
    working_allocations[] = 0
    exchange_result = try
        extension.exchange!(
            working_first, working_second, MPI.COMM_SELF;
            adapter=working_adapter,
            collective=(send, receive, _comm) -> copyto!(receive, send),
        )
    catch error
        error
    end
    @test exchange_result === working_second
    @test working_second.data == working_first.data
    @test working_allocations[] == 2
    @test length(extension._GPU_STAGING) <= 1

    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(8)
    working_allocations[] = 0
    working_barrier = Threads.Atomic{Int}(0)
    concurrent_working_values = [
        (MockMPIArray(Float32[2n - 1]), MockMPIArray(Float32[2n]))
        for n in 1:2
    ]
    concurrent_working_tasks = [Threads.@spawn extension._staged_gpu_call(
        working_adapter, :concurrent_multi_input, comm,
        (first, second) -> begin
            Threads.atomic_add!(working_barrier, 1)
            while working_barrier[] < 2
                yield()
            end
            first[1] + second[1]
        end,
        values...; validate_storage=false,
    ) for values in concurrent_working_values]
    @test fetch.(concurrent_working_tasks) == Float32[3, 7]
    @test working_allocations[] == 4
    @test length(extension._GPU_STAGING) <= 8

    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(8)
    result_barrier_count = Threads.Atomic{Int}(0)
    concurrent_result_inputs = [MockMPIArray(Float32[n, n + 1]) for n in 1:8]
    concurrent_result_tasks = [Threads.@spawn extension._staged_gpu_call(
        result_adapter, :full_cache_result, comm,
        host -> begin
            Threads.atomic_add!(result_barrier_count, 1)
            while result_barrier_count[] < 8
                yield()
            end
            Float64.(host)
        end,
        input; validate_storage=false,
    ) for input in concurrent_result_inputs]
    concurrent_result_status = Base.timedwait(
        () -> all(istaskdone, concurrent_result_tasks), 2.0,
    )
    if concurrent_result_status !== :ok
        extension.parallel_gpu_cache_limit!(16)
        for _ in 1:16
            notify(extension._GPU_STAGING_AVAILABLE)
        end
    end
    concurrent_results = fetch.(concurrent_result_tasks)
    @test concurrent_result_status === :ok
    @test all(enumerate(concurrent_results)) do (n, result)
        result.data == Float64[n, n + 1]
    end
    @test extension.parallel_gpu_cache_sizes().staging <= 8
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(8)

    # Device-aware cache keys follow the allocation behind views, not the
    # current task device.  Reusing an allocation after its fake device changes
    # must create independent awareness and pinned-staging entries.
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(8)
    multi = MockMultiDeviceArray(reshape(Float32.(1:4), 2, 2), 1)
    multi_view = view(multi, :, :)
    current_device = Ref(99)
    multi_host_allocations = Ref(0)
    multi_adapter = extension.ParallelGPUAdapter(
        :mock_multidevice,
        value -> extension._parallel_root_buffer(value) isa MockMultiDeviceArray,
        _ -> MockMultiDeviceArray,
        value -> extension._parallel_root_buffer(value).device,
        (f, device) -> begin
            previous = MOCK_CURRENT_DEVICE[]
            try
                MOCK_CURRENT_DEVICE[] = device
                f()
            finally
                MOCK_CURRENT_DEVICE[] = previous
            end
        end,
        _ -> false,
        _ -> nothing,
        (T, n) -> (multi_host_allocations[] += 1; Vector{T}(undef, n)),
        copyto!, copyto!,
    )
    multi_comm = Ref(:multidevice_subgroup)

    # Pinned allocation is a device operation too: it must observe the
    # allocation owner's device and restore the caller even when it throws.
    pin_devices = Int[]
    pin_failure = Ref(false)
    pin_context_adapter = extension.ParallelGPUAdapter(
        :mock_pin_context, multi_adapter.matches, multi_adapter.array_type,
        multi_adapter.device, multi_adapter.with_device,
        multi_adapter.gpu_aware, multi_adapter.synchronize,
        (T, n) -> begin
            push!(pin_devices, MOCK_CURRENT_DEVICE[])
            pin_failure[] && error("pin allocation failed")
            Vector{T}(undef, n)
        end,
        multi_adapter.device_to_host!, multi_adapter.host_to_device!,
    )
    extension.parallel_gpu_clear_caches!()
    MOCK_CURRENT_DEVICE[] = 99
    extension.allreduce!(
        multi, +, multi_comm; adapter=pin_context_adapter,
        collective=(host, _op, _comm) -> host,
    )
    @test pin_devices == [1]
    @test MOCK_CURRENT_DEVICE[] == 99
    extension.parallel_gpu_clear_caches!()
    pin_failure[] = true
    @test_throws ErrorException extension.allreduce!(
        multi, +, Ref(:pin_failure); adapter=pin_context_adapter,
        collective=(host, _op, _comm) -> host,
    )
    @test last(pin_devices) == 1
    @test MOCK_CURRENT_DEVICE[] == 99
    pin_failure[] = false

    # Native CPU mirror plans are single-flight, pinned, bounded, observable,
    # and recover after a failed first construction without stranding waiters.
    @test isdefined(extension, :_with_gpu_transpose_host_entry)
    if isdefined(extension, :_with_gpu_transpose_host_entry)
        native_cfg = SHTnsKit.create_gauss_config(2, 4; nlon=6)
        native_plan = SHTnsKit.DistTransposePlan(
            native_cfg; comm=MPI.COMM_SELF, nlev=1,
        )
        native_pins = Threads.Atomic{Int}(0)
        native_pin_devices = Int[]
        native_build_entered = Base.Event()
        native_build_release = Base.Event()
        native_adapter = extension.ParallelGPUAdapter(
            :mock_native_host_plan, multi_adapter.matches,
            multi_adapter.array_type, multi_adapter.device,
            multi_adapter.with_device, multi_adapter.gpu_aware,
            multi_adapter.synchronize,
            (T, n) -> begin
                allocation = Threads.atomic_add!(native_pins, 1)
                push!(native_pin_devices, MOCK_CURRENT_DEVICE[])
                if allocation == 0
                    notify(native_build_entered)
                    wait(native_build_release)
                end
                Vector{T}(undef, n)
            end,
            multi_adapter.device_to_host!, multi_adapter.host_to_device!,
        )
        extension.parallel_gpu_clear_caches!()
        start_native = Base.Event()
        native_ready = Channel{Nothing}(8)
        native_tasks = [Threads.@spawn begin
            put!(native_ready, nothing)
            wait(start_native)
            extension._with_gpu_transpose_host_entry(
                native_adapter, native_plan, multi,
            ) do entry
                (objectid(entry.fft_plan), objectid(entry.spatial),
                 objectid(entry.spectral))
            end
        end for _ in 1:8]
        foreach(_ -> take!(native_ready), 1:8)
        notify(start_native)
        wait(native_build_entered)
        for _ in 1:64
            yield()
        end
        GC.gc(true)
        notify(native_build_release)
        native_ids = fetch.(native_tasks)
        # Repeated collection/concurrent acquisition must never invalidate the
        # weak cache owner while this logical plan remains live.
        for _ in 2:100
            GC.gc(true)
            round_start = Base.Event()
            round_tasks = [Threads.@spawn begin
                wait(round_start)
                extension._with_gpu_transpose_host_entry(
                    native_adapter, native_plan, multi,
                ) do entry
                    (objectid(entry.fft_plan), objectid(entry.spatial),
                     objectid(entry.spectral))
                end
            end for _ in 1:8]
            notify(round_start)
            append!(native_ids, fetch.(round_tasks))
        end
        @test all(==(first(native_ids)), native_ids)
        @test native_pins[] == 2
        @test native_pin_devices == [1, 1]
        @test MOCK_CURRENT_DEVICE[] == 99
        @test extension.parallel_gpu_cache_sizes().native_host_plans == 1

        fail_next_pin = Ref(true)
        failing_native_adapter = extension.ParallelGPUAdapter(
            :mock_native_host_failure, multi_adapter.matches,
            multi_adapter.array_type, multi_adapter.device,
            multi_adapter.with_device, multi_adapter.gpu_aware,
            multi_adapter.synchronize,
            (T, n) -> begin
                if fail_next_pin[]
                    fail_next_pin[] = false
                    error("native pin failed")
                end
                Vector{T}(undef, n)
            end,
            multi_adapter.device_to_host!, multi_adapter.host_to_device!,
        )
        extension.parallel_gpu_clear_caches!()
        @test_throws ErrorException extension._with_gpu_transpose_host_entry(
            _ -> nothing, failing_native_adapter, native_plan, multi,
        )
        @test extension._with_gpu_transpose_host_entry(
            _ -> :recovered, failing_native_adapter, native_plan, multi,
        ) === :recovered

        extension.parallel_gpu_clear_caches!()
        live_native_plans = [SHTnsKit.DistTransposePlan(
            native_cfg; comm=MPI.COMM_SELF, nlev=1,
        ) for _ in 1:10]
        for plan in live_native_plans
            extension._with_gpu_transpose_host_entry(
                _ -> nothing, native_adapter, plan, multi,
            )
            @test extension.parallel_gpu_cache_sizes().native_host_plans <= 8
        end
        extension.parallel_gpu_clear_caches!()
        @test extension.parallel_gpu_cache_sizes().native_host_plans == 0

        # Clear never drops active or in-progress native entries, and an
        # in-progress builder remains the single publisher for its key.
        extension.parallel_gpu_clear_caches!()
        native_entered = Base.Event()
        native_release = Base.Event()
        active_native = Threads.@spawn extension._with_gpu_transpose_host_entry(
            native_adapter, native_plan, multi,
        ) do _entry
            notify(native_entered)
            wait(native_release)
        end
        wait(native_entered)
        extension.parallel_gpu_clear_caches!()
        @test length(extension._GPU_TRANSPOSE_HOST) == 1
        notify(native_release)
        fetch(active_native)
        @test length(extension._GPU_TRANSPOSE_HOST) == 0

        pending_native_allocations = Threads.Atomic{Int}(0)
        pending_native_entered = Base.Event()
        pending_native_release = Base.Event()
        pending_native_adapter = extension.ParallelGPUAdapter(
            :mock_pending_native, multi_adapter.matches,
            multi_adapter.array_type, multi_adapter.device,
            multi_adapter.with_device, multi_adapter.gpu_aware,
            multi_adapter.synchronize,
            (T, n) -> begin
                allocation = Threads.atomic_add!(pending_native_allocations, 1)
                if allocation == 0
                    notify(pending_native_entered)
                    wait(pending_native_release)
                end
                Vector{T}(undef, n)
            end,
            multi_adapter.device_to_host!, multi_adapter.host_to_device!,
        )
        first_pending_native = Threads.@spawn extension._with_gpu_transpose_host_entry(
            _ -> :first, pending_native_adapter, native_plan, multi,
        )
        wait(pending_native_entered)
        extension.parallel_gpu_clear_caches!()
        second_pending_native = Threads.@spawn extension._with_gpu_transpose_host_entry(
            _ -> :second, pending_native_adapter, native_plan, multi,
        )
        yield()
        notify(pending_native_release)
        @test fetch(first_pending_native) === :first
        @test fetch(second_pending_native) === :second
        @test pending_native_allocations[] == 2
        @test length(extension._GPU_TRANSPOSE_HOST) == 0
    end

    test_mpi_gpu_native_cache_collectives(extension)

    # Every sync/copy must run on the physical device owning that buffer while
    # host MPI/CPU callbacks run on the caller's original current device.
    context_events = Any[]
    context_aware = Ref(false)
    context_copy_error = Ref(false)
    context_adapter = extension.ParallelGPUAdapter(
        :mock_device_context,
        multi_adapter.matches, multi_adapter.array_type, multi_adapter.device,
        (f, device) -> begin
            previous = MOCK_CURRENT_DEVICE[]
            push!(context_events, (:enter, device, previous))
            try
                MOCK_CURRENT_DEVICE[] = device
                f()
            finally
                MOCK_CURRENT_DEVICE[] = previous
                push!(context_events, (:exit, device, previous))
            end
        end,
        _ -> context_aware[],
        value -> push!(context_events, (
            :sync, MOCK_CURRENT_DEVICE[], multi_adapter.device(value),
        )),
        (T, n) -> Vector{T}(undef, n),
        (host, value) -> begin
            push!(context_events, (
                :device_to_host, MOCK_CURRENT_DEVICE[],
                multi_adapter.device(value),
            ))
            context_copy_error[] && error("device copy failed")
            copyto!(host, value)
        end,
        (value, host) -> begin
            push!(context_events, (
                :host_to_device, MOCK_CURRENT_DEVICE[],
                multi_adapter.device(value),
            ))
            context_copy_error[] && error("device copy failed")
            copyto!(value, host)
        end,
    )
    context_value = MockMultiDeviceArray(Float32[1, 2], 1)
    context_receive = MockMultiDeviceArray(zeros(Float32, 2), 1)
    MOCK_CURRENT_DEVICE[] = 2

    function assert_context_events(events, callback_device)
        device_events = filter(
            event -> first(event) in (:sync, :device_to_host, :host_to_device),
            events,
        )
        @test !isempty(device_events)
        @test all(event -> event[2] == event[3] == 1, device_events)
        @test all(event -> event[2] == callback_device,
                  filter(event -> first(event) == :host, events))
        @test MOCK_CURRENT_DEVICE[] == 2
    end

    extension._staged_gpu_call(
        context_adapter, :device_context_staged_math, MPI.COMM_SELF,
        host -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            host .+= 1
            host
        end,
        context_value; mutated=(1,), validate_storage=false,
    )
    assert_context_events(context_events, 2)

    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(host, _op, _comm) -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            host
        end,
    )
    assert_context_events(context_events, 2)

    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    extension.exchange!(
        context_value, context_receive, MPI.COMM_SELF; adapter=context_adapter,
        collective=(send, receive, _comm) -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            copyto!(receive, send)
        end,
    )
    assert_context_events(context_events, 2)

    context_aware[] = true
    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(device, _op, _comm) -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            device
        end,
    )
    assert_context_events(context_events, 1)

    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    extension.exchange!(
        context_value, context_receive, MPI.COMM_SELF; adapter=context_adapter,
        collective=(send, receive, _comm) -> begin
            push!(context_events, (:host, MOCK_CURRENT_DEVICE[]))
            copyto!(receive, send)
        end,
    )
    assert_context_events(context_events, 1)

    # Device scopes restore caller state after copy and MPI exceptions.
    context_aware[] = false

    # Native vendor dispatch itself must select the allocation's physical
    # device and restore the caller's device on success and failure. This
    # covers vendor kernels as well as their direct PencilFFT calls.
    native_scope_adapter = extension.ParallelGPUAdapter(
        :mock_native_scope,
        context_adapter.matches, context_adapter.array_type,
        context_adapter.device, context_adapter.with_device,
        context_adapter.gpu_aware, context_adapter.synchronize,
        context_adapter.allocate_pinned, context_adapter.device_to_host!,
        context_adapter.host_to_device!,
    )
    MOCK_CURRENT_DEVICE[] = 2
    native_callback_device = Ref(0)
    @test extension._dist_transpose_gpu_analysis!(
        native_scope_adapter,
        () -> (native_callback_device[] = MOCK_CURRENT_DEVICE[]),
        context_receive, context_value,
    ) === context_receive
    @test native_callback_device[] == 1
    @test MOCK_CURRENT_DEVICE[] == 2
    @test_throws ErrorException extension._dist_transpose_gpu_analysis!(
        native_scope_adapter, () -> error("native kernel failed"),
        context_receive, context_value,
    )
    @test MOCK_CURRENT_DEVICE[] == 2
    context_copy_error[] = true
    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    @test_throws ErrorException extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(host, _op, _comm) -> host,
    )
    @test MOCK_CURRENT_DEVICE[] == 2
    context_copy_error[] = false
    empty!(context_events)
    extension.parallel_gpu_clear_caches!()
    @test_throws ErrorException extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(_host, _op, _comm) -> error("MPI failed"),
    )
    @test MOCK_CURRENT_DEVICE[] == 2

    context_aware[] = true
    extension.parallel_gpu_clear_caches!()
    @test_throws ErrorException extension.allreduce!(
        context_value, +, MPI.COMM_SELF; adapter=context_adapter,
        collective=(_device, _op, _comm) -> error("direct MPI failed"),
    )
    @test MOCK_CURRENT_DEVICE[] == 2
    extension.parallel_gpu_clear_caches!()
    @test_throws ErrorException extension.exchange!(
        context_value, context_receive, MPI.COMM_SELF;
        adapter=context_adapter,
        collective=(_send, _receive, _comm) -> error("direct MPI failed"),
    )
    @test MOCK_CURRENT_DEVICE[] == 2
    context_aware[] = false

    # Same-vendor buffers on different local devices are a collective error
    # before callbacks, communication counters, or output mutation.
    cross_device = MockMultiDeviceArray(Float32[9, 10], 2)
    @test_throws ArgumentError extension._validate_parallel_storage!(
        MPI.COMM_SELF, :cross_device_fake, context_value, cross_device;
        adapter=context_adapter,
    )
    cross_device_sentinel = copy(cross_device.data)
    cross_device_calls = Ref(0)
    before_cross_device = extension.parallel_gpu_stats()
    @test_throws ArgumentError extension.exchange!(
        context_value, cross_device, MPI.COMM_SELF; adapter=context_adapter,
        collective=(_send, receive, _comm) -> begin
            cross_device_calls[] += 1
            fill!(receive, -1)
        end,
    )
    @test cross_device.data == cross_device_sentinel
    @test cross_device_calls[] == 0
    @test extension.parallel_gpu_stats() == before_cross_device
    MPI.Barrier(MPI.COMM_SELF)

    # Original send/receive aliasing is rejected before either policy branch,
    # callback, traffic counters, or mutation. Distinct overlapping views are
    # just as invalid as exact identity.
    alias_root = MockMultiDeviceArray(Float32[1, 2, 3, 4], 1)
    alias_cases = (
        (view(alias_root, 1:3), view(alias_root, 1:3)),
        (view(alias_root, 1:3), view(alias_root, 2:4)),
    )
    for aware in (false, true), (send_alias, receive_alias) in alias_cases
        context_aware[] = aware
        alias_sentinel = copy(alias_root.data)
        alias_callbacks = Ref(0)
        alias_stats = extension.parallel_gpu_stats()
        @test_throws ArgumentError extension.exchange!(
            send_alias, receive_alias, MPI.COMM_SELF;
            adapter=context_adapter,
            collective=(_send, receive, _comm) -> begin
                alias_callbacks[] += 1
                fill!(receive, -1)
            end,
        )
        @test alias_root.data == alias_sentinel
        @test alias_callbacks[] == 0
        @test extension.parallel_gpu_stats() == alias_stats
        MPI.Barrier(MPI.COMM_SELF)
    end
    context_aware[] = false

    extension.parallel_gpu_clear_caches!()
    multi_host_allocations[] = 0
    # Equal-size logical views of one allocation need independent staging
    # snapshots. Sharing a root-keyed entry overwrites the first input before
    # the CPU callback observes it.
    first_view = view(multi, 1, :)
    second_view = view(multi, 2, :)
    distinct_view_result = extension._staged_gpu_call(
        multi_adapter, :distinct_views, multi_comm,
        (first_host, second_host) -> sum(10 .* first_host .+ second_host),
        first_view, second_view; validate_storage=false,
    )
    @test distinct_view_result == 46
    @test multi_host_allocations[] == 2
    @test extension.parallel_gpu_cache_sizes().staging == 2
    @test extension._staged_gpu_call(
        multi_adapter, :distinct_views, multi_comm,
        (first_host, second_host) -> sum(10 .* first_host .+ second_host),
        first_view, second_view; validate_storage=false,
    ) == 46
    @test multi_host_allocations[] == 2

    # Equivalent live wrappers describe the same transfer region and must
    # reuse one entry and one lock even when wrapper identities differ.
    extension.parallel_gpu_clear_caches!()
    multi_host_allocations[] = 0
    equivalent_first = view(multi, [1], :)
    equivalent_second = view(multi, [1], :)
    equivalent_entry = extension._staging_entry(
        multi_adapter, multi_comm, equivalent_first, 2,
    )
    @test extension._staging_entry(
        multi_adapter, multi_comm, equivalent_second, 2,
    ) === equivalent_entry
    @test multi_host_allocations[] == 1

    # Concurrent first use of equivalent live wrappers is single-flight: all
    # tasks receive the same entry/lock and only one pin allocation occurs.
    extension.parallel_gpu_clear_caches!()
    concurrent_allocations = Threads.Atomic{Int}(0)
    concurrent_adapter = extension.ParallelGPUAdapter(
        :mock_equivalent_views, multi_adapter.matches,
        multi_adapter.array_type, multi_adapter.device,
        multi_adapter.with_device, multi_adapter.gpu_aware,
        multi_adapter.synchronize,
        (T, n) -> begin
            Threads.atomic_add!(concurrent_allocations, 1)
            for _ in 1:16
                yield()
            end
            Vector{T}(undef, n)
        end,
        multi_adapter.device_to_host!, multi_adapter.host_to_device!,
    )
    start = Base.Event()
    concurrent_views = [view(multi, [1], :) for _ in 1:8]
    tasks = [Threads.@spawn begin
        wait(start)
        extension._staging_entry(
            concurrent_adapter, multi_comm, logical, length(logical),
        )
    end for logical in concurrent_views]
    notify(start)
    concurrent_entries = fetch.(tasks)
    @test all(entry -> entry === first(concurrent_entries), concurrent_entries)
    @test concurrent_allocations[] == 1

    # At capacity, different live regions borrow only idle compatible pinned
    # slots. Concurrent excess users wait; they never allocate uncached buffers
    # or corrupt one another's payload.
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(2)
    pool_allocations = Threads.Atomic{Int}(0)
    pool_adapter = extension.ParallelGPUAdapter(
        :mock_bounded_pool, multi_adapter.matches, multi_adapter.array_type,
        multi_adapter.device, multi_adapter.with_device,
        multi_adapter.gpu_aware, multi_adapter.synchronize,
        (T, n) -> begin
            Threads.atomic_add!(pool_allocations, 1)
            Vector{T}(undef, n)
        end,
        multi_adapter.device_to_host!, multi_adapter.host_to_device!,
    )
    pool_values = [MockMultiDeviceArray(Float32[n, n + 1], 1) for n in 1:8]
    for _ in 1:2, value in pool_values
        extension.allreduce!(
            value, +, multi_comm; adapter=pool_adapter,
            collective=(host, _op, _comm) -> host,
        )
    end
    @test pool_allocations[] <= 2
    extension.parallel_gpu_clear_caches!()
    pool_allocations[] = 0
    start_pool = Base.Event()
    pool_tasks = [Threads.@spawn begin
        wait(start_pool)
        extension.allreduce!(
            value, +, multi_comm; adapter=pool_adapter,
            collective=(host, _op, _comm) -> begin
                for _ in 1:8
                    yield()
                end
                host .+= 1
                host
            end,
        )
    end for value in pool_values]
    notify(start_pool)
    fetch.(pool_tasks)
    @test pool_allocations[] <= 2
    for (n, value) in enumerate(pool_values)
        @test value.data == Float32[n + 1, n + 2]
    end

    # Lowering the limit trims idle entries immediately. Active entries may
    # temporarily exceed the new limit, but releases must converge the raw
    # registry to the requested cap before admitting another allocation.
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(3)
    shrink_values = [MockMultiDeviceArray(Float32[n], 1) for n in 1:3]
    for value in shrink_values
        extension._staging_entry(
            pool_adapter, multi_comm, value, length(value),
        )
    end
    @test length(extension._GPU_STAGING) == 3
    extension.parallel_gpu_cache_limit!(1)
    @test length(extension._GPU_STAGING) == 1

    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(2)
    active_first = extension._staging_entry(
        pool_adapter, multi_comm, shrink_values[1], 1; acquire=true,
    )
    active_second = extension._staging_entry(
        pool_adapter, multi_comm, shrink_values[2], 1; acquire=true,
    )
    extension.parallel_gpu_cache_limit!(1)
    @test length(extension._GPU_STAGING) == 2
    extension._release_staging_entry(active_first)
    @test length(extension._GPU_STAGING) == 1
    extension._release_staging_entry(active_second)
    @test length(extension._GPU_STAGING) == 1

    # Clear preserves active/pending ownership only until current users finish;
    # their retired entries must then disappear without a second clear.
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(2)
    active_stage_entered = Base.Event()
    active_stage_release = Base.Event()
    active_stage = Threads.@spawn extension._with_staging(
        pool_adapter, multi_comm, shrink_values[1], Float32, 1,
    ) do _host
        notify(active_stage_entered)
        wait(active_stage_release)
    end
    wait(active_stage_entered)
    extension.parallel_gpu_clear_caches!()
    @test length(extension._GPU_STAGING) == 1
    notify(active_stage_release)
    fetch(active_stage)
    @test length(extension._GPU_STAGING) == 0

    pending_allocations = Threads.Atomic{Int}(0)
    pending_entered = Base.Event()
    pending_release = Base.Event()
    pending_adapter = extension.ParallelGPUAdapter(
        :mock_pending_stage, multi_adapter.matches, multi_adapter.array_type,
        multi_adapter.device, multi_adapter.with_device,
        multi_adapter.gpu_aware, multi_adapter.synchronize,
        (T, n) -> begin
            allocation = Threads.atomic_add!(pending_allocations, 1)
            if allocation == 0
                notify(pending_entered)
                wait(pending_release)
            end
            Vector{T}(undef, n)
        end,
        multi_adapter.device_to_host!, multi_adapter.host_to_device!,
    )
    first_pending = Threads.@spawn extension._with_staging(
        pending_adapter, multi_comm, shrink_values[1], Float32, 1,
    ) do host
        objectid(parent(host))
    end
    wait(pending_entered)
    extension.parallel_gpu_clear_caches!()
    second_pending = Threads.@spawn extension._with_staging(
        pending_adapter, multi_comm, shrink_values[1], Float32, 1,
    ) do host
        objectid(parent(host))
    end
    yield()
    notify(pending_release)
    first_pending_entry = fetch(first_pending)
    second_pending_entry = fetch(second_pending)
    @test pending_allocations[] == 1
    @test first_pending_entry == second_pending_entry
    @test length(extension._GPU_STAGING) == 0

    # Rolling back a partially acquired working set must retire entries that
    # were tombstoned by a concurrent cache clear.  A later allocation failure
    # used to decrement `users` without removing the now-idle entry.
    extension.parallel_gpu_clear_caches!()
    rollback_allocations = Threads.Atomic{Int}(0)
    rollback_entered = Base.Event()
    rollback_release = Base.Event()
    rollback_fail = Ref(true)
    rollback_adapter = extension.ParallelGPUAdapter(
        :mock_staging_rollback, multi_adapter.matches,
        multi_adapter.array_type, multi_adapter.device,
        multi_adapter.with_device, multi_adapter.gpu_aware,
        multi_adapter.synchronize,
        (T, n) -> begin
            allocation = Threads.atomic_add!(rollback_allocations, 1)
            if allocation == 1 && rollback_fail[]
                notify(rollback_entered)
                wait(rollback_release)
                error("rollback pin allocation failed")
            end
            Vector{T}(undef, n)
        end,
        multi_adapter.device_to_host!, multi_adapter.host_to_device!,
    )
    rollback_first = MockMultiDeviceArray(Float32[1], 1)
    rollback_second = MockMultiDeviceArray(Float32[2], 1)
    extension._staging_entry(
        rollback_adapter, multi_comm, rollback_first, 1,
    )
    rollback_task = Threads.@spawn try
        extension._staging_entries(
            rollback_adapter, multi_comm, rollback_first, rollback_second,
        )
    catch error
        error
    end
    wait(rollback_entered)
    extension.parallel_gpu_clear_caches!()
    @test length(extension._GPU_STAGING) == 1
    notify(rollback_release)
    @test fetch(rollback_task) isa ErrorException
    @test length(extension._GPU_STAGING) == 0
    rollback_fail[] = false
    @test extension._staged_gpu_call(
        rollback_adapter, :staging_rollback_retry, multi_comm,
        (first, second) -> first[1] + second[1],
        rollback_first, rollback_second; validate_storage=false,
    ) == 3
    extension.parallel_gpu_cache_limit!(8)

    # Wrapper signatures must retain the wrapped logical region. Equal-shape
    # reshapes over different views of one root need separate, reusable slots.
    extension.parallel_gpu_clear_caches!()
    multi_host_allocations[] = 0
    first_reshape = reshape(view(multi, 1, :), 1, :)
    second_reshape = reshape(view(multi, 2, :), 1, :)
    for _ in 1:2
        @test extension._staged_gpu_call(
            multi_adapter, :distinct_wrapped_views, multi_comm,
            (first_host, second_host) -> sum(10 .* first_host .+ second_host),
            first_reshape, second_reshape; validate_storage=false,
        ) == 46
    end
    @test multi_host_allocations[] == 2
    @test extension.parallel_gpu_cache_sizes().staging == 2

    # Out-of-place restoration must allocate on the prototype's physical
    # device, not whichever device happens to be current in this task.
    MOCK_CURRENT_DEVICE[] = 2
    prototype_device_result = extension._device_result(
        multi_adapter, multi, Float32[7, 8], (), (),
    )
    @test prototype_device_result isa MockMultiDeviceArray
    @test prototype_device_result.device == multi.device
    @test MOCK_CURRENT_DEVICE[] == 2
    failing_device_adapter = extension.ParallelGPUAdapter(
        :mock_multidevice_failure, multi_adapter.matches,
        multi_adapter.array_type, multi_adapter.device,
        multi_adapter.with_device, multi_adapter.gpu_aware,
        multi_adapter.synchronize, multi_adapter.allocate_pinned,
        multi_adapter.device_to_host!,
        (_device, _host) -> error("copy to device failed"),
    )
    @test_throws ErrorException extension._device_result(
        failing_device_adapter, multi, Float32[7, 8], (), (),
    )
    @test MOCK_CURRENT_DEVICE[] == 2

    # Every standard array wrapper admitted by a vendor adapter must resolve
    # to the allocation that owns its physical device and cache lifetime.
    mask_root = MockMultiDeviceArray(Bool[true, false, true, false], 1)
    wrappers = (
        PermutedDimsArray(multi, (2, 1)), adjoint(multi), transpose(multi),
        Symmetric(multi), Hermitian(multi), UpperTriangular(multi),
        LowerTriangular(multi), UnitUpperTriangular(multi),
        UnitLowerTriangular(multi), Diagonal(view(multi, 1:2, 1)),
        Bidiagonal(view(multi, 1:2, 1), view(multi, 1:1, 2), :U),
        Tridiagonal(
            view(multi, 1:1, 1), view(multi, 1:2, 1), view(multi, 1:1, 2),
        ),
        Base.LogicalIndex(mask_root),
    )
    expected_roots = (ntuple(_ -> multi, length(wrappers) - 1)..., mask_root)
    for (wrapper, expected_root) in zip(wrappers, expected_roots)
        @test extension._parallel_root_buffer(wrapper) === expected_root
        @test multi_adapter.matches(wrapper)
        @test multi_adapter.device(wrapper) == expected_root.device
    end
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(length(wrappers) + 1)
    GC.@preserve wrappers begin
        for wrapper in wrappers
            extension._staging_entry(
                multi_adapter, multi_comm, wrapper, length(wrapper),
            )
        end
        @test extension.parallel_gpu_cache_sizes().staging == length(wrappers)
    end

    # Live logical wrappers cannot make the registry exceed its cap, and a
    # cached view does not keep its physical allocation alive.
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(2)
    for wrapper in wrappers
        extension._staging_entry(
            multi_adapter, multi_comm, wrapper, length(wrapper),
        )
        @test extension.parallel_gpu_cache_sizes().staging <= 2
    end
    extension.parallel_gpu_clear_caches!()
    weak_root = _cache_temporary_view!(extension, multi_adapter, multi_comm)
    @test extension.parallel_gpu_cache_sizes().staging == 1
    GC.gc(true)
    @test weak_root.value === nothing
    @test extension.parallel_gpu_cache_sizes().staging == 0

    # A dead logical view must be removable even while its allocation root is
    # still live; the cache must not retain entries by objectid alone.
    extension.parallel_gpu_clear_caches!()
    live_root = MockMultiDeviceArray(reshape(Float32.(1:4), 2, 2), 1)
    weak_logical = _cache_temporary_logical_view!(
        extension, multi_adapter, multi_comm, live_root,
    )
    @test length(extension._GPU_STAGING) == 1
    GC.gc(true)
    @test weak_logical.value === nothing
    extension.parallel_gpu_cache_sizes()
    @test length(extension._GPU_STAGING) == 1
    reused_region = view(live_root, 1, :)
    before_region_reuse = multi_host_allocations[]
    extension._staging_entry(
        multi_adapter, multi_comm, reused_region, length(reused_region),
    )
    @test multi_host_allocations[] == before_region_reuse
    @test length(extension._GPU_STAGING) <= extension._GPU_STAGING_LIMIT[]

    # Awareness registries purge dead communicator identities and remain
    # bounded even when many live subcommunicators are observed.
    extension.parallel_gpu_clear_caches!()
    awareness_comms = [Ref(Symbol(:awareness_comm_, n)) for n in 1:80]
    for awareness_comm in awareness_comms
        extension._gpu_awareness(multi_adapter, awareness_comm, multi)
    end
    @test length(extension._GPU_AWARENESS) <= 64
    empty!(awareness_comms)
    awareness_comm = nothing
    GC.gc(true)
    extension.parallel_gpu_cache_sizes()
    @test length(extension._GPU_AWARENESS) == 0

    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(8)
    extension.allreduce!(multi_view, +, multi_comm; adapter=multi_adapter,
                         collective=(host, _op, _comm) -> host)
    multi.device = 2
    extension.allreduce!(multi_view, +, multi_comm; adapter=multi_adapter,
                         collective=(host, _op, _comm) -> host)
    @test current_device[] == 99
    @test extension.parallel_gpu_cache_sizes() ==
          (awareness=2, staging=2, native_host_plans=0)

    # A staged collective cannot return until its host-to-device copy has been
    # synchronized. Per-entry locks must also be released on MPI/copy errors.
    extension.parallel_gpu_clear_caches!()
    events = Symbol[]
    throw_copy = Ref(false)
    event_adapter = extension.ParallelGPUAdapter(
        :mock_events,
        value -> value isa MockMPIArray,
        _ -> MockMPIArray,
        _ -> 1,
        _ -> false,
        _ -> push!(events, :sync),
        (T, n) -> Vector{T}(undef, n),
        (host, device) -> (push!(events, :device_to_host); copyto!(host, device)),
        (device, host) -> begin
            push!(events, :host_to_device)
            throw_copy[] && error("copy-back failed")
            copyto!(device, host)
        end,
    )
    extension._register_parallel_gpu_adapter!(event_adapter)
    event_buffer = MockMPIArray(Float32[1, 2])
    event_comm = Ref(:event_subgroup)
    extension.allreduce!(
        event_buffer, +, event_comm; adapter=event_adapter,
        collective=(host, _op, _comm) -> (push!(events, :collective); host),
    )
    @test events == [
        :sync, :device_to_host, :sync, :collective,
        :host_to_device, :sync,
    ]

    empty!(events)
    event_receive = MockMPIArray(zeros(Float32, 2))
    extension.exchange!(
        event_buffer, event_receive, MPI.COMM_SELF;
        adapter=event_adapter,
        collective=(send, receive, _comm) -> begin
            push!(events, :collective)
            copyto!(receive, send)
        end,
    )
    @test events == [
        :sync, :device_to_host, :sync, :collective,
        :host_to_device, :sync,
    ]

    empty!(events)
    @test_throws ErrorException extension.allreduce!(
        event_buffer, +, Ref(:event_mpi_throw); adapter=event_adapter,
        collective=(_host, _op, _comm) -> begin
            push!(events, :collective_throw)
            error("MPI failed")
        end,
    )
    @test events == [:sync, :device_to_host, :sync, :collective_throw]
    empty!(events)
    extension.allreduce!(event_buffer, +, Ref(:event_mpi_throw);
                         adapter=event_adapter,
                         collective=(host, _op, _comm) -> host)
    @test last(events) == :sync

    empty!(events)
    throw_copy[] = true
    @test_throws ErrorException extension.allreduce!(
        event_buffer, +, Ref(:event_copy_throw); adapter=event_adapter,
        collective=(host, _op, _comm) -> host,
    )
    @test last(events) == :host_to_device
    throw_copy[] = false
    empty!(events)
    extension.allreduce!(event_buffer, +, Ref(:event_copy_throw);
                         adapter=event_adapter,
                         collective=(host, _op, _comm) -> host)
    @test last(events) == :sync
    lock(extension._PARALLEL_GPU_ADAPTER_LOCK) do
        delete!(extension._PARALLEL_GPU_ADAPTERS, :mock_events)
    end
    extension.parallel_gpu_clear_caches!()
    extension.parallel_gpu_cache_limit!(2)

    # Staged mathematical calls preserve `!` output identity and recursively
    # restore tuple results to the same vendor without leaking host arrays.
    staged_output = MockMPIArray(Float32[0, 0, 0])
    staged_input = MockMPIArray(Float32[2, 3, 4])
    staged_result = extension._staged_gpu_call(
        adapter, :mock_bang, comm,
        (host_output, host_input) -> begin
            host_output .= host_input .+ 1
            (host_output, copy(host_input))
        end,
        staged_output, staged_input; mutated=(1,), validate_storage=false,
    )
    @test first(staged_result) === staged_output
    @test staged_output.data == Float32[3, 4, 5]
    @test last(staged_result) isa MockMPIArray
    @test last(staged_result).data == staged_input.data

    # Pencil metadata and subgroup communicators survive staging intact.
    pencil = PencilArrays.Pencil(
        MockMPIArray, (3, 2), (1,), MPI.COMM_SELF,
    )
    device_pencil = PencilArrays.PencilArray{Float32}(undef, pencil)
    parent(device_pencil).data .= reshape(Float32.(1:6), 3, 2)
    pencil_result = extension._staged_gpu_call(
        adapter, :mock_pencil, MPI.COMM_SELF,
        host -> begin
            @test PencilArrays.size_global(host) == (3, 2)
            parent(host) .*= 3
            host
        end,
        device_pencil; mutated=(1,),
    )
    @test pencil_result === device_pencil
    @test parent(device_pencil).data == 3 .* reshape(Float32.(1:6), 3, 2)

    # A vendor-backed Pencil in any positional slot must reach collective
    # storage preflight before generic CPU mathematics.
    mixed_adapter = extension.ParallelGPUAdapter(
        :mock_mixed,
        value -> value isa MockMixedVendorArray,
        _ -> MockMixedVendorArray,
        _ -> 17,
        _ -> false,
        _ -> nothing,
        (T, n) -> Vector{T}(undef, n),
        (host, device) -> copyto!(host, device),
        (device, host) -> copyto!(device, host),
    )
    extension._register_parallel_gpu_adapter!(mixed_adapter)
    # Registration is weak: preserving the caller's adapter must keep the
    # same identity reachable throughout these storage-policy checks.
    GC.@preserve mixed_adapter begin
      try
    GC.gc(true)
    mixed_cfg = SHTnsKit.create_gauss_config(2, 4; nlon=6)
    cpu_spatial_pen = SHTnsKit.create_spatial_pencil(
        mixed_cfg; comm=MPI.COMM_SELF,
    )
    gpu_spatial_pen = similar(cpu_spatial_pen, MockMixedVendorArray)
    cpu_spatial = PencilArrays.PencilArray{Float64}(undef, cpu_spatial_pen)
    gpu_spatial = PencilArrays.PencilArray{Float64}(undef, gpu_spatial_pen)
    fill!(parent(cpu_spatial), 0.25)
    fill!(parent(gpu_spatial).data, 0.5)
    spatial_sentinels = (copy(parent(cpu_spatial)), copy(parent(gpu_spatial).data))
    mixed_before = extension.parallel_gpu_stats()
    for values in ((cpu_spatial, gpu_spatial), (gpu_spatial, cpu_spatial))
        error = try
            SHTnsKit.analysis_sphtor(mixed_cfg, values...)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("storage/vendor/device mismatch", sprint(showerror, error))
    end
    for vendor_position in 1:3
        values = ntuple(index -> index == vendor_position ? gpu_spatial :
                         cpu_spatial, 3)
        error = try
            SHTnsKit.analysis_qst(mixed_cfg, values...)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("storage/vendor/device mismatch", sprint(showerror, error))
    end
    @test parent(cpu_spatial) == first(spatial_sentinels)
    @test parent(gpu_spatial).data == last(spatial_sentinels)
    @test extension.parallel_gpu_stats() == mixed_before

    # Closed inventory of generic multi-Pencil diagnostic/local entry points.
    # Every position containing vendor storage must be rejected by the shared
    # collective preflight before numerical work or communication.
    cpu_spectral_pen = SHTnsKit.create_spectral_pencil(
        mixed_cfg; comm=MPI.COMM_SELF,
    )
    gpu_spectral_pen = similar(cpu_spectral_pen, MockMixedVendorArray)
    cpu_spectral = PencilArrays.PencilArray{ComplexF64}(
        undef, cpu_spectral_pen,
    )
    gpu_spectral = PencilArrays.PencilArray{ComplexF64}(
        undef, gpu_spectral_pen,
    )
    fill!(parent(cpu_spectral), 0.0)
    fill!(parent(gpu_spectral).data, 0.0)
    mixed_cases = Pair{Symbol,Function}[]
    append!(mixed_cases, (
        :dist_synthesis_coefficients => (() -> SHTnsKit.dist_synthesis(
            mixed_cfg, gpu_spectral; prototype_θφ=cpu_spatial,
        )),
        :dist_synthesis_prototype => (() -> SHTnsKit.dist_synthesis(
            mixed_cfg, cpu_spectral; prototype_θφ=gpu_spatial,
        )),
        :dist_synthesis_dense_coefficients => (() -> SHTnsKit.dist_synthesis(
            mixed_cfg, MockMixedVendorArray(zeros(
                ComplexF64, mixed_cfg.lmax + 1, mixed_cfg.mmax + 1,
            )); prototype_θφ=cpu_spatial,
        )),
        :dist_synthesis_dense_prototype => (() -> SHTnsKit.dist_synthesis(
            mixed_cfg, zeros(
                ComplexF64, mixed_cfg.lmax + 1, mixed_cfg.mmax + 1,
            ); prototype_θφ=gpu_spatial,
        )),
    ))
    for name in (:energy_vector_l_spectrum, :energy_vector_m_spectrum)
        function_object = getfield(SHTnsKit, name)
        push!(mixed_cases,
              Symbol(name, :_first) => (() -> function_object(
                  mixed_cfg, gpu_spectral, cpu_spectral,
              )))
        push!(mixed_cases,
              Symbol(name, :_second) => (() -> function_object(
                  mixed_cfg, cpu_spectral, gpu_spectral,
              )))
    end
    for (name, first, second) in (
            (:grid_energy_vector_first, gpu_spatial, cpu_spatial),
            (:grid_energy_vector_second, cpu_spatial, gpu_spatial))
        push!(mixed_cases, name => (() -> SHTnsKit.grid_energy_vector(
            mixed_cfg, first, second,
        )))
    end
    for name in (:dist_SHqst_to_point, :SHqst_to_point)
        function_object = getfield(SHTnsKit, name)
        for position in 1:3
            values = ntuple(index -> index == position ? gpu_spectral :
                             cpu_spectral, 3)
            push!(mixed_cases,
                  Symbol(name, :_, position) => (() -> function_object(
                      mixed_cfg, values..., 0.25, 0.5,
                  )))
        end
    end
    for name in (:dist_SHqst_to_lat, :SHqst_to_lat)
        function_object = getfield(SHTnsKit, name)
        for position in 1:3
            values = ntuple(index -> index == position ? gpu_spectral :
                             cpu_spectral, 3)
            push!(mixed_cases,
                  Symbol(name, :_, position) => (() -> function_object(
                      mixed_cfg, values..., 0.25,
                  )))
        end
    end
    for (name, first, second) in (
            (:SH_to_grad_point_first, gpu_spectral, cpu_spectral),
            (:SH_to_grad_point_second, cpu_spectral, gpu_spectral))
        push!(mixed_cases, name => (() -> SHTnsKit.SH_to_grad_point(
            mixed_cfg, first, second, 0.25, 0.5,
        )))
    end

    cpu_packed_pen = PencilArrays.Pencil(
        Array, (mixed_cfg.nlm, 1), (1,), MPI.COMM_SELF,
    )
    gpu_packed_pen = similar(cpu_packed_pen, MockMixedVendorArray)
    cpu_packed = PencilArrays.PencilArray{ComplexF64}(undef, cpu_packed_pen)
    gpu_packed = PencilArrays.PencilArray{ComplexF64}(undef, gpu_packed_pen)
    fill!(parent(cpu_packed), 0)
    fill!(parent(gpu_packed).data, 0)
    for name in (:synthesis_packed, :synthesis_packed_l)
        function_object = getfield(SHTnsKit, name)
        suffix = name === :synthesis_packed ? () : (mixed_cfg.lmax,)
        push!(mixed_cases,
              Symbol(name, :_coefficients) => (() -> function_object(
                  mixed_cfg, gpu_packed, suffix...;
                  prototype_θφ=cpu_spatial,
              )))
        push!(mixed_cases,
              Symbol(name, :_prototype) => (() -> function_object(
                  mixed_cfg, cpu_packed, suffix...;
                  prototype_θφ=gpu_spatial,
              )))
    end
    cpu_packed_vector = zeros(ComplexF64, mixed_cfg.nlm)
    gpu_packed_vector = MockMixedVendorArray(copy(cpu_packed_vector))
    append!(mixed_cases, (
        :dist_synthesis_packed_coefficients => (() ->
            SHTnsKit.dist_synthesis_packed(
                mixed_cfg, gpu_packed_vector; prototype_θφ=cpu_spatial,
            )),
        :dist_synthesis_packed_prototype => (() ->
            SHTnsKit.dist_synthesis_packed(
                mixed_cfg, cpu_packed_vector; prototype_θφ=gpu_spatial,
            )),
    ))
    complex_packed_length = SHTnsKit.nlm_cplx_calc(
        mixed_cfg.lmax, mixed_cfg.mmax, 1,
    )
    cpu_complex_packed_pen = PencilArrays.Pencil(
        Array, (complex_packed_length, 1), (1,), MPI.COMM_SELF,
    )
    gpu_complex_packed_pen = similar(
        cpu_complex_packed_pen, MockMixedVendorArray,
    )
    cpu_complex_packed = PencilArrays.PencilArray{ComplexF64}(
        undef, cpu_complex_packed_pen,
    )
    gpu_complex_packed = PencilArrays.PencilArray{ComplexF64}(
        undef, gpu_complex_packed_pen,
    )
    cpu_complex_spatial = PencilArrays.PencilArray{ComplexF64}(
        undef, cpu_spatial_pen,
    )
    gpu_complex_spatial = PencilArrays.PencilArray{ComplexF64}(
        undef, gpu_spatial_pen,
    )
    fill!(parent(cpu_complex_packed), 0)
    fill!(parent(gpu_complex_packed).data, 0)
    fill!(parent(cpu_complex_spatial), 0)
    fill!(parent(gpu_complex_spatial).data, 0)
    for name in (:synthesis_packed_cplx, :synthesis_packed_cplx_l)
        function_object = getfield(SHTnsKit, name)
        suffix = name === :synthesis_packed_cplx ? () : (mixed_cfg.lmax,)
        push!(mixed_cases,
              Symbol(name, :_coefficients) => (() -> function_object(
                  mixed_cfg, gpu_complex_packed, suffix...;
                  prototype_θφ=cpu_complex_spatial,
              )))
        push!(mixed_cases,
              Symbol(name, :_prototype) => (() -> function_object(
                  mixed_cfg, cpu_complex_packed, suffix...;
                  prototype_θφ=gpu_complex_spatial,
              )))
    end
    cpu_complex_packed_vector = zeros(ComplexF64, complex_packed_length)
    gpu_complex_packed_vector = MockMixedVendorArray(
        copy(cpu_complex_packed_vector),
    )
    append!(mixed_cases, (
        :dist_synthesis_packed_cplx_coefficients => (() ->
            SHTnsKit.dist_synthesis_packed_cplx(
                mixed_cfg, gpu_complex_packed_vector;
                prototype_θφ=cpu_complex_spatial,
            )),
        :dist_synthesis_packed_cplx_prototype => (() ->
            SHTnsKit.dist_synthesis_packed_cplx(
                mixed_cfg, cpu_complex_packed_vector;
                prototype_θφ=gpu_complex_spatial,
            )),
    ))

    for name in (:dist_SH_Zrotate, :dist_SH_Yrotate,
                 :dist_SH_Yrotate_allgatherm!,
                 :dist_SH_Yrotate_truncgatherm!)
        function_object = getfield(SHTnsKit, name)
        push!(mixed_cases,
              Symbol(name, :_input) => (() -> function_object(
                  mixed_cfg, gpu_spectral, 0.25, cpu_spectral,
              )))
        push!(mixed_cases,
              Symbol(name, :_output) => (() -> function_object(
                  mixed_cfg, cpu_spectral, 0.25, gpu_spectral,
              )))
    end
    for name in (:dist_SH_Yrotate90, :dist_SH_Xrotate90)
        function_object = getfield(SHTnsKit, name)
        push!(mixed_cases,
              Symbol(name, :_input) => (() -> function_object(
                  mixed_cfg, gpu_spectral, cpu_spectral,
              )))
        push!(mixed_cases,
              Symbol(name, :_output) => (() -> function_object(
                  mixed_cfg, cpu_spectral, gpu_spectral,
              )))
    end
    push!(mixed_cases,
          :dist_SH_rotate_euler_input => (() -> SHTnsKit.dist_SH_rotate_euler(
              mixed_cfg, gpu_spectral, 0.1, 0.2, 0.3, cpu_spectral,
          )))
    push!(mixed_cases,
          :dist_SH_rotate_euler_output => (() -> SHTnsKit.dist_SH_rotate_euler(
              mixed_cfg, cpu_spectral, 0.1, 0.2, 0.3, gpu_spectral,
          )))
    for name in (:dist_SH_Zrotate_packed, :dist_SH_Yrotate_packed)
        function_object = getfield(SHTnsKit, name)
        push!(mixed_cases,
              Symbol(name, :_coefficients) => (() -> function_object(
                  mixed_cfg, gpu_packed_vector, 0.25;
                  prototype_lm=cpu_spectral,
              )))
        push!(mixed_cases,
              Symbol(name, :_prototype) => (() -> function_object(
                  mixed_cfg, cpu_packed_vector, 0.25;
                  prototype_lm=gpu_spectral,
              )))
    end
    for name in (:dist_SH_Yrotate90_packed, :dist_SH_Xrotate90_packed)
        function_object = getfield(SHTnsKit, name)
        push!(mixed_cases,
              Symbol(name, :_coefficients) => (() -> function_object(
                  mixed_cfg, gpu_packed_vector; prototype_lm=cpu_spectral,
              )))
        push!(mixed_cases,
              Symbol(name, :_prototype) => (() -> function_object(
                  mixed_cfg, cpu_packed_vector; prototype_lm=gpu_spectral,
              )))
    end

    cpu_batch_spatial = PencilArrays.PencilArray{Float64}(
        undef, cpu_spatial_pen, 2,
    )
    gpu_batch_spatial = PencilArrays.PencilArray{Float64}(
        undef, gpu_spatial_pen, 2,
    )
    cpu_batch_spectral = PencilArrays.PencilArray{ComplexF64}(
        undef, cpu_spectral_pen, 2,
    )
    gpu_batch_spectral = PencilArrays.PencilArray{ComplexF64}(
        undef, gpu_spectral_pen, 2,
    )
    cpu_batch_complex_spatial = PencilArrays.PencilArray{ComplexF64}(
        undef, cpu_spatial_pen, 2,
    )
    gpu_batch_complex_spatial = PencilArrays.PencilArray{ComplexF64}(
        undef, gpu_spatial_pen, 2,
    )
    cpu_batch_real_spectral = PencilArrays.PencilArray{Float64}(
        undef, cpu_spectral_pen, 2,
    )
    for value in (cpu_batch_spatial, cpu_batch_spectral)
        fill!(parent(value), 0)
    end
    for value in (gpu_batch_spatial, gpu_batch_spectral)
        fill!(parent(value).data, 0)
    end
    fill!(parent(cpu_batch_complex_spatial), 0)
    fill!(parent(gpu_batch_complex_spatial).data, 0)
    fill!(parent(cpu_batch_real_spectral), 0)

    # Invalid CPU batch ranks must still enter a broad distributed entry point.
    # A local MethodError here becomes a deadlock when peer ranks dispatch to
    # the compound GPU firewall and wait in its storage collective.
    invalid_cpu_batch_calls = (
        () -> SHTnsKit.analysis_batch(mixed_cfg, cpu_spatial),
        () -> SHTnsKit.analysis_batch(
            mixed_cfg, cpu_batch_complex_spatial,
        ),
        () -> SHTnsKit.analysis_batch!(
            mixed_cfg, cpu_spectral, cpu_spatial,
        ),
        () -> SHTnsKit.analysis_batch!(
            mixed_cfg, cpu_batch_spectral, cpu_batch_complex_spatial,
        ),
        () -> SHTnsKit.synthesis_batch(
            mixed_cfg, cpu_spectral; prototype_θφ=cpu_spatial,
        ),
        () -> SHTnsKit.synthesis_batch(
            mixed_cfg, cpu_batch_real_spectral;
            prototype_θφ=cpu_batch_spatial,
        ),
        () -> SHTnsKit.synthesis_batch_cplx(
            mixed_cfg, cpu_spectral; prototype_θφ=cpu_complex_spatial,
        ),
        () -> SHTnsKit.synthesis_batch_cplx(
            mixed_cfg, cpu_batch_real_spectral;
            prototype_θφ=cpu_batch_complex_spatial,
        ),
        () -> SHTnsKit.synthesis_batch!(
            mixed_cfg, cpu_spatial, cpu_spectral;
            prototype_θφ=cpu_spatial,
        ),
        () -> SHTnsKit.synthesis_batch!(
            mixed_cfg, cpu_batch_spatial, cpu_batch_real_spectral;
            prototype_θφ=cpu_batch_spatial,
        ),
    )
    invalid_cpu_batch_values = (
        cpu_spatial, cpu_spectral, cpu_complex_spatial,
        cpu_batch_spatial, cpu_batch_complex_spatial,
        cpu_batch_spectral, cpu_batch_real_spectral,
    )
    invalid_cpu_batch_before = map(
        value -> copy(parent(value)), invalid_cpu_batch_values,
    )
    for call in invalid_cpu_batch_calls
        error = try
            call()
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
    end
    @test map(value -> parent(value), invalid_cpu_batch_values) ==
          invalid_cpu_batch_before
    append!(mixed_cases, (
        :analysis_batch_output => (() -> SHTnsKit.analysis_batch!(
            mixed_cfg, gpu_batch_spectral, cpu_batch_spatial,
        )),
        :analysis_batch_fields => (() -> SHTnsKit.analysis_batch!(
            mixed_cfg, cpu_batch_spectral, gpu_batch_spatial,
        )),
        :synthesis_batch_coefficients => (() -> SHTnsKit.synthesis_batch(
            mixed_cfg, gpu_batch_spectral; prototype_θφ=cpu_batch_spatial,
        )),
        :synthesis_batch_prototype => (() -> SHTnsKit.synthesis_batch(
            mixed_cfg, cpu_batch_spectral; prototype_θφ=gpu_batch_spatial,
        )),
        :synthesis_batch_cplx_coefficients => (() ->
            SHTnsKit.synthesis_batch_cplx(
                mixed_cfg, gpu_batch_spectral;
                prototype_θφ=cpu_batch_complex_spatial,
            )),
        :synthesis_batch_cplx_prototype => (() ->
            SHTnsKit.synthesis_batch_cplx(
                mixed_cfg, cpu_batch_spectral;
                prototype_θφ=gpu_batch_complex_spatial,
            )),
        :synthesis_batch_bang_output => (() -> SHTnsKit.synthesis_batch!(
            mixed_cfg, gpu_batch_spatial, cpu_batch_spectral;
            prototype_θφ=cpu_batch_spatial,
        )),
        :synthesis_batch_bang_coefficients => (() -> SHTnsKit.synthesis_batch!(
            mixed_cfg, cpu_batch_spatial, gpu_batch_spectral;
            prototype_θφ=cpu_batch_spatial,
        )),
        :synthesis_batch_bang_prototype => (() -> SHTnsKit.synthesis_batch!(
            mixed_cfg, cpu_batch_spatial, cpu_batch_spectral;
            prototype_θφ=gpu_batch_spatial,
        )),
    ))
    for (name, call) in mixed_cases
        error = try
            call()
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("storage/vendor/device mismatch", sprint(showerror, error))
    end
    @test extension.parallel_gpu_stats() == mixed_before

    # Dense compatibility spectra are ordinary full-field mathematical APIs.
    # The compound extension owns their dispatch, but rejects them before any
    # host transfer until a device-native implementation exists.
    dense_Q = zeros(ComplexF64, mixed_cfg.lmax + 1, mixed_cfg.mmax + 1)
    dense_S = similar(dense_Q); fill!(dense_S, 0)
    dense_T = similar(dense_Q); fill!(dense_T, 0)
    dense_Aminus = similar(dense_Q); fill!(dense_Aminus, 0)
    dense_Q[1, 1] = 0.2
    dense_S[2, 1] = -0.08
    dense_T[3, 2] = 0.03 - 0.02im
    dense_Aminus[3, 2] = 0.01 + 0.04im
    vendor_Q = MockMixedVendorArray(copy(dense_Q))
    vendor_S = MockMixedVendorArray(copy(dense_S))
    vendor_T = MockMixedVendorArray(copy(dense_T))
    vendor_Aminus = MockMixedVendorArray(copy(dense_Aminus))
    compound_extension = something(
        Base.get_extension(SHTnsKit, :SHTnsKitParallelCUDAExt),
        Base.get_extension(SHTnsKit, :SHTnsKitParallelAMDGPUExt),
    )
    staged_prototype_comm = compound_extension._internal_staged_vendor_call_with_adapter(
        mixed_adapter, MPI.COMM_SELF, :dense_prototype_comm,
        host_prototype -> MPI.Comm_compare(
            PencilArrays.get_comm(host_prototype), MPI.COMM_SELF,
        ), gpu_spatial,
    )
    @test staged_prototype_comm in (MPI.IDENT, MPI.CONGRUENT)
    dense_compat_cases = (
        scalar_aminus=(
            () -> compound_extension._dist_synthesis_dense_vendor(
                mixed_adapter, MPI.COMM_SELF, mixed_cfg, vendor_Q,
                gpu_complex_spatial; real_output=false,
                Aminus=vendor_Aminus,
            ),
            :dist_synthesis_dense,
        ),
        vector=(
            () -> compound_extension._dist_synthesis_sphtor_dense_vendor(
                mixed_adapter, MPI.COMM_SELF, mixed_cfg, vendor_S, vendor_T,
                gpu_spatial,
            ),
            :dist_synthesis_sphtor_dense,
        ),
        qst=(
            () -> compound_extension._dist_synthesis_qst_dense_vendor(
                mixed_adapter, MPI.COMM_SELF, mixed_cfg, vendor_Q, vendor_S,
                vendor_T, gpu_spatial,
            ),
            :dist_synthesis_qst_dense,
        ),
    )
    for (name, (call, operation)) in pairs(dense_compat_cases)
        before_dense_stats = extension.parallel_gpu_stats()
        before_dense_payload = map(
            value -> copy(value.data),
            (vendor_Q, vendor_S, vendor_T, vendor_Aminus),
        )
        caught = try
            call()
            nothing
        catch error
            error
        end
        @test caught isa SHTnsKit.BackendUnavailableError
        @test caught.operation === operation
        @test extension.parallel_gpu_stats() == before_dense_stats
        @test map(value -> value.data,
                  (vendor_Q, vendor_S, vendor_T, vendor_Aminus)) ==
              before_dense_payload
    end

    diagnostic_source = read(joinpath(
        ROOT, "ext", "ParallelDiagnostics.jl",
    ), String)
    local_source = read(joinpath(ROOT, "ext", "ParallelLocal.jl"), String)
    for marker in (
            "energy_vector_l_spectrum", "energy_vector_m_spectrum",
            "grid_energy_vector")
        @test occursin(marker, diagnostic_source)
    end
    @test occursin("_validate_parallel_storage!", diagnostic_source)
    for marker in (
            "dist_SHqst_to_point", "dist_SHqst_to_lat", "SHqst_to_point",
            "SHqst_to_lat", "SH_to_grad_point", "synthesis_packed_l",
            "synthesis_batch!", "_validate_parallel_storage!")
        @test occursin(marker, local_source)
    end

    world_cpu_pen = SHTnsKit.create_spatial_pencil(
        mixed_cfg; comm=MPI.COMM_WORLD,
    )
    world_gpu_pen = similar(world_cpu_pen, MockMixedVendorArray)
    world_cpu = PencilArrays.PencilArray{Float64}(undef, world_cpu_pen)
    world_gpu = PencilArrays.PencilArray{Float64}(undef, world_gpu_pen)
    fill!(parent(world_cpu), 0.75)
    fill!(parent(world_gpu).data, 1.25)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    world_gpu_comm = PencilArrays.get_comm(world_gpu)

    # Reusable cfg-form plans own CPU matrices and must reject vendor storage
    # before allocating scratch or entering any layout/FFT collective.
    for constructor in (
            (prototype -> extension.DistAnalysisPlan(mixed_cfg, prototype)),
            (prototype -> extension.DistPlan(mixed_cfg, prototype)),
            (prototype -> extension.DistSphtorPlan(mixed_cfg, prototype)),
            (prototype -> extension.DistQstPlan(mixed_cfg, prototype)),
        )
        uniform_vendor_error = try
            constructor(world_gpu)
            nothing
        catch error
            error
        end
        @test uniform_vendor_error isa SHTnsKit.BackendUnavailableError
        @test MPI.Allreduce(
            uniform_vendor_error isa SHTnsKit.BackendUnavailableError ? 1 : 0,
            min, MPI.COMM_WORLD,
        ) == 1
        MPI.Barrier(MPI.COMM_WORLD)

        mixed_storage_error = try
            constructor(iseven(rank) ? world_cpu : world_gpu)
            nothing
        catch error
            error
        end
        @test mixed_storage_error isa ArgumentError
        @test occursin(
            "storage/vendor/device mismatch",
            sprint(showerror, mixed_storage_error),
        )
        @test MPI.Allreduce(
            mixed_storage_error isa ArgumentError ? 1 : 0,
            min, MPI.COMM_WORLD,
        ) == 1
        MPI.Barrier(MPI.COMM_WORLD)
    end

    # Dense compatibility dispatch depends on coefficient residency.  Every
    # rank must therefore enter the same storage preflight before a CPU rank
    # starts generic conversion while a vendor rank starts staging.
    dense_collective_cases = (
        scalar_coefficients=(() -> iseven(rank) ? SHTnsKit.dist_synthesis(
            mixed_cfg, dense_Q; prototype_θφ=world_gpu,
        ) : compound_extension._dist_synthesis_dense_vendor(
            mixed_adapter, world_gpu_comm, mixed_cfg, vendor_Q, world_gpu,
        ), :dist_synthesis_dense),
        scalar_coefficients_aminus=(() -> iseven(rank) ?
            SHTnsKit.dist_synthesis(
                mixed_cfg, dense_Q; prototype_θφ=world_gpu,
                real_output=false, Aminus=vendor_Aminus,
            ) : compound_extension._dist_synthesis_dense_vendor(
                mixed_adapter, world_gpu_comm, mixed_cfg, vendor_Q, world_gpu;
                real_output=false, Aminus=vendor_Aminus,
            ), :dist_synthesis_dense),
        scalar_aminus=(() -> iseven(rank) ? SHTnsKit.dist_synthesis(
            mixed_cfg, vendor_Q; prototype_θφ=world_gpu,
            real_output=false, Aminus=dense_Aminus,
        ) : compound_extension._dist_synthesis_dense_vendor(
            mixed_adapter, world_gpu_comm, mixed_cfg, vendor_Q, world_gpu;
            real_output=false, Aminus=vendor_Aminus,
        ), :dist_synthesis_dense),
        vector_s=(() -> iseven(rank) ? SHTnsKit.dist_synthesis_sphtor(
            mixed_cfg, dense_S, vendor_T; prototype_θφ=world_gpu,
        ) : compound_extension._dist_synthesis_sphtor_dense_vendor(
            mixed_adapter, world_gpu_comm, mixed_cfg, vendor_S, vendor_T,
            world_gpu,
        ), :dist_synthesis_sphtor_dense),
        vector_t=(() -> iseven(rank) ? SHTnsKit.dist_synthesis_sphtor(
            mixed_cfg, vendor_S, dense_T; prototype_θφ=world_gpu,
        ) : compound_extension._dist_synthesis_sphtor_dense_vendor(
            mixed_adapter, world_gpu_comm, mixed_cfg, vendor_S, vendor_T,
            world_gpu,
        ), :dist_synthesis_sphtor_dense),
        qst_q=(() -> iseven(rank) ? SHTnsKit.dist_synthesis_qst(
            mixed_cfg, dense_Q, vendor_S, vendor_T;
            prototype_θφ=world_gpu,
        ) : compound_extension._dist_synthesis_qst_dense_vendor(
            mixed_adapter, world_gpu_comm, mixed_cfg, vendor_Q, vendor_S,
            vendor_T, world_gpu,
        ), :dist_synthesis_qst_dense),
        qst_s=(() -> iseven(rank) ? SHTnsKit.dist_synthesis_qst(
            mixed_cfg, vendor_Q, dense_S, vendor_T;
            prototype_θφ=world_gpu,
        ) : compound_extension._dist_synthesis_qst_dense_vendor(
            mixed_adapter, world_gpu_comm, mixed_cfg, vendor_Q, vendor_S,
            vendor_T, world_gpu,
        ), :dist_synthesis_qst_dense),
        qst_t=(() -> iseven(rank) ? SHTnsKit.dist_synthesis_qst(
            mixed_cfg, vendor_Q, vendor_S, dense_T;
            prototype_θφ=world_gpu,
        ) : compound_extension._dist_synthesis_qst_dense_vendor(
            mixed_adapter, world_gpu_comm, mixed_cfg, vendor_Q, vendor_S,
            vendor_T, world_gpu,
        ), :dist_synthesis_qst_dense),
    )
    for (call, operation) in values(dense_collective_cases)
        before_stats = extension.parallel_gpu_stats()
        before_payload = copy(parent(world_gpu).data)
        caught = try
            call()
            nothing
        catch error
            error
        end
        expected_message = "ArgumentError: $operation collective validation failed: storage/vendor/device mismatch"
        @test caught isa ArgumentError
        @test sprint(showerror, caught) == expected_message
        @test extension.parallel_gpu_stats() == before_stats
        @test parent(world_gpu).data == before_payload
        @test MPI.Allreduce(caught isa ArgumentError ? 1 : 0, min,
                            MPI.COMM_WORLD) == 1
        MPI.Barrier(MPI.COMM_WORLD)
    end

    if MPI.Comm_size(MPI.COMM_WORLD) > 1
        before_presence_stats = extension.parallel_gpu_stats()
        before_presence_payload = copy(parent(world_gpu).data)
        presence_error = try
            SHTnsKit.dist_synthesis(
                mixed_cfg, vendor_Q; prototype_θφ=world_gpu,
                real_output=false,
                Aminus=iseven(rank) ? nothing : vendor_Aminus,
            )
            nothing
        catch error
            error
        end
        @test presence_error isa ArgumentError
        @test sprint(showerror, presence_error) ==
              "ArgumentError: dist_synthesis_dense collective validation failed: rank-varying Aminus presence"
        @test extension.parallel_gpu_stats() == before_presence_stats
        @test parent(world_gpu).data == before_presence_payload
        @test MPI.Allreduce(presence_error isa ArgumentError ? 1 : 0, min,
                            MPI.COMM_WORLD) == 1
        MPI.Barrier(MPI.COMM_WORLD)
    else
        @test_skip "rank-varying Aminus presence requires at least two ranks"
    end

    alternating = iseven(rank) ? (world_cpu, world_gpu) :
                                 (world_gpu, world_cpu)
    caught_mixed = false
    try
        SHTnsKit.analysis_sphtor(mixed_cfg, alternating...)
    catch error
        caught_mixed = error isa ArgumentError && occursin(
            "storage/vendor/device mismatch", sprint(showerror, error),
        )
    end
    @test MPI.Allreduce(caught_mixed ? 1 : 0, min, MPI.COMM_WORLD) == 1
    MPI.Barrier(MPI.COMM_WORLD)

    world_cpu_spectral_pen = SHTnsKit.create_spectral_pencil(
        mixed_cfg; comm=MPI.COMM_WORLD,
    )
    world_gpu_spectral_pen = similar(
        world_cpu_spectral_pen, MockMixedVendorArray,
    )
    world_cpu_spectral = PencilArrays.PencilArray{ComplexF64}(
        undef, world_cpu_spectral_pen,
    )
    world_gpu_spectral = PencilArrays.PencilArray{ComplexF64}(
        undef, world_gpu_spectral_pen,
    )
    fill!(parent(world_cpu_spectral), 0)
    fill!(parent(world_gpu_spectral).data, 0)

    # Simulate the method split that real CUDA/ROCm PencilArrays create when
    # storage residency differs by rank: CPU ranks enter the ordinary generic
    # method while vendor ranks enter the compound-extension firewall. Both
    # branches must rendezvous in the same storage preflight before either
    # starts configuration/layout collectives or raises a local backend error.
    split_dispatch_cases = (
        matrix_to_spectral_pencil=(:matrix_to_spectral_pencil,
            () -> iseven(rank) ? SHTnsKit.matrix_to_spectral_pencil(
                mixed_cfg, dense_Q; comm=MPI.COMM_WORLD,
            ) : compound_extension._stage_vendor_call(
                :matrix_to_spectral_pencil, identity, vendor_Q;
                comm=MPI.COMM_WORLD,
            )),
        spectral_pencil_to_matrix=(:spectral_pencil_to_matrix,
            () -> iseven(rank) ? SHTnsKit.spectral_pencil_to_matrix(
                mixed_cfg, world_cpu_spectral,
            ) : compound_extension._stage_vendor_call(
                :spectral_pencil_to_matrix, identity, world_gpu_spectral;
                comm=PencilArrays.get_comm(world_gpu_spectral),
            )),
        analysis=(:analysis, () -> iseven(rank) ?
            SHTnsKit.analysis(mixed_cfg, world_cpu) :
            compound_extension._stage_vendor_call(
                :analysis, identity, world_gpu; comm=world_gpu_comm,
            )),
        dist_analysis=(:dist_analysis, () -> iseven(rank) ?
            SHTnsKit.dist_analysis(mixed_cfg, world_cpu) :
            compound_extension._stage_vendor_call(
                :dist_analysis, identity, world_gpu; comm=world_gpu_comm,
            )),
        synthesis=(:synthesis, () -> iseven(rank) ?
            SHTnsKit.synthesis(
                mixed_cfg, world_cpu_spectral; prototype_θφ=world_cpu,
            ) : compound_extension._stage_vendor_call(
                :synthesis, identity, world_gpu_spectral, world_gpu;
                comm=world_gpu_comm,
            )),
        dist_synthesis=(:dist_synthesis, () -> iseven(rank) ?
            SHTnsKit.dist_synthesis(
                mixed_cfg, world_cpu_spectral; prototype_θφ=world_cpu,
            ) : compound_extension._stage_vendor_call(
                :dist_synthesis, identity, world_gpu_spectral, world_gpu;
                comm=world_gpu_comm,
            )),
        analysis_sphtor=(:analysis_sphtor, () -> iseven(rank) ?
            SHTnsKit.analysis_sphtor(mixed_cfg, world_cpu, world_cpu) :
            compound_extension._stage_vendor_call(
                :analysis_sphtor, identity, world_gpu, world_gpu;
                comm=world_gpu_comm,
            )),
        dist_analysis_sphtor=(:dist_analysis_sphtor, () -> iseven(rank) ?
            SHTnsKit.dist_analysis_sphtor(mixed_cfg, world_cpu, world_cpu) :
            compound_extension._stage_vendor_call(
                :dist_analysis_sphtor, identity, world_gpu, world_gpu;
                comm=world_gpu_comm,
            )),
        synthesis_sphtor=(:synthesis_sphtor, () -> iseven(rank) ?
            SHTnsKit.synthesis_sphtor(
                mixed_cfg, world_cpu_spectral, world_cpu_spectral;
                prototype_θφ=world_cpu,
            ) : compound_extension._stage_vendor_call(
                :synthesis_sphtor, identity, world_gpu_spectral,
                world_gpu_spectral, world_gpu; comm=world_gpu_comm,
            )),
        synthesis_sph_l_cplx=(:synthesis_sph_l_cplx,
            () -> iseven(rank) ? SHTnsKit.synthesis_sph_l_cplx(
                mixed_cfg, world_cpu_spectral, mixed_cfg.lmax;
                prototype_θφ=world_cpu,
            ) : compound_extension._stage_vendor_call(
                :synthesis_sph_l_cplx, identity,
                world_gpu_spectral, world_gpu; comm=world_gpu_comm,
            )),
        synthesis_tor_l_cplx=(:synthesis_tor_l_cplx,
            () -> iseven(rank) ? SHTnsKit.synthesis_tor_l_cplx(
                mixed_cfg, world_cpu_spectral, mixed_cfg.lmax;
                prototype_θφ=world_cpu,
            ) : compound_extension._stage_vendor_call(
                :synthesis_tor_l_cplx, identity,
                world_gpu_spectral, world_gpu; comm=world_gpu_comm,
            )),
        dist_synthesis_sphtor=(:dist_synthesis_sphtor, () -> iseven(rank) ?
            SHTnsKit.dist_synthesis_sphtor(
                mixed_cfg, world_cpu_spectral, world_cpu_spectral;
                prototype_θφ=world_cpu,
            ) : compound_extension._stage_vendor_call(
                :dist_synthesis_sphtor, identity, world_gpu_spectral,
                world_gpu_spectral, world_gpu; comm=world_gpu_comm,
            )),
        analysis_qst=(:analysis_qst, () -> iseven(rank) ?
            SHTnsKit.analysis_qst(
                mixed_cfg, world_cpu, world_cpu, world_cpu,
            ) : compound_extension._stage_vendor_call(
                :analysis_qst, identity, world_gpu, world_gpu, world_gpu;
                comm=world_gpu_comm,
            )),
        dist_analysis_qst=(:dist_analysis_qst, () -> iseven(rank) ?
            SHTnsKit.dist_analysis_qst(
                mixed_cfg, world_cpu, world_cpu, world_cpu,
            ) : compound_extension._stage_vendor_call(
                :dist_analysis_qst, identity, world_gpu, world_gpu, world_gpu;
                comm=world_gpu_comm,
            )),
        synthesis_qst=(:synthesis_qst, () -> iseven(rank) ?
            SHTnsKit.synthesis_qst(
                mixed_cfg, world_cpu_spectral, world_cpu_spectral,
                world_cpu_spectral; prototype_θφ=world_cpu,
            ) : compound_extension._stage_vendor_call(
                :synthesis_qst, identity, world_gpu_spectral,
                world_gpu_spectral, world_gpu_spectral, world_gpu;
                comm=world_gpu_comm,
            )),
        dist_synthesis_qst=(:dist_synthesis_qst, () -> iseven(rank) ?
            SHTnsKit.dist_synthesis_qst(
                mixed_cfg, world_cpu_spectral, world_cpu_spectral,
                world_cpu_spectral; prototype_θφ=world_cpu,
            ) : compound_extension._stage_vendor_call(
                :dist_synthesis_qst, identity, world_gpu_spectral,
                world_gpu_spectral, world_gpu_spectral, world_gpu;
                comm=world_gpu_comm,
            )),
    )
    # Invalid CPU dimensionality must rendezvous with a peer's vendor method
    # before either branch reports its local argument/backend error.
    split_batch_cases = (
        (:analysis_batch, (world_cpu,), (world_gpu,), (;)),
        (:analysis_batch!, (world_cpu_spectral, world_cpu),
         (world_gpu_spectral, world_gpu), (;)),
        (:synthesis_batch, (world_cpu_spectral,), (world_gpu_spectral, world_gpu),
         (; prototype_θφ=world_cpu)),
        (:synthesis_batch_cplx, (world_cpu_spectral,),
         (world_gpu_spectral, world_gpu), (; prototype_θφ=world_cpu)),
        (:synthesis_batch!, (world_cpu, world_cpu_spectral),
         (world_gpu, world_gpu_spectral, world_gpu), (; prototype_θφ=world_cpu)),
    )
    split_batch_values = (world_cpu, world_cpu_spectral, world_gpu, world_gpu_spectral)
    split_batch_before = map(value -> Array(parent(value)), split_batch_values)
    split_batch_stats = extension.parallel_gpu_stats()
    for (operation, cpu_args, gpu_args, kwargs) in split_batch_cases
        caught = try
            if iseven(rank)
                getfield(SHTnsKit, operation)(mixed_cfg, cpu_args...; kwargs...)
            else
                compound_extension._stage_vendor_call(
                    operation, identity, gpu_args...; comm=world_gpu_comm,
                )
            end
            nothing
        catch error
            error
        end
        @test caught isa ArgumentError
        @test occursin("storage/vendor/device mismatch", sprint(showerror, caught))
        @test MPI.Allreduce(caught isa ArgumentError ? 1 : 0, min,
                            MPI.COMM_WORLD) == 1
        MPI.Barrier(MPI.COMM_WORLD)
    end
    @test map(value -> Array(parent(value)), split_batch_values) == split_batch_before
    @test extension.parallel_gpu_stats() == split_batch_stats

    for (operation, call) in values(split_dispatch_cases)
        caught = try
            call()
            nothing
        catch error
            error
        end
        @test caught isa ArgumentError
        @test occursin(
            "$operation collective validation failed: " *
            "storage/vendor/device mismatch",
            sprint(showerror, caught),
        )
        @test MPI.Allreduce(caught isa ArgumentError ? 1 : 0, min,
                            MPI.COMM_WORLD) == 1
        MPI.Barrier(MPI.COMM_WORLD)
    end

    alternating_synthesis = iseven(rank) ?
        (world_cpu_spectral, world_gpu) :
        (world_gpu_spectral, world_cpu)
    caught_synthesis = false
    try
        SHTnsKit.dist_synthesis(
            mixed_cfg, first(alternating_synthesis);
            prototype_θφ=last(alternating_synthesis),
        )
    catch error
        caught_synthesis = error isa ArgumentError && occursin(
            "storage/vendor/device mismatch", sprint(showerror, error),
        )
    end
    @test MPI.Allreduce(caught_synthesis ? 1 : 0, min, MPI.COMM_WORLD) == 1
    MPI.Barrier(MPI.COMM_WORLD)
      finally
        lock(extension._PARALLEL_GPU_ADAPTER_LOCK) do
            delete!(extension._PARALLEL_GPU_ADAPTERS, :mock_mixed)
        end
      end
    end

    # Execute real staged mathematical callbacks on COMM_SELF while this test
    # normally runs under a two-rank WORLD. Before subgroup propagation was
    # fixed, the generic validators entered WORLD and mismatched/hung here.
    subgroup_cfg = SHTnsKit.create_gauss_config(2, 4; nlon=6)
    subgroup_dense = zeros(ComplexF64, 3, 3)
    subgroup_dense[1, 1] = 0.2
    subgroup_dense[2, 1] = -0.05
    subgroup_dense[3, 2] = 0.03 - 0.02im
    subgroup_pen = PencilArrays.Pencil(
        MockMPIArray, size(subgroup_dense), (2,), MPI.COMM_SELF,
    )
    subgroup_spectral = PencilArrays.PencilArray{ComplexF64}(
        undef, subgroup_pen,
    )
    parent(subgroup_spectral).data .= subgroup_dense
    subgroup_point = extension._staged_gpu_call(
        adapter, :mock_subgroup_point, MPI.COMM_SELF,
        host -> SHTnsKit.synthesis_point(
            subgroup_cfg, host, 0.31, -0.27,
        ), subgroup_spectral,
    )
    @test subgroup_point ≈ SHTnsKit.synthesis_point(
        subgroup_cfg, subgroup_dense, 0.31, -0.27,
    ) atol=3e-12
    subgroup_diagonal = extension._staged_gpu_call(
        adapter, :mock_subgroup_diagonal, MPI.COMM_SELF,
        host -> SHTnsKit.divergence_from_spheroidal(
            subgroup_cfg, host,
        ), subgroup_spectral,
    )
    @test subgroup_diagonal isa PencilArrays.PencilArray
    @test parent(subgroup_diagonal).data ≈
          SHTnsKit.divergence_from_spheroidal(
              subgroup_cfg, subgroup_dense,
          ) atol=3e-12
    MPI.Barrier(MPI.COMM_SELF)

    concurrent_adapter = extension.ParallelGPUAdapter(
        :mock_concurrent,
        value -> value isa MockMPIArray,
        _ -> MockMPIArray,
        _ -> 9,
        _ -> false,
        _ -> nothing,
        (T, n) -> Vector{T}(undef, n),
        copyto!, copyto!,
    )
    concurrent = MockMPIArray(Float32[0])
    tasks = [Threads.@spawn extension.allreduce!(
        concurrent, +, comm; adapter=concurrent_adapter,
        collective=(host, _op, _comm) -> (host .+= 1; host),
    ) for _ in 1:8]
    fetch.(tasks)
    @test concurrent.data == Float32[8]
    @test extension.parallel_gpu_cache_sizes().staging <= 2

    direct_adapter = extension.ParallelGPUAdapter(
        :mock_direct,
        value -> value isa MockMPIArray,
        _ -> Vector,
        _ -> 8,
        _ -> true,
        _ -> (sync_calls[] += 1),
        (T, n) -> Vector{T}(undef, n),
        copyto!,
        copyto!,
    )
    direct = function(device, _op, _comm)
        direct_collectives[] += 1
        device.data .+= 1
        return device
    end
    extension.allreduce!(buffer, +, Ref(:subgroup_b);
                         adapter=direct_adapter, collective=direct)
    @test buffer.data == Float32[5, 9, 13]
    @test direct_collectives[] == 1

    # The bounded cache may flush old entries, but must never grow past its cap.
    for n in 1:8
        value = MockMPIArray(fill(Float32(n), n))
        extension.allreduce!(value, +, Ref(Symbol(:comm_, n)); adapter,
                             collective=(host, _op, _comm) -> host)
        @test extension.parallel_gpu_cache_sizes().staging <= 2
    end

    stats = extension.parallel_gpu_stats()
    @test stats.direct_calls >= 1
    @test stats.staged_calls >= 10
    @test stats.direct_bytes >= sizeof(Float32) * 3
    @test stats.staged_bytes >= sizeof(Float32) * (3 + 3)
    extension.parallel_gpu_clear_caches!()
    @test extension.parallel_gpu_cache_sizes() ==
          (awareness=0, staging=0, native_host_plans=0)
end

const MPI_GPU_FIREWALL_GROUPS = (
    :matrix_to_spectral_pencil, :spectral_pencil_to_matrix,
    :analysis, :synthesis, :synthesis_cplx,
    :analysis_sphtor, :analysis_sphtor_cplx, :synthesis_sphtor,
    :synthesis_sphtor_cplx, :synthesis_sph, :synthesis_sph_cplx,
    :synthesis_tor, :synthesis_tor_cplx, :analysis_sphtor_l,
    :analysis_sphtor_ml, :synthesis_sphtor_l, :synthesis_sphtor_l_cplx,
    :synthesis_sphtor_ml, :synthesis_sph_l, :synthesis_sph_l_cplx,
    :synthesis_sph_ml, :synthesis_tor_l, :synthesis_tor_l_cplx,
    :synthesis_tor_ml, :analysis_qst,
    :analysis_qst_cplx, :synthesis_qst, :synthesis_qst_cplx,
    :analysis_qst_l, :analysis_qst_ml, :synthesis_qst_l,
    :synthesis_qst_l_cplx, :synthesis_qst_ml, :analysis_batch,
    :analysis_batch!, :synthesis_batch, :synthesis_batch!,
    :synthesis_batch_cplx, :analysis_sphtor_batch,
    :synthesis_sphtor_batch, :synthesis_sphtor_batch_cplx,
    :analysis_qst_batch, :synthesis_qst_batch, :synthesis_qst_batch_cplx,
    :analysis_packed, :analysis_packed_l, :analysis_packed_cplx,
    :analysis_packed_cplx_l, :analysis_packed_ml, :analysis_axisym,
    :analysis_axisym_l, :synthesis_packed, :synthesis_packed_l,
    :synthesis_packed_cplx, :synthesis_packed_cplx_l,
    :synthesis_packed_ml, :synthesis_axisym, :synthesis_axisym_l,
    :synthesis_point, :synthesis_point_cplx, :SH_to_lat,
    :SH_to_lat_cplx, :SHqst_to_point, :SHqst_to_lat,
    :SH_to_grad_point, :synthesis_grad, :synthesis_grad_l,
    :synthesis_grad_ml, :dist_analysis, :dist_synthesis,
    :dist_SH_to_lat, :dist_SH_to_point,
    :dist_SHqst_to_point, :dist_SHqst_to_lat,
    :dist_analysis_packed, :dist_synthesis_packed,
    :dist_analysis_packed_cplx, :dist_synthesis_packed_cplx,
    :dist_analysis_sphtor, :dist_synthesis_sphtor,
    :dist_analysis_qst, :dist_synthesis_qst,
    :dist_scalar_roundtrip!, :dist_vector_roundtrip!,
    :divergence_from_spheroidal, :divergence_from_spheroidal!,
    :spheroidal_from_divergence, :spheroidal_from_divergence!,
    :vorticity_from_toroidal, :vorticity_from_toroidal!,
    :toroidal_from_vorticity, :toroidal_from_vorticity!,
    :dist_apply_laplacian!, :SH_mul_mx, :dist_SH_mul_mx!,
    :dist_spatial_divergence,
    :dist_spatial_vorticity, :dist_scalar_laplacian,
    :dist_scalar_laplacian!, :dist_SH_Zrotate, :dist_SH_Yrotate,
    :dist_SH_Yrotate_allgatherm!, :dist_SH_Yrotate_truncgatherm!,
    :dist_SH_Yrotate90, :dist_SH_Xrotate90, :dist_SH_rotate_euler,
    :dist_SH_Zrotate_packed, :dist_SH_Yrotate_packed,
    :dist_SH_Yrotate90_packed, :dist_SH_Xrotate90_packed, :energy_scalar,
    :energy_scalar_l_spectrum, :energy_scalar_m_spectrum,
    :energy_vector, :energy_vector_l_spectrum, :energy_vector_m_spectrum,
    :enstrophy, :enstrophy_l_spectrum, :enstrophy_m_spectrum,
    :grid_energy_scalar, :grid_energy_vector, :grid_enstrophy,
    :dist_analysis!, :dist_synthesis!, :dist_analysis_sphtor!,
    :dist_synthesis_sphtor!, :dist_analysis_qst!, :dist_synthesis_qst!,
)

# One representative operation for every ordinary compound family that used
# to stage a complete mathematical call through the CPU.  The compound
# extension may only expose these families once their mathematics is native;
# collective-only staging remains tested directly in `test_mpi_gpu_policy`.
const MPI_GPU_ORDINARY_EARLY_ERROR_FAMILIES = (
    cfg_scalar=:analysis,
    cfg_vector=:analysis_sphtor,
    cfg_qst=:analysis_qst,
    degree_restricted=:analysis_qst_l,
    order_restricted=:analysis_qst_ml,
    packed_axisymmetric=:analysis_packed,
    batch_bang=:analysis_batch!,
    local_evaluation_gradient=:SH_to_grad_point,
    diagonal_operator=:divergence_from_spheroidal!,
    distributed_operator=:dist_scalar_laplacian!,
    rotation=:dist_SH_rotate_euler,
    packed_rotation=:dist_SH_Yrotate_packed,
    diagnostics=:energy_vector_l_spectrum,
    diagnostics_scalar_total=:enstrophy,
    diagnostics_vector_total=:energy_vector,
    distributed_storage_reduce=:distributed_spectral_reduce!,
    distributed_storage_1d=:dist_analysis_distributed,
    distributed_storage_2d=:dist_analysis_distributed_2d,
    distributed_compatibility=:dist_analysis,
    dense_compatibility=:dist_synthesis_qst_dense,
)

const MPI_GPU_NATIVE_FIREWALL_GROUPS = (
    :dist_analysis!, :dist_synthesis!, :dist_analysis_sphtor!,
    :dist_synthesis_sphtor!, :dist_analysis_qst!, :dist_synthesis_qst!,
)

function test_mpi_gpu_ordinary_early_errors(extension, compound_extension)
    callback_calls = Ref(0)
    copies = Ref(0)
    adapter = extension.ParallelGPUAdapter(
        :mock_ordinary_firewall,
        value -> value isa MockMPIArray,
        _ -> MockMPIArray,
        _ -> 1,
        _ -> false,
        _ -> nothing,
        (T, n) -> Vector{T}(undef, n),
        (host, device) -> (copies[] += 1; copyto!(host, device)),
        (device, host) -> (copies[] += 1; copyto!(device, host)),
    )
    for (family, operation) in pairs(MPI_GPU_ORDINARY_EARLY_ERROR_FAMILIES)
        sentinel = MockMPIArray(Float32[11, 13, 17])
        before_payload = copy(sentinel.data)
        before_stats = extension.parallel_gpu_stats()
        caught = try
            compound_extension._stage_vendor_call_with_adapter(
                adapter, MPI.COMM_SELF, operation,
                host -> begin
                    callback_calls[] += 1
                    host .= -1
                    host
                end,
                sentinel; mutated=(1,), validate_storage=false,
            )
            nothing
        catch error
            error
        end
        @test caught isa SHTnsKit.BackendUnavailableError
        if caught isa SHTnsKit.BackendUnavailableError
            @test caught.operation === operation
            @test occursin("device-native", caught.detail)
        end
        @test sentinel.data == before_payload
        @test extension.parallel_gpu_stats() == before_stats
        MPI.Barrier(MPI.COMM_SELF)
    end
    @test callback_calls[] == 0
    @test copies[] == 0
    return nothing
end

function test_mpi_gpu_source_contract(root::AbstractString, vendor::Symbol,
                                      compound_extension)
    parallel_extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    parallel_gpu = read(joinpath(root, "ext", "ParallelGPU.jl"), String)
    @test occursin("WeakRef", parallel_gpu)
    @test occursin("ReentrantLock", parallel_gpu)
    @test occursin("function exchange!", parallel_gpu)
    @test occursin("function allreduce!", parallel_gpu)
    for wrapper in (
        "LogicalIndex", "PermutedDimsArray", "Adjoint", "Transpose",
        "Symmetric", "Hermitian", "Diagonal", "Bidiagonal", "Tridiagonal",
        "UpperTriangular", "LowerTriangular", "UnitUpperTriangular",
        "UnitLowerTriangular",
    )
        @test occursin(wrapper, parallel_gpu)
    end
    @test occursin("logical_id", parallel_gpu)
    @test occursin("logical_owner", parallel_gpu)
    @test occursin("adapter.with_device(device)", parallel_gpu)
    @test occursin("_with_owner_device", parallel_gpu)
    @test occursin("_device_to_host_snapshot!", parallel_gpu)
    @test occursin("_host_to_device_snapshot!", parallel_gpu)
    @test occursin("local_device_mismatch", parallel_gpu)
    @test occursin("_validate_parallel_storage!(comm, :exchange", parallel_gpu)
    @test occursin("Base.mightalias", parallel_gpu)
    @test occursin("_gpu_transpose_forward!", parallel_gpu)
    @test occursin("_gpu_transpose_inverse!", parallel_gpu)
    @test !occursin("using CUDA", parallel_gpu)
    @test !occursin("using AMDGPU", parallel_gpu)

    extension_file = vendor === :cuda ? "SHTnsKitParallelCUDAExt.jl" :
                     "SHTnsKitParallelAMDGPUExt.jl"
    source = read(joinpath(root, "ext", extension_file), String)
    @test occursin("_register_parallel_gpu_adapter!", source)
    @test occursin(vendor === :cuda ? "MPI.has_cuda" : "MPI.has_rocm", source)
    @test occursin("_parallel_root_buffer(value)", source)
    @test occursin(vendor === :cuda ? "CUDA.device(" : "AMDGPU.device(", source)
    @test occursin(vendor === :cuda ? "CUDA.device!(f, device)" :
                                    "AMDGPU.device!(f, device)", source)
    @test occursin("n == 0 && return Vector{T}(undef, 0)", source)
    @test occursin("_gpu_transpose_forward!", source)
    @test occursin("_gpu_transpose_inverse!", source)
    empty_pinned = vendor === :cuda ?
        compound_extension._cuda_pinned(Float32, 0) :
        compound_extension._amdgpu_pinned(Float32, 0)
    @test empty_pinned isa Vector{Float32}
    @test isempty(empty_pinned)

    runner_file = vendor === :cuda ?
        joinpath(root, "test", "gpu", "cuda", "mpi_runtests.jl") :
        joinpath(root, "test", "gpu", "amdgpu", "mpi_runtests.jl")
    runner = read(runner_file, String)
    @test isdefined(@__MODULE__, :run_mpi_gpu_full_parity)
    @test occursin("run_mpi_gpu_full_parity(", runner)
    for family in (
        "native scalar/vector/QST transpose nonzero numerics",
        "ordinary GPU APIs reject before whole-call staging",
        "actual allocation device cache key",
        "repeated-plan cache and residency",
    )
        @test occursin(family, read(@__FILE__, String))
    end
    matrix_source = read(@__FILE__, String)
    for call_marker in (
        "MPI_GPU_ORDINARY_EARLY_ERROR_FAMILIES",
        "public_calls = (",
        "dist_analysis!(", "dist_synthesis!(",
        "dist_analysis_sphtor!(", "dist_synthesis_sphtor!(",
        "dist_analysis_qst!(", "dist_synthesis_qst!(",
        "forced_staged_adapter", "parallel_gpu_stats() == before_stats",
    )
        @test occursin(call_marker, matrix_source)
    end

    project = read(joinpath(root, "Project.toml"), String)
    trigger = vendor === :cuda ? "SHTnsKitParallelCUDAExt" :
              "SHTnsKitParallelAMDGPUExt"
    @test occursin(trigger, project)

    firewall_path = joinpath(root, "ext", "ParallelGPUVendorFirewall.jl")
    @test isfile(firewall_path)
    isfile(firewall_path) || return
    firewall = read(firewall_path, String)
    @test compound_extension !== nothing
    @test isempty(Test.detect_ambiguities(
        SHTnsKit, compound_extension; recursive=true,
    ))
    # detect_ambiguities does not report the ambiguities a vendor alias that
    # is not a PencilArray subtype causes, so probe dispatch directly.
    @test compound_extension.VendorPencilArray <: PencilArrays.PencilArray
    probe = firewall_dispatch_probe(
        compound_extension, (T, N) -> compound_extension.VendorArrayType{T,N},
    )
    for failure in first(probe.failures, 5)
        @error "firewall dispatch probe failed" failure
    end
    @test probe.probes > 1000
    @test length(probe.failures) == 0
    vendor_adapter = vendor === :cuda ?
        compound_extension.CUDA_PARALLEL_ADAPTER :
        compound_extension.AMDGPU_PARALLEL_ADAPTER
    @test vendor_adapter.device_to_host! ===
          parallel_extension._bulk_device_to_host!
    @test vendor_adapter.host_to_device! ===
          parallel_extension._bulk_host_to_device!
    for api in MPI_GPU_FIREWALL_GROUPS
        function_object = getfield(SHTnsKit, api)
        owned = filter(method -> method.module === compound_extension,
                       methods(function_object))
        @test !isempty(owned)
        @test all(method -> method.module !== parallel_extension, owned)
    end
    classifications = Dict(
        api => (api in MPI_GPU_NATIVE_FIREWALL_GROUPS ? :native : :early_error)
        for api in MPI_GPU_FIREWALL_GROUPS
    )
    @test count(==(:native), values(classifications)) == 6
    @test count(==(:early_error), values(classifications)) ==
          length(MPI_GPU_FIREWALL_GROUPS) - 6
    @test all(value -> value in (:native, :early_error),
              values(classifications))
    @test occursin("include(\"ParallelGPUVendorFirewall.jl\")", source)
    @test !occursin("Array(", firewall)
    @test !occursin("collect(", firewall)
    @test !occursin("allowscalar", firewall)
    @test occursin("_dist_transpose_gpu_analysis!", firewall)
    @test length(findall("ParallelExt._staged_gpu_call(", firewall)) == 1
    @test length(findall(
        "function _internal_staged_vendor_call_with_adapter(", firewall,
    )) == 1
    @test occursin("return _ordinary_vendor_backend_unavailable(operation)",
                   firewall)
    @test occursin(":synthesis_sph_l_cplx", firewall)
    @test occursin(":synthesis_tor_l_cplx", firewall)
    @test !occursin("prototype_θφ::VendorPencilArray", firewall)
    @test !occursin("prototype isa VendorPencilArray || throw", firewall)
    @test occursin("prototype_θφ::PencilArrays.PencilArray", firewall)
    @test occursin("_validate_parallel_storage!", firewall)
    for marker in (
        ":dist_synthesis_dense", ":dist_synthesis_sphtor_dense",
        ":dist_synthesis_qst_dense", "host_aminus",
    )
        @test occursin(marker, firewall)
    end
    vendor_array = compound_extension.VendorArray
    for api in (:dist_synthesis, :dist_synthesis_sphtor,
                :dist_synthesis_qst)
        owned = filter(method -> method.module === compound_extension,
                       methods(getfield(SHTnsKit, api)))
        @test any(owned) do method
            parameters = Base.unwrap_unionall(method.sig).parameters
            length(parameters) >= 3 &&
                typeintersect(parameters[3], vendor_array) !== Union{} &&
                typeintersect(parameters[3], PencilArrays.PencilArray) === Union{}
        end
    end

    transpose_source = read(
        joinpath(root, "ext", "ParallelTransposeTransforms.jl"), String,
    )
    @test occursin(
        r"_validate_parallel_storage!\(\s*plan.comm, operation, plan.F_buf, plan.F_buf2,\s*spatial\.\.\., spectral\.\.\.,",
        transpose_source,
    )
    @test occursin("_scalar_precision_code(real_type)", transpose_source)
    @test occursin("comm, communicator(prototype)", transpose_source)
    @test occursin("array_type === prototype_array_type", transpose_source)
    @test occursin("_with_owner_device", transpose_source)

    # Dealiased decompositions may leave a rank owning only Fourier bins above
    # mmax. The native kernel offset must still be the rank's real first bin,
    # never a fallback to m=0 when `plan.m_local` is empty.
    cfg = SHTnsKit.create_gauss_config(3, 5; nlon=13)
    plan = SHTnsKit.DistTransposePlan(
        cfg; comm=MPI.COMM_WORLD, nlev=1, with_vector=true,
    )
    expected_first_m = first(PencilArrays.range_local(
        PencilArrays.pencil(plan.F_buf),
    )[1]) - 1
    @test compound_extension._first_m(plan) == expected_first_m
end
