using Test
using SHTnsKit
using CUDA
using GPUArrays
using GPUArraysCore
import KernelAbstractions

include("../wrapper_reference.jl")
include("../test_mres.jl")
if CUDA.functional()
    run_gpu_mres_tests()
else
    @test_skip CUDA.functional()
end

include("../../parity/scalar_full.jl")
include("../../parity/scalar_variants.jl")
include("../../parity/sphtor_full.jl")
include("../../parity/qst_full.jl")
include("../../parity/vector_variants.jl")
include("../../parity/vector_batches.jl")
include("../../parity/local_evaluation.jl")
include("../../parity/operators.jl")
include("../../parity/rotations.jl")
include("../../parity/shtns37_fixtures.jl")

struct CUDAScalarAdapter <: ScalarParityAdapter end
struct CUDAOperatorAdapter <: GPUOperatorAdapter end
struct CUDARotationAdapter <: RotationParityAdapter end
rotation_place(::CUDARotationAdapter, value) = CuArray(value)
rotation_collect(::CUDARotationAdapter, value) = Array(value)
rotation_resident(::CUDARotationAdapter, value) = @test value isa CUDA.AnyCuArray
rotation_z(::CUDARotationAdapter, cfg, input, angle, output) =
    SH_Zrotate(GPU(), cfg, input, angle, output)
rotation_y(::CUDARotationAdapter, cfg, input, angle, output) =
    SH_Yrotate(GPU(), cfg, input, angle, output)
rotation_apply_real(::CUDARotationAdapter, rotation, input, output) =
    shtns_rotation_apply_real(GPU(), rotation, input, output)
rotation_apply_cplx(::CUDARotationAdapter, rotation, input, output) =
    shtns_rotation_apply_cplx(GPU(), rotation, input, output)
operator_place(::CUDAOperatorAdapter, value) = CuArray(value)
function operator_strided_place(::CUDAOperatorAdapter, value::AbstractVector)
    storage = CUDA.zeros(eltype(value), 2length(value))
    result = @view storage[1:2:end]
    copyto!(result, CuArray(value))
    return result
end
function operator_overlapping_matrix_place(::CUDAOperatorAdapter, value)
    storage = CUDA.zeros(eltype(value), size(value, 1) + 1, size(value, 2))
    source = @view storage[1:size(value, 1), :]
    destination = @view storage[2:(size(value, 1) + 1), :]
    copyto!(source, CuArray(value))
    return source, destination
end
operator_collect(::CUDAOperatorAdapter, value) = Array(value)
operator_resident(::CUDAOperatorAdapter, value) = @test value isa CUDA.AnyCuArray
function place(::CUDAScalarAdapter, ::SHTConfig, value, ::Symbol)
    if value isa PermutedDimsArray
        padded = zeros(eltype(value), size(value, 1), 2size(value, 2))
        @views padded[:, 1:2:end] .= value
        storage = CuArray(padded)
        return @view storage[:, 1:2:end]
    end
    return CuArray(value)
end
collect_result(::CUDAScalarAdapter, value, ::SHTConfig) = Array(value)
analysis_call(::CUDAScalarAdapter, cfg, field; use_rfft=false) =
    analysis(GPU(), cfg, field; use_rfft)
synthesis_call(::CUDAScalarAdapter, cfg, coefficients, _prototype;
               real_output, use_rfft=false) =
    synthesis(GPU(), cfg, coefficients; real_output, use_rfft)
synthesis_cplx_call(::CUDAScalarAdapter, cfg, coefficients, _prototype) =
    synthesis_cplx(cfg, coefficients)
assert_resident(::CUDAScalarAdapter, value) = @test value isa CUDA.AnyCuArray

struct CUDAVectorAdapter <: VectorParityAdapter end
vector_place(::CUDAVectorAdapter, ::SHTConfig, value, ::Symbol) = CuArray(value)
vector_collect(::CUDAVectorAdapter, value, ::SHTConfig) = Array(value)
vector_resident(::CUDAVectorAdapter, value) = @test value isa CUDA.AnyCuArray
vector_analysis(::CUDAVectorAdapter, cfg, Vt, Vp; use_rfft=false) =
    analysis_sphtor(GPU(), cfg, Vt, Vp; use_rfft)
vector_analysis_cplx(::CUDAVectorAdapter, cfg, Vt, Vp) =
    analysis_sphtor_cplx(GPU(), cfg, Vt, Vp)
vector_synthesis(::CUDAVectorAdapter, cfg, S, T, _prototype;
                 real_output=true, use_rfft=false) =
    synthesis_sphtor(GPU(), cfg, S, T; real_output, use_rfft)
vector_synthesis_cplx(::CUDAVectorAdapter, cfg, S, T, _prototype) =
    synthesis_sphtor_cplx(GPU(), cfg, S, T)
vector_sph(::CUDAVectorAdapter, cfg, S, _prototype; real_output=true) =
    synthesis_sph(GPU(), cfg, S; real_output)
vector_sph_cplx(::CUDAVectorAdapter, cfg, S, _prototype) =
    synthesis_sph_cplx(GPU(), cfg, S)
vector_tor(::CUDAVectorAdapter, cfg, T, _prototype; real_output=true) =
    synthesis_tor(GPU(), cfg, T; real_output)
vector_tor_cplx(::CUDAVectorAdapter, cfg, T, _prototype) =
    synthesis_tor_cplx(GPU(), cfg, T)

struct CUDAQSTAdapter <: QSTParityAdapter end
qst_place(::CUDAQSTAdapter, ::SHTConfig, value, ::Symbol) = CuArray(value)
qst_collect(::CUDAQSTAdapter, value, ::SHTConfig) = Array(value)
qst_resident(::CUDAQSTAdapter, value) = @test value isa CUDA.AnyCuArray
qst_analysis(::CUDAQSTAdapter, cfg, Vr, Vt, Vp; use_rfft=false) =
    analysis_qst(GPU(), cfg, Vr, Vt, Vp; use_rfft)
qst_analysis_cplx(::CUDAQSTAdapter, cfg, Vr, Vt, Vp) =
    analysis_qst_cplx(GPU(), cfg, Vr, Vt, Vp)
qst_synthesis(::CUDAQSTAdapter, cfg, Q, S, Tlm, prototype;
              real_output=true, use_rfft=false) =
    synthesis_qst(GPU(), cfg, Q, S, Tlm; prototype, real_output, use_rfft)
qst_synthesis_cplx(::CUDAQSTAdapter, cfg, Q, S, Tlm, prototype) =
    synthesis_qst_cplx(GPU(), cfg, Q, S, Tlm; prototype)
qst_analysis_inferred(::CUDAQSTAdapter, cfg, Vr, Vt, Vp) =
    analysis_qst(cfg, Vr, Vt, Vp)
qst_analysis_cplx_inferred(::CUDAQSTAdapter, cfg, Vr, Vt, Vp) =
    analysis_qst_cplx(cfg, Vr, Vt, Vp)
qst_synthesis_inferred(::CUDAQSTAdapter, cfg, Q, S, Tlm, prototype) =
    synthesis_qst(cfg, Q, S, Tlm; prototype)
qst_synthesis_cplx_inferred(::CUDAQSTAdapter, cfg, Q, S, Tlm, _prototype) =
    synthesis_qst_cplx(cfg, Q, S, Tlm)

struct CUDALocalEvaluationAdapter <: LocalEvaluationAdapter end
function local_place(::CUDALocalEvaluationAdapter, value)
    if value isa SubArray
        if ndims(value) == 1
            padded = zeros(eltype(value), 2length(value))
            @views padded[1:2:end] .= value
            storage = CuArray(padded)
            return @view storage[1:2:end]
        end
        padded = zeros(eltype(value), size(value, 1), 2size(value, 2))
        @views padded[:, 1:2:end] .= value
        storage = CuArray(padded)
        return @view storage[:, 1:2:end]
    end
    return CuArray(value)
end
function local_collect(::CUDALocalEvaluationAdapter, value)
    host = Array(value)
    return ndims(host) == 0 ? host[] : host
end
local_scalar(::CUDALocalEvaluationAdapter, cfg, coefficients, cost, phi) =
    synthesis_point(GPU(), cfg, coefficients, cost, phi)
local_scalar_cplx(::CUDALocalEvaluationAdapter, cfg, coefficients, cost, phi) =
    synthesis_point_cplx(GPU(), cfg, coefficients, cost, phi)
local_lat(::CUDALocalEvaluationAdapter, cfg, coefficients, cost; kwargs...) =
    SH_to_lat(GPU(), cfg, coefficients, cost; kwargs...)
local_lat_cplx(::CUDALocalEvaluationAdapter, cfg, coefficients, cost; kwargs...) =
    SH_to_lat_cplx(GPU(), cfg, coefficients, cost; kwargs...)
local_qst_point(::CUDALocalEvaluationAdapter, cfg, Q, S, Tlm, cost, phi) =
    SHqst_to_point(GPU(), cfg, Q, S, Tlm, cost, phi)
local_qst_lat(::CUDALocalEvaluationAdapter, cfg, Q, S, Tlm, cost; kwargs...) =
    SHqst_to_lat(GPU(), cfg, Q, S, Tlm, cost; kwargs...)
local_grad_point(::CUDALocalEvaluationAdapter, cfg, Dr, S, cost, phi) =
    SH_to_grad_point(GPU(), cfg, Dr, S, cost, phi)
local_assert_resident(::CUDALocalEvaluationAdapter, value) =
    @test value isa CUDA.AnyCuArray
qst_supports_strided_views(::CUDAQSTAdapter) = true
function qst_strided_place(::CUDAQSTAdapter, ::SHTConfig, value, ::Symbol)
    padded = zeros(eltype(value), size(value, 1), 2size(value, 2))
    @views padded[:, 1:2:end] .= value
    storage = CuArray(padded)
    return @view storage[:, 1:2:end]
end
function assert_warm_device_noalloc(::CUDAScalarAdapter, call)
    call()
    CUDA.synchronize()
    @test CUDA.@allocated(begin
        call()
        CUDA.synchronize()
    end) == 0
    return nothing
end

# Mixed Float32/Float64 operands promote as on the CPU: an S/T pair, and all
# three QST components, take the widest precision among them. Values are
# compared with the CPU transform of the promoted operands.
function test_mixed_precision_single_field()
    @testset "Mixed Float32/Float64 single-field vector and QST transforms" begin
        cfg = create_gauss_config(5, 8; nlon=14, norm=:schmidt,
                                  real_norm=true, cs_phase=false)
        dense = _variant_coefficients(cfg, Float64)
        Q64, S64, T64 = dense, 2 .* conj.(dense), -0.5 .* dense
        Vr64 = synthesis(cfg, Q64)
        Vt64, Vp64 = synthesis_sphtor(cfg, S64, T64)
        promoted(xs...) = (P = promote_type(map(eltype, xs)...); map(x -> P.(x), xs))
        function test_promoted(result, cpu, expected)
            for (values, reference, exact) in zip(result, cpu, expected)
                @test values isa CUDA.AnyCuArray
                @test eltype(values) === eltype(reference)
                tol = _vector_tol(real(eltype(reference)))
                @test Array(values) ≈ exact atol=tol.atol rtol=tol.rtol
            end
        end

        plan = SHTPlan(cfg)
        for (A, B) in ((Float32, Float64), (Float64, Float32))
            Vt, Vp = A.(Vt64), B.(Vp64)
            cpu = analysis_sphtor(cfg, Vt, Vp)
            exact = analysis_sphtor(cfg, promoted(Vt, Vp)...)
            test_promoted(analysis_sphtor(cfg, CuArray(Vt), CuArray(Vp)), cpu, exact)
            # A host component moves to the device without being narrowed.
            test_promoted(analysis_sphtor(cfg, CuArray(Vt), Vp), cpu, exact)
            # Planned outputs carry the promoted precision the allocating call returns.
            S_out = CUDA.zeros(ComplexF64, size(S64)); T_out = similar(S_out)
            analysis_sphtor!(plan, S_out, T_out, CuArray(Vt), CuArray(Vp))
            test_promoted((S_out, T_out), cpu, exact)
            @test_throws ArgumentError analysis_sphtor!(
                plan, CUDA.zeros(ComplexF32, size(S64)),
                CUDA.zeros(ComplexF32, size(S64)), CuArray(Vt), CuArray(Vp),
            )

            S, Tlm = Complex{A}.(S64), Complex{B}.(T64)
            cpu = synthesis_sphtor(cfg, S, Tlm)
            exact = synthesis_sphtor(cfg, promoted(S, Tlm)...)
            test_promoted(synthesis_sphtor(cfg, CuArray(S), CuArray(Tlm)), cpu, exact)
            Vt_out = CUDA.zeros(Float64, size(Vt64)); Vp_out = similar(Vt_out)
            synthesis_sphtor!(plan, Vt_out, Vp_out, CuArray(S), CuArray(Tlm))
            test_promoted((Vt_out, Vp_out), cpu, exact)
            test_promoted(synthesis_sphtor_cplx(cfg, CuArray(S), CuArray(Tlm)),
                          synthesis_sphtor_cplx(cfg, S, Tlm),
                          synthesis_sphtor_cplx(cfg, promoted(S, Tlm)...))
        end

        for wide in 1:3
            fields = ntuple(i -> (i == wide ? Float64 : Float32).((Vr64, Vt64, Vp64)[i]), 3)
            test_promoted(analysis_qst(cfg, map(CuArray, fields)...),
                          analysis_qst(cfg, fields...),
                          analysis_qst(cfg, promoted(fields...)...))
            test_promoted(analysis_qst_l(cfg, map(CuArray, fields)..., 3),
                          analysis_qst_l(cfg, fields..., 3),
                          analysis_qst_l(cfg, promoted(fields...)..., 3))
            spectra = ntuple(i -> (i == wide ? ComplexF64 : ComplexF32).((Q64, S64, T64)[i]), 3)
            for real_output in (true, false)
                test_promoted(synthesis_qst(cfg, map(CuArray, spectra)...; real_output),
                              synthesis_qst(cfg, spectra...; real_output),
                              synthesis_qst(cfg, promoted(spectra...)...; real_output))
                test_promoted(synthesis_qst_l(cfg, map(CuArray, spectra)..., 3; real_output),
                              synthesis_qst_l(cfg, spectra..., 3; real_output),
                              synthesis_qst_l(cfg, promoted(spectra...)..., 3; real_output))
            end
        end

        stored_im = 1
        physical_m = stored_im * cfg.mres
        ltr = cfg.lmax
        Sl64 = S64[(physical_m + 1):end, physical_m + 1]
        Tl64 = T64[(physical_m + 1):end, physical_m + 1]
        for (A, B) in ((ComplexF32, ComplexF64), (ComplexF64, ComplexF32))
            Sl, Tl = A.(Sl64), B.(Tl64)
            exact = synthesis_sphtor_ml(cfg, stored_im, promoted(Sl, Tl)..., ltr)
            test_promoted(synthesis_sphtor_ml(cfg, stored_im, CuArray(Sl), CuArray(Tl), ltr),
                          synthesis_sphtor_ml(cfg, stored_im, Sl, Tl, ltr), exact)
            Vtl, Vpl = A.(exact[1]), B.(exact[2])
            test_promoted(analysis_sphtor_ml(cfg, stored_im, CuArray(Vtl), CuArray(Vpl), ltr),
                          analysis_sphtor_ml(cfg, stored_im, Vtl, Vpl, ltr),
                          analysis_sphtor_ml(cfg, stored_im, promoted(Vtl, Vpl)..., ltr))
        end
        Ql64 = Q64[(physical_m + 1):end, physical_m + 1]
        for wide in 1:3
            modes = ntuple(i -> (i == wide ? ComplexF64 : ComplexF32).((Ql64, Sl64, Tl64)[i]), 3)
            exact = synthesis_qst_ml(cfg, stored_im, promoted(modes...)..., ltr)
            test_promoted(synthesis_qst_ml(cfg, stored_im, map(CuArray, modes)..., ltr),
                          synthesis_qst_ml(cfg, stored_im, modes..., ltr), exact)
            profiles = ntuple(i -> (i == wide ? ComplexF64 : ComplexF32).(exact[i]), 3)
            test_promoted(analysis_qst_ml(cfg, stored_im, map(CuArray, profiles)..., ltr),
                          analysis_qst_ml(cfg, stored_im, profiles..., ltr),
                          analysis_qst_ml(cfg, stored_im, promoted(profiles...)..., ltr))
        end
    end
    return nothing
end

struct SafeFallbackRedispatchError <: Exception end

struct SafeFallbackArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
end

Base.size(array::SafeFallbackArray) = size(array.parent)
Base.getindex(array::SafeFallbackArray, indices...) = getindex(array.parent, indices...)
SHTnsKit.on_device(::SafeFallbackArray) = SHTnsKit.GPU()
SHTnsKit.analysis(::SHTConfig, ::SafeFallbackArray; kwargs...) =
    throw(SafeFallbackRedispatchError())
SHTnsKit.synthesis(::SHTConfig, ::SafeFallbackArray; kwargs...) =
    throw(SafeFallbackRedispatchError())

@testset "CUDA backend routing" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitGPUExt)
    @test extension !== nothing
    @test SHTnsKit._GPU_LOOP_AVAILABLE[]
    cfg_memory = create_gauss_config(4, 6; nlon=9)
    spatial_bytes = cfg_memory.nlat * cfg_memory.nlon * 16
    coefficient_bytes = (cfg_memory.lmax + 1) * (cfg_memory.mmax + 1) * 16
    legendre_bytes = cfg_memory.nlat * (cfg_memory.lmax + 1) *
                     (cfg_memory.mmax + 1) * 8
    @test estimate_memory_usage(cfg_memory, :vector) ==
          4spatial_bytes + 2coefficient_bytes + 6legendre_bytes
    test_gpu_rotation_contract(
        extension, CuArray{ComplexF32,1}, CuArray{ComplexF32,1},
    )
    run_shared_rotation_kernel_reference(
        extension.GPUCommon, KernelAbstractions.CPU(),
    )
    test_angle_axis_pi_singularity()
    if CUDA.functional()
        loop_values = CUDA.zeros(Int32, 8)
        @sht_loop loop_values[i] = Int32(i) over i ∈ eachindex(loop_values)
        @test Array(loop_values) == Int32.(1:8)
        plan = extension.create_cufft_plan(2, 8)
        input = CUDA.rand(ComplexF64, 2, 8)
        transformed = copy(input)
        extension.gpu_fft!(plan, transformed)
        extension.gpu_ifft!(plan, transformed)
        @test Array(transformed) ≈ Array(input)

        run_rotation_parity(CUDARotationAdapter())
        test_shtns37_gpu_fixtures(
            CuArray, value -> (@test value isa CUDA.AnyCuArray),
        )
    else
        @test_skip CUDA.functional()
    end
    test_gpu_vector_variant_contract(
        extension, CuArray{Float32,2}, CuArray{ComplexF32,2},
        CuArray{ComplexF32,1}, CuArray{Float32,3}, CuArray{ComplexF32,3},
    )
    @test isdefined(extension.GPUCommon, :scalar_analysis_kernel!)
    @test isdefined(extension.GPUCommon, :scalar_synthesis_kernel!)
    @test isdefined(extension.GPUCommon, :coefficient_conversion_kernel!)
    @test isdefined(extension.GPUCommon, :coefficient_batch_conversion_kernel!)
    @test isdefined(extension.GPUCommon, :ScalarWorkspaceCache)
    @test isdefined(extension.GPUCommon, :vector_derivative_table_kernel!)
    @test isdefined(extension.GPUCommon, :vector_analysis_kernel!)
    @test isdefined(extension.GPUCommon, :vector_synthesis_kernel!)
    @test isdefined(extension.GPUCommon, :vector_diagonal_kernel!)
    @test isdefined(extension.GPUCommon, :operator_matrix_kernel!)
    @test isdefined(extension.GPUCommon, :packed_operator_kernel!)
    @test which(mul_ct_matrix,
                Tuple{SHTnsKit.GPU,SHTConfig,CuArray{Float32,1}}).module === extension
    @test which(st_dt_matrix,
                Tuple{SHTnsKit.GPU,SHTConfig,CuArray{Float32,1}}).module === extension
    @test which(SH_mul_mx,
                Tuple{SHTnsKit.GPU,SHTConfig,CuArray{Float32,1},
                      CuArray{ComplexF32,1},CuArray{ComplexF32,1}}).module === extension
    @test which(divergence_from_spheroidal,
                Tuple{SHTnsKit.GPU,SHTConfig,CuArray{ComplexF32,2}}).module === extension
    @test which(gpu_apply_laplacian!,
                Tuple{SHTConfig,CuArray{ComplexF32,2}}).module === extension
    @test isdefined(extension.GPUCommon, :local_scalar_kernel!)
    @test isdefined(extension.GPUCommon, :local_complex_kernel!)
    @test isdefined(extension.GPUCommon, :local_qst_kernel!)
    @test isdefined(extension, :_cuda_scalar_analysis)
    @test isdefined(extension, :_cuda_scalar_synthesis)
    @test isdefined(extension, :_cuda_clear_scalar_cache!)
    @test isdefined(extension, :_cuda_scalar_analysis_direct!)
    @test isdefined(extension, :_cuda_batch_analysis_direct!)
    @test isdefined(extension, :_cuda_vector_analysis)
    @test isdefined(extension, :_cuda_vector_synthesis)
    @test isdefined(extension, :_cuda_vector_analysis_direct!)
    @test isdefined(extension, :_cuda_vector_synthesis_direct!)
    @test isdefined(extension, :_cuda_local_tables)
    @test extension._CUDA_LOCAL_CACHE.max_per_device == 8
    @test which(synthesis_point,
                Tuple{SHTnsKit.GPU,SHTConfig,CuArray{ComplexF32,2},Float32,Float32}).module === extension
    @test which(SHqst_to_point,
                Tuple{SHTnsKit.GPU,SHTConfig,CuArray{ComplexF32,1},CuArray{ComplexF32,1},CuArray{ComplexF32,1},Float32,Float32}).module === extension
    @test fieldnames(extension.CUDAScalarTables) == (:sint, :weights, :Plm, :scales)
    @test isdefined(extension, :CUDAVectorTables)
    @test isdefined(extension, :_CUDA_VECTOR_CACHE)
    @test isdefined(extension, :_cuda_vector_tables)
    @test extension._CUDA_SCALAR_CACHE.max_per_device == 8
    @test extension._CUDA_VECTOR_CACHE.max_per_device == 8
    @test which(
        analysis_sphtor,
        Tuple{SHTConfig,CuArray{Float32,2},CuArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis_sphtor,
        Tuple{SHTConfig,CuArray{ComplexF32,2},CuArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis_qst,
        Tuple{SHTConfig,CuArray{Float32,2},CuArray{Float32,2},CuArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis_qst,
        Tuple{SHTConfig,CuArray{ComplexF32,2},CuArray{ComplexF32,2},CuArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis_sphtor,
        Tuple{SHTnsKit.GPU,SHTConfig,CuArray{Float32,2},CuArray{Float32,2}},
    ).module === SHTnsKit
    for (function_name, signature) in (
        (:analysis_packed, Tuple{SHTConfig,CuArray{Float32,1}}),
        (:synthesis_packed, Tuple{SHTConfig,CuArray{ComplexF32,1}}),
        (:analysis_packed_l, Tuple{SHTConfig,CuArray{Float32,1},Int}),
        (:synthesis_packed_l, Tuple{SHTConfig,CuArray{ComplexF32,1},Int}),
        (:analysis_axisym, Tuple{SHTConfig,CuArray{Float32,1}}),
        (:synthesis_axisym, Tuple{SHTConfig,CuArray{ComplexF32,1}}),
        (:analysis_axisym_l, Tuple{SHTConfig,CuArray{Float32,1},Int}),
        (:synthesis_axisym_l, Tuple{SHTConfig,CuArray{ComplexF32,1},Int}),
        (:analysis_packed_ml, Tuple{SHTConfig,Int,CuArray{ComplexF32,1},Int}),
        (:synthesis_packed_ml, Tuple{SHTConfig,Int,CuArray{ComplexF32,1},Int}),
        (:analysis_packed_cplx, Tuple{SHTConfig,CuArray{ComplexF32,2}}),
        (:synthesis_packed_cplx, Tuple{SHTConfig,CuArray{ComplexF32,1}}),
        (:analysis_packed_cplx_l, Tuple{SHTConfig,CuArray{ComplexF32,2},Int}),
        (:synthesis_packed_cplx_l, Tuple{SHTConfig,CuArray{ComplexF32,1},Int}),
        (:analysis_batch, Tuple{SHTConfig,CuArray{Float32,3}}),
        (:analysis_batch!, Tuple{SHTConfig,CuArray{ComplexF32,3},CuArray{Float32,3}}),
        (:synthesis_batch, Tuple{SHTConfig,CuArray{ComplexF32,3}}),
        (:synthesis_batch!, Tuple{SHTConfig,CuArray{Float32,3},CuArray{ComplexF32,3}}),
        (:synthesis_batch_cplx, Tuple{SHTConfig,CuArray{ComplexF32,3}}),
    )
        @test which(getproperty(SHTnsKit, function_name), signature).module === extension
    end
    @test which(
        analysis!, Tuple{SHTPlan,CuArray{ComplexF32,2},CuArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis!, Tuple{SHTPlan,CuArray{Float32,2},CuArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis_sphtor!,
        Tuple{SHTPlan,CuArray{ComplexF32,2},CuArray{ComplexF32,2},
              CuArray{Float32,2},CuArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis_sphtor!,
        Tuple{SHTPlan,CuArray{Float32,2},CuArray{Float32,2},
              CuArray{ComplexF32,2},CuArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis!,
        Tuple{SHTnsKit.GPU,SHTPlan,CuArray{ComplexF32,2},CuArray{Float32,2}},
    ).module === extension
    @test which(gpu_clear_cache!, Tuple{SHTnsKit.GPU}).module === SHTnsKit
    @test hasmethod(
        SHTnsKit._gpu_adapter_clear_cache!, Tuple{typeof(extension.CUDA_ADAPTER)},
    )
    cache_device = typemax(Int)
    extension.GPUCommon.scalar_cache_insert!(
        extension._CUDA_SCALAR_CACHE, cache_device, UInt(1), Float32, UInt(1), :sentinel,
    )
    if isdefined(extension, :_CUDA_VECTOR_CACHE)
        extension.GPUCommon.scalar_cache_insert!(
            extension._CUDA_VECTOR_CACHE, cache_device,
            UInt(1), Float32, UInt(1), :vector_sentinel,
        )
    end
    extension._cuda_clear_scalar_cache!(; device=cache_device)
    @test extension.GPUCommon.scalar_cache_size(
        extension._CUDA_SCALAR_CACHE; device=cache_device,
    ) == 0
    if isdefined(extension, :_CUDA_VECTOR_CACHE)
        @test extension.GPUCommon.scalar_cache_size(
            extension._CUDA_VECTOR_CACHE; device=cache_device,
        ) == 0
    end
    workspace_cache = extension.GPUCommon.ScalarWorkspaceCache(2)
    workspace_owner = Ref(:owner)
    builds = Ref(0)
    builder = () -> (builds[] += 1; :workspace)
    use_workspace = value -> value
    @test extension.GPUCommon.scalar_workspace_use!(
        use_workspace, builder, workspace_cache, :mock, workspace_owner,
        Float32, :scalar, (1,), UInt(1),
    ) === :workspace
    @test extension.GPUCommon.scalar_workspace_use!(
        use_workspace, builder, workspace_cache, :mock, workspace_owner,
        Float32, :scalar, (1,), UInt(1),
    ) === :workspace
    @test builds[] == 1
    extension.GPUCommon.scalar_workspace_clear!(workspace_cache; device=:mock)
    @test extension.GPUCommon.scalar_workspace_size(workspace_cache) == 0
    @test which(
        synthesis_cplx, Tuple{SHTConfig,CuArray{ComplexF32,2}},
    ).module === extension
    @test which(
        synthesis_cplx, Tuple{SHTnsKit.GPU,SHTConfig,CuArray{ComplexF32,2}},
    ).module === SHTnsKit
    run_shared_scalar_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_shared_scalar_variant_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_shared_vector_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_shared_operator_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_scalar_workspace_cache_reference(extension.GPUCommon)
    source = read(joinpath(@__DIR__, "../../../ext/SHTnsKitGPUExt.jl"), String)
    local_table_builder = split(
        split(source, "function _cuda_local_tables"; limit=2)[2],
        "@inline function _cuda_local_precision"; limit=2,
    )[1]
    @test occursin("scalar_cache_publish!(CUDA.synchronize", local_table_builder)
    @test !occursin("scalar_cache_insert!", local_table_builder)
    @test occursin("function _cuda_vector_tables", source)
    if occursin("function _cuda_vector_tables", source)
        scalar_table_builder = split(
            split(source, "function _cuda_scalar_tables"; limit=2)[2],
            "function _cuda_vector_tables"; limit=2,
        )[1]
        @test occursin("scalar_host_tables", scalar_table_builder)
        @test !occursin("vector_host_tables", scalar_table_builder)
        @test !occursin("vector_derivative_table_kernel!", scalar_table_builder)
        @test !occursin("dtheta", scalar_table_builder)
    end
    @test !occursin("_GPU_POLE_TOL", source)
    @test !occursin("vector_analysis_contrib_kernel!", source)
    @test all(method -> method.module === SHTnsKit, methods(gpu_analysis_sphtor))
    @test all(method -> method.module === SHTnsKit, methods(gpu_synthesis_sphtor))
    @test which(
        gpu_analysis_sphtor,
        Tuple{SHTConfig,Matrix{Float32},Matrix{Float32}},
    ).module === SHTnsKit
    @test which(
        gpu_synthesis_sphtor,
        Tuple{SHTConfig,Matrix{ComplexF32},Matrix{ComplexF32}},
    ).module === SHTnsKit
    vector_pipeline = split(
        split(source, "function _cuda_vector_analysis"; limit=2)[2],
        "@inline function _cuda_lcap"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", vector_pipeline)
    @test !occursin(r"\bcollect\s*\(", vector_pipeline)
    ordinary_pipeline = split(
        split(source, "function _cuda_scalar_analysis"; limit=2)[2],
        "function _gpu_adapter_analysis"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", ordinary_pipeline)
    @test !occursin(r"\bcollect\s*\(", ordinary_pipeline)
    @test occursin("use_rfft=true requires a real-valued input", ordinary_pipeline)
    @test occursin("use_rfft=true implies real_output", ordinary_pipeline)
    variant_pipeline = split(
        split(source, "@inline function _cuda_lcap"; limit=2)[2],
        "\"\"\"\n    _to_gpu_impl"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", variant_pipeline)
    @test !occursin(r"\bcollect\s*\(", variant_pipeline)
    @test occursin("ScalarWorkspaceCache(8)", source)
    @test occursin("_cuda_batch_analysis_direct!", source)
    @test occursin("_cuda_batch_synthesis_direct!", source)
    @test occursin("_cuda_vector_analysis_direct!", source)
    @test occursin("_cuda_vector_synthesis_direct!", source)
    @test occursin("tables = _cuda_vector_tables(cfg, RT)", source)
    @test occursin("CUFFT.plan_rfft", source)
    @test !occursin(
        r"result\s*=\s*_cuda_(?:scalar_analysis|scalar_synthesis|batch_analysis|batch_synthesis)\(",
        source,
    )

    @test which(on_device, Tuple{CUDA.AnyCuArray}).module === extension
    @test which(
        SHTnsKit._gpu_adapter_matches,
        Tuple{typeof(extension.CUDA_ADAPTER),CUDA.AnyCuArray},
    ).module === extension
    host_view = @view zeros(Float32, 3, 4)[:, 1:2]
    @test on_device(host_view) isa SHTnsKit.CPU
    @test !SHTnsKit._gpu_adapter_matches(extension.CUDA_ADAPTER, host_view)

    cfg_qst_routing = create_gauss_config(2, 3; nlon=6)
    qst_field = SafeFallbackArray(zeros(ComplexF32, cfg_qst_routing.nlat,
                                        cfg_qst_routing.nlon))
    qst_coefficients = SafeFallbackArray(zeros(ComplexF32,
                                               cfg_qst_routing.lmax + 1,
                                               cfg_qst_routing.mmax + 1))
    @test_throws SHTnsKit.BackendUnavailableError analysis_qst_cplx(
        cfg_qst_routing, qst_field, qst_field, qst_field,
    )
    @test_throws SHTnsKit.BackendUnavailableError synthesis_qst_cplx(
        cfg_qst_routing, qst_coefficients, qst_coefficients,
        qst_coefficients,
    )

    if !CUDA.functional()
        @test_skip CUDA.functional()
        @test get_device() isa SHTnsKit.CPU
        @test_throws SHTnsKit.BackendUnavailableError to_device(SHTnsKit.GPU(), zeros(Float32, 2, 2))

        cfg = create_gauss_config(2, 3; nlon=6)
        field = Float64[sin(i / 3) + cos(j / 4) for i in 1:cfg.nlat, j in 1:cfg.nlon]
        coefficients = analysis(cfg, field)
        for (operation, call) in (
            (:gpu_analysis, () -> gpu_analysis(cfg, field)),
            (:gpu_analysis, () -> gpu_analysis(cfg, field; device=SHTnsKit.CPU())),
            (:gpu_synthesis, () -> gpu_synthesis(cfg, coefficients)),
            (:gpu_analysis_sphtor, () -> gpu_analysis_sphtor(cfg, field, field)),
            (:gpu_synthesis_sphtor, () -> gpu_synthesis_sphtor(cfg, coefficients, coefficients)),
            (:gpu_apply_laplacian!, () -> gpu_apply_laplacian!(cfg, copy(coefficients))),
            (:gpu_memory_info, () -> gpu_memory_info()),
            (:gpu_clear_cache!, () -> gpu_clear_cache!()),
        )
            err = try
                call()
                nothing
            catch caught
                caught
            end
            if err isa SHTnsKit.BackendUnavailableError
                @test err.operation == operation
                detail = sprint(showerror, err)
                if operation in (:gpu_analysis_sphtor, :gpu_synthesis_sphtor)
                    @test occursin("functional", detail)
                else
                    @test occursin("CUDA.functional()", detail)
                end
            else
                @test err isa SHTnsKit.BackendUnavailableError
            end
        end

        @test gpu_analysis_safe(cfg, field) ≈ coefficients
        @test gpu_synthesis_safe(cfg, coefficients) ≈ synthesis(cfg, coefficients)
        wrapped_field = SafeFallbackArray(field)
        wrapped_coefficients = SafeFallbackArray(coefficients)
        wrapped_analysis = try
            gpu_analysis_safe(cfg, wrapped_field; device=SHTnsKit.GPU())
        catch err
            err
        end
        wrapped_synthesis = try
            gpu_synthesis_safe(cfg, wrapped_coefficients; device=SHTnsKit.GPU())
        catch err
            err
        end
        @test wrapped_analysis isa Matrix
        @test wrapped_synthesis isa Matrix
        if wrapped_analysis isa Matrix
            @test wrapped_analysis ≈ coefficients
            @test on_device(wrapped_analysis) isa SHTnsKit.CPU
        end
        if wrapped_synthesis isa Matrix
            @test wrapped_synthesis ≈ synthesis(cfg, coefficients)
            @test on_device(wrapped_synthesis) isa SHTnsKit.CPU
        end

        oom_helper_defined = isdefined(extension, :_with_cuda_oom_fallback)
        @test oom_helper_defined
        if oom_helper_defined
            injected = SafeFallbackRedispatchError()
            @test_throws SafeFallbackRedispatchError extension._with_cuda_oom_fallback(
                () -> throw(injected),
                () -> :cpu,
            )
            @test_throws ErrorException extension._with_cuda_oom_fallback(
                () -> error("illegal memory access"),
                () -> :cpu,
            )
            @test_logs (:warn, r"GPU out of memory") begin
                @test extension._with_cuda_oom_fallback(
                    () -> throw(CUDA.OutOfGPUMemoryError()),
                    () -> :cpu,
                ) == :cpu
            end
        end
        safe_analysis = try
            gpu_analysis_safe(cfg, field; device=SHTnsKit.GPU())
        catch err
            err
        end
        safe_synthesis = try
            gpu_synthesis_safe(cfg, coefficients; device=SHTnsKit.GPU())
        catch err
            err
        end
        if safe_analysis isa AbstractMatrix
            @test safe_analysis ≈ coefficients
        else
            @test safe_analysis isa AbstractMatrix
        end
        if safe_synthesis isa AbstractMatrix
            @test safe_synthesis ≈ synthesis(cfg, coefficients)
        else
            @test safe_synthesis isa AbstractMatrix
        end
        @test !check_gpu_memory(1)
        @test isempty(get_available_gpus())
        @test !set_gpu_device(0)
    else
        cfg = create_gauss_config(3, 6; nlon=8)
        host = Float64[sin(i / 3) + cos(j / 4) for i in 1:cfg.nlat, j in 1:cfg.nlon]
        device = to_device(SHTnsKit.GPU(), host)

        @test device isa CuArray
        @test on_device(device) isa SHTnsKit.GPU
        device_view = @view device[:, :]
        @test device_view isa CUDA.AnyCuArray
        @test on_device(device_view) isa SHTnsKit.GPU
        @test SHTnsKit._gpu_adapter_matches(extension.CUDA_ADAPTER, device_view)
        @test to_device(SHTnsKit.GPU(), device_view) === device_view
        @test to_device(SHTnsKit.GPU(), host, device_view) isa CUDA.AnyCuArray
        @test_throws ArgumentError analysis(SHTnsKit.CPU(), cfg, device_view)
        @test to_device(SHTnsKit.GPU(), host, device) isa CuArray
        @test to_device(host, SHTnsKit.GPU(), device) isa CuArray

        coefficients = analysis(SHTnsKit.GPU(), cfg, device_view)
        @test coefficients isa CUDA.AnyCuArray
        @test analysis(cfg, device_view) isa CUDA.AnyCuArray
        coefficient_view = @view coefficients[:, :]
        @test synthesis(SHTnsKit.GPU(), cfg, coefficients) isa CuArray
        @test synthesis(cfg, coefficient_view) isa CUDA.AnyCuArray
        @test synthesis_cplx(SHTnsKit.GPU(), cfg, coefficient_view) isa CUDA.AnyCuArray
        @test synthesis_cplx(cfg, coefficient_view) isa CUDA.AnyCuArray
        legacy_coefficients = gpu_analysis(cfg, host)
        @test legacy_coefficients isa Matrix
        @test gpu_synthesis(cfg, legacy_coefficients) isa Matrix

        CUDA.allowscalar(false)
        vector_cfg = _vector_config(:regular_poles, 3, 8; norm=:schmidt,
                                    real_norm=true, cs_phase=false)
        vector_S, vector_T = _vector_modes(vector_cfg, Float32)
        vector_Vt, vector_Vp = synthesis_sphtor(
            vector_cfg, CuArray(vector_S), CuArray(vector_T),
        )
        @test vector_Vt isa CUDA.AnyCuArray
        @test vector_Vp isa CUDA.AnyCuArray
        @test analysis_sphtor(vector_cfg, vector_Vt, vector_Vp)[1] isa CUDA.AnyCuArray
        legacy_S, legacy_T = gpu_analysis_sphtor(
            vector_cfg, Array(vector_Vt), Array(vector_Vp),
        )
        @test legacy_S isa Matrix{ComplexF32}
        @test legacy_T isa Matrix{ComplexF32}
        legacy_Vt, legacy_Vp = gpu_synthesis_sphtor(
            vector_cfg, legacy_S, legacy_T,
        )
        @test legacy_Vt isa Matrix{Float32}
        @test legacy_Vp isa Matrix{Float32}

        # Ordinary and direct-plan vector paths accept supported strided device
        # matrix views without staging through host memory.
        Vt_storage = CUDA.zeros(Float32, vector_cfg.nlat, 2vector_cfg.nlon)
        Vp_storage = similar(Vt_storage)
        Vt_view = @view Vt_storage[:, 1:2:end]
        Vp_view = @view Vp_storage[:, 1:2:end]
        copyto!(Vt_view, vector_Vt)
        copyto!(Vp_view, vector_Vp)
        view_S, view_T = analysis_sphtor(vector_cfg, Vt_view, Vp_view)
        @test view_S isa CUDA.AnyCuArray
        @test view_T isa CUDA.AnyCuArray
        @test Array(view_S) ≈ vector_S atol=4f-4 rtol=4f-4
        @test Array(view_T) ≈ vector_T atol=4f-4 rtol=4f-4

        S_storage = CUDA.zeros(ComplexF32, size(vector_S, 1), 2size(vector_S, 2))
        T_storage = similar(S_storage)
        S_view = @view S_storage[:, 1:2:end]
        T_view = @view T_storage[:, 1:2:end]
        copyto!(S_view, CuArray(vector_S))
        copyto!(T_view, CuArray(vector_T))
        view_Vt, view_Vp = synthesis_sphtor(vector_cfg, S_view, T_view)
        @test view_Vt isa CUDA.AnyCuArray
        @test view_Vp isa CUDA.AnyCuArray
        @test Array(view_Vt) ≈ Array(vector_Vt) atol=4f-4 rtol=4f-4
        @test Array(view_Vp) ≈ Array(vector_Vp) atol=4f-4 rtol=4f-4

        vector_plan = SHTPlan(vector_cfg)
        plan_S_store = similar(S_storage); plan_T_store = similar(S_storage)
        plan_S = @view plan_S_store[:, 1:2:end]
        plan_T = @view plan_T_store[:, 1:2:end]
        analysis_sphtor!(vector_plan, plan_S, plan_T, vector_Vt, vector_Vp)
        plan_Vt_store = similar(Vt_storage); plan_Vp_store = similar(Vp_storage)
        plan_Vt = @view plan_Vt_store[:, 1:2:end]
        plan_Vp = @view plan_Vp_store[:, 1:2:end]
        synthesis_sphtor!(vector_plan, plan_Vt, plan_Vp, plan_S, plan_T)
        @test Array(plan_S) ≈ vector_S atol=4f-4 rtol=4f-4
        @test Array(plan_T) ≈ vector_T atol=4f-4 rtol=4f-4
        @test Array(plan_Vt) ≈ Array(vector_Vt) atol=4f-4 rtol=4f-4
        @test Array(plan_Vp) ≈ Array(vector_Vp) atol=4f-4 rtol=4f-4
        assert_warm_device_noalloc(CUDAScalarAdapter()) do
            analysis_sphtor!(vector_plan, plan_S, plan_T, vector_Vt, vector_Vp)
        end
        assert_warm_device_noalloc(CUDAScalarAdapter()) do
            synthesis_sphtor!(vector_plan, plan_Vt, plan_Vp, plan_S, plan_T)
        end
        test_mixed_precision_single_field()
        variant_cfg = create_gauss_config(
            5, 8; nlon=14, mres=2, norm=:schmidt,
            real_norm=true, cs_phase=false,
        )
        variant_dense = _variant_coefficients(variant_cfg, Float32)
        variant_packed = SHTnsKit.pack_lm(variant_cfg, variant_dense)
        variant_field = synthesis_packed(variant_cfg, variant_packed)
        device_field = CuArray(variant_field)
        device_packed = CuArray(variant_packed)
        @test analysis_packed(variant_cfg, device_field) isa CUDA.AnyCuArray
        @test Array(analysis_packed(SHTnsKit.GPU(), variant_cfg, device_field)) ≈
              variant_packed atol=2f-4 rtol=2f-4
        @test Array(synthesis_packed(variant_cfg, device_packed)) ≈
              variant_field atol=2f-4 rtol=2f-4
        ltr = 3
        @test Array(analysis_packed_l(variant_cfg, device_field, ltr)) ≈
              analysis_packed_l(variant_cfg, variant_field, ltr) atol=2f-4 rtol=2f-4
        @test Array(synthesis_packed_l(variant_cfg, device_packed, ltr)) ≈
              synthesis_packed_l(variant_cfg, variant_packed, ltr) atol=2f-4 rtol=2f-4

        axis_coefficients = ComplexF32.(variant_dense[:, 1])
        axis_field = synthesis_axisym(variant_cfg, axis_coefficients)
        @test Array(analysis_axisym(variant_cfg, CuArray(axis_field))) ≈
              axis_coefficients atol=2f-4 rtol=2f-4
        @test Array(synthesis_axisym(variant_cfg, CuArray(axis_coefficients))) ≈
              axis_field atol=2f-4 rtol=2f-4

        im, physical_m = 2, 4
        mode_coefficients = ComplexF32.(variant_dense[(physical_m + 1):end, physical_m + 1])
        mode = synthesis_packed_ml(variant_cfg, im, mode_coefficients, variant_cfg.lmax)
        @test Array(analysis_packed_ml(
            variant_cfg, im, CuArray(mode), variant_cfg.lmax,
        )) ≈ mode_coefficients atol=2f-4 rtol=2f-4
        @test Array(synthesis_packed_ml(
            variant_cfg, im, CuArray(mode_coefficients), variant_cfg.lmax,
        )) ≈ mode atol=2f-4 rtol=2f-4

        batch_fields = repeat(
            reshape(Float32.(reshape(variant_field, variant_cfg.nlat, variant_cfg.nlon)),
                    variant_cfg.nlat, variant_cfg.nlon, 1), 1, 1, 2,
        )
        device_batch = CuArray(batch_fields)
        batch_coefficients = analysis_batch(variant_cfg, device_batch)
        @test batch_coefficients isa CUDA.AnyCuArray
        @test Array(batch_coefficients) ≈ analysis_batch(variant_cfg, batch_fields) atol=2f-4 rtol=2f-4
        @test Array(synthesis_batch(variant_cfg, batch_coefficients)) ≈
              batch_fields atol=2f-4 rtol=2f-4
        analysis_output = similar(batch_coefficients)
        @test analysis_batch!(variant_cfg, analysis_output, device_batch) === analysis_output
        synthesis_output = similar(device_batch)
        @test synthesis_batch!(variant_cfg, synthesis_output, batch_coefficients) === synthesis_output

        complex_cfg = create_gauss_config(3, 6; nlon=10)
        complex_coefficients = zeros(ComplexF32, nlm_cplx_calc(3, 3, 1))
        complex_coefficients[LM_cplx_index(3, 3, 2, -1) + 1] = 0.2f0 - 0.1f0im
        complex_field = synthesis_packed_cplx(complex_cfg, complex_coefficients)
        device_complex = CuArray(complex_coefficients)
        @test Array(synthesis_packed_cplx(complex_cfg, device_complex)) ≈
              complex_field atol=2f-4 rtol=2f-4
        @test Array(analysis_packed_cplx(complex_cfg, CuArray(complex_field))) ≈
              complex_coefficients atol=2f-4 rtol=2f-4
        run_gpu_scalar_variant_matrix(CUDAScalarAdapter())
        run_scalar_full_parity(
            CUDAScalarAdapter();
            grid_kinds=_SCALAR_GRID_KINDS,
            precisions=(Float32, Float64),
            mres_values=(1, 2),
            norms=(:orthonormal, :fourpi, :schmidt),
            real_norm_values=(false, true),
            cs_phase_values=(false, true),
            pole_orders=(false, true),
        )
        run_gpu_vector_batch_parity(CUDAVectorAdapter())
        GPUWrapperReference.run_gpu_qst_regressions(SHTnsKit, CuArray, CUDA.AnyCuArray)
        run_shared_legendre_precision_reference(
            extension.GPUCommon, CUDABackend(); place=CuArray,
        )
        run_sphtor_full_parity(CUDAVectorAdapter())
        run_qst_full_parity(CUDAQSTAdapter())
        run_gpu_vector_mode_edge_parity(CUDAVectorAdapter())
        run_concurrent_local_first_use(
            CUDALocalEvaluationAdapter(), extension._cuda_clear_scalar_cache!,
        )
        run_local_evaluation_parity(CUDALocalEvaluationAdapter())
        run_gpu_operator_parity(CUDAOperatorAdapter())
    end
end
