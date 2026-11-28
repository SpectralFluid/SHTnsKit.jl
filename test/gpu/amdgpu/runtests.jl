using Test
using SHTnsKit
using AMDGPU
using GPUArrays
using GPUArraysCore
import KernelAbstractions

include("../wrapper_reference.jl")
include("../test_transform_storage.jl")

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

struct AMDGPUScalarAdapter <: ScalarParityAdapter end
struct AMDGPUOperatorAdapter <: GPUOperatorAdapter end
struct AMDGPURotationAdapter <: RotationParityAdapter end
rotation_place(::AMDGPURotationAdapter, value) = ROCArray(value)
rotation_collect(::AMDGPURotationAdapter, value) = Array(value)
rotation_resident(::AMDGPURotationAdapter, value) = @test value isa AMDGPU.AnyROCArray
rotation_z(::AMDGPURotationAdapter, cfg, input, angle, output) =
    SH_Zrotate(GPU(), cfg, input, angle, output)
rotation_y(::AMDGPURotationAdapter, cfg, input, angle, output) =
    SH_Yrotate(GPU(), cfg, input, angle, output)
rotation_apply_real(::AMDGPURotationAdapter, rotation, input, output) =
    shtns_rotation_apply_real(GPU(), rotation, input, output)
rotation_apply_cplx(::AMDGPURotationAdapter, rotation, input, output) =
    shtns_rotation_apply_cplx(GPU(), rotation, input, output)
operator_place(::AMDGPUOperatorAdapter, value) = ROCArray(value)
function operator_strided_place(::AMDGPUOperatorAdapter, value::AbstractVector)
    storage = AMDGPU.zeros(eltype(value), 2length(value))
    result = @view storage[1:2:end]
    copyto!(result, ROCArray(value))
    return result
end
function operator_overlapping_matrix_place(::AMDGPUOperatorAdapter, value)
    storage = AMDGPU.zeros(eltype(value), size(value, 1) + 1, size(value, 2))
    source = @view storage[1:size(value, 1), :]
    destination = @view storage[2:(size(value, 1) + 1), :]
    copyto!(source, ROCArray(value))
    return source, destination
end
operator_collect(::AMDGPUOperatorAdapter, value) = Array(value)
operator_resident(::AMDGPUOperatorAdapter, value) = @test value isa AMDGPU.AnyROCArray
function place(::AMDGPUScalarAdapter, ::SHTConfig, value, ::Symbol)
    if value isa PermutedDimsArray
        padded = zeros(eltype(value), size(value, 1), 2size(value, 2))
        @views padded[:, 1:2:end] .= value
        storage = ROCArray(padded)
        return @view storage[:, 1:2:end]
    end
    return ROCArray(value)
end
collect_result(::AMDGPUScalarAdapter, value, ::SHTConfig) = Array(value)
analysis_call(::AMDGPUScalarAdapter, cfg, field; use_rfft=false) =
    analysis(GPU(), cfg, field; use_rfft)
synthesis_call(::AMDGPUScalarAdapter, cfg, coefficients, _prototype;
               real_output, use_rfft=false) =
    synthesis(GPU(), cfg, coefficients; real_output, use_rfft)
synthesis_cplx_call(::AMDGPUScalarAdapter, cfg, coefficients, _prototype) =
    synthesis_cplx(cfg, coefficients)
assert_resident(::AMDGPUScalarAdapter, value) = @test value isa AMDGPU.AnyROCArray

struct AMDGPUVectorAdapter <: VectorParityAdapter end
vector_place(::AMDGPUVectorAdapter, ::SHTConfig, value, ::Symbol) = ROCArray(value)
vector_collect(::AMDGPUVectorAdapter, value, ::SHTConfig) = Array(value)
vector_resident(::AMDGPUVectorAdapter, value) = @test value isa AMDGPU.AnyROCArray
vector_analysis(::AMDGPUVectorAdapter, cfg, Vt, Vp; use_rfft=false) =
    analysis_sphtor(GPU(), cfg, Vt, Vp; use_rfft)
vector_analysis_cplx(::AMDGPUVectorAdapter, cfg, Vt, Vp) =
    analysis_sphtor_cplx(GPU(), cfg, Vt, Vp)
vector_synthesis(::AMDGPUVectorAdapter, cfg, S, T, _prototype;
                 real_output=true, use_rfft=false) =
    synthesis_sphtor(GPU(), cfg, S, T; real_output, use_rfft)
vector_synthesis_cplx(::AMDGPUVectorAdapter, cfg, S, T, _prototype) =
    synthesis_sphtor_cplx(GPU(), cfg, S, T)
vector_sph(::AMDGPUVectorAdapter, cfg, S, _prototype; real_output=true) =
    synthesis_sph(GPU(), cfg, S; real_output)
vector_sph_cplx(::AMDGPUVectorAdapter, cfg, S, _prototype) =
    synthesis_sph_cplx(GPU(), cfg, S)
vector_tor(::AMDGPUVectorAdapter, cfg, T, _prototype; real_output=true) =
    synthesis_tor(GPU(), cfg, T; real_output)
vector_tor_cplx(::AMDGPUVectorAdapter, cfg, T, _prototype) =
    synthesis_tor_cplx(GPU(), cfg, T)

struct AMDGPUQSTAdapter <: QSTParityAdapter end
qst_place(::AMDGPUQSTAdapter, ::SHTConfig, value, ::Symbol) = ROCArray(value)
qst_collect(::AMDGPUQSTAdapter, value, ::SHTConfig) = Array(value)
qst_resident(::AMDGPUQSTAdapter, value) = @test value isa AMDGPU.AnyROCArray
qst_analysis(::AMDGPUQSTAdapter, cfg, Vr, Vt, Vp; use_rfft=false) =
    analysis_qst(GPU(), cfg, Vr, Vt, Vp; use_rfft)
qst_analysis_cplx(::AMDGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    analysis_qst_cplx(GPU(), cfg, Vr, Vt, Vp)
qst_synthesis(::AMDGPUQSTAdapter, cfg, Q, S, Tlm, prototype;
              real_output=true, use_rfft=false) =
    synthesis_qst(GPU(), cfg, Q, S, Tlm; prototype, real_output, use_rfft)
qst_synthesis_cplx(::AMDGPUQSTAdapter, cfg, Q, S, Tlm, prototype) =
    synthesis_qst_cplx(GPU(), cfg, Q, S, Tlm; prototype)
qst_analysis_inferred(::AMDGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    analysis_qst(cfg, Vr, Vt, Vp)
qst_analysis_cplx_inferred(::AMDGPUQSTAdapter, cfg, Vr, Vt, Vp) =
    analysis_qst_cplx(cfg, Vr, Vt, Vp)
qst_synthesis_inferred(::AMDGPUQSTAdapter, cfg, Q, S, Tlm, prototype) =
    synthesis_qst(cfg, Q, S, Tlm; prototype)
qst_synthesis_cplx_inferred(::AMDGPUQSTAdapter, cfg, Q, S, Tlm, _prototype) =
    synthesis_qst_cplx(cfg, Q, S, Tlm)

struct AMDGPULocalEvaluationAdapter <: LocalEvaluationAdapter end
function local_place(::AMDGPULocalEvaluationAdapter, value)
    if value isa SubArray
        if ndims(value) == 1
            padded = zeros(eltype(value), 2length(value))
            @views padded[1:2:end] .= value
            storage = ROCArray(padded)
            return @view storage[1:2:end]
        end
        padded = zeros(eltype(value), size(value, 1), 2size(value, 2))
        @views padded[:, 1:2:end] .= value
        storage = ROCArray(padded)
        return @view storage[:, 1:2:end]
    end
    return ROCArray(value)
end
function local_collect(::AMDGPULocalEvaluationAdapter, value)
    host = Array(value)
    return ndims(host) == 0 ? host[] : host
end
local_scalar(::AMDGPULocalEvaluationAdapter, cfg, coefficients, cost, phi) =
    synthesis_point(GPU(), cfg, coefficients, cost, phi)
local_scalar_cplx(::AMDGPULocalEvaluationAdapter, cfg, coefficients, cost, phi) =
    synthesis_point_cplx(GPU(), cfg, coefficients, cost, phi)
local_lat(::AMDGPULocalEvaluationAdapter, cfg, coefficients, cost; kwargs...) =
    SH_to_lat(GPU(), cfg, coefficients, cost; kwargs...)
local_lat_cplx(::AMDGPULocalEvaluationAdapter, cfg, coefficients, cost; kwargs...) =
    SH_to_lat_cplx(GPU(), cfg, coefficients, cost; kwargs...)
local_qst_point(::AMDGPULocalEvaluationAdapter, cfg, Q, S, Tlm, cost, phi) =
    SHqst_to_point(GPU(), cfg, Q, S, Tlm, cost, phi)
local_qst_lat(::AMDGPULocalEvaluationAdapter, cfg, Q, S, Tlm, cost; kwargs...) =
    SHqst_to_lat(GPU(), cfg, Q, S, Tlm, cost; kwargs...)
local_grad_point(::AMDGPULocalEvaluationAdapter, cfg, Dr, S, cost, phi) =
    SH_to_grad_point(GPU(), cfg, Dr, S, cost, phi)
local_assert_resident(::AMDGPULocalEvaluationAdapter, value) =
    @test value isa AMDGPU.AnyROCArray
qst_supports_strided_views(::AMDGPUQSTAdapter) = true
function qst_strided_place(::AMDGPUQSTAdapter, ::SHTConfig, value, ::Symbol)
    padded = zeros(eltype(value), size(value, 1), 2size(value, 2))
    @views padded[:, 1:2:end] .= value
    storage = ROCArray(padded)
    return @view storage[:, 1:2:end]
end
function assert_warm_device_noalloc(::AMDGPUScalarAdapter, call)
    call()
    AMDGPU.synchronize()
    @test @allocated(begin
        call()
        AMDGPU.synchronize()
    end) <= 65_536
    return nothing
end

# Mixed Float32/Float64 operands promote as on the CPU: an S/T pair, and all
# three QST components, take the widest precision among them. Values are
# compared with the CPU transform of the promoted operands.
function test_device_routing_fixes()
    @testset "In-place, Laplacian, nested-wrapper and Z-rotation routing" begin
        cfg = create_gauss_config(5, 8; nlon=12)
        a = _variant_coefficients(cfg, Float32)
        field = synthesis(cfg, a)
        d = ROCArray(field)
        # The configuration-level in-place transforms run on the device.
        out = AMDGPU.zeros(ComplexF32, size(a))
        @test analysis!(cfg, out, d) === out
        @test Array(out) ≈ a atol=2f-4 rtol=2f-4
        grid = similar(d)
        @test synthesis!(cfg, grid, out) === grid
        @test Array(grid) ≈ field atol=2f-4 rtol=2f-4
        # The Laplacian covers one field, so a batch is rejected.
        batch = ROCArray(cat(a, a; dims=3))
        @test_throws DimensionMismatch gpu_apply_laplacian!(cfg, batch)
        @test Array(batch)[:, :, 2] == a
        # A view of a PermutedDimsArray is copied into a dense device array.
        nested = view(PermutedDimsArray(ROCArray(permutedims(field)), (2, 1)), :, :)
        @test on_device(nested) isa SHTnsKit.GPU
        result = analysis(cfg, nested)
        @test result isa AMDGPU.AnyROCArray
        @test Array(result) ≈ a atol=2f-4 rtol=2f-4
        fields = cat(field, 2f0 .* field; dims=3)
        stacked = view(PermutedDimsArray(ROCArray(permutedims(fields, (2, 1, 3))), (2, 1, 3)), :, :, :)
        @test Array(analysis_batch(cfg, stacked)) ≈ analysis_batch(cfg, fields) atol=2f-4 rtol=2f-4
        @test to_device(SHTnsKit.CPU(), nested) == field
        @test_throws SHTnsKit.BackendUnavailableError analysis!(cfg, out, nested)
        # A wider output sets the planned precision, as a CPU plan's Float64 does.
        wide = AMDGPU.zeros(ComplexF64, size(a))
        analysis!(cfg, wide, d)
        @test Array(wide) ≈ analysis(cfg, Float64.(field)) rtol=1e-10 atol=1e-12
        # Real vector fields may land in complex storage, as on the CPU.
        S, T = analysis_sphtor(cfg, field, 2f0 .* field)
        zt, zp = AMDGPU.zeros(ComplexF32, size(field)), AMDGPU.zeros(ComplexF32, size(field))
        synthesis_sphtor!(SHTPlan(cfg), zt, zp, ROCArray(S), ROCArray(T))
        expected = synthesis_sphtor(cfg, S, T)
        @test Array(zt) ≈ expected[1] atol=2f-4 rtol=2f-4
        @test Array(zp) ≈ expected[2] atol=2f-4 rtol=2f-4
        @test_throws ArgumentError synthesis_axisym_l(cfg, AMDGPU.zeros(ComplexF32, 2), 3)
        # Float32 Z rotations keep the CPU's phase accuracy at high orders.
        big = create_gauss_config(255, 257; nlon=512)
        q = _rotation_real_spectrum(big, Float32)
        expected = SH_Zrotate(SHTnsKit.CPU(), big, q, 6.18f0, similar(q))
        rotated = SH_Zrotate(big, ROCArray(q), 6.18f0, AMDGPU.zeros(ComplexF32, big.nlm))
        @test maximum(abs, Array(rotated) - expected) < 1f-6 * maximum(abs, expected)
    end
end

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
                @test values isa AMDGPU.AnyROCArray
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
            test_promoted(analysis_sphtor(cfg, ROCArray(Vt), ROCArray(Vp)), cpu, exact)
            # A host component moves to the device without being narrowed.
            test_promoted(analysis_sphtor(cfg, ROCArray(Vt), Vp), cpu, exact)
            # Planned outputs carry the promoted precision the allocating call returns.
            S_out = AMDGPU.zeros(ComplexF64, size(S64)); T_out = similar(S_out)
            analysis_sphtor!(plan, S_out, T_out, ROCArray(Vt), ROCArray(Vp))
            test_promoted((S_out, T_out), cpu, exact)
            @test_throws ArgumentError analysis_sphtor!(
                plan, AMDGPU.zeros(ComplexF32, size(S64)),
                AMDGPU.zeros(ComplexF32, size(S64)), ROCArray(Vt), ROCArray(Vp),
            )

            S, Tlm = Complex{A}.(S64), Complex{B}.(T64)
            cpu = synthesis_sphtor(cfg, S, Tlm)
            exact = synthesis_sphtor(cfg, promoted(S, Tlm)...)
            test_promoted(synthesis_sphtor(cfg, ROCArray(S), ROCArray(Tlm)), cpu, exact)
            Vt_out = AMDGPU.zeros(Float64, size(Vt64)); Vp_out = similar(Vt_out)
            synthesis_sphtor!(plan, Vt_out, Vp_out, ROCArray(S), ROCArray(Tlm))
            test_promoted((Vt_out, Vp_out), cpu, exact)
            test_promoted(synthesis_sphtor_cplx(cfg, ROCArray(S), ROCArray(Tlm)),
                          synthesis_sphtor_cplx(cfg, S, Tlm),
                          synthesis_sphtor_cplx(cfg, promoted(S, Tlm)...))
        end

        for wide in 1:3
            fields = ntuple(i -> (i == wide ? Float64 : Float32).((Vr64, Vt64, Vp64)[i]), 3)
            test_promoted(analysis_qst(cfg, map(ROCArray, fields)...),
                          analysis_qst(cfg, fields...),
                          analysis_qst(cfg, promoted(fields...)...))
            test_promoted(analysis_qst_l(cfg, map(ROCArray, fields)..., 3),
                          analysis_qst_l(cfg, fields..., 3),
                          analysis_qst_l(cfg, promoted(fields...)..., 3))
            spectra = ntuple(i -> (i == wide ? ComplexF64 : ComplexF32).((Q64, S64, T64)[i]), 3)
            for real_output in (true, false)
                test_promoted(synthesis_qst(cfg, map(ROCArray, spectra)...; real_output),
                              synthesis_qst(cfg, spectra...; real_output),
                              synthesis_qst(cfg, promoted(spectra...)...; real_output))
                test_promoted(synthesis_qst_l(cfg, map(ROCArray, spectra)..., 3; real_output),
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
            test_promoted(synthesis_sphtor_ml(cfg, stored_im, ROCArray(Sl), ROCArray(Tl), ltr),
                          synthesis_sphtor_ml(cfg, stored_im, Sl, Tl, ltr), exact)
            Vtl, Vpl = A.(exact[1]), B.(exact[2])
            test_promoted(analysis_sphtor_ml(cfg, stored_im, ROCArray(Vtl), ROCArray(Vpl), ltr),
                          analysis_sphtor_ml(cfg, stored_im, Vtl, Vpl, ltr),
                          analysis_sphtor_ml(cfg, stored_im, promoted(Vtl, Vpl)..., ltr))
        end
        Ql64 = Q64[(physical_m + 1):end, physical_m + 1]
        for wide in 1:3
            modes = ntuple(i -> (i == wide ? ComplexF64 : ComplexF32).((Ql64, Sl64, Tl64)[i]), 3)
            exact = synthesis_qst_ml(cfg, stored_im, promoted(modes...)..., ltr)
            test_promoted(synthesis_qst_ml(cfg, stored_im, map(ROCArray, modes)..., ltr),
                          synthesis_qst_ml(cfg, stored_im, modes..., ltr), exact)
            profiles = ntuple(i -> (i == wide ? ComplexF64 : ComplexF32).(exact[i]), 3)
            test_promoted(analysis_qst_ml(cfg, stored_im, map(ROCArray, profiles)..., ltr),
                          analysis_qst_ml(cfg, stored_im, profiles..., ltr),
                          analysis_qst_ml(cfg, stored_im, promoted(profiles...)..., ltr))
        end
    end
    return nothing
end

@testset "AMDGPU backend routing" begin
    extension = Base.get_extension(SHTnsKit, :SHTnsKitAMDGPUExt)
    @test extension !== nothing
    @test isdefined(extension, :mul!)
    @test SHTnsKit._GPU_LOOP_AVAILABLE[]
    test_gpu_rotation_contract(
        extension, ROCArray{ComplexF32,1}, ROCArray{ComplexF32,1},
    )
    run_shared_rotation_kernel_reference(
        extension.GPUCommon, KernelAbstractions.CPU(),
    )
    test_angle_axis_pi_singularity()
    if AMDGPU.functional()
        test_transform_storage(AMDGPU, ROCArray)
        loop_values = AMDGPU.zeros(Int32, 8)
        @sht_loop loop_values[i] = Int32(i) over i ∈ eachindex(loop_values)
        @test Array(loop_values) == Int32.(1:8)
        run_rotation_parity(AMDGPURotationAdapter())
        test_shtns37_gpu_fixtures(
            ROCArray, value -> (@test value isa AMDGPU.AnyROCArray),
        )
    else
        @test_skip AMDGPU.functional()
    end
    test_gpu_vector_variant_contract(
        extension, ROCArray{Float32,2}, ROCArray{ComplexF32,2},
        ROCArray{ComplexF32,1}, ROCArray{Float32,3}, ROCArray{ComplexF32,3},
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
                Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{Float32,1}}).module === extension
    @test which(st_dt_matrix,
                Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{Float32,1}}).module === extension
    @test which(SH_mul_mx,
                Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{Float32,1},
                      ROCArray{ComplexF32,1},ROCArray{ComplexF32,1}}).module === extension
    @test which(divergence_from_spheroidal,
                Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{ComplexF32,2}}).module === extension
    @test which(gpu_apply_laplacian!,
                Tuple{SHTConfig,ROCArray{ComplexF32,2}}).module === extension
    @test isdefined(extension.GPUCommon, :local_scalar_kernel!)
    @test isdefined(extension.GPUCommon, :local_complex_kernel!)
    @test isdefined(extension.GPUCommon, :local_qst_kernel!)
    @test isdefined(extension, :_amdgpu_scalar_analysis)
    @test isdefined(extension, :_amdgpu_scalar_synthesis)
    @test isdefined(extension, :_amdgpu_clear_scalar_cache!)
    @test isdefined(extension, :_amdgpu_scalar_analysis_direct!)
    @test isdefined(extension, :_amdgpu_batch_analysis_direct!)
    @test isdefined(extension, :_amdgpu_vector_analysis)
    @test isdefined(extension, :_amdgpu_vector_synthesis)
    @test isdefined(extension, :_amdgpu_vector_analysis_direct!)
    @test isdefined(extension, :_amdgpu_vector_synthesis_direct!)
    @test isdefined(extension, :_amdgpu_local_tables)
    @test extension._AMDGPU_LOCAL_CACHE.max_per_device == 8
    @test which(synthesis_point,
                Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{ComplexF32,2},Float32,Float32}).module === extension
    @test which(SHqst_to_point,
                Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{ComplexF32,1},ROCArray{ComplexF32,1},ROCArray{ComplexF32,1},Float32,Float32}).module === extension
    @test fieldnames(extension.AMDGPUScalarTables) == (:sint, :weights, :Plm, :scales)
    @test isdefined(extension, :AMDGPUVectorTables)
    @test isdefined(extension, :_AMDGPU_VECTOR_CACHE)
    @test isdefined(extension, :_amdgpu_vector_tables)
    @test extension._AMDGPU_SCALAR_CACHE.max_per_device == 8
    @test extension._AMDGPU_VECTOR_CACHE.max_per_device == 8
    @test which(
        analysis_sphtor,
        Tuple{SHTConfig,ROCArray{Float32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis_sphtor,
        Tuple{SHTConfig,ROCArray{ComplexF32,2},ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis_qst,
        Tuple{SHTConfig,ROCArray{Float32,2},ROCArray{Float32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis_qst,
        Tuple{SHTConfig,ROCArray{ComplexF32,2},ROCArray{ComplexF32,2},ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis_sphtor,
        Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{Float32,2},ROCArray{Float32,2}},
    ).module === SHTnsKit
    for (function_name, signature) in (
        (:analysis_packed, Tuple{SHTConfig,ROCArray{Float32,1}}),
        (:synthesis_packed, Tuple{SHTConfig,ROCArray{ComplexF32,1}}),
        (:analysis_packed_l, Tuple{SHTConfig,ROCArray{Float32,1},Int}),
        (:synthesis_packed_l, Tuple{SHTConfig,ROCArray{ComplexF32,1},Int}),
        (:analysis_axisym, Tuple{SHTConfig,ROCArray{Float32,1}}),
        (:synthesis_axisym, Tuple{SHTConfig,ROCArray{ComplexF32,1}}),
        (:analysis_axisym_l, Tuple{SHTConfig,ROCArray{Float32,1},Int}),
        (:synthesis_axisym_l, Tuple{SHTConfig,ROCArray{ComplexF32,1},Int}),
        (:analysis_packed_ml, Tuple{SHTConfig,Int,ROCArray{ComplexF32,1},Int}),
        (:synthesis_packed_ml, Tuple{SHTConfig,Int,ROCArray{ComplexF32,1},Int}),
        (:analysis_packed_cplx, Tuple{SHTConfig,ROCArray{ComplexF32,2}}),
        (:synthesis_packed_cplx, Tuple{SHTConfig,ROCArray{ComplexF32,1}}),
        (:analysis_packed_cplx_l, Tuple{SHTConfig,ROCArray{ComplexF32,2},Int}),
        (:synthesis_packed_cplx_l, Tuple{SHTConfig,ROCArray{ComplexF32,1},Int}),
        (:analysis_batch, Tuple{SHTConfig,ROCArray{Float32,3}}),
        (:analysis_batch!, Tuple{SHTConfig,ROCArray{ComplexF32,3},ROCArray{Float32,3}}),
        (:synthesis_batch, Tuple{SHTConfig,ROCArray{ComplexF32,3}}),
        (:synthesis_batch!, Tuple{SHTConfig,ROCArray{Float32,3},ROCArray{ComplexF32,3}}),
        (:synthesis_batch_cplx, Tuple{SHTConfig,ROCArray{ComplexF32,3}}),
    )
        @test which(getproperty(SHTnsKit, function_name), signature).module === extension
    end
    @test which(
        analysis!, Tuple{SHTPlan,ROCArray{ComplexF32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis!, Tuple{SHTPlan,ROCArray{Float32,2},ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis_sphtor!,
        Tuple{SHTPlan,ROCArray{ComplexF32,2},ROCArray{ComplexF32,2},
              ROCArray{Float32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(
        synthesis_sphtor!,
        Tuple{SHTPlan,ROCArray{Float32,2},ROCArray{Float32,2},
              ROCArray{ComplexF32,2},ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        analysis!,
        Tuple{SHTnsKit.GPU,SHTPlan,ROCArray{ComplexF32,2},ROCArray{Float32,2}},
    ).module === extension
    @test which(gpu_clear_cache!, Tuple{SHTnsKit.GPU}).module === SHTnsKit
    @test hasmethod(
        SHTnsKit._gpu_adapter_clear_cache!, Tuple{typeof(extension.AMDGPU_ADAPTER)},
    )
    cache_device = typemax(Int)
    extension.GPUCommon.scalar_cache_insert!(
        extension._AMDGPU_SCALAR_CACHE, cache_device, UInt(1), Float32, UInt(1), :sentinel,
    )
    if isdefined(extension, :_AMDGPU_VECTOR_CACHE)
        extension.GPUCommon.scalar_cache_insert!(
            extension._AMDGPU_VECTOR_CACHE, cache_device,
            UInt(1), Float32, UInt(1), :vector_sentinel,
        )
    end
    extension._amdgpu_clear_scalar_cache!(; device=cache_device)
    @test extension.GPUCommon.scalar_cache_size(
        extension._AMDGPU_SCALAR_CACHE; device=cache_device,
    ) == 0
    if isdefined(extension, :_AMDGPU_VECTOR_CACHE)
        @test extension.GPUCommon.scalar_cache_size(
            extension._AMDGPU_VECTOR_CACHE; device=cache_device,
        ) == 0
    end
    workspace_cache = extension.GPUCommon.ScalarWorkspaceCache(2)
    workspace_owner = Ref(:owner)
    builds = Ref(0)
    builder = () -> (builds[] += 1; :workspace)
    @test extension.GPUCommon.scalar_workspace_use!(
        identity, builder, workspace_cache, :mock, workspace_owner,
        Float32, :scalar, (1,), UInt(1),
    ) === :workspace
    @test extension.GPUCommon.scalar_workspace_use!(
        identity, builder, workspace_cache, :mock, workspace_owner,
        Float32, :scalar, (1,), UInt(1),
    ) === :workspace
    @test builds[] == 1
    extension.GPUCommon.scalar_workspace_clear!(workspace_cache; device=:mock)
    @test extension.GPUCommon.scalar_workspace_size(workspace_cache) == 0
    @test which(
        synthesis_cplx, Tuple{SHTConfig,ROCArray{ComplexF32,2}},
    ).module === extension
    @test which(
        synthesis_cplx, Tuple{SHTnsKit.GPU,SHTConfig,ROCArray{ComplexF32,2}},
    ).module === SHTnsKit
    run_shared_scalar_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_shared_scalar_variant_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_shared_vector_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_shared_operator_kernel_reference(extension.GPUCommon, KernelAbstractions.CPU())
    run_scalar_workspace_cache_reference(extension.GPUCommon)
    source = read(joinpath(@__DIR__, "../../../ext/SHTnsKitAMDGPUExt.jl"), String)
    local_table_builder = split(
        split(source, "function _amdgpu_local_tables"; limit=2)[2],
        "@inline function _amdgpu_local_precision"; limit=2,
    )[1]
    @test occursin("scalar_cache_publish!(AMDGPU.synchronize", local_table_builder)
    @test !occursin("scalar_cache_insert!", local_table_builder)
    @test occursin("function _amdgpu_vector_tables", source)
    if occursin("function _amdgpu_vector_tables", source)
        scalar_table_builder = split(
            split(source, "function _amdgpu_scalar_tables"; limit=2)[2],
            "function _amdgpu_vector_tables"; limit=2,
        )[1]
        @test occursin("scalar_host_tables", scalar_table_builder)
        @test !occursin("vector_host_tables", scalar_table_builder)
        @test !occursin("vector_derivative_table_kernel!", scalar_table_builder)
        @test !occursin("dtheta", scalar_table_builder)
    end
    vector_pipeline = split(
        split(source, r"(?m)^_amdgpu_vector_analysis\("; limit=2)[2],
        "@inline function _amdgpu_lcap"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", vector_pipeline)
    @test !occursin(r"\bcollect\s*\(", vector_pipeline)
    ordinary_pipeline = split(
        split(source, r"(?m)^_amdgpu_scalar_analysis_direct!\("; limit=2)[2],
        "function _gpu_adapter_analysis"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", ordinary_pipeline)
    @test !occursin(r"\bcollect\s*\(", ordinary_pipeline)
    @test occursin("use_rfft=true requires a real-valued input", ordinary_pipeline)
    @test occursin("use_rfft=true implies real_output", ordinary_pipeline)
    variant_pipeline = split(
        split(source, "@inline function _amdgpu_lcap"; limit=2)[2],
        "end # module SHTnsKitAMDGPUExt"; limit=2,
    )[1]
    @test !occursin(r"\bArray\s*\(", variant_pipeline)
    @test !occursin(r"\bcollect\s*\(", variant_pipeline)
    @test occursin("ScalarWorkspaceCache(8)", source)
    @test occursin("_amdgpu_batch_analysis_direct!", source)
    @test occursin("_amdgpu_batch_synthesis_direct!", source)
    @test occursin("_amdgpu_vector_analysis_direct!", source)
    @test occursin("_amdgpu_vector_synthesis_direct!", source)
    @test occursin("tables = _amdgpu_vector_tables(cfg, RT)", source)
    @test occursin("FFTW.plan_rfft", source)
    @test !occursin(
        r"result\s*=\s*_amdgpu_(?:scalar_analysis|scalar_synthesis|batch_analysis|batch_synthesis)\(",
        source,
    )

    @test which(on_device, Tuple{AMDGPU.AnyROCArray}).module === extension
    @test which(
        SHTnsKit._gpu_adapter_matches,
        Tuple{typeof(extension.AMDGPU_ADAPTER),AMDGPU.AnyROCArray},
    ).module === extension
    host_view = @view zeros(Float32, 3, 4)[:, 1:2]
    @test on_device(host_view) isa SHTnsKit.CPU
    @test !SHTnsKit._gpu_adapter_matches(extension.AMDGPU_ADAPTER, host_view)

    if !AMDGPU.functional() || !AMDGPU.functional(:rocfft)
        @test_skip AMDGPU.functional() && AMDGPU.functional(:rocfft)
        if !AMDGPU.functional()
            @test get_device() isa SHTnsKit.CPU
            @test_throws SHTnsKit.BackendUnavailableError to_device(
                SHTnsKit.GPU(), zeros(Float32, 2, 2),
            )
        else
            cfg = create_gauss_config(2, 3; nlon=6)
            field = ROCArray(zeros(Float32, cfg.nlat, cfg.nlon))
            @test_throws SHTnsKit.BackendUnavailableError analysis(cfg, field)
        end
    else
        cfg = create_gauss_config(2, 3; nlon=6)
        host = reshape(Float32.(1:(cfg.nlat * cfg.nlon)), cfg.nlat, cfg.nlon)
        device = to_device(SHTnsKit.GPU(), host)

        @test device isa ROCArray
        @test on_device(device) isa SHTnsKit.GPU
        device_view = @view device[:, :]
        @test device_view isa AMDGPU.AnyROCArray
        @test on_device(device_view) isa SHTnsKit.GPU
        @test SHTnsKit._gpu_adapter_matches(extension.AMDGPU_ADAPTER, device_view)
        @test to_device(SHTnsKit.GPU(), device_view) === device_view
        @test to_device(SHTnsKit.GPU(), host, device_view) isa AMDGPU.AnyROCArray
        @test_throws ArgumentError analysis(SHTnsKit.CPU(), cfg, device_view)
        @test to_device(SHTnsKit.GPU(), host, device) isa ROCArray
        @test to_device(host, SHTnsKit.GPU(), device) isa ROCArray

        @test analysis(cfg, device_view) isa AMDGPU.AnyROCArray
        coefficients = analysis(SHTnsKit.GPU(), cfg, device_view)
        @test coefficients isa AMDGPU.AnyROCArray
        @test synthesis_cplx(SHTnsKit.GPU(), cfg, coefficients) isa AMDGPU.AnyROCArray
        @test synthesis_cplx(cfg, coefficients) isa AMDGPU.AnyROCArray

        AMDGPU.allowscalar(false)
        vector_cfg = _vector_config(:regular_poles, 3, 8; norm=:schmidt,
                                    real_norm=true, cs_phase=false)
        vector_S, vector_T = _vector_modes(vector_cfg, Float32)
        vector_Vt, vector_Vp = synthesis_sphtor(
            vector_cfg, ROCArray(vector_S), ROCArray(vector_T),
        )
        @test vector_Vt isa AMDGPU.AnyROCArray
        @test vector_Vp isa AMDGPU.AnyROCArray
        @test analysis_sphtor(vector_cfg, vector_Vt, vector_Vp)[1] isa AMDGPU.AnyROCArray
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
        Vt_storage = AMDGPU.zeros(Float32, vector_cfg.nlat, 2vector_cfg.nlon)
        Vp_storage = similar(Vt_storage)
        Vt_view = @view Vt_storage[:, 1:2:end]
        Vp_view = @view Vp_storage[:, 1:2:end]
        copyto!(Vt_view, vector_Vt)
        copyto!(Vp_view, vector_Vp)
        view_S, view_T = analysis_sphtor(vector_cfg, Vt_view, Vp_view)
        @test view_S isa AMDGPU.AnyROCArray
        @test view_T isa AMDGPU.AnyROCArray
        @test Array(view_S) ≈ vector_S atol=4f-4 rtol=4f-4
        @test Array(view_T) ≈ vector_T atol=4f-4 rtol=4f-4

        S_storage = AMDGPU.zeros(ComplexF32, size(vector_S, 1), 2size(vector_S, 2))
        T_storage = similar(S_storage)
        S_view = @view S_storage[:, 1:2:end]
        T_view = @view T_storage[:, 1:2:end]
        copyto!(S_view, ROCArray(vector_S))
        copyto!(T_view, ROCArray(vector_T))
        view_Vt, view_Vp = synthesis_sphtor(vector_cfg, S_view, T_view)
        @test view_Vt isa AMDGPU.AnyROCArray
        @test view_Vp isa AMDGPU.AnyROCArray
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
        assert_warm_device_noalloc(AMDGPUScalarAdapter()) do
            analysis_sphtor!(vector_plan, plan_S, plan_T, vector_Vt, vector_Vp)
        end
        assert_warm_device_noalloc(AMDGPUScalarAdapter()) do
            synthesis_sphtor!(vector_plan, plan_Vt, plan_Vp, plan_S, plan_T)
        end
        test_mixed_precision_single_field()
        test_device_routing_fixes()
        variant_cfg = create_gauss_config(
            5, 8; nlon=14, mres=2, norm=:schmidt,
            real_norm=true, cs_phase=false,
        )
        variant_dense = _variant_coefficients(variant_cfg, Float32)
        variant_packed = SHTnsKit.pack_lm(variant_cfg, variant_dense)
        variant_field = synthesis_packed(variant_cfg, variant_packed)
        device_field = ROCArray(variant_field)
        device_packed = ROCArray(variant_packed)
        @test analysis_packed(variant_cfg, device_field) isa AMDGPU.AnyROCArray
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
        @test Array(analysis_axisym(variant_cfg, ROCArray(axis_field))) ≈
              axis_coefficients atol=2f-4 rtol=2f-4
        @test Array(synthesis_axisym(variant_cfg, ROCArray(axis_coefficients))) ≈
              axis_field atol=2f-4 rtol=2f-4

        im, physical_m = 2, 4
        mode_coefficients = ComplexF32.(variant_dense[(physical_m + 1):end, physical_m + 1])
        mode = synthesis_packed_ml(variant_cfg, im, mode_coefficients, variant_cfg.lmax)
        @test Array(analysis_packed_ml(
            variant_cfg, im, ROCArray(mode), variant_cfg.lmax,
        )) ≈ mode_coefficients atol=2f-4 rtol=2f-4
        @test Array(synthesis_packed_ml(
            variant_cfg, im, ROCArray(mode_coefficients), variant_cfg.lmax,
        )) ≈ mode atol=2f-4 rtol=2f-4

        batch_fields = repeat(
            reshape(Float32.(reshape(variant_field, variant_cfg.nlat, variant_cfg.nlon)),
                    variant_cfg.nlat, variant_cfg.nlon, 1), 1, 1, 2,
        )
        device_batch = ROCArray(batch_fields)
        batch_coefficients = analysis_batch(variant_cfg, device_batch)
        @test batch_coefficients isa AMDGPU.AnyROCArray
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
        device_complex = ROCArray(complex_coefficients)
        @test Array(synthesis_packed_cplx(complex_cfg, device_complex)) ≈
              complex_field atol=2f-4 rtol=2f-4
        @test Array(analysis_packed_cplx(complex_cfg, ROCArray(complex_field))) ≈
              complex_coefficients atol=2f-4 rtol=2f-4
        run_gpu_scalar_variant_matrix(AMDGPUScalarAdapter())
        run_scalar_full_parity(
            AMDGPUScalarAdapter();
            grid_kinds=_SCALAR_GRID_KINDS,
            precisions=(Float32, Float64),
            mres_values=(1, 2),
            norms=(:orthonormal, :fourpi, :schmidt),
            real_norm_values=(false, true),
            cs_phase_values=(false, true),
            pole_orders=(false, true),
        )
        run_gpu_vector_batch_parity(AMDGPUVectorAdapter())
        GPUWrapperReference.run_gpu_qst_regressions(SHTnsKit, ROCArray, AMDGPU.AnyROCArray)
        run_shared_legendre_precision_reference(
            extension.GPUCommon, ROCBackend(); place=ROCArray,
        )
        run_sphtor_full_parity(AMDGPUVectorAdapter())
        run_qst_full_parity(AMDGPUQSTAdapter())
        run_gpu_vector_mode_edge_parity(AMDGPUVectorAdapter())
        run_concurrent_local_first_use(
            AMDGPULocalEvaluationAdapter(), extension._amdgpu_clear_scalar_cache!,
        )
        run_local_evaluation_parity(AMDGPULocalEvaluationAdapter())
        run_gpu_operator_parity(AMDGPUOperatorAdapter())
    end
end
