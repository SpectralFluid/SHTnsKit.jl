# Exercise the production wrapper functions without vendor hardware. Only array
# allocation, device selection and FFT execution are modeled; the wrapper bodies
# and shared transform kernels are loaded directly from ext/.
module GPUWrapperReference

using Test, SHTnsKit
import KernelAbstractions
include("../../ext/GPUCommon.jl")
include("test_mres.jl")
include("test_qst_regressions.jl")

module HostVendor
    using KernelAbstractions
    # Vendor `Any*Array` unions admit views of device storage too.
    const AnyROCArray{T,N} = Union{Array{T,N}, SubArray{T,N,<:Array}}
    const AnyCuArray{T,N} = Union{Array{T,N}, SubArray{T,N,<:Array}}
    const ROCArray = Array
    const CuArray = Array
    # Tests switch rocFFT off to check which operations still run without it.
    const ROCFFT_AVAILABLE = Ref(true)
    functional(args...) = true
    functional(feature::Symbol) = feature === :rocfft ? ROCFFT_AVAILABLE[] : true
    device() = 0
    device_id() = 0
    deviceid(device) = device
    const SYNCHRONIZE_CALLS = Ref(0)
    synchronize() = (SYNCHRONIZE_CALLS[] += 1;
                     KernelAbstractions.synchronize(KernelAbstractions.CPU()))
    zeros(args...) = Base.zeros(args...)
    unsafe_free!(_) = nothing
end

module HostFFT
    using SHTnsKit
    const FFTW = SHTnsKit.FFTW
    # rocFFT implements * for in-place plans and mul! only for out-of-place
    # plans. Deliberately provide no mul! method for this wrapper.
    struct InplacePlan{P}
        plan::P
    end
    Base.:*(plan::InplacePlan, values::Array) = plan.plan * values
    plan_fft!(args...) = InplacePlan(FFTW.plan_fft!(args...))
    plan_ifft!(args...) = InplacePlan(FFTW.plan_ifft!(args...))
    plan_rfft(args...) = FFTW.plan_rfft(args...)
    plan_irfft(args...) = FFTW.plan_irfft(args...)
    fft!(args...) = FFTW.fft!(args...)
    ifft!(args...) = FFTW.ifft!(args...)
end

function function_name(expr)
    signature = expr.args[1]
    while signature isa Expr && signature.head === :where
        signature = signature.args[1]
    end
    return signature isa Expr && signature.head === :call ? signature.args[1] : nothing
end

function wrapper_module(vendor)
    prefix = vendor === :AMDGPU ? "amdgpu" : "cuda"
    filename = vendor === :AMDGPU ? "SHTnsKitAMDGPUExt.jl" : "SHTnsKitGPUExt.jl"
    source = Meta.parse(read(joinpath(@__DIR__, "../../ext", filename), String))
    sandbox = Module(gensym(:VendorWrapper))
    Core.eval(sandbox, :(const GPUCommon = $GPUCommon))
    Core.eval(sandbox, :(const $vendor = $HostVendor))
    Core.eval(sandbox, :(const FFTW = $(vendor === :AMDGPU ? HostFFT : SHTnsKit.FFTW)))
    Core.eval(sandbox, :(const CUFFT = $(SHTnsKit.FFTW)))
    Core.eval(sandbox, :(const gpu_ifft! = $(HostFFT.ifft!)))
    Core.eval(sandbox, :(const gpu_fft! = $(HostFFT.fft!)))
    Core.eval(sandbox, :(const ROCArray = Array))
    Core.eval(sandbox, :(const CuArray = Array))
    Core.eval(sandbox, :(const ROCBackend = $(KernelAbstractions.CPU)))
    Core.eval(sandbox, :(const CUDABackend = $(KernelAbstractions.CPU)))
    # Deterministically simulate stale allocator memory. A wrapper must clear
    # Fourier bins it does not write before passing them to the inverse FFT.
    Core.eval(sandbox, :(const similar = (args...) -> fill!(Base.similar(args...), NaN)))
    functions = Set(Symbol.("_" .* prefix .* [
        "_scalar_tables", "_vector_tables", "_workspace_builder",
        "_vector_workspace_builder", "_batch_scratch",
        "_scalar_analysis_direct!", "_scalar_synthesis_direct!",
        "_batch_analysis_direct!", "_batch_synthesis_direct!",
        "_vector_analysis_direct!", "_vector_synthesis_direct!",
        "_batch_analysis", "_batch_synthesis", "_vector_batch_analysis",
        "_vector_batch_synthesis", "_rotation_blocks",
        "_local_tables", "_local_precision", "_validate_local_arrays",
        "_local_scalar", "_local_qst", "_lcap", "_mode_synthesis",
        "_coefficient_scales", "_safe_required_memory",
        "_device_array", "_scalar_analysis", "_scalar_synthesis", "_local_constants",
        "_vector_analysis", "_vector_synthesis",
    ]))
    # Public wrappers are defined as sandbox-local functions:
    # the sandbox evaluates the extension's `using` lines but not its
    # `import SHTnsKit: ...` lines, so no SHTnsKit method is added.
    union!(functions, [Symbol("_with_", prefix, "_workspace"),
                       Symbol("_with_", prefix, "_vector_workspace"),
                       Symbol("_require_", prefix),
                       :synthesis_axisym, :synthesis_axisym_l,
                       :analysis_qst_batch, :synthesis_qst_batch, :synthesis_qst_batch_cplx,
                       :synthesis_point, :synthesis_point_cplx, :SHqst_to_point, :SH_to_grad_point,
                       :SH_to_lat, :SH_to_lat_cplx, :SHqst_to_lat,
                       :estimate_memory_usage, :_legendre_table_bytes,
                       :analysis!, :synthesis!, :gpu_apply_laplacian!,
                       :_gpu_adapter_matches, :_gpu_adapter_dense,
                       :_gpu_adapter_analysis, :_gpu_adapter_synthesis,
                       :_gpu_adapter_analysis_sphtor, :_gpu_adapter_synthesis_sphtor])
    typeprefix = vendor === :AMDGPU ? "AMDGPU" : "CUDA"
    structs = Set(Symbol.(typeprefix .* [
        "ScalarTables", "VectorTables", "RotationBlocks", "LocalTables", "Adapter",
    ]))
    constants = Set(Symbol.("_" .* typeprefix .* [
        "_SCALAR_CACHE", "_VECTOR_CACHE", "_WORKSPACE_CACHE", "_LOCAL_CACHE",
        "_LOCAL_CONSTANTS",
    ]))
    push!(constants, :_rotation_cache)
    for expr in source.args[3].args
        expr isa Expr || continue
        if expr.head === :using
            # Preserve the production imports (including LinearAlgebra) so a
            # missing mul! binding is caught, while replacing vendor/FFT APIs.
            for imported in expr.args
                imported.head === :. && first(imported.args) in (:AMDGPU, :CUDA, :FFTW) && continue
                Core.eval(sandbox, Expr(:using, imported))
            end
        elseif expr.head === :struct
            declaration = expr.args[2]
            name = declaration isa Expr ? declaration.args[1] : declaration
            name in structs && Core.eval(sandbox, expr)
        elseif expr.head === :const
            expr.args[1].args[1] in constants && Core.eval(sandbox, expr)
        elseif expr.head in (:function, :(=)) && function_name(expr) in functions
            Core.eval(sandbox, expr)
        elseif expr.head === :macrocall &&
               expr.args[1] in (Symbol("@inline"), GlobalRef(Core, Symbol("@doc"))) &&
               last(expr.args) isa Expr && last(expr.args).head in (:function, :(=)) &&
               function_name(last(expr.args)) in functions
            Core.eval(sandbox, expr)
        end
    end
    return sandbox
end

@testset "GPU wrapper host reference" begin
    rocm = wrapper_module(:AMDGPU)
    cuda = wrapper_module(:CUDA)
    for wrapper in (rocm, cuda)
        run_gpu_qst_regressions(wrapper, identity, Array)
    end
    @testset "Order stride through production wrappers" begin
        for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda"))
            scalar_analysis! = getproperty(wrapper, Symbol("_", prefix, "_scalar_analysis_direct!"))
            scalar_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_scalar_synthesis_direct!"))
            vector_analysis! = getproperty(wrapper, Symbol("_", prefix, "_vector_analysis_direct!"))
            vector_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_vector_synthesis_direct!"))
            run_gpu_mres_tests(
                scalar_analysis=(cfg, field) -> begin
                    output = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
                    scalar_analysis!(cfg, cfg, output, field)
                end,
                scalar_synthesis=(cfg, coefficients; real_output=true) -> begin
                    output = zeros(real_output ? Float64 : ComplexF64, cfg.nlat, cfg.nlon)
                    scalar_synthesis!(cfg, cfg, output, coefficients; real_output)
                end,
                vector_analysis=(cfg, vt, vp) -> begin
                    sout = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
                    tout = similar(sout)
                    vector_analysis!(cfg, cfg, sout, tout, vt, vp)
                end,
                vector_synthesis=(cfg, s, t; real_output=true) -> begin
                    vt = zeros(real_output ? Float64 : ComplexF64, cfg.nlat, cfg.nlon)
                    vp = similar(vt)
                    vector_synthesis!(cfg, cfg, vt, vp, s, t; real_output)
                end,
            )
        end
    end
    @testset "Planned FFT execution" begin
        for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda")),
            T in (Float32, Float64), use_rfft in (false, true)
            scalar_analysis! = getproperty(wrapper, Symbol("_", prefix, "_scalar_analysis_direct!"))
            scalar_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_scalar_synthesis_direct!"))
            batch_analysis! = getproperty(wrapper, Symbol("_", prefix, "_batch_analysis_direct!"))
            batch_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_batch_synthesis_direct!"))
            vector_analysis! = getproperty(wrapper, Symbol("_", prefix, "_vector_analysis_direct!"))
            vector_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_vector_synthesis_direct!"))
            cfg = create_gauss_config(3, 8; nlon=10)
            coeff = zeros(Complex{T}, 4, 4)
            coeff[2, 1] = T(0.25)
            coeff[3, 3] = Complex{T}(0.1, 0.2)
            field = synthesis(cfg, coeff)
            output = similar(coeff)
            rebuilt = similar(field)
            tol = T === Float32 ? 3e-5 : 2e-12
            @test scalar_analysis!(cfg, cfg, output, field; use_rfft) === output
            @test output ≈ coeff atol=tol rtol=tol
            @test scalar_synthesis!(cfg, cfg, rebuilt, coeff; use_rfft) === rebuilt
            @test rebuilt ≈ field atol=tol rtol=tol
            fields = cat(field, 2field; dims=3)
            coefficients = cat(coeff, 2coeff; dims=3)
            batchout = similar(coefficients)
            spatialout = similar(fields)
            bins = use_rfft ? cfg.nlon ÷ 2 + 1 : cfg.nlon
            scratch = zeros(Complex{T}, cfg.nlat, bins, 2)
            padded = fill(Complex{T}(17), cfg.nlat, 2bins, 2)
            strided = view(padded, :, 1:2:2bins, :)
            for fft_batch in (nothing, scratch, strided)
                @test batch_analysis!(cfg, batchout, fields; use_rfft, fft_batch) === batchout
                @test batchout ≈ coefficients atol=tol rtol=tol
                @test batch_synthesis!(cfg, spatialout, coefficients; use_rfft, fft_batch) === spatialout
                @test spatialout ≈ fields atol=tol rtol=tol
            end
            @test all(==(17), view(padded, :, 2:2:2bins, :))
            @test any(!=(17), strided)
            vt, vp = synthesis_sphtor(cfg, coeff, 2coeff)
            sout, tout = similar(coeff), similar(coeff)
            @test vector_analysis!(cfg, cfg, sout, tout, vt, vp; use_rfft) === (sout, tout)
            @test sout ≈ coeff atol=tol rtol=tol
            @test tout ≈ 2coeff atol=tol rtol=tol
            vtout, vpout = similar(vt), similar(vp)
            @test vector_synthesis!(cfg, cfg, vtout, vpout, coeff, 2coeff; use_rfft) === (vtout, vpout)
            @test vtout ≈ vt atol=tol rtol=tol
            @test vpout ≈ vp atol=tol rtol=tol
        end
    end
    @testset "Normalize coefficients before narrowing the destination" begin
        cfg = create_gauss_config(2, 4; nlon=7, norm=:fourpi)
        field = fill(1.5e38, cfg.nlat, cfg.nlon)
        fields = cat(field, -field; dims=3)
        expected = ComplexF32.(analysis(cfg, field))
        @test expected[1, 1] ≈ 1.5f38
        for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda")), use_rfft in (false, true)
            analyze! = getproperty(wrapper, Symbol("_", prefix, "_scalar_analysis_direct!"))
            batch! = getproperty(wrapper, Symbol("_", prefix, "_batch_analysis_direct!"))
            out = similar(expected)
            batch = similar(expected, size(expected)..., 2)
            analyze!(cfg, cfg, out, field; use_rfft)
            batch!(cfg, batch, fields; use_rfft)
            @test isfinite(out[1, 1]) && out[1, 1] ≈ expected[1, 1]
            @test batch[1, 1, :] ≈ [expected[1, 1], -expected[1, 1]]
        end
    end
    @testset "Rotation blocks follow the effective angles" begin
        # The block cache was keyed on the stored angle fields, but setter-built
        # rotations swap α and γ: a keyword rotation and a setter rotation with the
        # same fields shared one entry, and whichever ran second got the other's
        # blocks. Order-mixing rotations on an mmax < lmax layout must also be
        # rejected exactly as on the CPU instead of silently truncated.
        for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda")), T in (Float32, Float64)
            blocks = getproperty(wrapper, Symbol("_", prefix, "_rotation_blocks"))
            L = 5
            keyword = SHTRotation(L, L; α=0.3, β=0.5, γ=1.1)
            setter = SHTRotation(L, L)
            shtns_rotation_set_angles_ZYZ(setter, 0.3, 0.5, 1.1)
            Q = Complex{T}.(range(0.1, 1.3; length=SHTnsKit.nlm_calc(L, L, 1)) .+
                            0.2im .* cos.(1:SHTnsKit.nlm_calc(L, L, 1)))
            tol = T === Float32 ? 2e-5 : 1e-12
            for rotation in (keyword, setter)    # keyword first: setter must not reuse it
                b = blocks(rotation, T)
                R = similar(Q)
                GPUCommon.rotation_real_kernel!(KernelAbstractions.CPU())(
                    R, Q, b.offsets, b.values, b.input_factors, b.output_factors,
                    L, L; ndrange=length(Q),
                )
                KernelAbstractions.synchronize(KernelAbstractions.CPU())
                expected = similar(Q)
                shtns_rotation_apply_real(rotation, Q, expected)
                @test R ≈ expected atol=tol rtol=tol
            end
            @test_throws ArgumentError blocks(SHTRotation(L + 2, L - 1; β=0.7), T)
            @test blocks(SHTRotation(L + 2, L - 1; β=Float64(π)), T) isa Any  # half-turn keeps |m|
        end
    end
    @testset "Float32 tables keep Float32 accuracy" begin
        # Tables were built from nodes already rounded to Float32 with a Float32
        # recurrence, which cost about two digits at lmax=127 (three by 255).
        # They are now built in Float64 and rounded once, like the CPU tables.
        cfg = create_gauss_config(127, 130)
        coeff = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        for m in 0:cfg.mmax, l in m:cfg.lmax
            coeff[l + 1, m + 1] = complex(sin(1.3l + 0.7m), m == 0 ? 0.0 : cos(0.9l - 1.1m))
        end
        coeff[1, 1] = 0
        relerr(a, b) = maximum(abs, a .- b) / maximum(abs, b)
        expected = synthesis(cfg, coeff)
        vt_expected, vp_expected = synthesis_sphtor(cfg, coeff, 0.5 .* coeff)
        for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda"))
            scalar_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_scalar_synthesis_direct!"))
            vector_synthesis! = getproperty(wrapper, Symbol("_", prefix, "_vector_synthesis_direct!"))
            field = zeros(Float32, cfg.nlat, cfg.nlon)
            scalar_synthesis!(cfg, cfg, field, ComplexF32.(coeff))
            @test relerr(field, expected) < 2e-6
            vt = zeros(Float32, cfg.nlat, cfg.nlon)
            vp = similar(vt)
            vector_synthesis!(cfg, cfg, vt, vp, ComplexF32.(coeff), ComplexF32.(0.5 .* coeff))
            @test relerr(vt, vt_expected) < 2e-6
            @test relerr(vp, vp_expected) < 2e-6
        end
    end
    @testset "Device caches release memory they no longer need" begin
        # Table entries keyed by objectid(cfg) used to outlive the config.
        tables = GPUCommon.ScalarTableCache(4)
        function insert_for_temporary_owner!(cache)
            local temporary = [0.0]  # stands in for an SHTConfig
            GPUCommon.scalar_cache_insert!(
                cache, :device, objectid(temporary), Float64, UInt(1), :tables;
                owner=temporary,
            )
            @test GPUCommon.scalar_cache_size(cache) == 1
            return nothing
        end
        insert_for_temporary_owner!(tables)
        GC.gc(); GC.gc()
        @test GPUCommon.scalar_cache_size(tables) == 0

        # A live owner keeps its entry; a reused object id does not see it.
        owner = [1.0]
        identity = objectid(owner)
        GPUCommon.scalar_cache_insert!(tables, :device, identity, Float64,
                                       UInt(2), :live; owner)
        @test GPUCommon.scalar_cache_lookup(tables, :device, identity, Float64,
                                            UInt(2); owner) === :live
        @test GPUCommon.scalar_cache_lookup(tables, :device, identity, Float64,
                                            UInt(2); owner=[1.0]) === nothing

        # Planned transforms key their workspace on the SHTPlan, held weakly.
        # A live plan must keep it through a collection instead of rebuilding.
        workspaces = GPUCommon.ScalarWorkspaceCache(4)
        plan = SHTPlan(create_gauss_config(4, 6))
        builds = Ref(0)
        use_workspace() = GPUCommon.scalar_workspace_use!(
            workspace -> workspace, () -> (builds[] += 1; :workspace),
            workspaces, :device, plan, Float64, :fft, (1,), UInt(0),
        )
        use_workspace()
        GC.gc(); GC.gc()
        @test use_workspace() === :workspace
        @test builds[] == 1

        # Dense rotation blocks grow like lmax³, so the cache is byte-bounded.
        blocks = GPUCommon.RotationBlockCache(8; max_bytes_per_device=1000)
        block(n) = (values=zeros(UInt8, n), alpha=0.0)
        for k in 1:3
            GPUCommon.rotation_cache_insert!(blocks, (:device, k), block(400))
        end
        @test GPUCommon.rotation_cache_size(blocks; device=:device) == 2
        @test GPUCommon.rotation_cache_lookup(blocks, (:device, 1)) === nothing
        oversized = block(2000)
        @test GPUCommon.rotation_cache_insert!(blocks, (:device, 4), oversized) === oversized
        @test GPUCommon.rotation_cache_lookup(blocks, (:device, 4)) === nothing

        # The vector signature covers the Nlm column the kernels read without
        # building a Tuple of all of Nlm on every transform.
        cfg = create_gauss_config(200, 202)
        before = GPUCommon.vector_config_signature(cfg)
        @test (@allocated GPUCommon.vector_config_signature(cfg)) < 4096
        cfg.Nlm[3, 2] *= 2
        @test GPUCommon.vector_config_signature(cfg) != before
    end
    @testset "Local evaluators and axisymmetric synthesis follow phi_scale" begin
        # These paths dropped the φ factor `synthesis` carries (1 under :dft,
        # 1/2π under :quad), so under :quad every GPU point/latitude value and
        # m = 0 column came back 2π too large. They must match the CPU in both.
        withenv("SHTNSKIT_PHI_SCALE" => nothing) do  # the variable overrides cfg.phi_scale
            for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda")),
                T in (Float32, Float64), phi_scale in (:dft, :quad)
                local_scalar = getproperty(wrapper, Symbol("_", prefix, "_local_scalar"))
                local_qst = getproperty(wrapper, Symbol("_", prefix, "_local_qst"))
                cfg = create_gauss_config(5, 8; nlon=13)
                cfg.phi_scale = phi_scale
                @test SHTnsKit.phi_inv_scale(cfg) ≈ (phi_scale === :quad ? cfg.nlon / 2π : cfg.nlon)
                tol = T === Float32 ? 3e-5 : 1e-12
                alm = zeros(Complex{T}, 6, 6)
                alm[1, 1] = T(0.7)
                alm[3, 2] = Complex{T}(0.2, -0.3)
                alm[6, 4] = Complex{T}(-0.1, 0.25)
                Q, S, Tlm = (SHTnsKit.pack_lm(cfg, A) for A in (alm, 2alm, -alm))
                cost, phi = T(0.3), T(1.1)
                @test local_scalar(cfg, alm, cost, phi)[] ≈
                      synthesis_point(cfg, alm, cost, phi) rtol=tol
                @test collect(map(only, local_qst(cfg, Q, S, Tlm, cost, phi))) ≈
                      collect(SHqst_to_point(cfg, Q, S, Tlm, cost, phi)) rtol=tol
                @test collect(local_qst(cfg, Q, S, Tlm, cost, zero(T); nphi=7,
                                        ltr=4, mtr=3)) ≈
                      collect(SHqst_to_lat(cfg, Q, S, Tlm, cost; nphi=7, ltr=4, mtr=3)) rtol=tol
                Z = Complex{T}.(range(0.1, 0.9; length=SHTnsKit.nlm_cplx_calc(5, 5, 1)) .+
                                0.3im .* sin.(1:SHTnsKit.nlm_cplx_calc(5, 5, 1)))
                @test local_scalar(cfg, Z, cost, phi; complex_layout=true)[] ≈
                      synthesis_point_cplx(cfg, Z, cost, phi) rtol=tol
                @test local_scalar(cfg, Z, cost, zero(T); nphi=9, ltr=4,
                                   complex_layout=true) ≈
                      SH_to_lat_cplx(cfg, Z, cost; nphi=9, ltr=4) rtol=tol
                column = alm[:, 1]
                @test wrapper.synthesis_axisym(SHTnsKit.GPU(), cfg, column) ≈
                      synthesis_axisym(cfg, column) rtol=tol
                @test wrapper.synthesis_axisym_l(SHTnsKit.GPU(), cfg, column, 3) ≈
                      synthesis_axisym_l(cfg, column, 3) rtol=tol
            end
        end
    end
    @testset "Point evaluation builds no full-grid table" begin
        # The local tables took their scales from the scalar tables, building the
        # nlat×(lmax+1)×(mmax+1) Legendre table (8.6 GB at lmax=1023) for one point.
        for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda"))
            local_scalar = getproperty(wrapper, Symbol("_", prefix, "_local_scalar"))
            scalar_tables = getproperty(wrapper, Symbol("_", prefix, "_scalar_tables"))
            cache = getproperty(wrapper, Symbol("_", uppercase(prefix), "_SCALAR_CACHE"))
            cfg = create_gauss_config(6, 9; nlon=15, norm=:schmidt)
            alm = zeros(ComplexF64, 7, 7)
            alm[4, 3] = 0.4 - 0.2im
            alm[7, 6] = 0.1im
            @test local_scalar(cfg, alm, 0.2, 0.7)[] ≈ synthesis_point(cfg, alm, 0.2, 0.7) rtol=1e-12
            @test GPUCommon.scalar_cache_lookup(
                cache, 0, objectid(cfg), Float64,
                GPUCommon.scalar_config_signature(cfg); owner=cfg,
            ) === nothing
            # Resident scalar tables still share their scales.
            scales = getproperty(wrapper, Symbol("_", prefix, "_coefficient_scales"))
            @test scales(cfg, Float64) == GPUCommon.coefficient_scales(cfg, Float64)
            resident = scalar_tables(cfg, Float64)
            @test scales(cfg, Float64) === resident.scales
        end
    end
    @testset "Safe wrappers count resident tables once" begin
        # Resident tables were counted again, so once they filled half the free
        # memory, every gpu_*_safe call after the first fell back to the CPU.
        cfg = create_gauss_config(9, 12; nlon=19)
        table_bytes = cfg.nlat * (cfg.lmax + 1) * (cfg.mmax + 1) * 8
        field = zeros(cfg.nlat, cfg.nlon)
        coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        analysis_bytes = cuda.estimate_memory_usage(cfg, :analysis)
        synthesis_bytes = cuda.estimate_memory_usage(cfg, :synthesis)
        @test cuda._cuda_safe_required_memory(cfg, :analysis, field) == analysis_bytes
        cuda._cuda_scalar_tables(cfg, Float64)
        @test cuda._cuda_safe_required_memory(cfg, :analysis, field) ==
              analysis_bytes - table_bytes
        @test cuda._cuda_safe_required_memory(cfg, :synthesis, coefficients) ==
              synthesis_bytes - table_bytes
        # Float32 input needs Float32 tables, which are not resident yet.
        @test cuda._cuda_safe_required_memory(cfg, :analysis, Float32.(field)) ==
              analysis_bytes
    end
    @testset "In-place, Laplacian and nested-wrapper routing" begin
        for (wrapper, prefix, Adapter) in ((rocm, "amdgpu", rocm.AMDGPUAdapter),
                                           (cuda, "cuda", cuda.CUDAAdapter)),
            T in (Float32, Float64)
            cfg = create_gauss_config(3, 8; nlon=10)
            coeff = zeros(Complex{T}, 4, 4)
            coeff[2, 1] = T(0.25)
            coeff[3, 3] = Complex{T}(0.1, 0.2)
            field = synthesis(cfg, coeff)
            tol = T === Float32 ? 3e-5 : 2e-12
            # The configuration-level in-place transforms run the planned kernels.
            output = similar(coeff)
            @test wrapper.analysis!(cfg, output, field) === output
            @test output ≈ coeff atol=tol rtol=tol
            rebuilt = similar(field)
            @test wrapper.synthesis!(cfg, rebuilt, coeff) === rebuilt
            @test rebuilt ≈ field atol=tol rtol=tol
            @test_throws ArgumentError wrapper.analysis!(cfg, output, field;
                                                         fft_scratch=complex.(field))
            # The Laplacian kernel covers one field, so a batch is rejected.
            lap = copy(coeff)
            wrapper.gpu_apply_laplacian!(cfg, view(lap, :, :))
            @test lap ≈ [-l * (l + 1) * coeff[l + 1, m + 1] for l in 0:3, m in 0:3]
            batch = cat(coeff, coeff; dims=3)
            @test_throws DimensionMismatch wrapper.gpu_apply_laplacian!(cfg, batch)
            @test batch[:, :, 2] == coeff
            # A deeper wrapper matches the adapter and is copied into a dense
            # array before the kernels run.
            adapter = Adapter()
            nest(x) = view(PermutedDimsArray(permutedims(x), (2, 1)), :, :)
            @test wrapper._gpu_adapter_matches(adapter, nest(field))
            dense = wrapper._gpu_adapter_dense(adapter, nest(field))
            @test dense isa Matrix && dense == field
            @test wrapper._gpu_adapter_analysis(adapter, cfg, nest(field)) ≈ coeff atol=tol rtol=tol
            @test wrapper._gpu_adapter_synthesis(adapter, cfg, nest(coeff)) ≈ field atol=tol rtol=tol
            vt, vp = synthesis_sphtor(cfg, coeff, 2coeff)
            s, t = wrapper._gpu_adapter_analysis_sphtor(adapter, cfg, nest(vt), vp)
            @test s ≈ coeff atol=tol rtol=tol
            @test t ≈ 2coeff atol=tol rtol=tol
            v = wrapper._gpu_adapter_synthesis_sphtor(adapter, cfg, nest(coeff), 2coeff)
            @test v[1] ≈ vt atol=tol rtol=tol
            @test v[2] ≈ vp atol=tol rtol=tol
        end
    end
    @testset "Planned precision, output parity, caches and rocFFT-free calls" begin
        for (wrapper, prefix) in ((rocm, "amdgpu"), (cuda, "cuda"))
            fn(name) = getproperty(wrapper, Symbol("_", prefix, "_", name))
            cache(name) = getproperty(wrapper, Symbol("_", uppercase(prefix), "_", name))
            cfg = create_gauss_config(4, 8; nlon=12)
            coeff = zeros(ComplexF64, 5, 5)
            coeff[2, 1] = 0.25
            coeff[4, 3] = complex(0.1, 0.2)
            field = synthesis(cfg, coeff)
            # A wider output sets the planned precision, as a CPU plan's Float64 does.
            narrow = Float32.(field)
            spectrum = zeros(ComplexF64, size(coeff))
            fn("scalar_analysis_direct!")(cfg, cfg, spectrum, narrow)
            @test spectrum ≈ analysis(cfg, Float64.(narrow)) rtol=1e-12 atol=1e-14
            grid = zeros(cfg.nlat, cfg.nlon)
            fn("scalar_synthesis_direct!")(cfg, cfg, grid, ComplexF32.(coeff))
            @test grid ≈ synthesis(cfg, ComplexF64.(ComplexF32.(coeff))) rtol=1e-12 atol=1e-14
            # Real vector fields may land in complex storage, as on the CPU.
            vt, vp = synthesis_sphtor(cfg, coeff, 2coeff)
            zt, zp = zeros(ComplexF64, size(vt)), zeros(ComplexF64, size(vp))
            fn("vector_synthesis_direct!")(cfg, cfg, zt, zp, coeff, 2coeff)
            @test zt ≈ vt rtol=1e-12 atol=1e-14
            @test zp ≈ vp rtol=1e-12 atol=1e-14
            # A profile too short for the degree limit is an argument error, as on the CPU.
            @test_throws ArgumentError wrapper.synthesis_axisym_l(
                SHTnsKit.GPU(), cfg, ComplexF64[1, 2], 3)
            # A failing workspace body drains its submitted work before unlocking.
            with_workspace = getproperty(wrapper, Symbol("_with_", prefix, "_workspace"))
            before = HostVendor.SYNCHRONIZE_CALLS[]
            @test_throws ErrorException with_workspace(cfg, cfg, Float64, 0, false) do _
                error("kernel launch failed")
            end
            @test HostVendor.SYNCHRONIZE_CALLS[] > before
            # Vector tables share the small scalar tables but never build Plm.
            GPUCommon.scalar_cache_clear!(cache("SCALAR_CACHE"))
            GPUCommon.scalar_cache_clear!(cache("VECTOR_CACHE"))
            tables = fn("vector_tables")(cfg, Float64)
            @test GPUCommon.scalar_cache_size(cache("SCALAR_CACHE")) == 0
            @test tables.scales ≈ GPUCommon.coefficient_scales(cfg, Float64)
            # Every latitude shares the normalization table and convention scales,
            # even with no resident scalar tables to borrow them from.
            GPUCommon.scalar_cache_clear!(cache("SCALAR_CACHE"))
            GPUCommon.scalar_cache_clear!(cache("LOCAL_CACHE"))
            first_row = fn("local_tables")(cfg, Float64, cfg.x[1])
            second_row = fn("local_tables")(cfg, Float64, cfg.x[2])
            @test first_row.scales === second_row.scales
        end
        # Only the transforms need rocFFT.
        HostVendor.ROCFFT_AVAILABLE[] = false
        try
            cfg = create_gauss_config(3, 8; nlon=10)
            lap = zeros(ComplexF64, 4, 4)
            lap[3, 2] = 1
            rocm.gpu_apply_laplacian!(cfg, view(lap, :, :))
            @test lap[3, 2] == -6
            @test rocm._require_amdgpu(:SH_mul_mx; fft=false) === nothing
            @test_throws SHTnsKit.BackendUnavailableError rocm._require_amdgpu(:analysis)
        finally
            HostVendor.ROCFFT_AVAILABLE[] = true
        end
        # A live plan keeps its workspace however many plans are live. The cap
        # bounds the workspaces of configuration-level calls only.
        workspaces = GPUCommon.ScalarWorkspaceCache(1)
        builds = Ref(0)
        use(owner) = GPUCommon.scalar_workspace_use!(
            identity, () -> (builds[] += 1; owner), workspaces, :device, owner,
            Float64, :fft, (1,), UInt(0),
        )
        plans = [SHTPlan(create_gauss_config(3, 6)) for _ in 1:3]
        foreach(use, plans)
        foreach(use, plans)
        @test builds[] == 3
        owners = [Ref(1), Ref(2)]
        foreach(use, owners)
        foreach(use, owners)
        @test builds[] == 7
        @test GPUCommon.scalar_workspace_size(workspaces; device=:device) == 4
    end
    @testset "Vector batch synthesis clears unused Fourier bins" begin
        for (vendor, call) in ((:AMDGPU, rocm._amdgpu_vector_batch_synthesis),
                               (:CUDA, cuda._cuda_vector_batch_synthesis)),
            T in (Float32, Float64), mres in (1, 2), real_output in (false, true)
            cfg = create_gauss_config(3, 8; nlon=10, mres)
            coeff = zeros(Complex{T}, 4, 4, 2)
            for nonzero in (false, true)
                if nonzero
                    coeff[2, 1, 1] = T(0.25)
                    coeff[3, 3, 2] = Complex{T}(0.1, 0.2)
                end
                expected = synthesis_sphtor_batch(cfg, coeff, 2coeff; real_output)
                result = call(cfg, coeff, 2coeff; real_output)
                tol = T === Float32 ? 3e-5 : 2e-12
                @test result[1] ≈ expected[1] atol=tol rtol=tol
                @test result[2] ≈ expected[2] atol=tol rtol=tol
            end
        end
    end
end

end # module
