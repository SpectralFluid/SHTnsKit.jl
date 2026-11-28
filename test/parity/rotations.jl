using Test
using Random
using LinearAlgebra
using SHTnsKit

abstract type RotationParityAdapter end

struct CPURotationAdapter <: RotationParityAdapter end

rotation_place(::CPURotationAdapter, value) = value
rotation_collect(::CPURotationAdapter, value) = Array(value)
rotation_resident(::CPURotationAdapter, value) = nothing
rotation_z(::CPURotationAdapter, cfg, input, angle, output) =
    SH_Zrotate(CPU(), cfg, input, angle, output)
rotation_y(::CPURotationAdapter, cfg, input, angle, output) =
    SH_Yrotate(CPU(), cfg, input, angle, output)
rotation_apply_real(::CPURotationAdapter, rotation, input, output) =
    shtns_rotation_apply_real(CPU(), rotation, input, output)
rotation_apply_cplx(::CPURotationAdapter, rotation, input, output) =
    shtns_rotation_apply_cplx(CPU(), rotation, input, output)

function _rotation_real_spectrum(cfg, ::Type{T}; seed=0xA712) where {T<:AbstractFloat}
    rng = MersenneTwister(seed)
    coefficients = randn(rng, Complex{T}, cfg.nlm)
    @inbounds for k in eachindex(coefficients)
        cfg.mi[k] == 0 && (coefficients[k] = real(coefficients[k]))
    end
    return coefficients
end

function _rotation_external(cfg, canonical)
    external = similar(canonical)
    SHTnsKit.convert_alm_norm!(external, canonical, cfg; to_internal=false)
    return external
end

function _zyz_matrix(alpha, beta, gamma)
    ca, sa = cos(alpha), sin(alpha)
    cb, sb = cos(beta), sin(beta)
    cg, sg = cos(gamma), sin(gamma)
    return [
        ca*cb*cg - sa*sg  -ca*cb*sg - sa*cg  ca*sb
        sa*cb*cg + ca*sg  -sa*cb*sg + ca*cg  sa*sb
        -sb*cg              sb*sg                cb
    ]
end

function _angle_axis_matrix(theta, x, y, z)
    norm_axis = hypot(x, hypot(y, z))
    x, y, z = x / norm_axis, y / norm_axis, z / norm_axis
    c, s, t = cos(theta), sin(theta), one(theta) - cos(theta)
    return [
        c+t*x*x    t*x*y-s*z  t*x*z+s*y
        t*y*x+s*z  c+t*y*y    t*y*z-s*x
        t*z*x-s*y  t*z*y+s*x  c+t*z*z
    ]
end

function test_angle_axis_pi_singularity()
    @testset "angle-axis beta=pi ZYZ singularity" begin
        for T in (Float32, Float64)
            phi = T(0.37)
            cases = (
                (one(T), zero(T), zero(T), T(pi)),
                (zero(T), one(T), zero(T), zero(T)),
                (cos(phi), sin(phi), zero(T), T(pi) + 2phi),
            )
            tol = T === Float32 ? 2f-5 : 2e-12
            nc = nlm_cplx_calc(2, 2, 1)
            input = randn(MersenneTwister(0xA819 + sizeof(T)), Complex{T}, nc)
            for (x, y, z, expected_delta) in cases
                actual = SHTRotation(2, 2)
                shtns_rotation_set_angle_axis(actual, T(pi), x, y, z)
                @test _zyz_matrix(actual.α, actual.β, actual.γ) ≈
                      _angle_axis_matrix(T(pi), x, y, z) atol=tol rtol=tol

                # Direct constructors retain the matrix-order convention used
                # by angle-axis. The SHTns-compatible setter deliberately
                # reverses its two outer intrinsic-Z arguments on apply.
                expected = SHTRotation(
                    2, 2; α=expected_delta, β=T(pi), γ=zero(T),
                )
                actual_output = similar(input)
                expected_output = similar(input)
                shtns_rotation_apply_cplx(
                    SHTnsKit.CPU(), actual, input, actual_output,
                )
                shtns_rotation_apply_cplx(
                    SHTnsKit.CPU(), expected, input, expected_output,
                )
                @test actual_output ≈ expected_output atol=tol rtol=tol

                actual_d = zeros(T, 9); expected_d = zeros(T, 9)
                shtns_rotation_wigner_d_matrix(actual, 1, actual_d)
                shtns_rotation_wigner_d_matrix(expected, 1, expected_d)
                @test actual_d ≈ expected_d atol=tol rtol=tol

                actual_blocks = SHTnsKit._rotation_host_blocks(actual, T)
                expected_blocks = SHTnsKit._rotation_host_blocks(expected, T)
                @test actual_blocks.values ≈ expected_blocks.values atol=tol rtol=tol
                @test cis(actual_blocks.alpha) ≈ cis(expected_blocks.alpha) atol=tol
                @test cis(actual_blocks.gamma) ≈ cis(expected_blocks.gamma) atol=tol
            end
        end
    end
end

function run_rotation_parity(adapter::RotationParityAdapter;
                             precisions=(Float32, Float64))
    @testset "rotation mathematical parity" begin
        for T in precisions
            tol = T === Float32 ? T(8e-5) : T(2e-12)
            cfg = create_gauss_config(4, 7; nlon=12)
            q = _rotation_real_spectrum(cfg, T)
            placed = rotation_place(adapter, q)

            z = similar(placed)
            @test rotation_z(adapter, cfg, placed, T(0.37), z) === z
            rotation_resident(adapter, z)
            expected_z = similar(q)
            @inbounds for k in eachindex(q)
                # SHTns rotates the field by +alpha (equivalently, the
                # reference frame by -alpha), so exp(im*phi) coefficients
                # acquire exp(-im*alpha).
                expected_z[k] = q[k] * cis(-T(cfg.mi[k]) * T(0.37))
            end
            @test rotation_collect(adapter, z) ≈ expected_z atol=tol rtol=tol

            y = similar(placed)
            back = similar(placed)
            rotation_y(adapter, cfg, placed, T(0.41), y)
            rotation_y(adapter, cfg, y, T(-0.41), back)
            @test rotation_collect(adapter, back) ≈ q atol=tol rtol=tol

            # Full complex identity/inverse and norm conservation.
            nc = nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
            rng = MersenneTwister(0xC012 + sizeof(T))
            c = randn(rng, Complex{T}, nc)
            pc = rotation_place(adapter, c)
            rot = SHTRotation(cfg.lmax, cfg.mmax)
            shtns_rotation_set_angles_ZYZ(rot, T(0.2), T(-0.35), T(0.17))
            rc = similar(pc)
            rotation_apply_cplx(adapter, rot, pc, rc)
            invrot = SHTRotation(cfg.lmax, cfg.mmax)
            shtns_rotation_set_angles_ZYZ(invrot, T(-0.17), T(0.35), T(-0.2))
            cc = similar(pc)
            rotation_apply_cplx(adapter, invrot, rc, cc)
            @test eltype(rc) === Complex{T}
            @test rotation_collect(adapter, cc) ≈ c atol=tol rtol=tol
            @test sum(abs2, rotation_collect(adapter, rc)) ≈ sum(abs2, c) atol=tol rtol=tol

            # ZXZ is the documented Rz(a)Rx(b)Rz(c), independently reduced to ZYZ.
            zxz = SHTRotation(cfg.lmax, cfg.mmax)
            shtns_rotation_set_angles_ZXZ(zxz, T(0.31), T(0.52), T(-0.23))
            equivalent = SHTRotation(cfg.lmax, cfg.mmax)
            shtns_rotation_set_angles_ZYZ(
                equivalent, T(0.31 + pi / 2), T(0.52), T(-0.23 - pi / 2),
            )
            zx = similar(pc); zy = similar(pc)
            rotation_apply_cplx(adapter, zxz, pc, zx)
            rotation_apply_cplx(adapter, equivalent, pc, zy)
            @test rotation_collect(adapter, zx) ≈ rotation_collect(adapter, zy) atol=tol rtol=tol

            # l=1 little-d oracle and row-major public extraction.
            beta = T(0.43); cb = cos(beta); sb = sin(beta)
            expected_d1 = T[
                (1 + cb)/2  sb/sqrt(T(2))  (1 - cb)/2
                -sb/sqrt(T(2)) cb sb/sqrt(T(2))
                (1 - cb)/2 -sb/sqrt(T(2)) (1 + cb)/2
            ]
            d1 = Matrix{T}(undef, 3, 3)
            SHTnsKit.wigner_d_matrix!(d1, 1, beta)
            @test d1 ≈ expected_d1 atol=tol rtol=tol
            extract = zeros(T, 9)
            probe = SHTRotation(1, 1; β=beta)
            @test shtns_rotation_wigner_d_matrix(probe, 1, extract) == 3
            @test reshape(extract, 3, 3)' ≈ expected_d1 atol=tol rtol=tol
        end
    end
end

function test_cpu_rotation_conventions_and_validation()
    @testset "CPU rotation conventions, mres, aliases, and validation" begin
        canonical_cfg = create_gauss_config(5, 8; nlon=14)
        canonical = _rotation_real_spectrum(canonical_cfg, Float64)
        expected = similar(canonical)
        SH_Yrotate(CPU(), canonical_cfg, canonical, 0.39, expected)

        for convention in (
            (norm=:fourpi, real_norm=true, cs_phase=false),
            (norm=:schmidt, real_norm=true, cs_phase=true),
        )
            cfg = create_gauss_config(5, 8; nlon=14, convention...)
            external = _rotation_external(cfg, canonical)
            got = similar(external)
            SH_Yrotate(CPU(), cfg, external, 0.39, got)
            @test got ≈ _rotation_external(cfg, expected) atol=2e-12 rtol=2e-12

            inplace = copy(external)
            @test SH_Yrotate(CPU(), cfg, inplace, 0.39, inplace) === inplace
            @test inplace ≈ got atol=2e-12 rtol=2e-12
        end

        cfg2 = create_gauss_config(6, 9; nlon=16, mres=2)
        q2 = _rotation_real_spectrum(cfg2, Float32)
        zout = fill(ComplexF32(91, -17), cfg2.nlm)
        SH_Zrotate(CPU(), cfg2, q2, 0.2f0, zout)
        @test all(isfinite, zout)
        sentinel = fill(ComplexF32(91, -17), cfg2.nlm)
        @test_throws ArgumentError SH_Yrotate(CPU(), cfg2, q2, 0.2f0, sentinel)
        @test all(==(ComplexF32(91, -17)), sentinel)

        cfg = canonical_cfg
        short = fill(ComplexF64(91, -17), cfg.nlm - 1)
        output = fill(ComplexF64(91, -17), cfg.nlm)
        @test_throws DimensionMismatch SH_Zrotate(CPU(), cfg, short, 0.2, output)
        @test all(==(ComplexF64(91, -17)), output)
        @test_throws DimensionMismatch SH_Yrotate(CPU(), cfg, short, 0.2, output)
        @test all(==(ComplexF64(91, -17)), output)
        @test_throws ArgumentError SH_Zrotate(CPU(), cfg, canonical, NaN, output)
        @test all(==(ComplexF64(91, -17)), output)
        @test_throws ArgumentError SHTRotation(2, 2; α=NaN)
        probe = SHTRotation(2, 2; α=0.1, β=0.2, γ=0.3)
        @test_throws ArgumentError shtns_rotation_set_angle_axis(
            probe, Inf, 1.0, 0.0, 0.0,
        )
        @test (probe.α, probe.β, probe.γ) == (0.1, 0.2, 0.3)

        # Offset overlapping views may corrupt a later degree unless the input
        # is staged before the first output write.
        rot = SHTRotation(4, 4; α=0.2, β=0.3, γ=-0.1)
        nc = nlm_cplx_calc(4, 4, 1)
        storage = randn(MersenneTwister(0xA11A5), ComplexF64, nc + 1)
        source = @view storage[1:nc]
        destination = @view storage[2:(nc + 1)]
        reference = similar(source)
        shtns_rotation_apply_cplx(CPU(), rot, copy(source), reference)
        @test shtns_rotation_apply_cplx(CPU(), rot, source, destination) === destination
        @test destination ≈ reference atol=2e-12 rtol=2e-12

        mismatch = fill(ComplexF32(91, -17), nc)
        before = copy(mismatch)
        @test_throws ArgumentError shtns_rotation_apply_cplx(
            CPU(), rot, ComplexF64.(reference), mismatch,
        )
        @test mismatch == before
    end
end

function test_gpu_rotation_contract(extension, packed_type, complex_type)
    @testset "GPU rotation routing contract" begin
        common = extension.GPUCommon
        @test isdefined(common, :RotationBlockCache)
        @test isdefined(common, :rotation_z_real_kernel!)
        @test isdefined(common, :rotation_real_kernel!)
        @test isdefined(common, :rotation_cplx_kernel!)
        @test extension._rotation_cache.max_per_device == 8
        block_name = isdefined(extension, :CUDARotationBlocks) ?
                     :CUDARotationBlocks : :AMDGPURotationBlocks
        block_type = Base.unwrap_unionall(getfield(extension, block_name))
        @test fieldtype(block_type, :alpha) isa TypeVar
        @test fieldtype(block_type, :gamma) isa TypeVar

        cache = common.RotationBlockCache(2)
        completed = Ref(false)
        value = common.rotation_cache_publish!(cache, (:device0, :first), 1) do
            completed[] = true
        end
        @test completed[] && value == 1
        common.rotation_cache_insert!(cache, (:device0, :second), 2)
        common.rotation_cache_insert!(cache, (:device0, :third), 3)
        common.rotation_cache_insert!(cache, (:device1, :first), 4)
        @test common.rotation_cache_size(cache; device=:device0) == 2
        @test common.rotation_cache_size(cache; device=:device1) == 1
        @test which(
            SH_Zrotate,
            Tuple{SHTnsKit.GPU,SHTConfig,packed_type,Float32,packed_type},
        ).module === extension
        @test which(
            SH_Yrotate,
            Tuple{SHTnsKit.GPU,SHTConfig,packed_type,Float32,packed_type},
        ).module === extension
        @test which(
            shtns_rotation_apply_real,
            Tuple{SHTnsKit.GPU,SHTRotation,packed_type,packed_type},
        ).module === extension
        @test which(
            shtns_rotation_apply_cplx,
            Tuple{SHTnsKit.GPU,SHTRotation,complex_type,complex_type},
        ).module === extension
    end
end

function run_shared_rotation_kernel_reference(common, backend)
    @testset "shared rotation kernels" begin
        for T in (Float32, Float64)
            cfg = create_gauss_config(3, 6; nlon=10)
            q = _rotation_real_spectrum(cfg, T)
            z = similar(q)
            common.rotation_z_real_kernel!(backend)(
                z, q, SHTnsKit._zrotation_phases(cfg, T(0.37), Complex{T}); ndrange=cfg.nlm,
            )
            expected = similar(q)
            SH_Zrotate(SHTnsKit.CPU(), cfg, q, T(0.37), expected)
            tol = T === Float32 ? T(2e-5) : T(2e-13)
            @test z ≈ expected atol=tol rtol=tol

            rot = SHTRotation(3, 3; α=0.21, β=-0.34, γ=0.19)
            blocks = SHTnsKit._rotation_host_blocks(rot, T)
            output = similar(q)
            common.rotation_real_kernel!(backend)(
                output, q, blocks.offsets, blocks.values, blocks.input_scales,
                blocks.output_scales, blocks.alpha, blocks.gamma,
                rot.lmax, rot.mmax; ndrange=cfg.nlm,
            )
            reference = similar(q)
            shtns_rotation_apply_real(SHTnsKit.CPU(), rot, q, reference)
            @test output ≈ reference atol=tol rtol=tol

            nc = nlm_cplx_calc(3, 3, 1)
            c = randn(MersenneTwister(0x6A17 + sizeof(T)), Complex{T}, nc)
            cout = similar(c)
            common.rotation_cplx_kernel!(backend)(
                cout, c, blocks.offsets, blocks.values, blocks.input_scales,
                blocks.output_scales, blocks.alpha, blocks.gamma,
                rot.lmax, rot.mmax; ndrange=nc,
            )
            cref = similar(c)
            shtns_rotation_apply_cplx(SHTnsKit.CPU(), rot, c, cref)
            @test cout ≈ cref atol=tol rtol=tol
        end

        # High orders keep the CPU's phase accuracy in Float32. Forming m*angle
        # in Float32 put these coefficients about 1e-4 off at lmax = 255.
        cfg = create_gauss_config(255, 257; nlon=512)
        q = _rotation_real_spectrum(cfg, Float32)
        z = similar(q)
        common.rotation_z_real_kernel!(backend)(
            z, q, SHTnsKit._zrotation_phases(cfg, 6.18f0, ComplexF32); ndrange=cfg.nlm,
        )
        expected = SH_Zrotate(SHTnsKit.CPU(), cfg, q, 6.18f0, similar(q))
        @test maximum(abs, z - expected) < 1f-6 * maximum(abs, expected)
    end
end
