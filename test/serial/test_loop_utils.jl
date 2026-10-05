# SHTnsKit.jl - Loop Utilities Tests
# Tests for @sht_loop macro, loop backend, helper functions

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

# Minimal array wrapper whose type name exercises the GPU dispatch branch without
# requiring a CUDA-capable host. The injected launcher executes the body on the
# CPU; the contract under test is the macro-to-extension handoff.
struct TestCuArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.size(a::TestCuArray) = size(a.data)
Base.getindex(a::TestCuArray, I...) = getindex(a.data, I...)
Base.setindex!(a::TestCuArray, value, I...) = setindex!(a.data, value, I...)

@testset "Loop Utilities" begin
    @testset "loop_backend" begin
        @test SHTnsKit.loop_backend() isa String
        @test SHTnsKit.loop_backend() in ("auto", "SIMD")
    end

    @testset "set_loop_backend" begin
        old = SHTnsKit.loop_backend()

        SHTnsKit.set_loop_backend("SIMD")
        @test SHTnsKit.loop_backend() == "SIMD"

        SHTnsKit.set_loop_backend("auto")
        @test SHTnsKit.loop_backend() == "auto"

        @test_throws ArgumentError SHTnsKit.set_loop_backend("invalid")

        SHTnsKit.set_loop_backend(old)
    end

    @testset "_is_cpu_array" begin
        @test SHTnsKit._is_cpu_array(rand(5)) == true
        @test SHTnsKit._is_cpu_array(zeros(3, 4)) == true
        @test SHTnsKit._is_cpu_array(ones(ComplexF64, 2, 3)) == true
    end

    @testset "_is_pencil_array" begin
        # Regular arrays should not be PencilArrays
        @test SHTnsKit._is_pencil_array(rand(5)) == false
        @test SHTnsKit._is_pencil_array(zeros(3, 4)) == false
    end

    @testset "_get_local_data" begin
        arr = rand(5)
        @test SHTnsKit._get_local_data(arr) === arr
    end

    @testset "spectral_range" begin
        r = SHTnsKit.spectral_range(4, 4)
        @test r isa CartesianIndices
        @test size(r) == (5, 5)
    end

    @testset "spatial_range" begin
        r = SHTnsKit.spatial_range(8, 16)
        @test r isa CartesianIndices
        @test size(r) == (8, 16)
    end

    @testset "latitude_range" begin
        r = SHTnsKit.latitude_range(10)
        @test r == 1:10
    end

    @testset "mode_range" begin
        r = SHTnsKit.mode_range(5)
        @test r == 0:5
    end

    @testset "local_range" begin
        arr = rand(4, 8)
        r = SHTnsKit.local_range(arr)
        @test r == CartesianIndices(arr)
    end

    @testset "local_size" begin
        arr = rand(4, 8)
        @test SHTnsKit.local_size(arr) == (4, 8)
    end

    @testset "CI shorthand" begin
        idx = SHTnsKit.CI(2, 3)
        @test idx == CartesianIndex(2, 3)
    end

    @testset "δ unit index" begin
        d1 = SHTnsKit.δ(1, CartesianIndex(2, 3))
        @test d1 == CartesianIndex(1, 0)

        d2 = SHTnsKit.δ(2, CartesianIndex(2, 3))
        @test d2 == CartesianIndex(0, 1)
    end

    @testset "inside" begin
        arr = zeros(6, 8)
        r = SHTnsKit.inside(arr)
        # Default buff=1, so interior is [2:5, 2:7]
        @test size(r) == (4, 6)

        r2 = SHTnsKit.inside(arr; buff=2)
        @test size(r2) == (2, 4)
    end

    @testset "_loop_index_symbols" begin
        @test SHTnsKit._loop_index_symbols(:I) == [:I]
        @test SHTnsKit._loop_index_symbols(Expr(:tuple, :i, :j)) == [:i, :j]
    end

    @testset "@sht_loop basic" begin
        # Test that @sht_loop writes correct values
        dest = zeros(4, 8)
        src = rand(4, 8)
        SHTnsKit.@sht_loop dest[I] = src[I] over I ∈ CartesianIndices(dest)
        @test dest ≈ src

        tuple_dest = zeros(2, 3)
        tuple_src = reshape(collect(1.0:6.0), 2, 3)
        SHTnsKit.@sht_loop tuple_dest[i, j] = tuple_src[i, j] over (i, j) ∈ CartesianIndices(tuple_dest)
        @test tuple_dest == tuple_src
    end

    @testset "@sht_loop checks bounds before its @inbounds loop" begin
        # The shipped loop demo wrote an 8×8 array over the 10×10 interior:
        # silent heap corruption under @inbounds.
        data = reshape(collect(1.0:100.0), 10, 10)
        small = zeros(8, 8)
        @test_throws BoundsError SHTnsKit.@sht_loop small[I] = data[I + δ(1, I)] - data[I] over I ∈ inside(data)
        @test all(iszero, small)  # rejected before any write
        out = zeros(10, 10)
        SHTnsKit.@sht_loop out[I] = data[I + δ(1, I)] - data[I - δ(1, I)] over I ∈ inside(data)
        @test out[2:9, 2:9] == fill(2.0, 8, 8)
        @test_throws BoundsError SHTnsKit.@sht_loop out[I] = data[I + δ(2, I)] over I ∈ CartesianIndices(out)
        short = zeros(3)
        @test_throws BoundsError SHTnsKit.@sht_loop short[i] = 1.0 over i ∈ 1:4
        indices = [1, 2, 7]
        @test_throws BoundsError SHTnsKit.@sht_loop short[k] = 1.0 over k ∈ indices
        # Corner values do not bound indirect or non-affine indices, which are
        # therefore checked at every iteration.
        src3 = [1.0, 2.0, 3.0]
        dest3 = zeros(3)
        perm = [1, 4, 3]
        @test_throws BoundsError SHTnsKit.@sht_loop dest3[i] = src3[perm[i]] over i ∈ 1:3
        @test all(iszero, dest3)
        dest7 = zeros(7)
        @test_throws BoundsError SHTnsKit.@sht_loop dest7[i + 4] = src3[abs(i)] over i ∈ -3:3
        @test all(iszero, dest7)
        perm = [3, 1, 2]
        SHTnsKit.@sht_loop dest3[i] = src3[perm[i]] over i ∈ 1:3
        @test dest3 == [3.0, 1.0, 2.0]
    end

    @testset "@sht_loop checks only accesses every iteration makes" begin
        # A guarded access may be out of range where its guard fails, and an
        # operand that is not an array has no checkbounds; both ran before the
        # checks were added and must keep running.
        src = collect(1.0:5.0)
        shifted = [0.0, 1.0, 2.0, 3.0, 4.0]
        out = zeros(5)
        SHTnsKit.@sht_loop out[i] = i > 1 ? src[i - 1] : 0.0 over i ∈ 1:5
        @test out == shifted
        out = zeros(5)
        SHTnsKit.@sht_loop if i > 1; out[i] = src[i - 1]; end over i ∈ 1:5
        @test out == shifted
        out = zeros(5)
        SHTnsKit.@sht_loop begin
            i < 5 || return
            out[i] = src[i + 1]
        end over i ∈ 1:5
        @test out == [2.0, 3.0, 4.0, 5.0, 0.0]
        grid = reshape(collect(1.0:12.0), 3, 4)
        nlon = 4
        wrapped = zeros(3, 4)
        SHTnsKit.@sht_loop wrapped[i, j] = j == 1 ? grid[i, nlon] : grid[i, j - 1] over (i, j) ∈ CartesianIndices(wrapped)
        @test wrapped == circshift(grid, (0, 1))

        scale = Ref(2.0)
        weights = (1.0, 2.0, 3.0)
        offsets = Dict(1 => 10.0)
        out = zeros(3)
        SHTnsKit.@sht_loop out[i] = scale[] * weights[i] + offsets[1] over i ∈ 1:3
        @test out == [12.0, 14.0, 16.0]

        # An index that calls anything but integer arithmetic is not evaluated
        # ahead of the loop: it may have side effects.
        calls = Ref(0)
        counted(i) = (calls[] += 1; i)
        out = zeros(5)
        SHTnsKit.@sht_loop out[i] = src[counted(i)] over i ∈ 1:5
        @test calls[] == 5
        @test out == src
    end

    @testset "@sht_loop field-access hygiene" begin
        # A field access in the body must read THAT field, never an unrelated
        # caller local of the same name. Rewriting `cfg.scale` to the bare symbol
        # `scale` used to silently pick up the local below and compile cleanly.
        holder = (scale = 3.0,)
        scale = -100.0          # decoy: same name, wrong value
        dest = zeros(6)
        src = collect(1.0:6.0)
        SHTnsKit.@sht_loop dest[i] = holder.scale * src[i] over i ∈ 1:6
        @test dest ≈ 3.0 .* src
        @test scale == -100.0   # untouched

        # Same field name reached through two different objects in one body.
        a = (w = 2.0,)
        b = (w = 5.0,)
        out = zeros(4)
        ones4 = ones(4)
        SHTnsKit.@sht_loop out[i] = a.w * ones4[i] + b.w over i ∈ 1:4
        @test out ≈ fill(7.0, 4)

        # Repeated operands must not produce duplicate kernel argument names.
        d2 = zeros(5)
        s2 = collect(1.0:5.0)
        SHTnsKit.@sht_loop d2[i] = s2[i] + s2[i] over i ∈ 1:5
        @test d2 ≈ 2 .* s2
    end


    @testset "@sht_loop GPU launcher contract" begin
        old_available = SHTnsKit._GPU_LOOP_AVAILABLE[]
        old_launcher = SHTnsKit._GPU_KERNEL_LAUNCHER[]
        old_backend = SHTnsKit.loop_backend()
        launches = Ref(0)

        launcher = function (args...)
            range = args[end - 1]
            body = args[end]
            launches[] += 1
            foreach(body, range)
            return nothing
        end

        try
            SHTnsKit.set_loop_backend("auto")
            SHTnsKit._enable_gpu_loops!(launcher)

            src = TestCuArray(reshape(collect(1.0:6.0), 2, 3))
            dest = TestCuArray(zeros(2, 3))
            holder = (scale = 2.5,)
            offset = -0.25

            SHTnsKit.@sht_loop dest[i, j] = holder.scale * src[i, j] + offset over (i, j) ∈ CartesianIndices(dest)

            @test launches[] == 1
            @test dest.data ≈ holder.scale .* src.data .+ offset
        finally
            SHTnsKit._GPU_LOOP_AVAILABLE[] = old_available
            SHTnsKit._GPU_KERNEL_LAUNCHER[] = old_launcher
            SHTnsKit.set_loop_backend(old_backend)
        end
    end
end
