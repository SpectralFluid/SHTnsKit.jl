#!/usr/bin/env julia
#
# Hardware-free regressions for the MPI + GPU compound-extension boundary: the
# vendor PencilArray firewall (ext/ParallelGPUVendorFirewall.jl) and the staged
# collectives in ext/ParallelGPU.jl. A stand-in compound extension includes the
# same firewall source over CPU-backed mock device storage. The rank-varying
# residency cases need at least two ranks and are skipped on one.
# Run with:
#   mpiexec -n 2 julia --project test/parallel/test_mpi_gpu_vendor_boundary.jl

module MPIGPUVendorBoundaryTests

using MPI
MPI.Initialized() || MPI.Init()

using LinearAlgebra
using PencilArrays
using PencilFFTs
using SHTnsKit
using Test

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
const ParExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)

include("mpi_gpu_firewall_probe.jl")

"""
CPU-backed stand-in for CUDA/ROCm storage. Like a device array it has no scalar
indexing, and bulk `copyto!` exists only against `Array` (the linear forms that
CUDA.jl and AMDGPU.jl implement), so element-wise host copies fail.
"""
mutable struct MockDeviceArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

MockDeviceArray{T}(::UndefInitializer, dims::Dims) where {T} =
    MockDeviceArray(Array{T}(undef, dims))
MockDeviceArray{T}(::UndefInitializer, dims::Integer...) where {T} =
    MockDeviceArray{T}(undef, Dims(dims))
Base.size(array::MockDeviceArray) = size(array.data)
Base.IndexStyle(::Type{<:MockDeviceArray}) = IndexLinear()
Base.getindex(::MockDeviceArray, ::Int) =
    error("Scalar indexing is disallowed for mock device storage")
Base.setindex!(::MockDeviceArray, _, ::Int) =
    error("Scalar indexing is disallowed for mock device storage")
Base.similar(::MockDeviceArray, ::Type{T}, dims::Dims) where {T} =
    MockDeviceArray{T}(undef, dims)
Base.copyto!(destination::Array{T}, destination_offset::Integer,
             source::MockDeviceArray{T}, source_offset::Integer,
             n::Integer) where {T} =
    (copyto!(destination, destination_offset, source.data, source_offset, n);
     destination)
Base.copyto!(destination::MockDeviceArray{T}, destination_offset::Integer,
             source::Array{T}, source_offset::Integer, n::Integer) where {T} =
    (copyto!(destination.data, destination_offset, source, source_offset, n);
     destination)
Base.copyto!(destination::Array{T}, source::MockDeviceArray{T}) where {T} =
    copyto!(destination, 1, source, 1, length(source))
Base.copyto!(destination::MockDeviceArray{T}, source::Array{T}) where {T} =
    copyto!(destination, 1, source, 1, length(source))

# Shaped like CUDA.AnyCuArray and AMDGPU.AnyROCArray: the device type plus
# wrappers, including some (`LogicalIndex`, `Adjoint`) that change the rank.
const AnyMockDeviceArray{T,N} = Union{
    MockDeviceArray{T,N},
    SubArray{T,N,<:MockDeviceArray},
    Base.ReshapedArray{T,N,<:MockDeviceArray},
    Base.LogicalIndex{T,<:MockDeviceArray},
    LinearAlgebra.Adjoint{T,<:MockDeviceArray{T,N}},
}

"""Stand-in for `SHTnsKitParallel{CUDA,AMDGPU}Ext` over mock device storage."""
module MockParallelVendorExt
import SHTnsKit
import MPI
import PencilArrays
import PencilFFTs
import LinearAlgebra
const ParallelExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
const VendorArray = parentmodule(@__MODULE__).AnyMockDeviceArray
const VendorArrayType = parentmodule(@__MODULE__).MockDeviceArray
const VENDOR_NAME = Val(:mock_vendor)
include(joinpath(pkgdir(SHTnsKit), "ext", "ParallelGPUVendorFirewall.jl"))
end

# Registered like the CUDA/ROCm adapters and with their transfer callbacks. MPI
# is never device-aware here, so collectives go through pinned host staging.
const MOCK_ADAPTER = ParExt.ParallelGPUAdapter(
    :mock_vendor,
    value -> ParExt._parallel_root_buffer(value) isa MockDeviceArray,
    _ -> MockDeviceArray,
    _ -> 0,
    _ -> false,
    _ -> nothing,
    (T, n) -> Vector{T}(undef, n),
    (host, device) -> ParExt._bulk_device_to_host!(host, device),
    (device, host) -> ParExt._bulk_host_to_device!(device, host),
)

host_data(array::MockDeviceArray) = array.data
host_data(array::AbstractArray) = array

const cfg = create_gauss_config(3, 6; nlon=8)

function stored_pencil_array(storage, ::Type{T}, dims, decomposition,
                             extra_dims...) where {T}
    array = PencilArray{T}(
        undef, Pencil(storage, dims, decomposition, comm), extra_dims...,
    )
    fill!(host_data(parent(array)), zero(T))
    return array
end

spatial(storage=MockDeviceArray, T=Float64, extra_dims...) = stored_pencil_array(
    storage, T, (cfg.nlat, cfg.nlon), (1,), extra_dims...,
)
spectral(storage=MockDeviceArray, T=ComplexF64, extra_dims...) =
    stored_pencil_array(
        storage, T, (cfg.lmax + 1, cfg.mmax + 1), (2,), extra_dims...,
    )

"""The operation a call reports as unavailable, or a summary of what it did."""
function unavailable_operation(call)
    try
        call()
        return :returned
    catch error
        error isa SHTnsKit.BackendUnavailableError && return error.operation
        return string(typeof(error), ": ",
                      first(split(sprint(showerror, error), '\n')))
    end
end

"""
Run `f()`, killing this rank if it has not returned after `seconds`. A
mismatched collective sequence then fails the MPI job instead of hanging it.
"""
function with_watchdog(f, seconds::Integer)
    Sys.iswindows() && return f()
    notice = "rank $rank: no result after $seconds s, killing this rank"
    watchdog = run(pipeline(
        `sh -c "sleep $seconds >/dev/null 2>&1; echo '$notice' >&2; kill -9 $(getpid())"`;
        stdin=devnull, stderr,
    ); wait=false)
    try
        return f()
    finally
        kill(watchdog, Base.SIGKILL)
    end
end

"""Run `f()` under a watchdog: `:mismatch` if it throws the residency error."""
function residency_outcome(f)
    return with_watchdog(120) do
        try
            f()
            :returned
        catch error
            message = sprint(showerror, error)
            error isa ArgumentError &&
                occursin("storage/vendor/device mismatch", message) ?
                :mismatch : string(typeof(error), ": ",
                                   first(split(message, '\n')))
        end
    end
end

"""A stand-in communicator that records the collectives issued on it."""
mutable struct CollectiveLog
    events::Vector{Symbol}
end

MPI.Allreduce(value, _op, log::CollectiveLog) =
    (push!(log.events, :Allreduce); value)

ParExt._register_parallel_gpu_adapter!(MOCK_ADAPTER)
try
    @testset "MPI GPU vendor boundary ($nprocs ranks)" begin
        @testset "vendor PencilArray methods win dispatch" begin
            VendorPencilArray = MockParallelVendorExt.VendorPencilArray
            @test VendorPencilArray <: PencilArrays.PencilArray
            @test spatial() isa VendorPencilArray
            @test !(spatial(Array) isa VendorPencilArray)
            # Every method of the included firewall must be selected for
            # vendor-backed arguments, including keyword calls.
            probe = firewall_dispatch_probe(
                MockParallelVendorExt, (T, N) -> MockDeviceArray{T,N},
            )
            for failure in first(probe.failures, 5)
                @error "firewall dispatch probe failed" failure
            end
            @test probe.probes > 1000
            @test length(probe.failures) == 0
        end

        @testset "firewall entry points preflight storage, then reject" begin
            field = spatial()
            output_field = spatial()
            batch = spatial(MockDeviceArray, Float64, 2)
            coefficients = spectral()
            output = spectral()
            batch_coefficients = spectral(MockDeviceArray, ComplexF64, 2)
            dense = MockDeviceArray(
                zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1),
            )
            dense_output = MockDeviceArray(zeros(ComplexF64, size(dense)))
            packed = MockDeviceArray(zeros(ComplexF64, cfg.nlm))
            mx = zeros(Float64, 2cfg.nlm)
            plan = ParExt.create_distributed_spectral_plan(
                cfg.lmax, cfg.mmax, comm; mres=cfg.mres,
            )
            plan_2d = ParExt.create_distributed_spectral_plan_2d(
                cfg.lmax, cfg.mmax, comm; p_l=1, p_m=nprocs, mres=cfg.mres,
            )
            payloads() = map(
                value -> copy(host_data(value isa PencilArray ?
                                        parent(value) : value)),
                (field, output_field, batch, coefficients, output,
                 batch_coefficients, dense, dense_output, packed),
            )
            before = payloads()
            before_stats = ParExt.parallel_gpu_stats()
            calls = (
                # do-block `_stage_vendor_call` sites
                :analysis => () -> SHTnsKit.analysis(cfg, field),
                :energy_scalar =>
                    () -> SHTnsKit.energy_scalar(cfg, coefficients),
                :synthesis_sph_ml => () -> SHTnsKit.synthesis_sph_ml(
                    cfg, 0, coefficients, cfg.lmax),
                :analysis_packed =>
                    () -> SHTnsKit.analysis_packed(cfg, field),
                :analysis_packed_l =>
                    () -> SHTnsKit.analysis_packed_l(cfg, field, cfg.lmax),
                :synthesis_axisym =>
                    () -> SHTnsKit.synthesis_axisym(cfg, coefficients),
                :synthesis_axisym_l => () -> SHTnsKit.synthesis_axisym_l(
                    cfg, coefficients, cfg.lmax),
                :analysis_packed_ml => () -> SHTnsKit.analysis_packed_ml(
                    cfg, 0, field, cfg.lmax),
                :synthesis_packed_ml => () -> SHTnsKit.synthesis_packed_ml(
                    cfg, 0, coefficients, cfg.lmax),
                :analysis_batch => () -> SHTnsKit.analysis_batch(cfg, batch),
                :synthesis_point => () -> SHTnsKit.synthesis_point(
                    cfg, coefficients, 0.25, 0.5),
                :SH_to_lat =>
                    () -> SHTnsKit.SH_to_lat(cfg, coefficients, 0.25),
                :dist_SH_to_lat =>
                    () -> SHTnsKit.dist_SH_to_lat(cfg, coefficients, 0.25),
                :dist_SH_to_point => () -> SHTnsKit.dist_SH_to_point(
                    cfg, coefficients, 0.25, 0.5),
                :dist_analysis_packed =>
                    () -> SHTnsKit.dist_analysis_packed(cfg, field),
                :dist_analysis_packed_cplx =>
                    () -> SHTnsKit.dist_analysis_packed_cplx(cfg, field),
                :dist_analysis => () -> SHTnsKit.dist_analysis(cfg, field),
                :dist_scalar_roundtrip! =>
                    () -> SHTnsKit.dist_scalar_roundtrip!(cfg, field),
                :dist_SH_Zrotate =>
                    () -> SHTnsKit.dist_SH_Zrotate(cfg, coefficients, 0.3),
                :enstrophy => () -> SHTnsKit.enstrophy(cfg, coefficients),
                # communicator taken from the first vendor PencilArray
                :analysis_sphtor =>
                    () -> SHTnsKit.analysis_sphtor(cfg, field, field),
                :analysis_qst_ml => () -> SHTnsKit.analysis_qst_ml(
                    cfg, 0, coefficients, coefficients, coefficients, cfg.lmax),
                :SH_to_grad_point => () -> SHTnsKit.SH_to_grad_point(
                    cfg, coefficients, coefficients, 0.25, 0.5),
                :dist_SH_Yrotate90 => () -> SHTnsKit.dist_SH_Yrotate90(
                    cfg, coefficients, output),
                :energy_vector => () -> SHTnsKit.energy_vector(
                    cfg, coefficients, coefficients),
                :SH_mul_mx =>
                    () -> SHTnsKit.SH_mul_mx(cfg, mx, coefficients, output),
                :dist_spatial_divergence =>
                    () -> SHTnsKit.dist_spatial_divergence(cfg, field, field),
                :dist_scalar_laplacian =>
                    () -> SHTnsKit.dist_scalar_laplacian(cfg, field),
                :dist_apply_laplacian! =>
                    () -> SHTnsKit.dist_apply_laplacian!(cfg, coefficients),
                :dist_SH_rotate_euler => () -> SHTnsKit.dist_SH_rotate_euler(
                    cfg, coefficients, 0.1, 0.2, 0.3, output),
                # communicator taken from an explicit Pencil argument
                :spectral_pencil_to_matrix =>
                    () -> SHTnsKit.spectral_pencil_to_matrix(cfg, coefficients),
                :synthesis => () -> SHTnsKit.synthesis(
                    cfg, coefficients; prototype_θφ=field),
                :synthesis_sph => () -> SHTnsKit.synthesis_sph(
                    cfg, coefficients; prototype_θφ=field),
                :synthesis_sphtor_l => () -> SHTnsKit.synthesis_sphtor_l(
                    cfg, coefficients, coefficients, cfg.lmax;
                    prototype_θφ=field),
                :synthesis_sph_l_cplx => () -> SHTnsKit.synthesis_sph_l_cplx(
                    cfg, coefficients, cfg.lmax; prototype_θφ=field),
                :synthesis_qst => () -> SHTnsKit.synthesis_qst(
                    cfg, coefficients, coefficients, coefficients;
                    prototype_θφ=field),
                :divergence_from_spheroidal! =>
                    () -> SHTnsKit.divergence_from_spheroidal!(
                        cfg, output, coefficients),
                :synthesis_packed => () -> SHTnsKit.synthesis_packed(
                    cfg, coefficients; prototype_θφ=field),
                :analysis_batch! => () -> SHTnsKit.analysis_batch!(
                    cfg, batch_coefficients, batch),
                :synthesis_batch! => () -> SHTnsKit.synthesis_batch!(
                    cfg, batch, batch_coefficients),
                :dist_synthesis => () -> SHTnsKit.dist_synthesis(
                    cfg, coefficients; prototype_θφ=field),
                :dist_scalar_laplacian! =>
                    () -> SHTnsKit.dist_scalar_laplacian!(
                        cfg, output_field, field),
                # dense vendor coefficients with a distributed prototype
                :dist_synthesis_dense => () -> SHTnsKit.dist_synthesis(
                    cfg, dense; prototype_θφ=field),
                :dist_synthesis_sphtor_dense =>
                    () -> SHTnsKit.dist_synthesis_sphtor(
                        cfg, dense, dense; prototype_θφ=field),
                :dist_synthesis_qst_dense => () -> SHTnsKit.dist_synthesis_qst(
                    cfg, dense, dense, dense; prototype_θφ=field),
                :dist_synthesis_packed => () -> SHTnsKit.dist_synthesis_packed(
                    cfg, packed; prototype_θφ=field),
                :dist_SH_Yrotate_packed =>
                    () -> SHTnsKit.dist_SH_Yrotate_packed(
                        cfg, packed, 0.2; prototype_lm=coefficients),
                :dist_SH_Xrotate90_packed =>
                    () -> SHTnsKit.dist_SH_Xrotate90_packed(
                        cfg, packed; prototype_lm=coefficients),
                :matrix_to_spectral_pencil =>
                    () -> SHTnsKit.matrix_to_spectral_pencil(cfg, dense; comm),
                # distributed-storage plans
                :distributed_spectral_reduce! =>
                    () -> ParExt.distributed_spectral_reduce!(
                        plan, dense, dense_output),
                :dist_analysis_distributed =>
                    () -> ParExt.dist_analysis_distributed(cfg, field; plan),
                :dist_analysis_distributed_2d =>
                    () -> ParExt.dist_analysis_distributed_2d(
                        cfg, field; plan=plan_2d),
            )
            try
                for (operation, call) in calls
                    @test unavailable_operation(call) === operation
                end
            finally
                close(plan_2d)
            end
            @test payloads() == before
            @test ParExt.parallel_gpu_stats() == before_stats
        end

        @testset "exchange! issues one collective sequence for any residency" begin
            # Mock-specific copies keep this independent of the transfer path.
            data_adapter = ParExt.ParallelGPUAdapter(
                :mock_vendor, MOCK_ADAPTER.matches, MOCK_ADAPTER.array_type,
                MOCK_ADAPTER.device, MOCK_ADAPTER.gpu_aware,
                MOCK_ADAPTER.synchronize, MOCK_ADAPTER.allocate_pinned,
                (host, device) -> copyto!(host, device.data),
                (device, host) -> copyto!(device.data, host),
            )
            collective = function (send, receive, log)
                push!(log.events, :exchange)
                return copyto!(receive, send)
            end
            host_log = CollectiveLog(Symbol[])
            host_receive = zeros(Float32, 2)
            ParExt.exchange!(Float32[1, 2], host_receive, host_log; collective)
            device_log = CollectiveLog(Symbol[])
            device_receive = MockDeviceArray(zeros(Float32, 2))
            ParExt.exchange!(
                MockDeviceArray(Float32[1, 2]), device_receive, device_log;
                adapter=data_adapter, collective,
            )
            @test host_receive == device_receive.data == Float32[1, 2]
            @test last(host_log.events) === :exchange
            @test host_log.events == device_log.events
            ParExt.parallel_gpu_clear_caches!()
        end

        @testset "rank-varying residency fails collectively" begin
            if nprocs > 1
                # The last rank holds mock device storage, the others hold
                # CPU storage, so the generic and firewall methods meet in
                # the same storage preflight.
                device = rank == nprocs - 1
                storage = device ? MockDeviceArray : Array
                field = spatial(storage)
                dense = storage(zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1))
                send = storage(fill(Float32(rank), nprocs))
                receive = storage(zeros(Float32, nprocs))
                exchanged = Ref(0)
                alltoall = function (send, receive, comm)
                    exchanged[] += 1
                    return MPI.Alltoall!(
                        MPI.UBuffer(send, 1), MPI.UBuffer(receive, 1), comm,
                    )
                end
                cases = (
                    :analysis => () -> SHTnsKit.analysis(cfg, field),
                    :analysis_sphtor =>
                        () -> SHTnsKit.analysis_sphtor(cfg, field, field),
                    :dense_synthesis => () -> SHTnsKit.dist_synthesis(
                        cfg, dense; prototype_θφ=field),
                    :exchange => () -> ParExt.exchange!(
                        send, receive, comm; collective=alltoall),
                )
                @testset "$name" for (name, call) in cases
                    outcome = residency_outcome(call)
                    @test outcome === :mismatch
                    @test MPI.Allreduce(outcome === :mismatch, &, comm)
                end
                @test exchanged[] == 0
                @test host_data(receive) == zeros(Float32, nprocs)
            else
                @test_skip nprocs > 1
            end
        end

        @testset "staged transfers use bulk host copies" begin
            # Element-wise host copies hit the mock's scalar-indexing error,
            # as they would for CUDA and ROCm arrays.
            @test_throws ErrorException copyto!(
                view(zeros(Float32, 4), 1:3), MockDeviceArray(Float32[1, 2, 3]),
            )
            for file in ("SHTnsKitParallelCUDAExt.jl",
                         "SHTnsKitParallelAMDGPUExt.jl")
                source = read(joinpath(pkgdir(SHTnsKit), "ext", file), String)
                @test occursin("ParallelExt._bulk_device_to_host!", source)
                @test occursin("ParallelExt._bulk_host_to_device!", source)
            end
            ParExt.parallel_gpu_clear_caches!()

            # Plain, offset and reshaped host stages.
            expected = reshape(Float32.(1:6), 3, 2)
            for stage in (zeros(Float32, 6), view(zeros(Float32, 8), 2:7),
                          reshape(view(zeros(Float32, 8), 1:6), 2, 3))
                ParExt._device_to_host_snapshot!(
                    MOCK_ADAPTER, stage, MockDeviceArray(copy(expected)),
                )
                @test vec(collect(stage)) == vec(expected)
                target = MockDeviceArray(zeros(Float32, 3, 2))
                ParExt._host_to_device_snapshot!(MOCK_ADAPTER, target, stage)
                @test target.data == expected
            end
            @test_throws DimensionMismatch ParExt._bulk_device_to_host!(
                zeros(Float32, 5), MockDeviceArray(copy(expected)),
            )

            buffer = MockDeviceArray(Float32[1, 2, 3])
            @test ParExt.allreduce!(buffer, +, comm) === buffer
            @test buffer.data == nprocs .* Float32[1, 2, 3]

            width = 2
            send = MockDeviceArray(Float32.(100rank .+ (1:width * nprocs)))
            receive = MockDeviceArray(zeros(Float32, width * nprocs))
            alltoall = (send, receive, comm) -> MPI.Alltoall!(
                MPI.UBuffer(send, width), MPI.UBuffer(receive, width), comm,
            )
            @test ParExt.exchange!(
                send, receive, comm; collective=alltoall,
            ) === receive
            @test receive.data == reduce(vcat, [
                Float32.(100source .+ (width * rank .+ (1:width)))
                for source in 0:nprocs-1
            ])

            # Whole-call staging snapshots reshaped host views, restores the
            # mutated input, and returns array and Pencil results on device.
            matrix = MockDeviceArray(reshape(Float32.(1:6), 2, 3))
            field = spatial()
            fill!(parent(field).data, rank + 1)
            result = MockParallelVendorExt._internal_staged_vendor_call_with_adapter(
                MOCK_ADAPTER, comm, :staged_bulk_copies,
                (host_matrix, host_field) -> begin
                    host_matrix .*= 2
                    (host_matrix .+ 1, 3 .* host_field)
                end,
                matrix, field; mutated=(1,),
            )
            @test matrix.data == 2 .* reshape(Float32.(1:6), 2, 3)
            @test result[1] isa MockDeviceArray
            @test host_data(result[1]) == 2 .* reshape(Float32.(1:6), 2, 3) .+ 1
            @test result[2] isa PencilArray
            @test parent(result[2]) isa MockDeviceArray
            @test all(==(3(rank + 1)), host_data(parent(result[2])))
            ParExt.parallel_gpu_clear_caches!()
        end
    end
finally
    lock(ParExt._PARALLEL_GPU_ADAPTER_LOCK) do
        delete!(ParExt._PARALLEL_GPU_ADAPTERS, :mock_vendor)
    end
    ParExt.parallel_gpu_clear_caches!()
end

end # module
