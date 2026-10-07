#!/usr/bin/env julia
#
# Regression for communicator cleanup compatibility across MPI.jl releases.
# Run with:
#   mpiexec -n 2 julia --project test/parallel/test_mpi_comm_cleanup.jl

using MPI
MPI.Init()

using Test
using PencilArrays
using PencilFFTs
using SHTnsKit

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
const ParExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)

@testset "MPI communicator cleanup ($nprocs ranks)" begin
    duplicate = MPI.Comm_dup(comm)
    @test duplicate != MPI.COMM_NULL

    ParExt._safe_comm_free(duplicate)

    # MPI.jl invalidates a successfully freed mutable communicator handle by
    # replacing its value with MPI_COMM_NULL. This catches compatibility shims
    # that silently skip the release when only MPI.free is available.
    @test duplicate == MPI.COMM_NULL

    # Cleanup remains safe when called more than once.
    @test isnothing(ParExt._safe_comm_free(duplicate))
    @test duplicate == MPI.COMM_NULL
end

@testset "per-call pencils share cached topologies ($nprocs ranks)" begin
    # Every `Pencil(dims, decomp, comm)` creates a Cartesian communicator and
    # its subcommunicators, which only GC finalizers release. Transforms that
    # built one per call ran MPICH out of communicators (2048) after ~1000 calls.
    cfg = create_gauss_config(7, 10; nlon=16)
    spectral = SHTnsKit.create_spectral_pencil(cfg; comm)
    spatial = SHTnsKit.create_spatial_pencil(cfg; comm)
    @test PencilArrays.topology(SHTnsKit.create_spectral_pencil(cfg; comm)) ===
          PencilArrays.topology(spectral)
    @test PencilArrays.topology(spatial) === PencilArrays.topology(spectral)

    f = PencilArray{Float64}(undef, spatial)
    parent(f) .= [cos(0.3i) * sin(0.2j) for i in PencilArrays.range_local(spatial)[1],
                                            j in PencilArrays.range_local(spatial)[2]]
    a1 = analysis(cfg, f)
    a2 = analysis(cfg, f)
    @test PencilArrays.topology(pencil(a1)) === PencilArrays.topology(pencil(a2))

    # With finalizers unable to run, a per-call topology would fail long before
    # 1100 calls.
    completed = 0
    GC.enable(false)
    try
        for _ in 1:1100
            analysis(cfg, f)
            completed += 1
        end
    finally
        GC.enable(true)
    end
    @test completed == 1100
end

MPI.Barrier(comm)
rank == 0 && println("MPI communicator cleanup regression complete")
