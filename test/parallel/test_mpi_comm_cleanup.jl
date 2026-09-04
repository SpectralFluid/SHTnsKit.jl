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

MPI.Barrier(comm)
rank == 0 && println("MPI communicator cleanup regression complete")
