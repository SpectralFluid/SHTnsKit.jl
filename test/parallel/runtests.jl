# SHTnsKit.jl - Parallel Grid Resolution Tests
# Run all parametric tests across multiple lat/lon configurations
#
# This folder contains:
# - test_scalar_parametric.jl  : Scalar transforms across multiple grid sizes
# - test_vector_parametric.jl  : Vector (sphtor) transforms across multiple grid sizes
# - test_qst_parametric.jl     : QST (3D vector) transforms across multiple grid sizes
# - test_packed_storage.jl     : Non-MPI tests for packed storage utilities
# - test_threading.jl          : Native-thread correctness (run with `julia -t N`)
# - test_mpi_comprehensive.jl  : MPI distributed tests (run separately with mpiexec)
# - test_mpi_extended.jl       : Extended MPI tests (run separately with mpiexec)
# - test_transpose_sht.jl      : DistTransposePlan transforms, canonical grid (mpiexec)
# - test_disttranspose_dealiased.jl : DistTransposePlan on dealiased nlon>2*mmax+1 (mpiexec)
# - test_disttranspose_conventions.jl : DistTransposePlan configured coefficient conventions (mpiexec)
# - test_dist_plan_alloc.jl    : DistAnalysisPlan correctness + per-call allocation budget (mpiexec)
# - test_mpi_audit_fixes.jl    : Regressions for the 2026-08 distributed audit fixes (mpiexec)
# - test_mpi_ad_tangent_spaces.jl : Distributed ChainRules tangent-space contracts (mpiexec)
# - test_mpi_parallel_diagnostics_regressions.jl : Distributed diagnostic contracts (mpiexec)
# - test_mpi_parallel_local_correctness.jl : Distributed point/latitude evaluation contracts (mpiexec)
# - test_mpi_plan_preflight.jl : Rank-symmetric plan/configuration preflight (mpiexec)
# - test_mpi_transpose_operand_preflight.jl : DistTransposePlan operand contracts (mpiexec)
# - test_mpi_comm_cleanup.jl : MPI.jl-compatible subcommunicator cleanup (mpiexec)
#
# To run MPI tests:
#   mpiexec -n 4 julia --project test/parallel/test_mpi_comprehensive.jl
#   mpiexec -n 4 julia --project test/parallel/test_mpi_extended.jl
#   mpiexec -n 2 julia --project test/parallel/test_transpose_sht.jl
#   mpiexec -n 2 julia --project test/parallel/test_disttranspose_dealiased.jl
#   mpiexec -n 2 julia --project test/parallel/test_disttranspose_conventions.jl
#   mpiexec -n 2 julia --project test/parallel/test_dist_plan_alloc.jl
#   mpiexec -n 8 julia --project test/parallel/test_mpi_audit_fixes.jl
#   mpiexec -n 2 julia --project test/parallel/test_mpi_ad_tangent_spaces.jl
#   mpiexec -n 2 julia --project test/parallel/test_mpi_parallel_diagnostics_regressions.jl
#   mpiexec -n 2 julia --project test/parallel/test_mpi_parallel_local_correctness.jl
#   mpiexec -n 2 julia --project test/parallel/test_mpi_plan_preflight.jl
#   mpiexec -n 2 julia --project test/parallel/test_mpi_transpose_operand_preflight.jl
#   mpiexec -n 2 julia --project test/parallel/test_mpi_comm_cleanup.jl

using Test

@testset "SHTnsKit Parallel Grid Tests" begin
    include("test_scalar_parametric.jl")
    include("test_vector_parametric.jl")
    include("test_qst_parametric.jl")
    include("test_packed_storage.jl")
    include("test_threading.jl")
end

# Note: MPI tests are not included here as they require mpiexec to run.
# Run them separately with:
#   mpiexec -n 4 julia --project test/parallel/test_mpi_comprehensive.jl
#   mpiexec -n 4 julia --project test/parallel/test_mpi_extended.jl
