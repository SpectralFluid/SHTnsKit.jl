#!/usr/bin/env julia

# Collective operand validation for DistTransposePlan execution.
# Run with:
#   mpiexec -n 2 julia --project test/parallel/test_mpi_transpose_operand_preflight.jl

using MPI
MPI.Init()

using PencilArrays
using PencilFFTs
using SHTnsKit
using Test

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

"""Return true only when every rank rejects the call before continuing."""
function rejected_collectively(f)
    rejected = false
    try
        f()
    catch err
        rejected = err isa ArgumentError || err isa DimensionMismatch
    end
    return MPI.Allreduce(rejected ? 1 : 0, +, comm) == nprocs
end

all_ranks(predicate::Bool) = MPI.Allreduce(predicate, &, comm)

function equivalent_spectral(plan; T=ComplexF64, nlev=plan.nlev, comm_=comm)
    nbin = plan.nlon ÷ 2 + 1
    pen = Pencil((plan.lmax + 1, nbin), (2,), comm_)
    A = PencilArray{T}(undef, pen, nlev)
    fill!(parent(A), zero(T))
    return A
end

function equivalent_spatial(plan; T=Float64, nlev=plan.nlev, comm_=comm)
    pen = Pencil((plan.nlon, plan.nlat), (2,), comm_)
    A = PencilArray{T}(undef, pen, nlev)
    fill!(parent(A), zero(T))
    return A
end

function rank_asymmetric_nlev_spectral(plan)
    A = rank == 0 ?
        PencilArray{ComplexF64}(undef, plan.spectral_pencil, plan.nlev + 1) :
        allocate_spectral(plan)
    fill!(parent(A), 0)
    return A
end

try
    @testset "DistTransposePlan operand preflight ($nprocs ranks)" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        plan = DistTransposePlan(
            cfg; comm, nlev=2, use_rfft=true, with_vector=true)

        spatial() = (A = allocate_spatial(plan); fill!(parent(A), 0); A)
        spectral() = (A = allocate_spectral(plan); fill!(parent(A), 0); A)

        @testset "scalar roles use exact plan layout, communicator, type, and nlev" begin
            f = spatial()

            # Only rank zero is malformed. The verdict still has to reject on
            # every rank before the forward FFT collective.
            A_bad_nlev = rank_asymmetric_nlev_spectral(plan)
            @test rejected_collectively() do
                dist_analysis!(plan, A_bad_nlev, f)
            end

            # An independently constructed Pencil can have identical ranges but
            # is not accepted by PencilFFTs as this plan's input/output layout.
            A_equiv = equivalent_spectral(plan)
            @test rejected_collectively() do
                dist_analysis!(plan, A_equiv, f)
            end

            # A COMM_SELF output is safely indexable but does not belong to the
            # plan communicator. Make the mismatch rank-asymmetric to exercise
            # the collective verdict rather than a uniform local throw.
            A_bad_comm = rank == 0 ? equivalent_spectral(plan; comm_=MPI.COMM_SELF) : spectral()
            @test rejected_collectively() do
                dist_analysis!(plan, A_bad_comm, f)
            end

            A_bad_type = PencilArray{ComplexF32}(
                undef, plan.spectral_pencil, plan.nlev)
            fill!(parent(A_bad_type), 0)
            @test rejected_collectively() do
                dist_analysis!(plan, A_bad_type, f)
            end

            f_out = spatial()
            A_in_bad_nlev = rank_asymmetric_nlev_spectral(plan)
            @test rejected_collectively() do
                dist_synthesis!(plan, f_out, A_in_bad_nlev)
            end

            A_bad_type_in = PencilArray{ComplexF32}(
                undef, plan.spectral_pencil, plan.nlev)
            fill!(parent(A_bad_type_in), 0)
            @test rejected_collectively() do
                dist_synthesis!(plan, f_out, A_bad_type_in)
            end

            # The old path filled the Fourier buffer before PencilFFTs rejected
            # the independently constructed spatial output pencil.
            f_equiv = equivalent_spatial(plan)
            sentinel = 17.0 - 9.0im
            fill!(parent(plan.F_buf), sentinel)
            @test rejected_collectively() do
                dist_synthesis!(plan, f_equiv, spectral())
            end
            @test all_ranks(all(==(sentinel), parent(plan.F_buf)))
        end

        @testset "vector roles are validated together before either FFT" begin
            Vt, Vp = spatial(), spatial()
            S, T = spectral(), spectral()

            T_bad_nlev = rank_asymmetric_nlev_spectral(plan)
            @test rejected_collectively() do
                dist_analysis_sphtor!(plan, S, T_bad_nlev, Vt, Vp)
            end
            @test rejected_collectively() do
                dist_synthesis_sphtor!(plan, Vt, Vp, S, T_bad_nlev)
            end

            T_bad_type = PencilArray{ComplexF32}(
                undef, plan.spectral_pencil, plan.nlev)
            fill!(parent(T_bad_type), 0)
            @test rejected_collectively() do
                dist_analysis_sphtor!(plan, S, T_bad_type, Vt, Vp)
            end

            # Vp is malformed, but Vt is valid. Validation must happen before
            # the first component FFT mutates F_buf.
            Vp_equiv = equivalent_spatial(plan)
            sentinel = 21.0 + 4.0im
            fill!(parent(plan.F_buf), sentinel)
            @test rejected_collectively() do
                dist_analysis_sphtor!(plan, S, T, Vt, Vp_equiv)
            end
            @test all_ranks(all(==(sentinel), parent(plan.F_buf)))

            # Likewise synthesis must not update Vt before discovering that Vp
            # is not the plan's spatial pencil.
            Vt_out = spatial()
            fill!(parent(Vt_out), 33.0)
            @test rejected_collectively() do
                dist_synthesis_sphtor!(plan, Vt_out, Vp_equiv, S, T)
            end
            @test all_ranks(all(==(33.0), parent(Vt_out)))
        end

        @testset "QST validates all six operands before scalar work" begin
            Vr, Vt, Vp = spatial(), spatial(), spatial()
            Q, S, T = spectral(), spectral(), spectral()

            # A late T operand must be noticed before scalar Q analysis starts.
            T_bad_nlev = rank_asymmetric_nlev_spectral(plan)
            fill!(parent(Q), 41.0 + 2.0im)
            @test rejected_collectively() do
                dist_analysis_qst!(plan, Q, S, T_bad_nlev, Vr, Vt, Vp)
            end
            @test all_ranks(all(==(41.0 + 2.0im), parent(Q)))

            # A late Vp output must likewise be noticed before Vr is synthesized.
            Vp_equiv = equivalent_spatial(plan)
            fill!(parent(Vr), 57.0)
            @test rejected_collectively() do
                dist_synthesis_qst!(plan, Vr, Vt, Vp_equiv, Q, S, T)
            end
            @test all_ranks(all(==(57.0), parent(Vr)))
        end

        @testset "cached configuration mutations are rejected" begin
            original_mres = cfg.mres
            cfg.mres = 2
            try
                @test rejected_collectively() do
                    dist_analysis!(plan, spectral(), spatial())
                end
            finally
                cfg.mres = original_mres
            end

            original_node = cfg.x[1]
            cfg.x[1] = nextfloat(original_node)
            try
                @test rejected_collectively() do
                    dist_analysis!(plan, spectral(), spatial())
                end
            finally
                cfg.x[1] = original_node
            end
        end
    end
finally
    MPI.Barrier(comm)
    rank == 0 && println("transpose operand-preflight regression complete")
    MPI.Finalize()
end
