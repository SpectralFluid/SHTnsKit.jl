# SHTnsKit.jl - 2D spectral-plan alignment contracts (run with mpiexec -n 4)
#
# The optimized 2D routines reduce within `l_comm` (ranks sharing an `m_rank`)
# instead of over the whole communicator. That is only valid when the spatial θ
# split lines up with the spectral l split, so that every rank in such a group
# owns a DISTINCT θ slab. Where the precondition holds these paths agree with
# the safe ones to ~1e-14; where it does not they used to return silent garbage:
#
#   dist_synthesis_distributed_2d_optimized   max|err| = 84.4   (field of O(10))
#   dist_analysis_distributed_2d(assume_aligned=true)  rel err = 1.15
#
# Both now validate collectively and raise. This file pins both directions.
#
# (This is also why `_dist_analysis_2d_aligned` does NOT call
# `_keep_one_phi_partner!` the way its full-comm siblings do: within `l_comm`
# the alignment precondition already guarantees there are no duplicate slabs.)

using Test, MPI, PencilArrays, PencilFFTs, SHTnsKit, LinearAlgebra, Random

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NP   = MPI.Comm_size(COMM)
const EXT  = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)

NP == 4 || error("test_mpi_2d_alignment.jl expects 4 ranks, got $NP")

const LMAX = 8
const NLAT = LMAX + 2
const NLON = 2 * LMAX + 2
const CFG  = create_gauss_config(LMAX, NLAT; nlon=NLON)

Random.seed!(1234)
const FGLOB = randn(NLAT, NLON)
const AREF  = analysis(CFG, FGLOB)

const ALM = let a = zeros(ComplexF64, LMAX + 1, CFG.mmax + 1)
    for m in 0:CFG.mmax, l in m:LMAX
        a[l+1, m+1] = m == 0 ? Float64(l + 1) : ComplexF64(l + 1, m)
    end
    a
end
const FREF = synthesis(CFG, ALM)

"""Build a spatial PencilArray on a `pθ × pφ` process grid, filled from `FGLOB`."""
function spatial(pθ, pφ)
    dims = pφ == 1 ? (1,) : (pθ == 1 ? (2,) : (1, 2))
    pen = Pencil((NLAT, NLON), dims, COMM)
    f = PencilArray{Float64}(undef, pen)
    lr = PencilArrays.range_local(pen)
    loc = parent(f)
    for (jj, jg) in enumerate(lr[2]), (ii, ig) in enumerate(lr[1])
        loc[ii, jj] = FGLOB[ig, jg]
    end
    f
end

"""Largest spatial error against the serial reference, reduced over all ranks."""
function spatial_err(out, prototype)
    loc = out isa PencilArray ? parent(out) : out
    lr = PencilArrays.range_local(PencilArrays.pencil(prototype))
    e = 0.0
    for (jj, jg) in enumerate(lr[2]), (ii, ig) in enumerate(lr[1])
        e = max(e, abs(loc[ii, jj] - FREF[ig, jg]))
    end
    MPI.Allreduce(e, max, COMM)
end

@testset "2D plan alignment contracts (4 ranks)" begin
    for (pθ, pφ) in ((4, 1), (1, 4), (2, 2))
        f = spatial(pθ, pφ)
        plan = EXT.create_distributed_spectral_plan_2d(LMAX, CFG.mmax, COMM; p_l=pθ, p_m=pφ)
        try
            aligned, _ = EXT.validate_2d_distribution_alignment(plan, f)
            # The verdict must be identical on every rank, or the guards below
            # would throw on some ranks and enter a collective on others.
            @test MPI.Allreduce(aligned, &, COMM) == MPI.Allreduce(aligned, |, COMM)

            @testset "pθ=$pθ × pφ=$pφ (aligned=$aligned)" begin
                # The safe paths are correct on every decomposition.
                Asafe = EXT.dist_analysis_distributed_2d(CFG, f; plan=plan, assume_aligned=false)
                @test maximum(abs, EXT.gather_to_full_dense_2d(Asafe) .- AREF) <
                      1e-12 * maximum(abs, AREF)

                dsa = EXT.create_distributed_spectral_array_2d(plan)
                EXT.scatter_from_dense_2d!(dsa, ALM)
                @test spatial_err(EXT.dist_synthesis_distributed_2d(CFG, dsa; prototype_θφ=f), f) < 1e-11

                if aligned
                    # Optimized paths agree with the safe ones where the
                    # precondition holds.
                    Aal = EXT.dist_analysis_distributed_2d(CFG, f; plan=plan, assume_aligned=true)
                    @test maximum(abs, EXT.gather_to_full_dense_2d(Aal) .- AREF) <
                          1e-12 * maximum(abs, AREF)
                    @test spatial_err(
                        EXT.dist_synthesis_distributed_2d_optimized(CFG, dsa; prototype_θφ=f), f) < 1e-11
                else
                    # ...and raise, rather than returning garbage, where it does not.
                    @test_throws ArgumentError EXT.dist_analysis_distributed_2d(
                        CFG, f; plan=plan, assume_aligned=true)
                    @test_throws ArgumentError EXT.dist_synthesis_distributed_2d_optimized(
                        CFG, dsa; prototype_θφ=f)
                end
            end
        finally
            close(plan)
        end
    end
end

MPI.Barrier(COMM)
RANK == 0 && println("\nAll 2D alignment contract tests PASSED")
