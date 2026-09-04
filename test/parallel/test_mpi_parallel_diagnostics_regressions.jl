#!/usr/bin/env julia

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

function distribute(A::AbstractMatrix, pen::Pencil)
    Ap = PencilArray{eltype(A)}(undef, pen)
    gl_l, gl_m = PencilArrays.range_local(pen)
    for (ii, il) in enumerate(axes(Ap, 1)), (jj, jm) in enumerate(axes(Ap, 2))
        Ap[il, jm] = A[gl_l[ii], gl_m[jj]]
    end
    return Ap
end

function active_only(cfg, A)
    B = copy(A)
    for m in 0:cfg.mmax
        m % cfg.mres == 0 || fill!(@view(B[:, m + 1]), zero(eltype(B)))
    end
    return B
end

try
    @testset "MPI PencilArray diagnostics regressions" begin
        cfg = create_gauss_config(6, 8; mmax=6, mres=2, nlon=14)
        dims = (cfg.lmax + 1, cfg.mmax + 1)
        pen = create_spectral_pencil(cfg; comm)

        # Seed every dense column.  The odd-m columns are deliberately large:
        # an mres=2 diagnostic must ignore these non-representable orders.
        S = Matrix{ComplexF32}(undef, dims)
        T = similar(S)
        Q = similar(S)
        for m in 0:cfg.mmax, l in 0:cfg.lmax
            represented = m % cfg.mres == 0 && l >= m
            scale = represented ? 1.0f0 : 100.0f0
            S[l + 1, m + 1] = scale * ComplexF32(l + 1, m + 1)
            T[l + 1, m + 1] = scale * ComplexF32(2l + 1, -(m + 1))
            Q[l + 1, m + 1] = scale * ComplexF32(l - m, l + m + 1)
        end
        S0, T0, Q0 = active_only(cfg, S), active_only(cfg, T), active_only(cfg, Q)
        Sp, Tp, Qp = distribute(S, pen), distribute(T, pen), distribute(Q, pen)

        @testset "totals and mres" begin
            @test energy_scalar(cfg, Qp) ≈ energy_scalar(cfg, Q)
            @test energy_vector(cfg, Sp, Tp) ≈ energy_vector(cfg, S, T)
            @test enstrophy(cfg, Tp) ≈ enstrophy(cfg, T)
        end

        @testset "spectra values and result types" begin
            Ql = energy_scalar_l_spectrum(cfg, Qp)
            Qm = energy_scalar_m_spectrum(cfg, Qp)
            Vl = energy_vector_l_spectrum(cfg, Sp, Tp)
            Vm = energy_vector_m_spectrum(cfg, Sp, Tp)
            Zl = enstrophy_l_spectrum(cfg, Tp)
            Zm = enstrophy_m_spectrum(cfg, Tp)

            @test Ql ≈ energy_scalar_l_spectrum(cfg, Q)
            @test Qm ≈ energy_scalar_m_spectrum(cfg, Q)
            @test Vl ≈ energy_vector_l_spectrum(cfg, S, T)
            @test Vm ≈ energy_vector_m_spectrum(cfg, S, T)
            @test Zl ≈ enstrophy_l_spectrum(cfg, T)
            @test Zm ≈ enstrophy_m_spectrum(cfg, T)

            @test eltype(Ql) === eltype(energy_scalar_l_spectrum(cfg, Q0))
            @test eltype(Qm) === eltype(energy_scalar_m_spectrum(cfg, Q0))
            @test eltype(Vl) === eltype(energy_vector_l_spectrum(cfg, S0, T0))
            @test eltype(Vm) === eltype(energy_vector_m_spectrum(cfg, S0, T0))
            @test eltype(Zl) === eltype(enstrophy_l_spectrum(cfg, T0))
            @test eltype(Zm) === eltype(enstrophy_m_spectrum(cfg, T0))

            for m in 0:cfg.mmax
                m % cfg.mres == 0 && continue
                @test Qm[m + 1] == 0
                @test Vm[m + 1] == 0
                @test Zm[m + 1] == 0
            end
        end

        @testset "layout validation happens before paired indexing" begin
            # The second array has a different global shape but at least as much
            # local storage, so the unchecked implementation silently computed a
            # partial result instead of naturally throwing a bounds error.
            bad_shape_pen = Pencil((cfg.lmax + 1, cfg.mmax + 2), comm)
            T_bad_shape = PencilArray{ComplexF32}(undef, bad_shape_pen)
            fill!(T_bad_shape, 0)
            @test_throws ArgumentError energy_vector(cfg, Sp, T_bad_shape)

            # COMM_SELF gives every rank a full, safely indexable second operand.
            # It must still be rejected before reducing on Sp's communicator.
            self_pen = Pencil(dims, MPI.COMM_SELF)
            T_self = PencilArray{ComplexF32}(undef, self_pen)
            fill!(T_self, 0)
            @test_throws ArgumentError energy_vector(cfg, Sp, T_self)

            # A latitude/degree split has a communicator congruent with the
            # normal order-split pencil, but owns different logical ranges.
            # Congruence alone is therefore not enough for paired indexing.
            l_pen = Pencil(dims, (1,), comm)
            T_l_split = PencilArray{ComplexF32}(undef, l_pen)
            fill!(T_l_split, 0)
            @test MPI.Comm_compare(PencilArrays.get_comm(Sp),
                                   PencilArrays.get_comm(T_l_split)) == MPI.CONGRUENT
            @test any(dim -> PencilArrays.range_local(pencil(Sp))[dim] !=
                             PencilArrays.range_local(pencil(T_l_split))[dim], 1:2)
            @test_throws ArgumentError energy_vector(cfg, Sp, T_l_split)

            perm_pen = Pencil(dims, comm; permute=Permutation(2, 1))
            S_perm = PencilArray{ComplexF32}(undef, perm_pen)
            T_perm = PencilArray{ComplexF32}(undef, perm_pen)
            fill!(S_perm, 0)
            fill!(T_perm, 0)
            @test_throws ArgumentError energy_vector(cfg, S_perm, T_perm)

            grid_pen = Pencil((cfg.nlat, cfg.nlon), comm)
            Vt = PencilArray{Float32}(undef, grid_pen)
            fill!(Vt, 0)
            bad_grid_pen = Pencil((cfg.nlat, cfg.nlon + 1), comm)
            Vp_bad = PencilArray{Float32}(undef, bad_grid_pen)
            fill!(Vp_bad, 0)
            @test_throws ArgumentError grid_energy_vector(cfg, Vt, Vp_bad)

            cfg_divergent = create_gauss_config(
                cfg.lmax, cfg.nlat; mmax=cfg.mmax, nlon=cfg.nlon,
                mres=(rank == nprocs - 1 ? 1 : 2),
            )
            @test_throws ArgumentError energy_scalar(cfg_divergent, Qp)
            if nprocs > 1
                @test_throws ArgumentError energy_scalar(
                    cfg, Qp; real_field=(rank != nprocs - 1))

                # A result buffer's reduction datatype is inferred from the
                # local spectral eltype. Divergent eltypes must be rejected
                # before a public diagnostic enters MPI.Allreduce[!].
                mixed_T = rank == nprocs - 1 ? ComplexF64 : ComplexF32
                mixed = PencilArray{mixed_T}(undef, pen)
                fill!(parent(mixed), zero(mixed_T))
                @test_throws ArgumentError ParExt._require_diagnostic_array(
                    cfg, mixed, :spectral, "mixed-eltype spectrum")
            end
        end
    end
finally
    MPI.Barrier(comm)
    rank == 0 && println("MPI parallel diagnostics regression test finished")
    MPI.Finalize()
end
