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

"""Copy a globally replicated matrix into the block owned by `pen`."""
function scatter_spectral(pen::Pencil, A::AbstractMatrix)
    ranges = PencilArrays.range_local(pen)
    block = Array{eltype(A)}(undef, PencilArrays.size_local(pen))
    for (jm, gm) in enumerate(ranges[2]), (il, gl) in enumerate(ranges[1])
        block[il, jm] = A[gl, gm]
    end
    return PencilArray(pen, block)
end

@testset "parallel local-evaluation contracts ($nprocs ranks)" begin
    lmax = mmax = 6
    cfg = create_gauss_config(lmax, lmax + 2; mmax, nlon=2mmax + 1)
    spectral_dims = (lmax + 1, mmax + 1)
    pen_m = Pencil(spectral_dims, comm)

    @testset "local vector evaluations honor Robert form" begin
        Q = zeros(ComplexF64, spectral_dims)
        S = copy(Q)
        T = copy(Q)
        Q[1, 1] = 0.8
        S[2, 1] = 0.7
        S[3, 2] = -0.3 + 0.1im
        T[4, 2] = 0.25 - 0.2im
        S[5, 3] = 0.4 + 0.2im
        T[6, 3] = -0.2 + 0.6im
        Q_p, S_p, T_p = map(A -> scatter_spectral(pen_m, A), (Q, S, T))

        for grid_type in (:gauss, :regular_poles), robert_form in (false, true)
            local_cfg = create_config(lmax; mmax, nlat=lmax + 2,
                                      nlon=2mmax + 1, grid_type, robert_form,
                                      norm=:schmidt, real_norm=true, cs_phase=false)
            fields = synthesis_qst(local_cfg, Q, S, T)
            truncated_fields = synthesis_qst_l(local_cfg, Q, S, T, 3)
            for ilat in (1, 3, local_cfg.nlat)
                cost = local_cfg.x[ilat]
                point = SHTnsKit.dist_SHqst_to_point(
                    local_cfg, Q_p, S_p, T_p, cost, local_cfg.φ[2])
                lat = SHTnsKit.dist_SHqst_to_lat(local_cfg, Q_p, S_p, T_p, cost)
                truncated = SHTnsKit.dist_SHqst_to_lat(
                    local_cfg, Q_p, S_p, T_p, cost; ltr=3)
                for component in 1:3
                    @test point[component] ≈ fields[component][ilat, 2] rtol=1e-11 atol=1e-12
                    @test lat[component] ≈ fields[component][ilat, :] rtol=1e-11 atol=1e-12
                    @test truncated[component] ≈ truncated_fields[component][ilat, :] rtol=1e-11 atol=1e-12
                end
            end
        end
    end

    @testset "complex latitude evaluation is one-sided complex synthesis" begin
        A = zeros(ComplexF64, spectral_dims)
        A[5, 3] = 0.7 - 0.4im # (l,m) = (4,2), deliberately non-real
        A_p = scatter_spectral(pen_m, A)
        ilat = 3

        got = SHTnsKit.dist_SH_to_lat(
            cfg, A_p, cfg.x[ilat]; nphi=cfg.nlon, real_output=false)
        ref = vec(SHTnsKit.synthesis(cfg, A; real_output=false)[ilat, :])

        @test eltype(got) <: Complex
        @test isapprox(got, ref; rtol=1e-11, atol=1e-12)
        @test maximum(abs, imag.(got)) > 1e-4
    end

    @testset "configured global spectral dimensions are enforced" begin
        bad_dims = (lmax, mmax + 1)
        bad_pen = Pencil(bad_dims, comm)
        bad = scatter_spectral(bad_pen, zeros(ComplexF64, bad_dims))

        @test_throws DimensionMismatch SHTnsKit.dist_SH_to_point(cfg, bad, 0.2, 0.4)
        @test_throws DimensionMismatch SHTnsKit.dist_SH_to_lat(cfg, bad, 0.2)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_point(
            cfg, bad, bad, bad, 0.2, 0.4)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_lat(
            cfg, bad, bad, bad, 0.2)
    end

    @testset "Q/S/T layouts and communicator groups must match" begin
        zeros_global = zeros(ComplexF64, spectral_dims)
        Q = scatter_spectral(pen_m, zeros_global)

        # Same global dimensions and communicator, but a different distributed
        # logical dimension, so the local ranges do not describe the same modes.
        pen_l = Pencil(spectral_dims, (1,), comm)
        S_l = scatter_spectral(pen_l, zeros_global)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_point(
            cfg, Q, S_l, Q, 0.2, 0.4)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_lat(
            cfg, Q, S_l, Q, 0.2)

        # Logical ownership matches, but memory order does not. Mixing these
        # arrays in one component-wise kernel is rejected explicitly.
        pen_perm = Pencil(spectral_dims, comm; permute=Permutation(2, 1))
        S_perm = PencilArray{ComplexF64}(undef, pen_perm)
        fill!(parent(S_perm), 0)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_point(
            cfg, Q, S_perm, Q, 0.2, 0.4)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_lat(
            cfg, Q, S_perm, Q, 0.2)

        # COMM_SELF has a different group even though the global dimensions
        # agree. The reduction communicator must be shared by every component.
        pen_self = Pencil(spectral_dims, MPI.COMM_SELF)
        S_self = scatter_spectral(pen_self, zeros_global)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_point(
            cfg, Q, S_self, Q, 0.2, 0.4)
        @test_throws DimensionMismatch SHTnsKit.dist_SHqst_to_lat(
            cfg, Q, S_self, Q, 0.2)
    end

    @testset "latitude truncation validation matches serial helpers" begin
        Z = scatter_spectral(pen_m, zeros(ComplexF64, spectral_dims))
        for bad_ltr in (-1, lmax + 1)
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(cfg, Z, 0.2; ltr=bad_ltr)
            @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
                cfg, Z, Z, Z, 0.2; ltr=bad_ltr)
        end
        for bad_mtr in (-1, mmax + 1)
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(cfg, Z, 0.2; mtr=bad_mtr)
            @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
                cfg, Z, Z, Z, 0.2; mtr=bad_mtr)
        end
    end

    @testset "collective evaluation arguments are validated" begin
        Z = scatter_spectral(pen_m, zeros(ComplexF64, spectral_dims))

        # A zero-length longitude vector is not a meaningful latitude sweep.
        @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(cfg, Z, 0.2; nphi=0)
        @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
            cfg, Z, Z, Z, 0.2; nphi=0)

        # These routines reduce partial modal sums across ranks, so every rank
        # must evaluate the same function with the same configuration.
        cfg_divergent = create_gauss_config(
            lmax, lmax + 2; mmax, nlon=2mmax + 1,
            mres=(rank == nprocs - 1 ? 2 : 1),
        )
        @test_throws ArgumentError SHTnsKit.dist_SH_to_point(
            cfg_divergent, Z, 0.2, 0.4)

        if nprocs > 1
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(
                cfg, Z, 0.2;
                nphi=(rank == nprocs - 1 ? cfg.nlon - 1 : cfg.nlon),
            )
            @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
                cfg, Z, Z, Z, 0.2;
                nphi=(rank == nprocs - 1 ? cfg.nlon - 1 : cfg.nlon),
            )
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(
                cfg, Z, 0.2;
                real_output=(rank != nprocs - 1),
            )
            @test_throws ArgumentError SHTnsKit.dist_SH_to_lat(
                cfg, Z, 0.2;
                ltr=(rank == nprocs - 1 ? lmax - 1 : lmax),
            )
            @test_throws ArgumentError SHTnsKit.dist_SHqst_to_point(
                cfg, Z, Z, Z, rank == nprocs - 1 ? 0.3 : 0.2, 0.4)
        end
    end
end

rank == 0 && println("ParallelLocal correctness regression tests complete")
