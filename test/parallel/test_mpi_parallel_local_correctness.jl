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

    @testset "one-longitude latitude evaluations retain vector outputs" begin
        Q = zeros(ComplexF64, spectral_dims)
        S = copy(Q)
        T = copy(Q)
        Q[1, 1], Q[3, 2] = 0.8, 0.3 - 0.2im
        S[2, 1], T[4, 2] = 0.4, -0.1 + 0.3im
        Qp, Sp, Tp = map(A -> scatter_spectral(pen_m, A), (Q, S, T))
        Qpacked, Spacked, Tpacked = map(A -> SHTnsKit.pack_lm(cfg, A), (Q, S, T))
        cost = 0.2

        expected = SH_to_lat(cfg, Qpacked, cost; nphi=1)
        for actual in (SH_to_lat(cfg, Qp, cost; nphi=1),
                       SHTnsKit.dist_SH_to_lat(cfg, Qp, cost; nphi=1))
            @test actual isa Vector{Float64}
            @test size(actual) == (1,)
            @test actual ≈ expected
        end
        expected_qst = SHqst_to_lat(cfg, Qpacked, Spacked, Tpacked, cost; nphi=1)
        for actual in (SHqst_to_lat(cfg, Qp, Sp, Tp, cost; nphi=1),
                       SHTnsKit.dist_SHqst_to_lat(cfg, Qp, Sp, Tp, cost; nphi=1))
            for k in 1:3
                @test actual[k] isa Vector{Float64}
                @test size(actual[k]) == (1,)
                @test actual[k] ≈ expected_qst[k]
            end
        end
        @test synthesis_point(cfg, Qp, cost, 0.0) ≈ only(expected)
        actual_point = SHqst_to_point(cfg, Qp, Sp, Tp, cost, 0.0)
        @test all(isapprox.(actual_point, only.(expected_qst)))

        C = zeros(ComplexF64, SHTnsKit.nlm_cplx_calc(lmax, mmax, 1))
        C[SHTnsKit.LM_cplx_index(lmax, mmax, 0, 0) + 1] = 0.8 + 0.1im
        C[SHTnsKit.LM_cplx_index(lmax, mmax, 3, -2) + 1] = 0.3 - 0.2im
        Cpen = Pencil((length(C), 1), (1,), comm)
        Cp = scatter_spectral(Cpen, reshape(C, :, 1))
        actual_complex = SH_to_lat_cplx(cfg, Cp, cost; nphi=1)
        expected_complex = SH_to_lat_cplx(cfg, C, cost; nphi=1)
        @test actual_complex isa Vector{ComplexF64}
        @test size(actual_complex) == (1,)
        @test actual_complex ≈ expected_complex
        @test synthesis_point_cplx(cfg, Cp, cost, 0.0) ≈ only(expected_complex)
    end

    @testset "axisymmetric default latitude output remains a vector" begin
        axis_cfg = create_gauss_config(lmax, lmax + 2; mmax=0, nlon=1)
        A = zeros(ComplexF64, lmax + 1, 1)
        A[1, 1], A[3, 1] = 0.7, -0.2
        axis_pen = Pencil(size(A), (1,), comm)
        Ap = scatter_spectral(axis_pen, A)
        packed = SHTnsKit.pack_lm(axis_cfg, A)
        actual = SH_to_lat(axis_cfg, Ap, 0.2)
        @test actual isa Vector{Float64}
        @test actual ≈ SH_to_lat(axis_cfg, packed, 0.2)
        actual_qst = SHqst_to_lat(axis_cfg, Ap, Ap, Ap, 0.2)
        expected_qst = SHqst_to_lat(axis_cfg, packed, packed, packed, 0.2)
        for k in 1:3
            @test actual_qst[k] isa Vector{Float64}
            @test actual_qst[k] ≈ expected_qst[k]
        end
    end

    @testset "square parent blocks cannot conceal unsupported permutations" begin
        square_cfg = create_gauss_config(2, 6; nlon=6nprocs)
        A = zeros(ComplexF64, 3, 3)
        A[1, 1], A[3, 1], A[3, 2] = 1, 2, 0.5 + 0.2im
        field = synthesis(square_cfg, A)
        spatial_pen = Pencil(size(field), (2,), comm)
        permuted_pen = Pencil(spatial_pen; permute=Permutation(2, 1))
        ordinary = scatter_spectral(spatial_pen, field)
        permuted = PencilArray{Float64}(undef, permuted_pen)
        ranges = PencilArrays.range_local(permuted_pen)
        for (j, gj) in enumerate(ranges[2]), (i, gi) in enumerate(ranges[1])
            permuted[i, j] = field[gi, gj]
        end
        @test size(parent(ordinary)) == size(parent(permuted)) == (6, 6)
        # Reuse the same topology/communicator, with only rank 0 permuted, to
        # ensure the validation failure is collective before FFT/reduction work.
        rank_varying = rank == 0 ? permuted : ordinary
        for input in (permuted, rank_varying), use_rfft in (false, true)
            @test_throws ArgumentError analysis(square_cfg, input; use_rfft)
        end
        @test SHTnsKit.spectral_pencil_to_matrix(
            square_cfg, analysis(square_cfg, ordinary),
        ) ≈ A

        # Degree decomposition gives every rank a square 3x3 spectral block.
        # Point/latitude kernels must reject it too, even though shapes match.
        local_cfg = create_gauss_config(3nprocs - 1, 3nprocs + 1; mmax=2, nlon=5)
        local_pen = Pencil((local_cfg.lmax + 1, 3), (1,), comm)
        local_permuted_pen = Pencil(local_pen; permute=Permutation(2, 1))
        local_ordinary = PencilArray{ComplexF64}(undef, local_pen)
        local_permuted = PencilArray{ComplexF64}(undef, local_permuted_pen)
        fill!(parent(local_ordinary), 0)
        fill!(parent(local_permuted), 0)
        @test size(parent(local_ordinary)) == size(parent(local_permuted)) == (3, 3)
        local_rank_varying = rank == 0 ? local_permuted : local_ordinary
        for input in (local_permuted, local_rank_varying)
            @test_throws ArgumentError synthesis_point(local_cfg, input, 0.2, 0.4)
            @test_throws ArgumentError SH_to_lat(local_cfg, input, 0.2; nphi=1)
            @test_throws ArgumentError SHqst_to_point(
                local_cfg, input, input, input, 0.2, 0.4,
            )
        end

        batch_cfg = create_gauss_config(2, 6nprocs; nlon=6)
        batch_pen = Pencil((batch_cfg.nlat, batch_cfg.nlon), (1,), comm)
        batch_permuted_pen = Pencil(batch_pen; permute=Permutation(2, 1))
        batch_ordinary = PencilArray{Float64}(undef, batch_pen, 2)
        batch_permuted = PencilArray{Float64}(undef, batch_permuted_pen, 2)
        fill!(parent(batch_ordinary), 0)
        fill!(parent(batch_permuted), 0)
        @test size(parent(batch_ordinary)) == size(parent(batch_permuted)) == (6, 6, 2)
        batch_rank_varying = rank == 0 ? batch_permuted : batch_ordinary
        for input in (batch_permuted, batch_rank_varying)
            @test_throws ArgumentError analysis_sphtor_batch(batch_cfg, input, input)
            @test_throws ArgumentError analysis_qst_batch(batch_cfg, input, input, input)
        end
    end

    @testset "Robert analysis rejects lossy weighted-pole data collectively" begin
        robert_cfg = create_regular_config(2, 6; nlon=5, include_poles=true,
                                            robert_form=true)
        spatial_pen = Pencil((robert_cfg.nlat, robert_cfg.nlon), (1,), comm)
        field = PencilArray{Float64}(undef, spatial_pen)
        fill!(parent(field), 0)
        @test_throws ArgumentError analysis_sphtor(robert_cfg, field, field)
        @test_throws ArgumentError SHTnsKit.dist_analysis_sphtor(robert_cfg, field, field)
        @test_throws ArgumentError analysis_qst(robert_cfg, field, field, field)
        @test_throws ArgumentError analysis_sphtor_l(robert_cfg, field, field, 1)
        @test_throws ArgumentError analysis_qst_l(robert_cfg, field, field, field, 1)
        @test all(A -> all(iszero, parent(A)),
                  analysis_sphtor_l(robert_cfg, field, field, 0))

        ParExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        vector_plan = ParExt.DistSphtorPlan(robert_cfg, field)
        qst_plan = ParExt.DistQstPlan(robert_cfg, field)
        outputs = ntuple(_ -> fill(7.0 + 2.0im, 3, 3), 3)
        @test_throws ArgumentError SHTnsKit.dist_analysis_sphtor!(
            vector_plan, outputs[2], outputs[3], field, field,
        )
        @test_throws ArgumentError SHTnsKit.dist_analysis_qst!(
            qst_plan, outputs..., field, field, field,
        )
        @test all(A -> all(==(7.0 + 2.0im), A), outputs)

        mode_pen = Pencil((robert_cfg.nlat, 1), (1,), comm)
        mode = PencilArray{ComplexF64}(undef, mode_pen)
        fill!(parent(mode), 0)
        @test_throws ArgumentError analysis_sphtor_ml(robert_cfg, 1, mode, mode, 2)
        @test_throws ArgumentError analysis_qst_ml(robert_cfg, 1, mode, mode, mode, 2)
        for m in (0, 2)
            @test all(A -> all(iszero, parent(A)),
                      analysis_sphtor_ml(robert_cfg, m, mode, mode, 2))
        end

        # A divergent cfg must fail on every rank before the local Robert guard
        # can throw on only the rank whose configuration enables Robert form.
        if nprocs > 1
            divergent = create_regular_config(2, 6; nlon=5, include_poles=true,
                                               robert_form=(rank == 0))
            @test_throws ArgumentError analysis_sphtor(divergent, field, field)
        end
    end

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
        # Communicator preflight raises ArgumentError before layout validation;
        # DimensionMismatch is reserved here for the shape/layout cases above.
        pen_self = Pencil(spectral_dims, MPI.COMM_SELF)
        S_self = scatter_spectral(pen_self, zeros_global)
        @test_throws ArgumentError SHTnsKit.dist_SHqst_to_point(
            cfg, Q, S_self, Q, 0.2, 0.4)
        @test_throws ArgumentError SHTnsKit.dist_SHqst_to_lat(
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

    @testset "composite spatial operators keep the input communicator" begin
        spatial_dims = (cfg.nlat, cfg.nlon)
        Q = zeros(ComplexF64, spectral_dims)
        S = similar(Q); fill!(S, 0)
        T = similar(Q); fill!(T, 0)
        Q[3, 1], Q[4, 2] = 0.3, 0.4 - 0.2im
        S[2, 1], S[5, 3] = -0.2, 0.3 + 0.1im
        T[3, 1], T[4, 2] = 0.1, -0.2 + 0.4im
        scalar_values = synthesis(cfg, Q)
        theta_values, phi_values = synthesis_sphtor(cfg, S, T)
        degree_factors = [-l * (l + 1) for l in 0:lmax]
        input_pen = Pencil(spatial_dims, (1,), comm)
        input = scatter_spectral(input_pen, scalar_values)
        theta_input = scatter_spectral(input_pen, theta_values)

        duplicate_a = MPI.Comm_dup(comm)
        duplicate_b = MPI.Comm_dup(comm)
        try
            pen_a = Pencil(spatial_dims, (1,), duplicate_a)
            pen_b = Pencil(spatial_dims, (1,), duplicate_b)
            peer_a = scatter_spectral(pen_a, phi_values)
            peer_b = scatter_spectral(pen_b, phi_values)

            # Every candidate communicator is congruent to `comm`, but choosing
            # a different duplicate on each rank makes it unsafe as a collective
            # context. All composite stages must stay on `input`'s communicator.
            peer = iseven(rank) ? peer_a : peer_b
            for decomposition in ((1,), (2,))
                output_pen_a = Pencil(spatial_dims, decomposition, duplicate_a)
                output_pen_b = Pencil(spatial_dims, decomposition, duplicate_b)
                output_pen = iseven(rank) ? output_pen_b : output_pen_a
                ranges = PencilArrays.range_local(output_pen)
                for (use_rfft, real_output) in ((false, true), (true, true), (false, false))
                    @testset "output=$decomposition, rfft=$use_rfft, real=$real_output" begin
                        output = PencilArray{real_output ? Float64 : ComplexF64}(undef, output_pen)
                        fill!(parent(output), 0)
                        expected_divergence = synthesis(cfg, degree_factors .* S; real_output)[ranges...]
                        expected_vorticity = synthesis(cfg, degree_factors .* T; real_output)[ranges...]
                        expected_laplacian = synthesis(cfg, degree_factors .* Q; real_output)[ranges...]
                        @test SHTnsKit.dist_spatial_divergence(
                            cfg, theta_input, peer; prototype_θφ=output, use_rfft, real_output,
                        ) ≈ expected_divergence rtol=1e-11 atol=1e-12
                        @test SHTnsKit.dist_spatial_vorticity(
                            cfg, theta_input, peer; prototype_θφ=output, use_rfft, real_output,
                        ) ≈ expected_vorticity rtol=1e-11 atol=1e-12
                        @test SHTnsKit.dist_scalar_laplacian(
                            cfg, input; prototype_θφ=output, use_rfft, real_output,
                        ) ≈ expected_laplacian rtol=1e-11 atol=1e-12
                        @test SHTnsKit.dist_scalar_laplacian!(
                            cfg, output, input; use_rfft, real_output,
                        ) === output
                        @test parent(output) ≈ expected_laplacian rtol=1e-11 atol=1e-12
                    end
                end
            end

            self_pen = Pencil(spatial_dims, (1,), MPI.COMM_SELF)
            self_output = scatter_spectral(self_pen, zeros(Float64, spatial_dims))
            incongruent_output = rank == 0 ? self_output : peer_a
            output_before = copy(parent(incongruent_output))
            @test_throws ArgumentError SHTnsKit.dist_spatial_divergence(
                cfg, input, input; prototype_θφ=incongruent_output,
            )
            @test_throws ArgumentError SHTnsKit.dist_spatial_vorticity(
                cfg, input, input; prototype_θφ=incongruent_output,
            )
            @test_throws ArgumentError SHTnsKit.dist_scalar_laplacian(
                cfg, input; prototype_θφ=incongruent_output,
            )
            @test_throws ArgumentError SHTnsKit.dist_scalar_laplacian!(
                cfg, incongruent_output, input,
            )
            @test parent(incongruent_output) == output_before
        finally
            Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)._safe_comm_free(
                duplicate_a,
            )
            Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)._safe_comm_free(
                duplicate_b,
            )
        end
    end
end

rank == 0 && println("ParallelLocal correctness regression tests complete")
