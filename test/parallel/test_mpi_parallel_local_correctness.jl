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

"""Distribute a replicated vector as a `(n, 1)` PencilArray split along `n`."""
scatter_column(v::AbstractVector) =
    scatter_spectral(Pencil((length(v), 1), (1,), comm), reshape(v, :, 1))

"""Collect a `(n, 1)` PencilArray split along `n` on every rank."""
function gather_column(x::PencilArray)
    full = zeros(eltype(x), PencilArrays.size_global(x)[1])
    full[PencilArrays.range_local(PencilArrays.pencil(x))[1]] = parent(x)[:, 1]
    return MPI.Allreduce(full, +, comm)
end

@testset "parallel local-evaluation contracts ($nprocs ranks)" begin
    lmax = mmax = 6
    cfg = create_gauss_config(lmax, lmax + 2; mmax, nlon=2mmax + 1)
    spectral_dims = (lmax + 1, mmax + 1)
    pen_m = Pencil(spectral_dims, comm)

    @testset "one-longitude latitude outputs ($phi_scale)" for phi_scale in (:dft, :quad)
        withenv("SHTNSKIT_PHI_SCALE" => nothing) do  # the variable overrides cfg.phi_scale
            eval_cfg = deepcopy(cfg)
            eval_cfg.phi_scale = phi_scale
            @test SHTnsKit.phi_inv_scale(eval_cfg) ≈ (phi_scale === :quad ? eval_cfg.nlon / 2π : eval_cfg.nlon)
            Q = zeros(ComplexF64, spectral_dims)
            S = copy(Q)
            T = copy(Q)
            Q[1, 1], Q[3, 2] = 0.8, 0.3 - 0.2im
            S[2, 1], T[4, 2] = 0.4, -0.1 + 0.3im
            Qp, Sp, Tp = map(A -> scatter_spectral(pen_m, A), (Q, S, T))
            Qpacked, Spacked, Tpacked = map(A -> SHTnsKit.pack_lm(eval_cfg, A), (Q, S, T))
            cost = 0.2

            expected = SH_to_lat(eval_cfg, Qpacked, cost; nphi=1)
            for actual in (SH_to_lat(eval_cfg, Qp, cost; nphi=1),
                           SHTnsKit.dist_SH_to_lat(eval_cfg, Qp, cost; nphi=1))
                @test actual isa Vector{Float64}
                @test size(actual) == (1,)
                @test actual ≈ expected
            end
            expected_qst = SHqst_to_lat(eval_cfg, Qpacked, Spacked, Tpacked, cost; nphi=1)
            for actual in (SHqst_to_lat(eval_cfg, Qp, Sp, Tp, cost; nphi=1),
                           SHTnsKit.dist_SHqst_to_lat(eval_cfg, Qp, Sp, Tp, cost; nphi=1))
                for k in 1:3
                    @test actual[k] isa Vector{Float64}
                    @test size(actual[k]) == (1,)
                    @test actual[k] ≈ expected_qst[k]
                end
            end
            @test synthesis_point(eval_cfg, Qp, cost, 0.0) ≈ only(expected)
            actual_point = SHqst_to_point(eval_cfg, Qp, Sp, Tp, cost, 0.0)
            @test all(isapprox.(actual_point, only.(expected_qst)))

            C = zeros(ComplexF64, SHTnsKit.nlm_cplx_calc(lmax, mmax, 1))
            C[SHTnsKit.LM_cplx_index(lmax, mmax, 0, 0) + 1] = 0.8 + 0.1im
            C[SHTnsKit.LM_cplx_index(lmax, mmax, 3, -2) + 1] = 0.3 - 0.2im
            Cpen = Pencil((length(C), 1), (1,), comm)
            Cp = scatter_spectral(Cpen, reshape(C, :, 1))
            actual_complex = SH_to_lat_cplx(eval_cfg, Cp, cost; nphi=1)
            expected_complex = SH_to_lat_cplx(eval_cfg, C, cost; nphi=1)
            @test actual_complex isa Vector{ComplexF64}
            @test size(actual_complex) == (1,)
            @test actual_complex ≈ expected_complex
            @test synthesis_point_cplx(eval_cfg, Cp, cost, 0.0) ≈ only(expected_complex)
            @test SH_to_lat_cplx(eval_cfg, Cp, cost) ≈ SH_to_lat_cplx(eval_cfg, C, cost)
        end
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

    @testset "fixed-order vector transforms require a dimension-1 split" begin
        m, ltr = 1, lmax
        active = ltr - m + 1
        Vt = ComplexF64[0.3 + 0.1k + 0.05im * k^2 for k in 1:cfg.nlat]
        Vp = ComplexF64[-0.2 + 0.07k - 0.1im * k for k in 1:cfg.nlat]
        S, T = analysis_sphtor_ml(cfg, m, Vt, Vp, ltr)
        Sd, Td = analysis_sphtor_ml(cfg, m, scatter_column(Vt), scatter_column(Vp), ltr)
        @test gather_column(Sd) ≈ S
        @test gather_column(Td) ≈ T
        Vtd, Vpd = synthesis_sphtor_ml(cfg, m, scatter_column(S), scatter_column(T), ltr)
        @test all(map(≈, map(gather_column, (Vtd, Vpd)),
                      synthesis_sphtor_ml(cfg, m, S, T, ltr)))

        # Splitting the singleton column leaves ranks without the column the
        # kernels read, and a permutation reorders parent storage; both must
        # be rejected on every rank before the per-root reductions start.
        for make in (n -> Pencil((n, 1), (2,), comm),
                     n -> Pencil((n, 1), (1,), comm; permute=Permutation(2, 1)))
            field = PencilArray{ComplexF64}(undef, make(cfg.nlat))
            coefficients = PencilArray{ComplexF64}(undef, make(active))
            fill!(parent(field), 0)
            fill!(parent(coefficients), 0)
            @test_throws ArgumentError analysis_sphtor_ml(cfg, m, field, field, ltr)
            @test_throws ArgumentError analysis_qst_ml(cfg, m, field, field, field, ltr)
            @test_throws ArgumentError synthesis_sphtor_ml(
                cfg, m, coefficients, coefficients, ltr,
            )
            @test_throws ArgumentError synthesis_qst_ml(
                cfg, m, coefficients, coefficients, coefficients, ltr,
            )
        end
    end

    @testset "distributed axisymmetric transforms follow $phi_scale" for phi_scale in (:dft, :quad)
        withenv("SHTNSKIT_PHI_SCALE" => nothing) do  # the variable overrides cfg.phi_scale
            # Distributed synthesis_axisym used a unit φ factor, so under :quad it
            # was 2π larger than the serial m = 0 column it mirrors.
            axis_cfg = deepcopy(cfg)
            axis_cfg.phi_scale = phi_scale
            @test SHTnsKit.phi_inv_scale(axis_cfg) ≈ (phi_scale === :quad ? axis_cfg.nlon / 2π : axis_cfg.nlon)
            coefficients = ComplexF64[0.4, -0.2, 0.1, 0.03, -0.02, 0.01, 0.005]
            field = synthesis_axisym(axis_cfg, coefficients)
            @test gather_column(synthesis_axisym(axis_cfg, scatter_column(coefficients))) ≈ field
            @test gather_column(synthesis_axisym_l(axis_cfg, scatter_column(coefficients), 3)) ≈
                  synthesis_axisym_l(axis_cfg, coefficients, 3)
            @test gather_column(analysis_axisym(axis_cfg, scatter_column(field))) ≈ coefficients
        end
    end

    @testset "distributed calls accept what serial accepts" begin
        ParExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        spatial_pen = Pencil((cfg.nlat, cfg.nlon), (1,), comm)
        F = [sin(0.3i + 0.7j) for i in 1:cfg.nlat, j in 1:cfg.nlon]
        field = scatter_spectral(spatial_pen, F)

        # Real coefficient matrices, as for serial `synthesis`.
        real_alm = zeros(spectral_dims)
        real_alm[3, 1] = 0.5
        real_alm[4, 2] = -0.25
        expected = synthesis(cfg, real_alm)
        local_rows = PencilArrays.range_local(spatial_pen)
        @test SHTnsKit.dist_synthesis(cfg, real_alm; prototype_θφ=field) ≈
              expected[local_rows[1], local_rows[2]]

        # Any real angle, as for serial rotations: π, integers, rationals.
        Q = zeros(ComplexF64, spectral_dims)
        Q[3, 2] = 0.4 - 0.2im
        Q[5, 4] = 0.1 + 0.3im
        coefficients = scatter_spectral(pen_m, Q)
        for (exact, approx) in ((π, Float64(π)), (1, 1.0), (1//2, 0.5))
            by_real = SHTnsKit.dist_SH_rotate_euler(cfg, coefficients, exact, exact, exact,
                                                    similar(coefficients))
            by_float = SHTnsKit.dist_SH_rotate_euler(cfg, coefficients, approx, approx, approx,
                                                     similar(coefficients))
            @test parent(by_real) ≈ parent(by_float)
        end

        # The in-place Laplacian checks its destination before any work.
        mismatched = PencilArray{Float32}(undef, spatial_pen)
        fill!(parent(mismatched), 7)
        @test_throws ArgumentError SHTnsKit.dist_scalar_laplacian!(cfg, mismatched, field)
        @test all(==(7), parent(mismatched))

        # Float32 fields keep their precision in the 1D distributed analysis.
        plan = ParExt.create_distributed_spectral_plan(cfg.lmax, cfg.mmax, comm)
        coefficients32 = ParExt.dist_analysis_distributed(
            cfg, scatter_spectral(spatial_pen, Float32.(F)); plan,
        )
        @test eltype(coefficients32.local_coeffs) === ComplexF32
        @test ParExt.gather_to_dense(coefficients32) ≈ analysis(cfg, F) rtol=1e-5
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

    @testset "one-sided complex latitude synthesis ($phi_scale)" for phi_scale in (:dft, :quad)
        withenv("SHTNSKIT_PHI_SCALE" => nothing) do  # the variable overrides cfg.phi_scale
            eval_cfg = deepcopy(cfg)
            eval_cfg.phi_scale = phi_scale
            @test SHTnsKit.phi_inv_scale(eval_cfg) ≈ (phi_scale === :quad ? eval_cfg.nlon / 2π : eval_cfg.nlon)
            A = zeros(ComplexF64, spectral_dims)
            A[5, 3] = 0.7 - 0.4im # (l,m) = (4,2), deliberately non-real
            A_p = scatter_spectral(pen_m, A)
            ilat = 3

            got = SHTnsKit.dist_SH_to_lat(
                eval_cfg, A_p, eval_cfg.x[ilat]; nphi=eval_cfg.nlon, real_output=false)
            ref = vec(SHTnsKit.synthesis(eval_cfg, A; real_output=false)[ilat, :])

            @test eltype(got) <: Complex
            @test isapprox(got, ref; rtol=1e-11, atol=1e-12)
            @test maximum(abs, imag.(got)) > 1e-4
            @test SHTnsKit.dist_SH_to_lat(
                eval_cfg, A_p, eval_cfg.x[ilat]; nphi=1, real_output=false) ≈ ref[1:1]
        end
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

@testset "vector batches follow the batch's own layout ($nprocs ranks)" begin
    # Batch fields used to be copied linearly into the default θ-split
    # (spatial) or m-split (spectral) pencil whatever the batch decomposition,
    # which scrambled them: PencilArrays' own default 3-D layout splits φ.
    cfg = create_gauss_config(8, 9; nlon=17)       # odd sizes: uneven blocks
    nf = 2
    Vr = [sin(0.37i + 0.71j + 1.3k) for i in 1:cfg.nlat, j in 1:cfg.nlon, k in 1:nf]
    Vt = [cos(0.53i - 0.29j + 0.9k) for i in 1:cfg.nlat, j in 1:cfg.nlon, k in 1:nf]
    Vp = [sin(0.11i + 0.43j - 0.7k) + 0.3 for i in 1:cfg.nlat, j in 1:cfg.nlon, k in 1:nf]
    Qs, Ss, Ts = analysis_qst_batch(cfg, Vr, Vt, Vp)
    Rr, Rt, Rp = synthesis_qst_batch(cfg, Qs, Ss, Ts)

    # A 3-D pencil, or a 2-D pencil carrying the batch as an extra dimension.
    function place(pen, A)
        r = PencilArrays.range_local(pen)
        P = ndims(pen) == 3 ? PencilArray{eltype(A)}(undef, pen) :
                              PencilArray{eltype(A)}(undef, pen, size(A, 3))
        parent(P) .= ndims(pen) == 3 ? A[r...] : A[r[1], r[2], :]
        return P
    end
    function gathered(P)
        G = zeros(eltype(P), PencilArrays.size_global(P)...)
        r = PencilArrays.range_local(pencil(P))
        ndims(pencil(P)) == 3 ? (G[r...] .= parent(P)) : (G[r[1], r[2], :] .= parent(P))
        return MPI.Allreduce!(G, +, comm)
    end
    batch_local(pen) = ndims(pen) == 2 || PencilArrays.size_local(pen)[3] == nf

    layouts = ("PencilArrays default" => dims -> Pencil(dims, comm),
               "θ/l split" => dims -> Pencil(dims, (1,), comm),
               "φ/m split" => dims -> Pencil(dims, (2,), comm),
               "2-D pencil, batch as extra dimension" =>
                   dims -> Pencil(dims[1:2], (2,), comm))
    for (name, layout) in layouts
        @testset "$name" begin
            pen = layout((cfg.nlat, cfg.nlon, nf))
            if batch_local(pen)
                Q, S, T = analysis_qst_batch(cfg, place(pen, Vr), place(pen, Vt),
                                             place(pen, Vp))
                @test isapprox(gathered(Q), Qs; rtol=1e-12, atol=1e-13)
                @test isapprox(gathered(S), Ss; rtol=1e-12, atol=1e-13)
                @test isapprox(gathered(T), Ts; rtol=1e-12, atol=1e-13)
            end

            pen = layout((cfg.lmax + 1, cfg.mmax + 1, nf))
            if batch_local(pen)
                inputs = (place(pen, Qs), place(pen, Ss), place(pen, Ts))
                splits_l = PencilArrays.size_local(pen)[1] != cfg.lmax + 1
                if MPI.Allreduce(splits_l, |, comm)
                    # Distributed synthesis needs every degree on each rank.
                    @test_throws ArgumentError synthesis_qst_batch(cfg, inputs...)
                else
                    a, b, c = synthesis_qst_batch(cfg, inputs...)
                    @test isapprox(gathered(a), Rr; rtol=1e-12, atol=1e-13)
                    @test isapprox(gathered(b), Rt; rtol=1e-12, atol=1e-13)
                    @test isapprox(gathered(c), Rp; rtol=1e-12, atol=1e-13)
                end
            end
        end
    end
end

rank == 0 && println("ParallelLocal correctness regression tests complete")
