#!/usr/bin/env julia
#
# MPI regressions for distributed-plan/configuration preflight validation.
# Run with:
#   mpiexec -n 2 julia --project test/parallel/test_mpi_plan_preflight.jl

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

function zero_spatial(cfg, comm_=comm)
    pen = Pencil((cfg.nlat, cfg.nlon), (1,), comm_)
    return PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))
end

# Pin as the vendor adapters do, then keep only a reshaped wrapper, as the GPU
# transpose host mirrors do.
function pinned_mirror(released::Ref{Bool})
    host = Vector{Float64}(undef, 64)
    finalizer(_ -> (released[] = true), ParExt._pinned_storage(host))
    return reshape(host, 8, 8)
end

function distributed_spatial(cfg, values; decompose_theta::Bool=true)
    pen = decompose_theta ?
        Pencil((cfg.nlat, cfg.nlon), (1,), comm) :
        Pencil((cfg.nlat, cfg.nlon), comm)
    ranges = PencilArrays.range_local(pen)
    local_values = Matrix{eltype(values)}(
        undef, length(ranges[1]), length(ranges[2]),
    )
    for (j, global_j) in enumerate(ranges[2]),
        (i, global_i) in enumerate(ranges[1])
        local_values[i, j] = values[global_i, global_j]
    end
    return PencilArray(pen, local_values)
end

function distributed_spatial_error(local_values, reference, prototype)
    ranges = PencilArrays.range_local(pencil(prototype))
    values = local_values isa PencilArray ? parent(local_values) : local_values
    error = 0.0
    for (j, global_j) in enumerate(ranges[2]),
        (i, global_i) in enumerate(ranges[1])
        error = max(error, abs(values[i, j] - reference[global_i, global_j]))
    end
    return MPI.Allreduce(error, max, comm)
end

@testset "distributed plan preflight ($nprocs ranks)" begin
    lmax = 4
    nlat = 7
    nlon = 11
    cfg = create_gauss_config(lmax, nlat; nlon)
    field = zero_spatial(cfg)

    @testset "explicit transform communicators are validated metadata" begin
        duplicate_a = MPI.Comm_dup(comm)
        duplicate_b = MPI.Comm_dup(comm)
        try
            selected = iseven(rank) ? duplicate_a : duplicate_b
            spectral = SHTnsKit.create_spectral_array(cfg; comm)
            fill!(parent(spectral), 0)
            dense = zeros(ComplexF64, lmax + 1, lmax + 1)
            nfields = 2
            spatial_batch = PencilArray{Float64}(
                undef, SHTnsKit.create_spatial_pencil(cfg; comm), nfields,
            )
            spectral_batch = PencilArray{ComplexF64}(
                undef, SHTnsKit.create_spectral_pencil(cfg; comm), nfields,
            )
            fill!(parent(spatial_batch), 0)
            fill!(parent(spectral_batch), 0)

            analysis_calls = (
                candidate -> SHTnsKit.dist_analysis(
                    cfg, field; comm=candidate,
                ),
                candidate -> SHTnsKit.analysis_sphtor(
                    cfg, field, field; comm=candidate,
                ),
                candidate -> SHTnsKit.dist_analysis_sphtor(
                    cfg, field, field; comm=candidate,
                ),
                candidate -> SHTnsKit.analysis_sphtor_batch(
                    cfg, spatial_batch, spatial_batch; comm=candidate,
                ),
            )
            synthesis_calls = (
                candidate -> SHTnsKit.dist_synthesis(
                    cfg, dense; prototype_θφ=field, comm=candidate,
                ),
                candidate -> SHTnsKit.dist_synthesis(
                    cfg, spectral; prototype_θφ=field, comm=candidate,
                ),
                candidate -> SHTnsKit.synthesis(
                    cfg, spectral; prototype_θφ=field, comm=candidate,
                ),
                candidate -> SHTnsKit.dist_synthesis_sphtor(
                    cfg, spectral, spectral;
                    prototype_θφ=field, comm=candidate,
                ),
                candidate -> SHTnsKit.synthesis_sphtor(
                    cfg, spectral, spectral;
                    prototype_θφ=field, comm=candidate,
                ),
                candidate -> SHTnsKit.dist_synthesis_qst(
                    cfg, spectral, spectral, spectral;
                    prototype_θφ=field, comm=candidate,
                ),
                candidate -> SHTnsKit.synthesis_qst(
                    cfg, spectral, spectral, spectral;
                    prototype_θφ=field, comm=candidate,
                ),
                candidate -> SHTnsKit.synthesis_sphtor_batch(
                    cfg, spectral_batch, spectral_batch; comm=candidate,
                ),
                candidate -> SHTnsKit.synthesis_batch(
                    cfg, spectral_batch;
                    prototype_θφ=spatial_batch, comm=candidate,
                ),
                candidate -> SHTnsKit.synthesis_batch_cplx(
                    cfg, spectral_batch;
                    prototype_θφ=spatial_batch, comm=candidate,
                ),
            )

            # Congruent duplicates are valid metadata even when ranks select
            # different duplicate handles. Execution must stay on the stable
            # communicator owned by the input or spatial prototype.
            for call in (analysis_calls..., synthesis_calls...)
                result = call(selected)
                values = result isa Tuple ? result : (result,)
                @test all(values) do value
                    storage = value isa PencilArray ? parent(value) : value
                    all(iszero, storage)
                end
            end

            # These complex degree-limited aliases need distributed methods of
            # their own; otherwise a CPU PencilArray falls through to the
            # serial AbstractMatrix implementation while a vendor peer enters
            # the compound-extension collective firewall.
            for call in (
                    () -> SHTnsKit.synthesis_sph_l_cplx(
                        cfg, spectral, lmax; prototype_θφ=field,
                    ),
                    () -> SHTnsKit.synthesis_tor_l_cplx(
                        cfg, spectral, lmax; prototype_θφ=field,
                    ),
                )
                @test all(component -> all(iszero, parent(component)), call())
            end

            field_before = copy(parent(field))
            spectral_before = copy(parent(spectral))
            spatial_batch_before = copy(parent(spatial_batch))
            spectral_batch_before = copy(parent(spectral_batch))
            for call in (analysis_calls..., synthesis_calls...)
                @test_throws ArgumentError call(MPI.COMM_SELF)
            end
            @test parent(field) == field_before
            @test parent(spectral) == spectral_before
            @test parent(spatial_batch) == spatial_batch_before
            @test parent(spectral_batch) == spectral_batch_before
        finally
            ParExt._safe_comm_free(duplicate_a)
            ParExt._safe_comm_free(duplicate_b)
        end
    end

    @testset "distributed plan constructors require replicated signatures" begin
        if nprocs > 1
            rank_lmax = rank == nprocs - 1 ? lmax + 1 : lmax
            @test_throws ArgumentError ParExt.create_distributed_spectral_plan(
                rank_lmax, lmax, comm; mres=1,
            )
            @test_throws ArgumentError ParExt.create_distributed_spectral_plan_2d(
                rank_lmax, lmax, comm; p_l=1, p_m=nprocs, mres=1,
            )

            # Optional scratch behavior is also part of the collective call
            # signature; otherwise peers construct plans with incompatible
            # cached state even though both Comm_split calls happen to complete.
            rank_scratch = rank == nprocs - 1
            @test_throws ArgumentError ParExt.create_distributed_spectral_plan_2d(
                lmax, lmax, comm; p_l=1, p_m=nprocs, mres=1,
                with_scratch=rank_scratch, prototype_θφ=field, cfg=cfg,
            )

            # `with_vector` changes which cached tables the collective transpose
            # plan builds, even though its FFT plan happens to be identical.
            rank_vector = rank != nprocs - 1
            @test_throws ArgumentError DistTransposePlan(
                cfg; comm, nlev=1, use_rfft=true, with_vector=rank_vector,
            )
            @test_throws ArgumentError DistTransposePlan(
                cfg; comm,
                nlev=(rank == nprocs - 1 ? 2 : 1),
                use_rfft=true, with_vector=true,
            )
            @test_throws ArgumentError DistTransposePlan(
                cfg; comm, nlev=1,
                use_rfft=(rank != nprocs - 1), with_vector=true,
            )

            rank_rfft = rank == nprocs - 1
            @test_throws ArgumentError ParExt.DistAnalysisPlan(
                cfg, field; use_rfft=rank_rfft,
            )
            @test_throws ArgumentError ParExt.DistPlan(
                cfg, field; use_rfft=rank_rfft,
            )
            @test_throws ArgumentError ParExt.DistSphtorPlan(
                cfg, field; use_rfft=rank_rfft,
                with_spatial_scratch=rank_vector,
            )
        end
    end

    @testset "1D plan rejects cfg and communicator mismatches" begin
        plan = ParExt.create_distributed_spectral_plan(lmax, lmax, comm; mres=1)

        cfg_mres = create_gauss_config(lmax, nlat; nlon, mres=2)
        @test_throws ArgumentError ParExt.dist_analysis_distributed(cfg_mres, field; plan)

        cfg_lmax = create_gauss_config(lmax + 1, nlat; nlon, mmax=lmax)
        @test_throws ArgumentError ParExt.dist_analysis_distributed(cfg_lmax, field; plan)

        alm = ParExt.create_distributed_spectral_array(plan)
        @test_throws ArgumentError ParExt.dist_synthesis_distributed(
            cfg_mres, alm; prototype_θφ=field,
        )

        # COMM_SELF has the same local process but is not congruent to COMM_WORLD.
        # The rejection must happen before a world-communicator transform collective.
        self_field = zero_spatial(cfg, MPI.COMM_SELF)
        @test_throws ArgumentError ParExt.dist_analysis_distributed(cfg, self_field; plan)
        @test_throws ArgumentError ParExt.dist_synthesis_distributed(
            cfg, alm; prototype_θφ=self_field,
        )

        short_pen = Pencil((cfg.nlat - 1, cfg.nlon), (1,), comm)
        short_field = PencilArray(
            short_pen, zeros(Float64, PencilArrays.size_local(short_pen)...),
        )
        @test_throws DimensionMismatch ParExt.dist_analysis_distributed(
            cfg, short_field; plan,
        )
    end

    @testset "2D plan rejects cfg and communicator mismatches" begin
        plan = ParExt.create_distributed_spectral_plan_2d(
            lmax, lmax, comm; p_l=1, p_m=nprocs, mres=1,
        )
        try
            cfg_mres = create_gauss_config(lmax, nlat; nlon, mres=2)
            @test_throws ArgumentError ParExt.dist_analysis_distributed_2d(
                cfg_mres, field; plan,
            )

            cfg_lmax = create_gauss_config(lmax + 1, nlat; nlon, mmax=lmax)
            @test_throws ArgumentError ParExt.dist_analysis_distributed_2d(
                cfg_lmax, field; plan,
            )

            alm = ParExt.create_distributed_spectral_array_2d(plan)
            @test_throws ArgumentError ParExt.dist_synthesis_distributed_2d(
                cfg_mres, alm; prototype_θφ=field,
            )
            @test_throws ArgumentError ParExt.dist_synthesis_distributed_2d_optimized(
                cfg_lmax, alm; prototype_θφ=field,
            )

            self_field = zero_spatial(cfg, MPI.COMM_SELF)
            @test_throws ArgumentError ParExt.dist_analysis_distributed_2d(
                cfg, self_field; plan,
            )


            short_pen = Pencil((cfg.nlat - 1, cfg.nlon), (1,), comm)
            short_field = PencilArray(
                short_pen, zeros(Float64, PencilArrays.size_local(short_pen)...),
            )
            @test_throws DimensionMismatch ParExt.dist_analysis_distributed_2d(
                cfg, short_field; plan,
            )
        finally
            close(plan)
        end

        # When cfg is supplied to plan construction, the explicit dimensions
        # must agree even if scratch allocation is disabled.
        cfg_lmax = create_gauss_config(lmax + 1, nlat; nlon, mmax=lmax)
        @test_throws ArgumentError ParExt.create_distributed_spectral_plan_2d(
            lmax, lmax, comm; p_l=1, p_m=nprocs, mres=1, cfg=cfg_lmax,
        )
    end

    @testset "2D scratch plans remain bound to cfg and prototype" begin
        plan = ParExt.create_distributed_spectral_plan_2d(
            lmax, lmax, comm; p_l=1, p_m=nprocs, mres=1,
            with_scratch=true, prototype_θφ=field, cfg=cfg,
        )
        try
            changed_cfg = create_gauss_config(lmax, nlat; nlon)
            changed_cfg.w[1] = nextfloat(changed_cfg.w[1])
            @test_throws ArgumentError ParExt.dist_analysis_distributed_2d(
                changed_cfg, field; plan,
            )

            alternate_pen = Pencil((cfg.nlat, cfg.nlon), comm)
            alternate_field = PencilArray(
                alternate_pen,
                zeros(Float64, PencilArrays.size_local(alternate_pen)...),
            )
            @test_throws DimensionMismatch ParExt.dist_analysis_distributed_2d(
                cfg, alternate_field; plan,
            )
        finally
            close(plan)
        end

        short_pen = Pencil((cfg.nlat - 1, cfg.nlon), (1,), comm)
        short_field = PencilArray(
            short_pen, zeros(Float64, PencilArrays.size_local(short_pen)...),
        )
        @test_throws DimensionMismatch ParExt.create_distributed_spectral_plan_2d(
            lmax, lmax, comm; p_l=1, p_m=nprocs, mres=1,
            with_scratch=true, prototype_θφ=short_field, cfg=cfg,
        )
    end

    @testset "closed 2D plans are rejected before transform work" begin
        plan = ParExt.create_distributed_spectral_plan_2d(
            lmax, lmax, comm; p_l=1, p_m=nprocs, mres=1,
        )
        coefficients = ParExt.create_distributed_spectral_array_2d(plan)
        close(plan)
        @test_throws ArgumentError ParExt.validate_2d_distribution_alignment(
            plan, field,
        )
        @test_throws ArgumentError ParExt.dist_analysis_distributed_2d(
            cfg, field; plan,
        )
        @test_throws ArgumentError ParExt.dist_synthesis_distributed_2d(
            cfg, coefficients; prototype_θφ=field,
        )
        @test_throws ArgumentError ParExt.dist_synthesis_distributed_2d_optimized(
            cfg, coefficients; prototype_θφ=field,
        )
    end

    @testset "distributed-plan call options are replicated" begin
        if nprocs > 1
            plan1d = ParExt.create_distributed_spectral_plan(
                lmax, lmax, comm; mres=1,
            )
            @test_throws ArgumentError ParExt.dist_analysis_distributed(
                cfg, field; plan=plan1d,
                use_tables=(rank == nprocs - 1),
            )

            plan2d = ParExt.create_distributed_spectral_plan_2d(
                lmax, lmax, comm; p_l=1, p_m=nprocs, mres=1,
            )
            try
                @test_throws ArgumentError ParExt.dist_analysis_distributed_2d(
                    cfg, field; plan=plan2d,
                    assume_aligned=(rank == nprocs - 1),
                )
                @test_throws ArgumentError ParExt.dist_analysis_distributed_2d(
                    cfg, field; plan=plan2d,
                    use_tables=(rank == nprocs - 1),
                )

                coefficients = ParExt.create_distributed_spectral_array_2d(plan2d)
                @test_throws ArgumentError ParExt.dist_synthesis_distributed_2d_optimized(
                    cfg, coefficients; prototype_θφ=field,
                    real_output=(rank != nprocs - 1),
                )
            finally
                close(plan2d)
            end
        end
    end

    @testset "configuration replication covers conventions and quadrature" begin
        cfg_norm = create_gauss_config(
            lmax, nlat; nlon, real_norm=(rank == nprocs - 1),
        )
        @test_throws ArgumentError DistTransposePlan(
            cfg_norm; comm, nlev=1, use_rfft=true, with_vector=false,
        )

        cfg_grid = create_gauss_config(lmax, nlat; nlon)
        rank == nprocs - 1 && (cfg_grid.w[1] = nextfloat(cfg_grid.w[1]))
        @test_throws ArgumentError ParExt._validate_cfg_replicated(cfg_grid, comm)
    end

    @testset "exported plan constructors and local_size reach the extension" begin
        @test SHTnsKit.DistAnalysisPlan(cfg, field) isa ParExt.DistAnalysisPlan
        @test SHTnsKit.DistPlan(cfg, field) isa ParExt.DistPlan
        @test SHTnsKit.DistSphtorPlan(cfg, field) isa ParExt.DistSphtorPlan
        @test SHTnsKit.DistQstPlan(cfg, field) isa ParExt.DistQstPlan

        distributed = ParExt.create_distributed_spectral_array(
            ParExt.create_distributed_spectral_plan(lmax, lmax, comm),
        )
        @test ParExt.local_size === SHTnsKit.local_size
        @test SHTnsKit.local_size(distributed) == length(distributed.local_coeffs)
        @test SHTnsKit.local_size(parent(field)) == size(parent(field))
    end

    @testset "planned cfg-form pencils match their construction prototype" begin
        short_field = zero_spatial(
            create_gauss_config(lmax, nlat - 1; nlon),
        )
        @test_throws DimensionMismatch ParExt.DistAnalysisPlan(cfg, short_field)
        @test_throws DimensionMismatch ParExt.DistPlan(cfg, short_field)
        @test_throws DimensionMismatch ParExt.DistSphtorPlan(cfg, short_field)

        analysis_plan = ParExt.DistAnalysisPlan(cfg, field)
        sphtor_plan = ParExt.DistSphtorPlan(cfg, field)
        synthesis_plan = ParExt.DistPlan(cfg, field)
        self_field = zero_spatial(cfg, MPI.COMM_SELF)
        scalar_out = zeros(ComplexF64, lmax + 1, lmax + 1)
        vector_out = similar(scalar_out)

        @test_throws ArgumentError SHTnsKit.dist_analysis!(
            analysis_plan, scalar_out, self_field,
        )
        @test_throws ArgumentError SHTnsKit.dist_analysis_sphtor!(
            sphtor_plan, scalar_out, vector_out, self_field, self_field,
        )

        spectral = SHTnsKit.create_spectral_array(cfg; comm)
        fill!(parent(spectral), 0)
        @test_throws ArgumentError SHTnsKit.dist_synthesis!(
            synthesis_plan, self_field, spectral,
        )

        @test_throws ArgumentError SHTnsKit.dist_analysis_sphtor(
            cfg, field, self_field,
        )

        original_weight = cfg.w[1]
        cfg.w[1] = nextfloat(original_weight)
        try
            @test_throws ArgumentError SHTnsKit.dist_analysis!(
                analysis_plan, scalar_out, field,
            )
        finally
            cfg.w[1] = original_weight
        end

        original_node = cfg.x[1]
        cfg.x[1] = nextfloat(original_node)
        try
            @test_throws ArgumentError SHTnsKit.dist_analysis_sphtor!(
                sphtor_plan, scalar_out, vector_out, field, field,
            )
        finally
            cfg.x[1] = original_node
        end
    end

    @testset "planned scalar synthesis validates options before spectral work" begin
        if nprocs > 1
            complex_field = PencilArray(
                pencil(field), zeros(ComplexF64, size(parent(field))...),
            )
            synthesis_plan = ParExt.DistPlan(cfg, complex_field)
            wrong_pen = Pencil((cfg.lmax + 2, cfg.mmax + 1), (2,), comm)
            wrong_spectral = PencilArray{ComplexF64}(undef, wrong_pen)
            fill!(parent(wrong_spectral), 0)

            # The rank-divergent option must win before the deliberately wrong
            # spectral shape reaches its own collective preflight.
            @test_throws ArgumentError SHTnsKit.dist_synthesis!(
                synthesis_plan, complex_field, wrong_spectral;
                real_output=(rank != nprocs - 1),
            )

            spectral = SHTnsKit.create_spectral_array(cfg; comm)
            fill!(parent(spectral), 0)
            rank_output = rank == nprocs - 1 ? field : complex_field
            @test_throws ArgumentError SHTnsKit.dist_synthesis!(
                synthesis_plan, rank_output, spectral; real_output=false,
            )
        end
    end

    @testset "planned multi-output transforms reject rank-varying aliases" begin
        if nprocs > 1
            spectral_size = (cfg.lmax + 1, cfg.mmax + 1)
            zeros_lm = zeros(ComplexF64, spectral_size)

            sphtor_plan = ParExt.DistSphtorPlan(cfg, field)
            S_out = zeros(ComplexF64, spectral_size)
            T_out = rank == nprocs - 1 ? S_out : similar(S_out)
            @test_throws ArgumentError SHTnsKit.dist_analysis_sphtor!(
                sphtor_plan, S_out, T_out, field, field,
            )

            Vt_out = similar(field)
            Vp_out = rank == nprocs - 1 ? Vt_out : similar(field)
            @test_throws ArgumentError SHTnsKit.dist_synthesis_sphtor!(
                sphtor_plan, Vt_out, Vp_out, zeros_lm, zeros_lm,
            )

            qst_plan = ParExt.DistQstPlan(cfg, field)
            Q_out = zeros(ComplexF64, spectral_size)
            S_qst_out = rank == nprocs - 1 ? Q_out : similar(Q_out)
            T_qst_out = similar(Q_out)
            @test_throws ArgumentError SHTnsKit.dist_analysis_qst!(
                qst_plan, Q_out, S_qst_out, T_qst_out,
                field, field, field,
            )

            Vr_out = similar(field)
            Vt_qst_out = rank == nprocs - 1 ? Vr_out : similar(field)
            Vp_qst_out = similar(field)
            @test_throws ArgumentError SHTnsKit.dist_synthesis_qst!(
                qst_plan, Vr_out, Vt_qst_out, Vp_qst_out,
                zeros_lm, zeros_lm, zeros_lm,
            )
        end
    end

    @testset "dense cfg-form synthesis checks exact coefficient shapes" begin
        oversized = zeros(ComplexF64, lmax + 2, lmax + 1)
        @test_throws DimensionMismatch SHTnsKit.dist_synthesis(
            cfg, oversized; prototype_θφ=field,
        )

        alm = zeros(ComplexF64, lmax + 1, lmax + 1)
        bad_minus = zeros(ComplexF64, lmax + 2, lmax + 1)
        @test_throws DimensionMismatch SHTnsKit.dist_synthesis(
            cfg, alm; prototype_θφ=field, real_output=false, Aminus=bad_minus,
        )
    end

    @testset "spectral-pencil synthesis validates before gathering" begin
        spectral = SHTnsKit.create_spectral_array(cfg; comm)
        fill!(parent(spectral), 0)

        spectral_self = SHTnsKit.create_spectral_array(cfg; comm=MPI.COMM_SELF)
        fill!(parent(spectral_self), 0)
        @test_throws ArgumentError SHTnsKit.dist_synthesis(
            cfg, spectral_self; prototype_θφ=field,
        )
        @test_throws ArgumentError SHTnsKit.dist_synthesis_sphtor(
            cfg, spectral, spectral_self; prototype_θφ=field,
        )

        wrong_pen = Pencil((cfg.lmax + 2, cfg.mmax + 1), (2,), comm)
        wrong_shape = PencilArray{ComplexF64}(undef, wrong_pen)
        fill!(parent(wrong_shape), 0)
        @test_throws DimensionMismatch SHTnsKit.dist_synthesis(
            cfg, wrong_shape; prototype_θφ=field,
        )

        alternate_pen = Pencil(
            (cfg.lmax + 1, cfg.mmax + 1), (1,), comm,
        )
        alternate = PencilArray{ComplexF64}(undef, alternate_pen)
        fill!(parent(alternate), 0)
        @test_throws ArgumentError SHTnsKit.dist_synthesis_qst(
            cfg, spectral, spectral, alternate; prototype_θφ=field,
        )
    end

    @testset "Pencil QST transforms keep collectives on the prototype communicator" begin
        # Both communicators have the same group and rank ordering as `comm`, but
        # distinct MPI contexts. Selecting a different duplicate on each rank is
        # legal input metadata; no collective may subsequently re-anchor itself
        # on a coefficient Pencil's rank-varying context.
        duplicate_a = MPI.Comm_dup(comm)
        duplicate_b = MPI.Comm_dup(comm)
        try
            spectral_size = (cfg.lmax + 1, cfg.mmax + 1)
            pencil_a = Pencil(spectral_size, (2,), duplicate_a)
            pencil_b = Pencil(spectral_size, (2,), duplicate_b)

            Q = zeros(ComplexF64, spectral_size)
            S = zeros(ComplexF64, spectral_size)
            T = zeros(ComplexF64, spectral_size)
            Q[1, 1] = 0.3
            Q[3, 2] = 0.15 - 0.2im
            S[2, 1] = -0.1
            S[4, 3] = 0.08 + 0.04im
            T[3, 2] = -0.06 + 0.09im

            function place_on_pencil(values, pen)
                placed = PencilArray{eltype(values)}(undef, pen)
                l_indices, m_indices = PencilArrays.range_local(pen)
                for (local_m, global_m) in pairs(m_indices),
                    (local_l, global_l) in pairs(l_indices)
                    parent(placed)[local_l, local_m] = values[global_l, global_m]
                end
                return placed
            end

            Qa = place_on_pencil(Q, pencil_a)
            Qb = place_on_pencil(Q, pencil_b)
            Sa = place_on_pencil(S, pencil_a)
            Sb = place_on_pencil(S, pencil_b)
            Ta = place_on_pencil(T, pencil_a)
            Tb = place_on_pencil(T, pencil_b)

            # Mix communicator contexts both across ranks and across operands.
            Qrank = iseven(rank) ? Qa : Qb
            Srank = iseven(rank) ? Sb : Sa
            Trank = iseven(rank) ? Ta : Tb
            expected = SHTnsKit.synthesis_qst(cfg, Q, S, T; real_output=true)
            actual = SHTnsKit.dist_synthesis_qst(
                cfg, Qrank, Srank, Trank;
                prototype_θφ=field, real_output=true,
            )
            for component in eachindex(expected)
                @test distributed_spatial_error(
                    actual[component], expected[component], field,
                ) ≤ 5e-12
            end

            spatial_pencil_a = Pencil(
                (cfg.nlat, cfg.nlon), (1,), duplicate_a,
            )
            spatial_pencil_b = Pencil(
                (cfg.nlat, cfg.nlon), (1,), duplicate_b,
            )
            spatial_a = map(
                values -> place_on_pencil(values, spatial_pencil_a), expected,
            )
            spatial_b = map(
                values -> place_on_pencil(values, spatial_pencil_b), expected,
            )
            Vrrank = iseven(rank) ? spatial_a[1] : spatial_b[1]
            Vtrank = iseven(rank) ? spatial_b[2] : spatial_a[2]
            Vprank = iseven(rank) ? spatial_a[3] : spatial_b[3]
            plan = ParExt.DistQstPlan(cfg, field)
            Qout = zeros(ComplexF64, spectral_size)
            Sout = similar(Qout)
            Tout = similar(Qout)
            SHTnsKit.dist_analysis_qst!(
                plan, Qout, Sout, Tout, Vrrank, Vtrank, Vprank,
            )
            @test Qout ≈ Q atol=5e-12 rtol=5e-12
            @test Sout ≈ S atol=5e-12 rtol=5e-12
            @test Tout ≈ T atol=5e-12 rtol=5e-12
        finally
            ParExt._safe_comm_free(duplicate_a)
            ParExt._safe_comm_free(duplicate_b)
        end
    end

    @testset "packed Pencil synthesis keeps collectives on the prototype communicator" begin
        duplicate_a = MPI.Comm_dup(comm)
        duplicate_b = MPI.Comm_dup(comm)
        try
            function place_packed(values, duplicate)
                pen = Pencil((length(values), 1), (1,), duplicate)
                placed = PencilArray{eltype(values)}(undef, pen)
                indices = PencilArrays.range_local(pen)[1]
                for (local_index, global_index) in pairs(indices)
                    parent(placed)[local_index, 1] = values[global_index]
                end
                return placed
            end

            dense = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
            dense[1, 1] = 0.3
            dense[3, 2] = 0.15 - 0.2im
            dense[5, 3] = -0.04 + 0.08im
            packed = SHTnsKit.pack_lm(cfg, dense)
            packed_a = place_packed(packed, duplicate_a)
            packed_b = place_packed(packed, duplicate_b)
            rank_packed = iseven(rank) ? packed_a : packed_b
            expected_real = reshape(
                SHTnsKit.synthesis_packed(cfg, packed), cfg.nlat, cfg.nlon,
            )
            actual_real = SHTnsKit.synthesis_packed(
                cfg, rank_packed; prototype_θφ=field,
            )
            @test distributed_spatial_error(
                actual_real, expected_real, field,
            ) ≤ 5e-12

            complex_count = SHTnsKit.nlm_cplx_calc(
                cfg.lmax, cfg.mmax, 1,
            )
            complex_packed = zeros(ComplexF64, complex_count)
            complex_packed[
                SHTnsKit.LM_cplx_index(cfg.lmax, cfg.mmax, 0, 0) + 1
            ] = 0.2 + 0.1im
            complex_packed[
                SHTnsKit.LM_cplx_index(cfg.lmax, cfg.mmax, 2, 1) + 1
            ] = -0.06 + 0.09im
            complex_packed[
                SHTnsKit.LM_cplx_index(cfg.lmax, cfg.mmax, 2, -1) + 1
            ] = 0.04 - 0.03im
            complex_a = place_packed(complex_packed, duplicate_a)
            complex_b = place_packed(complex_packed, duplicate_b)
            rank_complex = iseven(rank) ? complex_b : complex_a
            complex_field = PencilArray{ComplexF64}(undef, pencil(field))
            fill!(parent(complex_field), 0)
            expected_complex = reshape(
                SHTnsKit.synthesis_packed_cplx(cfg, complex_packed),
                cfg.nlat, cfg.nlon,
            )
            actual_complex = SHTnsKit.synthesis_packed_cplx(
                cfg, rank_complex; prototype_θφ=complex_field,
            )
            @test distributed_spatial_error(
                actual_complex, expected_complex, complex_field,
            ) ≤ 5e-12
        finally
            ParExt._safe_comm_free(duplicate_a)
            ParExt._safe_comm_free(duplicate_b)
        end
    end

    @testset "batch Pencil synthesis uses stable spatial communicators" begin
        duplicate_a = MPI.Comm_dup(comm)
        duplicate_b = MPI.Comm_dup(comm)
        try
            nfields = 2
            dense_batch = zeros(
                ComplexF64, cfg.lmax + 1, cfg.mmax + 1, nfields,
            )
            dense_batch[1, 1, 1] = 0.2
            dense_batch[3, 2, 1] = -0.04 + 0.07im
            dense_batch[2, 1, 2] = 0.1
            dense_batch[4, 3, 2] = 0.03 - 0.02im
            expected = SHTnsKit.synthesis_batch(cfg, dense_batch)

            function place_batch(values, duplicate)
                pen = Pencil(size(values)[1:2], (2,), duplicate)
                placed = PencilArray{eltype(values)}(
                    undef, pen, size(values, 3),
                )
                first_indices, second_indices =
                    PencilArrays.range_local(pen)
                for k in axes(values, 3),
                    (local_j, global_j) in pairs(second_indices),
                    (local_i, global_i) in pairs(first_indices)
                    parent(placed)[local_i, local_j, k] =
                        values[global_i, global_j, k]
                end
                return placed
            end

            coefficients_a = place_batch(dense_batch, duplicate_a)
            coefficients_b = place_batch(dense_batch, duplicate_b)
            rank_coefficients = iseven(rank) ? coefficients_a : coefficients_b
            prototype = PencilArray{Float64}(
                undef, pencil(field), nfields,
            )
            fill!(parent(prototype), 0)
            actual = SHTnsKit.synthesis_batch(
                cfg, rank_coefficients; prototype_θφ=prototype,
            )
            local_ranges = PencilArrays.range_local(pencil(prototype))
            local_reference = @view expected[
                local_ranges[1], local_ranges[2], :,
            ]
            local_error = maximum(
                abs, parent(actual) .- local_reference; init=0.0,
            )
            @test MPI.Allreduce(local_error, max, comm) ≤ 5e-12

            spatial_a = PencilArray{Float64}(
                undef,
                Pencil((cfg.nlat, cfg.nlon), (1,), duplicate_a),
                nfields,
            )
            spatial_b = PencilArray{Float64}(
                undef,
                Pencil((cfg.nlat, cfg.nlon), (1,), duplicate_b),
                nfields,
            )
            fill!(parent(spatial_a), 0)
            fill!(parent(spatial_b), 0)
            rank_prototype = iseven(rank) ? spatial_b : spatial_a
            output = similar(prototype)
            fill!(parent(output), NaN)
            @test SHTnsKit.synthesis_batch!(
                cfg, output, rank_coefficients;
                prototype_θφ=rank_prototype,
            ) === output
            output_error = maximum(
                abs, parent(output) .- local_reference; init=0.0,
            )
            @test MPI.Allreduce(output_error, max, comm) ≤ 5e-12
        finally
            ParExt._safe_comm_free(duplicate_a)
            ParExt._safe_comm_free(duplicate_b)
        end
    end

    @testset "dense synthesis inputs are fully replicated" begin
        # More than 256 coefficients puts this valid triangular entry outside
        # both bounded samples used by the old replication check.
        cfg_big = create_gauss_config(20, 22; nlon=41)
        field_big = zero_spatial(cfg_big)
        A_big = zeros(ComplexF64, 21, 21)
        rank == nprocs - 1 && (A_big[11, 11] = 1 + 2im)
        @test_throws ArgumentError SHTnsKit.dist_synthesis(
            cfg_big, A_big; prototype_θφ=field_big,
        )

        S = zeros(ComplexF64, lmax + 1, lmax + 1)
        T = similar(S)
        rank == nprocs - 1 && (S[3, 2] = 0.25 - 0.5im)
        @test_throws ArgumentError SHTnsKit.dist_synthesis_sphtor(
            cfg, S, T; prototype_θφ=field,
        )

        if nprocs > 1
            expected = zeros(ComplexF64, lmax + 1, lmax + 1)
            rank_sized = rank == nprocs - 1 ?
                zeros(ComplexF64, lmax + 2, lmax + 1) : expected
            @test_throws DimensionMismatch SHTnsKit.dist_synthesis(
                cfg, rank_sized; prototype_θφ=field,
            )
            @test_throws DimensionMismatch SHTnsKit.dist_synthesis_sphtor(
                cfg, rank_sized, expected; prototype_θφ=field,
            )

            rank_minus = rank == nprocs - 1 ? expected : nothing
            @test_throws ArgumentError SHTnsKit.dist_synthesis(
                cfg, expected; prototype_θφ=field,
                real_output=false, Aminus=rank_minus,
            )

            analysis_plan = ParExt.DistAnalysisPlan(cfg, field)
            rank_output = rank == nprocs - 1 ?
                zeros(ComplexF64, lmax + 2, lmax + 1) : expected
            @test_throws DimensionMismatch SHTnsKit.dist_analysis!(
                analysis_plan, rank_output, field,
            )
        end
    end

    @testset "replication checks ignore semantically unused storage" begin
        zeros_lm = zeros(ComplexF64, lmax + 1, lmax + 1)

        minus = similar(zeros_lm)
        fill!(minus, 0)
        rank == nprocs - 1 && fill!(@view(minus[:, 1]), 3 + 4im)
        scalar = SHTnsKit.dist_synthesis(
            cfg, zeros_lm; prototype_θφ=field,
            real_output=false, Aminus=minus,
        )
        @test all(iszero, scalar)

        S = similar(zeros_lm)
        T = similar(zeros_lm)
        fill!(S, 0)
        fill!(T, 0)
        rank == nprocs - 1 && (S[1, 1] = 2 - 5im)
        Vt, Vp = SHTnsKit.dist_synthesis_sphtor(
            cfg, S, T; prototype_θφ=field, real_output=true,
        )
        @test all(iszero, Vt)
        @test all(iszero, Vp)
    end

    @testset "cfg-form collective options are replicated" begin
        if nprocs > 1
            rank_rfft = rank == nprocs - 1
            @test_throws ArgumentError SHTnsKit.dist_analysis(
                cfg, field; use_rfft=rank_rfft,
            )
            @test_throws ArgumentError SHTnsKit.dist_analysis_sphtor(
                cfg, field, field; use_rfft=rank_rfft,
            )
            @test_throws ArgumentError SHTnsKit.analysis(
                cfg, field; return_pencil=(rank != nprocs - 1),
            )
        end
    end
end

@testset "one-dimensional distributed spectral storage honors mres" begin
    lmax = mmax = 6
    mres = 2
    plan = ParExt.create_distributed_spectral_plan(lmax, mmax, comm; mres)
    distributed = ParExt.create_distributed_spectral_array(plan)
    dense = fill(99.0 + 7.0im, lmax + 1, mmax + 1)
    expected = zeros(ComplexF64, size(dense))
    for m in 0:mres:mmax, l in m:lmax
        expected[l + 1, m + 1] = complex(10l + m, l - m)
        dense[l + 1, m + 1] = expected[l + 1, m + 1]
    end
    ParExt.scatter_from_dense!(distributed, dense)
    @test ParExt.gather_to_dense(distributed) == expected
    @test all(m % mres == 0 for (_, m) in plan.local_lm_indices)
    @test length(plan.local_packed_indices) ==
          length(unique(plan.local_packed_indices))
end

@testset "distributed spectral reduction validates collective operands" begin
    plan = ParExt.create_distributed_spectral_plan(4, 4, comm; mres=1)

    local_contribution = fill(ComplexF32(rank + 1), 2, 3)
    reduced = similar(local_contribution)
    @test ParExt.distributed_spectral_reduce!(
        plan, local_contribution, reduced,
    ) === reduced
    @test all(==(ComplexF32(nprocs * (nprocs + 1) ÷ 2)), reduced)

    if nprocs > 1
        # Equal element counts are not enough: MPI collectives also require
        # the same matrix shape and element type on every rank.
        rank_shape = rank == nprocs - 1 ? (1, 6) : (2, 3)
        shaped_input = zeros(ComplexF64, rank_shape)
        shaped_output = similar(shaped_input)
        @test_throws ArgumentError ParExt.distributed_spectral_reduce!(
            plan, shaped_input, shaped_output,
        )

        typed_input = rank == nprocs - 1 ?
            zeros(ComplexF32, 2, 3) : zeros(ComplexF64, 2, 3)
        typed_output = similar(typed_input)
        @test_throws ArgumentError ParExt.distributed_spectral_reduce!(
            plan, typed_input, typed_output,
        )

        alias_input = zeros(ComplexF64, 2, 3)
        alias_output = rank == 0 ? alias_input : similar(alias_input)
        @test_throws ArgumentError ParExt.distributed_spectral_reduce!(
            plan, alias_input, alias_output,
        )
    end

    @test_throws ArgumentError ParExt.distributed_spectral_reduce!(
        plan, zeros(Int, 2, 3), zeros(Int, 2, 3),
    )
    @test_throws ArgumentError ParExt.distributed_spectral_reduce!(
        plan, zeros(ComplexF64, 2, 3), zeros(ComplexF32, 2, 3),
    )
end

@testset "distributed storage residency is the first collective preflight" begin
    if nprocs > 1
        storage_cfg = create_gauss_config(4, 7; nlon=11)
        storage_field = zero_spatial(storage_cfg)
        plan_1d = ParExt.create_distributed_spectral_plan(
            storage_cfg.lmax, storage_cfg.mmax, comm; mres=storage_cfg.mres,
        )
        plan_2d = ParExt.create_distributed_spectral_plan_2d(
            storage_cfg.lmax, storage_cfg.mmax, comm;
            p_l=1, p_m=nprocs, mres=storage_cfg.mres,
        )
        adapter = ParExt.ParallelGPUAdapter(
            :distributed_payload_rank_local_mock,
            value -> rank == nprocs - 1 && value isa Array,
            _ -> Array,
            _ -> rank,
            _ -> false,
            _ -> nothing,
            (T, n) -> Vector{T}(undef, n),
            (host, device) -> copyto!(host, device),
            (device, host) -> copyto!(device, host),
        )
        ParExt._register_parallel_gpu_adapter!(adapter)
        try
            divergent_cfg = create_gauss_config(4, 7; nlon=11)
            rank == nprocs - 1 &&
                (divergent_cfg.w[1] = nextfloat(divergent_cfg.w[1]))
            rank_shape = rank == nprocs - 1 ? (1, 6) : (2, 3)
            reduction_input = zeros(ComplexF64, rank_shape)
            reduction_output = similar(reduction_input)
            calls = (
                () -> ParExt.distributed_spectral_reduce!(
                    plan_1d, reduction_input, reduction_output,
                ),
                () -> ParExt.dist_analysis_distributed(
                    divergent_cfg, storage_field; plan=plan_1d,
                ),
                () -> ParExt.dist_analysis_distributed_2d(
                    divergent_cfg, storage_field; plan=plan_2d,
                ),
            )
            GC.@preserve adapter begin
                for call in calls
                    caught = try
                        call()
                        nothing
                    catch error
                        error
                    end
                    @test caught isa ArgumentError
                    if caught isa Exception
                        @test occursin(
                            "storage/vendor/device mismatch",
                            sprint(showerror, caught),
                        )
                    end
                end
            end
        finally
            lock(ParExt._PARALLEL_GPU_ADAPTER_LOCK) do
                delete!(
                    ParExt._PARALLEL_GPU_ADAPTERS,
                    :distributed_payload_rank_local_mock,
                )
            end
            close(plan_2d)
        end
    end
end

@testset "operator matrix residency is validated before Pencil work" begin
    if nprocs > 1
        operator_cfg = create_gauss_config(4, 7; nlon=11)
        input = SHTnsKit.create_spectral_array(operator_cfg; comm)
        output = similar(input)
        fill!(parent(input), 1 + 2im)
        sentinel = 19 - 7im
        fill!(parent(output), sentinel)
        mx = zeros(Float64, 2operator_cfg.nlm)
        adapter = ParExt.ParallelGPUAdapter(
            :operator_matrix_rank_local_mock,
            value -> rank == nprocs - 1 && value === mx,
            _ -> Array,
            _ -> rank,
            _ -> false,
            _ -> nothing,
            (T, n) -> Vector{T}(undef, n),
            (host, device) -> copyto!(host, device),
            (device, host) -> copyto!(device, host),
        )
        ParExt._register_parallel_gpu_adapter!(adapter)
        try
            GC.@preserve adapter begin
                # A live adapter must remain registered even when it captures
                # the matrix and garbage collection runs before validation.
                GC.gc(true)
                @test ParExt._parallel_gpu_adapter(mx) ===
                      (rank == nprocs - 1 ? adapter : nothing)
                @test_throws ArgumentError SHTnsKit.SH_mul_mx(
                    operator_cfg, mx, input, output,
                )
            end
            @test all(==(sentinel), parent(output))
        finally
            lock(ParExt._PARALLEL_GPU_ADAPTER_LOCK) do
                delete!(
                    ParExt._PARALLEL_GPU_ADAPTERS,
                    :operator_matrix_rank_local_mock,
                )
            end
        end
    end
end

@testset "pinned host storage lives as long as its reshaped mirrors" begin
    # Julia 1.11+ collects the allocated Vector wrapper separately from its
    # Memory, so pinning tied to the wrapper was released under live mirrors.
    released = Ref(false)
    mirror = pinned_mirror(released)
    for _ in 1:3
        GC.gc(true)
    end
    @test !released[]
    mirror[1] = 1.0
    host = Vector{ComplexF64}(undef, 5)
    storage = ParExt._pinned_storage(host)
    @test pointer(storage) == pointer(host)
    @test sizeof(storage) == sizeof(host)
end

@testset "distributed spectral payloads are validated before variable gathers" begin
    lmax = mmax = 4

    plan_1d = ParExt.create_distributed_spectral_plan(
        lmax, mmax, comm; mres=1,
    )
    coefficients_32 = ParExt.create_distributed_spectral_array(
        plan_1d, ComplexF32,
    )
    for (index, (l, m)) in pairs(plan_1d.local_lm_indices)
        coefficients_32.local_coeffs[index] = ComplexF32(10l + m, l - m)
    end
    gathered_32 = ParExt.gather_to_dense(coefficients_32)
    @test eltype(gathered_32) === ComplexF32
    for m in 0:mmax, l in m:lmax
        @test gathered_32[l + 1, m + 1] == ComplexF32(10l + m, l - m)
    end

    if nprocs > 1
        mixed_precision_1d = ParExt.create_distributed_spectral_array(
            plan_1d, rank == nprocs - 1 ? ComplexF32 : ComplexF64,
        )
        @test_throws ArgumentError ParExt.gather_to_dense(mixed_precision_1d)

        wrong_length_1d = ParExt.create_distributed_spectral_array(plan_1d)
        rank == 0 && pop!(wrong_length_1d.local_coeffs)
        @test_throws ArgumentError ParExt.gather_to_dense(wrong_length_1d)
    end

    # Put every rank in the same l-communicator so the successful Float32
    # case exercises the exact variable-count collective used in production.
    plan_2d = ParExt.create_distributed_spectral_plan_2d(
        lmax, mmax, comm; p_l=nprocs, p_m=1, mres=1,
    )
    try
        coefficients_2d_32 = ParExt.create_distributed_spectral_array_2d(
            plan_2d, ComplexF32,
        )
        for (index, (l, m)) in pairs(plan_2d.local_lm_indices)
            coefficients_2d_32.local_coeffs[index] =
                ComplexF32(10l + m, l - m)
        end
        gathered_2d_32 = ParExt.gather_to_dense_2d(coefficients_2d_32)
        @test eltype(gathered_2d_32) === ComplexF32
        for m in 0:mmax, l in m:lmax
            @test gathered_2d_32[l + 1, m + 1] ==
                  ComplexF32(10l + m, l - m)
        end

        if nprocs > 1
            mixed_precision_2d =
                ParExt.create_distributed_spectral_array_2d(
                    plan_2d,
                    rank == nprocs - 1 ? ComplexF32 : ComplexF64,
                )
            @test_throws ArgumentError ParExt.gather_to_dense_2d(
                mixed_precision_2d,
            )

            wrong_length_2d =
                ParExt.create_distributed_spectral_array_2d(plan_2d)
            rank == 0 && pop!(wrong_length_2d.local_coeffs)
            @test_throws ArgumentError ParExt.gather_to_dense_2d(
                wrong_length_2d,
            )
        end
    finally
        close(plan_2d)
    end

    if nprocs > 1
        # l-group metadata legitimately differs between m-groups, so it cannot
        # be included in the replicated plan signature. A failure in one
        # subgroup must nevertheless be reported on the world communicator so
        # that every rank stops before any later global collective.
        subgroup_plan = ParExt.create_distributed_spectral_plan_2d(
            lmax, mmax, comm; p_l=1, p_m=nprocs, mres=1,
        )
        try
            subgroup_coefficients =
                ParExt.create_distributed_spectral_array_2d(subgroup_plan)
            rank == 0 && (subgroup_plan.l_recv_counts[1] += 1)
            @test_throws ArgumentError ParExt.gather_to_dense_2d(
                subgroup_coefficients,
            )
        finally
            close(subgroup_plan)
        end


        # Separately constructed plans have distinct derived communicator
        # contexts even when every dimension and count is identical. Mixing
        # the plan objects across ranks must fail on the parent communicator
        # before any rank enters one of those derived contexts.
        plan_a = ParExt.create_distributed_spectral_plan_2d(
            lmax, mmax, comm; p_l=1, p_m=nprocs, mres=1,
        )
        plan_b = ParExt.create_distributed_spectral_plan_2d(
            lmax, mmax, comm; p_l=1, p_m=nprocs, mres=1,
        )
        try
            @test plan_a.instance_id != plan_b.instance_id
            alignment_field = zero_spatial(
                create_gauss_config(lmax, 7; nlon=11),
            )
            selected_plan = iseven(rank) ? plan_a : plan_b
            selected_coefficients =
                ParExt.create_distributed_spectral_array_2d(selected_plan)
            @test_throws ArgumentError ParExt.validate_2d_distribution_alignment(
                selected_plan, alignment_field,
            )
            @test_throws ArgumentError ParExt.gather_to_dense_2d(
                selected_coefficients,
            )
        finally
            close(plan_a)
            close(plan_b)
        end
    end
end

@testset "optimized 2D synthesis includes empty m-groups in payload preflight" begin
    if nprocs > 1
        sparse_cfg = create_gauss_config(
            1, 3; mmax=0, nlon=max(3, 2nprocs + 1),
        )
        reference = zeros(Float64, sparse_cfg.nlat, sparse_cfg.nlon)
        spatial = distributed_spatial(
            sparse_cfg, reference; decompose_theta=false,
        )
        sparse_plan = ParExt.create_distributed_spectral_plan_2d(
            sparse_cfg.lmax, sparse_cfg.mmax, comm;
            p_l=1, p_m=nprocs, mres=sparse_cfg.mres,
        )
        try
            # Only the first m-group owns m=0; all other ranks have an empty
            # m_range but must still join gather_to_dense_2d's world preflight.
            @test MPI.Allreduce(isempty(sparse_plan.m_range) ? 0 : 1, +, comm) == 1
            coefficients =
                ParExt.create_distributed_spectral_array_2d(sparse_plan)
            reconstructed = ParExt.dist_synthesis_distributed_2d_optimized(
                sparse_cfg, coefficients; prototype_θφ=spatial,
            )
            @test distributed_spatial_error(
                reconstructed, reference, spatial,
            ) ≤ 5e-12

            corrupted =
                ParExt.create_distributed_spectral_array_2d(sparse_plan)
            rank == 0 && pop!(corrupted.local_coeffs)
            @test_throws ArgumentError ParExt.dist_synthesis_distributed_2d_optimized(
                sparse_cfg, corrupted; prototype_θφ=spatial,
            )
        finally
            close(sparse_plan)
        end
    end
end

@testset "distributed spectral plans honor cfg conventions and alignment" begin
    lmax = mmax = 5
    cfg = create_gauss_config(
        lmax, lmax + 3; mmax, mres=2, nlon=2mmax + 2,
        norm=:schmidt, real_norm=true, cs_phase=false,
    )
    spatial = [sin(0.17i + 0.31j) + 0.2cos(0.23i - 0.11j)
               for i in 1:cfg.nlat, j in 1:cfg.nlon]
    reference_coefficients = analysis(cfg, spatial)
    reference_spatial = synthesis(cfg, reference_coefficients; real_output=true)
    theta_field = distributed_spatial(cfg, spatial; decompose_theta=true)
    phi_field = distributed_spatial(cfg, spatial; decompose_theta=false)

    plan_1d = ParExt.create_distributed_spectral_plan(
        lmax, mmax, comm; mres=cfg.mres,
    )
    coefficients_1d = ParExt.dist_analysis_distributed(
        cfg, theta_field; plan=plan_1d,
    )
    @test isapprox(
        ParExt.gather_to_dense(coefficients_1d), reference_coefficients;
        rtol=1e-10, atol=1e-11,
    )
    reconstructed_1d = ParExt.dist_synthesis_distributed(
        cfg, coefficients_1d; prototype_θφ=theta_field,
    )
    @test distributed_spatial_error(
        reconstructed_1d, reference_spatial, theta_field,
    ) < 1e-10

    plan_2d = ParExt.create_distributed_spectral_plan_2d(
        lmax, mmax, comm; p_l=1, p_m=nprocs, mres=cfg.mres,
    )
    try
        coefficients_safe = ParExt.dist_analysis_distributed_2d(
            cfg, theta_field; plan=plan_2d,
        )
        @test isapprox(
            ParExt.gather_to_full_dense_2d(coefficients_safe),
            reference_coefficients; rtol=1e-10, atol=1e-11,
        )

        coefficients_aligned = ParExt.dist_analysis_distributed_2d(
            cfg, phi_field; plan=plan_2d, assume_aligned=true,
        )
        @test isapprox(
            ParExt.gather_to_full_dense_2d(coefficients_aligned),
            reference_coefficients; rtol=1e-10, atol=1e-11,
        )
        reconstructed_2d = ParExt.dist_synthesis_distributed_2d_optimized(
            cfg, coefficients_aligned; prototype_θφ=phi_field,
        )
        @test distributed_spatial_error(
            reconstructed_2d, reference_spatial, phi_field,
        ) < 1e-10

        if nprocs > 1
            @test_throws ArgumentError ParExt.dist_analysis_distributed_2d(
                cfg, theta_field; plan=plan_2d, assume_aligned=true,
            )
            @test_throws ArgumentError ParExt.dist_synthesis_distributed_2d_optimized(
                cfg, coefficients_safe; prototype_θφ=theta_field,
            )
        end
    finally
        close(plan_2d)
    end

    cfg_robert = create_gauss_config(
        lmax, lmax + 3; mmax, nlon=2mmax + 2, robert_form=true,
    )
    robert_coefficients = analysis(cfg_robert, spatial)
    robert_reference = synthesis(
        cfg_robert, robert_coefficients; real_output=true,
    )
    robert_field = distributed_spatial(
        cfg_robert, spatial; decompose_theta=false,
    )
    robert_plan = ParExt.create_distributed_spectral_plan_2d(
        lmax, mmax, comm; p_l=1, p_m=nprocs,
    )
    try
        robert_distributed = ParExt.create_distributed_spectral_array_2d(
            robert_plan,
        )
        ParExt.scatter_from_dense_2d!(
            robert_distributed, robert_coefficients,
        )
        robert_got = ParExt.dist_synthesis_distributed_2d_optimized(
            cfg_robert, robert_distributed; prototype_θφ=robert_field,
        )
        @test distributed_spatial_error(
            robert_got, robert_reference, robert_field,
        ) < 1e-10
    finally
        close(robert_plan)
    end
end

MPI.Barrier(comm)
rank == 0 && println("plan/config preflight regressions complete")
