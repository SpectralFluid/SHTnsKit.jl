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

@testset "distributed plan preflight ($nprocs ranks)" begin
    lmax = 4
    nlat = 7
    nlon = 11
    cfg = create_gauss_config(lmax, nlat; nlon)
    field = zero_spatial(cfg)

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
        @test_throws DimensionMismatch SHTnsKit.dist_synthesis_qst(
            cfg, spectral, spectral, alternate; prototype_θφ=field,
        )
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
        end
    end
end

MPI.Barrier(comm)
rank == 0 && println("plan/config preflight regressions complete")
