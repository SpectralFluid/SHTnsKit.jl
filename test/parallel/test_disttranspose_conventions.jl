# Regression tests for configured coefficient conventions at DistTransposePlan
# boundaries. Run with:
#
#   mpiexec -n 2 julia --project test/parallel/test_disttranspose_conventions.jl

using MPI; MPI.Init()
using SHTnsKit, PencilArrays, PencilFFTs, Test

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)

function scatter_fields!(dest, fields)
    r = PencilArrays.range_local(pencil(dest))
    @assert length(fields) == size(parent(dest), 3)
    for lev in eachindex(fields),
        (ilat, glat) in enumerate(r[2]), (ilon, glon) in enumerate(r[1])
        parent(dest)[ilon, ilat, lev] = fields[lev][glat, glon]
    end
    return dest
end

function scatter_coefficients!(dest, plan, coeffs)
    fill!(parent(dest), 0)
    @assert length(coeffs) == plan.nlev
    for lev in eachindex(coeffs), (mi, m) in enumerate(plan.m_local), l in 0:plan.lmax
        parent(dest)[l + 1, mi, lev] = coeffs[lev][l + 1, m + 1]
    end
    return dest
end

function seed_unused_bins!(dest, plan, value)
    A = parent(dest)
    first_unused = length(plan.m_local) + 1
    if first_unused <= size(A, 2)
        for lev in axes(A, 3), mi in first_unused:size(A, 2), l in axes(A, 1)
            A[l, mi, lev] = value
        end
    end
    return dest
end

function global_coefficient_error(got, plan, refs)
    err = 0.0
    for lev in eachindex(refs), (mi, m) in enumerate(plan.m_local), l in m:plan.lmax
        err = max(err, abs(parent(got)[l + 1, mi, lev] - refs[lev][l + 1, m + 1]))
    end
    return MPI.Allreduce(err, MPI.MAX, comm)
end

function global_spatial_error(got, expected)
    err = isempty(parent(got)) ? 0.0 : maximum(abs.(parent(got) .- parent(expected)))
    return MPI.Allreduce(err, MPI.MAX, comm)
end

function scalar_coefficients(cfg, lev)
    A = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    for m in 0:cfg.mmax, l in m:cfg.lmax
        x = (lev + 0.2m + 0.1) / (l + 1)^2
        A[l + 1, m + 1] = m == 0 ? complex(x) : complex(x, -0.35x)
    end
    return A
end

function vector_coefficients(cfg, lev)
    S = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    T = similar(S); fill!(T, 0)
    for m in 0:cfg.mmax, l in max(1, m):cfg.lmax
        x = (lev + 0.15m + 0.2) / (l + 1)^2
        S[l + 1, m + 1] = m == 0 ? complex(x) : complex(x, 0.4x)
        T[l + 1, m + 1] = m == 0 ? complex(-0.6x) : complex(-0.6x, 0.25x)
    end
    return S, T
end

@testset "DistTransposePlan honors configured conventions" begin
    lmax, nlat, nlon, nlev = 6, 10, 20, 2
    cfg = create_gauss_config(lmax, nlat;
        nlon, norm=:schmidt, real_norm=true, cs_phase=false)
    @test cfg.nlon > 2cfg.mmax + 1

    plan = DistTransposePlan(cfg; comm, nlev, use_rfft=true, with_vector=true)

    Q = [scalar_coefficients(cfg, lev) for lev in 1:nlev]
    ST = [vector_coefficients(cfg, lev) for lev in 1:nlev]
    S = first.(ST)
    T = last.(ST)

    Vr_full = [synthesis(cfg, Q[lev]; real_output=true) for lev in 1:nlev]
    tangential_full = [synthesis_sphtor(cfg, S[lev], T[lev]; real_output=true)
                       for lev in 1:nlev]
    Vt_full = first.(tangential_full)
    Vp_full = last.(tangential_full)

    Vr = scatter_fields!(allocate_spatial(plan), Vr_full)
    Vt = scatter_fields!(allocate_spatial(plan), Vt_full)
    Vp = scatter_fields!(allocate_spatial(plan), Vp_full)

    @testset "scalar analysis and synthesis" begin
        Qgot = allocate_spectral(plan)
        dist_analysis!(plan, Qgot, Vr)
        Qref = [analysis(cfg, field) for field in Vr_full]
        @test global_coefficient_error(Qgot, plan, Qref) < 1e-10

        Qin = scatter_coefficients!(allocate_spectral(plan), plan, Q)
        Vr_got = allocate_spatial(plan)
        dist_synthesis!(plan, Vr_got, Qin)
        @test global_spatial_error(Vr_got, Vr) < 1e-10
    end

    @testset "sphtor analysis and synthesis" begin
        Sgot = allocate_spectral(plan)
        Tgot = allocate_spectral(plan)
        dist_analysis_sphtor!(plan, Sgot, Tgot, Vt, Vp)
        STref = [analysis_sphtor(cfg, Vt_full[lev], Vp_full[lev]) for lev in 1:nlev]
        @test global_coefficient_error(Sgot, plan, first.(STref)) < 1e-10
        @test global_coefficient_error(Tgot, plan, last.(STref)) < 1e-10

        Sin = scatter_coefficients!(allocate_spectral(plan), plan, S)
        Tin = scatter_coefficients!(allocate_spectral(plan), plan, T)
        Vt_got = allocate_spatial(plan)
        Vp_got = allocate_spatial(plan)
        dist_synthesis_sphtor!(plan, Vt_got, Vp_got, Sin, Tin)
        @test global_spatial_error(Vt_got, Vt) < 1e-10
        @test global_spatial_error(Vp_got, Vp) < 1e-10
    end

    @testset "QST delegates preserve the same boundary convention" begin
        Qgot = allocate_spectral(plan)
        Sgot = allocate_spectral(plan)
        Tgot = allocate_spectral(plan)
        dist_analysis_qst!(plan, Qgot, Sgot, Tgot, Vr, Vt, Vp)
        Qref = [analysis(cfg, field) for field in Vr_full]
        STref = [analysis_sphtor(cfg, Vt_full[lev], Vp_full[lev]) for lev in 1:nlev]
        @test global_coefficient_error(Qgot, plan, Qref) < 1e-10
        @test global_coefficient_error(Sgot, plan, first.(STref)) < 1e-10
        @test global_coefficient_error(Tgot, plan, last.(STref)) < 1e-10

        Qin = scatter_coefficients!(allocate_spectral(plan), plan, Q)
        Sin = scatter_coefficients!(allocate_spectral(plan), plan, S)
        Tin = scatter_coefficients!(allocate_spectral(plan), plan, T)
        Vr_got = allocate_spatial(plan)
        Vt_got = allocate_spatial(plan)
        Vp_got = allocate_spatial(plan)
        dist_synthesis_qst!(plan, Vr_got, Vt_got, Vp_got, Qin, Sin, Tin)
        @test global_spatial_error(Vr_got, Vr) < 1e-10
        @test global_spatial_error(Vt_got, Vt) < 1e-10
        @test global_spatial_error(Vp_got, Vp) < 1e-10
    end
end


@testset "configured transpose synthesis is low-allocation and preserves inputs" begin
    # This strongly dealiased layout leaves one of two ranks with no meaningful
    # m columns, while keeping each local spectral allocation large enough that
    # a full-array conversion copy cannot hide beneath the allocation budget.
    lmax, nlat, nlon, nlev = 16, 20, 128, 2
    cfg = create_gauss_config(lmax, nlat;
        nlon, norm=:schmidt, real_norm=true, cs_phase=false)
    plan = DistTransposePlan(cfg; comm, nlev, use_rfft=true, with_vector=true)

    empty_ranks = MPI.Allreduce(isempty(plan.m_local) ? 1 : 0, +, comm)
    @test empty_ranks >= 1

    Q = [scalar_coefficients(cfg, lev) for lev in 1:nlev]
    ST = [vector_coefficients(cfg, lev) for lev in 1:nlev]
    Qin = seed_unused_bins!(
        scatter_coefficients!(allocate_spectral(plan), plan, Q), plan, 3.0 + 4.0im)
    Sin = seed_unused_bins!(
        scatter_coefficients!(allocate_spectral(plan), plan, first.(ST)), plan, -2.0 + 0.5im)
    Tin = seed_unused_bins!(
        scatter_coefficients!(allocate_spectral(plan), plan, last.(ST)), plan, 1.5 - 0.25im)
    Qbefore = copy(parent(Qin))
    Sbefore = copy(parent(Sin))
    Tbefore = copy(parent(Tin))

    Vr = allocate_spatial(plan)
    Vt = allocate_spatial(plan)
    Vp = allocate_spatial(plan)

    # Compile and initialize the normalization cache before measuring.
    dist_synthesis!(plan, Vr, Qin)
    dist_synthesis_sphtor!(plan, Vt, Vp, Sin, Tin)
    dist_synthesis_qst!(plan, Vr, Vt, Vp, Qin, Sin, Tin)

    scalar_local = @allocated dist_synthesis!(plan, Vr, Qin)
    vector_local = @allocated dist_synthesis_sphtor!(plan, Vt, Vp, Sin, Tin)
    qst_local = @allocated dist_synthesis_qst!(plan, Vr, Vt, Vp, Qin, Sin, Tin)
    scalar_alloc = MPI.Allreduce(scalar_local, MPI.MAX, comm)
    vector_alloc = MPI.Allreduce(vector_local, MPI.MAX, comm)
    qst_alloc = MPI.Allreduce(qst_local, MPI.MAX, comm)
    rank == 0 && println(
        "configured transpose synthesis allocations: scalar=$scalar_alloc, " *
        "vector=$vector_alloc, qst=$qst_alloc B/call")

    @test scalar_alloc < 8192
    @test vector_alloc < 8192
    @test qst_alloc < 16384
    @test parent(Qin) == Qbefore
    @test parent(Sin) == Sbefore
    @test parent(Tin) == Tbefore
end

MPI.Finalize()
