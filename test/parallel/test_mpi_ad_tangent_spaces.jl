#!/usr/bin/env julia

# ChainRules tangent-space contract for distributed vector analysis.
# Run with: mpiexec -n 2 julia --project test/parallel/test_mpi_ad_tangent_spaces.jl

using MPI
MPI.Init()

using ChainRulesCore: ProjectTo, ZeroTangent, rrule
using PencilArrays
using PencilFFTs
using Random
using SHTnsKit
using Test

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
const ParADExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelADExt)

function scatter_field(pen::Pencil, field::AbstractMatrix)
    ranges = PencilArrays.range_local(pen)
    local_field = Array{eltype(field)}(undef, length(ranges[1]), length(ranges[2]))
    @inbounds for (jlocal, jglobal) in enumerate(ranges[2])
        for (ilocal, iglobal) in enumerate(ranges[1])
            local_field[ilocal, jlocal] = field[iglobal, jglobal]
        end
    end
    return PencilArray(pen, local_field)
end


@testset "shared adjoints used by distributed transforms" begin
    lmax = mmax = 6
    cfg_stride = create_gauss_config(lmax, lmax + 2;
                                     mmax=mmax, mres=2, nlon=2mmax + 1)
    inactive = zeros(ComplexF64, lmax + 1, mmax + 1)
    inactive[4, 2] = 0.8 - 0.35im              # (l,m)=(3,1), excluded
    active = zeros(ComplexF64, lmax + 1, mmax + 1)
    active[4, 3] = 0.8 - 0.35im                # (l,m)=(3,2), retained
    zcoeff = zeros(ComplexF64, lmax + 1, mmax + 1)
    gridbar = reshape(sin.(1:(cfg_stride.nlat * cfg_stride.nlon)),
                      cfg_stride.nlat, cfg_stride.nlon)

    @test all(iszero, SHTnsKit._adjoint_analysis(cfg_stride, inactive))
    @test maximum(abs, SHTnsKit._adjoint_analysis(cfg_stride, active)) > 1e-8
    Abar = SHTnsKit._adjoint_synthesis(cfg_stride, gridbar)
    @test all(iszero, @view Abar[:, 2])
    @test maximum(abs, @view Abar[:, 3]) > 1e-8

    vtbar, vpbar = SHTnsKit._adjoint_analysis_sphtor(
        cfg_stride, inactive, zcoeff)
    @test all(iszero, vtbar)
    @test all(iszero, vpbar)
    Sbar, Tbar = SHTnsKit._adjoint_synthesis_sphtor(
        cfg_stride, gridbar, 0.7 .* gridbar)
    @test all(iszero, @view Sbar[:, 2])
    @test all(iszero, @view Tbar[:, 2])

    # Scalar analysis is complex-linear for complex spatial input; its shared
    # adjoint must retain the imaginary cotangent until a real public primal
    # explicitly projects it away.
    cfg = create_gauss_config(4, 6; nlon=9)
    rng = MersenneTwister(4141)
    Cbar = randn(rng, ComplexF64, 5, 5)
    h = randn(rng, ComplexF64, 6, 9)
    raw = SHTnsKit._adjoint_analysis(cfg, Cbar)
    @test eltype(raw) === ComplexF64
    @test maximum(abs, imag.(raw)) > 1e-8
    @test real(sum(conj.(raw) .* h)) ≈
          real(sum(conj.(Cbar) .* analysis(cfg, h))) rtol=1e-10 atol=1e-11
end


@testset "Robert-form vector adjoints" begin
    lmax = mmax = 4
    cfg = create_gauss_config(lmax, lmax + 2;
                              mmax=mmax, nlon=2mmax + 1, robert_form=true)
    rng = MersenneTwister(5151)
    hS = zeros(ComplexF64, lmax + 1, mmax + 1)
    hT = zeros(ComplexF64, size(hS))
    CbarS = zeros(ComplexF64, size(hS))
    CbarT = zeros(ComplexF64, size(hS))
    for m in 0:mmax, l in max(1, m):lmax
        hS[l + 1, m + 1] = randn(rng, ComplexF64)
        hT[l + 1, m + 1] = randn(rng, ComplexF64)
        CbarS[l + 1, m + 1] = randn(rng, ComplexF64)
        CbarT[l + 1, m + 1] = randn(rng, ComplexF64)
    end
    hS[:, 1] .= real.(hS[:, 1]); hT[:, 1] .= real.(hT[:, 1])
    Vtbar = randn(rng, cfg.nlat, cfg.nlon)
    Vpbar = randn(rng, cfg.nlat, cfg.nlon)

    Vt, Vp = synthesis_sphtor(cfg, hS, hT; real_output=true)
    Sbar, Tbar = SHTnsKit._adjoint_synthesis_sphtor(cfg, Vtbar, Vpbar)
    lhs_syn = sum(Vtbar .* Vt) + sum(Vpbar .* Vp)
    rhs_syn = real(sum(conj.(Sbar) .* hS) + sum(conj.(Tbar) .* hT))
    @test rhs_syn ≈ lhs_syn rtol=1e-9 atol=1e-10

    hVt = randn(rng, cfg.nlat, cfg.nlon)
    hVp = randn(rng, cfg.nlat, cfg.nlon)
    Sh, Th = analysis_sphtor(cfg, hVt, hVp)
    rawVt, rawVp = SHTnsKit._adjoint_analysis_sphtor(cfg, CbarS, CbarT)
    lhs_ana = real(sum(conj.(CbarS) .* Sh) + sum(conj.(CbarT) .* Th))
    rhs_ana = real(sum(conj.(rawVt) .* hVt) + sum(conj.(rawVp) .* hVp))
    @test rhs_ana ≈ lhs_ana rtol=1e-9 atol=1e-10
end


@testset "distributed scalar AD tangent contracts ($nprocs ranks)" begin
    lmax = 4
    cfg = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1)
    rng = MersenneTwister(6161)
    pen = Pencil((cfg.nlat, cfg.nlon), comm)
    f_real = scatter_field(pen, randn(rng, cfg.nlat, cfg.nlon))
    f_complex = scatter_field(pen, randn(rng, ComplexF64, cfg.nlat, cfg.nlon))
    Cbar = randn(rng, ComplexF64, lmax + 1, lmax + 1)
    ranges = PencilArrays.range_local(pen)
    theta_rows = collect(ranges[1])
    phi_rows = collect(ranges[2])
    phi_window = length(phi_rows) == cfg.nlon ? nothing :
                 (isempty(phi_rows) ? (1:0) : first(phi_rows):last(phi_rows))

    _, pb_complex = rrule(SHTnsKit.dist_analysis, cfg, f_complex)
    _, _, fbar_complex = pb_complex(Cbar)
    raw = SHTnsKit._adjoint_analysis(
        cfg, Cbar; θ_globals=theta_rows, φ_window=phi_window)
    @test fbar_complex isa PencilArray
    @test eltype(fbar_complex) === ComplexF64
    @test parent(fbar_complex) ≈ raw

    packed, pb_packed = rrule(
        SHTnsKit.dist_analysis, cfg, f_real; use_packed_storage=true)
    packedbar = randn(rng, ComplexF64, length(packed))
    _, _, fbar_packed = pb_packed(packedbar)
    packed_expected = SHTnsKit._adjoint_analysis(
        cfg, SHTnsKit.unpack_lm(cfg, packedbar);
        θ_globals=theta_rows, φ_window=phi_window)
    @test fbar_packed isa PencilArray
    @test parent(fbar_packed) ≈ ProjectTo(parent(f_real))(packed_expected)
    _, _, fbar_zero = pb_packed(ZeroTangent())
    @test all(iszero, parent(fbar_zero))

    # The output of distributed analysis is one logical replicated value. A
    # rank-varying cotangent is ambiguous and must be rejected collectively.
    @test_throws ArgumentError pb_complex((rank + 1) .* Cbar)
end


@testset "distributed synthesis preserves spectral pencil tangents ($nprocs ranks)" begin
    lmax = mmax = 4
    cfg = create_gauss_config(lmax, lmax + 2; nlon=2mmax + 1)
    rng = MersenneTwister(7171)
    spatial_pen = Pencil((cfg.nlat, cfg.nlon), (1,), comm)
    prototype = scatter_field(spatial_pen, zeros(cfg.nlat, cfg.nlon))
    spectral_pen = Pencil((lmax + 1, mmax + 1), (2,), comm)

    A = zeros(ComplexF32, lmax + 1, mmax + 1)
    S = similar(A)
    T = zeros(ComplexF64, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        A[l + 1, m + 1] = randn(rng, ComplexF32)
        S[l + 1, m + 1] = randn(rng, ComplexF32)
        T[l + 1, m + 1] = randn(rng, ComplexF64)
    end
    A[:, 1] .= real.(A[:, 1])
    S[:, 1] .= real.(S[:, 1])
    T[:, 1] .= real.(T[:, 1])
    A_p = scatter_field(spectral_pen, A)
    S_p = scatter_field(spectral_pen, S)
    T_p = scatter_field(spectral_pen, T)
    θrows = collect(PencilArrays.range_local(spatial_pen)[1])

    y, pb = rrule(SHTnsKit.dist_synthesis, cfg, A_p;
                  prototype_θφ=prototype, real_output=true)
    ybar = randn(rng, size(y))
    _, _, Abar = pb(ybar)
    dense_Abar = MPI.Allreduce(
        SHTnsKit._adjoint_synthesis(cfg, ybar; θ_globals=θrows), +, comm)
    @test Abar isa PencilArray
    @test pencil(Abar) === pencil(A_p)
    @test eltype(Abar) === ComplexF32
    @test parent(Abar) ≈ parent(scatter_field(spectral_pen, ComplexF32.(dense_Abar)))

    _, pb_dense = rrule(SHTnsKit.dist_synthesis, cfg, A;
                        prototype_θφ=prototype, real_output=true)
    _, _, dense_input_bar = pb_dense(ybar)
    @test eltype(dense_input_bar) === ComplexF32
    _, _, dense_zero_bar = pb_dense(ZeroTangent())
    @test eltype(dense_zero_bar) === ComplexF32
    @test all(iszero, dense_zero_bar)

    (Vt, Vp), pbv = rrule(SHTnsKit.dist_synthesis_sphtor, cfg, S_p, T_p;
                           prototype_θφ=prototype, real_output=true)
    Vtbar = randn(rng, size(Vt))
    Vpbar = randn(rng, size(Vp))
    _, _, Sbar, Tbar = pbv((Vtbar, Vpbar))
    dense_Sbar_local, dense_Tbar_local = SHTnsKit._adjoint_synthesis_sphtor(
        cfg, Vtbar, Vpbar; θ_globals=θrows)
    dense_Sbar = MPI.Allreduce(dense_Sbar_local, +, comm)
    dense_Tbar = MPI.Allreduce(dense_Tbar_local, +, comm)
    @test Sbar isa PencilArray
    @test Tbar isa PencilArray
    @test pencil(Sbar) === pencil(S_p)
    @test pencil(Tbar) === pencil(T_p)
    @test eltype(Sbar) === ComplexF32
    @test eltype(Tbar) === ComplexF64
    @test parent(Sbar) ≈ parent(scatter_field(spectral_pen, ComplexF32.(dense_Sbar)))
    @test parent(Tbar) ≈ parent(scatter_field(spectral_pen, dense_Tbar))

    _, pbv_dense = rrule(SHTnsKit.dist_synthesis_sphtor, cfg, S, T;
                         prototype_θφ=prototype, real_output=true)
    _, _, dense_S_input_bar, dense_T_input_bar = pbv_dense((Vtbar, Vpbar))
    @test eltype(dense_S_input_bar) === ComplexF32
    @test eltype(dense_T_input_bar) === ComplexF64

    # Spectral and spatial communicator groups must agree before either primal
    # or pullback enters a collective.
    self_pen = Pencil((lmax + 1, mmax + 1), (2,), MPI.COMM_SELF)
    A_self = scatter_field(self_pen, A)
    @test_throws ArgumentError rrule(
        SHTnsKit.dist_synthesis, cfg, A_self; prototype_θφ=prototype)
    if nprocs > 1
        rank_asymmetric = rank == nprocs - 1 ? A_self : A_p
        @test_throws ArgumentError ParADExt._require_ad_communicator_match(
            rank_asymmetric, prototype)
    end
end

@testset "distributed analysis_sphtor tangent spaces ($nprocs ranks)" begin
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    pen = Pencil((nlat, nlon), comm)
    rng = MersenneTwister(20260904)

    Vt_real = scatter_field(pen, randn(rng, nlat, nlon))
    Vp_real = scatter_field(pen, randn(rng, nlat, nlon))
    Vt_complex = scatter_field(pen, randn(rng, ComplexF64, nlat, nlon))
    Vp_complex = scatter_field(pen, randn(rng, ComplexF64, nlat, nlon))
    Sbar = randn(rng, ComplexF64, lmax + 1, lmax + 1)
    Tbar = randn(rng, ComplexF64, lmax + 1, lmax + 1)

    ranges = PencilArrays.range_local(pen)
    theta_rows = collect(ranges[1])
    phi_rows = collect(ranges[2])
    phi_window = length(phi_rows) == nlon ? nothing :
                 (isempty(phi_rows) ? (1:0) : first(phi_rows):last(phi_rows))
    raw_Vtbar, raw_Vpbar = SHTnsKit._adjoint_analysis_sphtor(
        cfg, Sbar, Tbar; θ_globals=theta_rows, φ_window=phi_window)

    _, pullback_real = rrule(SHTnsKit.dist_analysis_sphtor, cfg, Vt_real, Vp_real)
    _, _, Vtbar_real, Vpbar_real = pullback_real((Sbar, Tbar))
    @test Vtbar_real isa PencilArray
    @test Vpbar_real isa PencilArray
    @test eltype(Vtbar_real) <: Real
    @test eltype(Vpbar_real) <: Real
    @test parent(Vtbar_real) ≈ real.(raw_Vtbar)
    @test parent(Vpbar_real) ≈ real.(raw_Vpbar)

    _, pullback_complex = rrule(
        SHTnsKit.dist_analysis_sphtor, cfg, Vt_complex, Vp_complex)
    _, _, Vtbar_complex, Vpbar_complex = pullback_complex((Sbar, Tbar))
    @test eltype(Vtbar_complex) <: Complex
    @test eltype(Vpbar_complex) <: Complex
    @test parent(Vtbar_complex) ≈ raw_Vtbar
    @test parent(Vpbar_complex) ≈ raw_Vpbar

    # Each component is projected independently when primal eltypes differ.
    _, pullback_mixed = rrule(
        SHTnsKit.dist_analysis_sphtor, cfg, Vt_real, Vp_complex)
    _, _, Vtbar_mixed, Vpbar_mixed = pullback_mixed((Sbar, Tbar))
    @test eltype(Vtbar_mixed) <: Real
    @test eltype(Vpbar_mixed) <: Complex

    @test_throws ArgumentError pullback_complex(((rank + 1) .* Sbar, Tbar))
end

MPI.Barrier(comm)
rank == 0 && println("Distributed analysis_sphtor tangent-space tests passed.")
MPI.Finalize()
