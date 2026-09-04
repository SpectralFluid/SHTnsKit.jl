#!/usr/bin/env julia
#
# MPI regression tests for the distributed defects found in the 2026-08 audit.
# Each testset pins one fix; every one of them either crashed, hung, or returned
# silently wrong numbers before.
#
#   1. dist_analysis_packed_cplx: LM_cplx negative-m rule (no (-1)^m) + complex input
#   2. packed storage: accumulation/normalization loops must stride by mres
#   3. sphtor table kernels: closed-form pole limits instead of the guarded 0
#   4. ranks owning zero φ columns (nranks > nlon) must not crash the φ gather
#   5. 2D (θ×φ) pencil: θ-slab reduction must not over-count φ-partners
#   6. 2D pencil with an empty θ-partition: the slab keeper must still be a rank
#      that owns θ rows, or a whole latitude slab drops out of the sum
#   7. sphtor pole limits on the OTF (no-table) branch, not just the table branch
#   8. complex dist_analysis_packed_cplx on a φ-decomposed pencil
#   9. direct complex dist_analysis preserves the gathered element type
#  10. local evaluation and diagnostics honor configured coefficient conventions
#  11. distributed Y rotation converts conventions around Wigner mixing
#  12. l-distributed spectral pencils are rejected before row-wise collectives
#
# Run with: mpiexec -n 8 julia --project test/parallel/test_mpi_audit_fixes.jl

using MPI
MPI.Init()

using Test
using Random
using PencilArrays
using PencilFFTs
using SHTnsKit

const ParExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

root_println(args...) = (rank == 0 && (println(args...); flush(stdout)))

"""Scatter a globally-known matrix into the local block of `pen`."""
function scatter_field(pen::Pencil, F::AbstractMatrix)
    r = PencilArrays.range_local(pen)
    loc = Array{eltype(F)}(undef, length(r[1]), length(r[2]))
    for (jl, jg) in enumerate(r[2]), (il, ig) in enumerate(r[1])
        loc[il, jl] = F[ig, jg]
    end
    return PencilArray(pen, loc)
end

"""Max |local - F| over the rank's block of `pen`, reduced over all ranks."""
function max_local_error(local_block::AbstractMatrix, F::AbstractMatrix, pen::Pencil)
    r = PencilArrays.range_local(pen)
    err = 0.0
    for (jl, jg) in enumerate(r[2]), (il, ig) in enumerate(r[1])
        err = max(err, abs(local_block[il, jl] - F[ig, jg]))
    end
    return MPI.Allreduce(err, max, comm)
end

max_local_error(pa::PencilArray, F::AbstractMatrix) =
    max_local_error(parent(pa), F, pencil(pa))

@testset "MPI audit-fix regressions ($nprocs ranks)" begin

    @testset "PencilArrays 0.19 API" begin
        pen = Pencil((4, 5), (1,), comm)
        a = scatter_field(pen, zeros(4, 5))
        @test ParExt.communicator(a) == PencilArrays.get_comm(a)
        @test ParExt.globalindices(a, 1) == PencilArrays.range_local(pen)[1]
        @test ParExt.globalindices(a, 2) == PencilArrays.range_local(pen)[2]
        @test !isdefined(ParExt, :pencilarray_version_info)
    end

    @testset "single distributed analysis path" begin
        @test !isdefined(ParExt, :_ParallelExtState)
        @test !isdefined(ParExt, :dist_analysis_cache_blocked)
        @test !isdefined(ParExt, :dist_analysis_fused_cache_blocked)
    end

    @testset "cfg-form transforms reject permuted parent storage" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1)
        # These kernels intentionally operate on `parent(A)` for performance.
        # A Pencil permutation changes parent memory order while logical global
        # indices remain (θ,φ), so accepting it silently swaps transform axes.
        pen_perm = Pencil((cfg.nlat, cfg.nlon), (1,), MPI.COMM_SELF;
                          permute=Permutation(2, 1))
        f_perm = PencilArray{Float64}(undef, pen_perm)
        for j in axes(f_perm, 2), i in axes(f_perm, 1)
            f_perm[i, j] = sin(0.2i) * cos(0.3j)
        end
        err_analysis = try
            SHTnsKit.dist_analysis(cfg, f_perm)
            nothing
        catch err
            err
        end
        @test err_analysis isa ArgumentError
        @test occursin("permut", lowercase(sprint(showerror, err_analysis)))

        err_synthesis = try
            SHTnsKit.dist_synthesis(
                cfg, zeros(ComplexF64, lmax + 1, lmax + 1);
                prototype_θφ=f_perm)
            nothing
        catch err
            err
        end
        @test err_synthesis isa ArgumentError
        @test occursin("permut", lowercase(sprint(showerror, err_synthesis)))

        spec_perm = Pencil((lmax + 1, lmax + 1), (2,), MPI.COMM_SELF;
                           permute=Permutation(2, 1))
        A_perm = PencilArray{ComplexF64}(undef, spec_perm)
        fill!(A_perm, 0)
        err_rotation = try
            SHTnsKit.dist_SH_Zrotate(cfg, A_perm, 0.2, similar(A_perm))
            nothing
        catch err
            err
        end
        @test err_rotation isa DimensionMismatch
        @test occursin("unpermuted", lowercase(sprint(showerror, err_rotation)))
        root_println("    [PASS] permuted parent storage rejected explicitly")
    end

    @testset "equivalent pencils have rank-symmetric topology lookup" begin
        nlat, nlon = 4, 5
        pen1 = Pencil((nlat, nlon), (1,), comm)
        pen2 = Pencil((nlat, nlon), (1,), comm)
        @test pen1 !== pen2

        F = zeros(nlat, nlon)
        a1 = scatter_field(pen1, F)
        a2 = scatter_field(pen2, F)
        topo1 = ParExt._pencil_topology(
            a1, comm, size(parent(a1), 1), size(parent(a1), 2), nlat, nlon,
        )
        MPI.Barrier(comm)

        # Equivalent decompositions may legitimately have different object-reuse
        # histories on different ranks. No rank may skip a collective because its
        # local Pencil object happened to be cached.
        selected = rank == 0 ? a1 : a2
        topo2 = ParExt._pencil_topology(
            selected, comm, size(parent(selected), 1), size(parent(selected), 2), nlat, nlon,
        )
        @test topo2 == topo1 == (true, true)
        root_println("    [PASS] equivalent-pencil topology lookup")
    end

    @testset "dist_analysis_packed_cplx LM_cplx layout" begin
        lmax = 5
        nlat, nlon = lmax + 2, 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(20260807)
        pen = Pencil((nlat, nlon), (1,), comm)   # θ decomposition

        # Real field: the −m half is conj(a_{+m}) with NO (-1)^m. The old code's
        # (-1)^m flipped every odd-m coefficient.
        F = randn(rng, nlat, nlon)
        got = SHTnsKit.dist_analysis_packed_cplx(cfg, scatter_field(pen, F))
        ref = SHTnsKit.analysis_packed_cplx(cfg, complex.(F))
        @test isapprox(got, ref; rtol=1e-9, atol=1e-11)

        # Genuinely complex field (independent ±m): the old code fabricated the
        # −m half from the +m half and silently returned a symmetrized spectrum.
        Z = randn(rng, ComplexF64, nlat, nlon)
        gotc = SHTnsKit.dist_analysis_packed_cplx(cfg, scatter_field(pen, Z))
        refc = SHTnsKit.analysis_packed_cplx(cfg, Z)
        @test isapprox(gotc, refc; rtol=1e-9, atol=1e-11)
        @test !isapprox(gotc, got; rtol=1e-3)   # sanity: the two inputs differ

        # mres > 1 has no LM_cplx layout — fail loudly instead of mis-indexing.
        cfg2 = create_gauss_config(lmax, nlat; mmax=lmax, mres=2, nlon=nlon)
        @test_throws ArgumentError SHTnsKit.dist_analysis_packed_cplx(cfg2, scatter_field(pen, F))
        root_println("    [PASS] LM_cplx packed analysis")
    end

    @testset "packed storage strides by mres" begin
        lmax = mmax = 6
        mres = 2
        nlat, nlon = lmax + 2, max(2*mmax + 1, 4)
        # :fourpi also exercises the packed norm-conversion loop, which wrote
        # Alm_local[0] under @inbounds for every m that is not a multiple of mres.
        cfg = create_gauss_config(lmax, nlat; mmax=mmax, mres=mres, nlon=nlon, norm=:fourpi)
        rng = MersenneTwister(4242)
        F = randn(rng, nlat, nlon)
        pen = Pencil((nlat, nlon), (1,), comm)
        fpa = scatter_field(pen, F)

        packed = SHTnsKit.dist_analysis(cfg, fpa; use_packed_storage=true)
        dense = SHTnsKit.dist_analysis(cfg, fpa)
        info = ParExt.create_packed_storage_info(cfg)
        @test length(packed) == info.nlm_packed
        maxerr = 0.0
        for m in 0:mres:mmax, l in m:lmax
            maxerr = max(maxerr, abs(packed[info.lm_to_packed[l+1, m+1]] - dense[l+1, m+1]))
        end
        @test maxerr < 1e-11
        root_println("    [PASS] packed storage with mres=$mres")
    end

    @testset "sphtor pole limits on a pole-inclusive grid" begin
        lmax = 6
        nlat, nlon = lmax + 2, 2*lmax + 1
        cfg = create_regular_config(lmax, nlat; nlon=nlon, include_poles=true,
                                    precompute_plm=true)
        @test cfg.use_plm_tables                       # the table kernels are live
        @test minimum(abs.(1.0 .- abs.(cfg.x))) == 0.0 # grid really includes ±1

        rng = MersenneTwister(99)
        Slm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
        Tlm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
        for m in 0:cfg.mmax, l in max(1, m):lmax
            Slm[l+1, m+1] = randn(rng, ComplexF64)
            Tlm[l+1, m+1] = randn(rng, ComplexF64)
        end
        Slm[:, 1] .= real.(Slm[:, 1]); Tlm[:, 1] .= real.(Tlm[:, 1])

        # Serial reference uses the closed-form pole branch (src/kernels.jl).
        Vt, Vp = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        @test maximum(abs, Vt) > 1e-6                  # non-trivial reference

        pen = Pencil((nlat, nlon), (1,), comm)
        proto = scatter_field(pen, Vt)
        Vt_d, Vp_d = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm;
                                                    prototype_θφ=proto, real_output=true)
        # Before the fix the θ=0 and θ=π rows came back 0 from the table path.
        @test max_local_error(Vt_d, Vt, pen) < 1e-9
        @test max_local_error(Vp_d, Vp, pen) < 1e-9

        # Analysis direction reads the same tables.
        Sref, Tref = SHTnsKit.analysis_sphtor(cfg, Vt, Vp)
        S_d, T_d = SHTnsKit.dist_analysis_sphtor(cfg, scatter_field(pen, Vt),
                                                 scatter_field(pen, Vp))
        @test isapprox(S_d, Sref; rtol=1e-8, atol=1e-10)
        @test isapprox(T_d, Tref; rtol=1e-8, atol=1e-10)
        root_println("    [PASS] sphtor pole limits")
    end

    @testset "rank owning zero φ columns" begin
        lmax = mmax = 1
        nlat, nlon = 3, 3
        if nprocs <= nlon
            root_println("    [SKIP] zero-φ rank needs nprocs > nlon=$nlon")
        else
            cfg = create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon)
            rng = MersenneTwister(11)
            F = randn(rng, nlat, nlon)
            pen = Pencil((nlat, nlon), comm)   # decomposes φ → last ranks own nothing
            fpa = scatter_field(pen, F)
            # Used to BoundsError on the empty rank while its partners blocked in
            # the gather collective, hanging the job.
            alm = SHTnsKit.dist_analysis(cfg, fpa)
            @test isapprox(alm, SHTnsKit.analysis(cfg, F); rtol=1e-9, atol=1e-11)
            root_println("    [PASS] zero-φ rank")
        end
    end

    @testset "2D (θ×φ) pencil reduction" begin
        lmax = 6
        nlat, nlon = lmax + 2, 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        pen2 = try
            Pencil((nlat, nlon), (1, 2), comm)
        catch err
            root_println("    [SKIP] 2D pencil unsupported here: ", err)
            nothing
        end
        if pen2 === nothing
            @test true
        else
            r = PencilArrays.range_local(pen2)
            both_split = MPI.Allreduce(
                (length(r[1]) < nlat && length(r[2]) < nlon) ? 1 : 0, +, comm) == nprocs
            if !both_split
                root_println("    [SKIP] topology did not split both dims")
                @test true
            else
                rng = MersenneTwister(2026)
                F = randn(rng, nlat, nlon)
                fpa = scatter_field(pen2, F)
                # φ-partners of a θ-slab hold identical partials; over-counting
                # them scaled the whole spectrum by the φ-partition factor.
                @test isapprox(SHTnsKit.dist_analysis(cfg, fpa),
                               SHTnsKit.analysis(cfg, F); rtol=1e-9, atol=1e-11)

                Slm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
                Tlm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
                for m in 0:cfg.mmax, l in max(1, m):lmax
                    Slm[l+1, m+1] = randn(rng, ComplexF64)
                    Tlm[l+1, m+1] = randn(rng, ComplexF64)
                end
                Slm[:, 1] .= real.(Slm[:, 1]); Tlm[:, 1] .= real.(Tlm[:, 1])
                Vt, Vp = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
                Sref, Tref = SHTnsKit.analysis_sphtor(cfg, Vt, Vp)
                # This path used to Comm_split on first([]) with no guard at all.
                S_d, T_d = SHTnsKit.dist_analysis_sphtor(cfg, scatter_field(pen2, Vt),
                                                         scatter_field(pen2, Vp))
                @test isapprox(S_d, Sref; rtol=1e-8, atol=1e-10)
                @test isapprox(T_d, Tref; rtol=1e-8, atol=1e-10)
                root_println("    [PASS] 2D pencil reduction")
            end
        end
    end

    @testset "2D pencil with an empty θ-partition" begin
        # More θ-partitions than latitudes, so at least one rank owns zero θ rows.
        # Electing the slab keeper by θ-colour put such a rank in the same
        # Comm_split group as the genuine owner of global θ index 1 (an empty
        # range still reports `first == 1`), and Comm_split orders by global rank
        # — so the empty rank could win group rank 0 and zero the real θ=1 slab
        # out of the reduction, silently dropping a whole latitude band.
        lmax = 2
        nlat, nlon = lmax + 1, 2*lmax + 1
        # An automatic topology will not leave a θ-partition empty, so pick the
        # process grid explicitly: pθ > nlat forces the empty partition, pφ ≥ 2
        # keeps the φ-partner dedup (the code under test) on the critical path.
        pθ = 0
        for p in (nlat + 1):nprocs
            if nprocs % p == 0 && nprocs ÷ p >= 2
                pθ = p
                break
            end
        end
        if pθ == 0
            root_println("    [SKIP] no (pθ>$nlat, pφ≥2) split of $nprocs ranks; needs e.g. 8")
            @test true
        else
            pφ = nprocs ÷ pθ
            pen2 = Pencil(MPITopology(comm, (pθ, pφ)), (nlat, nlon), (1, 2))
            cfg = create_gauss_config(lmax, nlat; nlon=nlon)
            r = PencilArrays.range_local(pen2)
            n_empty_θ = MPI.Allreduce(isempty(r[1]) ? 1 : 0, +, comm)
            @test n_empty_θ > 0     # the scenario really is set up
            rng = MersenneTwister(31337)
            F = randn(rng, nlat, nlon)
            @test isapprox(SHTnsKit.dist_analysis(cfg, scatter_field(pen2, F)),
                           SHTnsKit.analysis(cfg, F); rtol=1e-9, atol=1e-11)
            root_println("    [PASS] empty θ-partition ($(pθ)×$(pφ) grid, $n_empty_θ empty ranks)")
        end
    end

    @testset "sphtor pole limits on the OTF branch" begin
        # precompute_plm=false forces the on-the-fly branch, which computed
        # Y/sinθ as P̄ * (1/sinθ). At an exact pole node 1/sinθ is guarded to 0,
        # so that product is 0 and the entire m=1 contribution vanished from the
        # pole rows — while the table branch next to it had already been fixed.
        lmax = 6
        for precompute in (false, true)
            cfg = create_regular_config(lmax, lmax + 2; nlon=2*lmax + 3,
                                        include_poles=true, precompute_plm=precompute)
            rng = MersenneTwister(4242)
            Slm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
            Tlm = zeros(ComplexF64, lmax+1, cfg.mmax+1)
            for m in 0:cfg.mmax, l in max(1, m):lmax
                Slm[l+1, m+1] = randn(rng, ComplexF64)
                Tlm[l+1, m+1] = randn(rng, ComplexF64)
            end
            Slm[:, 1] .= real.(Slm[:, 1]); Tlm[:, 1] .= real.(Tlm[:, 1])

            Vt_ref, Vp_ref = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
            # Guard against a vacuous pass: the pole rows must carry signal.
            @test maximum(abs, view(Vt_ref, 1, :)) > 1e-6

            pen = Pencil((cfg.nlat, cfg.nlon), (1,), comm)
            proto = PencilArray{Float64}(undef, pen)
            Vt_d, Vp_d = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm;
                                                        prototype_θφ=proto, real_output=true)
            @test max_local_error(Vt_d, Vt_ref, pen) < 1e-10
            @test max_local_error(Vp_d, Vp_ref, pen) < 1e-10
        end
        root_println("    [PASS] sphtor OTF pole limits")
    end

    @testset "complex packed_cplx on a φ-decomposed pencil" begin
        # The complex path routed the field through dist_analysis, whose φ-gather
        # helper packs into a Vector{Float64} — so a ComplexF64 PencilArray threw
        # InexactError on every rank. Only the θ-decomposed layout was covered.
        lmax = 5
        nlat, nlon = lmax + 2, 2*lmax + 2
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(909)
        zref = randn(rng, ComplexF64, nlat, nlon)
        ref = SHTnsKit.analysis_packed_cplx(cfg, zref)
        for (label, pen) in (("φ-split", Pencil((nlat, nlon), comm)),
                             ("θ-split", Pencil((nlat, nlon), (1,), comm)))
            zdist = scatter_field(pen, zref)
            # The direct dense path must preserve the PencilArray element type too;
            # packed_cplx used to hide a Float64-only φ gather by splitting into
            # real and imaginary transforms.
            @test isapprox(SHTnsKit.dist_analysis(cfg, zdist), SHTnsKit.analysis(cfg, zref);
                           rtol=1e-9, atol=1e-11)
            got = SHTnsKit.dist_analysis_packed_cplx(cfg, zdist)
            @test isapprox(got, ref; rtol=1e-9, atol=1e-11)
            root_println("    [PASS] complex packed_cplx on $label pencil")
        end
    end
    @testset "distributed transforms match the configured serial convention" begin
        # These are coefficient equality checks, not only roundtrip checks: a
        # roundtrip closes under either convention and cannot detect a backend
        # that forgot the configured normalization or phase boundary.
        lmax = 5
        nlat, nlon = lmax + 3, 2*lmax + 2
        for (nrm, cs) in ((:orthonormal, true), (:schmidt, true), (:fourpi, false))
            cfg = create_gauss_config(lmax, nlat; nlon=nlon, norm=nrm, cs_phase=cs)
            A0 = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
            for m in 0:cfg.mmax, l in m:lmax
                A0[l+1, m+1] = randn(MersenneTwister(31 + 7l + m), ComplexF64)
            end
            A0[:, 1] .= real.(A0[:, 1])
            F = SHTnsKit.synthesis(cfg, A0; real_output=true)

            pen = Pencil((nlat, nlon), (1,), comm)
            fpa = scatter_field(pen, F)

            # dist_analysis must equal serial analysis coefficient-for-coefficient
            @test isapprox(SHTnsKit.dist_analysis(cfg, fpa), SHTnsKit.analysis(cfg, F);
                           rtol=1e-12, atol=1e-13)

            # dist_synthesis must reproduce the field from the SAME configured alm
            frec = SHTnsKit.dist_synthesis(cfg, SHTnsKit.analysis(cfg, F);
                                           prototype_θφ=fpa, real_output=true)
            @test max_local_error(frec, F, pen) < 1e-10

            # Public distributed coefficient containers use cfg's convention,
            # just like serial analysis. Local evaluation and diagnostics must
            # convert/apply its metric rather than treating values as canonical.
            pen_spec = Pencil((lmax + 1, cfg.mmax + 1), comm)
            A_p = scatter_field(pen_spec, A0)
            iθ, jφ = 3, 4
            cost = cfg.x[iθ]
            phi = 2π * (jφ - 1) / nlon
            @test isapprox(SHTnsKit.dist_SH_to_point(cfg, A_p, cost, phi), F[iθ, jφ];
                           rtol=1e-10, atol=1e-11)
            @test isapprox(SHTnsKit.dist_SH_to_lat(cfg, A_p, cost; nphi=nlon),
                           vec(F[iθ, :]); rtol=1e-10, atol=1e-11)
            @test isapprox(SHTnsKit.energy_scalar(cfg, A_p),
                           SHTnsKit.energy_scalar(cfg, A0); rtol=1e-12, atol=1e-13)
            @test isapprox(SHTnsKit.energy_scalar_l_spectrum(cfg, A_p),
                           SHTnsKit.energy_scalar_l_spectrum(cfg, A0); rtol=1e-12, atol=1e-13)
            @test isapprox(SHTnsKit.energy_scalar_m_spectrum(cfg, A_p),
                           SHTnsKit.energy_scalar_m_spectrum(cfg, A0); rtol=1e-12, atol=1e-13)

            S0 = 0.3 .* A0
            T0 = -0.2im .* A0
            S_p = scatter_field(pen_spec, S0)
            T_p = scatter_field(pen_spec, T0)
            Vr, Vt, Vp = SHTnsKit.synthesis_qst(cfg, A0, S0, T0; real_output=true)
            got_point = SHTnsKit.dist_SHqst_to_point(cfg, A_p, S_p, T_p, cost, phi)
            @test isapprox(collect(got_point), [Vr[iθ, jφ], Vt[iθ, jφ], Vp[iθ, jφ]];
                           rtol=1e-9, atol=1e-10)
            got_lat = SHTnsKit.dist_SHqst_to_lat(cfg, A_p, S_p, T_p, cost; nphi=nlon)
            @test isapprox(collect(got_lat[1]), vec(Vr[iθ, :]); rtol=1e-9, atol=1e-10)
            @test isapprox(collect(got_lat[2]), vec(Vt[iθ, :]); rtol=1e-9, atol=1e-10)
            @test isapprox(collect(got_lat[3]), vec(Vp[iθ, :]); rtol=1e-9, atol=1e-10)
            @test isapprox(SHTnsKit.energy_vector_l_spectrum(cfg, S_p, T_p),
                           SHTnsKit.energy_vector_l_spectrum(cfg, S0, T0); rtol=1e-12, atol=1e-13)
            @test isapprox(SHTnsKit.energy_vector_m_spectrum(cfg, S_p, T_p),
                           SHTnsKit.energy_vector_m_spectrum(cfg, S0, T0); rtol=1e-12, atol=1e-13)
            @test isapprox(SHTnsKit.enstrophy_l_spectrum(cfg, T_p),
                           SHTnsKit.enstrophy_l_spectrum(cfg, T0); rtol=1e-12, atol=1e-13)
            @test isapprox(SHTnsKit.enstrophy_m_spectrum(cfg, T_p),
                           SHTnsKit.enstrophy_m_spectrum(cfg, T0); rtol=1e-12, atol=1e-13)
            root_println("    [PASS] distributed == serial for norm=$nrm cs_phase=$cs")
        end
    end

    @testset "Y rotation convention and supported spectral decomposition" begin
        lmax = mmax = 5
        beta = 0.37
        cfg_can = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1)
        cfg_ext = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1,
                                      norm=:schmidt, cs_phase=false, real_norm=true)
        rng = MersenneTwister(717)
        A_can = zeros(ComplexF64, lmax + 1, mmax + 1)
        for m in 0:mmax, l in m:lmax
            A_can[l + 1, m + 1] = randn(rng, ComplexF64)
        end
        A_can[:, 1] .= real.(A_can[:, 1])
        A_ext = copy(A_can)
        SHTnsKit._externalize_coefficients!(A_ext, cfg_ext)

        # m-only distribution is supported. Rotating two representations of the
        # same field must produce representations of the same rotated field.
        pen_m = Pencil((lmax + 1, mmax + 1), comm)
        R_can = similar(scatter_field(pen_m, A_can))
        R_ext = similar(scatter_field(pen_m, A_ext))
        SHTnsKit.dist_SH_Yrotate(cfg_can, scatter_field(pen_m, A_can), beta, R_can)
        SHTnsKit.dist_SH_Yrotate(cfg_ext, scatter_field(pen_m, A_ext), beta, R_ext)
        scales = SHTnsKit._ensure_norm_scale_matrix!(cfg_ext)
        ranges = PencilArrays.range_local(pen_m)
        local_err = 0.0
        for (jm, gm) in enumerate(ranges[2]), (il, gl) in enumerate(ranges[1])
            l, m = gl - 1, gm - 1
            if l >= m
                local_err = max(local_err,
                    abs(parent(R_can)[il, jm] - scales[l + 1, m + 1] * parent(R_ext)[il, jm]))
            end
        end
        @test MPI.Allreduce(local_err, max, comm) < 1e-10

        # Public rotation signatures accept generic complex pencils and real
        # angles; the optimized truncated-gather kernel must not narrow that
        # contract to ComplexF64/Float64 internally.
        A32_p = scatter_field(pen_m, ComplexF32.(A_can))
        R32_p = similar(A32_p)
        SHTnsKit.dist_SH_Yrotate(cfg_can, A32_p, Float32(beta), R32_p)
        R32_ref = zeros(ComplexF32, size(A_can))
        SHTnsKit.dist_SH_Yrotate(
            cfg_can, ComplexF32.(A_can), Float32(beta), R32_ref)
        @test max_local_error(R32_p, R32_ref) < 2e-6

        # Y rotation performs collectives row-by-row and therefore cannot accept
        # a pencil split across l rows. It must fail on every rank before entering
        # the first collective instead of mixing rows or hanging.
        if nprocs > 1
            pen_l = Pencil((lmax + 1, mmax + 1), (1,), comm)
            A_l = scatter_field(pen_l, A_can)
            @test_throws ArgumentError SHTnsKit.dist_SH_Yrotate(cfg_can, A_l, beta, similar(A_l))
        else
            @test true
        end

        # Output buffers must match before either local @inbounds writes or the
        # first row-wise collective. The old methods trusted R blindly.
        wrong_pen = Pencil((lmax + 2, mmax + 1), (2,), comm)
        wrong_R = scatter_field(wrong_pen, zeros(ComplexF64, lmax + 2, mmax + 1))
        A_p = scatter_field(pen_m, A_can)
        @test_throws DimensionMismatch SHTnsKit.dist_SH_Zrotate(cfg_can, A_p, beta, wrong_R)
        @test_throws DimensionMismatch SHTnsKit.dist_SH_Yrotate(cfg_can, A_p, beta, wrong_R)

        # A Y component mixes all orders and has no representation in an
        # mres-strided coefficient space. Every Y-derived API must explain that
        # restriction before indexing packed storage or entering MPI.
        cfg_stride = create_gauss_config(lmax, lmax + 2;
                                         mmax=mmax, mres=2, nlon=2lmax + 1)
        pen_stride = Pencil((lmax + 1, mmax + 1), (2,), comm)
        A_stride = scatter_field(pen_stride, zeros(ComplexF64, lmax + 1, mmax + 1))
        R_stride = similar(A_stride)
        err_y = try
            SHTnsKit.dist_SH_Yrotate(cfg_stride, A_stride, beta, R_stride)
            nothing
        catch err
            err
        end
        @test err_y isa ArgumentError
        @test occursin("mres==1", sprint(showerror, err_y))

        # Even the communication-free Z kernel produces one logical distributed
        # spectrum. A rank-divergent cfg must be rejected collectively instead
        # of letting different ranks apply different active-order masks.
        if nprocs > 1
            cfg_divergent = rank == nprocs - 1 ? cfg_stride : cfg_can
            @test_throws ArgumentError SHTnsKit.dist_SH_Zrotate(
                cfg_divergent, A_p, beta, similar(A_p))
            @test_throws ArgumentError SHTnsKit.dist_SH_Zrotate(
                cfg_can, A_p,
                rank == nprocs - 1 ? beta + 0.1 : beta,
                similar(A_p))
        end

        q_stride = zeros(ComplexF64, cfg_stride.nlm)
        err_packed = try
            SHTnsKit.dist_SH_Yrotate_packed(
                cfg_stride, q_stride, beta; prototype_lm=A_stride)
            nothing
        catch err
            err
        end
        @test err_packed isa ArgumentError
        @test occursin("mres==1", sprint(showerror, err_packed))
        root_println("    [PASS] Y rotation convention/decomposition contract")
    end

    @testset "Euler rotation matches the serial ZYZ convention" begin
        lmax = mmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1)
        rng = MersenneTwister(20260904)
        A = zeros(ComplexF64, lmax + 1, mmax + 1)
        for m in 0:mmax, l in m:lmax
            A[l + 1, m + 1] = randn(rng, ComplexF64)
        end
        A[:, 1] .= real.(A[:, 1])

        α, β, γ = 0.31, 0.72, -0.43
        q = SHTnsKit.pack_lm(cfg, A)
        qref = similar(q)
        rot = SHTRotation(lmax, mmax; α, β, γ, conv=:ZYZ)
        shtns_rotation_apply_real(rot, q, qref)
        Aref = SHTnsKit.unpack_lm(cfg, qref)

        pen_m = Pencil((lmax + 1, mmax + 1), comm)
        A_p = scatter_field(pen_m, A)
        R_p = similar(A_p)
        SHTnsKit.dist_SH_rotate_euler(cfg, A_p, α, β, γ, R_p)

        @test max_local_error(R_p, Aref) < 1e-10

        qxref = similar(q)
        SH_Xrotate90(cfg, q, qxref)
        Xref = SHTnsKit.unpack_lm(cfg, qxref)
        X_p = similar(A_p)
        SHTnsKit.dist_SH_Xrotate90(cfg, A_p, X_p)
        @test max_local_error(X_p, Xref) < 1e-10
        root_println("    [PASS] distributed Euler rotation matches serial ZYZ")
    end

    @testset "local evaluation ignores orders excluded by mres" begin
        lmax = mmax = 6
        cfg = create_gauss_config(lmax, lmax + 2;
                                  mmax=mmax, mres=2, nlon=2mmax + 1)
        excluded = zeros(ComplexF64, lmax + 1, mmax + 1)
        excluded[4, 2] = 0.8 - 0.35im  # (l,m) = (3,1), absent when mres=2
        pen_m = Pencil((lmax + 1, mmax + 1), comm)
        Q_p = scatter_field(pen_m, excluded)
        S_p = scatter_field(pen_m, 0.4 .* excluded)
        T_p = scatter_field(pen_m, -0.7im .* excluded)
        cost, phi = 0.23, 0.41

        @test abs(SHTnsKit.dist_SH_to_point(cfg, Q_p, cost, phi)) < 1e-14
        @test maximum(abs, SHTnsKit.dist_SH_to_lat(cfg, Q_p, cost; nphi=cfg.nlon)) < 1e-14

        qst_point = SHTnsKit.dist_SHqst_to_point(cfg, Q_p, S_p, T_p, cost, phi)
        @test maximum(abs, qst_point) < 1e-14
        qst_lat = SHTnsKit.dist_SHqst_to_lat(cfg, Q_p, S_p, T_p, cost; nphi=cfg.nlon)
        @test maximum(maximum(abs, component) for component in qst_lat) < 1e-14
        root_println("    [PASS] local evaluation honors mres")
    end

    @testset "one-dimensional distributed spectral storage honors mres" begin
        lmax = mmax = 6
        mres = 2
        plan = ParExt.create_distributed_spectral_plan(lmax, mmax, comm; mres)
        dsa = ParExt.create_distributed_spectral_array(plan)

        dense = fill(99.0 + 7.0im, lmax + 1, mmax + 1)
        expected = zeros(ComplexF64, size(dense))
        for m in 0:mres:mmax, l in m:lmax
            expected[l + 1, m + 1] = complex(10l + m, l - m)
            dense[l + 1, m + 1] = expected[l + 1, m + 1]
        end

        ParExt.scatter_from_dense!(dsa, dense)
        gathered = ParExt.gather_to_dense(dsa)
        @test gathered == expected
        @test all(m % mres == 0 for (_, m) in plan.local_lm_indices)
        @test length(plan.local_packed_indices) == length(unique(plan.local_packed_indices))
        root_println("    [PASS] one-dimensional distributed spectral storage honors mres")
    end

    @testset "cfg-form distributed transforms ignore inactive mres columns" begin
        lmax = mmax = 6
        cfg = create_gauss_config(lmax, lmax + 2;
                                  mmax=mmax, mres=2, nlon=2mmax + 1)
        rng = MersenneTwister(2048)
        F = randn(rng, cfg.nlat, cfg.nlon)
        Vt = randn(rng, cfg.nlat, cfg.nlon)
        Vp = randn(rng, cfg.nlat, cfg.nlon)
        pen = Pencil((cfg.nlat, cfg.nlon), (1,), comm)
        F_p = scatter_field(pen, F)
        Vt_p = scatter_field(pen, Vt)
        Vp_p = scatter_field(pen, Vp)

        Aref = analysis(cfg, F)
        Agot_otf = SHTnsKit.dist_analysis(cfg, F_p; use_tables=false)
        @test isapprox(Agot_otf, Aref; rtol=1e-10, atol=1e-11)

        Sref, Tref = analysis_sphtor(cfg, Vt, Vp)
        Sgot_otf, Tgot_otf = SHTnsKit.dist_analysis_sphtor(
            cfg, Vt_p, Vp_p; use_tables=false)
        @test isapprox(Sgot_otf, Sref; rtol=1e-9, atol=1e-10)
        @test isapprox(Tgot_otf, Tref; rtol=1e-9, atol=1e-10)

        prepare_plm_tables!(cfg)
        @test cfg.use_plm_tables
        Agot_tbl = SHTnsKit.dist_analysis(cfg, F_p; use_tables=true)
        @test isapprox(Agot_tbl, Aref; rtol=1e-10, atol=1e-11)
        Sgot_tbl, Tgot_tbl = SHTnsKit.dist_analysis_sphtor(
            cfg, Vt_p, Vp_p; use_tables=true)
        @test isapprox(Sgot_tbl, Sref; rtol=1e-9, atol=1e-10)
        @test isapprox(Tgot_tbl, Tref; rtol=1e-9, atol=1e-10)

        inactive = zeros(ComplexF64, lmax + 1, mmax + 1)
        inactive[4, 2] = 0.8 - 0.35im  # (l,m) = (3,1)
        scalar_ref = synthesis(cfg, inactive; real_output=true)
        scalar_got = SHTnsKit.dist_synthesis(cfg, inactive;
                                              prototype_θφ=F_p, real_output=true)
        @test max_local_error(scalar_got, scalar_ref, pen) < 1e-12

        vector_ref = synthesis_sphtor(cfg, inactive, 0.6im .* inactive;
                                      real_output=true)
        vector_got = SHTnsKit.dist_synthesis_sphtor(
            cfg, inactive, 0.6im .* inactive; prototype_θφ=F_p, real_output=true)
        @test max_local_error(vector_got[1], vector_ref[1], pen) < 1e-12
        @test max_local_error(vector_got[2], vector_ref[2], pen) < 1e-12
        root_println("    [PASS] cfg-form distributed transforms honor mres")
    end

    @testset "transpose plans honor mres without shifting local FFT slots" begin
        lmax = mmax = 6
        cfg = create_gauss_config(lmax, lmax + 2;
                                  mmax=mmax, mres=2, nlon=2mmax + 1)
        plan = DistTransposePlan(cfg; comm, nlev=1, use_rfft=true, with_vector=true)
        @test all(m % cfg.mres == 0 for m in plan.m_local)

        rng = MersenneTwister(9191)
        F = randn(rng, cfg.nlat, cfg.nlon)
        Vt = randn(rng, cfg.nlat, cfg.nlon)
        Vp = randn(rng, cfg.nlat, cfg.nlon)
        f_p = allocate_spatial(plan)
        vt_p = allocate_spatial(plan)
        vp_p = allocate_spatial(plan)
        spatial_ranges = PencilArrays.range_local(pencil(f_p))
        for (iθ, gθ) in enumerate(spatial_ranges[2]),
            (jφ, gφ) in enumerate(spatial_ranges[1])
            parent(f_p)[jφ, iθ, 1] = F[gθ, gφ]
            parent(vt_p)[jφ, iθ, 1] = Vt[gθ, gφ]
            parent(vp_p)[jφ, iθ, 1] = Vp[gθ, gφ]
        end

        A_p = allocate_spectral(plan)
        S_p = allocate_spectral(plan)
        T_p = allocate_spectral(plan)
        dist_analysis!(plan, A_p, f_p)
        dist_analysis_sphtor!(plan, S_p, T_p, vt_p, vp_p)
        Aref = analysis(cfg, F)
        Sref, Tref = analysis_sphtor(cfg, Vt, Vp)
        spectral_m = collect(PencilArrays.range_local(pencil(A_p))[2]) .- 1
        local_err = 0.0
        for (slot, m) in enumerate(spectral_m), l in 0:lmax
            local_err = max(local_err,
                abs(parent(A_p)[l + 1, slot, 1] - Aref[l + 1, m + 1]),
                abs(parent(S_p)[l + 1, slot, 1] - Sref[l + 1, m + 1]),
                abs(parent(T_p)[l + 1, slot, 1] - Tref[l + 1, m + 1]))
        end
        @test MPI.Allreduce(local_err, max, comm) < 1e-10

        # Seed only the forbidden global m=1 bin.  Physical FFT slots are not
        # compressed by mres, so the implementation needs an explicit slot map.
        A_bad = allocate_spectral(plan)
        S_bad = allocate_spectral(plan)
        T_bad = allocate_spectral(plan)
        fill!(parent(A_bad), 0); fill!(parent(S_bad), 0); fill!(parent(T_bad), 0)
        for (slot, m) in enumerate(spectral_m)
            if m == 1
                parent(A_bad)[4, slot, 1] = 0.8 - 0.35im
                parent(S_bad)[4, slot, 1] = -0.2 + 0.6im
                parent(T_bad)[4, slot, 1] = 0.4 + 0.1im
            end
        end
        f_bad = allocate_spatial(plan)
        vt_bad = allocate_spatial(plan)
        vp_bad = allocate_spatial(plan)
        dist_synthesis!(plan, f_bad, A_bad)
        dist_synthesis_sphtor!(plan, vt_bad, vp_bad, S_bad, T_bad)
        bad_local = max(maximum(abs, parent(f_bad)),
                        maximum(abs, parent(vt_bad)), maximum(abs, parent(vp_bad)))
        @test MPI.Allreduce(bad_local, max, comm) < 1e-12
        root_println("    [PASS] transpose plans honor mres")
    end

    @testset "distributed spectral operators preserve the triangular domain" begin
        lmax = mmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2mmax + 1)
        rng = MersenneTwister(3331)
        A = randn(rng, ComplexF64, lmax + 1, mmax + 1)
        pen = Pencil((lmax + 1, mmax + 1), (2,), comm)

        # The dense helpers define the contract for entries outside l >= m:
        # Laplacian leaves them untouched; the out-of-place tridiagonal operator
        # zeroes them. Distributed kernels must not manufacture coefficients
        # below the spherical-harmonic triangle.
        lap_ref = copy(A)
        SHTnsKit.dist_apply_laplacian!(cfg, lap_ref)
        lap_p = scatter_field(pen, A)
        SHTnsKit.dist_apply_laplacian!(cfg, lap_p)
        @test max_local_error(lap_p, lap_ref, pen) < 1e-12

        mx = zeros(Float64, 2cfg.nlm)
        mul_ct_matrix(cfg, mx)
        op_ref = similar(A)
        SHTnsKit.dist_SH_mul_mx!(cfg, mx, A, op_ref)
        A_p = scatter_field(pen, A)
        op_p = PencilArray{ComplexF64}(undef, pen)
        SHTnsKit.dist_SH_mul_mx!(cfg, mx, A_p, op_p)
        @test max_local_error(op_p, op_ref, pen) < 1e-12

        # The out-of-place operator zeroes its destination before traversing the
        # input.  Aliasing the two operands must therefore be rejected explicitly
        # rather than silently returning an all-zero spectrum.
        alias_p = scatter_field(pen, A)
        @test_throws ArgumentError SHTnsKit.dist_SH_mul_mx!(
            cfg, mx, alias_p, alias_p)
        equivalent_pen = Pencil((lmax + 1, mmax + 1), (2,), comm)
        shared_parent = PencilArray(equivalent_pen, parent(alias_p))
        @test alias_p !== shared_parent
        @test_throws ArgumentError SHTnsKit.dist_SH_mul_mx!(
            cfg, mx, alias_p, shared_parent)

        @test_throws DimensionMismatch SHTnsKit.dist_SH_mul_mx!(
            cfg, mx[1:end-1], A_p, op_p)
        root_println("    [PASS] distributed spectral operators preserve valid storage")
    end

    @testset "Robert form does not alter scalar synthesis" begin
        lmax = mmax = 6
        cfg = create_gauss_config(lmax, lmax + 2;
                                  mmax=mmax, nlon=2mmax + 1, robert_form=true)
        rng = MersenneTwister(1717)
        A = zeros(ComplexF64, lmax + 1, mmax + 1)
        for m in 0:mmax, l in m:lmax
            A[l + 1, m + 1] = randn(rng, ComplexF64)
        end
        A[:, 1] .= real.(A[:, 1])
        ref = synthesis(cfg, A; real_output=true)

        pen_θ = Pencil((cfg.nlat, cfg.nlon), (1,), comm)
        proto_θ = scatter_field(pen_θ, ref)
        got = SHTnsKit.dist_synthesis(cfg, A; prototype_θφ=proto_θ, real_output=true)
        @test max_local_error(got, ref, pen_θ) < 1e-10

        # The optimized 2-D spectral-storage path had a second copy of the same
        # scalar-only Robert scaling.  A φ decomposition gives every m-group the
        # identical latitude range required by its m-communicator reduction.
        pen_φ = Pencil((cfg.nlat, cfg.nlon), comm)
        proto_φ = scatter_field(pen_φ, ref)
        plan2d = ParExt.create_distributed_spectral_plan_2d(
            lmax, mmax, comm; p_l=1, p_m=nprocs)
        dsa2d = ParExt.create_distributed_spectral_array_2d(plan2d)
        ParExt.scatter_from_dense_2d!(dsa2d, A)
        got2d = ParExt.dist_synthesis_distributed_2d_optimized(
            cfg, dsa2d; prototype_θφ=proto_φ, real_output=true)
        @test max_local_error(got2d, ref, pen_φ) < 1e-10
        close(plan2d)
        root_println("    [PASS] Robert form remains vector-only")
    end

    @testset "2-D alignment validation rejects duplicated latitude slabs" begin
        if nprocs >= 4 && iseven(nprocs)
            lmax = mmax = 6
            cfg = create_gauss_config(lmax, lmax + 2; nlon=2mmax + 1)
            # Each latitude slab has multiple φ partners.  With p_m=1 the old
            # validator compared singleton m-communicators and returned true,
            # although an l_comm reduction would count every slab pφ times.
            pθ, pφ = 2, nprocs ÷ 2
            topo = MPITopology(comm, (pθ, pφ))
            pen2 = Pencil(topo, (cfg.nlat, cfg.nlon), (1, 2))
            proto = scatter_field(pen2, zeros(cfg.nlat, cfg.nlon))
            plan2d = ParExt.create_distributed_spectral_plan_2d(
                lmax, mmax, comm; p_l=nprocs, p_m=1)

            aligned, message = ParExt.validate_2d_distribution_alignment(plan2d, proto)
            @test !aligned
            @test occursin("not", lowercase(message))
            close(plan2d)

            # Replicated full-latitude data are also valid: the aligned analysis
            # detects that θ is not distributed and deliberately skips l_comm
            # reduction.  The validator must distinguish this from duplicated
            # partial slabs.
            pen_φ = Pencil((cfg.nlat, cfg.nlon), comm)
            proto_φ = scatter_field(pen_φ, zeros(cfg.nlat, cfg.nlon))
            plan_replicated = ParExt.create_distributed_spectral_plan_2d(
                lmax, mmax, comm; p_l=2, p_m=nprocs ÷ 2)
            aligned_replicated, _ = ParExt.validate_2d_distribution_alignment(
                plan_replicated, proto_φ)
            @test aligned_replicated
            close(plan_replicated)
            root_println("    [PASS] duplicated θ slabs rejected by alignment validator")
        else
            @test true
            root_println("    [SKIP] duplicated-slab alignment case needs an even rank count ≥ 4")
        end
    end
end

MPI.Barrier(comm)
root_println("\nAll MPI audit-fix regression tests PASSED")
MPI.Finalize()
