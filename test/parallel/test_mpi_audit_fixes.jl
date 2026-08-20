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
#
# Run with: mpiexec -n 4 julia --project test/parallel/test_mpi_audit_fixes.jl

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
            got = SHTnsKit.dist_analysis_packed_cplx(cfg, scatter_field(pen, zref))
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
            root_println("    [PASS] distributed == serial for norm=$nrm cs_phase=$cs")
        end
    end
end

MPI.Barrier(comm)
root_println("\nAll MPI audit-fix regression tests PASSED")
MPI.Finalize()
