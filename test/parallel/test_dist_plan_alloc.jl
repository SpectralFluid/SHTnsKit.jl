# DistAnalysisPlan in-place analysis: correctness + per-call allocation budget.
# Run with:
#   mpiexec -n 1 julia --project test/parallel/test_dist_plan_alloc.jl
#   mpiexec -n 2 julia --project test/parallel/test_dist_plan_alloc.jl
using MPI; MPI.Init()
using SHTnsKit, PencilArrays, PencilFFTs, Test
using Random

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const ParExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)

function band_limited_field(cfg, pen, ::Type{T}=Float64) where {T<:AbstractFloat}
    lmax, mmax = cfg.lmax, cfg.mmax
    rng = MersenneTwister(7)
    alm = zeros(Complex{T}, lmax + 1, mmax + 1)
    for m in 0:mmax, l in m:lmax
        alm[l+1, m+1] = Complex{T}(randn(rng), randn(rng))
    end
    alm[:, 1] .= real.(alm[:, 1])
    MPI.Bcast!(alm, 0, comm)
    f_full = T.(SHTnsKit.synthesis(cfg, alm; real_output=true))
    ranges = PencilArrays.range_local(pen)
    flocal = zeros(T, PencilArrays.size_local(pen)...)
    for (i_local, i_global) in enumerate(ranges[1])
        for (j_local, j_global) in enumerate(ranges[2])
            flocal[i_local, j_local] = f_full[i_global, j_global]
        end
    end
    return PencilArray(pen, flocal), f_full
end

@testset "DistAnalysisPlan θ-decomposed (scaling layout)" begin
    lmax = 24; nlat = 32; nlon = 72
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    pen = Pencil((nlat, nlon), (1,), comm)   # decompose θ — the scaling layout
    for T in (Float32, Float64), use_rfft in (false, true)
        f_pa, f_full = band_limited_field(cfg, pen, T)
        a_ref = SHTnsKit.analysis(cfg, f_full)
        plan = ParExt.DistAnalysisPlan(cfg, f_pa; use_rfft)
        Alm_out = zeros(Complex{T}, lmax + 1, cfg.mmax + 1)
        SHTnsKit.dist_analysis!(plan, Alm_out, f_pa)
        tolerance = T === Float32 ? 3f-4 : 1e-10
        @test maximum(abs.(Alm_out .- a_ref)) < tolerance

        # match the cfg-form distributed result too
        Alm_direct = SHTnsKit.dist_analysis(cfg, f_pa; use_rfft)
        @test maximum(abs.(Alm_out .- Alm_direct)) < tolerance

        # Allocation budget: plan owns FFT buffer, Legendre scratch, work
        # matrix, index/weight caches, and the cached reduction subcomm —
        # per-call allocations must stay far below the ~30 KB the unplanned
        # path burns (Fθm + Alm zeros + collects + Comm_split).
        SHTnsKit.dist_analysis!(plan, Alm_out, f_pa)  # warmup
        a = @allocated SHTnsKit.dist_analysis!(plan, Alm_out, f_pa)
        rank == 0 && println(
            "dist_analysis! ($T, use_rfft=$use_rfft): $a B/call",
        )
        @test a < 8192

        # The ordinary same-name API is the public plan surface for Pencil
        # input and must retain the established warmed allocation contract.
        SHTnsKit.analysis!(plan, Alm_out, f_pa)
        a_same_name = @allocated SHTnsKit.analysis!(plan, Alm_out, f_pa)
        rank == 0 && println(
            "analysis! distributed ($T, use_rfft=$use_rfft): " *
            "$a_same_name B/call",
        )
        @test a_same_name < 8192
    end
end

@testset "DistSphtorPlan θ-decomposed analysis" begin
    lmax = 24; nlat = 32; nlon = 72
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    pen = Pencil((nlat, nlon), (1,), comm)

    rng = MersenneTwister(11)
    S0 = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
    T0 = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
    for m in 0:cfg.mmax, l in max(1, m):lmax
        S0[l+1, m+1] = randn(rng) + im * randn(rng)
        T0[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    S0[:, 1] .= real.(S0[:, 1]); T0[:, 1] .= real.(T0[:, 1])
    MPI.Bcast!(S0, 0, comm); MPI.Bcast!(T0, 0, comm)
    Vt_full, Vp_full = SHTnsKit.synthesis_sphtor(cfg, S0, T0; real_output=true)

    ranges = PencilArrays.range_local(pen)
    Vt_loc = zeros(Float64, PencilArrays.size_local(pen)...)
    Vp_loc = zeros(Float64, PencilArrays.size_local(pen)...)
    for (i_local, i_global) in enumerate(ranges[1]), (j_local, j_global) in enumerate(ranges[2])
        Vt_loc[i_local, j_local] = Vt_full[i_global, j_global]
        Vp_loc[i_local, j_local] = Vp_full[i_global, j_global]
    end
    Vt_pa = PencilArray(pen, Vt_loc)
    Vp_pa = PencilArray(pen, Vp_loc)

    S_ref, T_ref = SHTnsKit.analysis_sphtor(cfg, Vt_full, Vp_full)

    for use_rfft in (false, true)
        plan = ParExt.DistSphtorPlan(cfg, Vt_pa; use_rfft)
        Slm_out = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        Tlm_out = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        SHTnsKit.dist_analysis_sphtor!(plan, Slm_out, Tlm_out, Vt_pa, Vp_pa)
        @test maximum(abs.(Slm_out .- S_ref)) < 1e-10
        @test maximum(abs.(Tlm_out .- T_ref)) < 1e-10

        S_direct, T_direct = SHTnsKit.dist_analysis_sphtor(cfg, Vt_pa, Vp_pa; use_rfft)
        @test maximum(abs.(Slm_out .- S_direct)) < 1e-12
        @test maximum(abs.(Tlm_out .- T_direct)) < 1e-12

        SHTnsKit.dist_analysis_sphtor!(plan, Slm_out, Tlm_out, Vt_pa, Vp_pa)  # warmup
        a = @allocated SHTnsKit.dist_analysis_sphtor!(plan, Slm_out, Tlm_out, Vt_pa, Vp_pa)
        rank == 0 && println("dist_analysis_sphtor! (use_rfft=$use_rfft): $a B/call")
        @test a < 8192
    end

    # cfg-form (unplanned) sphtor analysis: legitimately allocates its outputs
    # and FFT buffers, but must not box through the accumulation loop
    # (the old inline loop burned ~2.9 MB/call).
    SHTnsKit.dist_analysis_sphtor(cfg, Vt_pa, Vp_pa)  # warmup
    a_cfg = @allocated SHTnsKit.dist_analysis_sphtor(cfg, Vt_pa, Vp_pa)
    rank == 0 && println("dist_analysis_sphtor cfg-form: $a_cfg B/call")
    @test a_cfg < 300_000

    # Precomputed-tables path must agree with OTF and stay in budget
    cfg_tbl = create_gauss_config(lmax, nlat; nlon=nlon)
    SHTnsKit.prepare_plm_tables!(cfg_tbl)
    plan_tbl = ParExt.DistSphtorPlan(cfg_tbl, Vt_pa)
    S_tbl = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
    T_tbl = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
    SHTnsKit.dist_analysis_sphtor!(plan_tbl, S_tbl, T_tbl, Vt_pa, Vp_pa)
    @test maximum(abs.(S_tbl .- S_ref)) < 1e-10
    @test maximum(abs.(T_tbl .- T_ref)) < 1e-10
    SHTnsKit.dist_analysis_sphtor!(plan_tbl, S_tbl, T_tbl, Vt_pa, Vp_pa)  # warmup
    a_tbl = @allocated SHTnsKit.dist_analysis_sphtor!(plan_tbl, S_tbl, T_tbl, Vt_pa, Vp_pa)
    rank == 0 && println("dist_analysis_sphtor! (tables): $a_tbl B/call")
    @test a_tbl < 8192

    # QST plan composes the scalar + sphtor planned paths
    Vr_pa, _ = band_limited_field(cfg, pen)
    Qr_ref = SHTnsKit.dist_analysis(cfg, Vr_pa)
    for use_rfft in (false, true)
        qplan = ParExt.DistQstPlan(cfg, Vr_pa; use_rfft)
        Qlm_out = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        Slm_out = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        Tlm_out = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        SHTnsKit.dist_analysis_qst!(qplan, Qlm_out, Slm_out, Tlm_out, Vr_pa, Vt_pa, Vp_pa)
        @test maximum(abs.(Qlm_out .- Qr_ref)) < 1e-12
        @test maximum(abs.(Slm_out .- S_ref)) < 1e-10
        @test maximum(abs.(Tlm_out .- T_ref)) < 1e-10
        SHTnsKit.dist_analysis_qst!(qplan, Qlm_out, Slm_out, Tlm_out, Vr_pa, Vt_pa, Vp_pa)  # warmup
        a = @allocated SHTnsKit.dist_analysis_qst!(qplan, Qlm_out, Slm_out, Tlm_out, Vr_pa, Vt_pa, Vp_pa)
        rank == 0 && println("dist_analysis_qst! (use_rfft=$use_rfft): $a B/call")
        @test a < 16384
    end
end

@testset "dist_SH_Yrotate pencil form: correctness + allocations" begin
    lmax = 16
    β = 0.37
    cfg = create_gauss_config(lmax, lmax + 2; nlon=2 * lmax + 2)
    spen = Pencil((lmax + 1, cfg.mmax + 1), comm)   # m distributed (last dim)
    ranges = PencilArrays.range_local(spen)

    rng = MersenneTwister(3)
    Qlm = zeros(ComplexF64, cfg.nlm)
    for m in 0:cfg.mmax, l in m:lmax
        Qlm[SHTnsKit.LM_index(lmax, 1, l, m) + 1] = randn(rng) + im * randn(rng)
    end
    MPI.Bcast!(Qlm, 0, comm)

    Alm_loc = zeros(ComplexF64, PencilArrays.size_local(spen)...)
    for (i_loc, i_g) in enumerate(ranges[1]), (j_loc, j_g) in enumerate(ranges[2])
        l = i_g - 1; m = j_g - 1
        Alm_loc[i_loc, j_loc] = m <= l ? Qlm[SHTnsKit.LM_index(lmax, 1, l, m) + 1] : zero(ComplexF64)
    end
    A_pa = PencilArray(spen, Alm_loc)
    R_pa = PencilArray(spen, zeros(ComplexF64, PencilArrays.size_local(spen)...))

    SHTnsKit.dist_SH_Yrotate(cfg, A_pa, β, R_pa)

    R_ref = zeros(ComplexF64, cfg.nlm)
    SHTnsKit.SH_Yrotate(cfg, Qlm, β, R_ref)
    R_loc = parent(R_pa)
    maxerr = 0.0
    for (i_loc, i_g) in enumerate(ranges[1]), (j_loc, j_g) in enumerate(ranges[2])
        l = i_g - 1; m = j_g - 1
        ref = m <= l ? R_ref[SHTnsKit.LM_index(lmax, 1, l, m) + 1] : zero(ComplexF64)
        maxerr = max(maxerr, abs(R_loc[i_loc, j_loc] - ref))
    end
    @test maxerr < 1e-10

    SHTnsKit.dist_SH_Yrotate(cfg, A_pa, β, R_pa)  # warmup
    a = @allocated SHTnsKit.dist_SH_Yrotate(cfg, A_pa, β, R_pa)
    rank == 0 && println("dist_SH_Yrotate: $a B/call")
    @test a < 65536
end

@testset "pencil rotations reject orders an mmax < lmax layout cannot hold" begin
    # The serial engine refuses order-mixing rotations when mmax < lmax; the
    # pencil Y/X/Euler rotations silently dropped the |m| > mmax components
    # (9-27% of the energy at lmax=8, mmax=5).
    cfg = create_gauss_config(8, 10; mmax=5)
    spen = Pencil((cfg.lmax + 1, cfg.mmax + 1), comm)
    ranges = PencilArrays.range_local(spen)
    rng = MersenneTwister(31)
    Qlm = randn(rng, ComplexF64, cfg.nlm)
    MPI.Bcast!(Qlm, 0, comm)
    local_block = zeros(ComplexF64, PencilArrays.size_local(spen)...)
    for (i, ig) in enumerate(ranges[1]), (j, jg) in enumerate(ranges[2])
        l, m = ig - 1, jg - 1
        m <= l && (local_block[i, j] = Qlm[SHTnsKit.LM_index(cfg.lmax, 1, l, m) + 1])
    end
    A = PencilArray(spen, local_block)
    R = PencilArray(spen, zeros(ComplexF64, PencilArrays.size_local(spen)...))

    @test_throws ArgumentError SHTnsKit.dist_SH_Yrotate(cfg, A, 0.7, R)
    @test_throws ArgumentError SHTnsKit.dist_SH_Yrotate90(cfg, A, R)
    @test_throws ArgumentError SHTnsKit.dist_SH_Xrotate90(cfg, A, R)
    @test_throws ArgumentError SHTnsKit.dist_SH_rotate_euler(cfg, A, 0.1, 0.7, 0.2, R)

    # A half-turn keeps every |m| and stays available, matching the serial engine.
    SHTnsKit.dist_SH_Yrotate(cfg, A, Float64(π), R)
    expected = similar(Qlm)
    SHTnsKit.SH_Yrotate(cfg, Qlm, Float64(π), expected)
    err = 0.0
    for (i, ig) in enumerate(ranges[1]), (j, jg) in enumerate(ranges[2])
        l, m = ig - 1, jg - 1
        m <= l && (err = max(err, abs(parent(R)[i, j] -
                                      expected[SHTnsKit.LM_index(cfg.lmax, 1, l, m) + 1])))
    end
    @test MPI.Allreduce(err, max, comm) < 1e-12
end

@testset "DistAnalysisPlan φ-decomposed (fallback layout)" begin
    lmax = 16; nlat = 20; nlon = 48
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    pen = Pencil((nlat, nlon), comm)   # default: decompose last dim (φ)
    f_pa, f_full = band_limited_field(cfg, pen)

    a_ref = SHTnsKit.analysis(cfg, f_full)
    plan = ParExt.DistAnalysisPlan(cfg, f_pa)
    Alm_out = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
    SHTnsKit.dist_analysis!(plan, Alm_out, f_pa)
    @test maximum(abs.(Alm_out .- a_ref)) < 1e-10
end

@testset "batch layout validation is metadata-only" begin
    # Keep the candidate large enough that allocating a duplicate expected
    # PencilArray would be unambiguously visible in the allocation count.
    cfg = create_gauss_config(16, 24; nlon=48)
    nfields = 512
    expected_pen = create_spectral_pencil(cfg; comm)
    output = PencilArray{ComplexF64}(undef, expected_pen, nfields)
    expected_global = (cfg.lmax + 1, cfg.mmax + 1, nfields)
    expected_local = (PencilArrays.size_local(expected_pen)..., nfields)

    ParExt._validate_pencil_layout_description!(
        expected_pen, expected_global, expected_local, output,
        :batch_allocation_probe; comm,
    )
    allocated = @allocated ParExt._validate_pencil_layout_description!(
        expected_pen, expected_global, expected_local, output,
        :batch_allocation_probe; comm,
    )
    candidate_bytes = sizeof(eltype(output)) * length(parent(output))
    rank == 0 && println(
        "metadata-only batch validation: $allocated B " *
        "(candidate payload $candidate_bytes B)",
    )
    @test allocated < 65_536
    @test allocated * 8 < candidate_bytes
end

rank == 0 && println("test_dist_plan_alloc: all testsets done on $(MPI.Comm_size(comm)) rank(s)")
