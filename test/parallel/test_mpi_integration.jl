module MPIIntegrationTests

using Test
using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit

MPI.Initialized() || MPI.Init()
const ParallelExt = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
_get_global_indices(A, dim) = PencilArrays.range_local(PencilArrays.pencil(A))[dim]

function local_block(pen, values)
    return PencilArray(pen, values[PencilArrays.range_local(pen)...])
end

@testset "Parallel rfft equivalence" begin
    lmax = 6
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    P = PencilArrays.Pencil((nlat, nlon), MPI.COMM_WORLD)

    # Scalar - use global indices for consistent field across processes
    fθφ = PencilArrays.PencilArray{Float64}(undef, P)
    gl_θ = _get_global_indices(fθφ, 1)
    gl_φ = _get_global_indices(fθφ, 2)
    for (iθ_local, iθ_global) in enumerate(gl_θ)
        for (iφ_local, iφ_global) in enumerate(gl_φ)
            fθφ[iθ_local, iφ_local] = sin(0.13*iθ_global) + cos(0.07*iφ_global)
        end
    end

    Alm_c = zeros(ComplexF64, lmax+1, lmax+1)
    Alm_r = similar(Alm_c)
    plan_c = ParallelExt.DistAnalysisPlan(cfg, fθφ; use_rfft=false)
    plan_r = ParallelExt.DistAnalysisPlan(cfg, fθφ; use_rfft=true)
    SHTnsKit.dist_analysis!(plan_c, Alm_c, fθφ)
    SHTnsKit.dist_analysis!(plan_r, Alm_r, fθφ)
    @test isapprox(Alm_c, Alm_r; rtol=1e-10, atol=1e-12)

    # Vector - use global indices for consistent field across processes
    Vt = PencilArrays.PencilArray{Float64}(undef, P)
    Vp = PencilArrays.PencilArray{Float64}(undef, P)
    for (iθ_local, iθ_global) in enumerate(gl_θ)
        for (iφ_local, iφ_global) in enumerate(gl_φ)
            Vt[iθ_local, iφ_local] = 0.1*iθ_global + 0.05*iφ_global
            Vp[iθ_local, iφ_local] = 0.2*sin(0.1*iθ_global)
        end
    end

    Slm_c = zeros(ComplexF64, lmax+1, lmax+1)
    Tlm_c = zeros(ComplexF64, lmax+1, lmax+1)
    Slm_r = similar(Slm_c)
    Tlm_r = similar(Tlm_c)
    vplan_c = ParallelExt.DistSphtorPlan(cfg, Vt; use_rfft=false)
    vplan_r = ParallelExt.DistSphtorPlan(cfg, Vt; use_rfft=true)
    SHTnsKit.dist_analysis_sphtor!(vplan_c, Slm_c, Tlm_c, Vt, Vp)
    SHTnsKit.dist_analysis_sphtor!(vplan_r, Slm_r, Tlm_r, Vt, Vp)

    @test isapprox(Slm_c, Slm_r; rtol=1e-10, atol=1e-12)
    @test isapprox(Tlm_c, Tlm_r; rtol=1e-10, atol=1e-12)
end

@testset "Pencil θ-φ decomposition" begin
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)

    lmax = 6
    cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
    pθ, pφ = suggest_pencil_grid(comm, cfg.nlat, cfg.nlon; allow_one_dim=false)
    @test pθ * pφ == nprocs
    if min(pθ, pφ) == 1
        @info "Skipping θ-φ decomposition test (requires pθ>1 and pφ>1)"
    else
        topo = Pencil(MPITopology(comm, (pθ, pφ)), (cfg.nlat, cfg.nlon), (1, 2))
        fθφ = PencilArrays.PencilArray{Float64}(undef, topo)
        # Use Y_2^0 spherical harmonic pattern - exactly representable
        gl_θ = _get_global_indices(fθφ, 1)
        gl_φ = _get_global_indices(fθφ, 2)
        for (iθ_local, iθ_global) in enumerate(gl_θ)
            x = cfg.x[iθ_global]  # cos(θ)
            val = (3 * x^2 - 1) / 2  # Y_2^0 (unnormalized)
            for (iφ_local, iφ_global) in enumerate(gl_φ)
                fθφ[iθ_local, iφ_local] = val
            end
        end

        aplan = ParallelExt.DistAnalysisPlan(cfg, fθφ; use_rfft=true)
        Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
        SHTnsKit.dist_analysis!(aplan, Alm, fθφ)

        P_spec = PencilArrays.Pencil((cfg.lmax+1, cfg.mmax+1), comm)
        Alm_p = local_block(P_spec, Alm)
        spln = ParallelExt.DistPlan(cfg, fθφ; use_rfft=true)
        fθφ_back = PencilArrays.PencilArray{Float64}(undef, topo)
        SHTnsKit.dist_synthesis!(spln, fθφ_back, Alm_p)

        plan2d = ParallelExt.create_distributed_spectral_plan_2d(lmax, cfg.mmax, comm; p_l=pθ, p_m=pφ)
        plan2d_scratch = ParallelExt.create_distributed_spectral_plan_2d(
            lmax, cfg.mmax, comm;
            p_l=pθ, p_m=pφ,
            with_scratch=true,
            prototype_θφ=fθφ,
            cfg=cfg,
        )
        @test plan2d_scratch.with_scratch
        @test plan2d_scratch.scratch !== nothing
        @test hasproperty(plan2d_scratch.scratch, :P_complex)

        loc_diff = sum(abs2, Array(fθφ_back) .- Array(fθφ))
        loc_ref  = sum(abs2, Array(fθφ))
        glob_diff = MPI.Allreduce(loc_diff, +, comm)
        glob_ref  = MPI.Allreduce(loc_ref, +, comm)
        rel = sqrt(glob_diff / (glob_ref + eps()))
        @test rel < 1e-8

        # Vector roundtrip with rfft + spatial scratch (allocates (:θ,:k) buffers)
        Vt = PencilArrays.PencilArray{Float64}(undef, topo)
        Vp = PencilArrays.PencilArray{Float64}(undef, topo)
        # A round trip is exact for a band-limited field. Use both vector components.
        S_ref = zeros(ComplexF64, lmax+1, lmax+1)
        T_ref = zero(S_ref)
        S_ref[3, 2] = 0.3 + 0.4im
        T_ref[4, 3] = -0.2 + 0.1im
        vt_ref, vp_ref = synthesis_sphtor(cfg, S_ref, T_ref)
        local_indices = PencilArrays.range_local(topo)
        parent(Vt) .= view(vt_ref, local_indices...)
        parent(Vp) .= view(vp_ref, local_indices...)

        vplan = ParallelExt.DistSphtorPlan(cfg, Vt; use_rfft=true, with_spatial_scratch=true)
        Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
        Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
        SHTnsKit.dist_analysis_sphtor!(vplan, Slm, Tlm, Vt, Vp)

        Vt_back = PencilArrays.PencilArray{Float64}(undef, topo)
        Vp_back = PencilArrays.PencilArray{Float64}(undef, topo)
        SHTnsKit.dist_synthesis_sphtor!(vplan, Vt_back, Vp_back, Slm, Tlm)

        vt_diff = sum(abs2, Array(Vt_back) .- Array(Vt))
        vp_diff = sum(abs2, Array(Vp_back) .- Array(Vp))
        vt_ref  = sum(abs2, Array(Vt))
        vp_ref  = sum(abs2, Array(Vp))
        glob_vdiff = MPI.Allreduce(vt_diff + vp_diff, +, comm)
        glob_vref  = MPI.Allreduce(vt_ref + vp_ref, +, comm)
        rel_v = sqrt(glob_vdiff / (glob_vref + eps()))
        @test rel_v < 5e-8

        # Cache control toggles
        initial_cache = SHTnsKit.fft_plan_cache_enabled()
        SHTnsKit.enable_fft_plan_cache!()
        @test SHTnsKit.fft_plan_cache_enabled()
        SHTnsKit.disable_fft_plan_cache!()
        @test !SHTnsKit.fft_plan_cache_enabled()
        SHTnsKit.set_fft_plan_cache!(initial_cache)

        close(plan2d_scratch)
        close(plan2d)
    end
end

@testset "Parallel QST + local evals" begin
    lmax = 5
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    P = PencilArrays.Pencil((nlat, nlon), MPI.COMM_WORLD)

    # Build simple fields using global indices for consistency across processes
    fθφ = PencilArrays.PencilArray{Float64}(undef, P)
    Vtθφ = PencilArrays.PencilArray{Float64}(undef, P)
    Vpθφ = PencilArrays.PencilArray{Float64}(undef, P)
    Vrθφ = PencilArrays.PencilArray{Float64}(undef, P)
    gl_θ = _get_global_indices(fθφ, 1)
    gl_φ = _get_global_indices(fθφ, 2)
    for (iθ_local, iθ_global) in enumerate(gl_θ)
        for (iφ_local, iφ_global) in enumerate(gl_φ)
            fθφ[iθ_local, iφ_local] = sin(0.11*iθ_global) + cos(0.07*iφ_global)
            Vrθφ[iθ_local, iφ_local] = 0.3*cos(0.09*iθ_global)
            Vtθφ[iθ_local, iφ_local] = 0.2*sin(0.15*iφ_global)
            Vpθφ[iθ_local, iφ_local] = 0.1*cos(0.21*iφ_global)
        end
    end

    # Scalar local eval: point/lat
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)
    Qlm = Vector{ComplexF64}(undef, cfg.nlm)
    for m in 0:cfg.mmax, l in m:cfg.lmax
        Qlm[LM_index(cfg.lmax, cfg.mres, l, m) + 1] = Alm[l+1, m+1]
    end
    cost = 0.3; phi = 1.2
    # Create spectral Pencil for coefficient arrays
    P_spec = PencilArrays.Pencil((lmax+1, lmax+1), MPI.COMM_WORLD)
    val_dist = SHTnsKit.dist_SH_to_point(cfg, local_block(P_spec, Alm), cost, phi)
    val_ref = synthesis_point(cfg, Alm, cost, phi)
    @test isapprox(val_dist, val_ref; rtol=1e-10, atol=1e-12)
    lat_dist = SHTnsKit.dist_SH_to_lat(cfg, local_block(P_spec, Alm), cost)
    lat_ref = SH_to_lat(cfg, Qlm, cost)  # SH_to_lat expects Vector (packed) form
    @test isapprox(lat_dist, lat_ref; rtol=1e-10, atol=1e-12)

    # Evaluate the analyzed QST coefficients at the same point on every rank.
    Q,S,T = SHTnsKit.dist_analysis_qst(cfg, Vrθφ, Vtθφ, Vpθφ)

    # QST point/lat evals (use spectral Pencil from earlier)
    Qp = local_block(P_spec, Q)
    Sp = local_block(P_spec, S)
    Tp = local_block(P_spec, T)
    vr_d, vt_d, vp_d = SHTnsKit.dist_SHqst_to_point(cfg, Qp, Sp, Tp, cost, phi)

    # Build packed references
    Qv = similar(Qlm); Sv = similar(Qlm); Tv = similar(Qlm)
    for m in 0:cfg.mmax, l in m:cfg.lmax
        idx = LM_index(cfg.lmax, cfg.mres, l, m) + 1
        Qv[idx] = Q[l+1, m+1]; Sv[idx] = S[l+1, m+1]; Tv[idx] = T[l+1, m+1]
    end
    vr_r, vt_r, vp_r = SHTnsKit.SHqst_to_point(cfg, Qv, Sv, Tv, cost, phi)
    @test isapprox(vr_d, vr_r; rtol=1e-9, atol=1e-11)
    @test isapprox(vt_d, vt_r; rtol=1e-9, atol=1e-11)
    @test isapprox(vp_d, vp_r; rtol=1e-9, atol=1e-11)
end

@testset "Parallel halo operator" begin
    lmax = 6
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    P = PencilArrays.Pencil((nlat, nlon), MPI.COMM_WORLD)
    fθφ = PencilArrays.PencilArray{Float64}(undef, P)
    # Use global indices for consistent field across processes
    gl_θ = _get_global_indices(fθφ, 1)
    gl_φ = _get_global_indices(fθφ, 2)
    for (iθ_local, iθ_global) in enumerate(gl_θ)
        for (iφ_local, iφ_global) in enumerate(gl_φ)
            fθφ[iθ_local, iφ_local] = sin(0.19*iθ_global) * cos(0.13*iφ_global)
        end
    end

    # Dense analysis
    aplan = ParallelExt.DistAnalysisPlan(cfg, fθφ)
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    SHTnsKit.dist_analysis!(aplan, Alm, fθφ)

    # Pencil versions
    P_spec = PencilArrays.Pencil((cfg.lmax+1, cfg.mmax+1), MPI.COMM_WORLD)
    Alm_p = local_block(P_spec, Alm)
    R_p = PencilArrays.PencilArray{ComplexF64}(undef, P_spec)

    # Operator coefficients
    mx = zeros(Float64, 2*cfg.nlm)
    mul_ct_matrix(cfg, mx)

    # Neighbor/Allgatherv halo path
    SHTnsKit.dist_SH_mul_mx!(cfg, mx, Alm_p, R_p)

    # Dense reference
    Rlm = zeros(ComplexF64, size(Alm))
    SHTnsKit.dist_SH_mul_mx!(cfg, mx, Alm, Rlm)

    # Compare the local block with the corresponding dense coefficients.
    lloc = axes(R_p, 1); mloc = axes(R_p, 2)
    gl_l = _get_global_indices(R_p, 1)
    gl_m = _get_global_indices(R_p, 2)
    maxdiff = 0.0
    for (ii, il) in enumerate(lloc)
        for (jj, jm) in enumerate(mloc)
            maxdiff = max(maxdiff, abs(R_p[il,jm] - Rlm[gl_l[ii], gl_m[jj]]))
        end
    end
    @test maxdiff < 1e-10
end

@testset "Parallel Z-rotation equivalence" begin
    lmax = 6
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    P = PencilArrays.Pencil((nlat, nlon), MPI.COMM_WORLD)
    fθφ = PencilArrays.PencilArray{Float64}(undef, P)
    # Use global indices for consistent field across processes
    gl_θ = _get_global_indices(fθφ, 1)
    gl_φ = _get_global_indices(fθφ, 2)
    for (iθ_local, iθ_global) in enumerate(gl_θ)
        for (iφ_local, iφ_global) in enumerate(gl_φ)
            fθφ[iθ_local, iφ_local] = sin(0.23*iθ_global) + cos(0.29*iφ_global)
        end
    end

    # Analysis
    aplan = ParallelExt.DistAnalysisPlan(cfg, fθφ)
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    SHTnsKit.dist_analysis!(aplan, Alm, fθφ)

    # Rotate
    α = 0.37
    Rlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    SHTnsKit.dist_SH_Zrotate(cfg, Alm, α, Rlm)
    # Pencil variant should match
    P_spec = PencilArrays.Pencil((cfg.lmax+1, cfg.mmax+1), MPI.COMM_WORLD)
    Alm_p = local_block(P_spec, Alm)
    Rlm_p = SHTnsKit.dist_SH_Zrotate(cfg, Alm_p, α)

    # Compare dense vs PencilArray Z-rotation
    Rlm_from_pencil = Array(parent(Rlm_p))
    @test isapprox(Rlm[PencilArrays.range_local(P_spec)...], Rlm_from_pencil; rtol=1e-10, atol=1e-12)
end

@testset "Parallel Y-rotation allgatherm" begin
    lmax = 5
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    P = PencilArrays.Pencil((nlat, nlon), MPI.COMM_WORLD)
    fθφ = PencilArrays.PencilArray{Float64}(undef, P)
    # Use global indices for consistent field across processes
    gl_θ = _get_global_indices(fθφ, 1)
    gl_φ = _get_global_indices(fθφ, 2)
    for (iθ_local, iθ_global) in enumerate(gl_θ)
        for (iφ_local, iφ_global) in enumerate(gl_φ)
            fθφ[iθ_local, iφ_local] = 0.3*sin(0.1*iθ_global) + 0.8*cos(0.07*iφ_global)
        end
    end

    # Analysis
    aplan = ParallelExt.DistAnalysisPlan(cfg, fθφ)
    Alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    SHTnsKit.dist_analysis!(aplan, Alm, fθφ)

    # Allgatherm rotation
    P_spec = PencilArrays.Pencil((cfg.lmax+1, cfg.mmax+1), MPI.COMM_WORLD)
    Alm_p = local_block(P_spec, Alm)
    R_p = PencilArrays.PencilArray{ComplexF64}(undef, P_spec)
    β = 0.41
    SHTnsKit.dist_SH_Yrotate_allgatherm!(cfg, Alm_p, β, R_p)

    # Dense reference
    Rlm = zeros(ComplexF64, size(Alm))
    SHTnsKit.dist_SH_Yrotate(cfg, Alm, β, Rlm)

    # Compare
    lloc = axes(R_p, 1); mloc = axes(R_p, 2)
    gl_l = _get_global_indices(R_p, 1)
    gl_m = _get_global_indices(R_p, 2)
    maxdiff = 0.0
    for (ii, il) in enumerate(lloc)
        for (jj, jm) in enumerate(mloc)
            maxdiff = max(maxdiff, abs(R_p[il, jm] - Rlm[gl_l[ii], gl_m[jj]]))
        end
    end
    @test maxdiff < 1e-9
end

@testset "Parallel diagnostics" begin
    lmax = 6
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    P = PencilArrays.Pencil((nlat, nlon), MPI.COMM_WORLD)
    fθφ = PencilArrays.PencilArray{Float64}(undef, P)
    # Use Y_2^0 spherical harmonic pattern - exactly representable in spectral space
    gl_θ = _get_global_indices(fθφ, 1)
    gl_φ = _get_global_indices(fθφ, 2)
    for (iθ_local, iθ_global) in enumerate(gl_θ)
        x = cfg.x[iθ_global]  # cos(θ)
        val = (3 * x^2 - 1) / 2  # Y_2^0 (unnormalized)
        for (iφ_local, iφ_global) in enumerate(gl_φ)
            fθφ[iθ_local, iφ_local] = val
        end
    end

    # Scalar energy: spectral vs grid
    Alm = SHTnsKit.dist_analysis(cfg, fθφ)
    P_spec = PencilArrays.Pencil((cfg.lmax+1, cfg.mmax+1), MPI.COMM_WORLD)
    E_spec_dense = energy_scalar(cfg, Alm)
    Alm_p = local_block(P_spec, Alm)
    E_spec_pencil = energy_scalar(cfg, Alm_p)
    E_grid = grid_energy_scalar(cfg, fθφ)
    @test isapprox(E_spec_dense, E_spec_pencil; rtol=1e-10, atol=1e-12)
    @test isapprox(E_spec_pencil, E_grid; rtol=1e-8, atol=1e-10)

    # Spectra sum equals total
    El = energy_scalar_l_spectrum(cfg, Alm_p)
    Em = energy_scalar_m_spectrum(cfg, Alm_p)
    @test isapprox(sum(El), E_spec_pencil; rtol=1e-10, atol=1e-12)
    @test isapprox(sum(Em), E_spec_pencil; rtol=1e-10, atol=1e-12)

    cfg.phi_scale = :quad
    cfg.robert_form = true
    @test energy_scalar(cfg, Alm_p) ≈ grid_energy_scalar(cfg, local_block(P, synthesis(cfg, Alm)))
    S, T = zero(Alm), zero(Alm)
    S[2, 1] = 1
    T[3, 2] = 0.2im
    vt, vp = synthesis_sphtor(cfg, S, T)
    Sp, Tp = local_block(P_spec, S), local_block(P_spec, T)
    K = grid_energy_vector(cfg, local_block(P, vt), local_block(P, vp))
    @test K ≈ energy_vector(cfg, S, T)
    @test K ≈ energy_vector(cfg, Sp, Tp)
    @test K ≈ sum(energy_vector_m_spectrum(cfg, Sp, Tp))
end

end # module MPIIntegrationTests
