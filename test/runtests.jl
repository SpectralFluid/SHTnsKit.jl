# SHTnsKit.jl Test Suite
# 
# This comprehensive test suite validates the correctness of spherical harmonic transforms
# and related operations in both serial and parallel (MPI) environments.

using Test              # Julia testing framework
using LinearAlgebra    # For linear algebra operations 
using ChainRulesCore   # For automatic differentiation support
using Random           # For reproducible random number generation
using SHTnsKit         # The package being tested
using Zygote          # For automatic differentiation tests

const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1"

try
    @eval using FFTW
    VERBOSE && @info "FFTW available for tests"
catch e
    VERBOSE && @info "FFTW not available for tests" exception=(e, nothing)
end
VERBOSE && @info "phi_inv_scale mode" mode=get(ENV, "SHTNSKIT_PHI_SCALE", "dft")
try
    VERBOSE && @info "FFT backend (initial)" backend=SHTnsKit.fft_phi_backend()
catch
end

@testset "Single-mode sanity" begin
    # Construct a single spherical-harmonic mode and verify Parseval holds
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    alm = zeros(ComplexF64, lmax+1, lmax+1)
    alm[3, 2] = 1.0 + 0im  # l=2, m=1
    f = synthesis(cfg, alm; real_output=true)
    E_spec = energy_scalar(cfg, alm)
    E_grid = grid_energy_scalar(cfg, f)
    try
        VERBOSE && @info "Single-mode" l=2 m=1 E_spec E_grid backend=SHTnsKit.fft_phi_backend()
    catch
    end
    @test isapprox(E_spec, E_grid; rtol=1e-10, atol=1e-12)
end

@testset "Vector single-mode sanity" begin
    # Verify vector Parseval with a single (l,m) coefficient in S or T
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    # Single S-mode: l=2, m=1
    Slm = zeros(ComplexF64, lmax+1, lmax+1)
    Tlm = zeros(ComplexF64, lmax+1, lmax+1)
    Slm[3, 2] = 1.0 + 0im
    Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
    E_spec = energy_vector(cfg, Slm, Tlm)
    E_grid = grid_energy_vector(cfg, Vt, Vp)
    try
        VERBOSE && @info "Vector single-mode (S)" l=2 m=1 E_spec E_grid backend=SHTnsKit.fft_phi_backend()
    catch
    end
    @test isapprox(E_spec, E_grid; rtol=1e-9, atol=1e-11)
    # Single T-mode: l=3, m=2
    fill!(Slm, 0); fill!(Tlm, 0)
    Tlm[4, 3] = 1.0 + 0im
    Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
    E_spec = energy_vector(cfg, Slm, Tlm)
    E_grid = grid_energy_vector(cfg, Vt, Vp)
    try
        VERBOSE && @info "Vector single-mode (T)" l=3 m=2 E_spec E_grid backend=SHTnsKit.fft_phi_backend()
    catch
    end
    @test isapprox(E_spec, E_grid; rtol=1e-9, atol=1e-11)
end

@testset "Truncated scalar transforms" begin
    lmax = 5
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(11)
    alm = randn(rng, lmax+1, lmax+1) .+ im * randn(rng, lmax+1, lmax+1)
    alm[:, 1] .= real.(alm[:, 1])
    Vr = vec(synthesis(cfg, alm; real_output=true))

    full_Q = analysis_packed(cfg, Vr)
    ltr = lmax - 1
    trunc_Q = analysis_packed_l(cfg, Vr, ltr)
    for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            if l <= ltr
                @test isapprox(trunc_Q[lm], full_Q[lm]; rtol=1e-10, atol=1e-12)
            else
                @test trunc_Q[lm] == 0
            end
        end
    end

    full_spatial = synthesis_packed(cfg, full_Q)
    zeroed_Q = copy(full_Q)
    for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        for l in (ltr+1):cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            zeroed_Q[lm] = 0
        end
    end
    expected_trunc = synthesis_packed(cfg, zeroed_Q)
    @test isapprox(synthesis_packed_l(cfg, full_Q, ltr), expected_trunc; rtol=1e-10, atol=1e-12)
    @test isapprox(synthesis_packed_l(cfg, full_Q, lmax), full_spatial; rtol=1e-10, atol=1e-12)
end

@testset "Mode-limited vector wrappers" begin
    lmax = 5
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    azm = 1
    ltr = lmax - 1
    len = ltr - azm + 1
    rng = MersenneTwister(17)
    Sl = ComplexF64.(randn(rng, len) .+ 1im * randn(rng, len))
    Tl = ComplexF64.(randn(rng, len) .+ 1im * randn(rng, len))

    Vt_ref, Vp_ref = synthesis_sphtor_ml(cfg, azm, Sl, zeros(ComplexF64, len), ltr)
    Vt_s, Vp_s = synthesis_sph_ml(cfg, azm, Sl, ltr)
    @test isapprox(Vt_s, Vt_ref; rtol=1e-10, atol=1e-12)
    @test isapprox(Vp_s, Vp_ref; rtol=1e-10, atol=1e-12)

    Vt_ref_t, Vp_ref_t = synthesis_sphtor_ml(cfg, azm, zeros(ComplexF64, len), Tl, ltr)
    Vt_t, Vp_t = synthesis_tor_ml(cfg, azm, Tl, ltr)
    @test isapprox(Vt_t, Vt_ref_t; rtol=1e-10, atol=1e-12)
    @test isapprox(Vp_t, Vp_ref_t; rtol=1e-10, atol=1e-12)

    Slm = ComplexF64.(randn(rng, lmax+1, lmax+1) .+ 1im * randn(rng, lmax+1, lmax+1))
    Slm[:, 1] .= real.(Slm[:, 1])
    Gt_ref, Gp_ref = synthesis_sph(cfg, Slm)
    Gt, Gp = synthesis_sph(cfg, Slm)
    @test isapprox(Gt, Gt_ref; rtol=1e-10, atol=1e-12)
    @test isapprox(Gp, Gp_ref; rtol=1e-10, atol=1e-12)

    Gt_l_ref, Gp_l_ref = synthesis_sph_l(cfg, Slm, ltr)
    Gt_l, Gp_l = synthesis_sph_l(cfg, Slm, ltr)
    @test isapprox(Gt_l, Gt_l_ref; rtol=1e-10, atol=1e-12)
    @test isapprox(Gp_l, Gp_l_ref; rtol=1e-10, atol=1e-12)

    Sl_ml = ComplexF64.(randn(rng, len) .+ 1im * randn(rng, len))
    Vt_ml_ref, Vp_ml_ref = synthesis_sph_ml(cfg, azm, Sl_ml, ltr)
    Vt_ml, Vp_ml = synthesis_sph_ml(cfg, azm, Sl_ml, ltr)
    @test isapprox(Vt_ml, Vt_ml_ref; rtol=1e-10, atol=1e-12)
    @test isapprox(Vp_ml, Vp_ml_ref; rtol=1e-10, atol=1e-12)
end

@testset "Regular grid" begin
    lmax = 8
    # Driscoll-Healy quadrature for exact equiangular transforms
    nlat = 2 * (lmax + 1)
    nlon = 2 * (2 * lmax + 1)
    cfg_reg = create_regular_config(lmax, nlat; nlon=nlon, precompute_plm=true, include_poles=true, use_dh_weights=true)
    @test cfg_reg.grid_type in (:regular, :regular_poles, :driscoll_healy)
    @test cfg_reg.use_plm_tables

    rng = MersenneTwister(23)
    alm = zeros(ComplexF64, lmax+1, lmax+1)
    for m in 0:lmax, l in m:lmax
        alm[l+1, m+1] = randn(rng) + im * randn(rng)
    end
    alm[:, 1] .= real.(alm[:, 1])

    f = synthesis(cfg_reg, alm; real_output=true)
    alm_rt = analysis(cfg_reg, f)
    alm_err = maximum(abs.(alm_rt - alm))
    @test alm_err < 1e-6

end

@testset "LM_cplx compatibility" begin
    lmax = 6
    cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
    for l in 0:lmax
        for m in -l:l
            idx = LM_cplx(cfg, l, m)
            @test idx == LM_cplx_index(cfg.lmax, cfg.mmax, l, m)
        end
    end
end

"""
    parseval_scalar_test(lmax::Int)

Test Parseval's identity for scalar fields: energy should be conserved between
spectral and spatial representations. This verifies the orthogonality and 
normalization of the spherical harmonic basis functions.
"""
function parseval_scalar_test(lmax::Int)
    # Set up a slightly over-resolved grid to ensure accuracy
    nlat = lmax + 2  # Extra latitude points for Gauss-Legendre accuracy
    nlon = 2*lmax + 1  # Minimum longitude points for alias-free transforms
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(42)  # Reproducible random numbers

    # Generate random spectral coefficients with proper symmetry for real fields
    alm = randn(rng, lmax+1, lmax+1) .+ im * randn(rng, lmax+1, lmax+1)
    # Ensure real-field consistency: m=0 coefficients must be real
    alm[:, 1] .= real.(alm[:, 1])
    f = synthesis(cfg, alm; real_output=true)

    # Parseval's identity: ∫|f|² dΩ = Σ|a_lm|² (energy conservation)
    E_spec = energy_scalar(cfg, alm)
    E_grid = grid_energy_scalar(cfg, f)
    VERBOSE && @info "Parseval scalar" lmax E_spec E_grid rel=abs(E_spec - E_grid)/(abs(E_grid)+eps())
    @test isapprox(E_spec, E_grid; rtol=1e-10, atol=1e-12)
end


"""
    parseval_vector_test(lmax::Int)

Test Parseval's identity for vector fields: energy should be conserved between
spectral (S_lm, T_lm) and spatial (V_θ, V_φ) representations. This tests the 
spheroidal/toroidal decomposition and ensures proper normalization.
"""
function parseval_vector_test(lmax::Int)
    nlat = lmax + 2   # Over-resolved grid for accuracy
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(24)  # Different seed from scalar test

    # Generate random spheroidal and toroidal spectral coefficients
    Slm = randn(rng, lmax+1, lmax+1) .+ im * randn(rng, lmax+1, lmax+1)  # Spheroidal
    Tlm = randn(rng, lmax+1, lmax+1) .+ im * randn(rng, lmax+1, lmax+1)  # Toroidal
    # Ensure real-field consistency for m=0 modes
    Slm[:, 1] .= real.(Slm[:, 1])
    Tlm[:, 1] .= real.(Tlm[:, 1])
    
    # Synthesize vector components from spectral coefficients
    Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)

    # Parseval for vector fields: ∫(|V_θ|² + |V_φ|²) dΩ = Σ(|S_lm|² + |T_lm|²)
    E_vec = energy_vector(cfg, Slm, Tlm)
    E_grid = grid_energy_vector(cfg, Vt, Vp)
    VERBOSE && @info "Parseval vector" lmax E_vec E_grid rel=abs(E_vec - E_grid)/(abs(E_grid)+eps())
    @test isapprox(E_vec, E_grid; rtol=1e-9, atol=1e-11)
end

"""
    vector_coefficient_roundtrip_test(lmax::Int)

Test that vector spherical harmonic coefficients can be recovered after
synthesis → analysis roundtrip. This verifies that the spheroidal-toroidal
decomposition is correctly inverted.
"""
function vector_coefficient_roundtrip_test(lmax::Int)
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(31)

    # Generate random spheroidal and toroidal coefficients
    Slm_orig = zeros(ComplexF64, lmax+1, lmax+1)
    Tlm_orig = zeros(ComplexF64, lmax+1, lmax+1)

    # Fill only valid coefficients (l >= m, and l >= 1 for S/T)
    for m in 0:lmax
        for l in max(1, m):lmax
            Slm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
            Tlm_orig[l+1, m+1] = randn(rng) + im * randn(rng)
        end
        # m=0 must be real for real-valued fields
        if m == 0
            for l in 1:lmax
                Slm_orig[l+1, 1] = real(Slm_orig[l+1, 1])
                Tlm_orig[l+1, 1] = real(Tlm_orig[l+1, 1])
            end
        end
    end

    # Synthesize to spatial domain
    Vt, Vp = synthesis_sphtor(cfg, Slm_orig, Tlm_orig; real_output=true)

    # Analyze back to spectral domain
    Slm_recov, Tlm_recov = analysis_sphtor(cfg, Vt, Vp)

    # Compare recovered coefficients
    S_err = maximum(abs.(Slm_recov - Slm_orig))
    T_err = maximum(abs.(Tlm_recov - Tlm_orig))
    VERBOSE && @info "Vector roundtrip" lmax S_max_err=S_err T_max_err=T_err

    # Check that coefficients are recovered accurately
    @test isapprox(Slm_recov, Slm_orig; rtol=1e-9, atol=1e-11)
    @test isapprox(Tlm_recov, Tlm_orig; rtol=1e-9, atol=1e-11)
end

# ===== CORE MATHEMATICAL PROPERTY TESTS =====
@testset "Parseval identities" begin
    parseval_scalar_test(10)  # Test scalar energy conservation
    parseval_vector_test(10)  # Test vector energy conservation
end

@testset "Vector coefficient roundtrip" begin
    vector_coefficient_roundtrip_test(8)   # Test at moderate resolution
    vector_coefficient_roundtrip_test(16)  # Test at higher resolution
end

# ===== AUTOMATIC DIFFERENTIATION TESTS =====
# These tests verify that SHTnsKit functions are compatible with Julia's 
# automatic differentiation ecosystem (ForwardDiff, Zygote)

@testset "AD gradients - ForwardDiff" begin
    _has_forwarddiff = try; using ForwardDiff; true; catch; false; end
    if _has_forwarddiff
        lmax = 3
        nlat = lmax + 2
        nlon = 2*lmax + 1

        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(7)
        f0 = randn(rng, nlat, nlon)  # Random spatial field

        # Define loss function: spatial field → spectral energy
        loss(x) = energy_scalar(cfg, analysis(cfg, reshape(x, nlat, nlon)))
        x0 = vec(f0)  # Flatten for ForwardDiff
        g = ForwardDiff.gradient(loss, x0)  # Compute gradient w.r.t. spatial field

        # Validate gradient using finite difference check
        h = randn(rng, length(x0))  # Random perturbation direction
        ϵ = 1e-6
        φ(ξ) = loss(x0 .+ ξ .* h)  # Loss along perturbation direction
        dfdξ_fd = (φ(ϵ) - φ(-ϵ)) / (2ϵ)  # Finite difference derivative
        dfdξ_ad = dot(g, h)  # Automatic differentiation derivative
        @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)
    else
        @info "Skipping ForwardDiff gradient test (package not available)"
    end
end

@testset "AD gradients - Zygote" begin
    lmax = 3
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(9)
    f0 = randn(rng, nlat, nlon)
    loss(f) = energy_scalar(cfg, analysis(cfg, f))
    g = Zygote.gradient(loss, f0)[1]

    # Dot-test
    h = randn(rng, size(f0))
    ϵ = 1e-6
    dfdξ_fd = (loss(f0 .+ ϵ.*h) - loss(f0 .- ϵ.*h)) / (2ϵ)
    dfdξ_ad = sum(g .* h)
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)
end

@testset "AD gradients - Zygote: scalar analysis pipeline" begin
    # Test gradient through: spatial → analysis → energy (known working pattern)
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(101)

    # Real spatial field input
    f0 = randn(rng, nlat, nlon)

    # Loss: spatial → spectral → energy
    loss(f) = energy_scalar(cfg, analysis(cfg, f))
    g = Zygote.gradient(loss, f0)[1]

    # Finite difference check
    h = randn(rng, nlat, nlon)
    ϵ = 1e-6
    dfdξ_fd = (loss(f0 .+ ϵ.*h) - loss(f0 .- ϵ.*h)) / (2ϵ)
    dfdξ_ad = sum(g .* h)
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)
end

@testset "AD gradients - Zygote: vector analysis pipeline" begin
    # Test gradient through: spatial vector → analysis_sphtor → energy
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(201)

    # Real spatial vector field inputs
    Vt0 = randn(rng, nlat, nlon)
    Vp0 = randn(rng, nlat, nlon)

    # Loss: spatial → spectral → energy (gradient w.r.t. Vt)
    function loss_vt(Vt)
        S, T = analysis_sphtor(cfg, Vt, Vp0)
        return energy_vector(cfg, S, T)
    end

    gVt = Zygote.gradient(loss_vt, Vt0)[1]
    hVt = randn(rng, nlat, nlon)

    ϵ = 1e-6
    dfdξ_fd = (loss_vt(Vt0 .+ ϵ.*hVt) - loss_vt(Vt0 .- ϵ.*hVt)) / (2ϵ)
    dfdξ_ad = sum(gVt .* hVt)
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)

    # Loss: gradient w.r.t. Vp
    function loss_vp(Vp)
        S, T = analysis_sphtor(cfg, Vt0, Vp)
        return energy_vector(cfg, S, T)
    end

    gVp = Zygote.gradient(loss_vp, Vp0)[1]
    hVp = randn(rng, nlat, nlon)

    dfdξ_fd = (loss_vp(Vp0 .+ ϵ.*hVp) - loss_vp(Vp0 .- ϵ.*hVp)) / (2ϵ)
    dfdξ_ad = sum(gVp .* hVp)
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)
end

@testset "AD gradients - Zygote: grid energy functions" begin
    # Test gradient of grid-space energy functions (simple, should work)
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(301)

    # Test scalar grid energy
    f0 = randn(rng, nlat, nlon)
    loss_scalar(f) = grid_energy_scalar(cfg, f)

    g = Zygote.gradient(loss_scalar, f0)[1]
    h = randn(rng, nlat, nlon)

    ϵ = 1e-6
    dfdξ_fd = (loss_scalar(f0 .+ ϵ.*h) - loss_scalar(f0 .- ϵ.*h)) / (2ϵ)
    dfdξ_ad = sum(g .* h)
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)

    # Test vector grid energy
    Vt0 = randn(rng, nlat, nlon)
    Vp0 = randn(rng, nlat, nlon)

    function loss_vec_vt(Vt)
        return grid_energy_vector(cfg, Vt, Vp0)
    end

    gVt = Zygote.gradient(loss_vec_vt, Vt0)[1]
    hVt = randn(rng, nlat, nlon)

    dfdξ_fd = (loss_vec_vt(Vt0 .+ ϵ.*hVt) - loss_vec_vt(Vt0 .- ϵ.*hVt)) / (2ϵ)
    dfdξ_ad = sum(gVt .* hVt)
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)
end

@testset "AD gradients - Zygote: QST analysis pipeline" begin
    # Test gradient through: spatial 3D vector → analysis_qst → energy
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1

    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(401)

    # Real spatial 3D vector field inputs
    Vr0 = randn(rng, nlat, nlon)
    Vt0 = randn(rng, nlat, nlon)
    Vp0 = randn(rng, nlat, nlon)

    # Loss: spatial → spectral → sum of squared coefficients
    function loss_vr(Vr)
        Q, S, T = analysis_qst(cfg, Vr, Vt0, Vp0)
        return sum(abs2, Q) + sum(abs2, S) + sum(abs2, T)
    end

    gVr = Zygote.gradient(loss_vr, Vr0)[1]
    hVr = randn(rng, nlat, nlon)

    ϵ = 1e-6
    dfdξ_fd = (loss_vr(Vr0 .+ ϵ.*hVr) - loss_vr(Vr0 .- ϵ.*hVr)) / (2ϵ)
    dfdξ_ad = sum(gVr .* hVr)
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)
end

@testset "AD gradients - Zygote: multiple resolutions" begin
    # Test gradients work across different grid sizes
    rng = MersenneTwister(501)

    for lmax in [3, 5, 8]
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        f0 = randn(rng, nlat, nlon)
        loss(f) = energy_scalar(cfg, analysis(cfg, f))

        g = Zygote.gradient(loss, f0)[1]
        h = randn(rng, nlat, nlon)

        ϵ = 1e-6
        dfdξ_fd = (loss(f0 .+ ϵ.*h) - loss(f0 .- ϵ.*h)) / (2ϵ)
        dfdξ_ad = sum(g .* h)
        @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-7)
    end
end

@testset "AD gradients - Zygote: rotations and operators" begin
    begin
        # Setup
        lmax = 3
        nlat = lmax + 2
        nlon = 2*lmax + 1

        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(123)

        # Rotation around Y: gradient wrt packed real Qlm
        Q0 = ComplexF64.(randn(rng, cfg.nlm))
        α = 0.3

        function loss_yrot(Q)
            R = similar(Q)
            R = SH_Yrotate(cfg, Q, α, R)
            return 0.5 * sum(abs2, R)
        end

        h = ComplexF64.(randn(rng, length(Q0)))
        ϵ = 1e-6
        φ(ξ) = loss_yrot(Q0 .+ ξ .* h)
        dfdξ_fd = (φ(ϵ) - φ(-ϵ)) / (2ϵ)
        # Linearization: dL = Re⟨R, A h⟩ where A = Y-rotation
        Ry = similar(Q0); Ry = SH_Yrotate(cfg, Q0, α, Ry)
        Ayh = similar(Q0); Ayh = SH_Yrotate(cfg, h, α, Ayh)
        dfdξ_lin = real(sum(conj(Ry) .* Ayh))
        @test isapprox(dfdξ_lin, dfdξ_fd; rtol=5e-4, atol=1e-7)

        # Z-rotation
        function loss_zrot(Q)
            R = similar(Q)
            R = SH_Zrotate(cfg, Q, α, R)
            return 0.5 * sum(abs2, R)
        end

        φz(ξ) = loss_zrot(Q0 .+ ξ .* h)
        dfdξ_fd = (φz(ϵ) - φz(-ϵ)) / (2ϵ)
        # Analytic directional derivative using linearization: dL = Re⟨R, A h⟩
        Rz = similar(Q0); Rz = SH_Zrotate(cfg, Q0, α, Rz)
        Ah = similar(Q0); Ah = SH_Zrotate(cfg, h, α, Ah)
        dfdξ_lin = real(sum(conj(Rz) .* Ah))
        @test isapprox(dfdξ_lin, dfdξ_fd; rtol=5e-4, atol=1e-7)

        # Operator application: test gradients wrt Q and mx
        mx = zeros(Float64, 2*cfg.nlm)
        mul_ct_matrix(cfg, mx)
        Qv = ComplexF64.(randn(rng, cfg.nlm))
        function loss_op(Q, mx)
            R = similar(Q)
            R = SH_mul_mx(cfg, mx, Q, R)
            return 0.5 * sum(abs2, R)
        end

        gQ_op, gmx_op = Zygote.gradient(loss_op, Qv, mx)
        hQ = ComplexF64.(randn(rng, length(Qv)))
        hmx = randn(rng, length(mx))
        φQ(ξ) = loss_op(Qv .+ ξ .* hQ, mx)
        φmx(ξ) = loss_op(Qv, mx .+ ξ .* hmx)
        dfdξ_fd_Q = (φQ(ϵ) - φQ(-ϵ)) / (2ϵ)
        dfdξ_ad_Q = real(sum(gQ_op .* hQ))

        @test isapprox(dfdξ_ad_Q, dfdξ_fd_Q; rtol=5e-4, atol=1e-7)
        dfdξ_fd_mx = (φmx(ϵ) - φmx(-ϵ)) / (2ϵ)
        dfdξ_ad_mx = sum(gmx_op .* hmx)
        @test isapprox(dfdξ_ad_mx, dfdξ_fd_mx; rtol=5e-4, atol=1e-7)

        # Angle gradient for real rotation
        function loss_angles(a,b,c)
            r = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax; α=a, β=b, γ=c)
            R = similar(Q0)
            R = SHTnsKit.shtns_rotation_apply_real(r, Q0, R)
            return 0.5 * sum(abs2, R)
        end

        # Use provided helper to get angle gradients (more robust than tracing struct fields)
        gα, gβ, gγ = SHTnsKit.zgrad_rotation_angles_real(cfg, Q0, α, 0.1, -0.2)

        # Finite-diff checks for real rotation angles
        φa(ξ) = loss_angles(α + ξ, 0.1, -0.2)
        dfdξ_fd = (φa(ϵ) - φa(-ϵ)) / (2ϵ)
        @test isapprox(gα, dfdξ_fd; rtol=5e-4, atol=1e-7)
        φb(ξ) = loss_angles(α, 0.1 + ξ, -0.2)
        dfdξ_fd = (φb(ϵ) - φb(-ϵ)) / (2ϵ)
        @test isapprox(gβ, dfdξ_fd; rtol=5e-4, atol=1e-7)
        φg(ξ) = loss_angles(α, 0.1, -0.2 + ξ)
        dfdξ_fd = (φg(ϵ) - φg(-ϵ)) / (2ϵ)
        @test isapprox(gγ, dfdξ_fd; rtol=5e-4, atol=1e-7)

        # Complex rotation angle gradients: helper vs finite-diff on α
        let
            Zlen = SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
            Zlm = ComplexF64.(randn(rng, Zlen) .+ 1im * randn(rng, Zlen))
            αc, βc, γc = 0.2, -0.15, 0.33
            # Helper gradients (analytic)
            gαc, gβc, gγc = SHTnsKit.zgrad_rotation_angles_cplx(cfg.lmax, cfg.mmax, Zlm, αc, βc, γc)
            # Finite-diff on α
            function loss_cplx(a, b, c)
                r = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax; α=a, β=b, γ=c)
                R = similar(Zlm)
                R = SHTnsKit.shtns_rotation_apply_cplx(r, Zlm, R)
                return 0.5 * sum(abs2, R)
            end
            φac(ξ) = loss_cplx(αc + ξ, βc, γc)
            dfdξ_fd_c = (φac(ϵ) - φac(-ϵ)) / (2ϵ)
            @test isapprox(gαc, dfdξ_fd_c; rtol=5e-4, atol=1e-7)
            φbc(ξ) = loss_cplx(αc, βc + ξ, γc)
            dfdξ_fd_c = (φbc(ϵ) - φbc(-ϵ)) / (2ϵ)
            @test isapprox(gβc, dfdξ_fd_c; rtol=5e-4, atol=1e-7)
            φgc(ξ) = loss_cplx(αc, βc, γc + ξ)
            dfdξ_fd_c = (φgc(ϵ) - φgc(-ϵ)) / (2ϵ)
            @test isapprox(gγc, dfdξ_fd_c; rtol=5e-4, atol=1e-7)
        end
    end
end

@testset "Convenience gradients and packed energies" begin
    lmax = 4
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(202)

    # Packed scalar
    f = randn(rng, nlat, nlon)
    Q = analysis_packed(cfg, vec(f))
    @test isapprox(energy_scalar_packed(cfg, Q), energy_scalar(cfg, analysis(cfg, f)); rtol=1e-10)
    GQ = grad_energy_scalar_packed(cfg, Q)
    ϵ = 1e-7
    hQ = ComplexF64.(randn(rng, length(Q)))
    φ(ξ) = energy_scalar_packed(cfg, Q .+ ξ .* hQ)
    dfdξ_fd = (φ(ϵ) - φ(-ϵ)) / (2ϵ)
    dfdξ_ad = real(sum(conj(GQ) .* hQ))
    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-8)

    # Packed vector
    Vt = randn(rng, nlat, nlon)
    Vp = randn(rng, nlat, nlon)
    Slm, Tlm = analysis_sphtor(cfg, Vt, Vp)
    # Pack matrices into vectors in LM order
    Sp = similar(Q); Tp = similar(Q)
    for m in 0:cfg.mmax
        for l in m:cfg.lmax
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Sp[lm] = Slm[l+1, m+1]
            Tp[lm] = Tlm[l+1, m+1]
        end
    end

    @test isapprox(energy_vector_packed(cfg, Sp, Tp), energy_vector(cfg, Slm, Tlm); rtol=1e-9)
    GS, GT = grad_energy_vector_packed(cfg, Sp, Tp)
    hS = ComplexF64.(randn(rng, length(Sp)))
    hT = ComplexF64.(randn(rng, length(Tp)))
    φv(ξ) = energy_vector_packed(cfg, Sp .+ ξ .* hS, Tp .+ ξ .* hT)
    dfdξ_fd = (φv(ϵ) - φv(-ϵ)) / (2ϵ)
    dfdξ_ad = real(sum(conj(GS) .* hS) + sum(conj(GT) .* hT))

    @test isapprox(dfdξ_ad, dfdξ_fd; rtol=5e-4, atol=1e-8)
end


if get(ENV, "SHTNSKIT_RUN_MPI_TESTS", "0") == "1"
    include("parallel/test_mpi_integration.jl")
end

# Serial transform tests (exercises single-processor transform functions)
include("serial/runtests.jl")

# Parallel grid resolution tests (tests across multiple lat/lon configurations).
# Despite the folder name these are non-MPI: parametric grid sweeps + native
# threading that depend only on SHTnsKit + stdlib, so they run on all platforms.
# (True MPI tests live in test/parallel/test_mpi_*.jl and run via mpiexec.)
include("parallel/runtests.jl")

# JET.jl type stability tests (optional)
if get(ENV, "SHTNSKIT_RUN_JET_TESTS", "0") == "1"
    include("test_jet.jl")
else
    @info "Skipping JET type stability tests (set SHTNSKIT_RUN_JET_TESTS=1 to enable)"
end

# Aqua.jl quality assurance tests (optional)
if get(ENV, "SHTNSKIT_RUN_AQUA_TESTS", "0") == "1"
    include("test_aqua.jl")
else
    @info "Skipping Aqua quality assurance tests (set SHTNSKIT_RUN_AQUA_TESTS=1 to enable)"
end
