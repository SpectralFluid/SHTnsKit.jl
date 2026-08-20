# SHTnsKit.jl - Vorticity inverse-problem / adjoint diagnostics tests
#
# Covers the optimization helpers in src/vorticity_diagnostics.jl that were
# previously exported but untested:
#   grad_grid_enstrophy_zeta, loss_vorticity_grid,
#   grad_loss_vorticity_Tlm, loss_and_grad_vorticity_Tlm
#
# The toroidal gradient uses the hermitian-packed convention: the (l,m>0)
# coefficients each stand in for the ±m pair, so the directional derivative
# weights m>0 modes by 2 (the `_wm` weight). m=0 modes get weight 1.

using Test
using Random
using LinearAlgebra
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

# Build a random toroidal spectrum consistent with a real field
function _rand_Tlm(rng, lmax)
    T = zeros(ComplexF64, lmax + 1, lmax + 1)
    for m in 0:lmax, l in max(1, m):lmax
        T[l + 1, m + 1] = randn(rng) + im * randn(rng)
    end
    T[:, 1] .= real.(T[:, 1])   # m=0 must be real
    return T
end

# Hermitian dot-test weight: 1 for m=0, 2 for m>0
_wmat(lmax) = Float64[(m == 0 ? 1.0 : 2.0) for l in 0:lmax, m in 0:lmax]

@testset "Vorticity inverse-problem diagnostics" begin
    lmax = 6
    nlat = lmax + 2
    nlon = 2 * lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng = MersenneTwister(909)

    Ttarget = _rand_Tlm(rng, lmax)
    ζ_target = vorticity_grid(cfg, Ttarget)

    @testset "grad_grid_enstrophy_zeta finite-difference" begin
        ζ = randn(rng, nlat, nlon)
        g = grad_grid_enstrophy_zeta(cfg, ζ)
        @test size(g) == size(ζ)
        h = randn(rng, nlat, nlon)
        ϵ = 1e-6
        fd = (grid_enstrophy(cfg, ζ .+ ϵ .* h) - grid_enstrophy(cfg, ζ .- ϵ .* h)) / (2ϵ)
        @test isapprox(sum(g .* h), fd; rtol=1e-5, atol=1e-9)
        # Enstrophy is quadratic ⇒ exact Euler identity ⟨ζ, ∇Z⟩ = 2 Z
        @test isapprox(sum(g .* ζ), 2 * grid_enstrophy(cfg, ζ); rtol=1e-10, atol=1e-12)
    end

    @testset "loss_vorticity_grid properties" begin
        # Loss vanishes exactly at the generating spectrum
        @test loss_vorticity_grid(cfg, Ttarget, ζ_target) < 1e-18
        # Loss equals grid-enstrophy of the residual field
        T0 = 0.3 .* _rand_Tlm(rng, lmax)
        ζ = vorticity_grid(cfg, T0)
        @test isapprox(loss_vorticity_grid(cfg, T0, ζ_target),
                       grid_enstrophy(cfg, ζ .- ζ_target); rtol=1e-12, atol=1e-14)
        # Non-negative away from the solution
        @test loss_vorticity_grid(cfg, T0, ζ_target) > 0
    end

    @testset "grad_loss_vorticity_Tlm weighted finite-difference" begin
        T0 = 0.3 .* _rand_Tlm(rng, lmax)
        g = grad_loss_vorticity_Tlm(cfg, T0, ζ_target)
        @test size(g) == size(T0)
        W = _wmat(lmax)
        h = _rand_Tlm(rng, lmax)   # hermitian-consistent perturbation
        ϵ = 1e-6
        fd = (loss_vorticity_grid(cfg, T0 .+ ϵ .* h, ζ_target) -
              loss_vorticity_grid(cfg, T0 .- ϵ .* h, ζ_target)) / (2ϵ)
        ad = real(sum(W .* conj(g) .* h))
        VERBOSE && @info "grad_loss_vorticity_Tlm" fd ad
        @test isapprox(ad, fd; rtol=1e-5, atol=1e-7)
        # Gradient is (near) zero at the optimum
        gopt = grad_loss_vorticity_Tlm(cfg, Ttarget, ζ_target)
        @test maximum(abs, gopt) < 1e-8
    end

    @testset "loss_and_grad_vorticity_Tlm consistency" begin
        T0 = 0.5 .* _rand_Tlm(rng, lmax)
        L, g = loss_and_grad_vorticity_Tlm(cfg, T0, ζ_target)
        @test isapprox(L, loss_vorticity_grid(cfg, T0, ζ_target); rtol=1e-12, atol=1e-14)
        @test isapprox(g, grad_loss_vorticity_Tlm(cfg, T0, ζ_target); rtol=1e-12, atol=1e-14)
    end

    @testset "noncanonical inverse gradients" begin
        cfgn = create_gauss_config(lmax, nlat; nlon, norm=:schmidt,
                                   real_norm=true, cs_phase=false)
        Ttarget_n = _rand_Tlm(rng, lmax)
        ζtarget_n = vorticity_grid(cfgn, Ttarget_n)
        T0 = 0.3 .* _rand_Tlm(rng, lmax)
        h = _rand_Tlm(rng, lmax)
        W = _wmat(lmax)
        epsilon = 1e-6
        fd = (loss_vorticity_grid(cfgn, T0 .+ epsilon .* h, ζtarget_n) -
              loss_vorticity_grid(cfgn, T0 .- epsilon .* h, ζtarget_n)) / (2epsilon)

        g = grad_loss_vorticity_Tlm(cfgn, T0, ζtarget_n)
        @test isapprox(real(sum(W .* conj(g) .* h)), fd; rtol=2e-5, atol=2e-7)

        loss, combined_g = loss_and_grad_vorticity_Tlm(cfgn, T0, ζtarget_n)
        @test loss ≈ loss_vorticity_grid(cfgn, T0, ζtarget_n)
        @test isapprox(real(sum(W .* conj(combined_g) .* h)), fd;
                       rtol=2e-5, atol=2e-7)
    end
end
