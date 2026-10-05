# SHTnsKit.jl - Vorticity inverse-problem / adjoint diagnostics tests
#
# Covers the optimization helpers in src/vorticity_diagnostics.jl that were
# previously exported but untested:
#   grad_grid_enstrophy_zeta, loss_vorticity_grid,
#   grad_loss_vorticity_Tlm, loss_and_grad_vorticity_Tlm
#
# The toroidal gradient follows the ChainRules/Zygote convention shared by the
# other `grad_*` helpers: dL(T)[h] = real(sum(conj(g) .* h)). Each (l,m>0)
# coefficient stands in for the ±m pair of the real vorticity field, so those
# gradient entries carry the factor 2 (the `_wm` weight).

using Test
using Random
using LinearAlgebra
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")
_vorticity_has_zygote = try; @eval using Zygote; true; catch; false; end

# Build a random toroidal spectrum consistent with a real field
function _rand_Tlm(rng, lmax)
    T = zeros(ComplexF64, lmax + 1, lmax + 1)
    for m in 0:lmax, l in max(1, m):lmax
        T[l + 1, m + 1] = randn(rng) + im * randn(rng)
    end
    T[:, 1] .= real.(T[:, 1])   # m=0 must be real
    return T
end

@testset "Vorticity inverse-problem diagnostics" begin
    @testset "inverse gradients include synthesis phi scaling" begin
        rng = MersenneTwister(910)
        for norm in (:orthonormal, :schmidt), scaling in (:configured_quad, :environment_quad, :environment_dft)
            cfg = create_gauss_config(3, 5; norm, real_norm=true, cs_phase=false)
            cfg.phi_scale = scaling === :environment_quad ? :dft : :quad
            override = scaling === :configured_quad ? nothing :
                       scaling === :environment_quad ? "quad" : "dft"
            withenv("SHTNSKIT_PHI_SCALE" => override) do
                T0 = _rand_Tlm(rng, cfg.lmax)
                target = randn(rng, cfg.nlat, cfg.nlon)
                # Check m=0 separately from the m>0 modes that stand for ±m.
                for m in (0, 1)
                    h = zeros(ComplexF64, size(T0))
                    h[3, m + 1] = m == 0 ? 1 : 0.7 + 0.4im
                    epsilon = 1e-6
                    fd = (loss_vorticity_grid(cfg, T0 .+ epsilon .* h, target) -
                          loss_vorticity_grid(cfg, T0 .- epsilon .* h, target)) / (2epsilon)
                    g = grad_loss_vorticity_Tlm(cfg, T0, target)
                    loss, combined_g = loss_and_grad_vorticity_Tlm(cfg, T0, target)
                    @test real(sum(conj(g) .* h)) ≈ fd rtol=2e-5 atol=2e-7
                    @test real(sum(conj(combined_g) .* h)) ≈ fd rtol=2e-5 atol=2e-7
                    @test loss ≈ loss_vorticity_grid(cfg, T0, target)
                end
            end
        end
    end

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

    @testset "grad_loss_vorticity_Tlm finite-difference" begin
        T0 = 0.3 .* _rand_Tlm(rng, lmax)
        g = grad_loss_vorticity_Tlm(cfg, T0, ζ_target)
        @test size(g) == size(T0)
        h = _rand_Tlm(rng, lmax)   # hermitian-consistent perturbation
        ϵ = 1e-6
        fd = (loss_vorticity_grid(cfg, T0 .+ ϵ .* h, ζ_target) -
              loss_vorticity_grid(cfg, T0 .- ϵ .* h, ζ_target)) / (2ϵ)
        ad = real(sum(conj(g) .* h))
        VERBOSE && @info "grad_loss_vorticity_Tlm" fd ad
        @test isapprox(ad, fd; rtol=1e-5, atol=1e-7)
        # Gradient is (near) zero at the optimum
        gopt = grad_loss_vorticity_Tlm(cfg, Ttarget, ζ_target)
        @test maximum(abs, gopt) < 1e-8
        # Same convention as grad_enstrophy_Tlm: the residual-free loss
        # against a zero target is the enstrophy of the vorticity field.
        @test grad_loss_vorticity_Tlm(cfg, T0, zeros(nlat, nlon)) ≈
              grad_enstrophy_Tlm(cfg, T0) rtol=1e-10
    end

    if _vorticity_has_zygote
        @testset "grad_loss_vorticity_Tlm matches Zygote ($phi_scale)" for phi_scale in (:dft, :quad)
            withenv("SHTNSKIT_PHI_SCALE" => nothing) do  # the variable overrides cfg.phi_scale
                zcfg = create_gauss_config(lmax, nlat; nlon=nlon)
                zcfg.phi_scale = phi_scale
                @test SHTnsKit.phi_inv_scale(zcfg) ≈ (phi_scale === :quad ? zcfg.nlon / 2π : zcfg.nlon)
                T0 = 0.4 .* _rand_Tlm(rng, lmax)
                target = randn(rng, nlat, nlon)
                # loss_vorticity_grid itself mutates arrays; this is the same loss
                # through operations Zygote differentiates.
                ζ_of = [-(l * (l + 1)) * (l >= m) for l in 0:lmax, m in 0:lmax]
                loss(T) = 0.5 * (2π / nlon) *
                          sum(zcfg.w .* abs2.(synthesis(zcfg, ζ_of .* T; real_output=true) .- target))
                @test loss(T0) ≈ loss_vorticity_grid(zcfg, T0, target)
                @test grad_loss_vorticity_Tlm(zcfg, T0, target) ≈
                      Zygote.gradient(loss, T0)[1] rtol=1e-10
            end
        end
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
        epsilon = 1e-6
        fd = (loss_vorticity_grid(cfgn, T0 .+ epsilon .* h, ζtarget_n) -
              loss_vorticity_grid(cfgn, T0 .- epsilon .* h, ζtarget_n)) / (2epsilon)

        g = grad_loss_vorticity_Tlm(cfgn, T0, ζtarget_n)
        @test isapprox(real(sum(conj(g) .* h)), fd; rtol=2e-5, atol=2e-7)

        loss, combined_g = loss_and_grad_vorticity_Tlm(cfgn, T0, ζtarget_n)
        @test loss ≈ loss_vorticity_grid(cfgn, T0, ζtarget_n)
        @test isapprox(real(sum(conj(combined_g) .* h)), fd;
                       rtol=2e-5, atol=2e-7)
    end
end
