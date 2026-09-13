# Regression coverage for dense diagnostics with mres-limited configurations.

using Test
using SHTnsKit

@testset "mres diagnostic masks" begin
    lmax = 6
    cfg = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1, mres=2)

    # `zero`, not `similar`: `similar` hands back UNINITIALIZED memory. Run
    # standalone the pages are freshly mmap'd (all zero) and this passes, but
    # late in the full suite the allocator recycles pages and these matrices
    # come back full of denormal garbage. The diagnostics then agree exactly
    # (`a == b`), yet `isapprox` still fails, because `LinearAlgebra.norm` of a
    # matrix carrying such denormals overflows its scaling step to NaN and
    # `0 <= NaN` is false. Zero them explicitly.
    Qclean = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
    Sclean = zero(Qclean)
    Tclean = zero(Qclean)

    # Representable orders for mres=2.
    Qclean[3, 1] = 0.25
    Qclean[5, 3] = 0.4 - 0.3im       # (l,m) = (4,2)
    Sclean[4, 1] = -0.2
    Sclean[6, 3] = 0.1 + 0.35im     # (l,m) = (5,2)
    Tclean[7, 5] = -0.3 + 0.15im    # (l,m) = (6,4)

    Qdirty = copy(Qclean)
    Sdirty = copy(Sclean)
    Tdirty = copy(Tclean)

    # These columns cannot be represented by an mres=2 transform and must not
    # contribute to a diagnostic, even if a caller left nonzero values there.
    Qdirty[4, 2] = 10.0 - 7.0im     # (l,m) = (3,1)
    Qdirty[6, 4] = -9.0 + 2.0im     # (l,m) = (5,3)
    Sdirty[5, 2] = 8.0 + 4.0im      # (l,m) = (4,1)
    Sdirty[7, 6] = -6.0 + 3.0im     # (l,m) = (6,5)
    Tdirty[6, 4] = 5.0 - 11.0im     # (l,m) = (5,3)

    @testset "totals" begin
        @test energy_scalar(cfg, Qdirty) ≈ energy_scalar(cfg, Qclean)
        @test energy_vector(cfg, Sdirty, Tdirty) ≈
              energy_vector(cfg, Sclean, Tclean)
        @test enstrophy(cfg, Tdirty) ≈ enstrophy(cfg, Tclean)
    end

    @testset "energy spectra" begin
        @test energy_scalar_l_spectrum(cfg, Qdirty) ≈
              energy_scalar_l_spectrum(cfg, Qclean)
        @test energy_scalar_m_spectrum(cfg, Qdirty) ≈
              energy_scalar_m_spectrum(cfg, Qclean)
        @test energy_scalar_lm(cfg, Qdirty) ≈ energy_scalar_lm(cfg, Qclean)

        @test energy_vector_l_spectrum(cfg, Sdirty, Tdirty) ≈
              energy_vector_l_spectrum(cfg, Sclean, Tclean)
        @test energy_vector_m_spectrum(cfg, Sdirty, Tdirty) ≈
              energy_vector_m_spectrum(cfg, Sclean, Tclean)
        @test energy_vector_lm(cfg, Sdirty, Tdirty) ≈
              energy_vector_lm(cfg, Sclean, Tclean)
    end

    @testset "vorticity diagnostics" begin
        @test enstrophy_l_spectrum(cfg, Tdirty) ≈
              enstrophy_l_spectrum(cfg, Tclean)
        @test enstrophy_m_spectrum(cfg, Tdirty) ≈
              enstrophy_m_spectrum(cfg, Tclean)
        @test enstrophy_lm(cfg, Tdirty) ≈ enstrophy_lm(cfg, Tclean)
        @test vorticity_spectral(cfg, Tdirty) ≈ vorticity_spectral(cfg, Tclean)
    end

    @testset "gradients" begin
        @test grad_energy_scalar_alm(cfg, Qdirty) ≈
              grad_energy_scalar_alm(cfg, Qclean)
        gdS, gdT = grad_energy_vector_Slm_Tlm(cfg, Sdirty, Tdirty)
        gcS, gcT = grad_energy_vector_Slm_Tlm(cfg, Sclean, Tclean)
        @test gdS ≈ gcS
        @test gdT ≈ gcT
        @test grad_enstrophy_Tlm(cfg, Tdirty) ≈ grad_enstrophy_Tlm(cfg, Tclean)
    end
end
