# SHTnsKit.jl - Energy Diagnostics Tests
# Tests for Parseval identity and energy spectra

using Test
using Random
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Energy Diagnostics" begin
    @testset "Convention metrics use the cached scale matrix" begin
        lmax = 8
        alm = zeros(ComplexF64, lmax + 1, lmax + 1)
        alm[3, 2] = 0.4 - 0.2im

        cfg = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1,
                                  norm=:schmidt, real_norm=true, cs_phase=false)
        @test isempty(cfg.norm_scale_matrix)
        energy_scalar(cfg, alm)
        @test size(cfg.norm_scale_matrix) == (lmax + 1, lmax + 1)
        cached = cfg.norm_scale_matrix
        energy_vector(cfg, alm, alm)
        @test cfg.norm_scale_matrix === cached

        cfg.norm_scale_matrix = Matrix{Float64}(undef, 0, 0)
        energy_scalar_l_spectrum(cfg, alm)
        @test size(cfg.norm_scale_matrix) == (lmax + 1, lmax + 1)

        cfg.norm_scale_matrix = Matrix{Float64}(undef, 0, 0)
        enstrophy_l_spectrum(cfg, alm)
        @test size(cfg.norm_scale_matrix) == (lmax + 1, lmax + 1)

        canonical = create_gauss_config(lmax, lmax + 2; nlon=2lmax + 1)
        @test isempty(canonical.norm_scale_matrix)
        energy_scalar(canonical, alm)
        energy_scalar_l_spectrum(canonical, alm)
        enstrophy(canonical, alm)
        @test isempty(canonical.norm_scale_matrix)
    end

    @testset "Scalar Parseval identity" begin
        lmax = 10
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(100)

        # Random coefficients
        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        f = synthesis(cfg, alm; real_output=true)

        E_spec = energy_scalar(cfg, alm)
        E_grid = grid_energy_scalar(cfg, f)

        @test isapprox(E_spec, E_grid; rtol=1e-10, atol=1e-12)
        VERBOSE && @info "Scalar Parseval" E_spec E_grid rel=abs(E_spec-E_grid)/E_grid
    end

    @testset "Vector Parseval identity" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(101)

        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Slm[1, :] .= 0; Tlm[1, :] .= 0

        Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)

        E_spec = energy_vector(cfg, Slm, Tlm)
        E_grid = grid_energy_vector(cfg, Vt, Vp)

        @test isapprox(E_spec, E_grid; rtol=1e-9, atol=1e-11)
    end

    @testset "Single-mode Parseval scalar" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Single mode l=3, m=2
        alm = zeros(ComplexF64, lmax+1, lmax+1)
        alm[4, 3] = 1.0 + 0.5im

        f = synthesis(cfg, alm; real_output=true)

        E_spec = energy_scalar(cfg, alm)
        E_grid = grid_energy_scalar(cfg, f)

        @test isapprox(E_spec, E_grid; rtol=1e-10, atol=1e-12)
    end

    @testset "Single-mode Parseval vector" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Single S mode
        Slm = zeros(ComplexF64, lmax+1, lmax+1)
        Tlm = zeros(ComplexF64, lmax+1, lmax+1)
        Slm[4, 2] = 1.0 + 0im  # l=3, m=1

        Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        E_spec = energy_vector(cfg, Slm, Tlm)
        E_grid = grid_energy_vector(cfg, Vt, Vp)

        @test isapprox(E_spec, E_grid; rtol=1e-9, atol=1e-11)

        # Single T mode
        fill!(Slm, 0)
        Tlm[5, 3] = 1.0 + 0im  # l=4, m=2

        Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm; real_output=true)
        E_spec = energy_vector(cfg, Slm, Tlm)
        E_grid = grid_energy_vector(cfg, Vt, Vp)

        @test isapprox(E_spec, E_grid; rtol=1e-9, atol=1e-11)
    end

    @testset "Scalar l-spectrum" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(102)

        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        E_total = energy_scalar(cfg, alm)
        El = energy_scalar_l_spectrum(cfg, alm)

        @test length(El) == lmax + 1
        @test isapprox(sum(El), E_total; rtol=1e-10)
        @test all(El .>= 0)  # Energy is non-negative
    end

    @testset "Scalar m-spectrum" begin
        lmax = 8
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(103)

        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        E_total = energy_scalar(cfg, alm)
        Em = energy_scalar_m_spectrum(cfg, alm)

        @test length(Em) == cfg.mmax + 1
        @test isapprox(sum(Em), E_total; rtol=1e-10)
        @test all(Em .>= 0)
    end

    @testset "Scalar per-mode energy" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(104)

        alm = randn(rng, ComplexF64, lmax+1, lmax+1)
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        E_total = energy_scalar(cfg, alm)
        Elm = energy_scalar_lm(cfg, alm)

        @test size(Elm) == (lmax+1, cfg.mmax+1)
        @test isapprox(sum(Elm), E_total; rtol=1e-10)
        @test all(Elm .>= 0)
    end

    @testset "Vector energy spectra" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(105)

        Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
        Slm[:, 1] .= real.(Slm[:, 1])
        Tlm[:, 1] .= real.(Tlm[:, 1])
        Slm[1, :] .= 0; Tlm[1, :] .= 0

        E_total = energy_vector(cfg, Slm, Tlm)

        # l-spectrum
        El = energy_vector_l_spectrum(cfg, Slm, Tlm)
        @test isapprox(sum(El), E_total; rtol=1e-10)
        @test all(El .>= 0)

        # m-spectrum
        Em = energy_vector_m_spectrum(cfg, Slm, Tlm)
        @test isapprox(sum(Em), E_total; rtol=1e-10)
        @test all(Em .>= 0)

        # lm matrix
        Elm = energy_vector_lm(cfg, Slm, Tlm)
        @test isapprox(sum(Elm), E_total; rtol=1e-10)
    end

    @testset "Packed energy functions" begin
        lmax = 6
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(106)

        # Scalar packed
        f = randn(rng, nlat, nlon)
        Qlm = analysis_packed(cfg, vec(f))
        alm = analysis(cfg, f)

        E_packed = energy_scalar_packed(cfg, Qlm)
        E_matrix = energy_scalar(cfg, alm)
        @test isapprox(E_packed, E_matrix; rtol=1e-10)

        # Vector packed
        Vt = randn(rng, nlat, nlon)
        Vp = randn(rng, nlat, nlon)
        Slm_mat, Tlm_mat = analysis_sphtor(cfg, Vt, Vp)

        # Pack to vectors
        Sp = zeros(ComplexF64, cfg.nlm)
        Tp = zeros(ComplexF64, cfg.nlm)
        for m in 0:cfg.mmax
            for l in m:cfg.lmax
                idx = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                Sp[idx] = Slm_mat[l+1, m+1]
                Tp[idx] = Tlm_mat[l+1, m+1]
            end
        end

        E_vec_packed = energy_vector_packed(cfg, Sp, Tp)
        E_vec_matrix = energy_vector(cfg, Slm_mat, Tlm_mat)
        @test isapprox(E_vec_packed, E_vec_matrix; rtol=1e-9)
    end

    @testset "Energy non-negativity" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        rng = MersenneTwister(107)

        # Test multiple random configurations
        for _ in 1:10
            alm = randn(rng, ComplexF64, lmax+1, lmax+1)
            alm[:, 1] .= real.(alm[:, 1])
            for m in 0:lmax, l in 0:(m-1)
                alm[l+1, m+1] = 0
            end

            @test energy_scalar(cfg, alm) >= 0

            Slm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Tlm = randn(rng, ComplexF64, lmax+1, lmax+1)
            Slm[:, 1] .= real.(Slm[:, 1])
            Tlm[:, 1] .= real.(Tlm[:, 1])

            @test energy_vector(cfg, Slm, Tlm) >= 0
        end
    end

    @testset "spectral energy rejects a mis-sized spectrum" begin
        # The accumulation loops are `@inbounds`; a too-small matrix used to
        # return a silently wrong number instead of raising.
        cfg = create_gauss_config(6, 8)
        small = zeros(ComplexF64, 2, 2)
        ok = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        @test_throws DimensionMismatch energy_scalar(cfg, small)
        @test_throws DimensionMismatch energy_vector(cfg, small, ok)
        @test_throws DimensionMismatch energy_vector(cfg, ok, small)
        @test energy_scalar(cfg, ok) == 0.0
    end

end
