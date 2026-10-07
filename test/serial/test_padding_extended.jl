# SHTnsKit.jl - Padded spatial allocation tests
# Exercises set_allow_padding!, allocate_padded_spatial, allocate_padded_spatial_batch,
# copy_to_padded! / copy_from_padded!, estimate_padding_overhead, get_spat_dist.

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Padding helpers" begin
    @testset "Default state: padding disabled" begin
        cfg = create_gauss_config(6, 8; nlon=13)
        @test cfg.allow_padding == false
        @test is_padding_enabled(cfg) == false
        @test get_nlat_padded(cfg) == cfg.nlat
        @test estimate_padding_overhead(cfg) == 0.0
    end

    @testset "Enable padding" begin
        cfg = create_gauss_config(6, 8; nlon=13)
        set_allow_padding!(cfg)
        @test cfg.allow_padding == true
        @test is_padding_enabled(cfg) == true

        nlat_p = get_nlat_padded(cfg)
        @test nlat_p ≥ cfg.nlat

        # get_spat_dist tracks padded spat_dist if > 0
        n_spat = get_spat_dist(cfg)
        @test n_spat ≥ cfg.nspat

        # Overhead ≥ 0%
        @test estimate_padding_overhead(cfg) ≥ 0.0
    end

    @testset "allocate_padded_spatial shape" begin
        cfg = create_gauss_config(6, 8; nlon=13)
        set_allow_padding!(cfg)

        buf = allocate_padded_spatial(cfg)
        @test size(buf, 1) == get_nlat_padded(cfg)
        @test size(buf, 2) == cfg.nlon
        @test all(iszero, buf)

        buf_f32 = allocate_padded_spatial(cfg, Float32)
        @test eltype(buf_f32) == Float32
    end

    @testset "allocate_padded_spatial_batch shape" begin
        cfg = create_gauss_config(5, 7; nlon=11)
        set_allow_padding!(cfg)

        nfields = 4
        buf = allocate_padded_spatial_batch(cfg, nfields)
        @test size(buf) == (get_nlat_padded(cfg), cfg.nlon, nfields)
        @test eltype(buf) == Float64
    end

    @testset "copy_to_padded! / copy_from_padded! roundtrip" begin
        cfg = create_gauss_config(5, 7; nlon=11)
        set_allow_padding!(cfg)

        src = reshape(collect(1.0:(cfg.nlat * cfg.nlon)), cfg.nlat, cfg.nlon)
        padded = allocate_padded_spatial(cfg)

        copy_to_padded!(padded, src, cfg)
        @test padded[1:cfg.nlat, :] == src
        # Padded rows past nlat stay zero
        if size(padded, 1) > cfg.nlat
            @test all(iszero, padded[cfg.nlat + 1:end, :])
        end

        back = zeros(cfg.nlat, cfg.nlon)
        copy_from_padded!(back, padded, cfg)
        @test back == src
    end

    @testset "Padded copy: dimension checks" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        set_allow_padding!(cfg)
        padded = allocate_padded_spatial(cfg)

        # src wrong rows
        @test_throws DimensionMismatch copy_to_padded!(padded,
            zeros(cfg.nlat + 1, cfg.nlon), cfg)
        # src wrong cols
        @test_throws DimensionMismatch copy_to_padded!(padded,
            zeros(cfg.nlat, cfg.nlon + 1), cfg)
        # dest too short
        @test_throws DimensionMismatch copy_to_padded!(zeros(cfg.nlat - 1, cfg.nlon),
            zeros(cfg.nlat, cfg.nlon), cfg)

        # copy_from_padded!: dest must equal nlat × nlon
        @test_throws DimensionMismatch copy_from_padded!(zeros(cfg.nlat + 1, cfg.nlon),
            padded, cfg)
        @test_throws DimensionMismatch copy_from_padded!(zeros(cfg.nlat, cfg.nlon - 1),
            padded, cfg)
    end

    @testset "disable_padding! resets state" begin
        cfg = create_gauss_config(4, 6; nlon=9)
        set_allow_padding!(cfg)
        @test cfg.allow_padding
        disable_padding!(cfg)
        @test cfg.allow_padding == false
        @test get_nlat_padded(cfg) == cfg.nlat
    end

    @testset "use_rfft into padded views and strided scratch" begin
        # The real-FFT plans assume a column-contiguous output, so a padded
        # `spatial_view` or a strided `fft_scratch` raised "FFTW plan applied
        # to wrong-strides output".
        cfg = create_gauss_config(8, 10; nlon=17)
        set_allow_padding!(cfg)
        @test get_nlat_padded(cfg) > cfg.nlat
        alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        alm[1, 1] = 0.7
        alm[3, 2] = 0.4 - 0.1im
        field = synthesis(cfg, alm)
        bins = cfg.nlon ÷ 2 + 1

        padded = allocate_padded_spatial(cfg)
        @test synthesis!(cfg, spatial_view(cfg, padded), alm; use_rfft=true) ≈ field
        coefficients = similar(alm)
        @test analysis!(cfg, coefficients, spatial_view(cfg, padded); use_rfft=true) ≈ alm

        scratch = view(zeros(ComplexF64, cfg.nlat + 3, bins), 1:cfg.nlat, :)
        @test analysis(cfg, field; fft_scratch=scratch, use_rfft=true) ≈ alm
        @test synthesis(cfg, alm; fft_scratch=scratch, use_rfft=true) ≈ field

        batch = allocate_padded_spatial_batch(cfg, 2)
        batch_view = spatial_view(cfg, batch)
        synthesis_batch!(cfg, batch_view, cat(alm, 2alm; dims=3); use_rfft=true)
        @test batch_view ≈ cat(field, 2field; dims=3)
        batch_coefficients = zeros(ComplexF64, size(alm)..., 2)
        analysis_batch!(cfg, batch_coefficients, batch_view; use_rfft=true)
        @test batch_coefficients ≈ cat(alm, 2alm; dims=3)
    end
end
