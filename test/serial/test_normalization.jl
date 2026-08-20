# SHTnsKit.jl - Normalization and Phase Convention Tests
# Tests for norm_scale_from_orthonormal, cs_phase_factor, convert_alm_norm!

using Test
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Normalization and Phase Conventions" begin
    @testset "norm_scale_from_orthonormal" begin
        # Orthonormal → orthonormal is identity
        for l in 0:10, m in 0:l
            @test SHTnsKit.norm_scale_from_orthonormal(l, m, :orthonormal) ≈ 1.0
        end

        # Orthonormal → fourpi gives sqrt(4π) for all (l,m)
        for l in 0:10, m in 0:l
            @test SHTnsKit.norm_scale_from_orthonormal(l, m, :fourpi) ≈ sqrt(4π)
        end

        # Schmidt semi-normalized: m=0 case
        for l in 0:10
            expected = sqrt(4π / (2l + 1))
            @test SHTnsKit.norm_scale_from_orthonormal(l, 0, :schmidt) ≈ expected
        end

        # Schmidt semi-normalized is independent of the optional real-basis factor
        for l in 1:10, m in 1:l
            expected = sqrt(4π / (2l + 1))
            @test SHTnsKit.norm_scale_from_orthonormal(l, m, :schmidt) ≈ expected
        end

        # Unsupported normalization should throw
        @test_throws ArgumentError SHTnsKit.norm_scale_from_orthonormal(2, 1, :unknown)
    end

    @testset "cs_phase_factor" begin
        # Same convention → factor is 1
        for m in 0:10
            @test SHTnsKit.cs_phase_factor(m, true, true) ≈ 1.0
            @test SHTnsKit.cs_phase_factor(m, false, false) ≈ 1.0
        end

        # Different conventions → factor is (-1)^m
        for m in 0:10
            expected = (-1.0)^m
            @test SHTnsKit.cs_phase_factor(m, true, false) ≈ expected
            @test SHTnsKit.cs_phase_factor(m, false, true) ≈ expected
        end

        # Even m: phase factor is +1 when switching
        @test SHTnsKit.cs_phase_factor(0, true, false) ≈ 1.0
        @test SHTnsKit.cs_phase_factor(2, true, false) ≈ 1.0
        @test SHTnsKit.cs_phase_factor(4, true, false) ≈ 1.0

        # Odd m: phase factor is -1 when switching
        @test SHTnsKit.cs_phase_factor(1, true, false) ≈ -1.0
        @test SHTnsKit.cs_phase_factor(3, true, false) ≈ -1.0
        @test SHTnsKit.cs_phase_factor(5, true, false) ≈ -1.0
    end

    @testset "convert_alm_norm! roundtrip" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Create random coefficients
        src = zeros(ComplexF64, lmax+1, lmax+1)
        for m in 0:lmax, l in m:lmax
            src[l+1, m+1] = randn(ComplexF64)
        end
        src[:, 1] .= real.(src[:, 1])  # m=0 real

        dest = similar(src)
        back = similar(src)

        # Internal → external → internal should roundtrip
        SHTnsKit.convert_alm_norm!(dest, src, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(back, dest, cfg; to_internal=true)
        @test isapprox(back, src; rtol=1e-12)
    end

    @testset "convert_alm_norm! fourpi convention" begin
        lmax = 4
        nlat = lmax + 2
        cfg_fourpi = create_gauss_config(lmax, nlat; norm=:fourpi, cs_phase=true)

        src = zeros(ComplexF64, lmax+1, lmax+1)
        src[1, 1] = 1.0 + 0im  # l=0, m=0
        src[2, 1] = 2.0 + 0im  # l=1, m=0

        dest = similar(src)
        SHTnsKit.convert_alm_norm!(dest, src, cfg_fourpi; to_internal=false)

        # For fourpi norm, conversion from internal divides by sqrt(4π)
        @test isapprox(dest[1, 1], src[1, 1] / sqrt(4π); rtol=1e-12)
    end

    @testset "real_norm is independent of Schmidt normalization" begin
        cfg = create_gauss_config(2, 4; norm=:schmidt, real_norm=true)
        src = zeros(ComplexF64, 3, 3)
        src[2, 2] = 1 + 2im
        dest = similar(src)
        SHTnsKit.convert_alm_norm!(dest, src, cfg; to_internal=false)

        schmidt_scale = sqrt(4π / 3)
        @test dest[2, 2] ≈ sqrt(2) * src[2, 2] / schmidt_scale
    end

    @testset "convert_alm_norm! dimension mismatch" begin
        lmax = 4
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        src = zeros(ComplexF64, lmax+1, lmax+1)
        dest_wrong = zeros(ComplexF64, lmax+2, lmax+1)

        @test_throws DimensionMismatch SHTnsKit.convert_alm_norm!(dest_wrong, src, cfg)
    end

    @testset "transform boundaries honor configured conventions" begin
        lmax = 5
        nlat = lmax + 2
        nlon = 2lmax + 1
        canonical_cfg = create_gauss_config(lmax, nlat; nlon)
        field = randn(nlat, nlon)
        canonical_alm = analysis(canonical_cfg, field)
        canonical_projection = synthesis(canonical_cfg, canonical_alm)

        for (norm, cs_phase, real_norm) in (
            (:fourpi, true, false),
            (:schmidt, false, false),
            (:orthonormal, false, true),
        )
            cfg = create_gauss_config(
                lmax,
                nlat;
                nlon,
                norm,
                cs_phase,
                real_norm,
            )
            expected = similar(canonical_alm)
            SHTnsKit.convert_alm_norm!(
                expected,
                canonical_alm,
                cfg;
                to_internal=false,
            )

            @test analysis(cfg, field) ≈ expected rtol=2e-12 atol=2e-12
            @test synthesis(cfg, expected) ≈ canonical_projection rtol=2e-12 atol=2e-12

            inplace = similar(expected)
            analysis!(cfg, inplace, field)
            @test inplace ≈ expected rtol=2e-12 atol=2e-12
        end
    end

    @testset "configured conventions compose across public transform families" begin
        lmax = 4
        canonical_cfg = create_gauss_config(lmax, 7; nlon=11)
        cfg = create_gauss_config(
            lmax,
            7;
            nlon=11,
            norm=:schmidt,
            real_norm=true,
            cs_phase=false,
        )

        canonical_Q = zeros(ComplexF64, lmax + 1, lmax + 1)
        canonical_S = zeros(ComplexF64, size(canonical_Q))
        canonical_T = zeros(ComplexF64, size(canonical_Q))
        canonical_Q[1, 1] = 0.7
        canonical_Q[3, 2] = -0.2 + 0.5im
        canonical_S[3, 2] = 0.3 - 0.2im
        canonical_T[4, 3] = -0.1 + 0.4im

        external_Q = similar(canonical_Q)
        external_S = similar(canonical_S)
        external_T = similar(canonical_T)
        SHTnsKit.convert_alm_norm!(external_Q, canonical_Q, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(external_S, canonical_S, cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(external_T, canonical_T, cfg; to_internal=false)

        scalar_reference = synthesis(canonical_cfg, canonical_Q)
        vector_reference = synthesis_sphtor(canonical_cfg, canonical_S, canonical_T)
        qst_reference = synthesis_qst(canonical_cfg, canonical_Q, canonical_S, canonical_T)

        @test all(isapprox.(
            synthesis_sphtor(cfg, external_S, external_T),
            vector_reference;
            rtol=2e-12,
            atol=2e-12,
        ))
        analyzed_S, analyzed_T = analysis_sphtor(cfg, vector_reference...)
        @test analyzed_S ≈ external_S rtol=2e-10 atol=2e-11
        @test analyzed_T ≈ external_T rtol=2e-10 atol=2e-11
        @test all(isapprox.(
            synthesis_qst(cfg, external_Q, external_S, external_T),
            qst_reference;
            rtol=2e-12,
            atol=2e-12,
        ))

        packed = SHTnsKit.pack_lm(cfg, external_Q)
        @test reshape(synthesis_packed(cfg, packed), cfg.nlat, cfg.nlon) ≈
              scalar_reference rtol=2e-12 atol=2e-12

        batch = reshape(external_Q, lmax + 1, lmax + 1, 1)
        @test synthesis_batch(cfg, batch)[:, :, 1] ≈ scalar_reference rtol=2e-12 atol=2e-12
        @test analysis_batch(cfg, reshape(scalar_reference, cfg.nlat, cfg.nlon, 1))[:, :, 1] ≈
              external_Q rtol=2e-10 atol=2e-11

        plan = SHTPlan(cfg)
        planned = zeros(cfg.nlat, cfg.nlon)
        synthesis!(plan, planned, external_Q)
        @test planned ≈ scalar_reference rtol=2e-12 atol=2e-12
        planned_back = similar(external_Q)
        analysis!(plan, planned_back, scalar_reference)
        @test planned_back ≈ external_Q rtol=2e-10 atol=2e-11

        @test energy_scalar(cfg, external_Q) ≈
              energy_scalar(canonical_cfg, canonical_Q) rtol=2e-12 atol=2e-12
        @test energy_scalar_l_spectrum(cfg, external_Q) ≈
              energy_scalar_l_spectrum(canonical_cfg, canonical_Q) rtol=2e-12 atol=2e-12
        @test enstrophy(cfg, external_T) ≈
              enstrophy(canonical_cfg, canonical_T) rtol=2e-12 atol=2e-12

        ql_canonical = ComplexF64[0.2, -0.4, 0.7, 0.1, -0.3]
        ql_external = [
            ql_canonical[l + 1] /
            SHTnsKit.coefficient_scale_to_canonical(cfg, l, 0)
            for l in 0:lmax
        ]
        axisymmetric_reference = synthesis_axisym(canonical_cfg, ql_canonical)
        @test synthesis_axisym(cfg, ql_external) ≈ axisymmetric_reference rtol=2e-12 atol=2e-12
        @test analysis_axisym(cfg, axisymmetric_reference) ≈ ql_external rtol=2e-10 atol=2e-11

        m = 2
        ml_canonical = ComplexF64[0.2 + 0.1im, -0.3 + 0.4im, 0.5 - 0.2im]
        ml_external = [
            ml_canonical[l - m + 1] /
            SHTnsKit.coefficient_scale_to_canonical(cfg, l, m)
            for l in m:lmax
        ]
        ml_reference = synthesis_packed_ml(canonical_cfg, m, ml_canonical, lmax)
        @test synthesis_packed_ml(cfg, m, ml_external, lmax) ≈ ml_reference rtol=2e-12 atol=2e-12
        @test analysis_packed_ml(cfg, m, ml_reference, lmax) ≈ ml_external rtol=2e-10 atol=2e-11

        complex_count = nlm_cplx_calc(lmax, lmax, 1)
        complex_canonical = zeros(ComplexF64, complex_count)
        complex_canonical[LM_cplx_index(lmax, lmax, 2, 1) + 1] = 0.4 - 0.3im
        complex_canonical[LM_cplx_index(lmax, lmax, 3, -2) + 1] = -0.2 + 0.5im
        complex_external = similar(complex_canonical)
        SHTnsKit.convert_alm_norm!(
            complex_external,
            complex_canonical,
            cfg;
            to_internal=false,
        )
        complex_reference = synthesis_packed_cplx(canonical_cfg, complex_canonical)
        @test synthesis_packed_cplx(cfg, complex_external) ≈ complex_reference rtol=2e-12 atol=2e-12
        @test analysis_packed_cplx(cfg, complex_reference) ≈ complex_external rtol=2e-10 atol=2e-11
    end

    @testset "adjoint boundaries follow configured conventions" begin
        lmax = 3
        canonical_cfg = create_gauss_config(lmax, 6; nlon=9)
        cfg = create_gauss_config(
            lmax,
            6;
            nlon=9,
            norm=:schmidt,
            real_norm=true,
            cs_phase=false,
        )
        grid_bar = reshape(collect(1.0:(cfg.nlat * cfg.nlon)), cfg.nlat, cfg.nlon) ./ 17

        canonical_bar = SHTnsKit._adjoint_synthesis(canonical_cfg, grid_bar)
        expected_bar = similar(canonical_bar)
        SHTnsKit.convert_alm_norm!(expected_bar, canonical_bar, cfg; to_internal=true)
        @test SHTnsKit._adjoint_synthesis(cfg, grid_bar) ≈ expected_bar rtol=2e-12 atol=2e-12

        configured_input = zeros(ComplexF64, lmax + 1, lmax + 1)
        configured_input[2, 1] = 0.2
        configured_input[3, 2] = -0.3 + 0.4im
        canonical_input = similar(configured_input)
        SHTnsKit.convert_alm_norm!(canonical_input, configured_input, cfg; to_internal=false)
        @test SHTnsKit._adjoint_analysis(cfg, configured_input) ≈
              SHTnsKit._adjoint_analysis(canonical_cfg, canonical_input) rtol=2e-12 atol=2e-12
    end
end
