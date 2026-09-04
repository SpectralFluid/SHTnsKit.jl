# Regression coverage for mres-limited dense transform schedules.

using Test
using SHTnsKit

@testset "mres transform schedules" begin
    lmax = 4
    nlat = lmax + 2
    nlon = 2lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon, mres=2)
    cfg_all_m = create_gauss_config(lmax, nlat; nlon=nlon)

    excluded = zeros(ComplexF64, lmax + 1, lmax + 1)
    excluded[3, 2] = 0.7 - 0.2im # (l,m) = (2,1), excluded by mres=2
    scalar_field = synthesis(cfg_all_m, excluded; real_output=true)

    @test synthesis(cfg, excluded; real_output=true) ≈ zeros(nlat, nlon) atol=1e-13
    @test synthesis_point(cfg, excluded, cfg.x[2], 0.4) ≈ 0.0 atol=1e-13
    scalar_coeffs = analysis(cfg, scalar_field)
    @test scalar_coeffs[:, 2:2:end] ≈ zeros(ComplexF64, lmax + 1, 2) atol=1e-13

    packed = analysis_packed(cfg, vec(scalar_field))
    @test SHTnsKit.unpack_lm(cfg, packed) ≈ scalar_coeffs atol=1e-13

    allowed = zeros(ComplexF64, lmax + 1, lmax + 1)
    allowed[4, 3] = 0.4 + 0.3im # (l,m) = (3,2), retained by mres=2
    allowed_field = synthesis(cfg, allowed; real_output=true)
    @test analysis(cfg, allowed_field) ≈ allowed rtol=1e-11 atol=1e-12
    @test reshape(synthesis_packed(cfg, SHTnsKit.pack_lm(cfg, allowed)), nlat, nlon) ≈
          allowed_field rtol=1e-12 atol=1e-13

    plan = SHTPlan(cfg)
    planned_field = zeros(nlat, nlon)
    synthesis!(plan, planned_field, excluded)
    @test planned_field ≈ zeros(nlat, nlon) atol=1e-13
    planned_coeffs = zeros(ComplexF64, lmax + 1, lmax + 1)
    analysis!(plan, planned_coeffs, scalar_field)
    @test planned_coeffs[:, 2:2:end] ≈ zeros(ComplexF64, lmax + 1, 2) atol=1e-13

    excluded_batch = reshape(excluded, lmax + 1, lmax + 1, 1)
    field_batch = reshape(scalar_field, nlat, nlon, 1)
    @test synthesis_batch(cfg, excluded_batch) ≈ zeros(nlat, nlon, 1) atol=1e-13
    batch_coeffs = analysis_batch(cfg, field_batch)
    @test batch_coeffs[:, 2:2:end, :] ≈ zeros(ComplexF64, lmax + 1, 2, 1) atol=1e-13

    excluded_s = copy(excluded)
    excluded_t = zeros(ComplexF64, size(excluded))
    vector_field = synthesis_sphtor(cfg_all_m, excluded_s, excluded_t; real_output=true)

    Vt, Vp = synthesis_sphtor(cfg, excluded_s, excluded_t; real_output=true)
    @test Vt ≈ zeros(nlat, nlon) atol=1e-13
    @test Vp ≈ zeros(nlat, nlon) atol=1e-13
    S, T = analysis_sphtor(cfg, vector_field...)
    @test S[:, 2:2:end] ≈ zeros(ComplexF64, lmax + 1, 2) atol=1e-13
    @test T[:, 2:2:end] ≈ zeros(ComplexF64, lmax + 1, 2) atol=1e-13

    planned_Vt = zeros(nlat, nlon)
    planned_Vp = zeros(nlat, nlon)
    synthesis_sphtor!(plan, planned_Vt, planned_Vp, excluded_s, excluded_t)
    @test planned_Vt ≈ zeros(nlat, nlon) atol=1e-13
    @test planned_Vp ≈ zeros(nlat, nlon) atol=1e-13
    planned_S = zeros(ComplexF64, lmax + 1, lmax + 1)
    planned_T = zeros(ComplexF64, lmax + 1, lmax + 1)
    analysis_sphtor!(plan, planned_S, planned_T, vector_field...)
    @test planned_S[:, 2:2:end] ≈ zeros(ComplexF64, lmax + 1, 2) atol=1e-13
    @test planned_T[:, 2:2:end] ≈ zeros(ComplexF64, lmax + 1, 2) atol=1e-13

    excluded_s_batch = reshape(excluded_s, lmax + 1, lmax + 1, 1)
    excluded_t_batch = reshape(excluded_t, lmax + 1, lmax + 1, 1)
    batch_Vt, batch_Vp = synthesis_sphtor_batch(cfg, excluded_s_batch, excluded_t_batch)
    @test batch_Vt ≈ zeros(nlat, nlon, 1) atol=1e-13
    @test batch_Vp ≈ zeros(nlat, nlon, 1) atol=1e-13
    vector_field_batch = (
        reshape(vector_field[1], nlat, nlon, 1),
        reshape(vector_field[2], nlat, nlon, 1),
    )
    batch_S, batch_T = analysis_sphtor_batch(cfg, vector_field_batch...)
    @test batch_S[:, 2:2:end, :] ≈ zeros(ComplexF64, lmax + 1, 2, 1) atol=1e-13
    @test batch_T[:, 2:2:end, :] ≈ zeros(ComplexF64, lmax + 1, 2, 1) atol=1e-13
end
