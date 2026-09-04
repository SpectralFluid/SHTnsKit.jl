# SHTnsKit.jl - Extended rotation tests
# Covers bits of src/rotations.jl not in test_rotations.jl:
# - WignerCache
# - wigner_d_matrix_deriv finite-difference check
# - shtns_rotation_wigner_d_matrix (row-major fill)
# - shtns_rotation_create / destroy
# - ZXZ convention application
# - Composed rotations vs. direct Euler

using Test
using Random
using LinearAlgebra
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Rotations (extended)" begin
    @testset "WignerCache consistency with direct build" begin
        lmax = 5
        β = 0.37
        cache = SHTnsKit.WignerCache(lmax, β)
        @test cache.β == β
        for l in 0:lmax
            d_cached = SHTnsKit.wigner_d(cache, l)
            d_direct = SHTnsKit.wigner_d_matrix(l, β)
            @test d_cached == d_direct
            @test size(d_cached) == (2l + 1, 2l + 1)
        end
        @test_throws ArgumentError SHTnsKit.wigner_d(cache, lmax + 1)
        @test_throws ArgumentError SHTnsKit.WignerCache(-1, β)
    end

    @testset "Wigner-d remains orthogonal at l=64" begin
        d = SHTnsKit.wigner_d_matrix(64, 0.7)
        @test opnorm(transpose(d) * d - I, Inf) < 1e-10

        # The derivative of an orthogonal matrix must be tangent to O(n):
        # dᵀ ḋ + ḋᵀ d = 0. This also guards the high-l derivative path
        # used by the rotation AD rules.
        dβ = SHTnsKit.wigner_d_matrix_deriv(64, 0.7)
        @test opnorm(transpose(d) * dβ + transpose(dβ) * d, Inf) < 1e-9
    end

    @testset "wigner_d_matrix_deriv: finite-difference check" begin
        h = 1e-6
        for l in 0:3
            β = 0.4
            d_plus  = SHTnsKit.wigner_d_matrix(l, β + h)
            d_minus = SHTnsKit.wigner_d_matrix(l, β - h)
            dd_fd = (d_plus .- d_minus) ./ (2h)
            dd = SHTnsKit.wigner_d_matrix_deriv(l, β)
            @test isapprox(dd, dd_fd; rtol=1e-5, atol=1e-7)
        end
    end

    @testset "wigner_d_matrix_deriv: l=0 is zero" begin
        dd = SHTnsKit.wigner_d_matrix_deriv(0, 0.5)
        @test size(dd) == (1, 1)
        @test dd[1, 1] == 0.0
    end

    @testset "shtns_rotation_wigner_d_matrix row-major layout" begin
        lmax = 3
        r = SHTRotation(lmax, lmax)
        β = 0.6
        shtns_rotation_set_angles_ZYZ(r, 0.0, β, 0.0)
        for l in 0:lmax
            n = 2l + 1
            buf = zeros(n * n)
            ret = shtns_rotation_wigner_d_matrix(r, l, buf)
            @test ret == n
            d_ref = SHTnsKit.wigner_d_matrix(l, β)
            # Row-major unpack
            for i in 1:n, j in 1:n
                @test buf[(i - 1) * n + j] ≈ d_ref[i, j]
            end
        end
        # Buffer-too-small → DimensionMismatch
        @test_throws DimensionMismatch shtns_rotation_wigner_d_matrix(r, 3, zeros(1))
    end

    @testset "shtns_rotation_create / destroy" begin
        r = shtns_rotation_create(4, 4, 0)
        @test r isa SHTRotation
        @test r.lmax == 4
        @test r.mmax == 4
        @test shtns_rotation_destroy(r) === nothing
        @test_throws ArgumentError shtns_rotation_create(4, 4, 1)  # non-orthonormal not supported
    end

    @testset "ZYZ vs ZXZ: 90° x-rotation consistency" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(400)
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax + 1] .= real.(Qlm[1:lmax + 1])

        Rlm_x = similar(Qlm)
        SH_Xrotate90(cfg, Qlm, Rlm_x)
        # Energy must be preserved
        @test isapprox(energy_scalar_packed(cfg, Rlm_x),
                       energy_scalar_packed(cfg, Qlm); rtol=1e-10)

        # Four 90° X-rotations return to identity
        tmp1 = similar(Qlm); tmp2 = similar(Qlm); tmp3 = similar(Qlm); tmp4 = similar(Qlm)
        SH_Xrotate90(cfg, Qlm, tmp1)
        SH_Xrotate90(cfg, tmp1, tmp2)
        SH_Xrotate90(cfg, tmp2, tmp3)
        SH_Xrotate90(cfg, tmp3, tmp4)
        @test isapprox(tmp4, Qlm; rtol=1e-8, atol=1e-10)
    end

    @testset "Fast axis helpers use the general rotation orientation" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(405)
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax + 1] .= real.(Qlm[1:lmax + 1])

        α = 0.37
        rz = SHTRotation(lmax, lmax)
        shtns_rotation_set_angles_ZYZ(rz, α, 0.0, 0.0)
        z_general = similar(Qlm)
        z_fast = similar(Qlm)
        shtns_rotation_apply_real(rz, Qlm, z_general)
        SH_Zrotate(cfg, Qlm, α, z_fast)
        @test z_fast ≈ z_general rtol=2e-13 atol=2e-13

        rx = SHTRotation(lmax, lmax)
        shtns_rotation_set_angle_axis(rx, π/2, 1.0, 0.0, 0.0)
        x_general = similar(Qlm)
        x_fast = similar(Qlm)
        shtns_rotation_apply_real(rx, Qlm, x_general)
        SH_Xrotate90(cfg, Qlm, x_fast)
        @test x_fast ≈ x_general rtol=2e-13 atol=2e-13
    end

    @testset "Serial axis wrappers honor configured coefficient conventions" begin
        lmax = 4
        canonical_cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1,
                                  norm=:schmidt, real_norm=true, cs_phase=false)
        rng = MersenneTwister(406)
        Qcanonical = randn(rng, ComplexF64, canonical_cfg.nlm)
        Qcanonical[1:lmax + 1] .= real.(Qcanonical[1:lmax + 1])
        Qconfigured = similar(Qcanonical)
        SHTnsKit.convert_alm_norm!(Qconfigured, Qcanonical, cfg; to_internal=false)

        rotations = (
            (q, out, c) -> SH_Yrotate(c, q, 0.41, out),
            (q, out, c) -> SH_Yrotate90(c, q, out),
            (q, out, c) -> SH_Xrotate90(c, q, out),
        )
        for rotate! in rotations
            expected = similar(Qcanonical)
            rotate!(Qcanonical, expected, canonical_cfg)
            got_configured = similar(Qconfigured)
            rotate!(Qconfigured, got_configured, cfg)
            got_canonical = similar(Qcanonical)
            SHTnsKit.convert_alm_norm!(got_canonical, got_configured, cfg; to_internal=true)
            @test got_canonical ≈ expected rtol=2e-12 atol=2e-12
        end
    end

    @testset "Composition: two Y-rotations = single Y-rotation" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(401)
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax + 1] .= real.(Qlm[1:lmax + 1])

        β1, β2 = 0.3, 0.5
        t1 = similar(Qlm); t2 = similar(Qlm); direct = similar(Qlm)
        SH_Yrotate(cfg, Qlm, β1, t1)
        SH_Yrotate(cfg, t1, β2, t2)
        SH_Yrotate(cfg, Qlm, β1 + β2, direct)

        @test isapprox(t2, direct; rtol=1e-9, atol=1e-11)
    end

    @testset "General rotation identity (α=β=γ=0)" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(402)
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax + 1] .= real.(Qlm[1:lmax + 1])

        r = SHTRotation(lmax, lmax)  # default α=β=γ=0
        Rlm = similar(Qlm)
        shtns_rotation_apply_real(r, Qlm, Rlm)
        @test isapprox(Rlm, Qlm; rtol=1e-12, atol=1e-14)
    end

    @testset "Rotation energy preservation (real field, packed)" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(403)
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax + 1] .= real.(Qlm[1:lmax + 1])

        r = SHTRotation(lmax, lmax)
        shtns_rotation_set_angles_ZYZ(r, 0.7, 1.1, -0.4)
        Rlm = similar(Qlm)
        shtns_rotation_apply_real(r, Qlm, Rlm)

        @test isapprox(energy_scalar_packed(cfg, Rlm),
                       energy_scalar_packed(cfg, Qlm); rtol=1e-9)
    end

    @testset "Angle-axis with zero-length axis → identity" begin
        r = SHTRotation(3, 3)
        r.α = 0.9; r.β = 0.9; r.γ = 0.9
        shtns_rotation_set_angle_axis(r, 1.2, 0.0, 0.0, 0.0)
        @test r.α == 0.0 && r.β == 0.0 && r.γ == 0.0
        @test r.conv == :ZYZ
    end

    @testset "Dimension checks on rotation apply" begin
        lmax, mmax = 3, 3
        r = SHTRotation(lmax, mmax)
        expected = nlm_calc(lmax, mmax, 1)
        @test_throws DimensionMismatch shtns_rotation_apply_real(r,
            zeros(ComplexF64, expected - 1), zeros(ComplexF64, expected))
        @test_throws DimensionMismatch shtns_rotation_apply_real(r,
            zeros(ComplexF64, expected), zeros(ComplexF64, expected + 1))

        nlmc = nlm_cplx_calc(lmax, mmax, 1)
        @test_throws DimensionMismatch shtns_rotation_apply_cplx(r,
            zeros(ComplexF64, nlmc), zeros(ComplexF64, nlmc - 1))
    end

    @testset "ZXZ convention is applied, not just stored" begin
        lmax = 3
        α, β, γ = 0.3, 0.5, 0.2
        r_xz = SHTRotation(lmax, lmax)
        shtns_rotation_set_angles_ZXZ(r_xz, α, β, γ)
        @test r_xz.conv == :ZXZ

        # Rx(β) = Rz(-π/2) Ry(β) Rz(π/2), hence this is the
        # equivalent ZYZ triple for Rz(α) Rx(β) Rz(γ).
        r_equiv = SHTRotation(lmax, lmax)
        shtns_rotation_set_angles_ZYZ(r_equiv, α - π/2, β, γ + π/2)
        r_same_numbers = SHTRotation(lmax, lmax)
        shtns_rotation_set_angles_ZYZ(r_same_numbers, α, β, γ)

        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(404)
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax + 1] .= real.(Qlm[1:lmax + 1])

        R_xz = similar(Qlm)
        R_equiv = similar(Qlm)
        R_same_numbers = similar(Qlm)
        shtns_rotation_apply_real(r_xz, Qlm, R_xz)
        shtns_rotation_apply_real(r_equiv, Qlm, R_equiv)
        shtns_rotation_apply_real(r_same_numbers, Qlm, R_same_numbers)
        @test R_xz ≈ R_equiv rtol=2e-12 atol=2e-12
        @test !isapprox(R_xz, R_same_numbers; rtol=1e-8, atol=1e-10)
        @test isapprox(energy_scalar_packed(cfg, R_xz),
                       energy_scalar_packed(cfg, Qlm); rtol=1e-9)
    end
end
