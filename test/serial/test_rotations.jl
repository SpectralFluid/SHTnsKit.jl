# SHTnsKit.jl - Rotation Tests
# Tests for spectral rotations and Wigner-d matrices

using Test
using Random
using LinearAlgebra
using SHTnsKit

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Rotations" begin
    @testset "Z-axis rotation phase" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(90)

        # Random coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])  # m=0 real

        # Rotation angle
        α = π/4
        Rlm = similar(Qlm)
        SH_Zrotate(cfg, Qlm, α, Rlm)

        # Z-rotation multiplies by exp(-imα) — the ACTIVE rotation of the field
        # by +α about ẑ, i.e. g(θ,φ) = f(θ,φ-α). See the convention section of
        # `SH_Zrotate`'s docstring, and the ground-truth testset below that pins
        # this against a real spatial φ shift and against the ZYZ engine.
        for m in 0:cfg.mmax
            for l in m:cfg.lmax
                idx = LM_index(cfg.lmax, cfg.mres, l, m) + 1
                expected = Qlm[idx] * cis(-m * α)
                @test isapprox(Rlm[idx], expected; rtol=1e-12, atol=1e-14)
            end
        end
    end

    @testset "Z-rotation sign convention is pinned to physical ground truth" begin
        # The `exp(-imα)` sign is not a free choice: flipping it to `exp(+imα)`
        # (the passive convention) silently desynchronises SH_Zrotate from the
        # actual spatial rotation, from the general Wigner engine, and from the
        # distributed twins. Pin all three so it cannot drift again.
        lmax = 6
        nlon = 2 * lmax + 2              # even, so an exact grid shift exists
        cfg = create_gauss_config(lmax, lmax + 2; nlon=nlon)
        rng = MersenneTwister(4242)

        A = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        for m in 0:cfg.mmax, l in m:lmax
            A[l + 1, m + 1] = m == 0 ? randn(rng) : complex(randn(rng), randn(rng))
        end
        Qlm = SHTnsKit.pack_lm(cfg, A)

        shift = 3                        # α is an exact multiple of the φ spacing
        α = 2π * shift / nlon
        Rlm = similar(Qlm)
        SH_Zrotate(cfg, Qlm, α, Rlm)

        # (1) Ground truth: rotating the FIELD by +α about ẑ is g(θ,φ) = f(θ,φ-α),
        #     which on this grid is a pure column shift.
        f = synthesis(cfg, A)
        g = similar(f)
        for j in 1:nlon
            g[:, j] .= @view f[:, mod(j - 1 - shift, nlon) + 1]
        end
        spatial = SHTnsKit.pack_lm(cfg, analysis(cfg, g))
        @test isapprox(Rlm, spatial; rtol=1e-10, atol=1e-12)

        # ...and the opposite sign is the OTHER rotation, not a rounding detail.
        flipped = [Qlm[k] * cis(+cfg.mi[k] * α) for k in 1:cfg.nlm]
        @test !isapprox(flipped, spatial; rtol=1e-3, atol=1e-6)

        # (2) The general ZYZ engine must agree for both the α and γ slots.
        for angles in ((α, 0.0, 0.0), (0.0, 0.0, α))
            r = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax)
            SHTnsKit.shtns_rotation_set_angles_ZYZ(r, angles...)
            Rzyz = similar(Qlm)
            SHTnsKit.shtns_rotation_apply_real(r, Qlm, Rzyz)
            @test isapprox(Rzyz, Rlm; rtol=1e-10, atol=1e-12)
        end

        # (3) The dense distributed twin must agree coefficient for coefficient.
        Rdense = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        SHTnsKit.dist_SH_Zrotate(cfg, A, α, Rdense)
        @test isapprox(SHTnsKit.pack_lm(cfg, Rdense), Rlm; rtol=1e-12, atol=1e-14)
    end

    @testset "Z-axis rotation in-place" begin
        lmax = 6
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(90)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        α = π/4
        Rlm = similar(Qlm)
        SH_Zrotate(cfg, Qlm, α, Rlm)

        # In-place rotation
        Qlm_copy = copy(Qlm)
        SH_Zrotate(cfg, Qlm_copy, α, Qlm_copy)
        @test isapprox(Qlm_copy, Rlm; rtol=1e-12, atol=1e-14)
    end

    @testset "Z-rotation periodicity" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(91)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        # Rotation by 2π should return original (for m-integer modes)
        Rlm = similar(Qlm)
        SH_Zrotate(cfg, Qlm, 2π, Rlm)
        @test isapprox(Rlm, Qlm; rtol=1e-10, atol=1e-12)
    end

    @testset "Z-rotation energy preservation" begin
        # Z-rotation preserves energy because it's just a phase shift
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(92)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        α = π/3
        Rlm = similar(Qlm)
        SH_Zrotate(cfg, Qlm, α, Rlm)

        E_before = sum(abs2, Qlm)
        E_after = sum(abs2, Rlm)
        @test isapprox(E_before, E_after; rtol=1e-12, atol=1e-14)
    end

    @testset "Y-axis rotation reversibility" begin
        # Y-rotation should be reversible: rotate forward then backward = identity
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(93)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        α = π/3
        Rlm = similar(Qlm)
        Qlm_back = similar(Qlm)

        # Rotate forward
        SH_Yrotate(cfg, Qlm, α, Rlm)
        # Rotate backward
        SH_Yrotate(cfg, Rlm, -α, Qlm_back)

        # Should recover original
        @test isapprox(Qlm_back, Qlm; rtol=1e-9, atol=1e-11)
    end

    @testset "90-degree rotation reversibility" begin
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(94)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        # Y-rotation by 90° four times should return to original
        Rlm1 = similar(Qlm)
        Rlm2 = similar(Qlm)
        Rlm3 = similar(Qlm)
        Rlm4 = similar(Qlm)

        SH_Yrotate90(cfg, Qlm, Rlm1)
        SH_Yrotate90(cfg, Rlm1, Rlm2)
        SH_Yrotate90(cfg, Rlm2, Rlm3)
        SH_Yrotate90(cfg, Rlm3, Rlm4)

        # Four 90° rotations = 360° = identity
        @test isapprox(Rlm4, Qlm; rtol=1e-8, atol=1e-10)
    end

    @testset "X-rotation 90 is Rx(+pi/2), not its inverse" begin
        # Through v2.0.0 `SH_Xrotate90` used the Euler triple ZYZ(π/2, π/2, -π/2),
        # which is Rx(-π/2) — the inverse of the +90° rotation the name promises.
        # The existing "four 90° rotations = identity" check passes for either
        # sign, so pin the direction explicitly.
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 2)
        rng = MersenneTwister(8801)
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        X = similar(Qlm)
        SH_Xrotate90(cfg, Qlm, X)

        # Ground truth: a pure Rx(+π/2) via the ZXZ entry point, whose own
        # converter documents Rx(β) = Rz(-π/2)·Ry(β)·Rz(π/2).
        fwd = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax)
        SHTnsKit.shtns_rotation_set_angles_ZXZ(fwd, 0.0, π/2, 0.0)
        Xref = similar(Qlm)
        SHTnsKit.shtns_rotation_apply_real(fwd, Qlm, Xref)
        @test isapprox(X, Xref; rtol=1e-10, atol=1e-12)

        # ...and it is NOT the inverse rotation, which is what shipped before.
        inv_rot = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax)
        SHTnsKit.shtns_rotation_set_angles_ZXZ(inv_rot, 0.0, -π/2, 0.0)
        Xinv = similar(Qlm)
        SHTnsKit.shtns_rotation_apply_real(inv_rot, Qlm, Xinv)
        @test !isapprox(X, Xinv; rtol=1e-3, atol=1e-6)

        # The CHANGELOG's porting recipe: three forward turns reproduce the old result.
        a = similar(Qlm); b = similar(Qlm); c = similar(Qlm)
        SH_Xrotate90(cfg, Qlm, a); SH_Xrotate90(cfg, a, b); SH_Xrotate90(cfg, b, c)
        @test isapprox(c, Xinv; rtol=1e-9, atol=1e-11)
    end

    @testset "Wigner-d matrix orthogonality" begin
        # Test orthogonality: d^T d = I for real orthogonal case
        for l in 0:4
            β = 0.7
            d = SHTnsKit.wigner_d_matrix(l, β)
            n = 2l + 1
            @test size(d) == (n, n)

            # Orthogonality check
            prod = d' * d
            @test isapprox(prod, I(n); rtol=1e-10, atol=1e-12)
        end
    end

    @testset "Wigner-d matrix identity at β=0" begin
        for l in 0:4
            d0 = SHTnsKit.wigner_d_matrix(l, 0.0)
            @test isapprox(d0, I(2l+1); rtol=1e-12)
        end
    end

    @testset "Wigner-d matrix β=π inversion" begin
        # At β=π, d^l_{mm'}(π) = (-1)^(l+m) δ_{m,-m'}
        for l in 0:3
            d_pi = SHTnsKit.wigner_d_matrix(l, Float64(π))  # Convert π to Float64
            n = 2l + 1
            for m in -l:l
                for mp in -l:l
                    expected = (mp == -m) ? (-1.0)^(l + m) : 0.0
                    @test isapprox(d_pi[m+l+1, mp+l+1], expected; rtol=1e-10, atol=1e-12)
                end
            end
        end
    end

    @testset "SHTRotation struct" begin
        lmax = 4
        mmax = 4

        # Create rotation
        rot = SHTRotation(lmax, mmax; α=0.1, β=0.2, γ=0.3)
        @test rot.lmax == lmax
        @test rot.mmax == mmax
        @test rot.α ≈ 0.1
        @test rot.β ≈ 0.2
        @test rot.γ ≈ 0.3
        @test rot.conv == :ZYZ

        # Set angles ZYZ
        shtns_rotation_set_angles_ZYZ(rot, 0.5, 0.6, 0.7)
        @test rot.α ≈ 0.5
        @test rot.β ≈ 0.6
        @test rot.γ ≈ 0.7
        @test rot.conv == :ZYZ

        # Set angles ZXZ
        shtns_rotation_set_angles_ZXZ(rot, 0.8, 0.9, 1.0)
        @test rot.α ≈ 0.8
        @test rot.β ≈ 0.9
        @test rot.γ ≈ 1.0
        @test rot.conv == :ZXZ
    end

    @testset "General rotation reversibility" begin
        lmax = 3
        mmax = 3
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(95)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        # Apply rotation
        rot = SHTRotation(lmax, mmax)
        shtns_rotation_set_angles_ZYZ(rot, π/6, π/4, π/3)

        Rlm = similar(Qlm)
        shtns_rotation_apply_real(rot, Qlm, Rlm)

        # Apply inverse rotation (negate angles in reverse order for ZYZ)
        rot_inv = SHTRotation(lmax, mmax)
        shtns_rotation_set_angles_ZYZ(rot_inv, -π/3, -π/4, -π/6)

        Qlm_back = similar(Qlm)
        shtns_rotation_apply_real(rot_inv, Rlm, Qlm_back)

        # Should recover original
        @test isapprox(Qlm_back, Qlm; rtol=1e-8, atol=1e-10)
    end

    @testset "Complex rotation energy preservation" begin
        # For full complex representation, energy IS preserved
        lmax = 3
        mmax = 3
        rng = MersenneTwister(96)

        # Complex coefficients (full m range)
        nlm_cplx = nlm_cplx_calc(lmax, mmax, 1)
        Zlm = randn(rng, ComplexF64, nlm_cplx)

        rot = SHTRotation(lmax, mmax)
        shtns_rotation_set_angles_ZYZ(rot, 0.3, 0.5, 0.2)

        Rlm = similar(Zlm)
        shtns_rotation_apply_cplx(rot, Zlm, Rlm)

        # Energy preserved in full complex representation
        @test isapprox(sum(abs2, Zlm), sum(abs2, Rlm); rtol=1e-9)
    end

    @testset "Z-rotation composition" begin
        lmax = 3
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(97)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        # Two Z-rotations should compose additively
        α1 = π/6
        α2 = π/4
        Rlm1 = similar(Qlm)
        Rlm2 = similar(Qlm)
        Rlm_direct = similar(Qlm)

        SH_Zrotate(cfg, Qlm, α1, Rlm1)
        SH_Zrotate(cfg, Rlm1, α2, Rlm2)
        SH_Zrotate(cfg, Qlm, α1 + α2, Rlm_direct)

        @test isapprox(Rlm2, Rlm_direct; rtol=1e-10, atol=1e-12)
    end

    @testset "Z-rotation inverse" begin
        lmax = 3
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(98)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        α = π/3
        Rlm = similar(Qlm)
        Qlm_back = similar(Qlm)

        # Rotate forward then backward
        SH_Zrotate(cfg, Qlm, α, Rlm)
        SH_Zrotate(cfg, Rlm, -α, Qlm_back)

        @test isapprox(Qlm_back, Qlm; rtol=1e-10, atol=1e-12)
    end

    @testset "Rotation produces different result" begin
        # Verify rotation actually changes coefficients (not a no-op)
        lmax = 4
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(99)

        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        Rlm = similar(Qlm)
        SH_Yrotate(cfg, Qlm, π/4, Rlm)

        # Should be different (except for m=0 axisymmetric case at special angles)
        @test !isapprox(Rlm, Qlm; rtol=0.1)
    end

    @testset "Angle-axis rotation" begin
        lmax = 5
        cfg = create_gauss_config(lmax, lmax + 2; nlon=2*lmax + 1)
        rng = MersenneTwister(100)

        # Create rotation object
        rot = SHTRotation(lmax, lmax)

        # Set rotation using angle-axis (rotate by π/3 around normalized axis)
        theta = π/3
        Vx, Vy, Vz = 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)  # Unit axis
        shtns_rotation_set_angle_axis(rot, theta, Vx, Vy, Vz)

        # Apply rotation
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Qlm[1:lmax+1] .= real.(Qlm[1:lmax+1])

        Rlm = similar(Qlm)
        shtns_rotation_apply_real(rot, Qlm, Rlm)

        # Should produce valid output
        @test all(!isnan, Rlm)
        @test all(!isinf, Rlm)

        # Rotated field should be different from original
        @test !isapprox(Rlm, Qlm; rtol=0.1)

        # Energy should be preserved (use proper packed energy function)
        E_orig = energy_scalar_packed(cfg, Qlm)
        E_rot = energy_scalar_packed(cfg, Rlm)
        @test isapprox(E_orig, E_rot; rtol=1e-10)
    end
end
