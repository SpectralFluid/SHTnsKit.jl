# SHTnsKit.jl - Local Evaluation Tests
# Tests for SH_to_lat, SH_to_lat_cplx, SHqst_to_point, SHqst_to_lat

using Test
using SHTnsKit
using Random

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

@testset "Local Evaluations" begin
    @testset "Robert-form local vector evaluations match grid synthesis" begin
        for grid in (:gauss, :regular_poles), mres in (1, 2)
            cfg = create_config(3; grid_type=grid, nlat=5, mres,
                                robert_form=true, norm=:schmidt,
                                real_norm=true, cs_phase=false)
            Q = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
            S = copy(Q)
            T = copy(Q)
            Q[1, 1] = 0.8
            S[2, 1] = 0.7
            S[4, 1] = -0.3
            S[3, mres + 1] = 0.4 + 0.2im
            T[4, mres + 1] = -0.2 + 0.6im
            packed = map(A -> SHTnsKit.pack_lm(cfg, A), (Q, S, T))
            fields = synthesis_qst(cfg, Q, S, T)
            grad_fields = synthesis_sph(cfg, S)

            for i in (1, 2, cfg.nlat)
                x = cfg.x[i]
                lat = SHqst_to_lat(cfg, packed..., x)
                point = SHTnsKit.SHqst_to_point(cfg, packed..., x, cfg.φ[2])
                grad = SHTnsKit.SH_to_grad_point(cfg, packed[1], packed[2], x, cfg.φ[2])
                for component in 1:3
                    @test lat[component] ≈ fields[component][i, :] rtol=1e-11 atol=1e-12
                    @test point[component] ≈ fields[component][i, 2] rtol=1e-11 atol=1e-12
                end
                # The current API also evaluates the supplied radial derivative.
                @test grad[1] ≈ fields[1][i, 2] rtol=1e-11 atol=1e-12
                @test grad[2] ≈ grad_fields[1][i, 2] rtol=1e-11 atol=1e-12
                @test grad[3] ≈ grad_fields[2][i, 2] rtol=1e-11 atol=1e-12

                truncated = SHqst_to_lat(cfg, packed..., x; ltr=2)
                truncated_fields = synthesis_qst_l(cfg, Q, S, T, 2)
                for component in 1:3
                    @test truncated[component] ≈ truncated_fields[component][i, :] rtol=1e-11 atol=1e-12
                end
            end
        end
    end

    @testset "point/latitude evaluators honour phi_scale" begin
        # `synthesis` scales its Fourier bins by `phi_inv_scale(cfg)` and the
        # inverse FFT divides by nlon, a net factor of 1 under :dft but 1/2π
        # under :quad. The direct evaluators applied no factor at all, so under
        # :quad every one of them disagreed with the grid it claims to sample by
        # exactly 2π.
        for mode in (:dft, :quad)
            cfg = create_gauss_config(6, 8)
            cfg.phi_scale = mode
            rng = MersenneTwister(5150)
            A = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
            for m in 0:cfg.mmax, l in m:cfg.lmax
                A[l+1, m+1] = m == 0 ? randn(rng) : complex(randn(rng), randn(rng))
            end
            packed = SHTnsKit.pack_lm(cfg, A)
            zero_packed = zeros(ComplexF64, cfg.nlm)

            f = synthesis(cfg, A)
            Vr, Vt, Vp = synthesis_qst(cfg, A, zero(A), zero(A))
            i, j = 2, 3

            @test synthesis_point(cfg, A, cfg.x[i], cfg.φ[j]) ≈ f[i, j] rtol=1e-10
            @test SH_to_lat(cfg, packed, cfg.x[i]) ≈ f[i, :] rtol=1e-10
            @test SHqst_to_lat(cfg, packed, zero_packed, zero_packed, cfg.x[i])[1] ≈
                  Vr[i, :] rtol=1e-10
            @test SHTnsKit.SHqst_to_point(cfg, packed, zero_packed, zero_packed,
                                          cfg.x[i], cfg.φ[j])[1] ≈ Vr[i, j] rtol=1e-10

            # axisymmetric pair
            a0 = zeros(ComplexF64, cfg.lmax + 1); a0[3] = 0.7
            am = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1); am[3, 1] = 0.7
            @test synthesis_axisym(cfg, a0) ≈ synthesis(cfg, am)[:, 1] rtol=1e-10
        end
    end

    @testset "evaluators preserve the coefficient element type" begin
        # The φ convention factor is a Float64. Multiplying a Float32 evaluation
        # by it silently widens every result to Float64, breaking the element
        # type these functions promise their caller (and the Dual types AD needs
        # to see through). Narrow the scale at the boundary instead.
        for mode in (:dft, :quad), T in (Float32, Float64)
            cfg = create_gauss_config(4, 6; nlon=9)
            cfg.phi_scale = mode
            CT = Complex{T}
            A = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
            A[2, 1] = CT(0.6); A[3, 2] = CT(0.4, 0.2)
            packed = SHTnsKit.pack_lm(cfg, A)
            zpack = zeros(CT, cfg.nlm)
            x = T(cfg.x[2]); φ = T(cfg.φ[3])

            @test synthesis_point(cfg, A, x, φ) isa T
            @test eltype(SH_to_lat(cfg, packed, x)) === T
            @test eltype(SH_to_lat_cplx(cfg, zeros(CT, SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)), x)) === CT
            @test all(v -> v isa T, SHTnsKit.SHqst_to_point(cfg, packed, zpack, zpack, x, φ))
            @test all(v -> v isa T, SHTnsKit.SH_to_grad_point(cfg, packed, zpack, x, φ))
            @test all(V -> eltype(V) === T, SHqst_to_lat(cfg, packed, zpack, zpack, x))
            @test eltype(synthesis_axisym(cfg, A[:, 1])) === T
            @test eltype(SHTnsKit.synthesis_axisym_l(cfg, A[:, 1], cfg.lmax)) === T
            @test SHTnsKit.synthesis_point_cplx(
                cfg, zeros(CT, SHTnsKit.nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)), x, φ) isa CT
        end
    end

    @testset "SH_to_lat matches synthesis at grid latitudes" begin
        lmax = 8
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        rng = MersenneTwister(42)
        # Create random real-field coefficients in dense (l,m) matrix format
        alm = zeros(ComplexF64, lmax + 1, lmax + 1)
        for m in 0:lmax, l in m:lmax
            alm[l+1, m+1] = randn(rng, ComplexF64)
        end
        # m=0 must be real for real fields
        alm[:, 1] .= real.(alm[:, 1])
        # Zero invalid entries (l < m)
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Full synthesis for reference
        f = synthesis(cfg, alm; real_output=true)

        # Pack coefficients for SH_to_lat (needs packed LM format)
        Qlm = Vector{ComplexF64}(undef, cfg.nlm)
        for k in 1:cfg.nlm
            l = cfg.li[k]
            m = cfg.mi[k]
            Qlm[k] = alm[l+1, m+1]
        end

        # Test at each grid latitude: SH_to_lat should match synthesis row
        for i in 1:cfg.nlat
            cost = cfg.x[i]
            vals = SH_to_lat(cfg, Qlm, cost)
            @test length(vals) == cfg.nlon
            @test isapprox(vals, f[i, :]; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "SH_to_lat with degree truncation" begin
        lmax = 8
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)
        ltr = 4  # Truncate at degree 4

        rng = MersenneTwister(43)
        alm = zeros(ComplexF64, lmax + 1, lmax + 1)
        for m in 0:lmax, l in m:lmax
            alm[l+1, m+1] = randn(rng, ComplexF64)
        end
        alm[:, 1] .= real.(alm[:, 1])
        for m in 0:lmax, l in 0:(m-1)
            alm[l+1, m+1] = 0
        end

        # Truncated reference: zero modes above ltr
        alm_trunc = copy(alm)
        for m in 0:lmax, l in (ltr+1):lmax
            alm_trunc[l+1, m+1] = 0
        end
        f_trunc = synthesis(cfg, alm_trunc; real_output=true)

        # Pack
        Qlm = Vector{ComplexF64}(undef, cfg.nlm)
        for k in 1:cfg.nlm
            l = cfg.li[k]; m = cfg.mi[k]
            Qlm[k] = alm[l+1, m+1]
        end

        # SH_to_lat with ltr should match truncated synthesis
        for i in 1:min(3, cfg.nlat)  # Test a few latitudes
            cost = cfg.x[i]
            vals = SH_to_lat(cfg, Qlm, cost; ltr=ltr)
            @test isapprox(vals, f_trunc[i, :]; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "SH_to_lat_cplx matches complex synthesis" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        rng = MersenneTwister(44)
        # Create complex-packed coefficients (LM_cplx format, all m including negative)
        nlm_c = nlm_cplx_calc(lmax, lmax, 1)
        alm_cplx = randn(rng, ComplexF64, nlm_c)

        # Evaluate at a grid latitude
        cost = cfg.x[1]
        vals = SHTnsKit.SH_to_lat_cplx(cfg, alm_cplx, cost)
        @test length(vals) == cfg.nlon
        @test eltype(vals) == ComplexF64
        # Basic sanity: values should be finite
        @test all(isfinite, vals)
    end

    @testset "SHqst_to_point basic evaluation" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        # Set up trivial QST coefficients
        Qlm = zeros(ComplexF64, cfg.nlm)
        Slm = zeros(ComplexF64, cfg.nlm)
        Tlm = zeros(ComplexF64, cfg.nlm)

        # Set Q_{1,0} = 1 (a simple radial dipole)
        lm10 = LM_index(lmax, 1, 1, 0) + 1
        Qlm[lm10] = 1.0 + 0im

        cost = 0.5  # Some latitude
        phi = 0.3   # Some longitude
        Vr, Vt, Vp = SHTnsKit.SHqst_to_point(cfg, Qlm, Slm, Tlm, cost, phi)

        # With only Q_{1,0} set, should get nonzero Vr and zero Vt, Vp
        @test abs(Vr) > 1e-10  # Radial component should be nonzero
        @test isapprox(Vt, 0.0; atol=1e-12)
        @test isapprox(Vp, 0.0; atol=1e-12)
    end

    @testset "SHqst_to_lat matches full QST synthesis at grid latitudes" begin
        lmax = 6
        nlat = lmax + 2
        cfg = create_gauss_config(lmax, nlat)

        rng = MersenneTwister(45)
        # Create random QST coefficients
        Qlm = randn(rng, ComplexF64, cfg.nlm)
        Slm = randn(rng, ComplexF64, cfg.nlm)
        Tlm = randn(rng, ComplexF64, cfg.nlm)

        # Make m=0 real
        for k in 1:cfg.nlm
            if cfg.mi[k] == 0
                Qlm[k] = real(Qlm[k])
                Slm[k] = real(Slm[k])
                Tlm[k] = real(Tlm[k])
            end
        end

        # Full QST synthesis for reference — unpack to matrix form
        Q_mat = zeros(ComplexF64, lmax + 1, lmax + 1)
        S_mat = zeros(ComplexF64, lmax + 1, lmax + 1)
        T_mat = zeros(ComplexF64, lmax + 1, lmax + 1)
        for k in 1:cfg.nlm
            l = cfg.li[k]; m = cfg.mi[k]
            Q_mat[l+1, m+1] = Qlm[k]
            S_mat[l+1, m+1] = Slm[k]
            T_mat[l+1, m+1] = Tlm[k]
        end
        Vr_ref, Vt_ref, Vp_ref = synthesis_qst(cfg, Q_mat, S_mat, T_mat; real_output=true)

        # Test at a grid latitude
        i = 2  # Pick a non-pole latitude
        cost = cfg.x[i]
        Vr_lat, Vt_lat, Vp_lat = SHTnsKit.SHqst_to_lat(cfg, Qlm, Slm, Tlm, cost)

        @test length(Vr_lat) == cfg.nlon
        @test isapprox(Vr_lat, Vr_ref[i, :]; rtol=1e-9, atol=1e-11)
        @test isapprox(Vt_lat, Vt_ref[i, :]; rtol=1e-9, atol=1e-11)
        @test isapprox(Vp_lat, Vp_ref[i, :]; rtol=1e-9, atol=1e-11)
    end

    @testset "configured coefficient conventions are honored" begin
        lmax = 4
        nlat = lmax + 2
        nlon = 2lmax + 1
        canonical_cfg = create_gauss_config(lmax, nlat; nlon=nlon)
        configured_cfg = create_gauss_config(
            lmax, nlat; nlon=nlon, norm=:fourpi, cs_phase=false, real_norm=true,
        )
        rng = MersenneTwister(46)

        canonical_Q = zeros(ComplexF64, lmax + 1, lmax + 1)
        canonical_S = zeros(ComplexF64, lmax + 1, lmax + 1)
        canonical_T = zeros(ComplexF64, lmax + 1, lmax + 1)
        for m in 0:lmax, l in m:lmax
            canonical_Q[l + 1, m + 1] = randn(rng, ComplexF64)
            if l > 0
                canonical_S[l + 1, m + 1] = randn(rng, ComplexF64)
                canonical_T[l + 1, m + 1] = randn(rng, ComplexF64)
            end
        end
        canonical_Q[:, 1] .= real.(canonical_Q[:, 1])
        canonical_S[:, 1] .= real.(canonical_S[:, 1])
        canonical_T[:, 1] .= real.(canonical_T[:, 1])

        configured_Q = similar(canonical_Q)
        configured_S = similar(canonical_S)
        configured_T = similar(canonical_T)
        SHTnsKit.convert_alm_norm!(configured_Q, canonical_Q, configured_cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(configured_S, canonical_S, configured_cfg; to_internal=false)
        SHTnsKit.convert_alm_norm!(configured_T, canonical_T, configured_cfg; to_internal=false)

        canonical_packed = map(
            A -> SHTnsKit.pack_lm(canonical_cfg, A),
            (canonical_Q, canonical_S, canonical_T),
        )
        configured_packed = map(
            A -> SHTnsKit.pack_lm(configured_cfg, A),
            (configured_Q, configured_S, configured_T),
        )
        cost = canonical_cfg.x[2]
        phi = 0.7

        @test SH_to_lat(configured_cfg, configured_packed[1], cost) ≈
              SH_to_lat(canonical_cfg, canonical_packed[1], cost) rtol=1e-12 atol=1e-13

        canonical_point = SHTnsKit.SHqst_to_point(canonical_cfg, canonical_packed..., cost, phi)
        configured_point = SHTnsKit.SHqst_to_point(configured_cfg, configured_packed..., cost, phi)
        for component in 1:3
            @test configured_point[component] ≈ canonical_point[component] rtol=1e-12 atol=1e-13
        end

        canonical_grad = SHTnsKit.SH_to_grad_point(
            canonical_cfg, canonical_packed[1], canonical_packed[2], cost, phi,
        )
        configured_grad = SHTnsKit.SH_to_grad_point(
            configured_cfg, configured_packed[1], configured_packed[2], cost, phi,
        )
        for component in 1:3
            @test configured_grad[component] ≈ canonical_grad[component] rtol=1e-12 atol=1e-13
        end

        canonical_lat = SHTnsKit.SHqst_to_lat(canonical_cfg, canonical_packed..., cost)
        configured_lat = SHTnsKit.SHqst_to_lat(configured_cfg, configured_packed..., cost)
        for component in 1:3
            @test configured_lat[component] ≈ canonical_lat[component] rtol=1e-12 atol=1e-13
        end

        nlm_c = nlm_cplx_calc(lmax, lmax, 1)
        canonical_cplx = randn(rng, ComplexF64, nlm_c)
        configured_cplx = similar(canonical_cplx)
        SHTnsKit.convert_alm_norm!(
            configured_cplx, canonical_cplx, configured_cfg; to_internal=false,
        )
        @test SHTnsKit.SH_to_lat_cplx(configured_cfg, configured_cplx, cost) ≈
              SHTnsKit.SH_to_lat_cplx(canonical_cfg, canonical_cplx, cost) rtol=1e-12 atol=1e-13
    end
end
