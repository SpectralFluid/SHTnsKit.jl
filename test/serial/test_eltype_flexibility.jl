# SHTnsKit.jl - Element-type flexibility and allocation tests
# Point/latitude evaluators and operators must propagate the input eltype
# (e.g. ForwardDiff.Dual) instead of hardcoding ComplexF64/Float64, and the
# point evaluators must not allocate O(nlm) scratch per call.

using Test
using SHTnsKit
using ForwardDiff: Dual, value, partials

@isdefined(VERBOSE) || (const VERBOSE = get(ENV, "SHTNSKIT_TEST_VERBOSE", "0") == "1")

# Complex Dual vector carrying d/dt of (t * base) at t = 1
_dualize(base::AbstractVector{<:Complex}) =
    [Complex(Dual(real(z), real(z)), Dual(imag(z), imag(z))) for z in base]
_dualize(base::AbstractMatrix{<:Complex}) =
    [Complex(Dual(real(z), real(z)), Dual(imag(z), imag(z))) for z in base]

_value(x::Dual) = value(x)
_value(z::Complex{<:Dual}) = Complex(value(real(z)), value(imag(z)))
_partial(x::Dual) = partials(x, 1)

@testset "Eltype flexibility (Dual propagation)" begin
    lmax = 8
    nlat = lmax + 4
    nlon = 2 * lmax + 2
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    rng_vals = [0.1 * (i + 1) + 0.05im * i for i in 0:(cfg.nlm - 1)]
    Qlm = collect(ComplexF64, rng_vals)
    Slm = reverse(Qlm)
    Tlm = 0.5 .* Qlm
    cost = 0.3

    @testset "SH_to_lat accepts Dual coefficients" begin
        Qd = _dualize(Qlm)
        vals_d = SH_to_lat(cfg, Qd, cost)
        vals = SH_to_lat(cfg, Qlm, cost)
        @test _value.(vals_d) ≈ vals
        # Field is linear in Qlm, so d/dt of (t*Qlm) at t=1 equals the value
        @test _partial.(vals_d) ≈ vals
    end

    @testset "SH_to_lat_cplx accepts Dual coefficients" begin
        nc = SHTnsKit.nlm_cplx_calc(lmax, cfg.mmax, 1)
        alm = [0.1 * k + 0.02im * k for k in 1:nc]
        vals = SHTnsKit.SH_to_lat_cplx(cfg, alm, cost)
        vals_d = SHTnsKit.SH_to_lat_cplx(cfg, _dualize(alm), cost)
        @test _value.(vals_d) ≈ vals
        @test _partial.(real.(vals_d)) ≈ real.(vals)
    end

    @testset "SHqst_to_lat accepts Dual coefficients" begin
        Vr, Vt, Vp = SHqst_to_lat(cfg, Qlm, Slm, Tlm, cost)
        Vrd, Vtd, Vpd = SHqst_to_lat(cfg, _dualize(Qlm), _dualize(Slm), _dualize(Tlm), cost)
        @test _value.(Vrd) ≈ Vr
        @test _value.(Vtd) ≈ Vt
        @test _value.(Vpd) ≈ Vp
        @test _partial.(Vtd) ≈ Vt
    end

    @testset "synthesis_packed_ml accepts Dual coefficients" begin
        m = 1
        Ql = Qlm[1:(lmax - m + 1)]
        out = synthesis_packed_ml(cfg, m, Ql, lmax)
        out_d = synthesis_packed_ml(cfg, m, _dualize(Ql), lmax)
        @test _value.(out_d) ≈ out
    end

    @testset "synthesis_point is inferable with Dual coefficients" begin
        Qmat = zeros(ComplexF64, lmax + 1, cfg.mmax + 1)
        for m in 0:cfg.mmax, l in m:lmax
            Qmat[l + 1, m + 1] = 0.1 * (l + 1) + 0.03im * m
        end
        Qmat_d = _dualize(Qmat)
        v = synthesis_point(cfg, Qmat, cost, 0.7)
        vd = @inferred synthesis_point(cfg, Qmat_d, cost, 0.7)
        @test _value(vd) ≈ v
        @test _partial(vd) ≈ v
    end

    @testset "sphtor synthesis kernels are inferable with Dual coefficients" begin
        Smat = _dualize(zeros(ComplexF64, lmax + 1, cfg.mmax + 1) .+ Slm[1] )
        Tmat = _dualize(zeros(ComplexF64, lmax + 1, cfg.mmax + 1) .+ Tlm[1])
        P = Vector{Float64}(undef, lmax + 1)
        dP = Vector{Float64}(undef, lmax + 1)
        Ps = Vector{Float64}(undef, lmax + 1)
        Pb = Vector{Float64}(undef, lmax + 2)
        gθ, gφ = @inferred SHTnsKit._sphtor_synthesis_kernel_otf(
            cfg, Smat, Tmat, P, dP, Ps, Pb, 1, 2, 1, lmax)
        @test isfinite(value(real(gθ)))
        @test isfinite(value(real(gφ)))
    end

    @testset "SH_mul_mx accepts Dual coefficients" begin
        mx = zeros(2 * cfg.nlm)
        mul_ct_matrix(cfg, mx)
        R = similar(Qlm)
        SH_mul_mx(cfg, mx, Qlm, R)
        Qd = _dualize(Qlm)
        Rd = similar(Qd)
        SH_mul_mx(cfg, mx, Qd, Rd)
        @test _value.(Rd) ≈ R
    end
end

@testset "Sphtor synthesis promotes both coefficient inputs" begin
    lmax = 5
    nlat = lmax + 2
    nlon = 2*lmax + 1
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    S32 = zeros(ComplexF32, lmax + 1, lmax + 1)
    T64 = zeros(ComplexF64, lmax + 1, lmax + 1)
    S32[3, 2] = 0.25f0 - 0.5f0im
    T64[4, 3] = 1.0 + 1.0e-10im

    for real_output in (true, false)
        Vt, Vp = synthesis_sphtor(cfg, S32, T64; real_output)
        Vt_ref, Vp_ref = synthesis_sphtor(cfg, ComplexF64.(S32), T64; real_output)
        expected_type = real_output ? Float64 : ComplexF64
        @test eltype(Vt) === expected_type
        @test eltype(Vp) === expected_type
        @test Vt ≈ Vt_ref
        @test Vp ≈ Vp_ref

        Vt_l, Vp_l = synthesis_sphtor_l(cfg, S32, T64, 4; real_output)
        Vt_l_ref, Vp_l_ref = synthesis_sphtor_l(
            cfg, ComplexF64.(S32), T64, 4; real_output)
        @test eltype(Vt_l) === expected_type
        @test eltype(Vp_l) === expected_type
        @test Vt_l ≈ Vt_l_ref
        @test Vp_l ≈ Vp_l_ref
    end
end

@testset "Adjoint FFT allocations (copy+re-plan pattern)" begin
    using ChainRulesCore: rrule
    lmax = 64
    nlat = lmax + 2
    nlon = 2 * lmax + 2
    cfg = create_gauss_config(lmax, nlat; nlon=nlon)
    matbytes = nlat * nlon * 16

    # _adjoint_synthesis must allocate ONE complex FFT buffer (cached plan),
    # not a complex copy + an out-of-place re-planned fft result.
    fbar = randn(nlat, nlon)
    ref = SHTnsKit._adjoint_synthesis(cfg, fbar)
    a = @allocated SHTnsKit._adjoint_synthesis(cfg, fbar)
    @test a < 2 * matbytes  # one buffer + ālm output + slack (was ~2.5 buffers)

    # sphtor synthesis pullback: one buffer per field, not two
    S = zeros(ComplexF64, lmax + 1, cfg.mmax + 1); S[2, 1] = 1.0
    T = zeros(ComplexF64, lmax + 1, cfg.mmax + 1); T[3, 2] = 0.5 + 0.25im
    (y, pb) = rrule(SHTnsKit.synthesis_sphtor, cfg, S, T)
    Vt, Vp = y
    _, _, S̄1, T̄1 = pb((Vt, Vp))
    a2 = @allocated pb((Vt, Vp))
    # 2 FFT buffers + S̄/T̄ outputs ≈ 3 matbytes; the old copy+re-plan path was
    # ~5 matbytes. 4.5× leaves headroom for Julia-version/platform allocation
    # noise (1.12.6/x64 measures ~0.8 matbytes above 1.11.1/arm) while still
    # failing if the per-field copy comes back.
    @test a2 < 9 * matbytes ÷ 2
    # adjoint result must be unchanged by the buffer strategy
    _, _, S̄2, T̄2 = pb((Vt, Vp))
    @test S̄1 ≈ S̄2 && T̄1 ≈ T̄2

    # analysis_packed_cplx: complex input avoids the copy already; guard that
    # the cached-plan path returns identical results
    z = randn(ComplexF64, nlat, nlon)
    alm1 = analysis_packed_cplx(cfg, z)
    alm2 = analysis_packed_cplx(cfg, z)
    @test alm1 ≈ alm2
end

@testset "Point evaluator allocations" begin
    lmax = 32
    cfg = create_gauss_config(lmax, lmax + 4; nlon=2 * lmax + 2)
    Slm = [0.1 * (i + 1) + 0.05im * i for i in 0:(cfg.nlm - 1)]
    Dr = zero(Slm)
    # Must match the (Q=0, T=0) special case of the general point evaluator
    vr, vt, vp = SHTnsKit.SH_to_grad_point(cfg, Dr, Slm, 0.3, 0.7)
    zq = zeros(ComplexF64, cfg.nlm)
    vr0, vt0, vp0 = SHTnsKit.SHqst_to_point(cfg, zq, Slm, zq, 0.3, 0.7)
    @test vr == 0.0
    @test vt ≈ vt0
    @test vp ≈ vp0
    a = @allocated SHTnsKit.SH_to_grad_point(cfg, Dr, Slm, 0.3, 0.7)
    # Must not allocate two O(nlm) zero vectors per call (2*nlm*16 = 17.9 KB
    # at lmax=32); only the three O(lmax) Legendre rows are acceptable.
    @test a < 4096
end
