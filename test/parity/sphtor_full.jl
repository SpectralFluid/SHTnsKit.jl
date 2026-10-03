using Test
using SHTnsKit

abstract type VectorParityAdapter end

struct CPUVectorAdapter <: VectorParityAdapter end

vector_place(::CPUVectorAdapter, ::SHTConfig, value, ::Symbol) = value
vector_collect(::CPUVectorAdapter, value, ::SHTConfig) = Array(value)
vector_resident(::CPUVectorAdapter, value) = @test on_device(value) isa CPU
vector_analysis(::CPUVectorAdapter, cfg, Vt, Vp; use_rfft=false) =
    analysis_sphtor(CPU(), cfg, Vt, Vp; use_rfft)
vector_analysis_cplx(::CPUVectorAdapter, cfg, Vt, Vp) =
    analysis_sphtor_cplx(CPU(), cfg, Vt, Vp)
vector_synthesis(::CPUVectorAdapter, cfg, S, T, _prototype;
                 real_output=true, use_rfft=false) =
    synthesis_sphtor(CPU(), cfg, S, T; real_output, use_rfft)
vector_synthesis_cplx(::CPUVectorAdapter, cfg, S, T, _prototype) =
    synthesis_sphtor_cplx(CPU(), cfg, S, T)
vector_sph(::CPUVectorAdapter, cfg, S, _prototype; real_output=true) =
    synthesis_sph(CPU(), cfg, S; real_output)
vector_sph_cplx(::CPUVectorAdapter, cfg, S, _prototype) =
    synthesis_sph_cplx(CPU(), cfg, S)
vector_tor(::CPUVectorAdapter, cfg, T, _prototype; real_output=true) =
    synthesis_tor(CPU(), cfg, T; real_output)
vector_tor_cplx(::CPUVectorAdapter, cfg, T, _prototype) =
    synthesis_tor_cplx(CPU(), cfg, T)

const _VECTOR_GRID_KINDS = (:gauss, :gauss_fly, :regular, :regular_poles)

function _vector_config(kind::Symbol, lmax::Int=3, nlat::Int=8;
                        mres::Int=1, norm::Symbol=:orthonormal,
                        real_norm::Bool=false, cs_phase::Bool=true,
                        robert_form::Bool=false, south_pole_first::Bool=false)
    nlon = 2lmax + 2
    cfg = if kind === :gauss
        create_gauss_config(lmax, nlat; nlon, mres, norm, real_norm,
                            cs_phase, robert_form)
    elseif kind === :gauss_fly
        create_gauss_fly_config(lmax, nlat; nlon, mres, norm, real_norm,
                                cs_phase, robert_form)
    elseif kind === :regular
        create_regular_config(lmax, nlat; nlon, mres, norm, real_norm,
                              cs_phase, robert_form, include_poles=false,
                              precompute_plm=true)
    elseif kind === :regular_poles
        create_regular_config(lmax, nlat; nlon, mres, norm, real_norm,
                              cs_phase, robert_form, include_poles=true,
                              precompute_plm=true)
    else
        error("unknown vector parity grid kind: $kind")
    end
    south_pole_first && set_south_pole_first!(cfg)
    return cfg
end

_vector_tol(::Type{Float32}) = (atol=4f-4, rtol=4f-4)
_vector_tol(::Type{Float64}) = (atol=6e-11, rtol=6e-11)

function _external_vector_coefficients(cfg, canonical)
    # Conversion visits only valid l >= m storage; initialize forbidden entries
    # so MPI replicated-input validation never observes rank-local garbage.
    external = zero(canonical)
    SHTnsKit.convert_alm_norm!(external, canonical, cfg; to_internal=false)
    return external
end

"""
Independent low-order vector-harmonic sum.  It deliberately does not call the
package Legendre or vector kernels: the l=1,m=0 and l=2,m=2 harmonics are
written in closed form, including their theta derivatives.
"""
function _direct_low_vector(cfg, S_can::AbstractMatrix{Complex{T}},
                            T_can::AbstractMatrix{Complex{T}};
                            real_output::Bool) where {T<:AbstractFloat}
    Vt = Matrix{Complex{T}}(undef, cfg.nlat, cfg.nlon)
    Vp = similar(Vt)
    c10 = sqrt(T(3) / T(4pi))
    c22 = sqrt(T(15) / T(32pi))
    for j in 1:cfg.nlon, i in 1:cfg.nlat
        x = T(cfg.x[i])
        s = sqrt(max(zero(T), one(T) - x*x))
        vt = zero(Complex{T})
        vp = zero(Complex{T})

        # l=1,m=0: dY/dtheta = -sqrt(3/4pi) sin(theta).
        d10 = -c10 * s
        vt += d10 * S_can[2, 1]
        vp += d10 * T_can[2, 1]

        if cfg.mres == 1 || cfg.mres == 2
            # l=2,m=2: Y=C sin(theta)^2 exp(2i phi),
            # dY/dtheta=2C sin(theta)cos(theta) exp(2i phi).
            wave = cis(T(2) * T(cfg.φ[j]))
            d22 = T(2) * c22 * s * x
            y22_over_s = c22 * s
            sv = (d22 * S_can[3, 3] - Complex{T}(0, 2) * y22_over_s * T_can[3, 3]) * wave
            tv = (Complex{T}(0, 2) * y22_over_s * S_can[3, 3] + d22 * T_can[3, 3]) * wave
            if real_output
                vt += Complex{T}(T(2) * real(sv), 0)
                vp += Complex{T}(T(2) * real(tv), 0)
            else
                vt += sv
                vp += tv
            end
        end
        if real_output
            vt = Complex{T}(real(vt), 0)
            vp = Complex{T}(real(vp), 0)
        end
        if cfg.robert_form
            vt *= s
            vp *= s
        end
        Vt[i, j] = vt
        Vp[i, j] = vp
    end
    return real_output ? (real.(Vt), real.(Vp)) : (Vt, Vp)
end

function _vector_modes(cfg, ::Type{T}) where {T<:AbstractFloat}
    CT = Complex{T}
    S = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    Tlm = zeros(CT, size(S))
    S[2, 1] = CT(T(0.19), 0)
    Tlm[2, 1] = CT(T(-0.11), 0)
    S[3, 3] = CT(T(0.13), T(-0.07))
    Tlm[3, 3] = CT(T(-0.09), T(0.05))
    return S, Tlm
end

function _test_vector_case(adapter::VectorParityAdapter, cfg, ::Type{T}) where {T<:AbstractFloat}
    tol = _vector_tol(T)
    unsupported_analysis = cfg.grid_type === :regular_poles && cfg.robert_form &&
                           cfg.mres == 1 && cfg.mmax >= 1
    S_can, T_can = _vector_modes(cfg, T)
    S = _external_vector_coefficients(cfg, S_can)
    Tlm = _external_vector_coefficients(cfg, T_can)
    Vt_ref, Vp_ref = _direct_low_vector(cfg, S_can, T_can; real_output=true)
    prototype = vector_place(adapter, cfg, Vt_ref, :spatial)
    Sd = vector_place(adapter, cfg, S, :spectral)
    Td = vector_place(adapter, cfg, Tlm, :spectral)

    Vt, Vp = vector_synthesis(adapter, cfg, Sd, Td, prototype; real_output=true)
    vector_resident(adapter, Vt); vector_resident(adapter, Vp)
    Vth = vector_collect(adapter, Vt, cfg); Vph = vector_collect(adapter, Vp, cfg)
    @test eltype(Vth) === T
    @test eltype(Vph) === T
    @test Vth ≈ Vt_ref atol=tol.atol rtol=tol.rtol
    @test Vph ≈ Vp_ref atol=tol.atol rtol=tol.rtol

    if unsupported_analysis
        @test_throws ArgumentError vector_analysis(
            adapter, cfg, prototype, vector_place(adapter, cfg, Vp_ref, :spatial))
    else
        Sa, Ta = vector_analysis(adapter, cfg, prototype, vector_place(adapter, cfg, Vp_ref, :spatial))
        Sah = vector_collect(adapter, Sa, cfg); Tah = vector_collect(adapter, Ta, cfg)
        @test eltype(Sah) === Complex{T}
        @test eltype(Tah) === Complex{T}
        @test Sah ≈ S atol=tol.atol rtol=tol.rtol
        @test Tah ≈ Tlm atol=tol.atol rtol=tol.rtol
    end

    Vtc_ref, Vpc_ref = _direct_low_vector(cfg, S_can, T_can; real_output=false)
    Vtc, Vpc = vector_synthesis_cplx(adapter, cfg, Sd, Td, prototype)
    @test vector_collect(adapter, Vtc, cfg) ≈ Vtc_ref atol=tol.atol rtol=tol.rtol
    @test vector_collect(adapter, Vpc, cfg) ≈ Vpc_ref atol=tol.atol rtol=tol.rtol
    if unsupported_analysis
        @test_throws ArgumentError vector_analysis_cplx(
            adapter, cfg, vector_place(adapter, cfg, Vtc_ref, :spatial),
            vector_place(adapter, cfg, Vpc_ref, :spatial))
    else
        Sac, Tac = vector_analysis_cplx(
            adapter, cfg, vector_place(adapter, cfg, Vtc_ref, :spatial),
            vector_place(adapter, cfg, Vpc_ref, :spatial),
        )
        @test vector_collect(adapter, Sac, cfg) ≈ S atol=tol.atol rtol=tol.rtol
        @test vector_collect(adapter, Tac, cfg) ≈ Tlm atol=tol.atol rtol=tol.rtol
    end

    zeroS = vector_place(adapter, cfg, zero.(S), :spectral)
    zeroT = vector_place(adapter, cfg, zero.(Tlm), :spectral)
    sph = vector_sph(adapter, cfg, Sd, prototype; real_output=true)
    sph_ref = vector_synthesis(adapter, cfg, Sd, zeroT, prototype; real_output=true)
    tor = vector_tor(adapter, cfg, Td, prototype; real_output=true)
    tor_ref = vector_synthesis(adapter, cfg, zeroS, Td, prototype; real_output=true)
    for k in 1:2
        @test vector_collect(adapter, sph[k], cfg) ≈ vector_collect(adapter, sph_ref[k], cfg) atol=tol.atol rtol=tol.rtol
        @test vector_collect(adapter, tor[k], cfg) ≈ vector_collect(adapter, tor_ref[k], cfg) atol=tol.atol rtol=tol.rtol
    end
    sphc = vector_sph_cplx(adapter, cfg, Sd, prototype)
    torc = vector_tor_cplx(adapter, cfg, Td, prototype)
    @test vector_collect(adapter, sphc[1], cfg) ≈
          vector_collect(adapter, vector_synthesis_cplx(adapter, cfg, Sd, zeroT, prototype)[1], cfg) atol=tol.atol rtol=tol.rtol
    @test vector_collect(adapter, torc[2], cfg) ≈
          vector_collect(adapter, vector_synthesis_cplx(adapter, cfg, zeroS, Td, prototype)[2], cfg) atol=tol.atol rtol=tol.rtol
    return nothing
end

function _test_vector_signs(adapter::VectorParityAdapter)
    cfg = _vector_config(:gauss, 2, 7)
    S, Tlm = _vector_modes(cfg, Float64)
    zeroS = zero.(S); zeroT = zero.(Tlm)
    refS = _direct_low_vector(cfg, S, zeroT; real_output=true)
    refT = _direct_low_vector(cfg, zeroS, Tlm; real_output=true)
    prototype = vector_place(adapter, cfg, refS[1], :spatial)
    gotS = vector_synthesis(adapter, cfg, vector_place(adapter, cfg, S, :spectral),
                            vector_place(adapter, cfg, zeroT, :spectral), prototype;
                            real_output=true)
    gotT = vector_synthesis(adapter, cfg, vector_place(adapter, cfg, zeroS, :spectral),
                            vector_place(adapter, cfg, Tlm, :spectral), prototype;
                            real_output=true)
    @test vector_collect(adapter, gotS[1], cfg) ≈ refS[1] atol=2e-12 rtol=2e-12
    @test vector_collect(adapter, gotS[2], cfg) ≈ refS[2] atol=2e-12 rtol=2e-12
    @test vector_collect(adapter, gotT[1], cfg) ≈ refT[1] atol=2e-12 rtol=2e-12
    @test vector_collect(adapter, gotT[2], cfg) ≈ refT[2] atol=2e-12 rtol=2e-12
end

function _test_vector_m1_pole_signal(adapter::VectorParityAdapter)
    cfg = _vector_config(:regular_poles, 2, 5; mres=1)
    CT = ComplexF64
    S_can = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    T_can = zero(S_can)
    S_can[2, 2] = CT(0.2, -0.1)
    T_can[2, 2] = CT(-0.12, 0.07)
    S = _external_vector_coefficients(cfg, S_can)
    Tlm = _external_vector_coefficients(cfg, T_can)
    prototype = vector_place(
        adapter, cfg, zeros(Float64, cfg.nlat, cfg.nlon), :spatial,
    )
    Vt, Vp = vector_synthesis(
        adapter, cfg, vector_place(adapter, cfg, S, :spectral),
        vector_place(adapter, cfg, Tlm, :spectral), prototype;
        real_output=true,
    )
    Vth = vector_collect(adapter, Vt, cfg)
    Vph = vector_collect(adapter, Vp, cfg)

    # Y_1^1 = -sqrt(3/(8π)) sin(θ) exp(iφ).  Its theta derivative
    # changes sign between poles, while Y/sin(θ) does not.
    c11 = sqrt(3 / (8pi))
    for i in findall(x -> isapprox(abs(x), 1; atol=8eps(Float64)), cfg.x),
        j in 1:cfg.nlon
        north = cfg.x[i] > 0
        d = north ? -c11 : c11
        p = -c11
        wave = cis(cfg.φ[j])
        expected_t = 2real(
            (d * S_can[2, 2] - im * p * T_can[2, 2]) * wave,
        )
        expected_p = 2real(
            (im * p * S_can[2, 2] + d * T_can[2, 2]) * wave,
        )
        @test Vth[i, j] ≈ expected_t atol=3e-12 rtol=3e-12
        @test Vph[i, j] ≈ expected_p atol=3e-12 rtol=3e-12
    end
    return nothing
end

function _test_vector_mres(adapter::VectorParityAdapter)
    cfg = _vector_config(:gauss, 3, 8; mres=2)
    unsupported = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    unsupported[2, 2] = 0.3 - 0.2im
    zeros_spectral = zero.(unsupported)
    prototype = vector_place(adapter, cfg, zeros(Float64, cfg.nlat, cfg.nlon), :spatial)
    Vt, Vp = vector_synthesis(
        adapter, cfg, vector_place(adapter, cfg, unsupported, :spectral),
        vector_place(adapter, cfg, zeros_spectral, :spectral), prototype;
        real_output=true,
    )
    @test all(iszero, vector_collect(adapter, Vt, cfg))
    @test all(iszero, vector_collect(adapter, Vp, cfg))

    c = sqrt(3 / (8pi))
    Vθ = [-2c * cfg.x[i] * sqrt(max(0.0, 1 - cfg.x[i]^2)) * cos(cfg.φ[j])
          for i in 1:cfg.nlat, j in 1:cfg.nlon]
    Vφ = [2c * sqrt(max(0.0, 1 - cfg.x[i]^2)) * sin(cfg.φ[j])
          for i in 1:cfg.nlat, j in 1:cfg.nlon]
    Sa, Ta = vector_analysis(
        adapter, cfg, vector_place(adapter, cfg, Vθ, :spatial),
        vector_place(adapter, cfg, Vφ, :spatial),
    )
    @test all(iszero, vector_collect(adapter, Sa, cfg)[:, 2])
    @test all(iszero, vector_collect(adapter, Ta, cfg)[:, 2])
end

function _test_vector_operators_and_l0()
    cfg = _vector_config(:gauss, 3, 8)
    S, Tlm = _vector_modes(cfg, Float64)
    S[1, 1] = 7 + 2im
    Tlm[1, 1] = -3 + im
    Vt, Vp = synthesis_sphtor(cfg, S, Tlm)
    S[1, 1] = 0; Tlm[1, 1] = 0
    @test Vt ≈ synthesis_sphtor(cfg, S, Tlm)[1]
    @test Vp ≈ synthesis_sphtor(cfg, S, Tlm)[2]
    div = divergence_from_spheroidal(cfg, S)
    vort = vorticity_from_toroidal(cfg, Tlm)
    @test iszero(div[1, 1]); @test iszero(vort[1, 1])
    @test spheroidal_from_divergence(cfg, div) ≈ S
    @test toroidal_from_vorticity(cfg, vort) ≈ Tlm
end

function _test_vector_operator_backend(adapter::VectorParityAdapter)
    cfg = _vector_config(:gauss, 3, 8; mres=2)
    S, Tlm = _vector_modes(cfg, Float32)
    S[1, 1] = 7f0 + 2f0im
    Tlm[1, 1] = -3f0 + 1f0im
    # Stored column 2 is physical m=1 and is forbidden when mres=2.
    # Operators must ignore adversarial nonzero storage there, not merely rely
    # on transforms to have cleared it earlier.
    S[2, 2] = 11f0 - 4f0im
    Tlm[3, 2] = -9f0 + 6f0im
    Sd = vector_place(adapter, cfg, S, :spectral)
    Td = vector_place(adapter, cfg, Tlm, :spectral)

    div = divergence_from_spheroidal(cfg, Sd)
    vort = vorticity_from_toroidal(cfg, Td)
    vector_resident(adapter, div); vector_resident(adapter, vort)
    divh = vector_collect(adapter, div, cfg)
    vorth = vector_collect(adapter, vort, cfg)
    @test divh ≈ divergence_from_spheroidal(cfg, S)
    @test vorth ≈ vorticity_from_toroidal(cfg, Tlm)
    @test iszero(divh[1, 1]); @test iszero(vorth[1, 1])
    @test all(iszero, divh[:, 2])
    @test all(iszero, vorth[:, 2])

    Sback = spheroidal_from_divergence(cfg, div)
    Tback = toroidal_from_vorticity(cfg, vort)
    expected_S = copy(S); expected_T = copy(Tlm)
    expected_S[1, 1] = 0; expected_T[1, 1] = 0
    expected_S[:, 2] .= 0; expected_T[:, 2] .= 0
    @test vector_collect(adapter, Sback, cfg) ≈ expected_S
    @test vector_collect(adapter, Tback, cfg) ≈ expected_T

    div_out = vector_place(adapter, cfg, zero.(S), :spectral)
    vort_out = vector_place(adapter, cfg, zero.(Tlm), :spectral)
    divergence_from_spheroidal!(cfg, div_out, Sd)
    vorticity_from_toroidal!(cfg, vort_out, Td)
    @test vector_collect(adapter, div_out, cfg) ≈ divh
    @test vector_collect(adapter, vort_out, cfg) ≈ vorth
    spheroidal_from_divergence!(cfg, div_out, div)
    toroidal_from_vorticity!(cfg, vort_out, vort)
    @test vector_collect(adapter, div_out, cfg) ≈ expected_S
    @test vector_collect(adapter, vort_out, cfg) ≈ expected_T
    return nothing
end

function _test_vector_views_and_plan()
    cfg = _vector_config(:gauss, 3, 8)
    S, Tlm = _vector_modes(cfg, Float64)
    Vt, Vp = synthesis_sphtor(cfg, S, Tlm)
    Vt_store = zeros(Float64, cfg.nlat, 2cfg.nlon)
    Vp_store = similar(Vt_store)
    @views Vt_store[:, 1:2:end] .= Vt
    @views Vp_store[:, 1:2:end] .= Vp
    Vtv = @view Vt_store[:, 1:2:end]
    Vpv = @view Vp_store[:, 1:2:end]
    @test analysis_sphtor(cfg, Vtv, Vpv)[1] ≈ S atol=4e-12 rtol=4e-12

    plan = SHTPlan(cfg)
    Sout = fill(99.0 + 2.0im, size(S)); Tout = fill(-88.0 + 3.0im, size(Tlm))
    bad = @view Vt_store[:, 1:(end - 1)]
    @test_throws DimensionMismatch analysis_sphtor!(plan, Sout, Tout, bad, Vpv)
    @test all(==(99 + 2im), Sout)
    @test all(==(-88 + 3im), Tout)
    analysis_sphtor!(plan, Sout, Tout, Vtv, Vpv)
    @test Sout ≈ S atol=4e-12 rtol=4e-12
    @test Tout ≈ Tlm atol=4e-12 rtol=4e-12
    Vto = similar(Vt); Vpo = similar(Vp)
    synthesis_sphtor!(plan, Vto, Vpo, S, Tlm)
    @test Vto ≈ Vt atol=4e-12 rtol=4e-12
    @test Vpo ≈ Vp atol=4e-12 rtol=4e-12
end

function run_sphtor_full_parity(adapter::VectorParityAdapter;
                                grid_kinds=_VECTOR_GRID_KINDS,
                                precisions=(Float32, Float64),
                                mres_values=(1, 2),
                                norms=(:orthonormal, :fourpi, :schmidt),
                                real_norm_values=(false, true),
                                cs_phase_values=(false, true),
                                robert_values=(false, true),
                                pole_orders=(false, true))
    @testset "spheroidal/toroidal full-grid parity $(nameof(typeof(adapter)))" begin
        for kind in grid_kinds, T in precisions, mres in mres_values,
            norm in norms, real_norm in real_norm_values,
            cs_phase in cs_phase_values, robert_form in robert_values,
            south_pole_first in pole_orders
            cfg = _vector_config(
                kind, 3, 8; mres, norm, real_norm, cs_phase,
                robert_form, south_pole_first,
            )
            _test_vector_case(adapter, cfg, T)
        end
        _test_vector_signs(adapter)
        _test_vector_m1_pole_signal(adapter)
        _test_vector_mres(adapter)
        _test_vector_operator_backend(adapter)
    end
    return nothing
end

function run_cpu_sphtor_full_parity()
    run_sphtor_full_parity(CPUVectorAdapter())
    @testset "CPU vector operators, views, and plans" begin
        _test_vector_operators_and_l0()
        _test_vector_views_and_plan()
    end
    return nothing
end

"""Compile and numerically validate the vendor-neutral vector kernels on KA CPU."""
function run_shared_vector_kernel_reference(common, backend)
    pole_cfg = _vector_config(:regular_poles, 2, 5; mres=1)
    pole_x, _, _, pole_Nlm = common.vector_host_tables(pole_cfg, Float32)
    pole_P = zeros(Float32, pole_cfg.nlat, pole_cfg.lmax + 1, pole_cfg.mmax + 1)
    pole_dtheta = similar(pole_P)
    pole_over_sin = similar(pole_P)
    event = common.vector_derivative_table_kernel!(backend)(
        pole_P, pole_dtheta, pole_over_sin, pole_x, pole_Nlm,
        pole_cfg.lmax, pole_cfg.mmax;
        ndrange=(pole_cfg.nlat, pole_cfg.mmax + 1),
    )
    event === nothing || wait(event)
    north = argmax(pole_x)
    south = argmin(pole_x)
    c11 = sqrt(Float32(3) / Float32(8pi))
    @test pole_dtheta[north, 2, 2] ≈ -c11 atol=8f-6 rtol=8f-6
    @test pole_over_sin[north, 2, 2] ≈ -c11 atol=8f-6 rtol=8f-6
    @test pole_dtheta[south, 2, 2] ≈ c11 atol=8f-6 rtol=8f-6
    @test pole_over_sin[south, 2, 2] ≈ -c11 atol=8f-6 rtol=8f-6

    @test isdefined(common, :vector_config_signature)
    if isdefined(common, :vector_config_signature)
        vector_cache = common.ScalarTableCache(2)
        identity = objectid(pole_cfg)
        scalar_signature = common.scalar_config_signature(pole_cfg)
        signature_before = common.vector_config_signature(pole_cfg)
        before = pole_dtheta[north, 2, 2]
        common.scalar_cache_insert!(
            vector_cache, :mock, identity, Float32, signature_before, :before,
        )
        pole_cfg.Nlm[2, 2] *= 2
        @test common.scalar_config_signature(pole_cfg) == scalar_signature
        signature_after = common.vector_config_signature(pole_cfg)
        @test signature_after != signature_before
        @test common.scalar_cache_lookup(
            vector_cache, :mock, identity, Float32, signature_after,
        ) === nothing
        common.scalar_cache_insert!(
            vector_cache, :mock, identity, Float32, signature_after, :after,
        )
        @test common.scalar_cache_size(vector_cache; device=:mock) == 1

        pole_x2, _, _, pole_Nlm2 = common.vector_host_tables(pole_cfg, Float32)
        event = common.vector_derivative_table_kernel!(backend)(
            pole_P, pole_dtheta, pole_over_sin, pole_x2, pole_Nlm2,
            pole_cfg.lmax, pole_cfg.mmax;
            ndrange=(pole_cfg.nlat, pole_cfg.mmax + 1),
        )
        event === nothing || wait(event)
        @test pole_dtheta[north, 2, 2] ≈ 2before atol=8f-6 rtol=8f-6
        @test pole_over_sin[north, 2, 2] ≈ -2c11 atol=8f-6 rtol=8f-6
    end

    cfg = _vector_config(
        :regular_poles, 3, 8; mres=2, norm=:schmidt,
        real_norm=true, cs_phase=false, robert_form=true,
    )
    T = Float32
    CT = ComplexF32
    x, weights, scales, Nlm = common.vector_host_tables(cfg, T)
    P = zeros(T, cfg.nlat, cfg.lmax + 1, cfg.mmax + 1)
    dtheta = similar(P)
    over_sin = similar(P)
    event = common.vector_derivative_table_kernel!(backend)(
        P, dtheta, over_sin, x, Nlm, cfg.lmax, cfg.mmax;
        ndrange=(cfg.nlat, cfg.mmax + 1),
    )
    event === nothing || wait(event)

    c10 = sqrt(T(3) / T(4pi))
    c22 = sqrt(T(15) / T(32pi))
    for i in 1:cfg.nlat
        xi = T(cfg.x[i]); si = sqrt(max(zero(T), one(T) - xi*xi))
        @test dtheta[i, 2, 1] ≈ -c10 * si atol=8f-6 rtol=8f-6
        @test dtheta[i, 3, 3] ≈ T(2) * c22 * si * xi atol=8f-6 rtol=8f-6
        @test over_sin[i, 3, 3] ≈ c22 * si atol=8f-6 rtol=8f-6
    end
    @test all(isfinite, dtheta)
    @test all(isfinite, over_sin)

    # Vector modes have no l=0 degree.  Analyze an independently synthesized
    # axisymmetric mode and require the shared kernel to leave that logical
    # storage slot finite and exactly zero while recovering active degrees.
    lcap0 = cfg.lmax
    S_mode0 = CT[0, 0.05 - 0.02im, -0.03 + 0.01im, 0.02 + 0.015im]
    T_mode0 = CT[0, -0.025 + 0.01im, 0.015 - 0.005im, -0.01 + 0.02im]
    Vt_mode0, Vp_mode0 = synthesis_sphtor_ml(
        SHTnsKit.CPU(), cfg, 0, S_mode0, T_mode0, lcap0,
    )
    S_mode_out = fill(CT(91, -17), lcap0 + 1)
    T_mode_out = fill(CT(-83, 23), lcap0 + 1)
    event = common.vector_mode_analysis_kernel!(backend)(
        S_mode_out, T_mode_out, Vt_mode0, Vp_mode0, dtheta, over_sin,
        weights, scales, x, T(cfg.cphi), 0, lcap0, cfg.robert_form;
        ndrange=lcap0 + 1,
    )
    event === nothing || wait(event)
    @test isfinite(S_mode_out[1]) && iszero(S_mode_out[1])
    @test isfinite(T_mode_out[1]) && iszero(T_mode_out[1])
    @test all(isfinite, S_mode_out)
    @test all(isfinite, T_mode_out)
    @test S_mode_out[2:end] ≈ S_mode0[2:end] atol=4f-4 rtol=4f-4
    @test T_mode_out[2:end] ≈ T_mode0[2:end] atol=4f-4 rtol=4f-4

    # Synthesis and the spheroidal-only gradient convention must ignore l=0,
    # even when its storage slot contains adversarial noise.
    noisy_S0 = copy(S_mode0); noisy_T0 = copy(T_mode0)
    noisy_S0[1] = CT(77, -31); noisy_T0[1] = CT(-59, 19)
    Vt_shared = zeros(CT, cfg.nlat); Vp_shared = similar(Vt_shared)
    event = common.vector_mode_synthesis_kernel!(backend)(
        Vt_shared, Vp_shared, noisy_S0, noisy_T0, dtheta, over_sin,
        scales, x, T(SHTnsKit.phi_inv_scale(cfg)), 0, lcap0,
        cfg.robert_form; ndrange=cfg.nlat,
    )
    event === nothing || wait(event)
    @test Vt_shared ≈ Vt_mode0 atol=8f-6 rtol=8f-6
    @test Vp_shared ≈ Vp_mode0 atol=8f-6 rtol=8f-6
    grad_ref = synthesis_grad_ml(SHTnsKit.CPU(), cfg, 0, noisy_S0, lcap0)
    @test grad_ref == synthesis_sph_ml(
        SHTnsKit.CPU(), cfg, 0, noisy_S0, lcap0,
    )

    S_can, T_can = _vector_modes(cfg, T)
    S = _external_vector_coefficients(cfg, S_can)
    Tlm = _external_vector_coefficients(cfg, T_can)
    fourier_t = zeros(CT, cfg.nlat, cfg.nlon)
    fourier_p = zeros(CT, cfg.nlat, cfg.nlon)
    event = common.vector_synthesis_kernel!(backend)(
        fourier_t, fourier_p, S, Tlm, dtheta, over_sin, scales, x,
        T(SHTnsKit.phi_inv_scale(cfg)), cfg.nlon, cfg.lmax, cfg.mmax,
        cfg.mres, true, cfg.robert_form;
        ndrange=(cfg.nlat, cfg.mmax + 1),
    )
    event === nothing || wait(event)
    @test all(iszero, fourier_t[:, 2])
    @test fourier_t[:, cfg.nlon - 1] ≈ conj.(fourier_t[:, 3])
    Vt = copy(fourier_t); Vp = copy(fourier_p)
    SHTnsKit.ifft_phi!(Vt, Vt); SHTnsKit.ifft_phi!(Vp, Vp)
    Vt_ref, Vp_ref = _direct_low_vector(cfg, S_can, T_can; real_output=true)
    @test real.(Vt) ≈ Vt_ref atol=4f-4 rtol=4f-4
    @test real.(Vp) ≈ Vp_ref atol=4f-4 rtol=4f-4

    SHTnsKit.fft_phi!(fourier_t, Vt_ref)
    SHTnsKit.fft_phi!(fourier_p, Vp_ref)
    S_out = zeros(CT, size(S)); T_out = similar(S_out)
    event = common.vector_analysis_kernel!(backend)(
        S_out, T_out, fourier_t, fourier_p, dtheta, over_sin,
        weights, scales, x, T(cfg.cphi), cfg.lmax, cfg.mmax,
        cfg.mres, cfg.robert_form;
        ndrange=size(S_out),
    )
    event === nothing || wait(event)
    @test S_out ≈ S atol=4f-4 rtol=4f-4
    @test T_out ≈ Tlm atol=4f-4 rtol=4f-4
    @test all(iszero, S_out[:, 2])
    @test all(iszero, T_out[:, 2])

    # As for scalar transpose plans, local columns are contiguous physical
    # orders.  Exercise a nonzero first order so configured convention scales
    # must be indexed by the global order on both public boundaries.
    first_m = 1
    local_orders = 2
    local_range = (first_m + 1):(first_m + local_orders)
    distributed_Ft = reshape(
        copy(fourier_t[:, local_range]), cfg.nlat, local_orders, 1,
    )
    distributed_Fp = reshape(
        copy(fourier_p[:, local_range]), cfg.nlat, local_orders, 1,
    )
    distributed_S = zeros(CT, cfg.lmax + 1, local_orders, 1)
    distributed_T = similar(distributed_S)
    event = common.distributed_vector_analysis_kernel!(backend)(
        distributed_S, distributed_T, distributed_Ft, distributed_Fp,
        dtheta, over_sin, weights, scales, x, T(cfg.cphi), first_m,
        cfg.lmax, cfg.mmax, cfg.mres, cfg.robert_form;
        ndrange=size(distributed_S),
    )
    event === nothing || wait(event)
    @test distributed_S[:, :, 1] ≈ S_out[:, local_range] atol=4f-4 rtol=4f-4
    @test distributed_T[:, :, 1] ≈ T_out[:, local_range] atol=4f-4 rtol=4f-4

    distributed_Vt = zeros(CT, cfg.nlat, local_orders, 1)
    distributed_Vp = similar(distributed_Vt)
    distributed_Sin = reshape(
        copy(S[:, local_range]), cfg.lmax + 1, local_orders, 1,
    )
    distributed_Tin = reshape(
        copy(Tlm[:, local_range]), cfg.lmax + 1, local_orders, 1,
    )
    event = common.distributed_vector_synthesis_kernel!(backend)(
        distributed_Vt, distributed_Vp, distributed_Sin, distributed_Tin,
        dtheta, over_sin, scales, x, T(SHTnsKit.phi_inv_scale(cfg)), first_m,
        cfg.lmax, cfg.mmax, cfg.mres, cfg.robert_form;
        ndrange=size(distributed_Vt),
    )
    event === nothing || wait(event)
    @test isapprox(
        distributed_Vt[:, :, 1], fourier_t[:, local_range];
        atol=8f-6, rtol=8f-6,
    )
    @test isapprox(
        distributed_Vp[:, :, 1], fourier_p[:, local_range];
        atol=8f-6, rtol=8f-6,
    )

    S[2, 2] = CT(13, -5)
    diagonal = similar(S)
    event = common.vector_diagonal_kernel!(backend)(
        diagonal, S, cfg.lmax, cfg.mmax, cfg.mres, false;
        ndrange=size(S),
    )
    event === nothing || wait(event)
    @test diagonal ≈ divergence_from_spheroidal(cfg, S)
    @test all(iszero, diagonal[:, 2])
    recovered = similar(S)
    event = common.vector_diagonal_kernel!(backend)(
        recovered, diagonal, cfg.lmax, cfg.mmax, cfg.mres, true;
        ndrange=size(S),
    )
    event === nothing || wait(event)
    expected = copy(S); expected[1, 1] = 0; expected[:, 2] .= 0
    @test recovered ≈ expected
    return nothing
end
