using Test
using SHTnsKit

abstract type LocalEvaluationAdapter end
struct CPULocalEvaluationAdapter <: LocalEvaluationAdapter end

local_place(::CPULocalEvaluationAdapter, value) = value
local_collect(::CPULocalEvaluationAdapter, value) = value
local_scalar(::CPULocalEvaluationAdapter, cfg, coefficients, cost, phi) =
    synthesis_point(CPU(), cfg, coefficients, cost, phi)
local_scalar_cplx(::CPULocalEvaluationAdapter, cfg, coefficients, cost, phi) =
    synthesis_point_cplx(CPU(), cfg, coefficients, cost, phi)
local_lat(::CPULocalEvaluationAdapter, cfg, coefficients, cost; kwargs...) =
    SH_to_lat(CPU(), cfg, coefficients, cost; kwargs...)
local_lat_cplx(::CPULocalEvaluationAdapter, cfg, coefficients, cost; kwargs...) =
    SH_to_lat_cplx(CPU(), cfg, coefficients, cost; kwargs...)
local_qst_point(::CPULocalEvaluationAdapter, cfg, Q, S, Tlm, cost, phi) =
    SHqst_to_point(CPU(), cfg, Q, S, Tlm, cost, phi)
local_qst_lat(::CPULocalEvaluationAdapter, cfg, Q, S, Tlm, cost; kwargs...) =
    SHqst_to_lat(CPU(), cfg, Q, S, Tlm, cost; kwargs...)
local_grad_point(::CPULocalEvaluationAdapter, cfg, Dr, S, cost, phi) =
    SH_to_grad_point(CPU(), cfg, Dr, S, cost, phi)
local_assert_resident(::CPULocalEvaluationAdapter, value) = @test on_device(value) isa CPU

_local_tol(::Type{Float32}) = (atol=4f-5, rtol=4f-5)
_local_tol(::Type{Float64}) = (atol=3e-13, rtol=3e-13)

function _local_config(kind::Symbol, ::Type{T}; mres::Int=1,
                       norm::Symbol=:orthonormal, real_norm::Bool=false,
                       cs_phase::Bool=true, robert_form::Bool=false) where {T}
    lmax = 2
    nlat = 6
    kwargs = (; mres, nlon=6, norm, real_norm, cs_phase, robert_form)
    cfg = if kind === :gauss
        create_gauss_config(lmax, nlat; kwargs...)
    elseif kind === :gauss_fly
        create_gauss_fly_config(lmax, nlat; kwargs...)
    elseif kind === :regular
        create_regular_config(lmax, nlat; kwargs..., include_poles=false,
                              precompute_plm=true)
    elseif kind === :regular_poles
        create_regular_config(lmax, nlat; kwargs..., include_poles=true,
                              precompute_plm=true)
    else
        error("unknown local grid kind: $kind")
    end
    return cfg
end

function _local_canonical_modes(cfg, ::Type{T}) where {T<:AbstractFloat}
    CT = Complex{T}
    Q = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    S = zeros(CT, size(Q))
    Tlm = zeros(CT, size(Q))
    Dr = zeros(CT, size(Q))
    Q[1, 1] = CT(T(0.31), 0)
    Q[2, 1] = CT(T(-0.17), 0)
    Dr[1, 1] = CT(T(0.73), 0)
    Dr[2, 1] = CT(T(-0.08), 0)
    S[2, 1] = CT(T(0.19), 0)
    Tlm[2, 1] = CT(T(-0.11), 0)
    if cfg.mres == 1
        Q[2, 2] = CT(T(0.13), T(-0.09))
        Dr[2, 2] = CT(T(-0.12), T(0.06))
        S[2, 2] = CT(T(0.07), T(-0.05))
        Tlm[2, 2] = CT(T(-0.04), T(0.09))
    end
    Q[3, 3] = CT(T(-0.08), T(0.04))
    Dr[3, 3] = CT(T(0.05), T(0.03))
    S[3, 3] = CT(T(0.13), T(-0.07))
    Tlm[3, 3] = CT(T(-0.09), T(0.05))
    return Q, S, Tlm, Dr
end

function _local_external(cfg, canonical)
    external = zero(canonical)
    SHTnsKit.convert_alm_norm!(external, canonical, cfg; to_internal=false)
    return external
end

function _local_packed(cfg, canonical)
    return SHTnsKit.pack_lm(cfg, _local_external(cfg, canonical))
end

"""
Independent low-order real scalar sum. The harmonics are closed forms and do
not call package Legendre or transform helpers.
"""
function _local_direct_scalar(canonical, cost, phi; real_output::Bool=true,
                              ltr::Int=2, mtr::Int=2)
    T = promote_type(real(eltype(canonical)), typeof(cost), typeof(phi))
    CT = Complex{T}
    x = T(cost)
    s = sqrt(max(zero(T), one(T) - x * x))
    value = zero(CT)
    if ltr >= 0
        value += canonical[1, 1] / sqrt(T(4pi))
    end
    if ltr >= 1
        value += canonical[2, 1] * sqrt(T(3) / T(4pi)) * x
        if size(canonical, 2) >= 2 && mtr >= 1
            y11 = -sqrt(T(3) / T(8pi)) * s * cis(T(phi))
            wave = canonical[2, 2] * y11
            value += real_output ? CT(T(2) * real(wave), 0) : wave
        end
    end
    if ltr >= 2 && size(canonical, 2) >= 3 && mtr >= 2
        y22 = sqrt(T(15) / T(32pi)) * s * s * cis(T(2) * T(phi))
        wave = canonical[3, 3] * y22
        value += real_output ? CT(T(2) * real(wave), 0) : wave
    end
    return real_output ? real(value) : value
end

"""Independent low-order Q/S/T sum with analytic m=1 pole limits."""
function _local_direct_qst(cfg, Q, S, Tlm, cost, phi)
    T = promote_type(real(eltype(Q)), real(eltype(S)), real(eltype(Tlm)), typeof(cost), typeof(phi))
    CT = Complex{T}
    x = T(cost)
    s = sqrt(max(zero(T), one(T) - x * x))
    vr = Q[1, 1] / sqrt(T(4pi))
    c10 = sqrt(T(3) / T(4pi))
    vr += Q[2, 1] * c10 * x
    vt = -c10 * s * S[2, 1]
    vp = -c10 * s * Tlm[2, 1]

    if cfg.mres == 1
        c11 = -sqrt(T(3) / T(8pi))
        phase = cis(T(phi))
        y = c11 * s
        d = c11 * x
        over = c11
        vr += CT(T(2) * real(Q[2, 2] * y * phase), 0)
        vt += CT(T(2) * real((d * S[2, 2] - CT(0, 1) * over * Tlm[2, 2]) * phase), 0)
        vp += CT(T(2) * real((CT(0, 1) * over * S[2, 2] + d * Tlm[2, 2]) * phase), 0)
    end

    c22 = sqrt(T(15) / T(32pi))
    phase2 = cis(T(2) * T(phi))
    y2 = c22 * s * s
    d2 = T(2) * c22 * s * x
    over2 = c22 * s
    vr += CT(T(2) * real(Q[3, 3] * y2 * phase2), 0)
    vt += CT(T(2) * real((d2 * S[3, 3] - CT(0, 2) * over2 * Tlm[3, 3]) * phase2), 0)
    vp += CT(T(2) * real((CT(0, 2) * over2 * S[3, 3] + d2 * Tlm[3, 3]) * phase2), 0)
    if cfg.robert_form
        vt *= s
        vp *= s
    end
    return real(vr), real(vt), real(vp)
end

function _local_complex_modes(cfg, ::Type{T}) where {T<:AbstractFloat}
    CT = Complex{T}
    canonical = zeros(CT, nlm_cplx_calc(cfg.lmax, cfg.mmax, 1))
    for (l, m, value) in (
        (0, 0, CT(T(0.21), T(-0.03))),
        (1, 0, CT(T(-0.17), T(0.08))),
        (1, 1, CT(T(0.13), T(-0.09))),
        (1, -1, CT(T(-0.04), T(0.07))),
        (2, 2, CT(T(0.06), T(0.02))),
        (2, -2, CT(T(-0.03), T(-0.05))),
    )
        canonical[LM_cplx_index(cfg.lmax, cfg.mmax, l, m) + 1] = value
    end
    external = similar(canonical)
    SHTnsKit.convert_alm_norm!(external, canonical, cfg; to_internal=false)
    return canonical, external
end

function _local_direct_complex(canonical, cost, phi; ltr::Int=2)
    T = promote_type(real(eltype(canonical)), typeof(cost), typeof(phi))
    x = T(cost)
    s = sqrt(max(zero(T), one(T) - x * x))
    value = canonical[LM_cplx_index(2, 2, 0, 0) + 1] / sqrt(T(4pi))
    if ltr >= 1
        value += canonical[LM_cplx_index(2, 2, 1, 0) + 1] * sqrt(T(3) / T(4pi)) * x
        c11 = -sqrt(T(3) / T(8pi)) * s
        value += c11 * (
            canonical[LM_cplx_index(2, 2, 1, 1) + 1] * cis(T(phi)) +
            canonical[LM_cplx_index(2, 2, 1, -1) + 1] * cis(-T(phi))
        )
    end
    if ltr >= 2
        c22 = sqrt(T(15) / T(32pi)) * s * s
        value += c22 * (
            canonical[LM_cplx_index(2, 2, 2, 2) + 1] * cis(T(2) * T(phi)) +
            canonical[LM_cplx_index(2, 2, 2, -2) + 1] * cis(-T(2) * T(phi))
        )
    end
    return value
end

function _test_local_case(adapter::LocalEvaluationAdapter, cfg, ::Type{T}) where {T<:AbstractFloat}
    tol = _local_tol(T)
    Qcan, Scan, Tcan, Drcan = _local_canonical_modes(cfg, T)
    Q = _local_external(cfg, Qcan)
    Qp = _local_packed(cfg, Qcan)
    Sp = _local_packed(cfg, Scan)
    Tp = _local_packed(cfg, Tcan)
    Drp = _local_packed(cfg, Drcan)
    cost = T(0.37)
    phi = T(0.61)

    dense_device = local_place(adapter, Q)
    q_device = local_place(adapter, Qp)
    s_device = local_place(adapter, Sp)
    t_device = local_place(adapter, Tp)
    dr_device = local_place(adapter, Drp)

    point = local_scalar(adapter, cfg, dense_device, cost, phi)
    @test local_collect(adapter, point) ≈ _local_direct_scalar(Qcan, cost, phi) atol=tol.atol rtol=tol.rtol
    @test local_collect(adapter, local_scalar(adapter, cfg, dense_device, cost, phi + T(2pi))) ≈
          local_collect(adapter, point) atol=tol.atol rtol=tol.rtol

    nphi = 7
    lat = local_lat(adapter, cfg, q_device, cost; nphi)
    local_assert_resident(adapter, lat)
    lat_host = local_collect(adapter, lat)
    @test eltype(lat_host) === T
    @test lat_host ≈ [_local_direct_scalar(Qcan, cost, T(2pi * j / nphi)) for j in 0:(nphi - 1)] atol=tol.atol rtol=tol.rtol

    qst = local_qst_point(adapter, cfg, q_device, s_device, t_device, cost, phi)
    qst_host = map(value -> local_collect(adapter, value), qst)
    @test all(value -> value isa T, qst_host)
    @test collect(qst_host) ≈ collect(_local_direct_qst(cfg, Qcan, Scan, Tcan, cost, phi)) atol=tol.atol rtol=tol.rtol

    qst_lat = local_qst_lat(adapter, cfg, q_device, s_device, t_device, cost; nphi)
    refs = [_local_direct_qst(cfg, Qcan, Scan, Tcan, cost, T(2pi * j / nphi)) for j in 0:(nphi - 1)]
    for component in 1:3
        local_assert_resident(adapter, qst_lat[component])
        host = local_collect(adapter, qst_lat[component])
        @test eltype(host) === T
        @test host ≈ getindex.(refs, component) atol=tol.atol rtol=tol.rtol
    end

    grad = local_grad_point(adapter, cfg, dr_device, s_device, cost, phi)
    grad_host = map(value -> local_collect(adapter, value), grad)
    expected_grad = _local_direct_qst(cfg, Drcan, Scan, zero(Tcan), cost, phi)
    @test collect(grad_host) ≈ collect(expected_grad) atol=tol.atol rtol=tol.rtol

    return nothing
end

function _test_local_poles(adapter::LocalEvaluationAdapter)
    for cost in (-1.0, 1.0), phi in (-2pi, -0.3, 0.0, 2pi)
        cfg = _local_config(:regular_poles, Float64; robert_form=false)
        Q, S, Tlm, Dr = _local_canonical_modes(cfg, Float64)
        args = map(x -> local_place(adapter, _local_packed(cfg, x)), (Q, S, Tlm, Dr))
        qst = map(value -> local_collect(adapter, value),
                  local_qst_point(adapter, cfg, args[1], args[2], args[3], cost, phi))
        @test collect(qst) ≈ collect(_local_direct_qst(cfg, Q, S, Tlm, cost, phi)) atol=4e-13 rtol=4e-13
        grad = map(value -> local_collect(adapter, value),
                   local_grad_point(adapter, cfg, args[4], args[2], cost, phi))
        @test collect(grad) ≈ collect(_local_direct_qst(cfg, Dr, S, zero(Tlm), cost, phi)) atol=4e-13 rtol=4e-13
    end
    return nothing
end

function _test_local_complex(adapter::LocalEvaluationAdapter)
    for T in (Float32, Float64), convention in (
        (norm=:orthonormal, real_norm=false, cs_phase=true),
        (norm=:schmidt, real_norm=true, cs_phase=false),
    )
        cfg = _local_config(:gauss, T; convention...)
        canonical, external = _local_complex_modes(cfg, T)
        placed = local_place(adapter, external)
        tol = _local_tol(T)
        cost = T(-0.28)
        phi = T(0.77)
        point = local_scalar_cplx(adapter, cfg, placed, cost, phi)
        @test local_collect(adapter, point) ≈ _local_direct_complex(canonical, cost, phi) atol=tol.atol rtol=tol.rtol
        lat = local_lat_cplx(adapter, cfg, placed, cost; nphi=7, ltr=cfg.lmax)
        local_assert_resident(adapter, lat)
        host = local_collect(adapter, lat)
        @test eltype(host) === Complex{T}
        @test host ≈ [_local_direct_complex(canonical, cost, T(2pi * j / 7)) for j in 0:6] atol=tol.atol rtol=tol.rtol
    end
    return nothing
end

function _test_local_views_and_filtering(adapter::LocalEvaluationAdapter)
    cfg = _local_config(:gauss, Float64; mres=2)
    Qcan, Scan, Tcan, Drcan = _local_canonical_modes(cfg, Float64)
    dense = _local_external(cfg, Qcan)
    dense[2, 2] = 99 - 41im
    padded_dense = zeros(ComplexF64, size(dense, 1), 2size(dense, 2))
    @views padded_dense[:, 1:2:end] .= dense
    dense_view = @view padded_dense[:, 1:2:end]
    @test stride(dense_view, 2) > size(dense_view, 1)
    cost, phi = 0.23, -0.91
    @test local_collect(adapter, local_scalar(adapter, cfg, local_place(adapter, dense_view), cost, phi)) ≈
          _local_direct_scalar(Qcan, cost, phi) atol=3e-13 rtol=3e-13

    packed = map(x -> _local_packed(cfg, x), (Qcan, Scan, Tcan, Drcan))
    views = map(packed) do values
        padded = zeros(eltype(values), 2length(values))
        @views padded[1:2:end] .= values
        local_place(adapter, @view padded[1:2:end])
    end
    got = map(value -> local_collect(adapter, value),
              local_qst_point(adapter, cfg, views[1], views[2], views[3], cost, phi))
    @test collect(got) ≈ collect(_local_direct_qst(cfg, Qcan, Scan, Tcan, cost, phi)) atol=3e-13 rtol=3e-13
    grad = map(value -> local_collect(adapter, value),
               local_grad_point(adapter, cfg, views[4], views[2], cost, phi))
    @test collect(grad) ≈ collect(_local_direct_qst(cfg, Drcan, Scan, zero(Tcan), cost, phi)) atol=3e-13 rtol=3e-13
    return nothing
end

"""Exercise racing first-use table builds on hardware-backed task streams."""
function run_concurrent_local_first_use(adapter::LocalEvaluationAdapter, clear_cache!)
    cfg = _local_config(:gauss, Float32)
    Qcan = first(_local_canonical_modes(cfg, Float32))
    coefficients = local_place(adapter, _local_external(cfg, Qcan))
    cost, phi = 0.37f0, 0.61f0
    expected = _local_direct_scalar(Qcan, cost, phi)
    clear_cache!()
    tasks = [Threads.@spawn local_scalar(
        adapter, cfg, coefficients, cost, phi,
    ) for _ in 1:8]
    values = map(task -> local_collect(adapter, fetch(task)), tasks)
    @test values ≈ fill(expected, length(values)) atol=4f-5 rtol=4f-5
    return nothing
end

function run_local_evaluation_parity(adapter::LocalEvaluationAdapter;
                                     exhaustive::Bool=true)
    @testset "local evaluation parity $(nameof(typeof(adapter)))" begin
        grids = exhaustive ? (:gauss, :gauss_fly, :regular, :regular_poles) : (:gauss,)
        precisions = exhaustive ? (Float32, Float64) : (Float32,)
        conventions = exhaustive ? (
            (norm=:orthonormal, real_norm=false, cs_phase=true),
            (norm=:fourpi, real_norm=false, cs_phase=false),
            (norm=:schmidt, real_norm=true, cs_phase=true),
            (norm=:orthonormal, real_norm=true, cs_phase=false),
        ) : ((norm=:schmidt, real_norm=true, cs_phase=false),)
        for kind in grids, T in precisions, mres in (1, 2), convention in conventions,
            robert_form in (false, true)
            cfg = _local_config(kind, T; mres, convention..., robert_form)
            _test_local_case(adapter, cfg, T)
        end
        _test_local_poles(adapter)
        _test_local_complex(adapter)
        _test_local_views_and_filtering(adapter)
    end
    return nothing
end

function test_cpu_local_compatibility_and_validation()
    cfg = _local_config(:gauss, Float64)
    Q, S, Tlm, Dr = _local_canonical_modes(cfg, Float64)
    dense = _local_external(cfg, Q)
    packed = map(x -> _local_packed(cfg, x), (Q, S, Tlm, Dr))
    cost, phi = 0.3, 0.7
    @test synthesis_point(CPU(), cfg, dense, cost, phi) == synthesis_point(cfg, dense, cost, phi)
    @test SH_to_lat(CPU(), cfg, packed[1], cost) == SH_to_lat(cfg, packed[1], cost)
    @test SHqst_to_point(CPU(), cfg, packed[1:3]..., cost, phi) == SHqst_to_point(cfg, packed[1:3]..., cost, phi)
    @test SHqst_to_lat(CPU(), cfg, packed[1:3]..., cost) == SHqst_to_lat(cfg, packed[1:3]..., cost)
    @test SH_to_grad_point(CPU(), cfg, packed[4], packed[2], cost, phi) == SH_to_grad_point(cfg, packed[4], packed[2], cost, phi)

    complex_canonical, complex_external = _local_complex_modes(cfg, Float64)
    @test synthesis_point_cplx(CPU(), cfg, complex_external, cost, phi) == synthesis_point_cplx(cfg, complex_external, cost, phi)
    @test SH_to_lat_cplx(CPU(), cfg, complex_external, cost) == SH_to_lat_cplx(cfg, complex_external, cost)

    @test_throws ArgumentError synthesis_point(cfg, dense, 1.01, phi)
    @test_throws ArgumentError synthesis_point(cfg, dense, cost, Inf)
    @test_throws DimensionMismatch synthesis_point(cfg, @view(dense[1:2, :]), cost, phi)
    @test_throws DimensionMismatch SH_to_lat(cfg, packed[1][1:end-1], cost)
    @test_throws ArgumentError SH_to_lat(cfg, packed[1], cost; nphi=0)
    @test_throws ArgumentError SH_to_lat(cfg, packed[1], cost; ltr=-1)
    @test_throws ArgumentError SH_to_lat(cfg, packed[1], cost; mtr=cfg.mmax + 1)
    @test_throws DimensionMismatch SHqst_to_point(cfg, packed[1][1:end-1], packed[2], packed[3], cost, phi)
    @test_throws DimensionMismatch SH_to_grad_point(cfg, packed[4][1:end-1], packed[2], cost, phi)
    @test_throws DimensionMismatch synthesis_point_cplx(cfg, complex_external[1:end-1], cost, phi)
    @test_throws ArgumentError SH_to_lat_cplx(cfg, complex_external, cost; nphi=-1)
    @test complex_canonical isa Vector{ComplexF64}
    return nothing
end

function test_cpu_local_mixed_coordinate_precision()
    @testset "Local results retain coefficient precision with wider coordinates" begin
        for (Tcoeff, Tcoord) in ((Float32, Float64), (Float64, Float32))
            cfg = _local_config(:gauss, Tcoeff)
            Qcan, Scan, Tcan, Drcan = _local_canonical_modes(cfg, Tcoeff)
            dense = _local_external(cfg, Qcan)
            Qp, Sp, Tp, Drp = map(
                values -> _local_packed(cfg, values),
                (Qcan, Scan, Tcan, Drcan),
            )
            complex_can, complex_external = _local_complex_modes(cfg, Tcoeff)
            cost, phi = Tcoord(0.37), Tcoord(0.61)
            tol = _local_tol(Tcoeff)
            PT = promote_type(Tcoeff, Tcoord)

            scalar = synthesis_point(CPU(), cfg, dense, cost, phi)
            @test scalar isa Tcoeff
            @test scalar ≈ _local_direct_scalar(
                Qcan, cost, phi,
            ) atol=tol.atol rtol=tol.rtol

            scalar_complex = synthesis_point_cplx(
                CPU(), cfg, complex_external, cost, phi,
            )
            @test scalar_complex isa Complex{Tcoeff}
            @test scalar_complex ≈ _local_direct_complex(
                complex_can, cost, phi,
            ) atol=tol.atol rtol=tol.rtol

            latitude = SH_to_lat(CPU(), cfg, Qp, cost; nphi=7)
            complex_latitude = SH_to_lat_cplx(
                CPU(), cfg, complex_external, cost; nphi=7,
            )
            @test eltype(latitude) === Tcoeff
            @test eltype(complex_latitude) === Complex{Tcoeff}
            @test latitude ≈ [
                _local_direct_scalar(Qcan, cost, PT(2pi*j/7))
                for j in 0:6
            ] atol=tol.atol rtol=tol.rtol
            @test complex_latitude ≈ [
                _local_direct_complex(complex_can, cost, PT(2pi*j/7))
                for j in 0:6
            ] atol=tol.atol rtol=tol.rtol

            qst = SHqst_to_point(CPU(), cfg, Qp, Sp, Tp, cost, phi)
            grad = SH_to_grad_point(CPU(), cfg, Drp, Sp, cost, phi)
            @test all(value -> value isa Tcoeff, qst)
            @test all(value -> value isa Tcoeff, grad)
            @test collect(qst) ≈ collect(_local_direct_qst(
                cfg, Qcan, Scan, Tcan, cost, phi,
            )) atol=tol.atol rtol=tol.rtol
            @test collect(grad) ≈ collect(_local_direct_qst(
                cfg, Drcan, Scan, zero(Tcan), cost, phi,
            )) atol=tol.atol rtol=tol.rtol

            qst_lat = SHqst_to_lat(CPU(), cfg, Qp, Sp, Tp, cost; nphi=7)
            @test all(values -> eltype(values) === Tcoeff, qst_lat)
            refs = [_local_direct_qst(
                cfg, Qcan, Scan, Tcan, cost, PT(2pi*j/7),
            ) for j in 0:6]
            for component in 1:3
                @test qst_lat[component] ≈ getindex.(refs, component) atol=tol.atol rtol=tol.rtol
            end
        end
    end
    return nothing
end
