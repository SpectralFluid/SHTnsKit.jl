"""
Local/partial evaluations along latitude circles and at points.
"""

@inline function _validate_local_storage(operation::Symbol, arrays...)
    devices = map(on_device, arrays)
    all(device -> typeof(device) === typeof(first(devices)), devices) ||
        throw(ArgumentError("$operation requires all coefficient arrays on the same backend"))
    first(devices) isa GPU && throw(ArgumentError(
        "$operation with GPU storage requires GPU() dispatch and one vendor's arrays",
    ))
    return nothing
end

"""
    SH_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real;
              nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vector{<:Real}

Evaluate a real field along a latitude (fixed cosθ = cost) at `nphi` equispaced longitudes.
Uses orthonormal harmonics and packed real coefficients `Qlm` (LM order).
"""
function SH_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    _validate_local_storage(:SH_to_lat, Qlm)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    _validate_local_cost(cost, :SH_to_lat)
    _validate_local_nphi(nphi, :SH_to_lat)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    (0 ≤ mtr ≤ cfg.mmax) || throw(ArgumentError("mtr must be within [0, mmax]"))
    Qlm_int = _internal_coefficients(Qlm, cfg)
    CT = eltype(Qlm_int)
    RT = typeof(real(zero(CT)))
    PT = _local_basis_type(RT, cost)
    AT, ACT = _local_accumulator_types(CT, PT)
    x = convert(PT, cost)
    lmax = cfg.lmax
    # Accumulator/output eltype follows the input so AD types (e.g.
    # ForwardDiff.Dual) propagate; defaults to Float64 for ComplexF64 input.
    P = Vector{PT}(undef, lmax + 1)
    vals = Vector{AT}(undef, nphi)
    fill!(vals, zero(AT))


    # m=0 contribution
    Plm_norm_row!(P, x, lmax, 0)
    g0 = zero(ACT)
    @inbounds for l in 0:ltr
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        a = Qlm_int[lm]
        g0 += P[l+1] * a
    end

    @inbounds for j in 0:(nphi-1)
        vals[j+1] = real(g0)
    end

    # m>0
    for m in 1:mtr
        (m % cfg.mres == 0) || continue
        Plm_norm_row!(P, x, lmax, m)
        gm = zero(ACT)
        @inbounds for l in m:min(ltr, lmax)
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            a = Qlm_int[lm]
            gm += P[l+1] * a
        end
        
        for j in 0:(nphi-1)
            vals[j+1] += 2 * real(gm * cis(PT(2π * m * j / nphi)))
        end
    end
    sφ = _evaluator_phi_scale(cfg)   # 1 under :dft; 1/2π under :quad
    sφ == 1 || (vals .*= sφ)
    return vals
end

"""
    SH_to_lat_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax) -> Vector{<:Complex}

Evaluate a complex field along a latitude using packed LM_cplx coefficients.
"""
function SH_to_lat_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax)
    _validate_local_storage(:SH_to_lat_cplx, alm_packed)
    lmax, mmax = cfg.lmax, cfg.mmax
    # The packing/indexing below assumes mres==1 (dense m). Fail loudly rather
    # than silently returning wrong values for a strided (mres>1) configuration.
    cfg.mres == 1 || throw(ArgumentError("SH_to_lat_cplx supports mres==1 only; got mres=$(cfg.mres)"))
    length(alm_packed) == nlm_cplx_calc(lmax, mmax, 1) || throw(DimensionMismatch("alm_packed length"))
    _validate_local_cost(cost, :SH_to_lat_cplx)
    _validate_local_nphi(nphi, :SH_to_lat_cplx)
    (0 <= ltr <= cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    alm_int = _internal_coefficients(alm_packed, cfg)
    CT = eltype(alm_int)
    RT = typeof(real(zero(CT)))
    PT = _local_basis_type(RT, cost)
    _, ACT = _local_accumulator_types(CT, PT)
    x = convert(PT, cost)
    P = Vector{PT}(undef, lmax + 1)
    vals = Vector{ACT}(undef, nphi)
    fill!(vals, zero(ACT))
    # m=0
    Plm_norm_row!(P, x, lmax, 0)
    g0 = zero(ACT)
    @inbounds for l in 0:min(ltr, lmax)
        idx = LM_cplx_index(lmax, mmax, l, 0) + 1
        a = alm_int[idx]
        g0 += P[l+1] * a
    end

    @inbounds for j in 1:nphi
        vals[j] += g0
    end

    # m ≠ 0
    for m in 1:mmax
        Plm_norm_row!(P, x, lmax, m)
        gm = zero(ACT); gn = zero(ACT)
        @inbounds for l in m:min(ltr, lmax)
            Ylm = P[l+1]
            # positive m
            ap = alm_int[LM_cplx_index(lmax, mmax, l, m) + 1]
            # negative m
            an = alm_int[LM_cplx_index(lmax, mmax, l, -m) + 1]
            gm += Ylm * ap
            gn += Ylm * an
        end
        for j in 0:(nphi-1)
            phase = cis(PT(2π * m * j / nphi))
            vals[j+1] += gm * phase + gn * conj(phase)
        end
    end
    sφ = _evaluator_phi_scale(cfg)   # 1 under :dft; 1/2π under :quad
    sφ == 1 || (vals .*= sφ)
    return vals
end

"""
    SHqst_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
        -> vr, vt, vp

Evaluate 3D field at a single point using packed real spectra.
"""
function SHqst_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    _validate_local_storage(:SHqst_to_point, Qlm, Slm, Tlm)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length"))
    length(Slm) == cfg.nlm || throw(DimensionMismatch("Slm length"))
    length(Tlm) == cfg.nlm || throw(DimensionMismatch("Tlm length"))
    _validate_local_coordinates(cost, phi, :SHqst_to_point)
    Qlm_int = _internal_coefficients(Qlm, cfg)
    Slm_int = _internal_coefficients(Slm, cfg)
    Tlm_int = _internal_coefficients(Tlm, cfg)
    CT = promote_type(eltype(Qlm_int), eltype(Slm_int), eltype(Tlm_int))
    RT = typeof(real(zero(CT)))
    PT = _local_basis_type(RT, cost, phi)
    _, ACT = _local_accumulator_types(CT, PT)
    x = convert(PT, cost)
    phiv = convert(PT, phi)
    lmax = cfg.lmax; mmax = cfg.mmax
    P = Vector{PT}(undef, lmax + 1)
    dPdtheta = Vector{PT}(undef, lmax + 1)
    P_over_sinth = Vector{PT}(undef, lmax + 1)
    Pbuf = Vector{PT}(undef, lmax + 2)  # scratch for the dθ recurrence
    imagunit = complex(zero(PT), one(PT))
    vr = zero(ACT)
    vt = zero(ACT)
    vp = zero(ACT)

    # m=0 (no 1/sinθ terms)
    Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, 0)
    for l in 0:lmax
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        aQ = Qlm_int[lm]; aS = Slm_int[lm]; aT = Tlm_int[lm]
        Y = P[l+1]
        dθY = dPdtheta[l+1]
        vr += Y   * aQ
        vt += dθY * aS
        # Vφ = (im/sinθ)*Y*S + dθY*T, for m=0 the first term is zero
        vp += dθY * aT
    end

    # m>0 (need pole-safe 1/sinθ handling)
    for m in 1:mmax
        (m % cfg.mres == 0) || continue
        # Single call computes P̄, dP̄/dθ, and P̄/sinθ (avoids redundant Plm_norm_row!)
        Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, m, Pbuf)
        gvr = zero(ACT)
        gvt = zero(ACT)
        gvp = zero(ACT)
        for l in m:lmax
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            aQ = Qlm_int[lm]; aS = Slm_int[lm]; aT = Tlm_int[lm]
            Y = P[l+1]
            dθY = dPdtheta[l+1]
            Y_over_sθ = P_over_sinth[l+1]
            gvr += Y   * aQ
            # Vθ = dθY*S - (im/sinθ)*Y*T
            gvt += dθY * aS - imagunit * m * Y_over_sθ * aT
            # Vφ = (im/sinθ)*Y*S + dθY*T
            gvp += imagunit * m * Y_over_sθ * aS + dθY * aT
        end
        ph = cis(convert(PT, m) * phiv)
        vr += 2 * real(gvr * ph)
        vt += 2 * real(gvt * ph)
        vp += 2 * real(gvp * ph)
    end
    if cfg.robert_form
        sinth = sqrt(max(zero(PT), one(PT) - x*x))
        vt *= sinth
        vp *= sinth
    end
    sφ = _evaluator_phi_scale(cfg)   # 1 under :dft; 1/2π under :quad
    return real(vr) * sφ, real(vt) * sφ, real(vp) * sφ
end

"""
    SH_to_grad_point(cfg::SHTConfig, DrSlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, cost::Real, phi::Real)
        -> vr, vt, vp

Evaluate the gradient of a scalar field at a point. `DrSlm` supplies the
radial derivative spectrum and `Slm` supplies the tangential gradient.
"""
function SH_to_grad_point(cfg::SHTConfig, DrSlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    _validate_local_storage(:SH_to_grad_point, DrSlm, Slm)
    length(DrSlm) == cfg.nlm || throw(DimensionMismatch("DrSlm length"))
    length(Slm) == cfg.nlm || throw(DimensionMismatch("Slm length"))
    _validate_local_coordinates(cost, phi, :SH_to_grad_point)
    DrSlm_int = _internal_coefficients(DrSlm, cfg)
    Slm_int = _internal_coefficients(Slm, cfg)
    CT = promote_type(eltype(DrSlm_int), eltype(Slm_int))
    RT = typeof(real(zero(CT)))
    PT = _local_basis_type(RT, cost, phi)
    _, ACT = _local_accumulator_types(CT, PT)
    x = convert(PT, cost)
    phiv = convert(PT, phi)
    lmax = cfg.lmax; mmax = cfg.mmax
    P = Vector{PT}(undef, lmax + 1)
    dPdtheta = Vector{PT}(undef, lmax + 1)
    P_over_sinth = Vector{PT}(undef, lmax + 1)
    Pbuf = Vector{PT}(undef, lmax + 2)  # scratch for the dθ recurrence
    imagunit = complex(zero(PT), one(PT))
    vr = zero(ACT)
    vt = zero(ACT)
    vp = zero(ACT)


    # m=0: Vθ = dθY*S; the (im/sinθ)*Y*S term in Vφ vanishes
    Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, 0)
    for l in 0:lmax
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        aDr = DrSlm_int[lm]
        aS = Slm_int[lm]
        vr += P[l+1] * aDr
        vt += dPdtheta[l+1] * aS
    end

    # m>0 (pole-safe 1/sinθ handling)
    for m in 1:mmax
        (m % cfg.mres == 0) || continue
        Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, m, Pbuf)
        gvt = zero(ACT)
        gvp = zero(ACT)
        gvr = zero(ACT)
        for l in m:lmax
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            aDr = DrSlm_int[lm]
            aS = Slm_int[lm]
            gvr += P[l+1] * aDr
            # Vθ = dθY*S, Vφ = (im/sinθ)*Y*S (T ≡ 0 for a scalar gradient)
            gvt += dPdtheta[l+1] * aS
            gvp += imagunit * m * P_over_sinth[l+1] * aS
        end
        ph = cis(convert(PT, m) * phiv)
        vr += 2 * real(gvr * ph)
        vt += 2 * real(gvt * ph)
        vp += 2 * real(gvp * ph)
    end
    if cfg.robert_form
        sinth = sqrt(max(zero(PT), one(PT) - x*x))
        vt *= sinth
        vp *= sinth
    end
    sφ = _evaluator_phi_scale(cfg)   # 1 under :dft; 1/2π under :quad
    return real(vr) * sφ, real(vt) * sφ, real(vp) * sφ
end

"""
    SHqst_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real;
                 nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vr, Vt, Vp

Evaluate 3D field along latitude (cosθ = cost) at `nphi` longitudes from packed real spectra.
Inputs `Qlm, Slm, Tlm` are all packed (LM order) vectors for each component.
"""
function SHqst_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real;
                      nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    _validate_local_storage(:SHqst_to_lat, Qlm, Slm, Tlm)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length"))
    length(Slm) == cfg.nlm || throw(DimensionMismatch("Slm length"))
    length(Tlm) == cfg.nlm || throw(DimensionMismatch("Tlm length"))
    _validate_local_cost(cost, :SHqst_to_lat)
    _validate_local_nphi(nphi, :SHqst_to_lat)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    (0 ≤ mtr ≤ cfg.mmax) || throw(ArgumentError("mtr must be within [0, mmax]"))
    Qlm_int = _internal_coefficients(Qlm, cfg)
    Slm_int = _internal_coefficients(Slm, cfg)
    Tlm_int = _internal_coefficients(Tlm, cfg)
    CT = promote_type(eltype(Qlm_int), eltype(Slm_int), eltype(Tlm_int))
    RT = typeof(real(zero(CT)))
    PT = _local_basis_type(RT, cost)
    AT, ACT = _local_accumulator_types(CT, PT)
    x = convert(PT, cost)
    lmax = cfg.lmax
    # Accumulator/output eltype follows the inputs so AD types propagate.
    P = Vector{PT}(undef, lmax + 1)
    dPdtheta = Vector{PT}(undef, lmax + 1)
    P_over_sinth = Vector{PT}(undef, lmax + 1)
    Pbuf = Vector{PT}(undef, lmax + 2)  # scratch for the dθ recurrence
    imagunit = complex(zero(PT), one(PT))
    Vr = Vector{AT}(undef, nphi)
    Vt = Vector{AT}(undef, nphi)
    Vp = Vector{AT}(undef, nphi)
    fill!(Vr, zero(AT)); fill!(Vt, zero(AT)); fill!(Vp, zero(AT))


    # m=0 (no 1/sinθ terms)
    Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, 0)
    g0 = zero(ACT)
    gθ0 = zero(ACT)
    gφ0 = zero(ACT)

    @inbounds for l in 0:ltr
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        aQ = Qlm_int[lm]; aS = Slm_int[lm]; aT = Tlm_int[lm]
        Y = P[l+1]
        dθY = dPdtheta[l+1]
        g0  += Y * aQ
        gθ0 += dθY * aS
        # Vφ = (im/sinθ)*Y*S + dθY*T, for m=0 the first term is zero
        gφ0 += dθY * aT
    end
    @inbounds for j in 1:nphi
        Vr[j] += real(g0); Vt[j] += real(gθ0); Vp[j] += real(gφ0)
    end

    # m>0 (need pole-safe 1/sinθ handling)
    for m in 1:mtr
        (m % cfg.mres == 0) || continue
        # Single call computes P̄, dP̄/dθ, and P̄/sinθ (avoids redundant Plm_norm_row!)
        Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, m, Pbuf)
        g  = zero(ACT)
        gθ = zero(ACT)
        gφ = zero(ACT)

        @inbounds for l in m:min(ltr, lmax)
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            aQ = Qlm_int[lm]; aS = Slm_int[lm]; aT = Tlm_int[lm]
            Y = P[l+1]
            dθY = dPdtheta[l+1]
            Y_over_sθ = P_over_sinth[l+1]
            g  += Y   * aQ
            # Vθ = dθY*S - (im/sinθ)*Y*T
            gθ += dθY * aS - imagunit * m * Y_over_sθ * aT
            # Vφ = (im/sinθ)*Y*S + dθY*T
            gφ += imagunit * m * Y_over_sθ * aS + dθY * aT
        end
        for j in 0:(nphi-1)
            phase = cis(PT(2π * m * j / nphi))
            Vr[j+1] += 2 * real(g * phase)
            Vt[j+1] += 2 * real(gθ * phase)
            Vp[j+1] += 2 * real(gφ * phase)
        end
    end
    if cfg.robert_form
        sinth = sqrt(max(zero(PT), one(PT) - x*x))
        Vt .*= sinth
        Vp .*= sinth
    end
    sφ = _evaluator_phi_scale(cfg)   # 1 under :dft; 1/2π under :quad
    if sφ != 1
        Vr .*= sφ; Vt .*= sφ; Vp .*= sφ
    end
    return Vr, Vt, Vp
end


function SH_to_lat(::CPU, cfg::SHTConfig, Qlm::AbstractVector{<:Complex},
                   cost::Real; kwargs...)
    _require_cpu_storage(:SH_to_lat, Qlm)
    return SH_to_lat(cfg, Qlm, cost; kwargs...)
end

function SH_to_lat_cplx(::CPU, cfg::SHTConfig,
                        alm::AbstractVector{<:Complex}, cost::Real; kwargs...)
    _require_cpu_storage(:SH_to_lat_cplx, alm)
    return SH_to_lat_cplx(cfg, alm, cost; kwargs...)
end

function SHqst_to_point(::CPU, cfg::SHTConfig,
                        Qlm::AbstractVector{<:Complex},
                        Slm::AbstractVector{<:Complex},
                        Tlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    _require_cpu_storage(:SHqst_to_point, Qlm)
    _require_cpu_storage(:SHqst_to_point, Slm)
    _require_cpu_storage(:SHqst_to_point, Tlm)
    return SHqst_to_point(cfg, Qlm, Slm, Tlm, cost, phi)
end

function SHqst_to_lat(::CPU, cfg::SHTConfig,
                      Qlm::AbstractVector{<:Complex},
                      Slm::AbstractVector{<:Complex},
                      Tlm::AbstractVector{<:Complex}, cost::Real; kwargs...)
    _require_cpu_storage(:SHqst_to_lat, Qlm)
    _require_cpu_storage(:SHqst_to_lat, Slm)
    _require_cpu_storage(:SHqst_to_lat, Tlm)
    return SHqst_to_lat(cfg, Qlm, Slm, Tlm, cost; kwargs...)
end

function SH_to_grad_point(::CPU, cfg::SHTConfig,
                          DrSlm::AbstractVector{<:Complex},
                          Slm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    _require_cpu_storage(:SH_to_grad_point, DrSlm)
    _require_cpu_storage(:SH_to_grad_point, Slm)
    return SH_to_grad_point(cfg, DrSlm, Slm, cost, phi)
end


SH_to_lat(::GPU, ::SHTConfig, ::AbstractVector{<:Complex}, ::Real; kwargs...) =
    throw(ArgumentError("SH_to_lat(GPU(), ...) requires one supported GPU vendor's storage"))
SH_to_lat_cplx(::GPU, ::SHTConfig, ::AbstractVector{<:Complex}, ::Real; kwargs...) =
    throw(ArgumentError("SH_to_lat_cplx(GPU(), ...) requires one supported GPU vendor's storage"))
SHqst_to_point(::GPU, ::SHTConfig, ::AbstractVector{<:Complex},
               ::AbstractVector{<:Complex}, ::AbstractVector{<:Complex},
               ::Real, ::Real) =
    throw(ArgumentError("SHqst_to_point(GPU(), ...) requires one supported GPU vendor's storage"))
SHqst_to_lat(::GPU, ::SHTConfig, ::AbstractVector{<:Complex},
             ::AbstractVector{<:Complex}, ::AbstractVector{<:Complex},
             ::Real; kwargs...) =
    throw(ArgumentError("SHqst_to_lat(GPU(), ...) requires one supported GPU vendor's storage"))
SH_to_grad_point(::GPU, ::SHTConfig, ::AbstractVector{<:Complex},
                 ::AbstractVector{<:Complex}, ::Real, ::Real) =
    throw(ArgumentError("SH_to_grad_point(GPU(), ...) requires one supported GPU vendor's storage"))
