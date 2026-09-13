"""
Local/partial evaluations along latitude circles and at points.
"""

"""
    SH_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real;
              nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vector{Float64}

Evaluate a real field along a latitude (fixed cosθ = cost) at `nphi` equispaced longitudes.
Uses orthonormal harmonics and packed real coefficients `Qlm` (LM order).
"""
function SH_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    (0 ≤ mtr ≤ cfg.mmax) || throw(ArgumentError("mtr must be within [0, mmax]"))
    Qlm_int = _internal_coefficients(Qlm, cfg)
    x = float(cost)
    lmax = cfg.lmax
    # Accumulator/output eltype follows the input so AD types (e.g.
    # ForwardDiff.Dual) propagate; defaults to Float64 for ComplexF64 input.
    CT = promote_type(eltype(Qlm_int), ComplexF64)
    RT = real(CT)
    P = Vector{Float64}(undef, lmax + 1)
    vals = Vector{RT}(undef, nphi)
    fill!(vals, zero(RT))


    # m=0 contribution
    Plm_norm_row!(P, x, lmax, 0)
    g0 = zero(CT)
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
        gm = zero(CT)
        @inbounds for l in m:min(ltr, lmax)
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            a = Qlm_int[lm]
            gm += P[l+1] * a
        end
        
        for j in 0:(nphi-1)
            vals[j+1] += 2 * real(gm * cis(2π * m * j / nphi))
        end
    end
    return vals
end

"""
    SH_to_lat_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax) -> Vector{ComplexF64}

Evaluate a complex field along a latitude using packed LM_cplx coefficients.
"""
function SH_to_lat_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}, cost::Real; nphi::Int=cfg.nlon, ltr::Int=cfg.lmax)
    lmax, mmax = cfg.lmax, cfg.mmax
    # The packing/indexing below assumes mres==1 (dense m). Fail loudly rather
    # than silently returning wrong values for a strided (mres>1) configuration.
    cfg.mres == 1 || throw(ArgumentError("SH_to_lat_cplx supports mres==1 only; got mres=$(cfg.mres)"))
    length(alm_packed) == nlm_cplx_calc(lmax, mmax, 1) || throw(DimensionMismatch("alm_packed length"))
    alm_int = _internal_coefficients(alm_packed, cfg)
    x = float(cost)
    CT = promote_type(eltype(alm_int), ComplexF64)
    P = Vector{Float64}(undef, lmax + 1)
    vals = Vector{CT}(undef, nphi)
    fill!(vals, zero(CT))
    # m=0
    Plm_norm_row!(P, x, lmax, 0)
    g0 = zero(CT)
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
        gm = zero(CT); gn = zero(CT)
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
            phase = cis(2π * m * j / nphi)
            vals[j+1] += gm * phase + gn * conj(phase)
        end
    end
    return vals
end

"""
    SHqst_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
        -> vr::Float64, vt::Float64, vp::Float64

Evaluate 3D field at a single point using packed real spectra.
Tangential components use Robert-form scaling when configured, as in grid synthesis.
"""
function SHqst_to_point(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length"))
    length(Slm) == cfg.nlm || throw(DimensionMismatch("Slm length"))
    length(Tlm) == cfg.nlm || throw(DimensionMismatch("Tlm length"))
    Qlm_int = _internal_coefficients(Qlm, cfg)
    Slm_int = _internal_coefficients(Slm, cfg)
    Tlm_int = _internal_coefficients(Tlm, cfg)
    x = float(cost)
    lmax = cfg.lmax; mmax = cfg.mmax
    P = Vector{Float64}(undef, lmax + 1)
    dPdtheta = Vector{Float64}(undef, lmax + 1)
    P_over_sinth = Vector{Float64}(undef, lmax + 1)
    Pbuf = Vector{Float64}(undef, lmax + 2)  # scratch for the dθ recurrence
    sθ = sqrt(max(0.0, 1 - x*x))
    CT = promote_type(eltype(Qlm_int), eltype(Slm_int), eltype(Tlm_int))
    vr = zero(CT)
    vt = zero(CT)
    vp = zero(CT)

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
        gvr = zero(CT)
        gvt = zero(CT)
        gvp = zero(CT)
        for l in m:lmax
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            aQ = Qlm_int[lm]; aS = Slm_int[lm]; aT = Tlm_int[lm]
            Y = P[l+1]
            dθY = dPdtheta[l+1]
            Y_over_sθ = P_over_sinth[l+1]
            gvr += Y   * aQ
            # Vθ = dθY*S - (im/sinθ)*Y*T
            gvt += dθY * aS - 1.0im * m * Y_over_sθ * aT
            # Vφ = (im/sinθ)*Y*S + dθY*T
            gvp += 1.0im * m * Y_over_sθ * aS + dθY * aT
        end
        ph = cis(m * phi)
        vr += 2 * real(gvr * ph)
        vt += 2 * real(gvt * ph)
        vp += 2 * real(gvp * ph)
    end
    if cfg.robert_form
        vt *= sθ
        vp *= sθ
    end
    return real(vr), real(vt), real(vp)
end

"""
    SH_to_grad_point(cfg::SHTConfig, DrSlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, cost::Real, phi::Real)
        -> vr::Float64, vt::Float64, vp::Float64

Evaluate gradient of a scalar field at a point. Vr is returned as 0.0.
`DrSlm` is ignored for this pure-Julia core.
Tangential components use Robert-form scaling when configured.
"""
function SH_to_grad_point(cfg::SHTConfig, ::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    length(Slm) == cfg.nlm || throw(DimensionMismatch("Slm length"))
    Slm_int = _internal_coefficients(Slm, cfg)
    x = float(cost)
    lmax = cfg.lmax; mmax = cfg.mmax
    P = Vector{Float64}(undef, lmax + 1)
    dPdtheta = Vector{Float64}(undef, lmax + 1)
    P_over_sinth = Vector{Float64}(undef, lmax + 1)
    Pbuf = Vector{Float64}(undef, lmax + 2)  # scratch for the dθ recurrence
    CT = promote_type(eltype(Slm_int), ComplexF64)
    vt = zero(CT)
    vp = zero(CT)


    # m=0: Vθ = dθY*S; the (im/sinθ)*Y*S term in Vφ vanishes
    Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, 0)
    for l in 0:lmax
        lm = LM_index(lmax, cfg.mres, l, 0) + 1
        aS = Slm_int[lm]
        vt += dPdtheta[l+1] * aS
    end

    # m>0 (pole-safe 1/sinθ handling)
    for m in 1:mmax
        (m % cfg.mres == 0) || continue
        Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, m, Pbuf)
        gvt = zero(CT)
        gvp = zero(CT)
        for l in m:lmax
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            aS = Slm_int[lm]
            # Vθ = dθY*S, Vφ = (im/sinθ)*Y*S (T ≡ 0 for a scalar gradient)
            gvt += dPdtheta[l+1] * aS
            gvp += 1.0im * m * P_over_sinth[l+1] * aS
        end
        ph = cis(m * phi)
        vt += 2 * real(gvt * ph)
        vp += 2 * real(gvp * ph)
    end
    if cfg.robert_form
        sθ = sqrt(max(0.0, 1 - x*x))
        vt *= sθ
        vp *= sθ
    end
    return zero(real(CT)), real(vt), real(vp)
end

"""
    SHqst_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real;
                 nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax) -> Vr, Vt, Vp

Evaluate 3D field along latitude (cosθ = cost) at `nphi` longitudes from packed real spectra.
Inputs `Qlm, Slm, Tlm` are all packed (LM order) vectors for each component.
Tangential components use Robert-form scaling when configured, as in grid synthesis.
"""
function SHqst_to_lat(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Slm::AbstractVector{<:Complex}, Tlm::AbstractVector{<:Complex}, cost::Real;
                      nphi::Int=cfg.nlon, ltr::Int=cfg.lmax, mtr::Int=cfg.mmax)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length"))
    length(Slm) == cfg.nlm || throw(DimensionMismatch("Slm length"))
    length(Tlm) == cfg.nlm || throw(DimensionMismatch("Tlm length"))
    Qlm_int = _internal_coefficients(Qlm, cfg)
    Slm_int = _internal_coefficients(Slm, cfg)
    Tlm_int = _internal_coefficients(Tlm, cfg)
    (0 ≤ ltr ≤ cfg.lmax) || throw(ArgumentError("ltr must be within [0, lmax]"))
    (0 ≤ mtr ≤ cfg.mmax) || throw(ArgumentError("mtr must be within [0, mmax]"))
    x = float(cost)
    lmax = cfg.lmax
    # Accumulator/output eltype follows the inputs so AD types propagate.
    CT = promote_type(eltype(Qlm_int), eltype(Slm_int), eltype(Tlm_int), ComplexF64)
    RT = real(CT)
    P = Vector{Float64}(undef, lmax + 1)
    dPdtheta = Vector{Float64}(undef, lmax + 1)
    P_over_sinth = Vector{Float64}(undef, lmax + 1)
    Pbuf = Vector{Float64}(undef, lmax + 2)  # scratch for the dθ recurrence
    Vr = Vector{RT}(undef, nphi)
    Vt = Vector{RT}(undef, nphi)
    Vp = Vector{RT}(undef, nphi)
    fill!(Vr, zero(RT)); fill!(Vt, zero(RT)); fill!(Vp, zero(RT))


    # m=0 (no 1/sinθ terms)
    Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, 0)
    g0 = zero(CT)
    gθ0 = zero(CT)
    gφ0 = zero(CT)

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
        g  = zero(CT)
        gθ = zero(CT)
        gφ = zero(CT)

        @inbounds for l in m:min(ltr, lmax)
            lm = LM_index(lmax, cfg.mres, l, m) + 1
            aQ = Qlm_int[lm]; aS = Slm_int[lm]; aT = Tlm_int[lm]
            Y = P[l+1]
            dθY = dPdtheta[l+1]
            Y_over_sθ = P_over_sinth[l+1]
            g  += Y   * aQ
            # Vθ = dθY*S - (im/sinθ)*Y*T
            gθ += dθY * aS - 1.0im * m * Y_over_sθ * aT
            # Vφ = (im/sinθ)*Y*S + dθY*T
            gφ += 1.0im * m * Y_over_sθ * aS + dθY * aT
        end
        for j in 0:(nphi-1)
            phase = cis(2π * m * j / nphi)
            Vr[j+1] += 2 * real(g * phase)
            Vt[j+1] += 2 * real(gθ * phase)
            Vp[j+1] += 2 * real(gφ * phase)
        end
    end
    if cfg.robert_form
        sθ = sqrt(max(0.0, 1 - x*x))
        Vt .*= sθ
        Vp .*= sθ
    end
    return Vr, Vt, Vp
end
