#=
================================================================================
complex_packed.jl - Complex Packed Layout (LM_cplx) Support
================================================================================

This file implements the packed coefficient storage for complex-valued fields
where both positive and negative m values are stored (no Hermitian symmetry).

WHEN IS THIS NEEDED?
--------------------
Real-valued fields have Hermitian symmetry: a_{l,-m} = (-1)^m conj(a_{l,m})
So we only need to store m ≥ 0 coefficients (the "real" packed layout).

Complex-valued fields have NO symmetry - we must store all m from -l to +l.
This doubles the storage but is necessary for:
- Complex scalar fields
- Intermediate calculations in rotations
- Some mathematical analyses

LAYOUT (LM_cplx)
----------------
Coefficients are packed in order:
    l=0: m=0                    (1 coefficient)
    l=1: m=-1, 0, +1            (3 coefficients)
    l=2: m=-2, -1, 0, +1, +2    (5 coefficients)
    ...

For l ≤ mmax: index = l(l+1) + m
For l > mmax: index = mmax(2l - mmax) + l + m

This matches the SHTns C library LM_cplx macro (for mres=1).

FUNCTIONS
---------
    LM_cplx_index(lmax, mmax, l, m)    : Get packed index for (l,m)
    LM_cplx(cfg, l, m)                 : Compatibility wrapper
    nlm_cplx_calc(lmax, mmax, mres)    : Count total complex coefficients

    synthesis_packed_cplx(cfg, alm_packed)   : Synthesis for complex field
    analysis_packed_cplx(cfg, z)            : Analysis for complex field
    synthesis_point_cplx(cfg, alm, cosθ, φ): Point evaluation for complex field

COMPARISON WITH REAL LAYOUT
---------------------------
For lmax=mmax=2:
    Real layout:    (0,0), (1,0), (2,0), (1,1), (2,1), (2,2)     = 6 coeffs
    Complex layout: (0,0), (1,-1),(1,0),(1,1), (2,-2),...,(2,2)  = 9 coeffs

================================================================================
=#

"""
Complex packed layout (LM_cplx) support and transforms in pure Julia.
Compatible with SHTns `LM_cplx` macro for `mres == 1`.
"""

"""
    LM_cplx_index(lmax::Int, mmax::Int, l::Int, m::Int) -> Int

Packed complex index for coefficient `(l,m)` where `-min(l,mmax) ≤ m ≤ min(l,mmax)`.
Matches SHTns `LM_cplx` macro (assumes `mres == 1`).
"""
function LM_cplx_index(lmax::Int, mmax::Int, l::Int, m::Int)
    (0 ≤ l ≤ lmax) || throw(ArgumentError("l out of range"))
    mm = min(l, mmax)
    (-mm ≤ m ≤ mm) || throw(ArgumentError("m out of range for given l and mmax"))
    if l ≤ mmax
        return l*(l + 1) + m
    else
        return mmax*(2*l - mmax) + l + m
    end
end

"""
    LM_cplx(cfg::SHTConfig, l::Integer, m::Integer) -> Int

Compatibility helper that mirrors the C macro `LM_cplx` for `mres == 1`,
returning the zero-based packed complex index for coefficient `(l,m)`.
"""
function LM_cplx(cfg::SHTConfig, l::Integer, m::Integer)
    cfg.mres == 1 || throw(ArgumentError("LM_cplx is only defined for mres == 1"))
    return LM_cplx_index(cfg.lmax, cfg.mmax, Int(l), Int(m))
end

"""
    synthesis_packed_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex}) -> Matrix{ComplexF64}

Synthesize complex spatial field from packed complex coefficients (LM_cplx order).
Returns an `nlat × nlon` complex array.
"""
function synthesis_packed_cplx(cfg::SHTConfig, alm_packed::AbstractVector{<:Complex})
    mres = cfg.mres
    mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    expected = nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(alm_packed) == expected || throw(DimensionMismatch("alm length $(length(alm_packed)) != expected $(expected)"))
    alm_int = _internal_coefficients(alm_packed, cfg)

    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm_int)
    Fφ = Matrix{CT}(undef, nlat, nlon)
    fill!(Fφ, zero(CT))

    lmax, mmax = cfg.lmax, cfg.mmax
    P = Vector{Float64}(undef, lmax + 1)
    # Scale continuous Fourier coefficients to DFT bins for ifft
    inv_scaleφ = phi_inv_scale(cfg)

    # Complex packed layout stores negative and positive m explicitly, so this
    # loop writes each Fourier bin directly instead of relying on Hermitian
    # symmetry as the real-field packed layout does.
    xv = cfg.x  # hoist field read out of the m/l loops (cfg is mutable, so not auto-hoisted)
    # Loop over |m| once: P̄_l^{|m|}, the CS-phase factor ((-1)^m == (-1)^{-m}) and
    # the norm scale all depend only on |m|, so the +m and -m Fourier bins share
    # the same Legendre row — compute it once instead of twice.
    for am in 0:mmax
        jp = am + 1                     # DFT bin for +am
        jn = nlon - am + 1              # DFT bin for -am ((j-1) ≡ -am mod nlon)
        for i in 1:nlat
            Plm_norm_row!(P, xv[i], lmax, am)
            gp = zero(CT); gn = zero(CT)
            @inbounds for l in am:lmax
                Pl = P[l+1]
                ap = alm_int[LM_cplx_index(lmax, mmax, l, am) + 1]
                gp += Pl * ap
                if am > 0
                    an = alm_int[LM_cplx_index(lmax, mmax, l, -am) + 1]
                    gn += Pl * an
                end
            end
            Fφ[i, jp] = inv_scaleφ * gp
            am > 0 && (Fφ[i, jn] = inv_scaleφ * gn)
        end
    end

    z = ifft_phi(Fφ)
    return z
end

"""
    analysis_packed_cplx(cfg::SHTConfig, z::AbstractMatrix{<:Complex}) -> Vector{ComplexF64}

Analyze complex spatial field into packed complex coefficients (LM_cplx order).
Input `z` must be `nlat × nlon` complex.
"""
function analysis_packed_cplx(cfg::SHTConfig, z::AbstractMatrix{<:Complex})
    size(z,1) == cfg.nlat || throw(DimensionMismatch("z first dim must be nlat"))
    size(z,2) == cfg.nlon || throw(DimensionMismatch("z second dim must be nlon"))
    mres = cfg.mres
    mres == 1 || throw(ArgumentError("LM_cplx layout only defined for mres==1"))
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(z)
    alm = Vector{CT}(undef, nlm_cplx_calc(lmax, mmax, 1))
    fill!(alm, zero(CT))

    # FFT along φ — one buffer + cached FFTW plan (the bare out-of-place
    # fft_phi re-plans each call). eltype preserved for the AD/DFT fallback.
    Fφ = fft_phi!(Matrix{complex(float(eltype(z)))}(undef, size(z)...), z)
    P = Vector{Float64}(undef, lmax + 1)
    scaleφ = cfg.cphi

    xv = cfg.x; wv = cfg.w  # hoist field reads out of the m/l loops (cfg is mutable, so not auto-hoisted)
    # Read both signs of m from the FFT output and store them in LM_cplx order.
    # Negative modes live at DFT column nlon+m+1. Loop over |m| once — P̄_l^{|m|},
    # the CS-phase factor and the norm scale depend only on |m| — so the +m and
    # -m columns share one Legendre row instead of recomputing it twice.
    for am in 0:mmax
        colp = am + 1
        coln = cfg.nlon - am + 1
        for i in 1:cfg.nlat
            Plm_norm_row!(P, xv[i], lmax, am)
            wi = wv[i]
            Fip = Fφ[i, colp]
            Fin = am > 0 ? Fφ[i, coln] : zero(eltype(Fφ))
            @inbounds for l in am:lmax
                base = (wi * P[l+1]) * scaleφ
                ap = base * Fip
                alm[LM_cplx_index(lmax, mmax, l, am) + 1] += ap
                if am > 0
                    an = base * Fin
                    alm[LM_cplx_index(lmax, mmax, l, -am) + 1] += an
                end
            end
        end
    end
    return _externalize_coefficients!(alm, cfg)
end

"""
    synthesis_point_cplx(cfg::SHTConfig, alm::AbstractVector{<:Complex}, cost::Real, phi::Real) -> ComplexF64

Evaluate a complex field represented by packed `alm` at a single point.
"""
function synthesis_point_cplx(cfg::SHTConfig, alm::AbstractVector{<:Complex}, cost::Real, phi::Real)
    expected = nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
    length(alm) == expected || throw(DimensionMismatch("alm length mismatch"))
    alm_int = _internal_coefficients(alm, cfg)
    x = float(cost)
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = eltype(alm_int)
    P = Vector{Float64}(undef, lmax + 1)
    acc = zero(CT)
    # Loop over |m| once (P̄_l^{|m|}, CS-phase and norm scale depend only on |m|)
    # and accumulate the +m and -m contributions from the shared Legendre row.
    for am in 0:mmax
        Plm_norm_row!(P, x, lmax, am)
        gp = zero(CT); gn = zero(CT)
        @inbounds for l in am:lmax
            Pl = P[l+1]
            ap = alm_int[LM_cplx_index(lmax, mmax, l, am) + 1]
            gp += Pl * ap
            if am > 0
                an = alm_int[LM_cplx_index(lmax, mmax, l, -am) + 1]
                gn += Pl * an
            end
        end
        acc += gp * cis(am * phi)
        am > 0 && (acc += gn * cis(-am * phi))
    end
    return acc
end
