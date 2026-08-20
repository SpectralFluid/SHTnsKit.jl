#=
================================================================================
normalization.jl - Spherical Harmonic Normalization and Phase Conventions
================================================================================

This file handles conversions between different spherical harmonic normalization
and phase conventions. This is essential for interoperability with other
libraries and datasets that may use different conventions.

WHY DIFFERENT CONVENTIONS?
--------------------------
Different scientific fields evolved their own spherical harmonic conventions:

- Physics: orthonormal + Condon-Shortley phase (our internal convention)
- Geodesy: 4π normalization
- Geomagnetism: Schmidt semi-normalized

These differ by factors and phases, which must be carefully tracked.

INTERNAL CONVENTION
-------------------
SHTnsKit internally uses:
    Y_l^m(θ,φ) = N_lm P_l^m(cos θ) exp(imφ)

where N_lm = sqrt[(2l+1)/(4π) * (l-m)!/(l+m)!]

This makes the spherical harmonics ORTHONORMAL:
    ∫∫ Y_l^m (Y_{l'}^{m'})* dΩ = δ_{ll'} δ_{mm'}

CONDON-SHORTLEY PHASE
---------------------
The Condon-Shortley (CS) phase is a (-1)^m factor included in P_l^m.
- With CS phase: P_l^m contains (-1)^m factor (physics standard)
- Without CS phase: No (-1)^m factor (some math texts)

This affects the sign of coefficients for odd m values.

SUPPORTED CONVENTIONS
---------------------
:orthonormal - Our internal convention, standard physics
    N_lm = sqrt[(2l+1)/(4π) * (l-m)!/(l+m)!]

:fourpi - Geodesy/meteorology
    N_lm = sqrt[(2l+1) * (l-m)!/(l+m)!]
    (factor of sqrt(4π) larger than orthonormal)

:schmidt - Semi-normalized, geomagnetism (IGRF, WMM models)
    N_lm = sqrt[(l-m)!/(l+m)!] for m>0, 1 for m=0
    Common in geomagnetic field modeling

CONVERSION FORMULAS
-------------------
To convert from orthonormal to target:
    a_lm^target = a_lm^orthonormal / scale_factor

where scale_factor = norm_scale_from_orthonormal(l, m, target)

For CS phase conversion:
    a_lm^new = (-1)^m * a_lm^old  (when switching CS convention)

DEBUGGING
---------
```julia
# Check scale factors
@assert norm_scale_from_orthonormal(2, 1, :orthonormal) ≈ 1.0
@assert norm_scale_from_orthonormal(2, 1, :fourpi) ≈ sqrt(4π)

# Phase factor
@assert cs_phase_factor(3, true, false) ≈ -1.0  # odd m
@assert cs_phase_factor(2, true, false) ≈ 1.0   # even m
```

================================================================================
=#

"""
Spherical Harmonic Normalization and Phase Conversion Utilities

This module handles conversions between different normalization conventions and
phase definitions used in spherical harmonic analysis. Different fields use
different conventions, so we provide conversion utilities to maintain compatibility.

Internal Convention:
- SHTnsKit internally uses orthonormal spherical harmonics with Condon-Shortley phase
- This ensures numerical stability and follows physics conventions

External Conventions Supported:
- :orthonormal - Standard physics normalization: ∫ Y_l^m (Y_{l'}^{m'})* dΩ = δ_{ll'} δ_{mm'}
- :fourpi - Geodesy convention: Y_l^m scaled by sqrt(4π)
- :schmidt - Semi-normalized: common in geomagnetism and geodesy

Phase Conventions:
- Condon-Shortley phase: includes (-1)^m factor (standard in physics)
- No CS phase: omits the (-1)^m factor (used in some mathematics texts)
"""

"""
    norm_scale_from_orthonormal(l::Int, m::Int, to::Symbol) -> Float64

Calculate the scaling factor to convert from orthonormal to target normalization.

Returns k such that Y_target = k * Y_orthonormal, allowing conversion between
different spherical harmonic normalization conventions while preserving the
mathematical relationships.

The scale factor depends on the target convention:
- :orthonormal → k = 1 (no scaling)
- :fourpi → k = sqrt(4π) (geodesy convention)
- :schmidt → k = sqrt(4π/(2l+1)) (Schmidt semi-normalization)

The optional real-basis `sqrt(2)` factor is deliberately separate from Schmidt
normalization and is controlled by `cfg.real_norm`.
"""
function norm_scale_from_orthonormal(l::Int, m::Int, to::Symbol)
    if to === :orthonormal
        # No conversion needed
        return 1.0

    elseif to === :fourpi
        # Geodesy convention: multiply by sqrt(4π)
        # This removes the 1/sqrt(4π) factor from orthonormal normalization
        return sqrt(4π)

    elseif to === :schmidt
        # Schmidt semi-normalized spherical harmonics
        # Used extensively in geomagnetic field modeling (e.g., IGRF, WMM)
        return sqrt(4π / (2l + 1))

    else
        throw(ArgumentError("Unsupported normalization: $to"))
    end
end

"""
    coefficient_scale_to_canonical(cfg, l, m)

Return the multiplier `s` such that `a_canonical = s * a_configured` for the
same physical field. The canonical convention is orthonormal with the
Condon–Shortley phase. With `real_norm=true`, coefficients for `m>0` are stored
`sqrt(2)` larger, so their multiplier to canonical form is `1/sqrt(2)`.
"""
@inline function coefficient_scale_to_canonical(cfg, l::Int, m::Int)
    norm = norm_scale_from_orthonormal(l, m, cfg.norm)
    real_scale = cfg.real_norm && m > 0 ? inv(sqrt(2.0)) : 1.0
    phase = cs_phase_factor(m, true, cfg.cs_phase)
    return norm * real_scale * phase
end

@inline _uses_canonical_convention(cfg) =
    cfg.norm === :orthonormal && !cfg.real_norm && cfg.cs_phase

"""
    cs_phase_factor(m::Int, cs_from::Bool, cs_to::Bool) -> Float64

Calculate the phase factor for converting between Condon-Shortley conventions.

The Condon-Shortley phase is a (-1)^m factor included in some spherical harmonic
definitions. This function returns the scaling factor α such that:
Y_to = α * Y_from when switching phase conventions.

The conversion rule is:
- If cs_from = cs_to: α = 1 (no change needed)
- If switching: α = (-1)^m (the CS phase factor itself)

Note: This applies to the basis functions. For coefficients, the transformation
may need to be inverted depending on the context.
"""
function cs_phase_factor(m::Int, cs_from::Bool, cs_to::Bool)
    if cs_from == cs_to
        # No phase conversion needed
        return 1.0
    else
        # Apply Condon-Shortley phase toggle: (-1)^m
        # This handles switching between CS and non-CS conventions
        return (-1.0)^m
    end
end

"""
    _ensure_norm_scale_matrix!(cfg) -> Matrix{Float64}

Lazily build (or return cached) `(lmax+1, mmax+1)` matrix holding
`coefficient_scale_to_canonical(cfg, l, m)`
so `convert_alm_norm!` reduces to one elementwise multiply per coefficient.
"""
function _ensure_norm_scale_matrix!(cfg)
    M = cfg.norm_scale_matrix
    size_matches = size(M, 1) == cfg.lmax + 1 && size(M, 2) == cfg.mmax + 1
    cache_is_current = size_matches &&
        M[1, 1] == coefficient_scale_to_canonical(cfg, 0, 0) &&
        (cfg.lmax == 0 || M[2, 1] == coefficient_scale_to_canonical(cfg, 1, 0)) &&
        (cfg.mmax == 0 || M[2, 2] == coefficient_scale_to_canonical(cfg, 1, 1))
    if cache_is_current
        return M
    end
    lmax, mmax = cfg.lmax, cfg.mmax
    M = Matrix{Float64}(undef, lmax + 1, mmax + 1)
    fill!(M, 1.0)  # entries with l < m stay at 1 but are never consumed
    @inbounds for m in 0:mmax
        for l in m:lmax
            M[l+1, m+1] = coefficient_scale_to_canonical(cfg, l, m)
        end
    end
    cfg.norm_scale_matrix = M
    return M
end

"""
    convert_alm_norm!(dest, src, cfg; to_internal::Bool=false)

Convert coefficient matrix `src` between cfg's normalization/phase and internal
orthonormal+CS. If `to_internal=true`, maps from cfg to internal. Otherwise maps
from internal to cfg. Writes into `dest` which must match `src` size.
"""
function convert_alm_norm!(dest::AbstractMatrix, src::AbstractMatrix, cfg; to_internal::Bool=false)
    size(dest) == size(src) || throw(DimensionMismatch("dest/src dims mismatch"))
    lmax, mmax = cfg.lmax, cfg.mmax
    size(src) == (lmax + 1, mmax + 1) ||
        throw(DimensionMismatch("coefficient matrix must have size ($(lmax+1), $(mmax+1))"))
    M = _ensure_norm_scale_matrix!(cfg)
    z = zero(eltype(dest))
    # to_internal=true: alm_int = alm_cfg * M. to_internal=false: alm_cfg = alm_int / M.
    @inbounds for m in 0:mmax
        for l in 0:(m-1)
            dest[l+1, m+1] = z  # l<m entries are unused; zero them to avoid uninitialized garbage
        end
        for l in m:lmax
            s = M[l+1, m+1]
            dest[l+1, m+1] = to_internal ? s * src[l+1, m+1] : src[l+1, m+1] / s
        end
    end
    return dest
end

"""Convert packed coefficients between configured and canonical conventions."""
function convert_alm_norm!(dest::AbstractVector, src::AbstractVector, cfg; to_internal::Bool=false)
    length(dest) == length(src) || throw(DimensionMismatch("dest/src lengths mismatch"))
    M = _ensure_norm_scale_matrix!(cfg)
    if length(src) == cfg.nlm
        @inbounds for k in eachindex(src)
            s = M[cfg.li[k] + 1, cfg.mi[k] + 1]
            dest[k] = to_internal ? s * src[k] : src[k] / s
        end
    elseif cfg.mres == 1 && length(src) == nlm_cplx_calc(cfg.lmax, cfg.mmax, 1)
        @inbounds for l in 0:cfg.lmax, m in -min(l, cfg.mmax):min(l, cfg.mmax)
            k = LM_cplx_index(cfg.lmax, cfg.mmax, l, m) + 1
            s = M[l + 1, abs(m) + 1]
            dest[k] = to_internal ? s * src[k] : src[k] / s
        end
    else
        throw(DimensionMismatch("packed coefficient length does not match real or complex SHTns storage"))
    end
    return dest
end

"""Convert a `(l,m,batch)` coefficient array without changing its element type."""
function convert_alm_norm!(dest::AbstractArray{<:Any,3}, src::AbstractArray{<:Any,3}, cfg;
                           to_internal::Bool=false)
    size(dest) == size(src) || throw(DimensionMismatch("dest/src dims mismatch"))
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    size(src, 1) == expected[1] && size(src, 2) == expected[2] ||
        throw(DimensionMismatch("coefficient batch first dimensions must be $expected"))
    M = _ensure_norm_scale_matrix!(cfg)
    z = zero(eltype(dest))
    @inbounds for k in axes(src, 3), m in 0:cfg.mmax
        for l in 0:(m - 1)
            dest[l + 1, m + 1, k] = z
        end
        for l in m:cfg.lmax
            s = M[l + 1, m + 1]
            dest[l + 1, m + 1, k] = to_internal ?
                s * src[l + 1, m + 1, k] : src[l + 1, m + 1, k] / s
        end
    end
    return dest
end

@inline function _convert_mode_norm!(dest::AbstractVector, src::AbstractVector,
                                     cfg, m::Int, ltr::Int; to_internal::Bool=false)
    expected = ltr - m + 1
    length(dest) == length(src) == expected ||
        throw(DimensionMismatch("mode coefficients must have length $expected"))
    M = _ensure_norm_scale_matrix!(cfg)
    @inbounds for l in m:ltr
        s = M[l + 1, m + 1]
        k = l - m + 1
        dest[k] = to_internal ? s * src[k] : src[k] / s
    end
    return dest
end

@inline function _internal_coefficients(src, cfg)
    _uses_canonical_convention(cfg) && return src
    dest = similar(src)
    return convert_alm_norm!(dest, src, cfg; to_internal=true)
end

@inline function _externalize_coefficients!(dest, cfg)
    _uses_canonical_convention(cfg) && return dest
    return convert_alm_norm!(dest, dest, cfg; to_internal=false)
end

# Adjoint maps are transposes of the real diagonal convention maps.
@inline function _analysis_cotangent_to_canonical(src, cfg)
    _uses_canonical_convention(cfg) && return src
    dest = similar(src)
    return convert_alm_norm!(dest, src, cfg; to_internal=false)
end

@inline function _synthesis_cotangent_to_configured!(dest, cfg)
    _uses_canonical_convention(cfg) && return dest
    return convert_alm_norm!(dest, dest, cfg; to_internal=true)
end
