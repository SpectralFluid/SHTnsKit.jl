#=
================================================================================
transforms.jl - Specialized Spherical Harmonic Transform Functions
================================================================================

This file provides specialized transform variants for cases where full 2D
transforms are unnecessary or inefficient.

WHEN TO USE SPECIALIZED TRANSFORMS
----------------------------------
- Axisymmetric fields (m=0 only): Use *_axisym functions
- Single azimuthal mode: Use *_ml functions
- Point evaluation: Use synthesis_point
- Degree truncation: Use *_l variants
- Packed coefficient layout: Use synthesis_packed / analysis_packed

FUNCTION CATEGORIES
-------------------
1. Packed Layout Transforms:
   analysis_packed(cfg, Vr)      : Grid → packed coefficients (1D vector)
   synthesis_packed(cfg, Qlm)     : Packed coefficients → flattened grid

2. Axisymmetric (m=0) Transforms:
   analysis_axisym(cfg, Vr)     : Latitude values → l-coefficients
   synthesis_axisym(cfg, Qlm)    : l-coefficients → latitude values
   *_l_axisym variants            : Degree-limited versions

3. Mode-Limited (single m) Transforms:
   analysis_packed_ml(cfg, m, Vr_m, ltr)  : Single-mode analysis
   synthesis_packed_ml(cfg, m, Ql, ltr)    : Single-mode synthesis

4. Degree-Limited Transforms:
   analysis_packed_l(cfg, Vr, ltr)  : Analysis with l ≤ ltr
   synthesis_packed_l(cfg, Qlm, ltr) : Synthesis with l ≤ ltr

5. Point Evaluation:
   synthesis_point(cfg, Qlm, cosθ, φ)  : Evaluate at single point

PERFORMANCE BENEFITS
--------------------
- Axisymmetric: O(lmax × nlat) vs O(lmax × nlat × nlon) for full transform
- Mode-limited: Process only needed azimuthal modes
- Point evaluation: No grid storage needed
- Degree truncation: Skip high-degree computations

USAGE EXAMPLES
--------------
```julia
cfg = create_gauss_config(32, 64)

# Axisymmetric field (zonal average)
f_zonal = mean(f, dims=2)[:, 1]  # Average over longitude
Ql = analysis_axisym(cfg, f_zonal)

# Point evaluation (avoid full synthesis for single point)
val = synthesis_point(cfg, Qlm, cos(θ), φ)

# Degree-limited synthesis (e.g., for low-pass filtering)
f_smooth = synthesis_packed_l(cfg, Qlm, 10)  # Only l ≤ 10
```

================================================================================
=#

"""
Specialized Spherical Harmonic Transforms

This module provides specialized transform functions for specific use cases:
- Vector transforms for individual l,m modes
- Point evaluation without full grid computation
- Degree-limited and mode-limited transforms

These functions are optimized for cases where only partial spectral information
is needed, avoiding the computational overhead of full 2D transforms.
"""

"""
    analysis_axisym(cfg, Vr) -> Vector{ComplexF64}

Axisymmetric (m=0) transform from Gauss latitudes to degree-only coefficients.
Input `Vr` should contain values at Gauss latitudes for a specific longitude mode.
Returns coefficients Q_l for l = 0..lmax.
"""
function analysis_axisym(cfg::SHTConfig, Vr::AbstractVector{<:Real})
    nlat, lmax = cfg.nlat, cfg.lmax
    length(Vr) == nlat || throw(DimensionMismatch("Vr length must be nlat=$(nlat)"))
    
    CT = complex(float(eltype(Vr)))  # AD/Float32-safe output eltype
    Ql = Vector{CT}(undef, lmax + 1)
    fill!(Ql, zero(CT))

    P = Vector{Float64}(undef, lmax + 1)
    xv = cfg.x; wv = cfg.w  # hoist field reads out of the i/l loops (cfg is mutable, so not auto-hoisted)

    for i in 1:nlat
        x = xv[i]
        Plm_norm_row!(P, x, lmax, 0)  # m=0 case (axisymmetric); P̄ already orthonormal-normalized

        weighted_Vr = Vr[i] * wv[i]
        @inbounds for l in 0:lmax
            Ql[l+1] += weighted_Vr * P[l+1]
        end
    end

    # φ quadrature factor. The full `analysis` applies `cfg.cphi` to an FFT output
    # whose bin 0 already carries an implicit `nlon` from the DFT sum; an
    # axisymmetric profile is constant in φ, so the whole `cphi*nlon = 2π` must be
    # applied explicitly here. Without it this is NOT the inverse of
    # `synthesis_axisym` (which matches `synthesis` exactly) and disagrees with
    # the m=0 column of `analysis` by 1/2π.
    scaleφ = cfg.cphi * cfg.nlon
    @inbounds for l in 0:lmax
        Ql[l+1] *= scaleφ
    end
    return _convert_mode_norm!(Ql, Ql, cfg, 0, lmax; to_internal=false)
end

"""
    analysis_packed(cfg, Vr_flat::AbstractVector{<:Real}) -> Vector{ComplexF64}

Packed scalar analysis from flattened grid values (length nlat*nlon) to Qlm (LM order).
"""
function analysis_packed(cfg::SHTConfig, Vr::AbstractVector{<:Real})
    length(Vr) == cfg.nspat || throw(DimensionMismatch("Vr must have length $(cfg.nspat)"))
    f = reshape(Vr, cfg.nlat, cfg.nlon)
    alm_mat = analysis(cfg, f)
    # Dense matrix output is converted back to SHTns-compatible packed LM
    # order, skipping unsupported m values when mres > 1.
    return pack_lm(cfg, alm_mat)
end

"""
    synthesis_packed(cfg, Qlm::AbstractVector{<:Complex}) -> Vector{Float64}

Packed scalar synthesis from Qlm (LM order) to flattened real grid (length nlat*nlon).
"""
function synthesis_packed(cfg::SHTConfig, Qlm::AbstractVector{<:Complex})
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    # Packed LM order stores only valid (l,m) pairs. Expand to dense
    # (l+1,m+1) so the core synthesis kernel can be reused.
    alm_mat = unpack_lm(cfg, Qlm)
    f = synthesis(cfg, alm_mat; real_output=true)
    return vec(f)
end

"""
    analysis_packed_l(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Integer) -> Vector{ComplexF64}

Scalar analysis truncated to degrees `l ≤ ltr`. The returned packed coefficient
vector has length `cfg.nlm`; coefficients with `l > ltr` are set to zero.
"""
function analysis_packed_l(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Integer)
    length(Vr) == cfg.nspat || throw(DimensionMismatch("Vr must have length $(cfg.nspat)"))
    lcap = min(Int(ltr), cfg.lmax)
    lcap ≥ 0 || throw(ArgumentError("ltr must be ≥ 0"))
    f = reshape(Vr, cfg.nlat, cfg.nlon)
    alm_mat = analysis(cfg, f)
    Qlm = zeros(eltype(alm_mat), cfg.nlm)
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        m > lcap && continue
        for l in m:lcap
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            Qlm[lm] = alm_mat[l+1, m+1]
        end
    end
    return Qlm
end

"""
    synthesis_packed_l(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Integer) -> Vector{Float64}

Scalar synthesis truncated to degrees `l ≤ ltr`. Contributions from higher
degrees are ignored.
"""
function synthesis_packed_l(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Integer)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm must have length $(cfg.nlm)"))
    lcap = min(Int(ltr), cfg.lmax)
    lcap ≥ 0 || throw(ArgumentError("ltr must be ≥ 0"))
    alm_mat = zeros(eltype(Qlm), cfg.lmax+1, cfg.mmax+1)
    # Ignore packed coefficients above lcap without mutating the caller's
    # source vector; this is the spectral low-pass behavior of `_l` variants.
    @inbounds for m in 0:cfg.mmax
        (m % cfg.mres == 0) || continue
        m > lcap && continue
        for l in m:lcap
            lm = LM_index(cfg.lmax, cfg.mres, l, m) + 1
            alm_mat[l+1, m+1] = Qlm[lm]
        end
    end
    f = synthesis(cfg, alm_mat; real_output=true)
    return vec(f)
end

"""
    synthesis_axisym(cfg, Qlm) -> Vector{Float64}

Axisymmetric synthesis from degree-only coefficients to Gauss latitudes.
Input `Qlm` should contain coefficients Q_l for l = 0..lmax.
Returns spatial values at Gauss latitudes.
"""
function synthesis_axisym(cfg::SHTConfig, Qlm::AbstractVector{<:Complex})
    nlat, lmax = cfg.nlat, cfg.lmax
    length(Qlm) == lmax + 1 || throw(DimensionMismatch("Qlm length must be lmax+1=$(lmax+1)"))
    
    Qlm_int = _uses_canonical_convention(cfg) ? Qlm :
              _convert_mode_norm!(similar(Qlm), Qlm, cfg, 0, lmax; to_internal=true)
    RT = real(float(eltype(Qlm_int)))  # AD/Float32-safe output eltype
    Vr = Vector{RT}(undef, nlat)
    P = Vector{Float64}(undef, lmax + 1)
    xv = cfg.x  # hoist field reads out of the i/l loops (cfg is mutable, so not auto-hoisted)

    for i in 1:nlat
        x = xv[i]
        Plm_norm_row!(P, x, lmax, 0)  # m=0 case; P̄ already orthonormal-normalized

        val = zero(RT)
        @inbounds for l in 0:lmax
            val += real(Qlm_int[l+1] * P[l+1])  # Take real part for spatial field
        end
        Vr[i] = val
    end

    return Vr
end

"""
    analysis_axisym_l(cfg, Vr, ltr) -> Vector{ComplexF64}

Axisymmetric degree-limited transform up to degree ltr.
"""
function analysis_axisym_l(cfg::SHTConfig, Vr::AbstractVector{<:Real}, ltr::Int)
    nlat = cfg.nlat
    length(Vr) == nlat || throw(DimensionMismatch("Vr length must be nlat=$(nlat)"))
    ltr <= cfg.lmax || throw(ArgumentError("ltr must be <= lmax=$(cfg.lmax)"))
    
    CT = complex(float(eltype(Vr)))  # AD/Float32-safe output eltype
    Ql = Vector{CT}(undef, ltr + 1)
    fill!(Ql, zero(CT))

    P = Vector{Float64}(undef, ltr + 1)
    xv = cfg.x; wv = cfg.w  # hoist field reads out of the i/l loops (cfg is mutable, so not auto-hoisted)

    for i in 1:nlat
        x = xv[i]
        Plm_norm_row!(P, x, ltr, 0)  # P̄ already orthonormal-normalized

        weighted_Vr = Vr[i] * wv[i]
        @inbounds for l in 0:ltr
            Ql[l+1] += weighted_Vr * P[l+1]
        end
    end

    # Same φ quadrature factor as `analysis_axisym` — see the comment there.
    scaleφ = cfg.cphi * cfg.nlon
    @inbounds for l in eachindex(Ql)
        Ql[l] *= scaleφ
    end
    return _convert_mode_norm!(Ql, Ql, cfg, 0, ltr; to_internal=false)
end

"""
    synthesis_axisym_l(cfg, Qlm, ltr) -> Vector{Float64}

Axisymmetric degree-limited synthesis using degrees up to ltr.
"""
function synthesis_axisym_l(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    ltr_qlm = length(Qlm) - 1  # Convert length to max degree
    ltr <= cfg.lmax || throw(ArgumentError("ltr must be <= lmax=$(cfg.lmax)"))
    ltr <= ltr_qlm || throw(ArgumentError("ltr must be <= length(Qlm)-1=$(ltr_qlm)"))
    
    Qlm_used = view(Qlm, 1:(ltr + 1))
    Qlm_int = _uses_canonical_convention(cfg) ? Qlm_used :
              _convert_mode_norm!(similar(Qlm_used), Qlm_used, cfg, 0, ltr; to_internal=true)
    RT = real(float(eltype(Qlm_int)))  # AD/Float32-safe output eltype
    Vr = Vector{RT}(undef, nlat)
    P = Vector{Float64}(undef, ltr + 1)
    xv = cfg.x  # hoist field reads out of the i/l loops (cfg is mutable, so not auto-hoisted)

    for i in 1:nlat
        x = xv[i]
        Plm_norm_row!(P, x, ltr, 0)  # P̄ already orthonormal-normalized

        val = zero(RT)
        @inbounds for l in 0:ltr
            val += real(Qlm_int[l+1] * P[l+1])
        end
        Vr[i] = val
    end

    return Vr
end

"""
    analysis_packed_ml(cfg, im, Vr_m, ltr) -> Vector{ComplexF64}

Transform spatial field for specific azimuthal mode m to spherical harmonic coefficients.
`im` is the m-index (0-based), `Vr_m` contains complex spatial values for that mode.
Returns coefficients Q_l for degrees l = m..ltr.
"""
function analysis_packed_ml(cfg::SHTConfig, im::Int, Vr_m::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    length(Vr_m) == nlat || throw(DimensionMismatch("Vr_m length must be nlat=$(nlat)"))
    im >= 0 || throw(ArgumentError("im must be >= 0"))
    im <= cfg.mmax || throw(ArgumentError("im must be <= mmax=$(cfg.mmax)"))
    ltr <= cfg.lmax || throw(ArgumentError("ltr must be <= lmax=$(cfg.lmax)"))
    ltr >= im || throw(ArgumentError("ltr must be >= im=$(im)"))

    num_l = ltr - im + 1
    CT = complex(float(real(eltype(Vr_m))))  # AD/Float32-safe output eltype
    Ql = Vector{CT}(undef, num_l)
    fill!(Ql, zero(CT))

    P = Vector{Float64}(undef, ltr + 1)
    scaleφ = cfg.cphi  # Match full transform normalization
    xv = cfg.x; wv = cfg.w  # hoist field reads out of the i/l loops (cfg is mutable, so not auto-hoisted)

    for i in 1:nlat
        x = xv[i]
        Plm_norm_row!(P, x, ltr, im)  # P̄ already orthonormal-normalized

        weighted_Vr = Vr_m[i] * wv[i]
        @inbounds for l in im:ltr
            Ql[l-im+1] += weighted_Vr * P[l+1]
        end
    end

    # Apply phi scaling to match full transform normalization
    Ql .*= scaleφ
    return _convert_mode_norm!(Ql, Ql, cfg, im, ltr; to_internal=false)
end

"""
    synthesis_packed_ml(cfg, im, Ql, ltr) -> Vector{ComplexF64}

Transform spherical harmonic coefficients for specific mode m to spatial field.
`im` is the m-index, `Ql` contains coefficients for degrees l = im..ltr.
Returns complex spatial values for that azimuthal mode.
"""
function synthesis_packed_ml(cfg::SHTConfig, im::Int, Ql::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    im >= 0 || throw(ArgumentError("im must be >= 0"))
    im <= cfg.mmax || throw(ArgumentError("im must be <= mmax=$(cfg.mmax)"))
    ltr <= cfg.lmax || throw(ArgumentError("ltr must be <= lmax=$(cfg.lmax)"))
    ltr >= im || throw(ArgumentError("ltr must be >= im=$(im)"))

    expected_len = ltr - im + 1
    length(Ql) == expected_len || throw(DimensionMismatch("Ql length must be $(expected_len)"))

    Ql_int = _uses_canonical_convention(cfg) ? Ql :
             _convert_mode_norm!(similar(Ql), Ql, cfg, im, ltr; to_internal=true)
    # Output eltype follows the input so AD types propagate.
    CT = promote_type(eltype(Ql_int), ComplexF64)
    Vr_m = Vector{CT}(undef, nlat)
    P = Vector{Float64}(undef, ltr + 1)
    inv_scaleφ = phi_inv_scale(cfg)  # Match full transform normalization
    xv = cfg.x  # hoist field reads out of the i/l loops (cfg is mutable, so not auto-hoisted)

    for i in 1:nlat
        x = xv[i]
        Plm_norm_row!(P, x, ltr, im)  # P̄ already orthonormal-normalized

        val = zero(CT)
        @inbounds for l in im:ltr
            val += Ql_int[l-im+1] * P[l+1]
        end
        Vr_m[i] = val * inv_scaleφ
    end

    return Vr_m
end

"""
    synthesis_point(cfg, Qlm, cost, phi) -> Float64

Evaluate spherical harmonic expansion at a single point (θ,φ) for a real-valued field.
`cost` = cos(θ), `phi` is the azimuthal angle.
`Qlm` should be a matrix of size (lmax+1, mmax+1) with standard indexing
(only m ≥ 0 stored; negative-m reconstructed via Hermitian symmetry).
Returns the real field value at the specified point.
"""
function synthesis_point(cfg::SHTConfig, Qlm::AbstractMatrix{<:Complex}, cost::Real, phi::Real)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Qlm, 1) == lmax + 1 || throw(DimensionMismatch("Qlm first dim must be lmax+1"))
    size(Qlm, 2) == mmax + 1 || throw(DimensionMismatch("Qlm second dim must be mmax+1"))
    Qlm_int = _internal_coefficients(Qlm, cfg)

    # Accumulator eltype follows the input so AD types propagate.
    CT = promote_type(eltype(Qlm), ComplexF64)
    result = zero(real(CT))
    P = Vector{Float64}(undef, lmax + 1)

    # m = 0 contribution (no conjugate partner)
    Plm_norm_row!(P, cost, lmax, 0)  # P̄ already orthonormal-normalized
    @inbounds for l in 0:lmax
        result += real(Qlm_int[l+1, 1]) * P[l+1]
    end

    # m > 0 contributions: add both +m and -m via 2*real(...)
    for m in 1:mmax
        Plm_norm_row!(P, cost, lmax, m)  # P̄ already orthonormal-normalized
        phase = cis(m * phi)  # e^(imφ)
        gm = zero(CT)
        @inbounds for l in m:lmax
            gm += Qlm_int[l+1, m+1] * P[l+1]
        end
        result += 2 * real(gm * phase)
    end

    return result
end
