#=
================================================================================
operators.jl - Spherical Differential Operators in Spectral Space
================================================================================

This file implements differential operators that act directly on spherical
harmonic coefficients WITHOUT going to physical space. This enables efficient
spectral methods for solving PDEs on the sphere.

WHY SPECTRAL OPERATORS?
-----------------------
Many differential equations on the sphere can be solved more efficiently in
spectral space. Instead of computing derivatives by finite differences, we
apply recurrence relations that couple neighboring degrees.

Key insight: Multiplication by cos(θ) or differentiation ∂/∂θ ONLY couples
modes with adjacent l values (l-1 and l+1) at fixed m. This makes the
operators sparse (tridiagonal in l).

OPERATORS IMPLEMENTED
---------------------
1. cos(θ) multiplication:
   cos(θ) Y_l^m = a_l^m Y_{l-1}^m + b_l^m Y_{l+1}^m

   where:
   a_l^m = sqrt[(l² - m²) / ((2l-1)(2l+1))]
   b_l^m = sqrt[((l+1)² - m²) / ((2l+1)(2l+3))]

2. sin(θ) ∂/∂θ derivative:
   sin(θ) ∂/∂θ Y_l^m = l b_l^m Y_{l+1}^m - (l+1) a_l^m Y_{l-1}^m

   The sin(θ) factor removes the coordinate singularity at poles.

MATRIX STORAGE FORMAT
---------------------
Operators are stored as vectors of coupling coefficients in packed format:
    mx[2*lm + 1] = c_minus  (coupling to l-1)
    mx[2*lm + 2] = c_plus   (coupling to l+1)

where lm is the packed index for (l,m) in SHTns LM order (m≥0 only).

APPLICATIONS
------------
- Solving Laplace/Poisson equations spectrally
- Computing Coriolis terms in shallow water equations
- Latitude-weighted integration
- Building spectral Jacobians for implicit solvers

USAGE EXAMPLE
-------------
```julia
cfg = create_gauss_config(32, 64)
mx = zeros(2 * cfg.nlm)  # Allocate coefficient array

# Fill with cos(θ) operator coefficients
mul_ct_matrix(cfg, mx)

# Apply operator to spectral field
Qlm = pack_alm(cfg, alm)  # Convert to packed format
Rlm = similar(Qlm)
SH_mul_mx(cfg, mx, Qlm, Rlm)  # R = cos(θ) * Q in spectral space
```

DEBUGGING
---------
```julia
# Verify operator application matches spatial multiplication
f = synthesis(cfg, alm)
f_times_costh = f .* cos.(acos.(cfg.x'))  # Spatial multiplication
alm_check = analysis(cfg, f_times_costh)

# Should match spectral operator result
@assert alm_check ≈ unpack_alm(cfg, Rlm)
```

================================================================================
=#

"""
Spherical Differential Operators in Spectral Space

This module implements differential operators that act on spherical harmonic coefficients.
The operators are represented as sparse matrices that couple neighboring degrees l at fixed
azimuthal order m, taking advantage of the recurrence relations for spherical harmonics.

The key operators implemented are:
- cos(θ) multiplication: couples Y_l^m to Y_{l±1}^m
- sin(θ) ∂/∂θ derivative: couples Y_l^m to Y_{l±1}^m with different coefficients

SHTns-style functions (SHTns names and packed LM order; the operator layout
differs from SHTns, see "Differences from SHTns" in docs/src/norms.md):
- `mul_ct_matrix(cfg, mx)`: fill `mx` (length 2*nlm) with coefficients for cos(θ) operator
- `st_dt_matrix(cfg, mx)`: fill `mx` (length 2*nlm) with coefficients for sin(θ) ∂/∂θ operator
- `SH_mul_mx(cfg, mx, Qlm, Rlm)`: apply a tridiagonal operator that couples (l,m) to l±1 at fixed m

Matrix storage format:
For each packed index `lm` (SHTns LM order, m≥0), we store two coupling coefficients:
- `mx[2*lm+1] = c_minus` for coupling to Y_{l-1}^m (lower degree neighbor)
- `mx[2*lm+2] = c_plus` for coupling to Y_{l+1}^m (higher degree neighbor)
Out-of-range neighbors (l<m or l>lmax) are automatically ignored.
"""

"""Fill `mx` with the cosθ (`derivative=false`) or sinθ ∂θ couplings."""
function _operator_matrix!(cfg::SHTConfig, mx::AbstractVector{<:Real},
                           derivative::Bool)
    # Validate coefficient matrix size (2 coefficients per (l,m) mode)
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    
    # Fill coupling coefficients for each (l,m) mode in packed order
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]  # Degree for this packed index
        m = cfg.mi[lm0+1]  # Order for this packed index
        
        # Coupling coefficient to Y_{l-1}^m (downward in degree)
        # From recurrence: cos(θ) Y_l^m = a_l^m Y_{l-1}^m + b_l^m Y_{l+1}^m
        a = (l == 0 || l == m) ? 0.0 :
            sqrt(max(0.0, (l^2 - m^2) / ((2l - 1) * (2l + 1))))
        
        # Coupling coefficient to Y_{l+1}^m (upward in degree)  
        b = (l == cfg.lmax) ? 0.0 :
            sqrt(max(0.0, ((l + 1)^2 - m^2) / ((2l + 1) * (2l + 3))))

        # mx acts on the configured coefficients at the public boundary.  The
        # orthonormal recurrence therefore needs the source/destination
        # coefficient-scale ratio for conventions whose scale varies with l
        # (Schmidt).  CS and REAL_NORM cancel because both neighbours share m.
        source_scale = coefficient_scale_to_canonical(cfg, l, m)
        if a != 0
            a *= source_scale / coefficient_scale_to_canonical(cfg, l - 1, m)
        end
        if b != 0
            b *= source_scale / coefficient_scale_to_canonical(cfg, l + 1, m)
        end
        
        # Store in packed format: [c_minus, c_plus] for each (l,m)
        mx[2*lm0 + 1] = derivative ? -(l + 1) * a : a
        mx[2*lm0 + 2] = derivative ? l * b : b
    end
    return mx
end

"""Typed host metadata consumed by the vendor-neutral coefficient kernels."""
function _gpu_operator_metadata(cfg::SHTConfig, ::Type{T}) where {T<:Real}
    li = Int32.(cfg.li)
    mi = Int32.(cfg.mi)
    down_ratios = ones(T, cfg.nlm)
    up_ratios = ones(T, cfg.nlm)
    lower = zeros(Int32, cfg.nlm)
    upper = zeros(Int32, cfg.nlm)
    @inbounds for k in 1:cfg.nlm
        l = cfg.li[k]; m = cfg.mi[k]
        source = coefficient_scale_to_canonical(cfg, l, m)
        if l > m
            down_ratios[k] = T(source /
                coefficient_scale_to_canonical(cfg, l - 1, m))
            lower[k] = LM_index(cfg.lmax, cfg.mres, l - 1, m) + 1
        end
        if l < cfg.lmax
            up_ratios[k] = T(source /
                coefficient_scale_to_canonical(cfg, l + 1, m))
            upper[k] = LM_index(cfg.lmax, cfg.mres, l + 1, m) + 1
        end
    end
    return (; li, mi, down_ratios, up_ratios, lower, upper)
end

"""
    mul_ct_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})

Fill `mx` with coupling coefficients for cosθ operator: cosθ Y_l^m = a_l^m Y_{l-1}^m + b_l^m Y_{l+1}^m.

`mx[2lm+1]` and `mx[2lm+2]`, with `lm` the 0-based packed index of `(l, m)`, hold
the couplings of `Y_l^m` to `Y_{l-1}^m` and `Y_{l+1}^m`: the layout
[`SH_mul_mx`](@ref) expects. cosθ is symmetric, so this is also the array SHTns's
`mul_ct_matrix` produces.
"""
mul_ct_matrix(::CPU, cfg::SHTConfig, mx::AbstractVector{<:Real}) =
    _operator_matrix!(cfg, mx, false)
mul_ct_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real}) =
    mul_ct_matrix(CPU(), cfg, mx)

"""
    st_dt_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real})

Fill `mx` with coupling coefficients for sinθ ∂_θ operator:
sinθ ∂_θ Y_l^m = l b_l^m Y_{l+1}^m - (l+1) a_l^m Y_{l-1}^m.

Same layout as [`mul_ct_matrix`](@ref): each entry pair holds how `Y_l^m` couples
to its neighbours. SHTns stores, for each `(l, m)`, what it receives from its
neighbours instead, which differs for this non-symmetric operator; see
"Differences from SHTns" in the normalization guide for the conversion.
"""
st_dt_matrix(::CPU, cfg::SHTConfig, mx::AbstractVector{<:Real}) =
    _operator_matrix!(cfg, mx, true)
st_dt_matrix(cfg::SHTConfig, mx::AbstractVector{<:Real}) =
    st_dt_matrix(CPU(), cfg, mx)

"""
    SH_mul_mx(cfg::SHTConfig, mx::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Apply a nearest-neighbor-in-l operator represented by `mx` to `Qlm` and write to `Rlm`.
Both `Qlm` and `Rlm` are length `cfg.nlm` packed vectors (m≥0, SHTns LM order).
`mx` uses the layout of [`mul_ct_matrix`](@ref) and [`st_dt_matrix`](@ref), which
for non-symmetric operators differs from SHTns's (see "Differences from SHTns"
in the normalization guide).

!!! warning
    `Rlm` must not alias `Qlm`. Each output couples the l-neighbors of `Qlm`, and
    those neighbors are overwritten by earlier iterations, so an in-place call
    (`Rlm === Qlm`) reads already-mutated values and gives wrong results. Use a
    separate output buffer.
"""
function SH_mul_mx(::CPU, cfg::SHTConfig, mx::AbstractVector{<:Real},
                   Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    # Validate input array dimensions
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    length(Rlm) == cfg.nlm || throw(DimensionMismatch("Rlm length must be nlm=$(cfg.nlm)"))
    Base.mightalias(Qlm, Rlm) && throw(ArgumentError(
        "SH_mul_mx input and output must not alias",
    ))
    
    lmax = cfg.lmax; mres = cfg.mres
    
    # Apply the tridiagonal operator to each (l,m) mode
    # Key insight: mx stores coefficients describing how (l,m) contributes to its neighbors.
    # For R_l^m = (Op * Q)_lm, we need coefficients from neighbors that contribute TO (l,m):
    # - Y_{l-1}^m contributes to Y_l^m via b_{l-1}^m (the upward coefficient from l-1)
    # - Y_{l+1}^m contributes to Y_l^m via a_{l+1}^m (the downward coefficient from l+1)
    @inbounds for lm0 in 0:(cfg.nlm-1)
        l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]  # Get (l,m) for this packed index

        acc = zero(promote_type(eltype(Qlm), eltype(Rlm), complex(eltype(mx))))  # eltype-preserving accumulator (AD-safe)

        # Contribution from lower degree neighbor Y_{l-1}^m
        # Y_{l-1}^m couples upward to Y_l^m via b_{l-1}^m
        if l > m && l > 0  # Check bounds: l-1 ≥ m and l-1 ≥ 0
            lm_prev = LM_index(lmax, mres, l-1, m)  # Get packed index for (l-1,m)
            c_from_below = mx[2*lm_prev + 2]        # b_{l-1}^m: upward coeff from neighbor
            acc += c_from_below * Qlm[lm_prev + 1]  # Add contribution from lower neighbor
        end

        # Contribution from higher degree neighbor Y_{l+1}^m
        # Y_{l+1}^m couples downward to Y_l^m via a_{l+1}^m
        if l < lmax  # Check bounds: l+1 ≤ lmax
            lm_next = LM_index(lmax, mres, l+1, m)  # Get packed index for (l+1,m)
            c_from_above = mx[2*lm_next + 1]        # a_{l+1}^m: downward coeff from neighbor
            acc += c_from_above * Qlm[lm_next + 1]  # Add contribution from upper neighbor
        end

        Rlm[lm0 + 1] = acc  # Store result for this (l,m) mode
    end
    return Rlm
end

SH_mul_mx(cfg::SHTConfig, mx::AbstractVector{<:Real},
          Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex}) =
    SH_mul_mx(CPU(), cfg, mx, Qlm, Rlm)
