#=
================================================================================
legendre.jl - Associated Legendre Polynomial Computation
================================================================================

This file implements the core Legendre polynomial computations needed for
spherical harmonic transforms. These polynomials are the latitude (θ) basis
functions in the expansion of functions on the sphere.

MATHEMATICAL BACKGROUND
-----------------------
Associated Legendre polynomials P_l^m(x) are solutions to the Legendre
differential equation:

    (1-x²) d²P/dx² - 2x dP/dx + [l(l+1) - m²/(1-x²)] P = 0

where x = cos(θ), l is the degree, and m is the order (|m| ≤ l).

For spherical harmonics:
    Y_l^m(θ,φ) = N_lm * P_l^m(cos θ) * exp(imφ)

where N_lm is a normalization factor.

IMPLEMENTATION NOTES
--------------------
1. Recurrence Relations:
   - P_l^m values are computed using three-term recurrence
   - This is numerically stable and efficient: O(lmax) per m value
   - The recurrence CANNOT be vectorized due to data dependencies

2. Condon-Shortley Phase:
   - We include the (-1)^m factor in P_m^m (physics convention)
   - This affects the sign of odd-m polynomials

3. Numerical Stability:
   - Normalization factors use log-space arithmetic to avoid overflow
   - (1-x²) is clamped to avoid sqrt of negative due to roundoff

4. Indexing Convention:
   - P[l+1] stores P_l^m (1-based Julia indexing)
   - Valid for l = m, m+1, ..., lmax

PERFORMANCE CONSIDERATIONS
--------------------------
- Plm_row!: Zero allocations, modifies P in place
- Cannot use SIMD for main recurrence (iteration dependency)
- Derivative computation CAN use SIMD (no iteration dependency)

DEBUGGING
---------
```julia
# Test P_0^0(x) = 1 for all x
P = zeros(10)
Plm_row!(P, 0.5, 9, 0)
@assert P[1] ≈ 1.0

# Test P_1^0(x) = x
@assert P[2] ≈ 0.5

# Test P_1^1(x) = -sqrt(1-x²) with CS phase
Plm_row!(P, 0.5, 9, 1)
@assert P[2] ≈ -sqrt(1 - 0.5^2)  # ≈ -0.866
```

================================================================================
=#

# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================

"""
Multiplier for machine epsilon to detect near-pole conditions.
When sin(θ) < POLE_TOLERANCE_FACTOR * eps(T), we're effectively at a pole
and need special handling to avoid division by zero or numerical instability.
"""
const POLE_TOLERANCE_FACTOR = 100

"""
Convergence criterion for Newton-Raphson iteration in Gauss-Legendre quadrature.
Root finding stops when |z - z_prev| < NEWTON_CONVERGENCE_TOL.
"""
const NEWTON_CONVERGENCE_TOL = 1e-15

# =============================================================================
# LEGENDRE POLYNOMIAL COMPUTATION
# =============================================================================

"""
    Plm_row!(P::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}

Compute associated Legendre polynomials P_l^m(x) for all degrees l = 0..lmax at fixed order m.

This function implements the stable three-term recurrence relations for associated 
Legendre polynomials. The algorithm follows the Ferrers definition with the 
Condon-Shortley phase factor (-1)^m included, which is standard in physics.

The input x = cos(θ) where θ is the colatitude angle.
Results are stored as P[l+1] = P_l^m(x) for l = 0..lmax (1-based indexing).

!!! warning "Overflows for lmax ≳ 150"
    These are the UN-normalized P_l^m; their magnitude grows factorially with m
    (~1e305 at m≈150) and overflows `Float64` for lmax ≳ 155. The transform
    kernels no longer use this directly — they use [`Plm_norm_row!`], which
    computes the bounded orthonormal `P̄_l^m = Nlm·P_l^m` and works at any lmax.
    Prefer `Plm_norm_row!` (and `Plm_norm_and_dPdtheta_row!` /
    `Plm_norm_dPdtheta_over_sinth_row!`) for new code. This function is retained
    for low-lmax use and as a reference.
"""
function Plm_row!(P::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    # Initialize output array to zero
    @inbounds fill!(P, zero(T))
    
    # Validate order parameter
    m < 0 && throw(ArgumentError("m must be ≥ 0"))
    
    # Early return if no valid polynomials exist
    lmax >= m || return P

    # ===== SPECIAL CASE: m = 0 (ordinary Legendre polynomials) =====
    if m == 0
        # Base cases for ordinary Legendre polynomials
        P[1] = one(T)                 # P_0^0(x) = 1
        if lmax >= 1
            P[2] = x                  # P_1^0(x) = x
        end
        
        # Three-term recurrence for P_l^0(x) - CANNOT vectorize due to dependencies!
        # P[l+1] depends on P[l] and P[l-1], so each iteration depends on previous ones
        for l in 2:lmax
            # Bonnet's recurrence: (l+1)P_{l+1} = (2l+1)x P_l - l P_{l-1}
            # Rearranged: P_l^0(x) = ((2l-1)x P_{l-1}^0 - (l-1) P_{l-2}^0)/l
            P[l+1] = ((2l - 1) * x * P[l] - (l - 1) * P[l-1]) / l
        end
        return P
    end

    # ===== GENERAL CASE: m > 0 (associated Legendre polynomials) =====

    # Start with P_m^m(x) using explicit formula
    # P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^{m/2}
    #
    # For large m (≳170), the direct product (2m-1)!! × (1-x²)^{m/2} overflows/underflows
    # Float64. Use log-space arithmetic to avoid this:
    #   log|P_m^m| = Σ_{k=1}^{m} log(2k-1) + (m/2) log(1-x²)
    # then exponentiate with the correct sign (-1)^m.
    sx2 = max(zero(T), 1 - x*x)  # (1-x²), guarded against roundoff for |x|≈1

    if m ≤ 140
        # Direct computation is safe for moderate m (no overflow risk)
        pmm = one(T)
        fact = one(T)
        for k in 1:m
            pmm *= -fact * sqrt(sx2)
            fact += 2
        end
    else
        # Log-space computation for large m to prevent overflow/underflow
        if sx2 ≤ zero(T)
            # At poles (x = ±1), P_m^m = 0 for m > 0
            P[m+1] = zero(T)
            if lmax > m
                P[m+2] = zero(T)
            end
            return P
        end
        log_abs = zero(T)
        for k in 1:m
            log_abs += log(T(2k - 1))  # log((2m-1)!!)
        end
        log_abs += T(m) / 2 * log(sx2)  # × (1-x²)^{m/2}
        sign = iseven(m) ? one(T) : -one(T)  # (-1)^m
        pmm = sign * exp(log_abs)
    end
    P[m+1] = pmm

    # If lmax = m, we're done
    if lmax == m
        return P
    end

    # Compute P_{m+1}^m(x) using explicit formula
    # P_{m+1}^m(x) = x (2m+1) P_m^m(x)
    P[m+2] = x * (2m + 1) * pmm

    # Three-term recurrence for remaining degrees l ≥ m+2 - CANNOT vectorize!
    # P[l+1] depends on P[l] and P[l-1], so each iteration depends on previous ones
    for l in (m+2):lmax
        # Recurrence relation for associated Legendre polynomials:
        # P_l^m(x) = ((2l-1)x P_{l-1}^m - (l+m-1) P_{l-2}^m)/(l-m)
        P[l+1] = ((2l - 1) * x * P[l] - (l + m - 1) * P[l-1]) / (l - m)
    end
    
    return P
end

const _INV_SQRT_4PI = 0.28209479177387814  # sqrt(1/(4π)) = orthonormal P̄_0^0

"""
    Plm_norm_row!(P, x, lmax, m)

Fill `P[l+1] = P̄_l^m(x)` for l = m..lmax with the ORTHONORMAL + Condon–Shortley
associated Legendre functions (i.e. exactly `cfg.Nlm[l,m] * (raw P_l^m)`), via an
analytical fully-normalized recurrence that stays bounded (|P̄| ≲ 1) at all l, m —
no overflow at high lmax. Entries l < m are zeroed.
"""
function Plm_norm_row!(P::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    @inbounds fill!(P, zero(T))
    m < 0 && throw(ArgumentError("m must be ≥ 0"))
    lmax >= m || return P
    s = sqrt(max(zero(T), one(T) - x*x))
    pmm = T(_INV_SQRT_4PI)
    @inbounds for k in 1:m
        pmm = -sqrt(T(2k + 1) / T(2k)) * s * pmm
    end
    P[m+1] = pmm
    lmax == m && return P
    P[m+2] = sqrt(T(2m + 3)) * x * pmm
    @inbounds for l in (m+2):lmax
        a = sqrt((T(2l - 1) * T(2l + 1)) / (T(l - m) * T(l + m)))
        b = sqrt((T(2l + 1) * T(l - 1 - m) * T(l - 1 + m)) / (T(2l - 3) * T(l - m) * T(l + m)))
        P[l+1] = a * x * P[l] - b * P[l-1]
    end
    return P
end

"""
    Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, m)

Compute orthonormal normalized P̄_l^m(x) and dP̄_l^m/dθ for l = m..lmax.

`P` is filled with `P̄_l^m` (identical to `Plm_norm_row!`).
`dPdtheta` is filled with `dP̄_l^m/dθ` using the bounded recurrence:

    d(l,m) = l==0 ? 0 : sqrt((l^2 - m^2) / (4l^2 - 1))
    sinθ · dP̄_l^m/dθ = l·d(l+1,m)·P̄_{l+1}^m − (l+1)·d(l,m)·P̄_{l-1}^m

This recurrence is numerically stable (no overflow) at all lmax because it only
involves the bounded P̄ values (|P̄| ≲ 1).

At poles (sinθ → 0):
- m = 0: dP̄_l^0/dθ = 0 (symmetry)
- m = 1: analytic limit = N_{l,1} · (−l(l+1)/2) at north pole
- m > 1: dP̄_l^m/dθ = 0 (P̄_l^m ~ sin^m, derivative ~ sin^{m−1} → 0)
"""
function Plm_norm_and_dPdtheta_row!(P::AbstractVector{T}, dPdtheta::AbstractVector{T},
                                     x::T, lmax::Int, m::Int) where {T<:Real}
    # Allocating fallback: provide a local Pbuf and delegate.
    Pbuf = zeros(T, lmax + 2)
    return Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, m, Pbuf)
end

"""
    Plm_norm_and_dPdtheta_row!(P, dPdtheta, x, lmax, m, Pbuf)

Buffer-taking form: `Pbuf` is caller-supplied scratch of length ≥ `lmax+2`,
avoiding a heap allocation per call in hot loops (`prepare_plm_tables!` calls
this `nlat*(mmax+1)` times).
"""
function Plm_norm_and_dPdtheta_row!(P::AbstractVector{T}, dPdtheta::AbstractVector{T},
                                     x::T, lmax::Int, m::Int,
                                     Pbuf::AbstractVector{T}) where {T<:Real}
    length(Pbuf) >= lmax + 2 || throw(ArgumentError("Pbuf must have length ≥ lmax+2"))

    @inbounds begin
        fill!(dPdtheta, zero(T))

        sinth = sqrt(max(zero(T), one(T) - x*x))
        interior = lmax >= m && sinth >= POLE_TOLERANCE_FACTOR * eps(T)

        if interior
            # The dθ recurrence needs P̄ up to lmax+1, and the first lmax+1 of
            # those ARE the row we must return. Computing `P` with its own
            # `Plm_norm_row!` call and then `Pbuf` with a second one ran the
            # (inherently serial) recurrence twice for one row — about 40 % of
            # this function's cost. Run it once into the longer buffer instead.
            Plm_norm_row!(Pbuf, x, lmax + 1, m)
            fill!(P, zero(T))
            copyto!(P, 1, Pbuf, 1, lmax + 1)
        else
            Plm_norm_row!(P, x, lmax, m)
        end

        lmax < m && return P, dPdtheta

        # Handle poles (sinθ ≈ 0)
        if !interior
            if m == 1
                # Analytic limit: dP̄_l^1/dθ|_{θ=0} = N_{l,1} * (−l(l+1)/2)
                # N_{l,1} = sqrt[(2l+1)/(4π) / (l*(l+1))], so
                # dP̄_l^1/dθ|_{θ=0} = −(1/2)*sqrt[(2l+1)*l*(l+1)/(4π)]
                for l in 1:lmax
                    mag = T(0.5) * sqrt(T(2l + 1) * T(l) * T(l + 1) * T(_INV_SQRT_4PI^2 * 4π))
                    # = 0.5 * sqrt((2l+1)*l*(l+1) / (4π))
                    # Simplify: _INV_SQRT_4PI = 1/sqrt(4π), so 1/(4π) = _INV_SQRT_4PI^2
                    # mag = 0.5 * sqrt((2l+1)*l*(l+1)) * _INV_SQRT_4PI
                    # Recompute cleanly:
                    mag = T(0.5) * T(_INV_SQRT_4PI) * sqrt(T(2l + 1) * T(l) * T(l + 1))
                    if x > 0  # North pole
                        dPdtheta[l+1] = -mag
                    else  # South pole: dP̄/dθ(π) = (-1)^(l+1)·mag (parity of P̄_l^1)
                        dPdtheta[l+1] = (isodd(l+1) ? -one(T) : one(T)) * mag
                    end
                end
            end
            # m == 0 or m > 1: stays zero
            return P, dPdtheta
        end

        # Standard case: `Pbuf` already holds P̄ up to degree lmax+1 from the
        # single recurrence pass above.

        # d(l,m) = sqrt((l^2 - m^2) / (4l^2 - 1))  [0 for l=0]
        inv_sinth = one(T) / sinth

        for l in m:lmax
            dl1m = (l + 1 == 0) ? zero(T) :
                   sqrt(max(zero(T), T((l+1)^2 - m^2) / T(4*(l+1)^2 - 1)))
            # d(l,m) is analytically zero at l=m.  Branch before sqrt so
            # coordinate Duals do not evaluate sqrt(Dual(0, 0)), whose
            # undefined generic derivative would contaminate the row with NaN.
            dlm  = (l == 0 || l == m) ? zero(T) :
                   sqrt(max(zero(T), T(l^2      - m^2) / T(4*l^2      - 1)))
            # sinθ · dP̄_l/dθ = l*d(l+1,m)*P̄_{l+1} - (l+1)*d(l,m)*P̄_{l-1}
            Pm1 = l > 0 ? Pbuf[l] : zero(T)   # P̄_{l-1}^m (index l, 1-based)
            Pp1 = Pbuf[l + 2]                  # P̄_{l+1}^m (index l+2, 1-based)
            dPdtheta[l+1] = (T(l) * dl1m * Pp1 - T(l + 1) * dlm * Pm1) * inv_sinth
        end
    end

    return P, dPdtheta
end

"""
    Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, m[, Pbuf])

Combined computation of P̄_l^m(x), dP̄_l^m/dθ, and P̄_l^m/sin(θ) in a single call,
using the bounded orthonormal normalized convention (no overflow at high lmax).

- `P`           ← P̄_l^m   (same as `Plm_norm_row!`)
- `dPdtheta`    ← dP̄_l^m/dθ  (same as `Plm_norm_and_dPdtheta_row!`)
- `P_over_sinth`← P̄_l^m / sin(θ)
- `Pbuf`        — optional caller-supplied scratch of length ≥ lmax+2 (avoids heap
                  allocation in hot loops; the allocating 6-arg form falls back when
                  omitted).

All three quantities are finite for all l, m at interior points. Pole limits mirror
those in `Plm_dPdtheta_over_sinth_row!` but scaled to the normalized convention.
"""
function Plm_norm_dPdtheta_over_sinth_row!(P::AbstractVector{T}, dPdtheta::AbstractVector{T},
                                            P_over_sinth::AbstractVector{T},
                                            x::T, lmax::Int, m::Int) where {T<:Real}
    # Allocating fallback: provide a local Pbuf and delegate.
    Pbuf = zeros(T, lmax + 2)
    return Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, m, Pbuf)
end

function Plm_norm_dPdtheta_over_sinth_row!(P::AbstractVector{T}, dPdtheta::AbstractVector{T},
                                            P_over_sinth::AbstractVector{T},
                                            x::T, lmax::Int, m::Int,
                                            Pbuf::AbstractVector{T}) where {T<:Real}
    length(Pbuf) >= lmax + 2 || throw(ArgumentError("Pbuf must have length ≥ lmax+2"))

    @inbounds begin
        fill!(dPdtheta, zero(T))
        fill!(P_over_sinth, zero(T))

        sinth = sqrt(max(zero(T), one(T) - x*x))
        interior = lmax >= m && sinth >= POLE_TOLERANCE_FACTOR * eps(T)

        if interior
            # Single recurrence pass: the dθ formula needs P̄ to degree lmax+1 and
            # the first lmax+1 entries of that row ARE `P`. Filling `P` and `Pbuf`
            # with two separate `Plm_norm_row!` calls ran the serial recurrence
            # twice per row (~40 % of this function's cost).
            Plm_norm_row!(Pbuf, x, lmax + 1, m)
            fill!(P, zero(T))
            copyto!(P, 1, Pbuf, 1, lmax + 1)
        else
            Plm_norm_row!(P, x, lmax, m)
        end

        lmax < m && return P, dPdtheta, P_over_sinth

        if !interior
            if m == 1
                for l in 1:lmax
                    mag = T(0.5) * T(_INV_SQRT_4PI) * sqrt(T(2l + 1) * T(l) * T(l + 1))
                    if x > 0  # North pole
                        dPdtheta[l+1]    = -mag
                        P_over_sinth[l+1] = -mag
                    else  # South pole (parity of P̄_l^1 about θ=π/2)
                        sign_dP  = isodd(l+1) ? -one(T) : one(T)   # (-1)^(l+1)
                        sign_Pos = isodd(l)   ? -one(T) : one(T)   # (-1)^l
                        dPdtheta[l+1]    = sign_dP  * mag
                        P_over_sinth[l+1] = sign_Pos * mag
                    end
                end
            end
            # m == 0 or m > 1: both stay zero
            return P, dPdtheta, P_over_sinth
        end

        # Standard case: `Pbuf` already holds P̄ to degree lmax+1 from the single
        # recurrence pass above.

        inv_sinth = one(T) / sinth

        for l in m:lmax
            dl1m = sqrt(max(zero(T), T((l+1)^2 - m^2) / T(4*(l+1)^2 - 1)))
            # See the allocating derivative-row form above: d(m,m) is exactly
            # zero and must not pass through sqrt for coordinate AD numbers.
            dlm  = (l == 0 || l == m) ? zero(T) :
                   sqrt(max(zero(T), T(l^2 - m^2) / T(4*l^2 - 1)))
            Pm1 = l > 0 ? Pbuf[l] : zero(T)
            Pp1 = Pbuf[l + 2]
            dPdtheta[l+1]    = (T(l) * dl1m * Pp1 - T(l + 1) * dlm * Pm1) * inv_sinth
            P_over_sinth[l+1] = Pbuf[l + 1] * inv_sinth
        end
    end

    return P, dPdtheta, P_over_sinth
end

"""
    Plm_and_dPdx_row!(P, dPdx, x, lmax, m)

Simultaneously compute associated Legendre polynomials P_l^m(x) and their derivatives.

This function efficiently computes both P_l^m(x) and dP_l^m/dx for all degrees
l = m..lmax at fixed order m. The derivatives are computed with respect to the
argument x = cos(θ), not with respect to the angle θ itself.

The derivative calculation uses the standard recurrence relation:
dP_l^m/dx = [l*x*P_l^m - (l+m)*P_{l-1}^m] / (x²-1)

At poles (x = ±1), special handling is used to avoid division by zero:
- For m = 0: dP_l^0/dx|_{x=±1} = (±1)^{l+1} * l(l+1)/2 (known analytical result)
- For m > 0: derivatives are set to 0 (P_l^m(±1) = 0 for m > 0, and the vector
  transform code handles the 0/0 limit via Plm_and_dPdtheta_row!)

This is essential for computing gradients and differential operators on the sphere.
"""
function Plm_and_dPdx_row!(P::AbstractVector{T}, dPdx::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    # First compute the Legendre polynomials
    Plm_row!(P, x, lmax, m)

    @inbounds begin
        # Initialize derivative array
        fill!(dPdx, zero(T))

        # Early return if no valid polynomials
        if lmax < m
            return P, dPdx
        end

        # Precompute common factor (x² - 1) for derivative formula
        x2m1 = x*x - one(T)

        # Handle poles (x = ±1) where x² - 1 = 0 causes division by zero
        if abs(x2m1) < POLE_TOLERANCE_FACTOR * eps(T)
            if m == 0
                # For m = 0: dP_l^0/dx|_{x=+1} = l(l+1)/2
                #            dP_l^0/dx|_{x=-1} = (-1)^{l+1} * l(l+1)/2
                for l in 1:lmax
                    sign_factor = x > 0 ? one(T) : T((-1)^(l+1))
                    dPdx[l+1] = sign_factor * T(l * (l + 1)) / 2
                end
            else
                # For m > 0: P_l^m(±1) = 0 (because of sin^m factor)
                # The derivative exists but requires L'Hôpital's rule
                # For vector transforms, use Plm_and_dPdtheta_row! which handles this properly
                # Here we set to 0 as a safe fallback
                fill!(dPdx, zero(T))
            end
            return P, dPdx
        end

        # Standard case: not at a pole
        # Handle l = m case (base case for derivatives)
        l = m
        dPdx[l+1] = (l == 0) ? zero(T) : (m * x * P[l+1]) / x2m1

        # Compute derivatives for l ≥ m+1 using recurrence relation - SAFE to vectorize!
        # Each dPdx[l+1] depends only on already-computed P[l+1] and P[l], no iteration dependencies
        @simd ivdep for l in (m+1):lmax
            # Standard derivative recurrence:
            # dP_l^m/dx = [l*x*P_l^m - (l+m)*P_{l-1}^m] / (x²-1)
            dPdx[l+1] = (l * x * P[l+1] - (l + m) * P[l]) / x2m1
        end
    end

    return P, dPdx
end

"""
    Plm_and_dPdtheta_row!(P, dPdtheta, x, lmax, m)

Compute associated Legendre polynomials P_l^m(x) and their θ-derivatives dP_l^m/dθ.

This function computes dP/dθ directly (not dP/dx), which is singularity-free at poles.
The relationship is: dP/dθ = -sin(θ) * dP/dx = -√(1-x²) * dP/dx

For m > 0, this uses the recurrence relation in a form that avoids the pole singularity:
    dP_l^m/dθ = l*cos(θ)*P_l^m/sin(θ) - (l+m)*P_{l-1}^m/sin(θ)
              = l*x*P_l^m/√(1-x²) - (l+m)*P_{l-1}^m/√(1-x²)

At poles (θ=0 or π, x=±1):
- For m = 0: dP_l^0/dθ = 0 (by symmetry)
- For m = 1: dP_l^1/dθ has a finite nonzero limit
- For m > 1: dP_l^m/dθ = 0 (P_l^m vanishes faster than linearly)

This is the preferred function for vector spherical harmonic transforms.
"""
function Plm_and_dPdtheta_row!(P::AbstractVector{T}, dPdtheta::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    # First compute the Legendre polynomials
    Plm_row!(P, x, lmax, m)

    @inbounds begin
        # Initialize derivative array
        fill!(dPdtheta, zero(T))

        # Early return if no valid polynomials
        if lmax < m
            return P, dPdtheta
        end

        # Compute sin(θ) = √(1-x²)
        sinth = sqrt(max(zero(T), one(T) - x*x))

        # Handle poles (x = ±1) where sin(θ) = 0
        if sinth < POLE_TOLERANCE_FACTOR * eps(T)
            # At poles, dP/dθ has special values:
            # - For m = 0: dP_l^0/dθ = 0 (symmetry)
            # - For m = 1: dP_l^1/dθ|_{θ=0} = -√(l(l+1)/2) (with normalization)
            #              dP_l^1/dθ|_{θ=π} = (-1)^l * √(l(l+1)/2)
            # - For m > 1: dP_l^m/dθ = 0 (P_l^m ~ sin^m, so derivative ~ sin^{m-1} → 0)

            if m == 0
                # dP_l^0/dθ = 0 at poles
                fill!(dPdtheta, zero(T))
            elseif m == 1
                # For m = 1, use the analytical limit
                # P_l^1 = -sin(θ) * dP_l^0/dx (Condon-Shortley phase)
                # dP_l^1/dθ = -cos(θ)*dP_l^0/dx + sin²(θ)*d²P_l^0/dx²
                # At north pole (θ=0, x=1): dP_l^1/dθ = -1*P'_l(1) = -l(l+1)/2
                # At south pole (θ=π, x=-1): dP_l^1/dθ = +1*P'_l(-1) = (-1)^{l+1}*l(l+1)/2
                for l in 1:lmax
                    if x > 0  # North pole (θ = 0)
                        # dP_l^1/dθ|_{θ=0} = -l(l+1)/2 (always negative, no sign alternation)
                        dPdtheta[l+1] = -T(l * (l + 1)) / 2
                    else  # South pole (θ = π)
                        # dP_l^1/dθ|_{θ=π} = (-1)^{l+1} * l(l+1)/2
                        dPdtheta[l+1] = T((-1)^(l+1)) * T(l * (l + 1)) / 2
                    end
                end
            else
                # For m > 1, dP_l^m/dθ = 0 at poles
                fill!(dPdtheta, zero(T))
            end
            return P, dPdtheta
        end

        # Standard case: not at a pole
        # Use: dP/dθ = -sin(θ) * dP/dx = -sin(θ) * [l*x*P - (l+m)*P_{l-1}] / (x²-1)
        #            = -sin(θ) * [l*x*P - (l+m)*P_{l-1}] / (-sin²θ)
        #            = [l*x*P - (l+m)*P_{l-1}] / sin(θ)
        inv_sinth = one(T) / sinth

        # Handle l = m case
        l = m
        if l == 0
            dPdtheta[l+1] = zero(T)  # dP_0^0/dθ = 0
        else
            # dP_m^m/dθ = m*x*P_m^m/sin(θ) (since P_{m-1}^m = 0)
            dPdtheta[l+1] = m * x * P[l+1] * inv_sinth
        end

        # Compute derivatives for l ≥ m+1
        @simd ivdep for l in (m+1):lmax
            # dP_l^m/dθ = [l*x*P_l^m - (l+m)*P_{l-1}^m] / sin(θ)
            dPdtheta[l+1] = (l * x * P[l+1] - (l + m) * P[l]) * inv_sinth
        end
    end

    return P, dPdtheta
end

"""
    Plm_over_sinth_row!(P, P_over_sinth, x, lmax, m)

Compute P_l^m(x) and P_l^m(x)/sin(θ) with proper pole handling.

For m > 0, P_l^m(x)/sin(θ) has a finite limit at poles because P_l^m ~ sin^m(θ).
This function computes that limit correctly, avoiding the 0/0 indeterminate form.

At poles (x = ±1):
- For m = 0: Not applicable (m must be > 0 for this to be meaningful in vector transforms)
- For m = 1: P_l^1/sin(θ) → finite limit related to dP_l^0/dx
- For m > 1: P_l^m/sin(θ) → 0 (numerator has higher order zero)

This is essential for vector spherical harmonic transforms which require (im/sinθ)*Y terms.
"""
function Plm_over_sinth_row!(P::AbstractVector{T}, P_over_sinth::AbstractVector{T}, x::T, lmax::Int, m::Int) where {T<:Real}
    # First compute the Legendre polynomials
    Plm_row!(P, x, lmax, m)

    @inbounds begin
        # Initialize output array
        fill!(P_over_sinth, zero(T))

        # Early return if no valid polynomials
        if lmax < m
            return P, P_over_sinth
        end

        # Compute sin(θ) = √(1-x²)
        sinth = sqrt(max(zero(T), one(T) - x*x))

        # Handle poles (x = ±1) where sin(θ) = 0
        if sinth < POLE_TOLERANCE_FACTOR * eps(T)
            if m == 0
                # P_l^0/sin(θ) is genuinely singular for m=0, but this case
                # shouldn't be used in vector transforms (m=0 terms don't have 1/sinθ factor)
                fill!(P_over_sinth, zero(T))
            elseif m == 1
                # For m = 1: P_l^1 = -sin(θ) * dP_l^0/dx (Condon-Shortley)
                # So P_l^1/sin(θ) = -dP_l^0/dx
                # At x = 1 (north pole): dP_l^0/dx = l(l+1)/2, so P_l^1/sinθ = -l(l+1)/2
                # At x = -1 (south pole): dP_l^0/dx = (-1)^{l+1}*l(l+1)/2, so P_l^1/sinθ = (-1)^l*l(l+1)/2
                for l in 1:lmax
                    if x > 0  # North pole
                        P_over_sinth[l+1] = -T(l * (l + 1)) / 2
                    else  # South pole
                        P_over_sinth[l+1] = T((-1)^l) * T(l * (l + 1)) / 2
                    end
                end
            else
                # For m > 1: P_l^m ~ sin^m(θ), so P_l^m/sin(θ) ~ sin^{m-1}(θ) → 0
                fill!(P_over_sinth, zero(T))
            end
            return P, P_over_sinth
        end

        # Standard case: not at a pole
        inv_sinth = one(T) / sinth
        for l in m:lmax
            P_over_sinth[l+1] = P[l+1] * inv_sinth
        end
    end

    return P, P_over_sinth
end

"""
    Plm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, m)

Combined computation of P_l^m(x), dP_l^m/dθ, and P_l^m/sin(θ) in a single pass.

This avoids the redundant Legendre polynomial computation that occurs when calling
`Plm_and_dPdtheta_row!` and `Plm_over_sinth_row!` separately (each internally calls
`Plm_row!`). For vector spherical harmonic transforms that need all three quantities,
this function is ~33% faster than two separate calls.
"""
function Plm_dPdtheta_over_sinth_row!(P::AbstractVector{T}, dPdtheta::AbstractVector{T},
                                       P_over_sinth::AbstractVector{T}, x::T,
                                       lmax::Int, m::Int) where {T<:Real}
    # Compute Legendre polynomials once
    Plm_row!(P, x, lmax, m)

    @inbounds begin
        fill!(dPdtheta, zero(T))
        fill!(P_over_sinth, zero(T))

        if lmax < m
            return P, dPdtheta, P_over_sinth
        end

        sinth = sqrt(max(zero(T), one(T) - x*x))

        # Handle poles (x = ±1)
        if sinth < POLE_TOLERANCE_FACTOR * eps(T)
            # dP/dθ at poles
            if m == 0
                # dP_l^0/dθ = 0 at poles; P_l^0/sinθ singular but unused for m=0
            elseif m == 1
                for l in 1:lmax
                    if x > 0  # North pole
                        dPdtheta[l+1] = -T(l * (l + 1)) / 2
                        P_over_sinth[l+1] = -T(l * (l + 1)) / 2
                    else  # South pole
                        dPdtheta[l+1] = T((-1)^(l+1)) * T(l * (l + 1)) / 2
                        P_over_sinth[l+1] = T((-1)^l) * T(l * (l + 1)) / 2
                    end
                end
            end
            # For m == 0 or m > 1: both dPdtheta and P_over_sinth stay zero
            return P, dPdtheta, P_over_sinth
        end

        # Standard case: not at a pole
        inv_sinth = one(T) / sinth

        # dP/dθ: l = m case
        l = m
        if l == 0
            dPdtheta[l+1] = zero(T)
        else
            dPdtheta[l+1] = m * x * P[l+1] * inv_sinth
        end

        # dP/dθ: l ≥ m+1
        @simd ivdep for l in (m+1):lmax
            dPdtheta[l+1] = (l * x * P[l+1] - (l + m) * P[l]) * inv_sinth
        end

        # P/sinθ
        for l in m:lmax
            P_over_sinth[l+1] = P[l+1] * inv_sinth
        end
    end

    return P, dPdtheta, P_over_sinth
end

"""
    Nlm_table(lmax::Int, mmax::Int)

Precompute normalization factors for orthonormal spherical harmonics.

The normalization ensures that the spherical harmonics form an orthonormal basis:
∫ Y_l^m(θ,φ) [Y_{l'}^{m'}(θ,φ)]* dΩ = δ_{ll'} δ_{mm'}

The normalization factor is:
N_{l,m} = sqrt[(2l+1)/(4π) * (l-m)!/(l+m)!]

This function computes all factors for 0≤m≤mmax, m≤l≤lmax using stable
logarithmic arithmetic to avoid factorial overflow.

Returns matrix N[l+1,m+1] with 1-based indexing.
"""
function Nlm_table(lmax::Int, mmax::Int)
    # Allocate normalization table
    N = Matrix{Float64}(undef, lmax + 1, mmax + 1)

    for m in 0:mmax
        for l in 0:lmax
            if l < m
                # No spherical harmonic exists for l < m
                N[l+1, m+1] = 0.0
            else
                # Compute normalization factor in log space for numerical stability
                # log(N_{l,m}) = 0.5 * [log(2l+1) - log(4π) + log(Γ(l-m+1)) - log(Γ(l+m+1))]
                # Using Γ(n) = (n-1)! for integer n
                lr = 0.5 * (log(2l + 1.0) - log(4π)) + 0.5 * (_loggamma(l - m + 1) - _loggamma(l + m + 1))

                # Convert back from log space
                N[l+1, m+1] = exp(lr)
            end
        end
    end

    return N
end

#=
================================================================================
Gauss-Legendre Quadrature Nodes and Weights
================================================================================

This section computes the nodes (abscissas) and weights for Gauss-Legendre
numerical integration on the interval [-1, 1].

WHY GAUSS-LEGENDRE FOR SPHERICAL HARMONICS?
-------------------------------------------
Spherical harmonic transforms require integrating functions against Legendre
polynomials over the latitude (θ) direction. The standard substitution
x = cos(θ) transforms this to an integral over [-1, 1]:

    ∫₀^π f(θ) P_l(cos θ) sin θ dθ = ∫₋₁¹ f(arccos x) P_l(x) dx

Gauss-Legendre quadrature with n points is EXACT for polynomials up to
degree 2n-1. Since P_l(x) has degree l, using n = lmax+1 nodes ensures
exact integration of the Legendre polynomial component.

ALGORITHM
---------
1. Initial guess for roots using Abramowitz & Stegun formula (10.18.10):
   z₀ = cos(π(k - 0.25)/(n + 0.5))

2. Newton-Raphson iteration to refine root of P_n(z):
   z_{new} = z - P_n(z)/P'_n(z)

3. Weights computed from derivative at converged root:
   w_k = 2 / ((1 - z_k²) [P'_n(z_k)]²)

PROPERTIES
----------
- Nodes are symmetric: x_k = -x_{n-k+1}
- Weights are symmetric: w_k = w_{n-k+1}
- Weights sum to 2: Σ w_k = ∫₋₁¹ 1 dx = 2
- Nodes are roots of P_n(x)

DEBUGGING
---------
```julia
x, w = gausslegendre(16)
@assert length(x) == length(w) == 16
@assert abs(sum(w) - 2.0) < 1e-14     # Weights sum to 2
@assert all(x[i] ≈ -x[17-i] for i in 1:8)  # Symmetry
@assert all(w[i] ≈ w[17-i] for i in 1:8)   # Weight symmetry
```

================================================================================
=#

"""
    gausslegendre(n::Int)

Compute Gauss–Legendre nodes and weights for integrating functions on [-1, 1].
Returns `(x::Vector{Float64}, w::Vector{Float64})` with `length == n` where
`∫_{-1}^1 f(x) dx ≈ sum(w .* f.(x))`.
"""
function gausslegendre(n::Int)
    n > 0 || throw(ArgumentError("n must be positive"))
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    # Number of roots computed (symmetry)
    m = (n + 1) >>> 1
    for k in 1:m
        # Initial guess (Abramowitz & Stegun 10.18.10)
        z = cos(pi * (k - 0.25) / (n + 0.5))
        z1 = 0.0
        # Newton iterations
        converged = false
        for _ in 1:50
            pnm1 = 1.0
            pn = z
            # Compute Legendre P_n(z) using recurrence
            for l in 2:n
                pnp1 = ((2l - 1) * z * pn - (l - 1) * pnm1) / l
                pnm1, pn = pn, pnp1
            end
            # Derivative using stable relation
            pd = n * (z * pn - pnm1) / (z^2 - 1.0)
            z1 = z
            z -= pn / pd
            if abs(z - z1) < NEWTON_CONVERGENCE_TOL
                converged = true
                break
            end
        end
        converged || @warn "Gauss-Legendre root $k did not converge in 50 iterations for n=$n"
        # Compute P_n and derivative at converged root for weights
        pnm1 = 1.0
        pn = z
        for l in 2:n
            pnp1 = ((2l - 1) * z * pn - (l - 1) * pnm1) / l
            pnm1, pn = pn, pnp1
        end
        pd = n * (z * pn - pnm1) / (z^2 - 1.0)

        x[k] = -z
        x[n - k + 1] = z
        wk = 2.0 / ((1.0 - z^2) * pd^2)
        w[k] = wk
        w[n - k + 1] = wk
    end
    return x, w
end

"""
    _fejer1_weights(n::Int) -> Vector{Float64}

Return Fejér's first-rule weights for the `n` midpoint nodes
`cos((j + 1/2)π/n)`, `j = 0:n-1`. The weights integrate with respect to
`dx` on `[-1, 1]`, matching the latitude weights expected by the spherical
harmonic analysis kernels.

The cosine series is accumulated in `Float64`, as is the rest of the stored
grid geometry in `SHTConfig`. With `n` nodes the rule has algebraic degree
`n - 1` for even `n` and degree `n` for odd `n`.
"""
function _fejer1_weights(n::Int)
    n > 0 || throw(ArgumentError("n must be positive"))
    w = Vector{Float64}(undef, n)
    n_inv = 1.0 / n
    last_mode = fld(n, 2)

    @inbounds for j in 0:(n - 1)
        θj = (j + 0.5) * π * n_inv
        series = 0.0
        for k in 1:last_mode
            series += cos(2k * θj) / (4k^2 - 1)
        end
        w[j + 1] = 2n_inv * (1 - 2series)
    end
    return w
end

"""
    _clenshaw_curtis_weights(n::Int) -> Vector{Float64}

Return Clenshaw–Curtis weights for the `n` pole-inclusive nodes
`cos(jπ/(n-1))`, `j = 0:n-1`. The returned weights integrate with respect
to `dx` on `[-1, 1]` and include the parity-dependent endpoint correction.
With `n` nodes the rule has algebraic degree `n - 1` for even `n` and degree
`n` for odd `n`.
"""
function _clenshaw_curtis_weights(n::Int)
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    N = n - 1
    w = Vector{Float64}(undef, n)

    if iseven(N)
        endpoint = 1.0 / (N^2 - 1)
        w[1] = endpoint
        w[end] = endpoint
        @inbounds for j in 1:(N - 1)
            θj = j * π / N
            value = 1.0
            for k in 1:(N ÷ 2 - 1)
                value -= 2cos(2k * θj) / (4k^2 - 1)
            end
            value -= cos(N * θj) / (N^2 - 1)
            w[j + 1] = 2value / N
        end
    else
        endpoint = 1.0 / N^2
        w[1] = endpoint
        w[end] = endpoint
        @inbounds for j in 1:(N - 1)
            θj = j * π / N
            value = 1.0
            for k in 1:((N - 1) ÷ 2)
                value -= 2cos(2k * θj) / (4k^2 - 1)
            end
            w[j + 1] = 2value / N
        end
    end
    return w
end

"""
    thetaphi_from_nodes(nlat::Int, nlon::Int)

Return `θ` and `φ` arrays where `θ ∈ [0, π]` (Gauss–Legendre nodes mapped) and
`φ ∈ [0, 2π)` equally spaced longitudes suitable for FFT-based azimuthal transforms.

Note: The returned arrays follow the gausslegendre ordering (south-to-north,
`x` from -1 to +1). Callers that need north-to-south ordering should reverse
the latitude-dependent arrays.
"""
function thetaphi_from_nodes(nlat::Int, nlon::Int)
    x, w = gausslegendre(nlat)
    θ = acos.(x)
    φ = (2π / nlon) .* (0:(nlon-1))
    return θ, φ, x, w
end
