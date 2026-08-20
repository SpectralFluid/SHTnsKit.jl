#=
================================================================================
energy_diagnostics.jl - Energy Diagnostics for Spherical Harmonic Fields
================================================================================

This file provides functions to compute energy integrals from spherical harmonic
fields, both in spectral and physical space. Energy diagnostics are essential
for validating transforms (Parseval identity) and monitoring simulations.

ENERGY DEFINITIONS
------------------
For a scalar field f(θ,φ) = Σ a_lm Y_l^m(θ,φ), the energy is:
    E = (1/2) ∫∫ |f|² dΩ = (1/2) Σ |a_lm|²  (Parseval's identity)

For a vector field V with spheroidal/toroidal decomposition:
    KE = (1/2) ∫∫ |V|² dΩ = (1/2) Σ l(l+1) [|S_lm|² + |T_lm|²]

The l(l+1) factor arises from the gradient operator acting on scalar potentials.

HERMITIAN SYMMETRY
------------------
For real-valued fields, spectral coefficients satisfy:
    a_{l,-m} = (-1)^m conj(a_{l,m})

We only store m ≥ 0 coefficients. In energy sums, m > 0 modes must be
counted twice to account for the symmetric negative-m contributions:
    wm[m] = 1 for m=0, 2 for m>0

FUNCTION CATEGORIES
-------------------
Spectral energy:
    energy_scalar(cfg, alm)           : Total scalar energy from coefficients
    energy_vector(cfg, Slm, Tlm)      : Total vector kinetic energy
    energy_*_packed(cfg, Qlm)         : Same for packed coefficient format

Grid energy (physical space integration):
    grid_energy_scalar(cfg, f)        : Energy by Gauss-Legendre quadrature
    grid_energy_vector(cfg, Vt, Vp)   : Vector energy on grid

Energy gradients (for optimization):
    grad_energy_scalar_alm(cfg, alm)  : ∂E/∂a_lm
    grad_energy_vector_Slm_Tlm(...)   : ∂KE/∂S_lm, ∂KE/∂T_lm
    grad_grid_energy_*_field(...)     : ∂E/∂f for spatial fields

VALIDATION
----------
Parseval's identity: spectral and grid energies should match:
```julia
E_spectral = energy_scalar(cfg, alm)
f = synthesis(cfg, alm)
E_grid = grid_energy_scalar(cfg, f)
@assert isapprox(E_spectral, E_grid, rtol=1e-10)
```

================================================================================
=#

"""
Energy Diagnostics for Spherical Harmonic Fields

This module provides functions to compute energy diagnostics from spectral
coefficients and spatial fields, including total energy calculations and
energy gradients for optimization applications.

Key assumptions:
- Uses orthonormal spherical harmonics with Condon-Shortley phase convention
- All spectral coefficient matrices have shape (lmax+1, mmax+1) with m ≥ 0
- For real fields, m>0 modes contribute twice due to Hermitian symmetry
"""

# ===== HERMITIAN SYMMETRY WEIGHTS =====
# For real fields, we store only m≥0 coefficients but need to account for
# the symmetric m<0 modes in energy calculations.
# Inline weight: avoids allocating a vector on every energy/gradient call.
@inline _wm(m::Integer, real_field::Bool) = (real_field && m > 0) ? 2.0 : 1.0
@inline _convention_metric(cfg, l::Integer, m::Integer) =
    abs2(coefficient_scale_to_canonical(cfg, Int(l), Int(m)))

"""
    energy_scalar(cfg, alm; real_field=true) -> Float64

Compute the total energy of a scalar field from its spherical harmonic coefficients.

For a scalar field f(θ,φ) = Σ a_lm Y_l^m(θ,φ), the energy is defined as:
E = (1/2) ∫ |f|² dΩ = (1/2) Σ |a_lm|²

This represents the L² norm of the field, which is conserved under orthonormal
spherical harmonic transforms (Parseval's identity).
"""
function energy_scalar(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    # Type-stable accumulator (stays inferrable for Float32 / ForwardDiff.Dual inputs).
    E = zero(promote_type(Float64, real(float(eltype(alm)))))
    @inbounds for m in 0:mmax, l in m:lmax
        E += _wm(m, real_field) * _convention_metric(cfg, l, m) * abs2(alm[l+1, m+1])
    end
    return 0.5 * E
end

"""
    energy_vector(cfg, Slm, Tlm; real_field=true) -> Float64

Compute the total kinetic energy of a vector field from spheroidal/toroidal coefficients.

For a vector field V = ∇×(T Y_l^m êᵣ) + ∇ₕ(S Y_l^m), the kinetic energy is:
KE = (1/2) ∫ |V|² dΩ = (1/2) Σ [l(l+1)|S_lm|² + l(l+1)|T_lm|²]
"""
function energy_vector(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    E = zero(promote_type(Float64, real(float(eltype(Slm))), real(float(eltype(Tlm)))))
    @inbounds for m in 0:mmax, l in max(1,m):lmax  # Vector fields start at l=1
        ll1 = l * (l + 1)
        E += _wm(m, real_field) * _convention_metric(cfg, l, m) * ll1 *
             (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
    end
    return 0.5 * E
end

"""
    grid_energy_scalar(cfg, f) -> Float64

Compute energy of a scalar field directly from its spatial representation.
Uses Gauss-Legendre quadrature for accurate integration.
"""
function grid_energy_scalar(cfg::SHTConfig, f::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    wlat = cfg.w  # Gauss-Legendre weights
    
    E = 0.0
    for j in 1:nlon, i in 1:nlat
        E += wlat[i] * abs2(f[i, j])
    end
    return 0.5 * E * (2π / nlon)  # φ integration weight
end

"""
    grid_energy_vector(cfg, Vt, Vp) -> Float64

Compute kinetic energy of a vector field from its θ and φ components.
"""
function grid_energy_vector(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    wlat = cfg.w
    
    E = 0.0
    for j in 1:nlon, i in 1:nlat
        E += wlat[i] * (abs2(Vt[i, j]) + abs2(Vp[i, j]))
    end
    return 0.5 * E * (2π / nlon)
end

"""
    grad_energy_scalar_alm(cfg, alm; real_field=true) -> Matrix

Compute gradient of scalar field energy with respect to spectral coefficients.
Returns ∂E/∂a_lm for use in optimization problems.
"""
function grad_energy_scalar_alm(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    # Use zeros to ensure l < m positions are properly initialized to zero
    grad = zeros(eltype(alm), size(alm))
    for m in 0:mmax, l in m:lmax
        grad[l+1, m+1] = _wm(m, real_field) * _convention_metric(cfg, l, m) *
                         conj(alm[l+1, m+1])
    end
    return grad
end

"""
    energy_scalar_packed(cfg, Qlm; real_field=true) -> Float64

Compute energy from packed spectral coefficients (1D vector format).
"""
function energy_scalar_packed(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}; real_field::Bool=true)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    E = zero(promote_type(Float64, real(float(eltype(Qlm)))))
    @inbounds for k in eachindex(Qlm)
        m = cfg.mi[k]
        l = cfg.li[k]
        E += _wm(m, real_field) * _convention_metric(cfg, l, m) * abs2(Qlm[k])
    end
    return 0.5 * E
end

"""
    grad_energy_scalar_packed(cfg, Qlm; real_field=true) -> Vector

Compute energy gradient for packed coefficients format.
"""
function grad_energy_scalar_packed(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}; real_field::Bool=true)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    grad = similar(Qlm)
    @inbounds for k in eachindex(Qlm)
        m = cfg.mi[k]
        l = cfg.li[k]
        grad[k] = _wm(m, real_field) * _convention_metric(cfg, l, m) * conj(Qlm[k])
    end
    return grad
end

"""
    grad_energy_vector_Slm_Tlm(cfg, Slm, Tlm; real_field=true) -> (grad_S, grad_T)

Compute gradients of vector field kinetic energy with respect to S and T coefficients.
"""
function grad_energy_vector_Slm_Tlm(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    grad_S, grad_T = allocate_spectral_pair(Slm, Tlm)
    fill!(grad_S, zero(eltype(grad_S)))
    fill!(grad_T, zero(eltype(grad_T)))

    for m in 0:mmax, l in max(1,m):lmax
        ll1 = l * (l + 1)
        w = _wm(m, real_field) * _convention_metric(cfg, l, m) * ll1
        grad_S[l+1, m+1] = w * conj(Slm[l+1, m+1])
        grad_T[l+1, m+1] = w * conj(Tlm[l+1, m+1])
    end
    return grad_S, grad_T
end

"""
    grad_grid_energy_scalar_field(cfg, f) -> Matrix

Compute gradient of grid-based scalar energy with respect to spatial field values.
"""
function grad_grid_energy_scalar_field(cfg::SHTConfig, f::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    wlat = cfg.w
    scale = (2π / nlon)
    
    grad = similar(f)
    for j in 1:nlon, i in 1:nlat
        grad[i, j] = scale * wlat[i] * conj(f[i, j])
    end
    return grad
end

"""
    grad_grid_energy_vector_fields(cfg, Vt, Vp) -> (grad_Vt, grad_Vp)

Compute gradients of grid-based vector energy with respect to vector components.
"""
function grad_grid_energy_vector_fields(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = cfg.nlat, cfg.nlon
    wlat = cfg.w
    scale = (2π / nlon)
    
    grad_Vt, grad_Vp = allocate_spatial_pair(Vt, Vp)
    
    for j in 1:nlon, i in 1:nlat
        w = scale * wlat[i]
        grad_Vt[i, j] = w * conj(Vt[i, j])
        grad_Vp[i, j] = w * conj(Vp[i, j])
    end
    return grad_Vt, grad_Vp
end

"""
    energy_vector_packed(cfg, Spacked, Tpacked; real_field=true) -> Float64

Compute vector field kinetic energy from packed S and T coefficients.
"""
function energy_vector_packed(cfg::SHTConfig, Spacked::AbstractVector{<:Complex}, 
                            Tpacked::AbstractVector{<:Complex}; 
                            real_field::Bool=true)

    length(Spacked) == cfg.nlm || throw(DimensionMismatch("Spacked length must be nlm=$(cfg.nlm)"))
    length(Tpacked) == cfg.nlm || throw(DimensionMismatch("Tpacked length must be nlm=$(cfg.nlm)"))
    E = zero(promote_type(Float64, real(float(eltype(Spacked))), real(float(eltype(Tpacked)))))
    @inbounds for k in eachindex(Spacked)
        l = cfg.li[k]; m = cfg.mi[k]
        if l >= 1
            ll1 = l * (l + 1)
            E += _wm(m, real_field) * _convention_metric(cfg, l, m) * ll1 *
                 (abs2(Spacked[k]) + abs2(Tpacked[k]))
        end
    end
    return 0.5 * E
end

"""
    grad_energy_vector_packed(cfg, Spacked, Tpacked; real_field=true) -> (grad_S, grad_T)

Compute energy gradients for packed vector coefficients.
"""
function grad_energy_vector_packed(cfg::SHTConfig, 
                                Spacked::AbstractVector{<:Complex}, 
                                Tpacked::AbstractVector{<:Complex}; 
                                real_field::Bool=true)
    
    length(Spacked) == cfg.nlm || throw(DimensionMismatch("Spacked length must be nlm=$(cfg.nlm)"))
    length(Tpacked) == cfg.nlm || throw(DimensionMismatch("Tpacked length must be nlm=$(cfg.nlm)"))
    grad_S = similar(Spacked)
    grad_T = similar(Tpacked)
    fill!(grad_S, zero(eltype(grad_S)))
    fill!(grad_T, zero(eltype(grad_T)))
    @inbounds for k in eachindex(Spacked)
        l = cfg.li[k]; m = cfg.mi[k]
        if l >= 1
            w = _wm(m, real_field) * _convention_metric(cfg, l, m) * (l * (l + 1))
            grad_S[k] = w * conj(Spacked[k])
            grad_T[k] = w * conj(Tpacked[k])
        end
    end
    return grad_S, grad_T
end
