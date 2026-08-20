#=
================================================================================
spectral_diagnostics.jl - Spectral Analysis and Energy Spectrum Functions
================================================================================

This file provides functions for computing energy spectra and spectral
diagnostics from spherical harmonic coefficients.

WHY SPECTRAL DIAGNOSTICS?
-------------------------
Energy spectra reveal the scale distribution of physical fields:
- E(l): Energy vs spherical harmonic degree (length scale ~ R/l)
- E(m): Energy vs azimuthal order (zonal vs eddy contributions)
- E(l,m): Full 2D spectral energy distribution

These are essential for:
- Turbulence analysis (checking power law scalings)
- Model validation (comparing spectra to observations)
- Numerical debugging (detecting aliasing, truncation errors)
- Understanding energy cascades in geophysical flows

SPECTRAL SLOPES
---------------
Different physical processes produce characteristic spectral slopes:
- E(l) ~ l^{-3}: 2D enstrophy cascade (geostrophic turbulence)
- E(l) ~ l^{-5/3}: 2D inverse energy cascade
- E(l) ~ l^{-5}: 3D rotating turbulence
- E(l) flat at high l: numerical noise or aliasing

FUNCTIONS PROVIDED
------------------
Degree spectra (summed over m):
    energy_scalar_l_spectrum(cfg, alm)     : E(l) for scalar
    energy_vector_l_spectrum(cfg, Slm, Tlm): KE(l) for vector

Order spectra (summed over l):
    energy_scalar_m_spectrum(cfg, alm)     : E(m) for scalar
    energy_vector_m_spectrum(cfg, Slm, Tlm): KE(m) for vector

2D spectra (per-mode):
    energy_scalar_lm(cfg, alm)             : E_{lm} matrix
    energy_vector_lm(cfg, Slm, Tlm)        : KE_{lm} matrix

USAGE EXAMPLE
-------------
```julia
cfg = create_gauss_config(64, 128)
alm = analysis(cfg, f)

# Get energy spectrum vs degree
El = energy_scalar_l_spectrum(cfg, alm)
ls = 0:cfg.lmax

# Check for -3 power law
using Plots
loglog(ls[2:end], El[2:end], label="E(l)")
plot!(ls[2:end], El[5] .* (ls[2:end] ./ 5).^(-3), ls="--", label="l^{-3}")
```

================================================================================
=#

"""
Spectral Analysis and Spectrum Functions

This module provides functions for analyzing the spectral properties of
spherical harmonic fields, including energy spectra by degree (l) and order (m),
and per-mode energy distributions.

These diagnostics are essential for understanding the scale-dependent properties
of geophysical flows and validating numerical methods.
"""

"""
    energy_scalar_l_spectrum(cfg, alm; real_field=true) -> Vector{Float64}

Compute energy spectrum as a function of spherical harmonic degree l.
Returns E(l) = Σₘ |a_lm|² for each l = 0..lmax.
"""
function energy_scalar_l_spectrum(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax

    El = zeros(real(float(eltype(alm))), lmax + 1)
    for l in 0:lmax, m in 0:min(l, mmax)
        El[l+1] += _wm(m, real_field) * _convention_metric(cfg, l, m) * abs2(alm[l+1, m+1])
    end
    return 0.5 * El
end

"""
    energy_scalar_m_spectrum(cfg, alm; real_field=true) -> Vector{Float64}

Compute energy spectrum as a function of spherical harmonic order m.
Returns E(m) = Σₗ |a_lm|² for each m = 0..mmax.
"""
function energy_scalar_m_spectrum(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax

    Em = zeros(real(float(eltype(alm))), mmax + 1)
    for m in 0:mmax, l in m:lmax
        Em[m+1] += _wm(m, real_field) * _convention_metric(cfg, l, m) * abs2(alm[l+1, m+1])
    end
    return 0.5 * Em
end

"""
    energy_vector_l_spectrum(cfg, Slm, Tlm; real_field=true) -> Vector{Float64}

Compute kinetic energy spectrum by degree l for vector fields.
Returns KE(l) = Σₘ l(l+1)[|S_lm|² + |T_lm|²] for each l = 1..lmax.
"""
function energy_vector_l_spectrum(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax

    El = zeros(real(float(promote_type(eltype(Slm), eltype(Tlm)))), lmax + 1)
    for l in 1:lmax, m in 0:min(l, mmax)  # Vector fields start at l=1
        ll1 = l * (l + 1)
        El[l+1] += _wm(m, real_field) * _convention_metric(cfg, l, m) * ll1 *
                   (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
    end
    return 0.5 * El
end

"""
    energy_vector_m_spectrum(cfg, Slm, Tlm; real_field=true) -> Vector{Float64}

Compute kinetic energy spectrum by order m for vector fields.
Returns KE(m) = Σₗ l(l+1)[|S_lm|² + |T_lm|²] for each m = 0..mmax.
"""
function energy_vector_m_spectrum(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax

    Em = zeros(real(float(promote_type(eltype(Slm), eltype(Tlm)))), mmax + 1)
    for m in 0:mmax, l in max(1,m):lmax
        ll1 = l * (l + 1)
        Em[m+1] += _wm(m, real_field) * _convention_metric(cfg, l, m) * ll1 *
                   (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
    end
    return 0.5 * Em
end

"""
    energy_scalar_lm(cfg, alm; real_field=true) -> Matrix{Float64}

Compute per-mode energy distribution E_lm = |a_lm|² for each (l,m) mode.
Returns matrix of size (lmax+1, mmax+1) with energy contributions.
"""
function energy_scalar_lm(cfg::SHTConfig, alm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax

    RT = real(float(eltype(alm)))
    Elm = Matrix{RT}(undef, lmax+1, mmax+1)
    fill!(Elm, zero(RT))

    for m in 0:mmax, l in m:lmax
        Elm[l+1, m+1] = 0.5 * _wm(m, real_field) * _convention_metric(cfg, l, m) *
                        abs2(alm[l+1, m+1])
    end
    return Elm
end

"""
    energy_vector_lm(cfg, Slm, Tlm; real_field=true) -> Matrix{Float64}

Compute per-mode kinetic energy KE_lm = l(l+1)[|S_lm|² + |T_lm|²] for vector fields.
Returns matrix of size (lmax+1, mmax+1) with energy contributions.
"""
function energy_vector_lm(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_field::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax

    RT = real(float(promote_type(eltype(Slm), eltype(Tlm))))
    Elm = Matrix{RT}(undef, lmax+1, mmax+1)
    fill!(Elm, zero(RT))

    for m in 0:mmax, l in max(1,m):lmax
        ll1 = l * (l + 1)
        Elm[l+1, m+1] = 0.5 * _wm(m, real_field) * _convention_metric(cfg, l, m) * ll1 *
                        (abs2(Slm[l+1, m+1]) + abs2(Tlm[l+1, m+1]))
    end
    return Elm
end
