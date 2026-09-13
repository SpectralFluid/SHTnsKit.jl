"""
Zygote Extension for Reverse-Mode Automatic Differentiation

This extension provides Zygote.jl integration for reverse-mode automatic differentiation 
through SHTnsKit operations. Zygote excels at gradients of scalar-valued functions and 
is particularly well-suited for optimization and machine learning workflows.

Key Features:
- Reverse-mode AD through spherical harmonic transforms
- Custom adjoints (pullback rules) for SHTnsKit operations  
- Efficient gradient computation for energy functionals
- Support for rotation and operator gradients
- Compatible with both distributed and regular arrays

Comparison with ForwardDiff:
- Zygote: Better for many inputs → one output (optimization problems)
- ForwardDiff: Better for few inputs → many outputs (Jacobians)

The extension includes custom @adjoint definitions to ensure proper gradient flow
through complex mathematical operations like rotations and matrix operations.
"""
module SHTnsKitZygoteExt

using Zygote
using SHTnsKit

# ===== SCALAR FIELD GRADIENTS =====

"""
    zgrad_scalar_energy(cfg, f) -> ∂E/∂f

Zygote gradient of scalar energy E = 0.5 ∫ |f|^2 under spectral transform.

This function uses Zygote's reverse-mode automatic differentiation to compute
the gradient of the scalar energy functional. The computation flows through:
spatial field → spherical harmonic analysis → energy computation → gradient.

Parameters:
- cfg: SHTnsKit configuration
- f: Input scalar field matrix [nlat × nlon]

Returns:
- Gradient matrix of same size as f, computed via reverse-mode AD
"""
function SHTnsKit.zgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)
    loss(x) = SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, x))
    return Zygote.gradient(loss, f)[1]
end

# ===== DISTRIBUTED ARRAY SUPPORT =====
# Generic distributed/array wrappers (avoid hard dependency on PencilArrays)
# These provide automatic differentiation support for distributed computing

"""
    zgrad_scalar_energy(cfg, fθφ::AbstractArray) -> ∂E/∂fθφ

Zygote gradient computation for distributed arrays using reverse-mode AD.

This overload works with any AbstractArray type, including distributed arrays
like PencilArrays. Zygote automatically handles the gradient computation through
the distributed array operations without requiring explicit conversions.

The advantage over ForwardDiff here is that Zygote can work directly with
the distributed array types, preserving their structure throughout the
automatic differentiation process.
"""
function SHTnsKit.zgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, fθφ::AbstractArray)
    loss(x) = SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, x))
    return Zygote.gradient(loss, fθφ)[1]
end

# ===== VECTOR FIELD GRADIENTS =====

"""
    zgrad_vector_energy(cfg, Vtθφ, Vpθφ) -> (∂E/∂Vt, ∂E/∂Vp)

Zygote gradient of vector field energy for distributed arrays.

This function computes gradients of vector energy functionals using Zygote's
reverse-mode automatic differentiation. The vector energy typically involves
kinetic energy, enstrophy, or other quadratic functionals in fluid dynamics.

Parameters:
- cfg: SHTnsKit configuration
- Vtθφ: Theta component of vector field (distributed array)
- Vpθφ: Phi component of vector field (distributed array)

Returns:
- Tuple of gradient arrays (∂E/∂Vt, ∂E/∂Vp) with same structure as inputs
"""
function SHTnsKit.zgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vtθφ::AbstractArray, Vpθφ::AbstractArray)
    # Validate dimensions match
    size(Vtθφ) == size(Vpθφ) || throw(DimensionMismatch("Vt and Vp must have the same dimensions"))
    # Define vector energy functional for two-argument case
    loss(Xt, Xp) = begin
        Slm, Tlm = SHTnsKit.analysis_sphtor(cfg, Xt, Xp)      # Spheroidal/toroidal analysis
        SHTnsKit.energy_vector(cfg, Slm, Tlm)                  # Compute vector energy
    end
    
    # Use Zygote to compute gradients with respect to both vector components
    g = Zygote.gradient(loss, Vtθφ, Vpθφ)
    return g[1], g[2]                                          # Return (∂E/∂Vt, ∂E/∂Vp)
end

"""
    zgrad_vector_energy(cfg, Vt, Vp) -> (∂E/∂Vt, ∂E/∂Vp)
"""
function SHTnsKit.zgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    # Validate dimensions match
    size(Vt) == size(Vp) || throw(DimensionMismatch("Vt and Vp must have the same dimensions"))
    loss(Xt, Xp) = begin
        Slm, Tlm = SHTnsKit.analysis_sphtor(cfg, Xt, Xp)
        SHTnsKit.energy_vector(cfg, Slm, Tlm)
    end
    g = Zygote.gradient(loss, Vt, Vp)
    return g[1], g[2]
end

"""
    zgrad_enstrophy_Tlm(cfg, Tlm) -> ∂Z/∂Tlm

Zygote gradient of enstrophy with respect to toroidal spectrum Tlm.
"""
function SHTnsKit.zgrad_enstrophy_Tlm(cfg::SHTnsKit.SHTConfig, Tlm::AbstractMatrix)
    loss(X) = SHTnsKit.enstrophy(cfg, X)
    return Zygote.gradient(loss, Tlm)[1]
end

"""
    zgrad_rotation_angles_real(cfg, Qlm, α, β, γ) -> (∂L/∂α, ∂L/∂β, ∂L/∂γ)

Gradient of L = 0.5 || R ||^2 where R = rotation_apply_real(cfg, Qlm; α,β,γ) using Zygote.
Assumes mres == 1.
"""
function SHTnsKit.zgrad_rotation_angles_real(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector, α::Real, β::Real, γ::Real)
    lmax, mmax = cfg.lmax, cfg.mmax
    # Forward rotate to get R (used as cotangent in loss 0.5||R||^2)
    R = similar(Qlm)
    r = SHTnsKit.SHTRotation(lmax, mmax; α=float(α), β=float(β), γ=float(γ))
    SHTnsKit.shtns_rotation_apply_real(r, Qlm, R)
    gα = 0.0; gβ = 0.0; gγ = 0.0
    for l in 0:lmax
        mm = min(l, mmax)
        dl = SHTnsKit.wigner_d_matrix(l, float(β))
        ddl = SHTnsKit.wigner_d_matrix_deriv(l, float(β))
        n = 2l + 1
        b = zeros(ComplexF64, n)   # zeros, NOT undef: only |mp| <= min(l,mmax) is filled below,
                                   # and `dl * b` reads the whole 2l+1 range — with `undef` the
                                   # l > mmax slots fed malloc garbage into the gradients.
        # Build complex A_m' then b = e^{-i m' γ} A
        for mp in -mm:mm
            idxp = SHTnsKit.LM_index(lmax, 1, l, abs(mp)) + 1
            if mp == 0
                A = Qlm[idxp]
            elseif mp > 0
                A = Qlm[idxp]
            else
                A = (-1)^(-mp) * conj(Qlm[SHTnsKit.LM_index(lmax, 1, l, -mp) + 1])
            end
            b[mp + l + 1] = A * cis(-mp * float(γ))
        end
        c = dl * b
        for m in 0:mm
            idxp = SHTnsKit.LM_index(lmax, 1, l, m) + 1
            Rm = c[m + l + 1] * cis(-m * float(α))
            ȳ = R[idxp]
            gα += real(conj(ȳ) * ((0 - 1im) * m * Rm))
            # β gradient via d'(β)
            sβ = zero(ComplexF64)
            sγ = zero(ComplexF64)
            for mp in -l:l
                sβ += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
                sγ += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
            end
            gβ += real(conj(ȳ) * (sβ * cis(-m * float(α))))
            gγ += real(conj(ȳ) * (sγ * cis(-m * float(α))))
        end
    end
    return gα, gβ, gγ
end

"""
    zgrad_rotation_angles_cplx(lmax, mmax, Zlm, α, β, γ) -> (∂L/∂α, ∂L/∂β, ∂L/∂γ)

Gradient of L = 0.5 || R ||^2 where R = rotation_apply_cplx(lmax,mmax,Zlm; α,β,γ) using Zygote.
"""
function SHTnsKit.zgrad_rotation_angles_cplx(lmax::Integer, mmax::Integer, Zlm::AbstractVector, α::Real, β::Real, γ::Real)
    lmax = Int(lmax); mmax = Int(mmax)
    R = similar(Zlm)
    r = SHTnsKit.SHTRotation(lmax, mmax; α=float(α), β=float(β), γ=float(γ))
    SHTnsKit.shtns_rotation_apply_cplx(r, Zlm, R)
    # The loop below reimplements the Wigner engine, which works in the Y_l^m
    # basis, while the public transform is `ε ∘ engine ∘ ε` in the packed layout.
    # `|εx| = |x|`, so L is unchanged — but the intermediates must be the
    # engine's, i.e. built from εZ and compared against εR.
    ε = SHTnsKit._lmcplx_ybasis_signs(lmax, mmax)
    Zlm = ε .* Zlm
    R = ε .* R
    gα = 0.0; gβ = 0.0; gγ = 0.0
    for l in 0:lmax
        mm = min(l, mmax)
        dl = SHTnsKit.wigner_d_matrix(l, float(β))
        ddl = SHTnsKit.wigner_d_matrix_deriv(l, float(β))
        n = 2l + 1
        # Build b_m' = e^{-i m' γ} Z_{l,m'} for m' in [-l..l]
        b = zeros(ComplexF64, n)   # zeros, NOT undef: only |mp| <= min(l,mmax) is filled below,
                                   # and `dl * b` reads the whole 2l+1 range — with `undef` the
                                   # l > mmax slots fed malloc garbage into the gradients.
        # Only |mp| <= min(l, mmax) exists in the LM_cplx layout; `LM_cplx_index`
        # throws beyond that, while the primal simply truncates there. Iterate the
        # stored range and leave the rest at zero, matching the primal.
        mml = min(l, mmax)
        for mp in -mml:mml
            idx = SHTnsKit.LM_cplx_index(lmax, mmax, l, mp) + 1
            b[mp + l + 1] = Zlm[idx] * cis(-mp * float(γ))
        end
        # c_m = sum_{m'} d_{m m'}(β) b_{m'}
        c = dl * b
        for m in -mm:mm
            idxm = SHTnsKit.LM_cplx_index(lmax, mmax, l, m) + 1
            ȳ = R[idxm]
            Rm = c[m + l + 1] * cis(-m * float(α))
            # α-gradient: -i m R_m
            gα += real(conj(ȳ) * ((0 - 1im) * m * Rm))
            # γ-gradient: through input phase of b
            sγ = zero(ComplexF64)
            for mp in -l:l
                sγ += dl[m + l + 1, mp + l + 1] * ((0 - 1im) * mp * b[mp + l + 1])
            end
            gγ += real(conj(ȳ) * (sγ * cis(-m * float(α))))
            # β-gradient: via d'(β)
            sβ = zero(ComplexF64)
            for mp in -l:l
                sβ += ddl[m + l + 1, mp + l + 1] * b[mp + l + 1]
            end
            gβ += real(conj(ȳ) * (sβ * cis(-m * float(α))))
        end
    end
    return gα, gβ, gγ
end

# -----------------------------
## Zygote-specific adjoints for rotations/operators to ensure gradients are not `nothing`
## These mirror the ChainRules rrules but live here to guarantee Zygote picks them up.
## The Q̄ formulas match SHTnsKitAdvancedADExt and are FD-verified in
## test/serial/test_rotation_gradients.jl (m>0 packed modes carry double field
## weight ⇒ standard-inner-product adjoint is Q̄ = W·R⁻¹·(W⁻¹ ȳ), W = diag(wm)).
_zyg_rot_wm(cfg) = Float64[cfg.mi[k] == 0 ? 1.0 : 2.0 for k in 1:cfg.nlm]

function _zyg_configured_rotation_adjoint!(cfg, r, ȳ, Q̄)
    wm = _zyg_rot_wm(cfg)
    ȳ_canonical = SHTnsKit._analysis_cotangent_to_canonical(ȳ, cfg)
    Q̄_canonical = SHTnsKit._uses_canonical_convention(cfg) ? Q̄ : similar(Q̄)
    SHTnsKit.shtns_rotation_apply_real(r, ȳ_canonical ./ wm, Q̄_canonical)
    Q̄_canonical .*= wm
    if Q̄_canonical !== Q̄
        SHTnsKit.convert_alm_norm!(Q̄, Q̄_canonical, cfg; to_internal=true)
    end
    return Q̄
end

Zygote.@adjoint function SHTnsKit.SH_Zrotate(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    y = SHTnsKit.SH_Zrotate(cfg, Qlm, alpha, Rlm)
    # In-place rotation overwrites Qlm, and callers may reuse either buffer
    # before the pullback. Preserve the primal values needed for dR/dα.
    rotated = copy(y)
    function back(ȳ)
        # Diagonal Rlm = Qlm·e^{-imα} ⇒ Q̄ = ȳ·e^{imα} = SH_Zrotate(ȳ, -α).
        Q̄ = similar(Qlm)
        SHTnsKit.SH_Zrotate(cfg, ȳ, -alpha, Q̄)
        dα = 0.0
        for m in 0:cfg.mmax
            (m % cfg.mres == 0) || continue
            for l in m:cfg.lmax
                lm = SHTnsKit.LM_index(cfg.lmax, cfg.mres, l, m) + 1
                dα += real(conj(ȳ[lm]) * ((0 - 1im) * m * rotated[lm]))
            end
        end
        return (nothing, Q̄, dα, nothing)
    end
    return y, back
end

Zygote.@adjoint function SHTnsKit.SH_Yrotate(cfg::SHTnsKit.SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    # Snapshot BEFORE the primal: an in-place rotation (Rlm === Qlm) overwrites
    # Qlm, and callers may reuse either buffer before the pullback runs. The dα
    # formula needs the *input* coefficients, so they must be preserved here.
    Qlm_saved = copy(Qlm)
    y = SHTnsKit.SH_Yrotate(cfg, Qlm, alpha, Rlm)
    function back(ȳ)
        inverse = SHTnsKit.SHTRotation(cfg.lmax, cfg.mmax)
        SHTnsKit.shtns_rotation_set_angles_ZYZ(inverse, 0.0, -alpha, 0.0)
        Q̄ = similar(Qlm_saved)
        _zyg_configured_rotation_adjoint!(cfg, inverse, ȳ, Q̄)
        # angle gradient via derivative of Wigner-d at beta=alpha
        dα = 0.0
        lmax, mmax = cfg.lmax, cfg.mmax
        Qlm_canonical = SHTnsKit._internal_coefficients(Qlm_saved, cfg)
        ȳ_canonical = SHTnsKit._analysis_cotangent_to_canonical(ȳ, cfg)
        for l in 0:lmax
            mm = min(l, mmax)
            b = zeros(eltype(ȳ_canonical), 2l+1)
            for mp in -mm:mm
                idxp = SHTnsKit.LM_index(lmax, 1, l, abs(mp)) + 1
                if mp == 0
                    b[mp + l + 1] = Qlm_canonical[idxp]
                elseif mp > 0
                    b[mp + l + 1] = Qlm_canonical[idxp]
                    b[-mp + l + 1] = (-1)^mp * conj(Qlm_canonical[idxp])
                end
            end
            dd = SHTnsKit.wigner_d_matrix_deriv(l, float(alpha))
            for m in 0:mm
                lm = SHTnsKit.LM_index(lmax, 1, l, m) + 1
                s = zero(eltype(ȳ_canonical))
                for mp in -l:l
                    s += dd[m + l + 1, mp + l + 1] * b[mp + l + 1]
                end
                dα += real(conj(ȳ_canonical[lm]) * s)
            end
        end
        return (nothing, Q̄, dα, nothing)
    end
    return y, back
end

Zygote.@adjoint function SHTnsKit.SH_mul_mx(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real}, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    y = SHTnsKit.SH_mul_mx(cfg, mx, Qlm, Rlm)
    function back(ȳ)
        lmax = cfg.lmax; mres = cfg.mres
        Q̄ = zeros(eltype(Qlm), length(Qlm))
        mx̄ = zeros(eltype(mx), length(mx))
        # Forward pass: R[lm0+1] = mx[2*lm_prev+2]*Q[lm_prev+1] + mx[2*lm_next+1]*Q[lm_next+1]
        # where lm_prev = LM_index(l-1,m), lm_next = LM_index(l+1,m)
        @inbounds for lm0 in 0:(cfg.nlm-1)
            l = cfg.li[lm0+1]; m = cfg.mi[lm0+1]
            rbar = ȳ[lm0 + 1]
            # Contribution from lower neighbor Y_{l-1}^m
            if l > m && l > 0
                lm_prev = SHTnsKit.LM_index(lmax, mres, l-1, m)
                c_from_below = mx[2*lm_prev + 2]  # b_{l-1}^m coefficient
                Q̄[lm_prev + 1] += c_from_below * rbar  # mx is real, so conj not needed
                mx̄[2*lm_prev + 2] += real(conj(rbar) * Qlm[lm_prev + 1])
            end
            # Contribution from upper neighbor Y_{l+1}^m
            if l < lmax
                lm_next = SHTnsKit.LM_index(lmax, mres, l+1, m)
                c_from_above = mx[2*lm_next + 1]  # a_{l+1}^m coefficient
                Q̄[lm_next + 1] += c_from_above * rbar  # mx is real, so conj not needed
                mx̄[2*lm_next + 1] += real(conj(rbar) * Qlm[lm_next + 1])
            end
        end
        return (nothing, mx̄, Q̄, nothing)
    end
    return y, back
end

end # module
