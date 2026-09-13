#=
================================================================================
rotations.jl - Rotations of Spherical Harmonic Expansions
================================================================================

This file implements rotation operations on spherical harmonic coefficients.
Rotations can be applied directly in spectral space without going through
physical space, which is efficient for rotating fields on the sphere.

WHY SPECTRAL ROTATIONS?
-----------------------
Rotating a function on the sphere in physical space requires:
1. Synthesize to grid: O((lmax)² × nlon)
2. Interpolate to new grid positions: O(nlat × nlon)
3. Analyze back: O((lmax)² × nlon)

Spectral rotation is more direct:
- Rotation of Y_l^m produces a linear combination of Y_l^{m'} for |m'| ≤ l
- The mixing is given by Wigner-d matrices d^l_{mm'}(β)
- Complexity: O((lmax)³) but exact with no interpolation errors

EULER ANGLE CONVENTIONS
-----------------------
Rotations are specified using Euler angles (α, β, γ) in either:
- ZYZ convention (default): R = Rz(α) Ry(β) Rz(γ)
- ZXZ convention: R = Rz(α) Rx(β) Rz(γ)

Special cases implemented efficiently:
- Z-rotation: Just phase multiplication (m-dependent), O(nlm)
- Y-rotation: Requires full Wigner-d matrix application
- 90° rotations: Common in coordinate transformations

WIGNER-D MATRICES
-----------------
The little Wigner-d matrix d^l_{mm'}(β) gives the transformation of
spherical harmonics under rotation by angle β about the y-axis:

    R_y(β) Y_l^m = Σ_{m'} d^l_{m m'}(β) Y_l^{m'}

Full rotation R(α,β,γ) in ZYZ convention:
    a'_{lm} = Σ_{m'} e^{-imα} d^l_{mm'}(β) e^{-im'γ} a_{lm'}

FUNCTIONS
---------
Fast axis rotations:
    SH_Zrotate(cfg, Qlm, α, Rlm)     : Z-axis rotation (very fast)
    SH_Yrotate(cfg, Qlm, α, Rlm)     : Y-axis rotation
    SH_Xrotate90(cfg, Qlm, Rlm)      : X-axis 90° rotation
    SH_Yrotate90(cfg, Qlm, Rlm)      : Y-axis 90° rotation

General rotations:
    SHTRotation                       : Rotation specification struct
    shtns_rotation_set_angles_ZYZ     : Set Euler angles (ZYZ)
    shtns_rotation_apply_real         : Apply to real-field coefficients
    shtns_rotation_apply_cplx         : Apply to complex-field coefficients

Wigner-d computation:
    wigner_d_matrix(l, β)            : Compute d^l_{mm'}(β) matrix
    wigner_d_matrix_deriv(l, β)      : Derivative ∂d^l/∂β

USAGE EXAMPLE
-------------
```julia
cfg = create_gauss_config(32, 64)
Qlm = pack_alm(cfg, alm)  # Original coefficients
Rlm = similar(Qlm)        # Output

# Z-rotation by 45°
SH_Zrotate(cfg, Qlm, π/4, Rlm)

# General rotation using Euler angles
rot = SHTRotation(cfg.lmax, cfg.mmax)
shtns_rotation_set_angles_ZYZ(rot, α=π/6, β=π/3, γ=π/4)
shtns_rotation_apply_real(rot, Qlm, Rlm)
```

DEBUGGING
---------
```julia
# Z-rotation should just multiply by exp(-imα)
# For m=2 mode, rotation by α should multiply by exp(-2iα)
rot_coeff = Rlm[idx] / Qlm[idx]  # where idx is a mode with m=2
@assert rot_coeff ≈ cis(-2 * α)
```

================================================================================
=#

"""
Rotations of spherical harmonic expansions.

Currently supports fast rotation around the Z-axis by angle `alpha` in radians.
"""

"""
    SH_Zrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})

Rotate a real-field SH expansion around the Z-axis by angle `alpha`.
Input and output are packed `Qlm` vectors (LM order, m ≥ 0). In-place supported if `Rlm === Qlm`.

# Sign convention

`R_lm = Q_lm · exp(-i m α)`. This is the **active** rotation of the field by `+α`
about `ẑ`: the rotated field is `g(θ, φ) = f(θ, φ - α)`. Equivalently, a feature
at longitude `φ₀` moves to `φ₀ + α`.

Three things pin this sign and would all break if it were flipped to `+imα`
(which is the *passive* convention, `f(θ, φ + α)`):

  * the spatial rotation above, verified directly in
    `test/serial/test_rotations.jl` against an FFT-grid φ shift;
  * the general Wigner engine — `shtns_rotation_apply_real` with
    `ZYZ(α, 0, 0)` (or `ZYZ(0, 0, α)`) must equal this function, and it builds
    `diag(e^{-imα}) · d(β) · diag(e^{-imγ})`;
  * the distributed twins `dist_SH_Zrotate` in `src/parallel_dense.jl` and
    `ext/ParallelRotationsPencil.jl`, and every rotation `rrule`.
"""
function SH_Zrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be nlm=$(cfg.nlm)"))
    length(Rlm) == cfg.nlm || throw(DimensionMismatch("Rlm length must be nlm=$(cfg.nlm)"))
    lmax = cfg.lmax; mres = cfg.mres
    @inbounds for m in 0:cfg.mmax
        (m % mres == 0) || continue
        phase = cis(-m * alpha)
        for l in m:lmax
            lm = LM_index(lmax, mres, l, m) + 1
            Rlm[lm] = Qlm[lm] * phase
        end
    end
    return Rlm
end

"""
    struct SHTRotation
Holds Euler angles and target sizes for rotation.
- `lmax, mmax`: degrees/orders supported by the rotation.
- `α, β, γ`: Euler angles (ZYZ by default) in radians.
- `conv`: `:ZYZ` or `:ZXZ` convention.
"""
Base.@kwdef mutable struct SHTRotation
    lmax::Int
    mmax::Int
    α::Float64 = 0.0
    β::Float64 = 0.0
    γ::Float64 = 0.0
    conv::Symbol = :ZYZ
end

# Convenient outer constructor with keyword defaults (to mirror SHTns usage)
function SHTRotation(lmax::Integer, mmax::Integer; α::Real=0.0, β::Real=0.0, γ::Real=0.0, conv::Symbol=:ZYZ)
    return SHTRotation(Int(lmax), Int(mmax), float(α), float(β), float(γ), conv)
end

"""Apply an orthonormal-basis rotation at a configured-coefficient boundary."""
function _apply_configured_rotation_real(cfg::SHTConfig, r::SHTRotation,
                                         Qlm::AbstractVector{<:Complex},
                                         Rlm::AbstractVector{<:Complex})
    if _uses_canonical_convention(cfg)
        return shtns_rotation_apply_real(r, Qlm, Rlm)
    end

    # The Wigner engine acts on canonical orthonormal+CS coefficients. A rotation
    # mixes m at fixed l, so m-dependent real-normalization and phase factors do
    # not commute with it: convert on entry and undo the conversion on exit.
    Qlm_int = _internal_coefficients(Qlm, cfg)
    Rlm_int = similar(Rlm)
    shtns_rotation_apply_real(r, Qlm_int, Rlm_int)
    return convert_alm_norm!(Rlm, Rlm_int, cfg; to_internal=false)
end

"""
    SH_Yrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})

Rotate a real-field SH expansion around the Y-axis by angle `alpha`.
Uses Wigner-d mixing per l; dispatches to the general rotation engine.
"""
function SH_Yrotate(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, alpha::Real, Rlm::AbstractVector{<:Complex})
    r = SHTRotation(cfg.lmax, cfg.mmax)
    shtns_rotation_set_angles_ZYZ(r, 0.0, float(alpha), 0.0)
    return _apply_configured_rotation_real(cfg, r, Qlm, Rlm)
end

"""
    SH_Yrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
"""
function SH_Yrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    return SH_Yrotate(cfg, Qlm, π/2, Rlm)
end

"""
    SH_Xrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Rotate around X-axis by 90 degrees using ZYZ equivalence: Rz(-π/2)·Ry(π/2)·Rz(π/2).
"""
function SH_Xrotate90(cfg::SHTConfig, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    r = SHTRotation(cfg.lmax, cfg.mmax)
    shtns_rotation_set_angles_ZYZ(r, -π/2, π/2, π/2)
    return _apply_configured_rotation_real(cfg, r, Qlm, Rlm)
end

"""
    shtns_rotation_set_angle_axis(r::SHTRotation, theta::Real, Vx::Real, Vy::Real, Vz::Real)

Define rotation from angle-axis (theta around vector V).
Angles are set in ZYZ convention.
"""
function shtns_rotation_set_angle_axis(r::SHTRotation, theta::Real, Vx::Real, Vy::Real, Vz::Real)
    θ = float(theta)
    v = collect(float.((Vx, Vy, Vz)))
    n = hypot(v[1], hypot(v[2], v[3]))
    if n == 0
        r.α = 0.0; r.β = 0.0; r.γ = 0.0; r.conv = :ZYZ
        return nothing
    end
    kx, ky, kz = v ./ n
    c = cos(θ); s = sin(θ); t = 1 - c
    
    # Rotation matrix R = c I + s [k]_x + t k k^T
    R11 = c + t*kx*kx
    R12 = t*kx*ky - s*kz
    R13 = t*kx*kz + s*ky
    R21 = t*ky*kx + s*kz
    R22 = c + t*ky*ky
    R23 = t*ky*kz - s*kx
    R31 = t*kz*kx - s*ky
    R32 = t*kz*ky + s*kx
    R33 = c + t*kz*kz
    
    # Extract ZYZ Euler angles
    # For R = Rz(α)Ry(β)Rz(γ): R13 = cα*sβ, R23 = sα*sβ, R31 = -sβ*cγ, R32 = sβ*sγ
    β = acos(clamp(R33, -1.0, 1.0))
    if abs(sin(β)) > 1e-12
        α = atan(R23, R13)    # atan2(sα*sβ, cα*sβ) = α
        γ = atan(R32, -R31)   # atan2(sβ*sγ, sβ*cγ) = γ
    elseif R33 > 0
        # β ≈ 0: R = Rz(α)·Rz(γ) = Rz(α+γ), so R11 = cos(α+γ), R21 = sin(α+γ).
        # Only the sum is observable; fold it all into α.
        α = atan(R21, R11)
        γ = 0.0
    else
        # β ≈ π: R = Rz(α)·Ry(π)·Rz(γ). Working it out,
        #     R = [-cos(α-γ)  -sin(α-γ)   0
        #          -sin(α-γ)   cos(α-γ)   0
        #           0          0         -1]
        # so the observable combination is the DIFFERENCE α-γ, and it is read off
        # the NEGATED first column: α-γ = atan2(-R21, -R11).
        #
        # Reusing the β≈0 formula here (which is what this branch used to do for
        # both poles) silently returns a rotation about the wrong axis: a 180°
        # turn about x̂ came back as ZYZ(0, π, 0), which is exactly Ry(π) — the
        # relative error against the true Rx(π) was 1.33, and against Ry(π) it
        # was 0. Non-degenerate angles were unaffected, so this only bit exact
        # half-turns.
        α = atan(-R21, -R11)
        γ = 0.0
    end
    r.α = α; r.β = β; r.γ = γ; r.conv = :ZYZ
    return nothing
end

"""
    wigner_d_matrix!(d::AbstractMatrix{Float64}, l::Int, beta::Float64)

In-place computation of little Wigner-d matrix d^l_{m m'}(β). Writes into the
top-left (2l+1)×(2l+1) block of `d`. Caller must ensure `size(d,1) ≥ 2l+1`.
"""
function wigner_d_matrix!(d::AbstractMatrix{Float64}, l::Int, beta::Float64)
    l ≥ 0 || throw(ArgumentError("l must be ≥ 0"))
    work = Matrix{Float64}(undef, size(d))
    return _wigner_d_matrix_stable!(d, work, l, beta)
end

"""
    wigner_d_matrix!(d, l, beta, lg)

Compatibility overload for callers that supplied the former log-factorial
scratch buffer. The stable recurrence no longer consumes those values.
"""
function wigner_d_matrix!(d::AbstractMatrix{Float64}, l::Int, beta::Float64,
                          lg::AbstractVector{Float64})
    l ≥ 0 || throw(ArgumentError("l must be ≥ 0"))
    length(lg) ≥ 2l + 1 || throw(ArgumentError("lg must have length ≥ 2l+1"))
    work = Matrix{Float64}(undef, size(d))
    return _wigner_d_matrix_stable!(d, work, l, beta)
end

"""Scratch-matrix overload used by repeated rotations."""
function wigner_d_matrix!(d::AbstractMatrix{Float64}, l::Int, beta::Float64,
                          work::AbstractMatrix{Float64})
    l ≥ 0 || throw(ArgumentError("l must be ≥ 0"))
    return _wigner_d_matrix_stable!(d, work, l, beta)
end

"""
Build `dˡ(β)` by repeatedly coupling the current representation with spin 1/2.

The Clebsch–Gordan recurrence only combines bounded rotation entries with
coefficients in `[0,1]`; unlike the factorial formula, it never forms huge terms
that must cancel to produce an O(1) answer.
"""
function _wigner_d_matrix_stable!(d::AbstractMatrix{Float64},
                                  work::AbstractMatrix{Float64},
                                  l::Int, beta::Float64)
    n = 2l + 1
    size(d, 1) ≥ n && size(d, 2) ≥ n ||
        throw(DimensionMismatch("d must contain a (2l+1)×(2l+1) block"))
    size(work, 1) ≥ n && size(work, 2) ≥ n ||
        throw(DimensionMismatch("work must contain a (2l+1)×(2l+1) block"))

    d[1, 1] = 1.0
    l == 0 && return d

    cb = cos(beta / 2)
    sb = sin(beta / 2)
    # Rows/columns are ordered s=-1/2,+1/2, matching this file's convention.
    dhalf = (cb, sb, -sb, cb)
    src = d
    dest = work

    # `two_j_new` grows 0 -> 1/2 -> 1 -> ... -> l. Coupling coefficients are
    # sqrt((J ± M)/(2J)); doubled integer indices avoid half-integer arithmetic.
    for two_j_new in 1:(2l)
        two_j_old = two_j_new - 1
        denom = 2two_j_new
        @inbounds for i in 0:two_j_new
            two_m = -two_j_new + 2i
            for j in 0:two_j_new
                two_mp = -two_j_new + 2j
                acc = 0.0
                for si in 1:2
                    two_s = 2si - 3
                    two_m_old = two_m - two_s
                    abs(two_m_old) ≤ two_j_old || continue
                    old_i = (two_m_old + two_j_old) ÷ 2 + 1
                    ci = sqrt((two_j_new + two_s * two_m) / denom)
                    for sj in 1:2
                        two_sp = 2sj - 3
                        two_mp_old = two_mp - two_sp
                        abs(two_mp_old) ≤ two_j_old || continue
                        old_j = (two_mp_old + two_j_old) ÷ 2 + 1
                        cj = sqrt((two_j_new + two_sp * two_mp) / denom)
                        dh = dhalf[2(si - 1) + sj]
                        acc += ci * cj * src[old_i, old_j] * dh
                    end
                end
                dest[i + 1, j + 1] = acc
            end
        end
        src, dest = dest, src
    end

    # There are 2l (an even number of) half-steps, so the final result is in d.
    return d
end

"""
    wigner_d_matrix(l::Int, beta::Float64) -> Matrix{Float64}

Compute little Wigner-d matrix d^l_{m m'}(β) with m,m' in [-l..l], returned as a
(2l+1)×(2l+1) real matrix where index is `m+l+1, m'+l+1`.
"""
function wigner_d_matrix(l::Int, beta::Float64)
    n = 2l + 1
    d = Matrix{Float64}(undef, n, n)
    return wigner_d_matrix!(d, l, beta)
end

"""
    WignerCache(lmax::Int, β::Real) -> WignerCache

Precompute Wigner-d matrices `d^l(β)` for `l = 0:lmax` and hand them out via
[`wigner_d(cache, l)`]. Reuse across many rotations at fixed β (e.g.
time-stepping), amortizing the per-call construction cost.
"""
struct WignerCache
    β::Float64
    matrices::Vector{Matrix{Float64}}
end

function WignerCache(lmax::Int, β::Real)
    lmax ≥ 0 || throw(ArgumentError("lmax must be ≥ 0"))
    βf = float(β)
    mats = Vector{Matrix{Float64}}(undef, lmax + 1)
    for l in 0:lmax
        mats[l + 1] = wigner_d_matrix(l, βf)
    end
    return WignerCache(βf, mats)
end

"""
    wigner_d(cache::WignerCache, l::Int) -> Matrix{Float64}

Retrieve cached `d^l(β)`. Errors if `l` exceeds the cache's `lmax`.
"""
@inline function wigner_d(cache::WignerCache, l::Int)
    (0 ≤ l < length(cache.matrices)) || throw(ArgumentError("l=$l outside cache (lmax=$(length(cache.matrices) - 1))"))
    return cache.matrices[l + 1]
end

"""
    wigner_d_matrix_deriv(l::Int, beta::Float64) -> Matrix{Float64}

Derivative d/dβ of little Wigner-d matrix d^l_{m m'}(β).
"""
function wigner_d_matrix_deriv(l::Int, beta::Float64)
    l ≥ 0 || throw(ArgumentError("l must be ≥ 0"))
    n = 2l + 1
    d = wigner_d_matrix(l, beta)
    dβ = Matrix{Float64}(undef, n, n)
    # d(β)=exp(βG), where G is the real skew-symmetric y-rotation
    # generator. Therefore ḋ=Gd, which is stable and O(l²) once d is known.
    @inbounds for m in -l:l, mp in -l:l
        value = 0.0
        if m > -l
            value -= 0.5 * sqrt((l + m) * (l - m + 1)) * d[m + l, mp + l + 1]
        end
        if m < l
            value += 0.5 * sqrt((l - m) * (l + m + 1)) * d[m + l + 2, mp + l + 1]
        end
        dβ[m + l + 1, mp + l + 1] = value
    end
    return dβ
end

"""
    shtns_rotation_create(lmax::Integer, mmax::Integer, norm::Integer) -> SHTRotation
"""
function shtns_rotation_create(lmax::Integer, mmax::Integer, norm::Integer)
    norm == 0 || throw(ArgumentError("only orthonormal normalization supported"))
    return SHTRotation(Int(lmax), Int(mmax))
end

"""shtns_rotation_destroy(r::SHTRotation)"""
shtns_rotation_destroy(::SHTRotation) = nothing

"""shtns_rotation_set_angles_ZYZ(r, alpha, beta, gamma)"""
function shtns_rotation_set_angles_ZYZ(r::SHTRotation, alpha::Real, beta::Real, gamma::Real)
    r.α = float(alpha); r.β = float(beta); r.γ = float(gamma); r.conv = :ZYZ; return nothing
end

"""shtns_rotation_set_angles_ZXZ(r, alpha, beta, gamma)"""
function shtns_rotation_set_angles_ZXZ(r::SHTRotation, alpha::Real, beta::Real, gamma::Real)
    r.α = float(alpha); r.β = float(beta); r.γ = float(gamma); r.conv = :ZXZ; return nothing
end

"""Return the equivalent ZYZ angles consumed by the Wigner-d engine."""
@inline function _rotation_zyz_angles(r::SHTRotation)
    if r.conv === :ZYZ
        return r.α, r.β, r.γ
    elseif r.conv === :ZXZ
        # Rx(β) = Rz(-π/2) Ry(β) Rz(π/2).
        return r.α - π/2, r.β, r.γ + π/2
    end
    throw(ArgumentError("unsupported Euler convention $(r.conv)"))
end

"""
    shtns_rotation_wigner_d_matrix(r::SHTRotation, l::Integer, mx::AbstractVector{<:Real}) -> Int

Fill `mx` (length ≥ (2l+1)^2) with d^l in row-major order. Returns size 2l+1.
"""
function shtns_rotation_wigner_d_matrix(r::SHTRotation, l::Integer, mx::AbstractVector{<:Real})
    l = Int(l)
    n = 2l + 1
    length(mx) ≥ n*n || throw(DimensionMismatch("mx must have length ≥ (2l+1)^2"))
    d = wigner_d_matrix(l, r.β)
    @inbounds for i in 1:n, j in 1:n
        mx[(i-1)*n + j] = d[i, j]
    end
    return n
end

"""
    _lmcplx_ybasis_signs(lmax, mmax) -> Vector{Float64}

The diagonal ε relating this package's LM_cplx layout to the Y_l^m basis that
the Wigner-d engine works in: `ε_m = (-1)^m` for `m < 0`, `1` otherwise.

Both signs of m share the SAME P̄_l^{|m|} row in this layout, so a real field
satisfies `a_{-m} = conj(a_m)` with no CS factor — unlike the Y_l^m convention,
whose rule is `a_{-m} = (-1)^m conj(a_m)`. ε is real and self-inverse, so
`ε ∘ P ∘ ε` is the rotation expressed in the packed layout, and because
`|ε x| = |x|` any norm-based loss is unchanged by it (which is why the angle
gradients only need ε applied to their inputs).
"""
function _lmcplx_ybasis_signs(lmax::Integer, mmax::Integer)
    lmax = Int(lmax); mmax = Int(mmax)
    v = ones(Float64, nlm_cplx_calc(lmax, mmax, 1))
    for l in 0:lmax, m in -min(l, mmax):-1
        isodd(m) && (v[LM_cplx_index(lmax, mmax, l, m) + 1] = -1.0)
    end
    return v
end

"""
    shtns_rotation_apply_cplx(r::SHTRotation, Zlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Apply rotation with Euler angles (ZYZ/ZXZ) to complex SH coefficients in LM_cplx packing (mres==1).
"""
function shtns_rotation_apply_cplx(r::SHTRotation, Zlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    r.lmax ≥ 0 || return Rlm
    length(Zlm) == length(Rlm) || throw(DimensionMismatch("Zlm and Rlm length mismatch"))
    mres = 1
    expected = nlm_cplx_calc(r.lmax, r.mmax, mres)
    length(Zlm) == expected || throw(DimensionMismatch("LM_cplx size mismatch"))
    α, β, γ = _rotation_zyz_angles(r)
    _require_full_m_range(r, β)

    # Pre-allocate working arrays at maximum size to avoid per-l allocations
    nmax = 2 * r.lmax + 1
    b = Vector{ComplexF64}(undef, nmax)
    c = Vector{ComplexF64}(undef, nmax)
    dl = Matrix{Float64}(undef, nmax, nmax)  # Reusable Wigner d-matrix buffer
    dwork = similar(dl)

    # Apply R = diag(e^{-i m α}) * d^l(β) * diag(e^{-i m γ}) for each l
    for l in 0:r.lmax
        mm = min(l, r.mmax)
        n = 2l + 1
        # Build input vector b_m' = e^{-i m' γ} A_{m'} for m' in [-mm..mm].
        #
        # The Wigner-d machinery below is written for the Y_l^m basis, whose
        # Hermitian rule is a_{-m} = (-1)^m conj(a_m). This package's LM_cplx
        # layout is NOT that basis: both signs of m share the SAME P̄_l^{|m|} row,
        # so a real field there satisfies a_{-m} = conj(a_m) with no (-1)^m (see
        # `synthesis_packed_cplx` / `SH_to_lat_cplx`). The two differ by the
        # diagonal ε_m = (-1)^m on m < 0 only, applied on the way in and undone
        # on the way out.
        #
        # Without it this function was a valid rotation in the wrong basis: it
        # stayed unitary (per-l norm preserved), so nothing caught it, but
        # rotating a REAL field produced a complex one — a_{l,0} came back with a
        # non-zero imaginary part. Verified against a spatial-rotation reference.
        # ε is diagonal and commutes with the α/γ phase diagonals, so pure
        # Z-rotations are unaffected.
        fill!(view(b, 1:n), zero(ComplexF64))
        for mp in -mm:mm
            idx = LM_cplx_index(r.lmax, r.mmax, l, mp) + 1
            εp = (mp < 0 && isodd(mp)) ? -1.0 : 1.0
            b[mp + l + 1] = (εp * Zlm[idx]) * cis(-mp * γ)
        end
        # Multiply with d^l(β) — computed in-place into pre-allocated buffer
        wigner_d_matrix!(dl, l, β, dwork)
        fill!(view(c, 1:n), zero(ComplexF64))
        # c_m = sum_{m'} d_{m m'} b_{m'}
        for mi in -l:l
            acc = zero(ComplexF64)
            for mp in -l:l
                acc += dl[mi + l + 1, mp + l + 1] * b[mp + l + 1]
            end
            c[mi + l + 1] = acc
        end
        # Apply phase e^{-i m α} and write back only for allowed |m| ≤ mm,
        # undoing the ε basis change applied when b was built.
        for m in -mm:mm
            idx = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            εm = (m < 0 && isodd(m)) ? -1.0 : 1.0
            Rlm[idx] = εm * (c[m + l + 1] * cis(-m * α))
        end
    end
    return Rlm
end

"""
    _require_full_m_range(r::SHTRotation, β::Real)

Reject an order-mixing rotation on a layout that cannot hold every order it
produces.

A Wigner-d rotation through a general `β` couples `Y_l^m` to every `Y_l^{m'}`
with `|m'| ≤ l`. If the storage stops at `mmax < lmax`, the `|m'| > mmax`
components have nowhere to go and were silently dropped — measured at
`lmax = 8`, that quietly discarded **14.8 %** of the field's energy at
`mmax = 5` and **24.0 %** at `mmax = 3`, with no error and no warning.

The two degenerate angles are exempt because their `d^l` is not order-mixing:
`β ≡ 0` is diagonal, and `β ≡ π` is anti-diagonal (`m' = -m`), so `|m'| = |m|`
and a truncated layout still holds the result. That keeps pure Z-rotations
expressed as `ZYZ(α, 0, γ)` working at any `mmax`.

Mirrors the `mres > 1` restriction stated by `dist_SH_Yrotate` and the packed
distributed rotations, for the same reason.
"""
function _require_full_m_range(r::SHTRotation, β::Real)
    r.mmax >= r.lmax && return nothing
    abs(sin(float(β))) <= 1e-12 && return nothing   # β ≡ 0 (mod π): no m mixing
    throw(ArgumentError(
        "rotation with β=$(β) mixes azimuthal orders, but this configuration " *
        "stores only m ≤ mmax=$(r.mmax) with lmax=$(r.lmax); the |m| > mmax " *
        "components such a rotation generates cannot be represented and would " *
        "be silently discarded. Use a configuration with mmax == lmax for " *
        "Y/X rotations and general Euler angles. Pure Z-rotations (β ≡ 0 mod π) " *
        "are unaffected and still work at any mmax."))
end

"""
    _rotation_packed_length_check(v, expected, name)

Length guard for the packed rotation inputs, with an `mres`-aware message.
"""
function _rotation_packed_length_check(v::AbstractVector, expected::Int,
                                       name::AbstractString, r::SHTRotation)
    length(v) == expected && return nothing
    throw(DimensionMismatch(
        "$name has length $(length(v)), expected $expected for the packed (mres=1) " *
        "layout at lmax=$(r.lmax), mmax=$(r.mmax). A Y/X rotation mixes azimuthal " *
        "orders, so it cannot be represented in an mres-strided layout at all — an " *
        "mres>1 config produces a shorter packed vector and lands here. Use mres=1 " *
        "for rotations other than SH_Zrotate."))
end

"""
    shtns_rotation_apply_real(r::SHTRotation, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})

Apply rotation to real-field SH coefficients in packed LM layout (m ≥ 0). Requires `mres==1`.
"""
function shtns_rotation_apply_real(r::SHTRotation, Qlm::AbstractVector{<:Complex}, Rlm::AbstractVector{<:Complex})
    expected = nlm_calc(r.lmax, r.mmax, 1)
    # A length mismatch here is almost always an `mres > 1` config reaching a
    # rotation that mixes orders, which no mres-strided layout can represent.
    # Say that, rather than reporting a bare size mismatch the caller has to
    # reverse-engineer. (`dist_SH_Yrotate` and the packed distributed rotations
    # state the same restriction up front.)
    _rotation_packed_length_check(Qlm, expected, "Qlm", r)
    _rotation_packed_length_check(Rlm, expected, "Rlm", r)
    # Build LM_cplx array Zlm from real-packed Qlm using this layout's Hermitian
    # rule a_{-m} = conj(a_m) — NO (-1)^m; see the note at the write below and
    # `_lmcplx_ybasis_signs` for why the CS factor lives inside apply_cplx now.
    Z = Vector{ComplexF64}(undef, nlm_cplx_calc(r.lmax, r.mmax, 1))
    
    # initialize zeros
    fill!(Z, zero(ComplexF64))
    for l in 0:r.lmax
        mm = min(l, r.mmax)
        # m = 0
        idxp = LM_index(r.lmax, 1, l, 0) + 1
        idxc = LM_cplx_index(r.lmax, r.mmax, l, 0) + 1
        Z[idxc] = Qlm[idxp]
        for m in 1:mm
            idxp = LM_index(r.lmax, 1, l, m) + 1
            idxc_p = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            idxc_n = LM_cplx_index(r.lmax, r.mmax, l, -m) + 1
            Am = Qlm[idxp]
            Z[idxc_p] = Am
            # LM_cplx here is the P̄_l^{|m|} layout, whose real-field rule carries
            # NO (-1)^m; `shtns_rotation_apply_cplx` now converts to/from the
            # Y_l^m basis itself, so this must not pre-apply that factor too.
            Z[idxc_n] = conj(Am)
        end
    end
    R = similar(Z)
    shtns_rotation_apply_cplx(r, Z, R)
    # Pack back to positive-m layout
    for l in 0:r.lmax
        mm = min(l, r.mmax)
        idxp0 = LM_index(r.lmax, 1, l, 0) + 1
        idxc0 = LM_cplx_index(r.lmax, r.mmax, l, 0) + 1
        Rlm[idxp0] = R[idxc0]
        for m in 1:mm
            idxp = LM_index(r.lmax, 1, l, m) + 1
            idxc = LM_cplx_index(r.lmax, r.mmax, l, m) + 1
            Rlm[idxp] = R[idxc]
        end
    end
    return Rlm
end
