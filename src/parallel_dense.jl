"""
Dense (non-distributed) helpers that are used by distributed APIs and examples.
These do not depend on MPI/Pencil packages and live in src/ to keep extensions lean.
"""

"""
    dist_apply_laplacian!(cfg, Alm::AbstractMatrix)

In-place multiply by -l(l+1) for dense (l×m) coefficients.
"""
function dist_apply_laplacian!(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Alm,1)==lmax+1 && size(Alm,2)==mmax+1 || throw(DimensionMismatch("Alm dims"))
    @inbounds for m in 0:mmax, l in m:lmax
        Alm[l+1, m+1] *= -(l*(l+1))
    end
    return Alm
end

"""
    dist_SH_Zrotate(cfg, Alm::AbstractMatrix, alpha::Real, Rlm::AbstractMatrix)

Z-rotation by alpha (radians) on dense (l×m) coefficients; Rlm = e^{-imα} Alm.
"""
function dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix, alpha::Real, Rlm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Alm,1)==lmax+1 && size(Alm,2)==mmax+1 || throw(DimensionMismatch("Alm dims"))
    size(Rlm,1)==lmax+1 && size(Rlm,2)==mmax+1 || throw(DimensionMismatch("Rlm dims"))
    @inbounds for m in 0:mmax
        phase = cis(-m*alpha)
        for l in m:lmax
            Rlm[l+1, m+1] = phase * Alm[l+1, m+1]
        end
    end
    return Rlm
end

"""
    dist_SH_Yrotate(cfg, Alm::AbstractMatrix, beta::Real, Rlm::AbstractMatrix)

Gather/apply/unpack rotation on dense (l×m): packs to LM vector, applies SH_Yrotate, unpacks.
Useful for validation and small problems; distributed variants should prefer per-l Allgatherv.
"""
function dist_SH_Yrotate(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix, beta::Real, Rlm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Alm,1)==lmax+1 && size(Alm,2)==mmax+1 || throw(DimensionMismatch("Alm dims"))
    size(Rlm,1)==lmax+1 && size(Rlm,2)==mmax+1 || throw(DimensionMismatch("Rlm dims"))
    # A Y-rotation mixes orders, so it cannot be expressed in an mres-strided
    # layout at all: rotating an mres=2 field produces m=1 components with nowhere
    # to live. `shtns_rotation_apply_real` states the same restriction. Say so
    # here rather than failing downstream with `m must be a multiple of mres`
    # (what the un-strided loop used to do) or `LM packed size mismatch`.
    cfg.mres == 1 || throw(ArgumentError("dist_SH_Yrotate requires mres==1 (got mres=$(cfg.mres)); " *
                                         "a Y-rotation mixes orders and cannot be represented in an mres-strided layout"))
    # Canonical packed↔dense pair (src/layout.jl) instead of an open-coded loop.
    Q = SHTnsKit.pack_lm(cfg, complex(float(eltype(Alm))).(Alm))
    R = similar(Q)
    SHTnsKit.SH_Yrotate(cfg, Q, beta, R)
    # Orders absent from the packed layout have no rotated value to write back;
    # zero them rather than leaving whatever the caller's buffer held.
    fill!(Rlm, zero(eltype(Rlm)))
    SHTnsKit.unpack_lm!(Rlm, cfg, R)
    return Rlm
end

"""
    dist_SH_mul_mx!(cfg, mx, Alm, Rlm)

Apply 3-diagonal l±1 operator to dense (l×m) Alm into Rlm. No communication.
`mx` is 2*nlm packed coefficients as in mul_ct_matrix/st_dt_matrix.
"""
function dist_SH_mul_mx!(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real}, Alm::AbstractMatrix, Rlm::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Alm,1)==lmax+1 && size(Alm,2)==mmax+1 || throw(DimensionMismatch("Alm dims"))
    size(Rlm,1)==lmax+1 && size(Rlm,2)==mmax+1 || throw(DimensionMismatch("Rlm dims"))
    length(mx) == 2*cfg.nlm || throw(DimensionMismatch("mx length must be 2*nlm=$(2*cfg.nlm)"))
    fill!(Rlm, zero(eltype(Rlm)))
    # Key insight: mx stores coefficients describing how (l,m) contributes to its neighbors.
    # For R_l^m = (Op * A)_lm, we need coefficients from neighbors that contribute TO (l,m):
    # - Y_{l-1}^m contributes to Y_l^m via b_{l-1}^m (the upward coefficient from l-1)
    # - Y_{l+1}^m contributes to Y_l^m via a_{l+1}^m (the downward coefficient from l+1)
    @inbounds for m in 0:mmax, l in m:lmax
        # `LM_index` throws unless m is a multiple of mres, so stride like every
        # other packed-index loop in the package (`analysis_packed`,
        # `synthesis_packed`, `pack_lm!`). Without this the whole dense operator
        # path died on the first m=1 for any mres>1 config.
        (m % cfg.mres == 0) || continue
        acc = zero(promote_type(eltype(Alm), eltype(Rlm), complex(eltype(mx))))  # eltype-preserving accumulator (AD-safe)
        # Contribution from lower degree neighbor Y_{l-1}^m
        if l > m && l > 0
            idx_prev = SHTnsKit.LM_index(lmax, cfg.mres, l-1, m)
            c_from_below = mx[2*idx_prev + 2]  # b_{l-1}^m: upward coeff from neighbor
            acc += c_from_below * Alm[l, m+1]
        end
        # Contribution from higher degree neighbor Y_{l+1}^m
        if l < lmax
            idx_next = SHTnsKit.LM_index(lmax, cfg.mres, l+1, m)
            c_from_above = mx[2*idx_next + 1]  # a_{l+1}^m: downward coeff from neighbor
            acc += c_from_above * Alm[l+2, m+1]
        end
        Rlm[l+1, m+1] = acc
    end
    return Rlm
end
