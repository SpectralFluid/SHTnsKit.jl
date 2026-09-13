#=
================================================================================
ParallelTransposeTransforms.jl — Transpose-based distributed SHT (Task 2)
================================================================================

DistTransposePlan wraps a PencilFFTs plan whose FORWARD pass performs

    rFFT(φ)  +  internal θ↔m transpose

landing data in **θ-LOCAL / m-DISTRIBUTED** layout, which is exactly what the
Legendre stage needs (no extra communication before Legendre contraction).

Data-flow overview
------------------
Analysis (spatial → spectral):
  PencilArray[φ_local, θ_dist, lev]   ← allocate_spatial(plan)
    → mul!(F_buf, fft_plan, f_spatial) → F_buf[m_dist, θ_local, lev]
    → Legendre contraction per local m → Alm[l_local, m_local, lev]

Synthesis (spectral → spatial):
  (reverse of the above, Task 3+)

Key invariants
--------------
* `plan.m_local[mi]` is the active 0-based global m index this rank owns and
  `plan.m_slots[mi]` is its physical local rFFT/spectral column.
* `plan.NP[mi]` is the (lmax+1, nlat) matrix P̄_l^{m_local[mi]}(cos θ_i).
* The m-distribution of `F_buf` (PencilFFTs output, decomposed on dim 1) is
  guaranteed to match the m-distribution of `spectral_pencil` (Alm, decomposed
  on dim 2) because the spectral pencil is sized to the SAME number of rFFT bins
  (`nbin = nlon÷2+1`) and uses the same block-distribution formula.  This holds
  for both canonical (`nlon = 2*mmax+1`, `nbin = mmax+1`) and dealiased
  (`nlon > 2*mmax+1`, `nbin > mmax+1`) grids; in the dealiased case the owned
  coefficients are the columns selected by `m_slots`; excluded `mres` orders
  and high bins (`m > mmax`) are unused/zero. Asserted in the constructor.
================================================================================
=#

import LinearAlgebra: mul!, ldiv!

# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------

"""
    DistTransposePlan

Plan for a transpose-based distributed spherical harmonic transform.

Constructed via `DistTransposePlan(cfg; comm, nlev, use_rfft, with_vector)`.

# Fields
- `cfg`             : SHTConfig (replicated across ranks)
- `cfg_fingerprint` : snapshot used to reject stale cached tables after cfg mutation
- `nlat`, `nlon`, `lmax`, `mmax`, `nlev` : grid / spectral dimensions
- `comm`            : MPI communicator
- `fft_plan`        : `PencilFFTPlan((nlon, nlat), …)` — rFFT(φ) + internal transpose
- `F_buf`           : pre-allocated output of `fft_plan` (m-distributed, θ-local)
- `F_buf2`          : second pre-allocated output buffer for the second vector component
- `spectral_pencil` : `Pencil` for Alm arrays, global `(lmax+1, nbin)` where
                       `nbin = nlon÷2+1` (rFFT bins), dim-2 (m/bin) decomposed to
                       match the FFT output bin-for-bin. Owned coefficients are the
                       columns mapped by `m_slots`.
- `m_local`         : active 0-based global m indices owned by this rank
- `m_slots`         : physical local rFFT/spectral columns for `m_local`
- `NP`              : `NP[mi]` = `(lmax+1, nlat)` matrix of P̄_l^{m_local[mi]}(cos θ_i)
- `dP`              : `dP[mi]` = `(lmax+1, nlat)` matrix of dP̄_l^m/dθ at each latitude
- `Pos`             : `Pos[mi]` = `(lmax+1, nlat)` matrix of P̄_l^m/sinθ at each latitude

Coefficients crossing this plan's API use `cfg`'s normalization, real-basis, and
Condon–Shortley phase conventions, like the serial and other distributed
transforms. The Legendre tables remain orthonormal+CS internally; the plan
converts coefficients at its analysis and synthesis boundaries.
"""
struct DistTransposePlan{TP, TFB, TSP}
    cfg           :: SHTnsKit.SHTConfig
    cfg_fingerprint :: UInt
    nlat          :: Int
    nlon          :: Int
    lmax          :: Int
    mmax          :: Int
    nlev          :: Int
    comm          :: MPI.Comm
    fft_plan      :: TP                      # PencilFFTPlan
    F_buf         :: TFB                     # allocate_output(fft_plan): m-dist/θ-local + extra (nlev,)
    F_buf2        :: TFB                     # second buffer for vector component
    spectral_pencil :: TSP                   # Pencil for Alm: global (lmax+1,mmax+1), m-dist on dim2
    m_local       :: Vector{Int}             # 0-based global m indices this rank owns
    m_slots       :: Vector{Int}             # physical local columns corresponding to m_local
    NP            :: Vector{Matrix{Float64}} # NP[mi] = (lmax+1, nlat) normalized Legendre table
    dP            :: Vector{Matrix{Float64}} # dP[mi] = (lmax+1, nlat) dP̄_l^m/dθ table
    Pos           :: Vector{Matrix{Float64}} # Pos[mi] = (lmax+1, nlat) P̄_l^m/sinθ table
    with_vector   :: Bool                    # whether dP/Pos were built (sphtor/qst capable)
end


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

function SHTnsKit.DistTransposePlan(
        cfg::SHTnsKit.SHTConfig;
        comm        :: MPI.Comm = MPI.COMM_WORLD,
        nlev        :: Int      = 1,
        use_rfft    :: Bool     = true,
        with_vector :: Bool     = true)

    # PencilFFTPlan construction is collective. Detect a rank-divergent cfg
    # first so every rank raises together instead of constructing incompatible
    # FFT/spectral pencils (or hanging in their setup collectives).
    _validate_cfg_replicated(cfg, comm)
    _validate_replicated_call_signature(
        comm, "DistTransposePlan", (nlev, use_rfft, with_vector),
    )
    nlev > 0 || throw(ArgumentError("DistTransposePlan requires nlev > 0"))
    cfg_fingerprint = _cfg_fingerprint(cfg)

    nlat = cfg.nlat
    nlon = cfg.nlon
    lmax = cfg.lmax
    mmax = cfg.mmax

    use_rfft || error("DistTransposePlan currently requires use_rfft=true")
    # The transpose Legendre stages do not apply the Robert-form sinθ scaling that
    # the cfg-form paths do; guard rather than silently return wrong results.
    cfg.robert_form && error("DistTransposePlan does not support robert_form grids; use the cfg-form dist_analysis/dist_synthesis instead")

    # 1. Build the PencilFFTs plan: global (φ, θ) = (nlon, nlat),
    #    rFFT on dim1 (φ), NoTransform on dim2 (θ).
    #    extra_dims=(nlev,) carries radial levels as a trailing LOCAL dimension.
    proc     = (MPI.Comm_size(comm),)
    fft_plan = PencilFFTPlan(
        (nlon, nlat),
        (Transforms.RFFT(), Transforms.NoTransform()),
        proc,
        comm;
        extra_dims = (nlev,),
    )

    # 2. Allocate the spectral output buffer: shape (m_local, θ_ALL, nlev),
    #    decomposed on dim1 (m).
    F_buf  = allocate_output(fft_plan)
    F_buf2 = allocate_output(fft_plan)   # second buffer for vector (sphtor) component

    # 3. Determine which m values this rank owns (from F_buf's pencil).
    #    dim1 of F_buf's pencil is the m dimension (after the internal FFT+transpose).
    #    The rFFT output has nbin = nlon÷2+1 Fourier bins (0..nlon÷2), distributed
    #    across ranks.  Each rank owns a CONTIGUOUS ASCENDING block of bins.
    nbin   = nlon ÷ 2 + 1
    mr     = range_local(pencil(F_buf))     # tuple of ranges; mr[1] = local bin-range (1-based)
    # Global bin index (0-based) = 1-based pencil index − 1
    all_m_0based = collect(mr[1]) .- 1
    # Keep only orders represented by the configured m stride. `keep` is also the
    # physical local column mapping: with mres > 1 the active bins are not a prefix
    # (for example local bins 5:9 contain active m=6,8 at columns 2,4).
    keep   = findall(m -> 0 <= m <= mmax && m % cfg.mres == 0, all_m_0based)
    m_local = all_m_0based[keep]
    m_slots = keep

    # 4. Build the spectral Pencil for Alm sized to the rFFT BINS: global
    #    (lmax+1, nbin), decomposed on dim2 (the m/bin axis).  Sizing dim2 to nbin
    #    (instead of an independent mmax+1 block-split) GUARANTEES that dim2 of this
    #    pencil uses the SAME block distribution as dim1 of F_buf's pencil — so on
    #    every rank, the local columns of Alm correspond bin-for-bin to F_buf's local
    #    bins; m_slots maps active configured orders onto those physical columns.
    #    For the canonical grid nbin == mmax+1 so this is bitwise-identical to the old
    #    (lmax+1, mmax+1) pencil; for dealiased grids the extra (m > mmax) columns are
    #    unused/zero and the irFFT zero-pads them on synthesis.
    spectral_pencil = Pencil((lmax + 1, nbin), (2,), comm)

    # 5. Assert alignment: active m-columns of spectral_pencil (dim2, 0-based,
    #    restricted by mmax/mres) must equal m_local. This catches any future
    #    divergence in distribution strategies between the FFT and spectral pencils.
    sp_mr    = range_local(spectral_pencil)       # (l-range, m-range) 1-based
    sp_m_0   = collect(sp_mr[2]) .- 1             # 0-based m from spectral_pencil
    sp_m_kept = filter(m -> 0 <= m <= mmax && m % cfg.mres == 0, sp_m_0)
    if sp_m_kept != m_local
        error("m-distribution mismatch on rank $(MPI.Comm_rank(comm)): " *
              "F_buf m_local=$m_local  ≠  spectral_pencil m(≤mmax)=$sp_m_kept")
    end

    # 6. Pre-compute normalized Legendre tables for each local m.
    #    NP[mi][l+1, i] = P̄_l^{m_local[mi]}(cos θ_i)  for i=1..nlat, l=0..lmax
    #    When with_vector=true also compute dP (dP̄/dθ) and Pos (P̄/sinθ) tables.
    P_buf   = Vector{Float64}(undef, lmax + 1)
    dP_buf  = Vector{Float64}(undef, lmax + 1)
    Pos_buf = Vector{Float64}(undef, lmax + 1)
    # Only allocate the vector (dP/Pos) tables when requested; otherwise leave them
    # empty so a vector transform on a scalar-only plan errors cleanly instead of
    # reading the previously-`undef` (garbage) matrices.
    NP  = Vector{Matrix{Float64}}(undef, length(m_local))
    dP  = with_vector ? Vector{Matrix{Float64}}(undef, length(m_local)) : Matrix{Float64}[]
    Pos = with_vector ? Vector{Matrix{Float64}}(undef, length(m_local)) : Matrix{Float64}[]
    for (mi, m) in enumerate(m_local)
        tbl_NP = Matrix{Float64}(undef, lmax + 1, nlat)
        if with_vector
            tbl_dP  = Matrix{Float64}(undef, lmax + 1, nlat)
            tbl_Pos = Matrix{Float64}(undef, lmax + 1, nlat)
            for i in 1:nlat
                SHTnsKit.Plm_norm_dPdtheta_over_sinth_row!(
                    P_buf, dP_buf, Pos_buf, cfg.x[i], lmax, m)
                @inbounds for l in 0:lmax
                    tbl_NP[l+1, i]  = P_buf[l+1]
                    tbl_dP[l+1, i]  = dP_buf[l+1]
                    tbl_Pos[l+1, i] = Pos_buf[l+1]
                end
            end
            dP[mi]  = tbl_dP
            Pos[mi] = tbl_Pos
        else
            for i in 1:nlat
                SHTnsKit.Plm_norm_row!(P_buf, cfg.x[i], lmax, m)
                @inbounds for l in 0:lmax
                    tbl_NP[l+1, i] = P_buf[l+1]
                end
            end
        end
        NP[mi] = tbl_NP
    end

    return DistTransposePlan(
        cfg, cfg_fingerprint, nlat, nlon, lmax, mmax, nlev, comm,
        fft_plan, F_buf, F_buf2, spectral_pencil,
        m_local, m_slots, NP, dP, Pos, with_vector,
    )
end

# Externalize only the meaningful, locally owned m columns after analysis. A
# dealiased spectral pencil may also contain local rFFT-bin columns with
# m > mmax; those columns remain zero and are ignored by every Legendre stage.
function _externalize_local_coefficients!(dest, plan::DistTransposePlan)
    SHTnsKit._uses_canonical_convention(plan.cfg) && return dest

    scales = SHTnsKit._ensure_norm_scale_matrix!(plan.cfg)
    @inbounds for lev in axes(dest, 3)
        for (mi, m) in enumerate(plan.m_local)
            slot = plan.m_slots[mi]
            for l in m:plan.lmax
                scale = scales[l + 1, m + 1]
                dest[l + 1, slot, lev] /= scale
            end
        end
    end
    return dest
end

# ---------------------------------------------------------------------------
# Collective operand preflight
# ---------------------------------------------------------------------------

"""Return the exact spatial Pencil used to construct the FFT plan."""
@inline _transpose_spatial_pencil(plan::DistTransposePlan) =
    PencilFFTs.pencil_input(plan.fft_plan)

@inline function _transpose_operand_matches(
        plan::DistTransposePlan, A::PencilArray, expected_pencil,
        expected_global::Tuple, expected_parent::Tuple, expected_eltype::Type)
    comm_ok = _communicators_congruent(plan.comm, communicator(A))
    return comm_ok &&
           pencil(A) === expected_pencil &&
           PencilArrays.size_global(A) == expected_global &&
           PencilArrays.range_local(pencil(A)) ==
               PencilArrays.range_local(expected_pencil) &&
           PencilArrays.permutation(A) ==
               PencilArrays.permutation(expected_pencil) &&
           size(parent(A)) == expected_parent &&
           eltype(A) === expected_eltype
end

@inline function _transpose_outputs_distinct(arrays::Tuple)
    for j in 2:length(arrays), i in 1:(j - 1)
        Base.mightalias(parent(arrays[i]), parent(arrays[j])) && return false
    end
    return true
end

"""
    _require_transpose_operands(plan, operation, spatial, spectral,
                                distinct_spatial, distinct_spectral,
                                needs_vector)

Collectively validate every operand of one public transpose operation before
the first FFT collective or direct `parent` indexing.  The reduction is always
performed on `plan.comm`, never on an operand communicator: even a malformed
`COMM_SELF` operand therefore produces one rank-symmetric failure instead of
splitting ranks across different collectives.

PencilFFTs itself requires its input/output Pencil objects by identity.  Apply
the same exact-plan contract to coefficient pencils, in addition to checking
global logical dimensions (including `nlev`), local parent layout, permutation,
communicator, and element type.
"""
function _require_transpose_operands(
        plan::DistTransposePlan, operation::AbstractString,
        spatial::Tuple, spectral::Tuple,
        distinct_spatial::Tuple, distinct_spectral::Tuple,
        needs_vector::Bool)
    spatial_pencil = _transpose_spatial_pencil(plan)
    spectral_pencil = plan.spectral_pencil
    spatial_global = (plan.nlon, plan.nlat, plan.nlev)
    spectral_global = (plan.lmax + 1, plan.nlon ÷ 2 + 1, plan.nlev)
    spatial_parent = (
        PencilArrays.size_local(
            spatial_pencil, PencilArrays.MemoryOrder())...,
        plan.nlev,
    )
    spectral_parent = (
        PencilArrays.size_local(
            spectral_pencil, PencilArrays.MemoryOrder())...,
        plan.nlev,
    )
    spatial_eltype = Transforms.eltype_input(plan.fft_plan)
    spectral_eltype = Transforms.eltype_output(plan.fft_plan)

    # Also reject a mutated plan configuration collectively. The cached FFT and
    # Legendre objects were built for these immutable dimensions.
    local_ok = (!needs_vector || plan.with_vector) &&
               _cfg_fingerprint(plan.cfg) == plan.cfg_fingerprint &&
               plan.cfg.nlat == plan.nlat &&
               plan.cfg.nlon == plan.nlon &&
               plan.cfg.lmax == plan.lmax &&
               plan.cfg.mmax == plan.mmax &&
               _communicators_congruent(
                   plan.comm, PencilArrays.get_comm(spatial_pencil)) &&
               _communicators_congruent(
                   plan.comm, PencilArrays.get_comm(spectral_pencil))

    for A in spatial
        local_ok &= _transpose_operand_matches(
            plan, A, spatial_pencil, spatial_global, spatial_parent,
            spatial_eltype)
    end
    for A in spectral
        local_ok &= _transpose_operand_matches(
            plan, A, spectral_pencil, spectral_global, spectral_parent,
            spectral_eltype)
    end
    local_ok &= _transpose_outputs_distinct(distinct_spatial)
    local_ok &= _transpose_outputs_distinct(distinct_spectral)

    all_ok = MPI.Allreduce(local_ok, &, plan.comm)
    all_ok || throw(ArgumentError(
        "$operation operands must use this DistTransposePlan's exact spatial " *
        "and spectral pencils, congruent communicator, configured nlev, local " *
        "parent layout/permutation, and element types; output arrays must not alias",
    ))
    return nothing
end

# ---------------------------------------------------------------------------
# Analysis: spatial → spectral
# ---------------------------------------------------------------------------

"""
    dist_analysis!(plan::DistTransposePlan, Alm::PencilArray, f::PencilArray) -> Alm

Distributed scalar spherical harmonic analysis using the transpose approach.

Takes a spatial field `f` (global `(nlon, nlat)`, θ-distributed, produced by
`allocate_spatial(plan)`) and writes spherical harmonic coefficients into `Alm`
(global `(lmax+1, mmax+1)`, m-distributed, produced by `allocate_spectral(plan)`).

The forward pass is:
1. `mul!(F_buf, fft_plan, f)` — rFFT(φ) + internal pencil transpose → `F_buf[mi, θ, lev]`
   where θ is now fully local on every rank and m is distributed across ranks.
2. Legendre contraction per local m:
   `Alm[l+1, mi, lev] = Σ_i w[i] · NP[mi][l+1, i] · cphi · F[mi, i, lev]`
"""
function SHTnsKit.dist_analysis!(plan::DistTransposePlan, Alm::PencilArray, f::PencilArray)
    _require_transpose_operands(
        plan, "dist_analysis!", (f,), (Alm,), (), (Alm,), false)
    return _dist_transpose_analysis_unchecked!(plan, Alm, f)
end

function _dist_transpose_analysis_unchecked!(
        plan::DistTransposePlan, Alm::PencilArray, f::PencilArray)
    # Step 1: rFFT(φ) + internal pencil transpose.
    # After mul!, F_buf has logical dims (m, θ) with permutation (2,1), so the
    # physical parent storage order is (θ, m, lev):
    #   parent(F_buf)[i, mi, lev]   where i=theta index, mi=m-slot index
    # This is because PencilFFTs outputs F with Permutation(2,1) (physical dim1
    # = logical dim2 = θ, physical dim2 = logical dim1 = m).
    # Single FFT + Alltoall collective for the ENTIRE nlev batch.
    # This is the key amortization: one MPI collective serves all radial levels,
    # so cost is O(1) in nlev rather than O(nlev).
    mul!(plan.F_buf, plan.fft_plan, f)

    F = parent(plan.F_buf)   # (nlat, n_m_local, nlev)  — physical storage (θ, m, lev)
    A = parent(Alm)          # (lmax+1, n_m_local, nlev)

    fill!(A, zero(eltype(A)))

    w       = plan.cfg.w
    scaleφ  = SHTnsKit._analysis_phi_scale(plan.cfg)   # 2π/nlon under :dft; inverts synthesis under any phi_scale
    lmax    = plan.lmax
    nlat    = plan.nlat
    nlev    = plan.nlev

    # Step 2: Legendre contraction.
    # Access pattern: NP[mi] is (lmax+1, nlat) column-major; iterating i in the
    # inner loop walks NP columns sequentially. F[i, mi, lev] is also column-major
    # friendly with i as the fast index (dim1 of parent).
    @inbounds for lev in 1:nlev
        for (mi, m) in enumerate(plan.m_local)
            slot = plan.m_slots[mi]
            NP_mi = plan.NP[mi]          # (lmax+1, nlat) matrix for this m
            for i in 1:nlat
                wi_cphi_Fi = w[i] * scaleφ * F[i, slot, lev]
                for l in m:lmax
                    A[l+1, slot, lev] += NP_mi[l+1, i] * wi_cphi_Fi
                end
            end
        end
    end
    _externalize_local_coefficients!(A, plan)
    return Alm
end

# ---------------------------------------------------------------------------
# Synthesis: spectral → spatial
# ---------------------------------------------------------------------------

"""
    dist_synthesis!(plan::DistTransposePlan, f::PencilArray, Alm::PencilArray) -> f

Distributed scalar spherical harmonic synthesis using the transpose approach.

Takes spectral coefficients `Alm` (global `(lmax+1, mmax+1)`, m-distributed,
produced by `allocate_spectral(plan)`) and writes the reconstructed spatial
field into `f` (global `(nlon, nlat)`, θ-distributed, produced by
`allocate_spatial(plan)`).

The reverse pass is:
1. Legendre expansion per local m:
   `F[i, mi, lev] = inv_scaleφ · Σ_{l=m}^{lmax} NP[mi][l+1, i] · Alm[l+1, mi, lev]`
2. `ldiv!(f, fft_plan, F_buf)` — inverse transpose + irFFT(φ) → real spatial field.
"""
function SHTnsKit.dist_synthesis!(plan::DistTransposePlan, f::PencilArray, Alm::PencilArray)
    _require_transpose_operands(
        plan, "dist_synthesis!", (f,), (Alm,), (f,), (), false)
    return _dist_transpose_synthesis_unchecked!(plan, f, Alm)
end

function _dist_transpose_synthesis_unchecked!(
        plan::DistTransposePlan, f::PencilArray, Alm::PencilArray)
    A = parent(Alm)           # (lmax+1, n_m_local, nlev)
    F = parent(plan.F_buf)    # (nlat, n_m_local, nlev)  — physical storage (θ fast)

    fill!(F, zero(eltype(F)))

    inv_scaleφ = SHTnsKit.phi_inv_scale(plan.cfg)
    lmax = plan.lmax
    nlat = plan.nlat
    nlev = plan.nlev
    scales = SHTnsKit._uses_canonical_convention(plan.cfg) ? nothing :
        SHTnsKit._ensure_norm_scale_matrix!(plan.cfg)

    # Legendre expansion: for each local m, sum over l → F[i, mi, lev]
    # NP[mi] is (lmax+1, nlat) column-major; iterating i (fast dim of F) is cache-friendly.
    @inbounds for lev in 1:nlev
        for (mi, m) in enumerate(plan.m_local)
            slot = plan.m_slots[mi]
            NP_mi = plan.NP[mi]          # (lmax+1, nlat) matrix for this m
            for i in 1:nlat
                acc = zero(ComplexF64)
                for l in m:lmax
                    Alm_lm = A[l+1, slot, lev]
                    if scales !== nothing
                        Alm_lm *= scales[l+1, m+1]
                    end
                    acc += NP_mi[l+1, i] * Alm_lm
                end
                F[i, slot, lev] = inv_scaleφ * acc
            end
        end
    end

    # Single inverse Alltoall + irFFT for the ENTIRE nlev batch (one collective).
    ldiv!(f, plan.fft_plan, plan.F_buf)
    return f
end

# ---------------------------------------------------------------------------
# Vector (sphtor) transforms
# ---------------------------------------------------------------------------

"""
    dist_analysis_sphtor!(plan::DistTransposePlan, Slm, Tlm, Vt, Vp) -> (Slm, Tlm)

Distributed spheroidal/toroidal analysis using the transpose approach.

Takes spatial vector field components `Vt` (colatitude) and `Vp` (azimuthal),
both global `(nlon, nlat)` PencilArrays, and writes spheroidal/toroidal spectral
coefficients into `Slm` and `Tlm` (both global `(lmax+1, mmax+1)` PencilArrays).

Two FFT+transpose collectives are performed (one per component), then a local
Legendre contraction over (lev, m, θ, l) using the pre-built dP and Pos tables.
"""
function SHTnsKit.dist_analysis_sphtor!(plan::DistTransposePlan,
                                         Slm::PencilArray, Tlm::PencilArray,
                                         Vt::PencilArray,  Vp::PencilArray)
    _require_transpose_operands(
        plan, "dist_analysis_sphtor!", (Vt, Vp), (Slm, Tlm), (),
        (Slm, Tlm), true)
    return _dist_transpose_analysis_sphtor_unchecked!(
        plan, Slm, Tlm, Vt, Vp)
end

function _dist_transpose_analysis_sphtor_unchecked!(
        plan::DistTransposePlan,
        Slm::PencilArray, Tlm::PencilArray,
        Vt::PencilArray, Vp::PencilArray)
    # Step 1: rFFT(φ) + internal transpose for both components.
    mul!(plan.F_buf,  plan.fft_plan, Vt)
    mul!(plan.F_buf2, plan.fft_plan, Vp)

    Ft = parent(plan.F_buf)    # (nlat, n_m_local, nlev) — θ fast, complex Fourier m-modes
    Fp = parent(plan.F_buf2)

    S = parent(Slm)            # (lmax+1, n_m_local, nlev)
    T = parent(Tlm)

    fill!(S, zero(eltype(S)))
    fill!(T, zero(eltype(T)))

    w       = plan.cfg.w
    scaleφ  = SHTnsKit._analysis_phi_scale(plan.cfg)    # 2π/nlon under :dft; inverts synthesis under any phi_scale
    lmax    = plan.lmax
    nlat    = plan.nlat
    nlev    = plan.nlev

    # Step 2: sphtor Legendre contraction per local m.
    # Kernel (from kernels.jl _sphtor_analysis_kernel_otf!):
    #   coeff = w[i] * scaleφ / (l*(l+1))
    #   term  = im*m * Y_over_s
    #   Sacc[l] += coeff * (Ft_i * dtheta_Y + conj(term) * Fp_i)
    #   Tacc[l] += coeff * (-conj(term) * Ft_i + dtheta_Y * Fp_i)
    @inbounds for lev in 1:nlev
        for (mi, m) in enumerate(plan.m_local)
            slot = plan.m_slots[mi]
            dP_mi  = plan.dP[mi]   # (lmax+1, nlat)
            Pos_mi = plan.Pos[mi]  # (lmax+1, nlat)
            for i in 1:nlat
                wi_scale = w[i] * scaleφ
                Ft_i = Ft[i, slot, lev]
                Fp_i = Fp[i, slot, lev]
                for l in max(1, m):lmax
                    dtheta_Y = dP_mi[l+1, i]
                    Y_over_s = Pos_mi[l+1, i]
                    coeff    = wi_scale / (l * (l + 1))
                    term     = (1.0im * m) * Y_over_s   # im*m * P̄/sinθ
                    S[l+1, slot, lev] += coeff * (Ft_i * dtheta_Y + conj(term) * Fp_i)
                    T[l+1, slot, lev] += coeff * (-conj(term) * Ft_i + dtheta_Y * Fp_i)
                end
            end
        end
    end
    _externalize_local_coefficients!(S, plan)
    _externalize_local_coefficients!(T, plan)
    return Slm, Tlm
end

"""
    dist_synthesis_sphtor!(plan::DistTransposePlan, Vt, Vp, Slm, Tlm) -> (Vt, Vp)

Distributed spheroidal/toroidal synthesis using the transpose approach.

Takes spectral coefficients `Slm` and `Tlm` (both global `(lmax+1, mmax+1)`
PencilArrays) and writes reconstructed vector field components into `Vt` and `Vp`
(both global `(nlon, nlat)` PencilArrays).

Local Legendre expansion Slm,Tlm → Ft,Fp, then two inverse FFT+transpose collectives
(one per component) recover the real spatial fields.
"""
function SHTnsKit.dist_synthesis_sphtor!(plan::DistTransposePlan,
                                          Vt::PencilArray,  Vp::PencilArray,
                                          Slm::PencilArray, Tlm::PencilArray)
    _require_transpose_operands(
        plan, "dist_synthesis_sphtor!", (Vt, Vp), (Slm, Tlm),
        (Vt, Vp), (), true)
    return _dist_transpose_synthesis_sphtor_unchecked!(
        plan, Vt, Vp, Slm, Tlm)
end

function _dist_transpose_synthesis_sphtor_unchecked!(
        plan::DistTransposePlan,
        Vt::PencilArray, Vp::PencilArray,
        Slm::PencilArray, Tlm::PencilArray)
    S = parent(Slm)            # (lmax+1, n_m_local, nlev)
    T = parent(Tlm)

    Ft = parent(plan.F_buf)    # (nlat, n_m_local, nlev)
    Fp = parent(plan.F_buf2)

    fill!(Ft, zero(eltype(Ft)))
    fill!(Fp, zero(eltype(Fp)))

    inv_scaleφ = SHTnsKit.phi_inv_scale(plan.cfg)
    lmax = plan.lmax
    nlat = plan.nlat
    nlev = plan.nlev
    scales = SHTnsKit._uses_canonical_convention(plan.cfg) ? nothing :
        SHTnsKit._ensure_norm_scale_matrix!(plan.cfg)

    # Legendre expansion: for each local m, sum over l → Ft[i,mi,lev], Fp[i,mi,lev]
    # Kernel (from kernels.jl _sphtor_synthesis_kernel_otf):
    #   g_theta += dtheta_Y * Sl - im*m * Y_over_s * Tl
    #   g_phi   += im*m * Y_over_s * Sl + dtheta_Y * Tl
    # Then scale by inv_scaleφ (same as scalar synthesis) to undo the rFFT normalization.
    @inbounds for lev in 1:nlev
        for (mi, m) in enumerate(plan.m_local)
            slot = plan.m_slots[mi]
            dP_mi  = plan.dP[mi]
            Pos_mi = plan.Pos[mi]
            for i in 1:nlat
                g_theta = zero(ComplexF64)
                g_phi   = zero(ComplexF64)
                for l in max(1, m):lmax
                    dtheta_Y = dP_mi[l+1, i]
                    Y_over_s = Pos_mi[l+1, i]
                    Sl = S[l+1, slot, lev]
                    Tl = T[l+1, slot, lev]
                    if scales !== nothing
                        scale = scales[l+1, m+1]
                        Sl *= scale
                        Tl *= scale
                    end
                    g_theta += dtheta_Y * Sl - (1.0im * m) * Y_over_s * Tl
                    g_phi   += (1.0im * m) * Y_over_s * Sl + dtheta_Y * Tl
                end
                Ft[i, slot, lev] = inv_scaleφ * g_theta
                Fp[i, slot, lev] = inv_scaleφ * g_phi
            end
        end
    end

    # Two inverse Alltoall + irFFT collectives (one per component).
    ldiv!(Vt, plan.fft_plan, plan.F_buf)
    ldiv!(Vp, plan.fft_plan, plan.F_buf2)
    return Vt, Vp
end

# ---------------------------------------------------------------------------
# QST transforms (by delegation)
# ---------------------------------------------------------------------------

"""
    dist_analysis_qst!(plan::DistTransposePlan, Qlm, Slm, Tlm, Vr, Vt, Vp) -> (Qlm, Slm, Tlm)

Distributed QST analysis: scalar radial component (Q) via `dist_analysis!` and
spheroidal/toroidal components (S,T) via `dist_analysis_sphtor!`.
"""
function SHTnsKit.dist_analysis_qst!(plan::DistTransposePlan,
                                      Qlm::PencilArray, Slm::PencilArray, Tlm::PencilArray,
                                      Vr::PencilArray,  Vt::PencilArray,  Vp::PencilArray)
    _require_transpose_operands(
        plan, "dist_analysis_qst!", (Vr, Vt, Vp), (Qlm, Slm, Tlm), (),
        (Qlm, Slm, Tlm), true)
    _dist_transpose_analysis_unchecked!(plan, Qlm, Vr)
    _dist_transpose_analysis_sphtor_unchecked!(plan, Slm, Tlm, Vt, Vp)
    return Qlm, Slm, Tlm
end

"""
    dist_synthesis_qst!(plan::DistTransposePlan, Vr, Vt, Vp, Qlm, Slm, Tlm) -> (Vr, Vt, Vp)

Distributed QST synthesis: scalar radial component (Q) via `dist_synthesis!` and
spheroidal/toroidal components (S,T) via `dist_synthesis_sphtor!`.
"""
function SHTnsKit.dist_synthesis_qst!(plan::DistTransposePlan,
                                       Vr::PencilArray,  Vt::PencilArray,  Vp::PencilArray,
                                       Qlm::PencilArray, Slm::PencilArray, Tlm::PencilArray)
    _require_transpose_operands(
        plan, "dist_synthesis_qst!", (Vr, Vt, Vp), (Qlm, Slm, Tlm),
        (Vr, Vt, Vp), (), true)
    _dist_transpose_synthesis_unchecked!(plan, Vr, Qlm)
    _dist_transpose_synthesis_sphtor_unchecked!(plan, Vt, Vp, Slm, Tlm)
    return Vr, Vt, Vp
end

# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

"""
    allocate_spatial(plan::DistTransposePlan) -> PencilArray

Return a freshly allocated real PencilArray in the spatial input layout
expected by `plan`: global `(nlon, nlat)` with θ distributed, φ local,
plus `extra_dims=(nlev,)` as a trailing local dimension.

The caller fills this array then passes it to the forward transform.
"""
SHTnsKit.allocate_spatial(plan::DistTransposePlan) = allocate_input(plan.fft_plan)

"""
    allocate_spectral(plan::DistTransposePlan) -> PencilArray

Return a freshly allocated complex PencilArray in the spectral layout:
global `(lmax+1, nbin)` with the m/bin axis (dim 2) distributed, l fully local,
plus `nlev` as a trailing local dimension, where `nbin = nlon÷2+1` is the number
of rFFT Fourier bins.

# Shape contract (IMPORTANT)

The dim-2 global size is `nbin = nlon÷2+1`, NOT `mmax+1`.

- For the **canonical** grid (`nlon == 2*mmax+1`) `nbin == mmax+1`, so the layout
  is identical to the historical `(lmax+1, mmax+1)`.
- For **dealiased** grids (`nlon > 2*mmax+1`, e.g. the 3/2 rule) `nbin > mmax+1`.
  The dim-2 distribution is aligned bin-for-bin with the rFFT output, so on every
  rank the local columns map to that rank's Fourier bins. The MEANINGFUL spherical
  harmonic coefficients are the columns whose orders obey both `m ≤ mmax` and
  `m % mres == 0`. Their physical columns are `plan.m_slots`; columns for
  excluded orders and bins `m > mmax` are unused, zeroed by analysis, and
  ignored by synthesis (the irFFT zero-pads them).

So callers should pair `plan.m_local` with `plan.m_slots` (as the `dist_*`
kernels do), and treat the local column count as possibly exceeding
`length(plan.m_local)` for non-unit `mres` or dealiased grids.
"""
function SHTnsKit.allocate_spectral(plan::DistTransposePlan)
    return PencilArray{ComplexF64}(undef, plan.spectral_pencil, plan.nlev)
end
