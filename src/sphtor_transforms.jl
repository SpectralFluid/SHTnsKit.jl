#=
================================================================================
sphtor_transforms.jl - Spheroidal-Toroidal Vector Field Transforms
================================================================================

This file implements spherical harmonic transforms for 2D horizontal (tangential)
vector fields on the sphere using the spheroidal-toroidal (S/T) decomposition.

PHYSICAL MOTIVATION
-------------------
Any smooth horizontal vector field V on the sphere can be uniquely decomposed:
    V = V_S + V_T  (spheroidal + toroidal)

where:
- Spheroidal (S): curl-free, associated with DIVERGENT flow (sources/sinks)
- Toroidal (T):   div-free, associated with ROTATIONAL flow (vortices)

This is analogous to the Helmholtz decomposition in 3D, but specialized for
tangent vectors on a 2-sphere.

MATHEMATICAL FORMULATION
------------------------
Given spheroidal potential S(θ,φ) and toroidal potential T(θ,φ), the vector
components are:

    V_θ = ∂S/∂θ - (1/sin θ) ∂T/∂φ    (colatitude/meridional component)
    V_φ = (1/sin θ) ∂S/∂φ + ∂T/∂θ    (azimuthal/zonal component)

In spectral space with Y_l^m(θ,φ) = N_lm P_l^m(cos θ) exp(imφ):

    S(θ,φ) = Σ_{l,m} S_lm Y_l^m(θ,φ)
    T(θ,φ) = Σ_{l,m} T_lm Y_l^m(θ,φ)

The scalar invariants (divergence and vorticity) are directly related:

    δ_lm = -l(l+1) S_lm    (divergence spectrum)
    ζ_lm = -l(l+1) T_lm    (vorticity spectrum)

This makes S/T decomposition extremely useful in fluid dynamics where
divergence and vorticity are fundamental quantities.

APPLICATIONS
------------
- Atmospheric/oceanic flows: wind velocity, ocean currents
- Geophysics: surface plate motions, magnetic field horizontal components
- Astrophysics: stellar surface flows, solar wind

IMPLEMENTATION STRUCTURE
------------------------
Main transforms:
    synthesis_sphtor(cfg, Slm, Tlm)   : Synthesis (spectral → spatial)
    analysis_sphtor(cfg, Vt, Vp)      : Analysis (spatial → spectral)

Helper functions:
    synthesis_sph(cfg, Slm)           : Spheroidal-only synthesis (T=0)
    synthesis_tor(cfg, Tlm)           : Toroidal-only synthesis (S=0)

Spectral operators:
    divergence_from_spheroidal(cfg, Slm)   : δ_lm = -l(l+1) S_lm
    vorticity_from_toroidal(cfg, Tlm)      : ζ_lm = -l(l+1) T_lm
    spheroidal_from_divergence(cfg, δlm)   : Invert for S_lm
    toroidal_from_vorticity(cfg, ζlm)      : Invert for T_lm

Degree-limited variants (suffix _l):
    synthesis_sphtor_l, analysis_sphtor_l, etc.

Mode-limited variants (suffix _ml):
    For single azimuthal mode m processing

DEBUGGING TIPS
--------------
1. Test with pure spheroidal field (should have zero vorticity):
   ```julia
   cfg = create_gauss_config(32, 64)
   Slm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
   Slm[3, 1] = 1.0  # l=2, m=0 mode
   Tlm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
   Vt, Vp = synthesis_sphtor(cfg, Slm, Tlm)
   S_back, T_back = analysis_sphtor(cfg, Vt, Vp)
   @assert norm(T_back) < 1e-10 "Pure spheroidal should have no toroidal"
   ```

2. Verify divergence/vorticity relationships:
   ```julia
   δ = divergence_from_spheroidal(cfg, Slm)
   @assert δ[3,1] ≈ -6.0 * Slm[3,1]  # -l(l+1) = -2*3 = -6
   ```

3. Check pole behavior - sin θ → 0 at poles:
   - V_φ term (1/sinθ)∂S/∂φ can be singular
   - Implementation uses pole-safe functions (Plm_and_dPdtheta_row!, Plm_over_sinth_row!)
     that compute analytical limits at poles instead of dividing by zero

4. Robert form scaling:
   - If cfg.robert_form = true, outputs are scaled by sin(θ)
   - This regularizes pole singularities for numerical stability

PERFORMANCE NOTES
-----------------
- Uses thread-local Legendre polynomial arrays to enable @threads parallelism
- Can use precomputed tables (cfg.plm_tables, cfg.dplm_tables) if available
- Requires BOTH P_l^m AND dP_l^m/dx for vector transforms (unlike scalar)

================================================================================
=#

"""
Spheroidal-Toroidal Vector Field Transforms

This module handles transforms for horizontal (tangential) vector fields using
spheroidal (S) and toroidal (T) decomposition. This is the natural representation
for 2D vector fields on the sphere, such as horizontal velocity components.

Key relationships:
- Vt = ∂S/∂θ - (1/sin θ) ∂T/∂φ  (colatitude component)
- Vp = (1/sin θ) ∂S/∂φ + ∂T/∂θ  (azimuthal component)

Where S represents the spheroidal (curl-free) part and T the toroidal (div-free) part.
The corresponding scalar invariants are:
- Divergence: δ = ∇·V = -∑_{l,m} l(l+1) S_lm Y_l^m
- Vorticity:  ζ = (∇×V)·r̂ = -∑_{l,m} l(l+1) T_lm Y_l^m
so spheroidal/toroidal spectra are directly tied to divergence and vorticity.
"""

struct _ZeroSpectralMatrix{T} <: AbstractMatrix{T}
    nrows::Int
    ncols::Int
end

struct _ZeroSpectralVector{T} <: AbstractVector{T}
    len::Int
end

# Lightweight read-only zero spectra used by spheroidal-only and toroidal-only
# wrappers. They avoid allocating dense zero Slm/Tlm arrays while still
# satisfying the AbstractArray interface used by the shared kernels.
Base.size(A::_ZeroSpectralMatrix) = (A.nrows, A.ncols)
Base.axes(A::_ZeroSpectralMatrix) = (Base.OneTo(A.nrows), Base.OneTo(A.ncols))
@inline Base.getindex(::_ZeroSpectralMatrix{T}, ::Int, ::Int) where {T} = zero(T)
Base.similar(A::_ZeroSpectralMatrix{T}) where {T} = zeros(T, size(A))
Base.similar(A::_ZeroSpectralMatrix, ::Type{T}) where {T} = zeros(T, size(A))
Base.similar(::_ZeroSpectralMatrix, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

Base.size(A::_ZeroSpectralVector) = (A.len,)
Base.axes(A::_ZeroSpectralVector) = (Base.OneTo(A.len),)
@inline Base.getindex(::_ZeroSpectralVector{T}, ::Int) where {T} = zero(T)
Base.similar(A::_ZeroSpectralVector{T}) where {T} = zeros(T, size(A))
Base.similar(A::_ZeroSpectralVector, ::Type{T}) where {T} = zeros(T, size(A))
Base.copy(A::_ZeroSpectralVector{T}) where {T} = zeros(T, length(A))

@inline _zero_spectral_matrix(::Type{T}, cfg::SHTConfig) where {T} =
    _ZeroSpectralMatrix{T}(cfg.lmax + 1, cfg.mmax + 1)
@inline _zero_spectral_vector(::Type{T}, len::Int) where {T} =
    _ZeroSpectralVector{T}(len)

"""
    synthesis_sphtor(cfg, Slm, Tlm; real_output=true) -> (Vt, Vp)

Transform spheroidal/toroidal coefficients to horizontal vector field components.
Returns colatitude (Vt) and azimuthal (Vp) components on the spatial grid.
"""
Base.@constprop :aggressive function synthesis_sphtor(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true, use_rfft::Bool=false)
    # `real_output` and `use_rfft` choose different buffer shapes and return
    # element types, so dispatch through Val for literal keyword calls.
    return _synthesis_sphtor(cfg, Slm, Tlm, Val(real_output), Val(use_rfft))
end

function _synthesis_sphtor(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix,
                           ::Val{real_output}, ::Val{use_rfft}) where {real_output, use_rfft}
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Slm,1) == lmax+1 && size(Slm,2) == mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1) == lmax+1 && size(Tlm,2) == mmax+1 || throw(DimensionMismatch("Tlm dims"))

    Slm_int = _internal_coefficients(Slm, cfg)
    Tlm_int = _internal_coefficients(Tlm, cfg)

    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(Slm_int)

    if use_rfft
        # Vector rfft synthesis mirrors scalar synthesis: write only
        # nonnegative m columns, then let irfft_phi! reconstruct real output.
        real_output === true || throw(ArgumentError("use_rfft=true implies real_output"))
        mmax ≤ nlon ÷ 2 || throw(ArgumentError("use_rfft=true requires mmax ≤ nlon÷2"))
        nbins = nlon ÷ 2 + 1
        Ftheta = zeros(CT, nlat, nbins)
        Fphi   = zeros(CT, nlat, nbins)
        _synthesis_sphtor_mloop!(Ftheta, Fphi, cfg, Slm_int, Tlm_int;
                                  ltr=lmax, real_output=false)  # skip Hermitian fill
        RT = typeof(real(zero(CT)))
        Vt = Matrix{RT}(undef, nlat, nlon)
        Vp = Matrix{RT}(undef, nlat, nlon)
        irfft_phi!(Vt, Ftheta, nlon)
        irfft_phi!(Vp, Fphi, nlon)
    else
        Ftheta = zeros(CT, nlat, nlon)
        Fphi = zeros(CT, nlat, nlon)
        _synthesis_sphtor_mloop!(Ftheta, Fphi, cfg, Slm_int, Tlm_int;
                                  ltr=lmax, real_output=real_output)
        ifft_phi!(Ftheta, Ftheta)
        ifft_phi!(Fphi, Fphi)
        if real_output
            RT = typeof(real(zero(CT)))
            Vt = Matrix{RT}(undef, nlat, nlon)
            Vp = Matrix{RT}(undef, nlat, nlon)
            @inbounds for j in 1:nlon, i in 1:nlat
                Vt[i, j] = real(Ftheta[i, j])
                Vp[i, j] = real(Fphi[i, j])
            end
        else
            Vt = Ftheta
            Vp = Fphi
        end
    end

    if cfg.robert_form
        @inbounds for i in 1:nlat
            s_theta = sqrt(max(0.0, 1 - cfg.x[i]^2))
            for j in 1:nlon
                Vt[i, j] *= s_theta
                Vp[i, j] *= s_theta
            end
        end
    end
    return Vt, Vp
end

"""
    analysis_sphtor(cfg, Vt, Vp) -> (Slm, Tlm)

Transform horizontal vector field components to spheroidal/toroidal coefficients.
"""
function analysis_sphtor(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix; use_rfft::Bool=false)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(Vt,1) == nlat && size(Vt,2) == nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == nlat && size(Vp,2) == nlon || throw(DimensionMismatch("Vp dims"))

    # Use the in-place, cached-plan FFTs (one buffer each) instead of the old
    # out-of-place `fft_phi(_as_complex(V))` / `rfft_phi(V)` which copied the
    # input and re-planned FFTW on every call. Mirrors the scalar `analysis`
    # fix; eltype is preserved so AD types still take the DFT fallback.
    if use_rfft
        eltype(Vt) <: Real && eltype(Vp) <: Real || throw(ArgumentError("use_rfft=true requires real-valued Vt, Vp"))
        nbins = nlon ÷ 2 + 1
        RT = float(real(promote_type(eltype(Vt), eltype(Vp))))
        Fthetam = rfft_phi!(Matrix{Complex{RT}}(undef, nlat, nbins), Vt)
        Fphim   = rfft_phi!(Matrix{Complex{RT}}(undef, nlat, nbins), Vp)
    else
        CT = complex(float(promote_type(eltype(Vt), eltype(Vp))))
        Fthetam = fft_phi!(Matrix{CT}(undef, nlat, nlon), Vt)
        Fphim   = fft_phi!(Matrix{CT}(undef, nlat, nlon), Vp)
    end

    lmax, mmax = cfg.lmax, cfg.mmax
    # Output eltype follows the input so AD types (e.g. ForwardDiff.Dual) flow
    # through; defaults to ComplexF64 for ordinary real inputs.
    OT = complex(float(promote_type(eltype(Vt), eltype(Vp))))
    Slm_int = zeros(OT, lmax + 1, mmax + 1)
    Tlm_int = zeros(OT, lmax + 1, mmax + 1)

    _analysis_sphtor_mloop!(Slm_int, Tlm_int, cfg, Fthetam, Fphim; ltr=lmax)

    _externalize_coefficients!(Slm_int, cfg)
    _externalize_coefficients!(Tlm_int, cfg)
    return Slm_int, Tlm_int
end

# ============================================================================
# SPHTOR ORCHESTRATORS
# ============================================================================

"""
    _adjoint_analysis_sphtor(cfg, Slm̄, Tlm̄; θ_globals=1:cfg.nlat, φ_window=nothing)
      -> (V̄t, V̄p)

Adjoint operator of `analysis_sphtor`. Maps replicated coefficient cotangents
(Slm̄, Tlm̄) to a pair of spatial cotangents for this rank's θ rows (and
optional φ window). Shared between non-plan scalar AD and the distributed
AD extension; lives in src so both exts can reuse without circular deps.
"""
function _adjoint_analysis_sphtor(cfg::SHTConfig, Slm̄::AbstractMatrix, Tlm̄::AbstractMatrix;
                                   θ_globals::AbstractVector{<:Integer}=1:cfg.nlat,
                                   φ_window::Union{Nothing,UnitRange{Int}}=nothing)
    nlon = cfg.nlon
    lmax, mmax = cfg.lmax, cfg.mmax
    nlat_local = length(θ_globals)
    Slm̄_int = _analysis_cotangent_to_canonical(Slm̄, cfg)
    Tlm̄_int = _analysis_cotangent_to_canonical(Tlm̄, cfg)

    F̄θ = Matrix{ComplexF64}(undef, nlat_local, nlon)
    F̄φ = Matrix{ComplexF64}(undef, nlat_local, nlon)
    fill!(F̄θ, zero(ComplexF64)); fill!(F̄φ, zero(ComplexF64))

    P = Vector{Float64}(undef, lmax + 1)
    dPdtheta = Vector{Float64}(undef, lmax + 1)
    P_over_sinth = Vector{Float64}(undef, lmax + 1)
    Pbuf = Vector{Float64}(undef, lmax + 2)   # scratch for extended P̄ row (avoids per-call alloc)
    φadj = 2π

    for m in 0:mmax
        col = m + 1
        for (ii, iglob) in pairs(θ_globals)
            Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, cfg.x[iglob], lmax, m, Pbuf)
            wi = cfg.w[iglob]
            sθ = zero(ComplexF64); sφ = zero(ComplexF64)
            @inbounds for l in max(1, m):lmax
                dθY = dPdtheta[l+1]          # already orthonormal-normalized
                Y_over_sθ = P_over_sinth[l+1]
                ll1 = l * (l + 1)
                term = 1.0im * m * Y_over_sθ
                S_bar = Slm̄_int[l+1, col]; T_bar = Tlm̄_int[l+1, col]
                sθ += (dθY * S_bar + conj(term) * T_bar) / ll1
                sφ += (-conj(term) * S_bar + dθY * T_bar) / ll1
            end
            F̄θ[ii, col] = φadj * wi * sθ
            F̄φ[ii, col] = φadj * wi * sφ
        end
    end

    ifft_phi!(F̄θ, F̄θ); ifft_phi!(F̄φ, F̄φ)
    if φ_window === nothing
        return real.(F̄θ), real.(F̄φ)
    end
    V̄t = Matrix{Float64}(undef, nlat_local, length(φ_window))
    V̄p = Matrix{Float64}(undef, nlat_local, length(φ_window))
    @inbounds for (jj, jglob) in pairs(φ_window)
        for i in 1:nlat_local
            V̄t[i, jj] = real(F̄θ[i, jglob])
            V̄p[i, jj] = real(F̄φ[i, jglob])
        end
    end
    return V̄t, V̄p
end

"""Sphtor synthesis orchestrator. Parallelizes over m-modes."""
function _synthesis_sphtor_mloop!(Ftheta::AbstractMatrix, Fphi::AbstractMatrix,
                                   cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix;
                                   ltr::Int=cfg.lmax, real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlon = cfg.nlon
    ltr_eff = min(ltr, lmax)
    inv_scale_phi = phi_inv_scale(cfg)
    m_order = cached_m_order(cfg)

    if has_fused_vector_tables(cfg)
        # Fused tables store normalization with P and dP values, reducing
        # inner-loop multiplies at the cost of table memory.
        _synthesis_sphtor_mloop_tbl!(Ftheta, Fphi, cfg, Slm, Tlm, m_order, ltr_eff, inv_scale_phi)
    else
        # On-the-fly mode uses thread-local Legendre scratch to avoid shared
        # mutable buffers while keeping memory bounded by nthreads*lmax.
        _synthesis_sphtor_mloop_otf!(Ftheta, Fphi, cfg, Slm, Tlm, m_order, ltr_eff, inv_scale_phi)
    end

    if real_output
        # Full complex IFFT needs negative-m bins explicitly. rfft callers set
        # real_output=false here and reconstruct Hermitian symmetry in irfft.
        nlat = cfg.nlat
        for m in 1:mmax
            col = m + 1
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                Ftheta[i, conj_index] = conj(Ftheta[i, col])
                Fphi[i, conj_index] = conj(Fphi[i, col])
            end
        end
    end
    return Ftheta, Fphi
end

@inline function _synthesis_sphtor_mloop_tbl!(Ftheta, Fphi, cfg, Slm, Tlm, m_order, ltr_eff, inv_scale_phi)
    nlat = cfg.nlat
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        NP = cfg.NP_tables[col]
        NdP = cfg.NdP_tables[col]
        @inbounds for i in 1:nlat
            g_theta, g_phi = _sphtor_synthesis_kernel(cfg, Slm, Tlm, NP, NdP, i, col, m, ltr_eff)
            Ftheta[i, col] = inv_scale_phi * g_theta
            Fphi[i, col] = inv_scale_phi * g_phi
        end
    end
end

@inline function _synthesis_sphtor_mloop_otf!(Ftheta, Fphi, cfg, Slm, Tlm, m_order, ltr_eff, inv_scale_phi)
    lmax = cfg.lmax
    nlat = cfg.nlat
    thread_P = _ensure_otf_scratch!(cfg._otf_scratch_P, lmax)
    thread_dP = _ensure_otf_scratch!(cfg._otf_scratch_dP, lmax)
    thread_Ps = _ensure_otf_scratch!(cfg._otf_scratch_Ps, lmax)
    thread_Pb = _ensure_otf_scratch!(cfg._otf_scratch_Pb, lmax + 1)  # lmax+2 for extended P̄ row
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        tid = Threads.threadid()
        P = thread_P[tid]; dP = thread_dP[tid]; Ps = thread_Ps[tid]; Pb = thread_Pb[tid]
        @inbounds for i in 1:nlat
            g_theta, g_phi = _sphtor_synthesis_kernel_otf(cfg, Slm, Tlm, P, dP, Ps, Pb, i, col, m, ltr_eff)
            Ftheta[i, col] = inv_scale_phi * g_theta
            Fphi[i, col] = inv_scale_phi * g_phi
        end
    end
end

"""Sphtor analysis orchestrator. Parallelizes over m-modes."""
function _analysis_sphtor_mloop!(Slm::AbstractMatrix, Tlm::AbstractMatrix,
                                  cfg::SHTConfig, Fthetam::AbstractMatrix, Fphim::AbstractMatrix;
                                  ltr::Int=cfg.lmax)
    lmax, mmax = cfg.lmax, cfg.mmax
    ltr_eff = min(ltr, lmax)
    scale_phi = cfg.cphi
    m_order = cached_m_order(cfg)

    if has_fused_vector_tables(cfg)
        _analysis_sphtor_mloop_tbl!(Slm, Tlm, cfg, Fthetam, Fphim, m_order, ltr_eff, scale_phi)
    else
        _analysis_sphtor_mloop_otf!(Slm, Tlm, cfg, Fthetam, Fphim, m_order, ltr_eff, scale_phi)
    end
    return Slm, Tlm
end

@inline function _analysis_sphtor_mloop_tbl!(Slm, Tlm, cfg, Fthetam, Fphim, m_order, ltr_eff, scale_phi)
    lmax = cfg.lmax
    nlat = cfg.nlat
    # Hoist cfg field reads to locals: cfg is mutable, so reads inside the loops below aren't auto-hoisted.
    w = cfg.w
    x = cfg.x
    robert_form = cfg.robert_form
    # Accumulate directly into the output columns: Slm/Tlm are pre-zeroed by the
    # caller and each m owns a distinct column (col=m+1), so this is race-free and
    # drops the per-call thread-accumulator allocation. The column views carry the
    # output eltype, so AD types (e.g. ForwardDiff.Dual) still propagate.
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        Sacc = view(Slm, :, col)
        Tacc = view(Tlm, :, col)
        NP = cfg.NP_tables[col]
        NdP = cfg.NdP_tables[col]
        for i in 1:nlat
            wi = w[i]
            Ftheta_i = Fthetam[i, col]; Fphi_i = Fphim[i, col]
            if robert_form
                s_theta = sqrt(max(0.0, 1 - x[i]^2))
                if s_theta > 0
                    Ftheta_i /= s_theta; Fphi_i /= s_theta
                end
            end
            _sphtor_analysis_kernel!(Sacc, Tacc, cfg, Ftheta_i, Fphi_i, wi,
                NP, NdP, i, col, m, ltr_eff, scale_phi)
        end
    end
end

@inline function _analysis_sphtor_mloop_otf!(Slm, Tlm, cfg, Fthetam, Fphim, m_order, ltr_eff, scale_phi)
    lmax = cfg.lmax
    nlat = cfg.nlat
    # Hoist cfg field reads to locals: cfg is mutable, so reads inside the loops below aren't auto-hoisted.
    w = cfg.w
    x = cfg.x
    robert_form = cfg.robert_form
    thread_P = _ensure_otf_scratch!(cfg._otf_scratch_P, lmax)
    thread_dP = _ensure_otf_scratch!(cfg._otf_scratch_dP, lmax)
    thread_Ps = _ensure_otf_scratch!(cfg._otf_scratch_Ps, lmax)
    thread_Pb = _ensure_otf_scratch!(cfg._otf_scratch_Pb, lmax + 1)  # lmax+2 for extended P̄ row
    # Accumulate directly into the output columns (see _analysis_sphtor_mloop_tbl!):
    # pre-zeroed, one m per column → race-free, alloc-free, AD eltype preserved.
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        tid = Threads.threadid()
        Sacc = view(Slm, :, col)
        Tacc = view(Tlm, :, col)
        P = thread_P[tid]; dP = thread_dP[tid]; Ps = thread_Ps[tid]; Pb = thread_Pb[tid]
        for i in 1:nlat
            wi = w[i]
            Ftheta_i = Fthetam[i, col]; Fphi_i = Fphim[i, col]
            if robert_form
                s_theta = sqrt(max(0.0, 1 - x[i]^2))
                if s_theta > 0
                    Ftheta_i /= s_theta; Fphi_i /= s_theta
                end
            end
            _sphtor_analysis_kernel_otf!(Sacc, Tacc, cfg, Ftheta_i, Fphi_i, wi,
                P, dP, Ps, Pb, i, col, m, ltr_eff, scale_phi)
        end
    end
end

"""
    synthesis_sphtor_cplx(cfg, Slm, Tlm) -> (Vt, Vp)

Complex version preserving complex values in output.
"""
function synthesis_sphtor_cplx(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix)
    return _synthesis_sphtor(cfg, Slm, Tlm, Val(false), Val(false))
end

"""
    analysis_sphtor_cplx(cfg, Vt, Vp) -> (Slm, Tlm)

Transform complex horizontal vector components to spheroidal/toroidal coefficients.
"""
function analysis_sphtor_cplx(cfg::SHTConfig, Vt::AbstractMatrix{<:Complex}, Vp::AbstractMatrix{<:Complex})
    return analysis_sphtor(cfg, Vt, Vp)  # Same implementation works for complex
end

"""
    synthesis_sph(cfg, Slm; real_output=true) -> (Vt, Vp)

Transform only spheroidal component to spatial vector field (Tlm = 0).
"""
function synthesis_sph(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)
    Tlm_zero = _zero_spectral_matrix(eltype(Slm), cfg)
    return synthesis_sphtor(cfg, Slm, Tlm_zero; real_output=real_output)
end

"""
    synthesis_sph_cplx(cfg, Slm) -> (Vt, Vp)

Complex-output spheroidal-only synthesis without allocating a dense zero
toroidal spectrum.
"""
function synthesis_sph_cplx(cfg::SHTConfig, Slm::AbstractMatrix)
    Tlm_zero = _zero_spectral_matrix(eltype(Slm), cfg)
    return _synthesis_sphtor(cfg, Slm, Tlm_zero, Val(false), Val(false))
end

"""
    synthesis_tor(cfg, Tlm; real_output=true) -> (Vt, Vp)

Transform only toroidal component to spatial vector field (Slm = 0).
"""
function synthesis_tor(cfg::SHTConfig, Tlm::AbstractMatrix; real_output::Bool=true)
    Slm_zero = _zero_spectral_matrix(eltype(Tlm), cfg)
    return synthesis_sphtor(cfg, Slm_zero, Tlm; real_output=real_output)
end

"""
    synthesis_tor_cplx(cfg, Tlm) -> (Vt, Vp)

Complex-output toroidal-only synthesis without allocating a dense zero
spheroidal spectrum.
"""
function synthesis_tor_cplx(cfg::SHTConfig, Tlm::AbstractMatrix)
    Slm_zero = _zero_spectral_matrix(eltype(Tlm), cfg)
    return _synthesis_sphtor(cfg, Slm_zero, Tlm, Val(false), Val(false))
end

"""
    divergence_from_spheroidal(cfg, Slm) -> Matrix

Return divergence spectral coefficients δ_lm corresponding to spheroidal modes
via δ_lm = −l(l+1) S_lm (`δ_{00} = 0`).
"""
function divergence_from_spheroidal(cfg::SHTConfig, Slm::AbstractMatrix)
    δ = similar(Slm)
    return divergence_from_spheroidal!(cfg, δ, Slm)
end

function divergence_from_spheroidal!(cfg::SHTConfig, δ::AbstractMatrix, Slm::AbstractMatrix)
    size(δ) == size(Slm) || throw(DimensionMismatch("divergence output dims"))
    fill!(δ, zero(eltype(δ)))
    lmax, mmax = cfg.lmax, cfg.mmax
    @inbounds for m in 0:mmax, l in max(1, m):lmax
        row = l + 1; col = m + 1
        δ[row, col] = -(l * (l + 1)) * Slm[row, col]
    end
    return δ
end

"""
    spheroidal_from_divergence(cfg, δlm) -> Matrix

Invert δ_lm = −l(l+1) S_lm for l ≥ 1 to recover spheroidal coefficients.
"""
function spheroidal_from_divergence(cfg::SHTConfig, δlm::AbstractMatrix)
    Slm = similar(δlm)
    return spheroidal_from_divergence!(cfg, Slm, δlm)
end

function spheroidal_from_divergence!(cfg::SHTConfig, Slm::AbstractMatrix, δlm::AbstractMatrix)
    size(Slm) == size(δlm) || throw(DimensionMismatch("spheroidal output dims"))
    fill!(Slm, zero(eltype(Slm)))
    lmax, mmax = cfg.lmax, cfg.mmax
    @inbounds for m in 0:mmax, l in max(1, m):lmax
        row = l + 1; col = m + 1
        ll1 = l * (l + 1)
        Slm[row, col] = ll1 == 0 ? zero(eltype(Slm)) : -(δlm[row, col] / ll1)
    end
    return Slm
end

"""
    vorticity_from_toroidal(cfg, Tlm) -> Matrix

Return vorticity spectral coefficients ζ_lm = −l(l+1) T_lm.
"""
function vorticity_from_toroidal(cfg::SHTConfig, Tlm::AbstractMatrix)
    ζ = similar(Tlm)
    return vorticity_from_toroidal!(cfg, ζ, Tlm)
end

function vorticity_from_toroidal!(cfg::SHTConfig, ζ::AbstractMatrix, Tlm::AbstractMatrix)
    size(ζ) == size(Tlm) || throw(DimensionMismatch("vorticity output dims"))
    fill!(ζ, zero(eltype(ζ)))
    lmax, mmax = cfg.lmax, cfg.mmax
    @inbounds for m in 0:mmax, l in max(1, m):lmax
        row = l + 1; col = m + 1
        ζ[row, col] = -(l * (l + 1)) * Tlm[row, col]
    end
    return ζ
end

"""
    toroidal_from_vorticity(cfg, ζlm) -> Matrix

Recover toroidal stream function coefficients from vorticity spectra.
"""
function toroidal_from_vorticity(cfg::SHTConfig, ζlm::AbstractMatrix)
    Tlm = similar(ζlm)
    return toroidal_from_vorticity!(cfg, Tlm, ζlm)
end

function toroidal_from_vorticity!(cfg::SHTConfig, Tlm::AbstractMatrix, ζlm::AbstractMatrix)
    size(Tlm) == size(ζlm) || throw(DimensionMismatch("toroidal output dims"))
    fill!(Tlm, zero(eltype(Tlm)))
    lmax, mmax = cfg.lmax, cfg.mmax
    @inbounds for m in 0:mmax, l in max(1, m):lmax
        row = l + 1; col = m + 1
        ll1 = l * (l + 1)
        Tlm[row, col] = ll1 == 0 ? zero(eltype(Tlm)) : -(ζlm[row, col] / ll1)
    end
    return Tlm
end

Base.@constprop :aggressive function synthesis_sphtor_l(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    return _synthesis_sphtor_l(cfg, Slm, Tlm, ltr, Val(real_output))
end

function synthesis_sphtor_l_cplx(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int)
    # Dedicated complex helper avoids inference depending on a runtime keyword.
    return _synthesis_sphtor_l(cfg, Slm, Tlm, ltr, Val(false))
end

function _synthesis_sphtor_l(cfg::SHTConfig, Slm::AbstractMatrix, Tlm::AbstractMatrix, ltr::Int,
                             ::Val{real_output}) where {real_output}
    lmax, mmax = cfg.lmax, cfg.mmax
    size(Slm,1) == lmax+1 && size(Slm,2) == mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1) == lmax+1 && size(Tlm,2) == mmax+1 || throw(DimensionMismatch("Tlm dims"))

    Slm_int = _internal_coefficients(Slm, cfg)
    Tlm_int = _internal_coefficients(Tlm, cfg)

    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(Slm_int)
    Ftheta = zeros(CT, nlat, nlon)
    Fphi = zeros(CT, nlat, nlon)
    _synthesis_sphtor_mloop!(Ftheta, Fphi, cfg, Slm_int, Tlm_int;
                              ltr=ltr, real_output=real_output)
    ifft_phi!(Ftheta, Ftheta)
    ifft_phi!(Fphi, Fphi)
    if real_output
        RT = typeof(real(zero(CT)))
        Vt = Matrix{RT}(undef, nlat, nlon)
        Vp = Matrix{RT}(undef, nlat, nlon)
        @inbounds for j in 1:nlon, i in 1:nlat
            Vt[i, j] = real(Ftheta[i, j])
            Vp[i, j] = real(Fphi[i, j])
        end
    else
        Vt = Ftheta
        Vp = Fphi
    end

    # Robert form: scale by sin(theta) — must match synthesis_sphtor behavior
    if cfg.robert_form
        x = cfg.x  # hoist field read out of the loop below (cfg is mutable, so the compiler can't lift it)
        @inbounds for i in 1:nlat
            s_theta = sqrt(max(0.0, 1 - x[i]^2))
            for j in 1:nlon
                Vt[i, j] *= s_theta
                Vp[i, j] *= s_theta
            end
        end
    end
    return Vt, Vp
end

function analysis_sphtor_l(cfg::SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix, ltr::Int)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(Vt,1) == nlat && size(Vt,2) == nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1) == nlat && size(Vp,2) == nlon || throw(DimensionMismatch("Vp dims"))
    # In-place cached-plan FFT (one buffer each); avoids the copy + re-plan of
    # the old `fft_phi(_as_complex(V))`. See `analysis_sphtor` for rationale.
    CT = complex(float(promote_type(eltype(Vt), eltype(Vp))))  # AD/Float32-safe; promote both inputs
    Fthetam = fft_phi!(Matrix{CT}(undef, nlat, nlon), Vt)
    Fphim   = fft_phi!(Matrix{CT}(undef, nlat, nlon), Vp)
    lmax, mmax = cfg.lmax, cfg.mmax
    Slm = zeros(CT, lmax + 1, mmax + 1)
    Tlm = zeros(CT, lmax + 1, mmax + 1)
    _analysis_sphtor_mloop!(Slm, Tlm, cfg, Fthetam, Fphim; ltr=ltr)

    _externalize_coefficients!(Slm, cfg)
    _externalize_coefficients!(Tlm, cfg)
    return Slm, Tlm
end

"""
    synthesis_sph_l(cfg, Slm, ltr; real_output=true) -> (Vt, Vp)

Degree-limited spheroidal-only transform.
"""
function synthesis_sph_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    # Use a zero-view toroidal spectrum so the shared S/T implementation does
    # not pay for an all-zero dense allocation.
    Tlm_zero = _zero_spectral_matrix(eltype(Slm), cfg)
    return synthesis_sphtor_l(cfg, Slm, Tlm_zero, ltr; real_output=real_output)
end

function synthesis_sph_l_cplx(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int)
    Tlm_zero = _zero_spectral_matrix(eltype(Slm), cfg)
    return _synthesis_sphtor_l(cfg, Slm, Tlm_zero, ltr, Val(false))
end

"""
    synthesis_tor_l(cfg, Tlm, ltr; real_output=true) -> (Vt, Vp)

Degree-limited toroidal-only transform.
"""
function synthesis_tor_l(cfg::SHTConfig, Tlm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    # Use a zero-view spheroidal spectrum so the shared S/T implementation
    # stays allocation-light.
    Slm_zero = _zero_spectral_matrix(eltype(Tlm), cfg)
    return synthesis_sphtor_l(cfg, Slm_zero, Tlm, ltr; real_output=real_output)
end

function synthesis_tor_l_cplx(cfg::SHTConfig, Tlm::AbstractMatrix, ltr::Int)
    Slm_zero = _zero_spectral_matrix(eltype(Tlm), cfg)
    return _synthesis_sphtor_l(cfg, Slm_zero, Tlm, ltr, Val(false))
end

"""
    synthesis_sph_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, ltr::Int)

Mode-limited spheroidal-only synthesis wrapper.
"""
function synthesis_sph_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, ltr::Int)
    # Mode-limited wrappers use zero-vector views for the missing component;
    # this avoids an O(ltr-im) allocation on repeated per-mode calls.
    Tl_zero = _zero_spectral_vector(eltype(Sl), length(Sl))
    return synthesis_sphtor_ml(cfg, im, Sl, Tl_zero, ltr)
end

"""
    synthesis_tor_ml(cfg::SHTConfig, im::Int, Tl::AbstractVector{<:Complex}, ltr::Int)

Mode-limited toroidal-only synthesis wrapper.
"""
function synthesis_tor_ml(cfg::SHTConfig, im::Int, Tl::AbstractVector{<:Complex}, ltr::Int)
    # Mode-limited wrappers use zero-vector views for the missing component;
    # this avoids an O(ltr-im) allocation on repeated per-mode calls.
    Sl_zero = _zero_spectral_vector(eltype(Tl), length(Tl))
    return synthesis_sphtor_ml(cfg, im, Sl_zero, Tl, ltr)
end

"""
    analysis_sphtor_ml(cfg, mval, Vt_m, Vp_m, ltr) -> (Sl, Tl)

Mode-limited transform for a single azimuthal order `mval` (named `mval`, not
`im`, so it cannot shadow Julia's imaginary unit — see `synthesis_sphtor_ml`).
Input vectors contain Fourier coefficients for mode m at all latitudes.
Implements the proper spheroidal-toroidal decomposition using the adjoint of
the synthesis formulas:
    Vθ = ∂S/∂θ - (im/sinθ) * T
    Vφ = (im/sinθ) * S + ∂T/∂θ
"""
function analysis_sphtor_ml(cfg::SHTConfig, mval::Int, Vt_m::AbstractVector{<:Complex}, Vp_m::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    length(Vt_m) == nlat || throw(DimensionMismatch("Vt_m length must be nlat"))
    length(Vp_m) == nlat || throw(DimensionMismatch("Vp_m length must be nlat"))

    num_l = ltr - mval + 1
    CT = complex(float(promote_type(eltype(Vt_m), eltype(Vp_m))))  # AD/Float32-safe output eltype
    Sl = Vector{CT}(undef, num_l)
    Tl = Vector{CT}(undef, num_l)
    fill!(Sl, zero(CT))
    fill!(Tl, zero(CT))

    P = Vector{Float64}(undef, ltr + 1)
    dPdtheta = Vector{Float64}(undef, ltr + 1)
    P_over_sinth = Vector{Float64}(undef, ltr + 1)
    Pbuf = Vector{Float64}(undef, ltr + 2)   # scratch for extended P̄ row (avoids per-call alloc)
    scaleφ = cfg.cphi

    # Integrate using Legendre polynomials and derivatives (pole-safe)
    for i in 1:nlat
        x = cfg.x[i]
        sθ = sqrt(max(0.0, 1 - x*x))
        wi = cfg.w[i]
        Fθ = Vt_m[i]
        Fφ = Vp_m[i]

        # Handle Robert form scaling if needed
        if cfg.robert_form && sθ > 0
            Fθ = Fθ / sθ
            Fφ = Fφ / sθ
        end

        # Single call computes P̄, dP̄/dθ, and P̄/sinθ (bounded normalized; no overflow)
        Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, ltr, mval, Pbuf)

        @inbounds for l in max(1, mval):ltr
            dθY = dPdtheta[l+1]          # already orthonormal-normalized
            Y_over_sθ = P_over_sinth[l+1]
            ll1 = l * (l + 1)
            coeff = wi * scaleφ / ll1
            term = 1.0im * mval * Y_over_sθ

            # Adjoint of synthesis: Vθ = dθY*S - term*T, Vφ = term*S + dθY*T
            Sl[l-mval+1] += coeff * (Fθ * dθY + conj(term) * Fφ)
            Tl[l-mval+1] += coeff * (-conj(term) * Fθ + dθY * Fφ)
        end
    end

    _convert_mode_norm!(Sl, Sl, cfg, mval, ltr; to_internal=false)
    _convert_mode_norm!(Tl, Tl, cfg, mval, ltr; to_internal=false)
    return Sl, Tl
end

"""
    synthesis_sphtor_ml(cfg, mval, Sl, Tl, ltr) -> (Vt_m, Vp_m)

Mode-limited synthesis for a single azimuthal order `mval`:
    Vθ = ∂S/∂θ - (i·mval/sinθ) * T
    Vφ = (i·mval/sinθ) * S + ∂T/∂θ

The order argument is named `mval`, NOT `im`: a binding called `im` shadows
Julia's imaginary unit, and `1.0im` is juxtaposition (`1.0 * im`), so the
coupling terms above silently became real. Do not rename it back.
"""
function synthesis_sphtor_ml(cfg::SHTConfig, mval::Int, Sl::AbstractVector{<:Complex}, Tl::AbstractVector{<:Complex}, ltr::Int)
    nlat = cfg.nlat
    expected_len = ltr - mval + 1
    length(Sl) == expected_len || throw(DimensionMismatch("Sl length mismatch"))
    length(Tl) == expected_len || throw(DimensionMismatch("Tl length mismatch"))
    Sl_int = _uses_canonical_convention(cfg) ? Sl :
             _convert_mode_norm!(similar(Sl), Sl, cfg, mval, ltr; to_internal=true)
    Tl_int = _uses_canonical_convention(cfg) ? Tl :
             _convert_mode_norm!(similar(Tl), Tl, cfg, mval, ltr; to_internal=true)

    CT = complex(float(promote_type(eltype(Sl), eltype(Tl))))  # AD/Float32-safe output eltype
    inv_scaleφ = phi_inv_scale(cfg)  # Match full transform normalization
    Vt_m = Vector{CT}(undef, nlat)
    Vp_m = Vector{CT}(undef, nlat)
    P = Vector{Float64}(undef, ltr + 1)
    dPdtheta = Vector{Float64}(undef, ltr + 1)
    P_over_sinth = Vector{Float64}(undef, ltr + 1)
    Pbuf = Vector{Float64}(undef, ltr + 2)   # scratch for extended P̄ row (avoids per-call alloc)
    # Synthesize vector components for this mode using bounded normalized functions
    for i in 1:nlat
        x = cfg.x[i]
        sθ = sqrt(max(0.0, 1 - x*x))

        # Single call computes P̄, dP̄/dθ, and P̄/sinθ (bounded normalized; no overflow)
        Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, ltr, mval, Pbuf)

        gθ = zero(CT)
        gφ = zero(CT)

        @inbounds for l in max(1, mval):ltr
            dθY = dPdtheta[l+1]          # already orthonormal-normalized
            Y_over_sθ = P_over_sinth[l+1]
            S_coef = Sl_int[l-mval+1]
            T_coef = Tl_int[l-mval+1]

            # Vθ = ∂S/∂θ - (mval/sinθ) * T
            gθ += dθY * S_coef - 1.0im * mval * Y_over_sθ * T_coef
            # Vφ = (mval/sinθ) * S + ∂T/∂θ
            gφ += 1.0im * mval * Y_over_sθ * S_coef + dθY * T_coef
        end

        # Apply Robert form scaling if needed
        if cfg.robert_form
            gθ *= sθ
            gφ *= sθ
        end

        # Match the full transform's φ normalization, exactly as
        # `synthesis_packed_ml` does. Its absence left this pair off by 1/nlon
        # from being the inverse of `analysis_sphtor_ml` (which applies cphi).
        Vt_m[i] = gθ * inv_scaleφ
        Vp_m[i] = gφ * inv_scaleφ
    end

    return Vt_m, Vp_m
end

"""
    synthesis_grad(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)

Gradient synthesis alias for compatibility with SHTns.
"""
function synthesis_grad(cfg::SHTConfig, Slm::AbstractMatrix; real_output::Bool=true)
    return synthesis_sph(cfg, Slm; real_output=real_output)
end

"""
    synthesis_grad_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)

Degree-limited gradient synthesis alias.
"""
function synthesis_grad_l(cfg::SHTConfig, Slm::AbstractMatrix, ltr::Int; real_output::Bool=true)
    return synthesis_sph_l(cfg, Slm, ltr; real_output=real_output)
end

"""
    synthesis_grad_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, ltr::Int)

Mode-limited gradient synthesis alias.
"""
function synthesis_grad_ml(cfg::SHTConfig, im::Int, Sl::AbstractVector{<:Complex}, ltr::Int)
    return synthesis_sph_ml(cfg, im, Sl, ltr)
end
