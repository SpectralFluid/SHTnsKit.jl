#=
================================================================================
core_transforms.jl - Core Spherical Harmonic Transform Implementations
================================================================================

This file implements the fundamental forward (analysis) and backward (synthesis)
transforms between 2D spatial grids and spherical harmonic coefficients.

TRANSFORM OVERVIEW
------------------
Analysis (spatial → spectral):
    a_lm = ∫∫ f(θ,φ) Y_l^m*(θ,φ) sin(θ) dθ dφ

    Implementation:
    1. FFT along φ: f(θ,φ) → F_m(θ) for each m = 0..mmax
    2. Legendre integration: a_lm = Σ_θ w(θ) * F_m(θ) * P_l^m(cos θ) * N_lm

Synthesis (spectral → spatial):
    f(θ,φ) = Σ_l Σ_m a_lm Y_l^m(θ,φ)

    Implementation:
    1. Legendre summation: F_m(θ) = Σ_l a_lm * P_l^m(cos θ) * N_lm
    2. Inverse FFT along φ: F_m(θ) → f(θ,φ)

KEY FUNCTIONS
-------------
- analysis(cfg, f)       : Forward transform, allocates output
- analysis!(cfg, alm, f) : Forward transform, in-place output
- synthesis(cfg, alm)    : Backward transform, allocates output
- synthesis!(cfg, f, alm): Backward transform, in-place output

IMPLEMENTATION ARCHITECTURE
---------------------------
This file uses a 3-layer architecture:
1. Public API (analysis, synthesis, analysis!, synthesis!)
   - Input validation, FFT, output allocation
2. Orchestrators (_analysis_scalar_mloop!, _synthesis_scalar_mloop!)
   - Threading over m-modes, table vs on-the-fly dispatch
3. Kernels (in kernels.jl)
   - Per-latitude Legendre accumulation, single source of truth

LEGENDRE COMPUTATION MODES
---------------------------
1. Table-based (cfg.use_plm_tables=true):
   - Uses precomputed Legendre polynomial tables
   - Faster but requires more memory

2. On-the-fly (cfg.on_the_fly=true):
   - Computes Legendre polynomials during transform
   - Lower memory, slightly slower

MULTITHREADING
--------------
- Transforms parallelize over m (azimuthal order)
- Each thread has its own Legendre polynomial buffer
- Use `Threads.nthreads()` to check available threads

NUMERICAL PRECISION
-------------------
- Roundtrip error: f ≈ synthesis(cfg, analysis(cfg, f))
- Expected: O(ε) where ε is machine epsilon (~1e-15 for Float64)
- Larger errors may indicate: insufficient nlat, nlon, or numerical instability

DEBUGGING CHECKLIST
-------------------
1. Dimension checks:
   - f must be (nlat, nlon)
   - alm must be (lmax+1, mmax+1)

2. Coefficient interpretation:
   - alm[1,1] = a_{0,0}: mean value (monopole)
   - alm[2,1] = a_{1,0}: north-south gradient
   - alm[2,2] = a_{1,1}: east-west gradient

3. Common issues:
   - Wrong grid size → DimensionMismatch error
   - alm not zeroed → accumulation of old values
   - Missing normalization → wrong magnitudes

EXAMPLE USAGE
-------------
```julia
cfg = create_gauss_config(32, 48)

# Create test field (Y_2^0 pattern)
f = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat
    f[i, :] .= 0.5 * (3*cfg.x[i]^2 - 1)  # P_2^0(x)
end

# Transform to spectral
alm = analysis(cfg, f)
@assert abs(alm[3,1]) > 0.1  # a_{2,0} should be non-zero

# Transform back
f2 = synthesis(cfg, alm)
@assert maximum(abs, f - f2) < 1e-12
```

================================================================================
=#

"""
Core Spherical Harmonic Transforms

Fundamental forward (analysis) and backward (synthesis) transforms between
2D spatial grids and spherical harmonic spectra. Uses Gauss–Legendre quadrature
in latitude and FFT along longitude. Orthonormal spherical harmonics with
Condon–Shortley phase are used internally; normalization conversion is handled
by higher-level helpers when needed.
"""

# ============================================================================
# THREAD UTILIZATION DIAGNOSTICS
# ============================================================================

"""
    check_thread_utilization(cfg::SHTConfig; warn::Bool=true) -> NamedTuple

Check if threading configuration is optimal for this transform configuration.
Returns utilization statistics and optionally warns about suboptimal setups.

SHTnsKit parallelizes transforms over m-modes (azimuthal order) using
`@threads :static`. If `mmax + 1 < nthreads`, some threads will be idle.

# Arguments
- `cfg::SHTConfig`: The transform configuration to check
- `warn::Bool=true`: If true, emit a warning when utilization is below 50%

# Returns
A named tuple with:
- `nthreads`: Number of Julia threads available
- `mmax`: Maximum azimuthal order from config
- `active_threads`: Number of threads that will actually do work
- `utilization`: Fraction of threads utilized (0.0 to 1.0)

# Example
```julia
cfg = create_gauss_config(8, 10)  # Small problem with mmax=8
stats = check_thread_utilization(cfg)
# If running with 16 threads, this warns about only 9 active threads (56%)

# For better utilization, either:
# 1. Use fewer threads: julia -t 8
# 2. Use larger mmax: cfg = create_gauss_config(32, 34)
```

See also: [`configure_threading!`](@ref)
"""
function check_thread_utilization(cfg::SHTConfig; warn::Bool=true)
    nthreads = Threads.nthreads()
    mmax = cfg.mmax

    # With @threads :static for m in 0:mmax, we have (mmax + 1) work items
    work_items = mmax + 1
    active_threads = min(nthreads, work_items)
    utilization = nthreads > 0 ? active_threads / nthreads : 1.0

    if warn && utilization < 0.5 && nthreads > 1
        @warn "Thread underutilization detected" mmax=mmax nthreads=nthreads active_threads=active_threads utilization_pct=round(utilization*100, digits=1) suggestion="Consider using fewer threads (julia -t $(work_items)) or larger mmax"
    end

    return (nthreads=nthreads, mmax=mmax, active_threads=active_threads,
            utilization=utilization)
end

"""
    analysis(cfg::SHTConfig, f::AbstractMatrix) -> Matrix{ComplexF64}

Forward transform on Gauss-Legendre x equiangular grid.
Returns coefficients `alm[l+1, m+1]` in the normalization, real-normalization,
and phase convention configured by `cfg`.

Parallelizes over m-modes using static scheduling for consistent performance.
"""
function analysis(cfg::SHTConfig, f::AbstractMatrix; fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing, use_rfft::Bool=false)
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))
    if use_rfft
        # Real FFTs expose bins 0..nlon/2. These align with the full complex
        # FFT columns used by the Legendre m-loop for all supported m values.
        eltype(f) <: Real || throw(ArgumentError("use_rfft=true requires a real-valued input"))
        nbins = nlon ÷ 2 + 1
        Fph = if fft_scratch === nothing
            rfft_phi(f)                   # (nlat, nlon÷2+1)
        else
            size(fft_scratch) == (nlat, nbins) || throw(ArgumentError("fft_scratch size must be (nlat, nlon÷2+1) for use_rfft=true"))
            rfft_phi!(fft_scratch, f)
        end
    else
        # Use the in-place, cached-plan FFT for both paths. The no-scratch path
        # allocates a single complex buffer instead of the two temporaries the
        # old `fft_phi(_as_complex(f))` produced (a `complex.(f)` copy plus an
        # out-of-place `fft` that re-planned each call). eltype is preserved so
        # AD element types still take fft_phi!'s DFT fallback.
        if fft_scratch === nothing
            CT = complex(float(eltype(f)))
            Fph = fft_phi!(Matrix{CT}(undef, nlat, nlon), f)
        else
            Fph = fft_phi!(fft_scratch, f)
        end
    end
    CT = eltype(Fph)
    alm = zeros(CT, cfg.lmax + 1, cfg.mmax + 1)
    _analysis_scalar_mloop!(alm, cfg, Fph)
    return _externalize_coefficients!(alm, cfg)
end

"""
    analysis!(cfg::SHTConfig, alm_out::AbstractMatrix, f::AbstractMatrix)

In-place forward transform. Writes coefficients into `alm_out`.
`alm_out` must be size `(lmax+1, mmax+1)`.
"""
Base.@constprop :aggressive function analysis!(cfg::SHTConfig, alm_out::AbstractMatrix, f::AbstractMatrix; fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing, use_rfft::Bool=false)
    # Function barrier: `Val(use_rfft)` makes the scratch shape and FFT branch
    # compile-time constants for callers that pass a literal keyword.
    return _analysis!(cfg, alm_out, f, fft_scratch, Val(use_rfft))
end

function _analysis!(cfg::SHTConfig, alm_out::AbstractMatrix, f::AbstractMatrix,
                    fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}},
                    ::Val{use_rfft}) where {use_rfft}
    size(alm_out, 1) == cfg.lmax + 1 || throw(DimensionMismatch("alm_out first dim must be lmax+1=$(cfg.lmax+1)"))
    size(alm_out, 2) == cfg.mmax + 1 || throw(DimensionMismatch("alm_out second dim must be mmax+1=$(cfg.mmax+1)"))
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f, 1) == nlat || throw(DimensionMismatch("first dim must be nlat=$(nlat)"))
    size(f, 2) == nlon || throw(DimensionMismatch("second dim must be nlon=$(nlon)"))
    fill!(alm_out, zero(eltype(alm_out)))
    if use_rfft
        eltype(f) <: Real || throw(ArgumentError("use_rfft=true requires a real-valued input"))
        nbins = nlon ÷ 2 + 1
        Fph = if fft_scratch === nothing
            rfft_phi(f)
        else
            size(fft_scratch) == (nlat, nbins) || throw(ArgumentError("fft_scratch size must be (nlat, nlon÷2+1) for use_rfft=true"))
            rfft_phi!(fft_scratch, f)
        end
    else
        # Use the in-place, cached-plan FFT for both paths. The no-scratch path
        # allocates a single complex buffer instead of the two temporaries the
        # old `fft_phi(_as_complex(f))` produced (a `complex.(f)` copy plus an
        # out-of-place `fft` that re-planned each call). eltype is preserved so
        # AD element types still take fft_phi!'s DFT fallback.
        if fft_scratch === nothing
            CT = complex(float(eltype(f)))
            Fph = fft_phi!(Matrix{CT}(undef, nlat, nlon), f)
        else
            Fph = fft_phi!(fft_scratch, f)
        end
    end
    _analysis_scalar_mloop!(alm_out, cfg, Fph)
    return _externalize_coefficients!(alm_out, cfg)
end

"""
    synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true) -> Matrix

Inverse transform back to a grid `(nlat, nlon)`. If `real_output=true`,
Hermitian symmetry is enforced before IFFT.
"""
Base.@constprop :aggressive function synthesis(cfg::SHTConfig, alm::AbstractMatrix; real_output::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing, use_rfft::Bool=false)
    # Function barrier: output element type and FFT buffer shape depend on
    # keyword values, so route literals through Val-dispatched methods.
    return _synthesis(cfg, alm, Val(real_output), fft_scratch, Val(use_rfft))
end

"""
    synthesis_cplx(cfg::SHTConfig, alm::AbstractMatrix) -> Matrix{<:Complex}

Complex-output inverse scalar transform. This is equivalent to
`synthesis(cfg, alm; real_output=false)` but dispatches through a compile-time
output mode so callers can rely on a concrete return type.
"""
function synthesis_cplx(cfg::SHTConfig, alm::AbstractMatrix)
    return _synthesis(cfg, alm, Val(false), nothing, Val(false))
end

function _synthesis(cfg::SHTConfig, alm::AbstractMatrix, ::Val{real_output},
                    fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}},
                    ::Val{use_rfft}) where {real_output, use_rfft}
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))
    alm_int = _internal_coefficients(alm, cfg)
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    if use_rfft
        # The rfft path returns real output by construction. Hermitian
        # symmetry is not materialized in Fph; irfft_phi! reconstructs it.
        real_output === true || throw(ArgumentError("use_rfft=true implies real_output"))
        mmax ≤ nlon ÷ 2 || throw(ArgumentError("use_rfft=true requires mmax ≤ nlon÷2, got mmax=$mmax, nlon=$nlon"))
        nbins = nlon ÷ 2 + 1
        Fph = if fft_scratch === nothing
            Matrix{CT}(undef, nlat, nbins)
        else
            size(fft_scratch) == (nlat, nbins) || throw(ArgumentError("fft_scratch size must be (nlat, nlon÷2+1) for use_rfft=true"))
            fft_scratch
        end
        fill!(Fph, zero(CT))
        _synthesis_scalar_mloop!(Fph, cfg, alm_int; real_output=false, use_rfft=true)
        RT = real(CT)
        out = Matrix{RT}(undef, nlat, nlon)
        irfft_phi!(out, Fph, nlon)
        return out
    end
    Fph = fft_scratch === nothing ? Matrix{CT}(undef, nlat, nlon) : fft_scratch
    size(Fph, 1) == nlat && size(Fph, 2) == nlon || throw(DimensionMismatch("fft_scratch wrong size"))
    fill!(Fph, zero(CT))
    _synthesis_scalar_mloop!(Fph, cfg, alm_int; real_output=real_output)
    ifft_phi!(Fph, Fph)
    if real_output
        RT = typeof(real(zero(CT)))
        out = Matrix{RT}(undef, nlat, nlon)
        @inbounds for j in 1:nlon, i in 1:nlat
            out[i, j] = real(Fph[i, j])
        end
        return out
    else
        return Fph
    end
end

function _synthesis_l(cfg::SHTConfig, alm::AbstractMatrix, ltr::Int, ::Val{real_output}) where {real_output}
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))
    alm_int = _internal_coefficients(alm, cfg)
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    Fph = Matrix{CT}(undef, nlat, nlon)
    fill!(Fph, zero(CT))
    _synthesis_scalar_mloop!(Fph, cfg, alm_int; real_output=real_output, ltr=ltr)
    ifft_phi!(Fph, Fph)
    if real_output
        RT = typeof(real(zero(CT)))
        out = Matrix{RT}(undef, nlat, nlon)
        @inbounds for j in 1:nlon, i in 1:nlat
            out[i, j] = real(Fph[i, j])
        end
        return out
    else
        return Fph
    end
end

"""
    synthesis!(cfg::SHTConfig, f_out::AbstractMatrix, alm::AbstractMatrix; real_output::Bool=true)

In-place inverse transform. Writes the spatial field into `f_out`.
"""
Base.@constprop :aggressive function synthesis!(cfg::SHTConfig, f_out::AbstractMatrix, alm::AbstractMatrix; real_output::Bool=true, fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}}=nothing, use_rfft::Bool=false)
    # Same Val barrier as out-of-place synthesis, but preserving the caller's
    # output buffer contract.
    return _synthesis!(cfg, f_out, alm, Val(real_output), fft_scratch, Val(use_rfft))
end

function _synthesis!(cfg::SHTConfig, f_out::AbstractMatrix, alm::AbstractMatrix,
                     ::Val{real_output},
                     fft_scratch::Union{Nothing,AbstractMatrix{<:Complex}},
                     ::Val{use_rfft}) where {real_output, use_rfft}
    size(f_out, 1) == cfg.nlat || throw(DimensionMismatch("f_out first dim must be nlat=$(cfg.nlat)"))
    size(f_out, 2) == cfg.nlon || throw(DimensionMismatch("f_out second dim must be nlon=$(cfg.nlon)"))
    lmax, mmax = cfg.lmax, cfg.mmax
    size(alm, 1) == lmax + 1 || throw(DimensionMismatch("first dim must be lmax+1=$(lmax+1)"))
    size(alm, 2) == mmax + 1 || throw(DimensionMismatch("second dim must be mmax+1=$(mmax+1)"))
    alm_int = _internal_coefficients(alm, cfg)
    nlat, nlon = cfg.nlat, cfg.nlon
    CT = eltype(alm)
    if use_rfft
        real_output || throw(ArgumentError("use_rfft=true implies real_output"))
        eltype(f_out) <: Real || throw(ArgumentError("use_rfft=true requires real-valued f_out"))
        mmax ≤ nlon ÷ 2 || throw(ArgumentError("use_rfft=true requires mmax ≤ nlon÷2, got mmax=$mmax, nlon=$nlon"))
        nbins = nlon ÷ 2 + 1
        Fph = if fft_scratch === nothing
            Matrix{CT}(undef, nlat, nbins)
        else
            size(fft_scratch) == (nlat, nbins) || throw(ArgumentError("fft_scratch size must be (nlat, nlon÷2+1) for use_rfft=true"))
            fft_scratch
        end
        fill!(Fph, zero(eltype(Fph)))
        _synthesis_scalar_mloop!(Fph, cfg, alm_int; real_output=false, use_rfft=true)
        irfft_phi!(f_out, Fph, nlon)
        return f_out
    end
    Fph = fft_scratch === nothing ? Matrix{CT}(undef, nlat, nlon) : fft_scratch
    size(Fph, 1) == nlat && size(Fph, 2) == nlon || throw(DimensionMismatch("fft_scratch wrong size"))
    fill!(Fph, zero(eltype(Fph)))
    _synthesis_scalar_mloop!(Fph, cfg, alm_int; real_output=real_output)
    ifft_phi!(Fph, Fph)
    if real_output
        @inbounds for j in 1:nlon, i in 1:nlat
            f_out[i, j] = real(Fph[i, j])
        end
    else
        @inbounds for j in 1:nlon, i in 1:nlat
            f_out[i, j] = Fph[i, j]
        end
    end
    return f_out
end

# ============================================================================
# SCALAR ORCHESTRATORS
# ============================================================================

"""
    _adjoint_synthesis(cfg, f̄; θ_globals=1:cfg.nlat, real_output=true)
        -> Matrix{ComplexF64}

True mathematical adjoint of `synthesis`. NOT the same as `analysis` — the
analysis operator carries quadrature weights `w_i` and the `cphi = 2π/nlon`
azimuthal scaling, neither of which belongs in the synthesis adjoint.

Forward (real output, `inv_scaleφ = nlon` cancels the `1/nlon` in `ifft`):
    f[i,j] = Re(c_0[i]) + 2 Σ_{m=1..mmax} Re(c_m[i] · exp(i·m·φ_j))
    where c_m[i] = Σ_l Nlm · P_l^m(θ_i) · alm[l,m]

Wirtinger adjoint, Zygote convention (`g = 2 · ∂L/∂ā`):

For m > 0, forward synthesis writes both `alm[l,m]·q` (at bin m) AND
`ā[l,m]·conj(q)` (at bin nlon-m via Hermitian fill), with `q = Nlm·P·exp(imφ)`.
Both halves contribute to the Wirtinger gradient:
    ∂L/∂ā[l,m>0] = Σ ȳ · conj(q) = Σ_i Nlm·P_l^m · FFT(f̄)[m+1]
    g[l,m>0]     = 2 · ∂L/∂ā                                    (Zygote scaling)

For m = 0, `alm[l,0]` is a complex parameter but only Re(alm[l,0]) enters f
(because the forward pipeline takes `real.(Fph)` at the end). That gives
`∂f/∂alm = ∂f/∂ā = ½ Nlm·P_l^0`, so `g[l,0] = 2 · ½ · FFT(ȳ)[1] · Σ = FFT(ȳ)[1] · Σ`.

Net effect:
    g[l,m]  = wm(m) · Σ_i Nlm · P_l^m · FFT(f̄[i,:])[m+1]
    wm(0) = 1,  wm(m>0) = 2

No `conj` on the FFT bin (FFT already carries `exp(-imφ)`). No quadrature
weights or `cphi` — those belong to `analysis`, not the synthesis adjoint.

The forward's `inv_scaleφ = phi_inv_scale(cfg)` does NOT cancel against the
`1/nlon` in the ifft adjoint in general: the surviving factor is
`phi_inv_scale(cfg)/nlon`, which is 1 in the `:dft` mode but `1/2π` under
`:quad`. Treating it as always-1 made every synthesis-family gradient exactly
2π too large whenever `cfg.phi_scale === :quad` or `SHTNSKIT_PHI_SCALE=quad`.
"""
function _adjoint_synthesis(cfg::SHTConfig, f̄::AbstractMatrix;
                            θ_globals::AbstractVector{<:Integer}=1:cfg.nlat,
                            real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    # FFT along φ. For real f̄ the FFTW rfft would be enough, but we accept
    # complex f̄ too so use the general routine. One buffer + cached FFTW plan;
    # the old `fft_phi(_as_complex(f̄))` made a complex copy AND re-planned an
    # out-of-place fft each call. eltype preserved for the AD/DFT fallback.
    Fph̄ = fft_phi!(Matrix{complex(float(eltype(f̄)))}(undef, size(f̄)...), f̄)
    # eltype-derived accumulator (mirrors _adjoint_synthesis_sphtor): keep Float32
    # / ForwardDiff.Dual cotangents intact instead of upcasting to ComplexF64.
    ālm = zeros(complex(float(eltype(f̄))), lmax + 1, mmax + 1)
    use_tbl = has_fused_scalar_tables(cfg)
    P = use_tbl ? nothing : Vector{Float64}(undef, lmax + 1)
    xv = cfg.x  # hoist field read out of the m/l loops (cfg is mutable, so not auto-hoisted)
    # Forward scales bins by `inv_scaleφ = phi_inv_scale(cfg)`; adjoint of `ifft`
    # is `(1/nlon)·fft`. The surviving factor `inv_scaleφ/nlon` is 1 only in the
    # `:dft` mode — under `:quad` it is `1/2π`, and assuming it cancelled made
    # every synthesis-family gradient exactly 2π too large. See the docstring.
    φadj = phi_inv_scale(cfg) / cfg.nlon
    for m in 0:mmax
        col = m + 1
        wm = ((m == 0 || !real_output) ? 1.0 : 2.0) * φadj
        if use_tbl
            NP = cfg.NP_tables[col]
            @inbounds for (ii, iglob) in pairs(θ_globals)
                Fi = wm * Fph̄[ii, col]
                for l in m:lmax
                    ālm[l+1, col] += NP[l+1, iglob] * Fi
                end
            end
        else
            @inbounds for (ii, iglob) in pairs(θ_globals)
                Plm_norm_row!(P, xv[iglob], lmax, m)
                Fi = wm * Fph̄[ii, col]
                for l in m:lmax
                    ālm[l+1, col] += P[l+1] * Fi
                end
            end
        end
    end
    return _synthesis_cotangent_to_configured!(ālm, cfg)
end

"""
    _adjoint_synthesis_sphtor(cfg, V̄t, V̄p; θ_globals=1:cfg.nlat, real_output=true)

Adjoint operator of `synthesis_sphtor` (spheroidal/toroidal vector synthesis).
Maps spatial cotangents (V̄t, V̄p) → (S̄lm, T̄lm). Like `_adjoint_synthesis` this
carries NO Gauss quadrature weights or `cphi` (those belong to *analysis*, not
the synthesis adjoint), and applies the `wm` Hermitian doubling for real output.

The synthesis matrix M = [[dθY, -term]; [term, dθY]] (term = i·m·Y/sinθ) is
Hermitian, so its adjoint reuses the same matrix with conjugated off-diagonal.

Parametric over `θ_globals` so the distributed adjoint (rank-local θ slab) reuses
this exact helper; pass a full-`nlon`-width (zero-padded) cotangent and Allreduce
the partial (S̄, T̄) across ranks. Accumulator eltypes derive from the inputs so
ForwardDiff `Dual` / `Float32` cotangents flow through.
"""
function _adjoint_synthesis_sphtor(cfg::SHTConfig, V̄t::AbstractMatrix, V̄p::AbstractMatrix;
                                   θ_globals::AbstractVector{<:Integer}=1:cfg.nlat,
                                   real_output::Bool=true)
    lmax, mmax = cfg.lmax, cfg.mmax
    CT = complex(float(promote_type(eltype(V̄t), eltype(V̄p))))
    # Adjoint of ifft_phi is fft_phi; one buffer per field + cached FFTW plan.
    F̄θ = fft_phi!(Matrix{CT}(undef, size(V̄t)...), V̄t)
    F̄φ = fft_phi!(Matrix{CT}(undef, size(V̄p)...), V̄p)

    S̄ = zeros(CT, lmax + 1, mmax + 1)
    T̄ = zeros(CT, lmax + 1, mmax + 1)

    P = Vector{Float64}(undef, lmax + 1)
    dPdtheta = Vector{Float64}(undef, lmax + 1)
    P_over_sinth = Vector{Float64}(undef, lmax + 1)
    Pbuf = Vector{Float64}(undef, lmax + 2)  # scratch for normalized dθ recurrence
    xv = cfg.x  # hoist field read out of the m/i/l loops (cfg is mutable)

    # The forward scales its Fourier bins by `inv_scaleφ = phi_inv_scale(cfg)`
    # and the adjoint of `ifft` is `(1/nlon)·fft`, so the surviving factor is
    # `inv_scaleφ / nlon`. That is 1 only in the `:dft` mode; under `:quad`
    # (`phi_scale=:quad`, or SHTNSKIT_PHI_SCALE=quad) it is `1/2π`, and assuming
    # the cancellation made every gradient exactly 2π too large.
    # wm accounts for the Hermitian "fill conjugate" step (real output).
    φadj = phi_inv_scale(cfg) / cfg.nlon
    for m in 0:mmax
        col = m + 1
        wm = ((m == 0 || !real_output) ? 1.0 : 2.0) * φadj
        @inbounds for (ii, iglob) in pairs(θ_globals)
            x = xv[iglob]
            Plm_norm_dPdtheta_over_sinth_row!(P, dPdtheta, P_over_sinth, x, lmax, m, Pbuf)
            Fθ_im = F̄θ[ii, col]
            Fφ_im = F̄φ[ii, col]
            for l in max(1, m):lmax
                dθY = dPdtheta[l+1]
                term = 1.0im * m * P_over_sinth[l+1]
                # Hermitian adjoint of [dθY, -term; term, dθY]:
                S̄[l+1, col] += wm * (dθY * Fθ_im + conj(term) * Fφ_im)
                T̄[l+1, col] += wm * (-conj(term) * Fθ_im + dθY * Fφ_im)
            end
        end
    end
    _synthesis_cotangent_to_configured!(S̄, cfg)
    _synthesis_cotangent_to_configured!(T̄, cfg)
    return S̄, T̄
end

"""
    _adjoint_analysis(cfg, Alm̄; θ_globals=1:cfg.nlat, φ_window=nothing)

Adjoint operator of `analysis`. Shares the scalar synthesis kernels; only the
per-row scale differs (`φadj * cfg.w[i]` instead of `inv_scale_phi`). Negative-m
columns stay zero — the adjoint places mass only in measured bins.

Parametric over the θ set we integrate against so the distributed adjoint
(rank-local θ slab) can reuse this exact helper with a different `θ_globals`
range and optional φ-window slice. Lives in src so AD extensions don't have
to duplicate it.
"""
function _adjoint_analysis(cfg::SHTConfig, Alm̄::AbstractMatrix;
                           θ_globals::AbstractVector{<:Integer}=1:cfg.nlat,
                           φ_window::Union{Nothing,UnitRange{Int}}=nothing)
    nlon = cfg.nlon
    nlat_local = length(θ_globals)
    Alm̄_int = _analysis_cotangent_to_canonical(Alm̄, cfg)
    # eltype-derived buffers so Float32 / ForwardDiff.Dual cotangents survive.
    CT = complex(float(eltype(Alm̄_int)))
    Fφ = Matrix{CT}(undef, nlat_local, nlon)
    fill!(Fφ, zero(eltype(Fφ)))
    lmax, mmax = cfg.lmax, cfg.mmax
    φadj = 2π  # nlon (ifft adjoint) × cphi (2π/nlon) = 2π
    use_tbl = has_fused_scalar_tables(cfg)
    P = use_tbl ? nothing : Vector{Float64}(undef, lmax + 1)
    for m in 0:mmax
        col = m + 1
        if use_tbl
            NP = cfg.NP_tables[m+1]
            @inbounds for (ii, iglob) in pairs(θ_globals)
                Fφ[ii, col] = (φadj * cfg.w[iglob]) *
                    _scalar_synthesis_kernel(cfg, Alm̄_int, NP, iglob, col, m, lmax)
            end
        else
            @inbounds for (ii, iglob) in pairs(θ_globals)
                Fφ[ii, col] = (φadj * cfg.w[iglob]) *
                    _scalar_synthesis_kernel_otf(cfg, Alm̄_int, P, iglob, col, m, lmax)
            end
        end
    end
    ifft_phi!(Fφ, Fφ)
    if φ_window === nothing
        return real.(Fφ)
    end
    out = Matrix{real(CT)}(undef, nlat_local, length(φ_window))
    @inbounds for (jj, jglob) in pairs(φ_window)
        for i in 1:nlat_local
            out[i, jj] = real(Fφ[i, jglob])
        end
    end
    return out
end

"""Scalar analysis orchestrator. Parallelizes Legendre integration over m-modes."""
function _analysis_scalar_mloop!(alm::AbstractMatrix, cfg::SHTConfig, Fph::AbstractMatrix)
    lmax, mmax = cfg.lmax, cfg.mmax
    scale_phi = cfg.cphi
    m_order = cached_m_order(cfg)
    if has_fused_scalar_tables(cfg)
        _analysis_scalar_mloop_tbl!(alm, cfg, Fph, m_order, scale_phi)
    else
        _analysis_scalar_mloop_otf!(alm, cfg, Fph, m_order, scale_phi)
    end
    return alm
end

@inline function _analysis_scalar_mloop_tbl!(alm, cfg, Fph, m_order, scale_phi)
    # Avoid @threads setup and closure allocation on single-thread runs. The
    # serial and threaded bodies stay separate so performance tests can lock
    # down the low-allocation path.
    if Threads.nthreads() == 1
        return _analysis_scalar_mloop_tbl_serial!(alm, cfg, Fph, m_order, scale_phi)
    else
        return _analysis_scalar_mloop_tbl_threaded!(alm, cfg, Fph, m_order, scale_phi)
    end
end

@inline function _analysis_scalar_mloop_tbl_serial!(alm, cfg, Fph, m_order, scale_phi)
    lmax = cfg.lmax
    nlat = cfg.nlat
    @inbounds for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        NP = cfg.NP_tables[m+1]
        for i in 1:nlat
            _scalar_analysis_kernel!(alm, cfg, Fph, NP, i, col, m, lmax, scale_phi)
        end
    end
    return alm
end

@inline function _analysis_scalar_mloop_tbl_threaded!(alm, cfg, Fph, m_order, scale_phi)
    lmax = cfg.lmax
    nlat = cfg.nlat
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        NP = cfg.NP_tables[m+1]
        @inbounds for i in 1:nlat
            _scalar_analysis_kernel!(alm, cfg, Fph, NP, i, col, m, lmax, scale_phi)
        end
    end
    return alm
end

@inline function _analysis_scalar_mloop_otf!(alm, cfg, Fph, m_order, scale_phi)
    lmax = cfg.lmax
    thread_local_P = _ensure_otf_scratch!(cfg._otf_scratch_P, lmax)
    # See table path above: single-thread transforms use a direct loop and
    # the first scratch buffer instead of paying threaded-loop overhead.
    if Threads.nthreads() == 1
        return _analysis_scalar_mloop_otf_serial!(alm, cfg, Fph, m_order, scale_phi, thread_local_P, lmax)
    else
        return _analysis_scalar_mloop_otf_threaded!(alm, cfg, Fph, m_order, scale_phi, thread_local_P, lmax)
    end
end

@inline function _analysis_scalar_mloop_otf_serial!(alm, cfg, Fph, m_order, scale_phi, thread_local_P, lmax)
    nlat = cfg.nlat
    P = thread_local_P[1]
    @inbounds for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        for i in 1:nlat
            _scalar_analysis_kernel_otf!(alm, cfg, Fph, P, i, col, m, lmax, scale_phi)
        end
    end
    return alm
end

@inline function _analysis_scalar_mloop_otf_threaded!(alm, cfg, Fph, m_order, scale_phi, thread_local_P, lmax)
    nlat = cfg.nlat
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        P = thread_local_P[Threads.threadid()]
        @inbounds for i in 1:nlat
            _scalar_analysis_kernel_otf!(alm, cfg, Fph, P, i, col, m, lmax, scale_phi)
        end
    end
    return alm
end

"""Scalar synthesis orchestrator. Parallelizes Legendre summation over m-modes."""
function _synthesis_scalar_mloop!(Fph::AbstractMatrix, cfg::SHTConfig, alm::AbstractMatrix;
                                   real_output::Bool=true, use_rfft::Bool=false,
                                   ltr::Int=cfg.lmax)
    lmax, mmax = cfg.lmax, cfg.mmax
    ltr_eff = min(lmax, ltr)
    nlat, nlon = cfg.nlat, cfg.nlon
    inv_scale_phi = phi_inv_scale(cfg)
    m_order = cached_m_order(cfg)
    if has_fused_scalar_tables(cfg)
        _synthesis_scalar_mloop_tbl!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff)
    else
        _synthesis_scalar_mloop_otf!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff)
    end
    # Fill Hermitian conjugate columns for real-output IFFT — only the full
    # complex buffer (size nlon) can hold these. rfft's output half has no
    # slot for the negative-m bins; irfft_phi! reconstructs them implicitly.
    if real_output && !use_rfft
        for m in 1:mmax
            col = m + 1
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                Fph[i, conj_index] = conj(Fph[i, col])
            end
        end
    end
    return Fph
end

@inline function _synthesis_scalar_mloop_tbl!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff)
    # Keep the single-thread path free of @threads overhead. This matters for
    # allocation-sensitive small transforms and scalar in-place APIs.
    if Threads.nthreads() == 1
        return _synthesis_scalar_mloop_tbl_serial!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff)
    else
        return _synthesis_scalar_mloop_tbl_threaded!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff)
    end
end

@inline function _synthesis_scalar_mloop_tbl_serial!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff)
    nlat = cfg.nlat
    @inbounds for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        NP = cfg.NP_tables[m+1]
        for i in 1:nlat
            Fph[i, col] = inv_scale_phi * _scalar_synthesis_kernel(cfg, alm, NP, i, col, m, ltr_eff)
        end
    end
    return nothing
end

@inline function _synthesis_scalar_mloop_tbl_threaded!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff)
    nlat = cfg.nlat
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        NP = cfg.NP_tables[m+1]
        @inbounds for i in 1:nlat
            Fph[i, col] = inv_scale_phi * _scalar_synthesis_kernel(cfg, alm, NP, i, col, m, ltr_eff)
        end
    end
end

@inline function _synthesis_scalar_mloop_otf!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff)
    thread_local_P = _ensure_otf_scratch!(cfg._otf_scratch_P, max(ltr_eff, 0))
    # `thread_local_P[1]` is valid in serial mode because _ensure_otf_scratch!
    # sizes scratch for maxthreadid().
    if Threads.nthreads() == 1
        return _synthesis_scalar_mloop_otf_serial!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff, thread_local_P)
    else
        return _synthesis_scalar_mloop_otf_threaded!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff, thread_local_P)
    end
end

@inline function _synthesis_scalar_mloop_otf_serial!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff, thread_local_P)
    nlat = cfg.nlat
    P = thread_local_P[1]
    @inbounds for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        for i in 1:nlat
            Fph[i, col] = inv_scale_phi * _scalar_synthesis_kernel_otf(cfg, alm, P, i, col, m, ltr_eff)
        end
    end
    return nothing
end

@inline function _synthesis_scalar_mloop_otf_threaded!(Fph, cfg, alm, m_order, inv_scale_phi, ltr_eff, thread_local_P)
    nlat = cfg.nlat
    @threads :static for idx in 1:length(m_order)
        m = m_order[idx]
        col = m + 1
        P = thread_local_P[Threads.threadid()]
        @inbounds for i in 1:nlat
            Fph[i, col] = inv_scale_phi * _scalar_synthesis_kernel_otf(cfg, alm, P, i, col, m, ltr_eff)
        end
    end
end
