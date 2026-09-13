#=
================================================================================
plan.jl - Optimized Transform Planning for Spherical Harmonic Operations
================================================================================

This file implements a planning system for spherical harmonic transforms that
pre-allocates working arrays and FFT plans to minimize runtime overhead.

MOTIVATION: WHY PLANNING?
-------------------------
The basic transform functions (analysis, synthesis) allocate temporary arrays
on every call:
- Legendre polynomial working arrays: O(lmax)
- Fourier coefficient matrices: O(nlat × nlon)
- FFT planning overhead

For applications that call transforms repeatedly (e.g., time-stepping PDEs),
these allocations dominate runtime and cause GC pressure.

THE PLANNING APPROACH
---------------------
Inspired by FFTW: spend time upfront to optimize repeated operations.

1. Pre-allocate ALL working arrays once
2. Pre-compute optimized FFTW plans (can be slow but only done once)
3. Reuse everything across many transform calls

Result: Near-zero allocations per transform call.

IN-PLACE TRANSFORM FUNCTIONS
----------------------------
    analysis!(plan, alm_out, f)             : f → alm (scalar)
    synthesis!(plan, f_out, alm)            : alm → f (scalar)
    analysis_sphtor!(plan, S, T, Vt, Vp)   : (Vt,Vp) → (S,T) (vector)
    synthesis_sphtor!(plan, Vt, Vp, S, T)   : (S,T) → (Vt,Vp) (vector)

USAGE EXAMPLE
-------------
```julia
cfg = create_gauss_config(64, 128)

# Create plan (does all allocation and FFT planning)
plan = SHTPlan(cfg)

# Preallocate output arrays
f_out = Matrix{Float64}(undef, cfg.nlat, cfg.nlon)
alm_out = Matrix{ComplexF64}(undef, cfg.lmax+1, cfg.mmax+1)

# Now transforms are allocation-free
for timestep in 1:10000
    analysis!(plan, alm_out, f)    # No allocations!
    # ... modify alm_out ...
    synthesis!(plan, f_out, alm_out)  # No allocations!
end
```

REAL FFT OPTIMIZATION (use_rfft=true)
-------------------------------------
For real-valued scalar fields, pass `use_rfft=true` to `SHTPlan(cfg; ...)`.
- Fourier buffer reduced from nlon to nlon÷2+1 complex numbers.
- Uses pre-planned `FFTW.plan_rfft` / `FFTW.plan_irfft`.
- Vector sphtor transforms on the same plan still use the complex buffer.

DEBUGGING
---------
```julia
# Check that planned transforms match basic transforms
plan = SHTPlan(cfg)
alm1 = analysis(cfg, f)
alm2 = similar(alm1)
analysis!(plan, alm2, f)
@assert alm1 ≈ alm2

# Benchmark allocation-free operation
using BenchmarkTools
@btime analysis!($plan, $alm_out, $f)  # Should show 0 allocations
```

================================================================================
=#

"""
Optimized Transform Planning for Spherical Harmonic Operations

This module implements a planning system for spherical harmonic transforms that
pre-allocates working arrays and FFT plans to minimize runtime overhead. The
planning approach is inspired by FFTW's philosophy: spend time upfront to
optimize repeated operations.

Benefits of Planning:
- Eliminates repeated memory allocations during transforms
- Pre-optimizes FFTW plans for maximum performance
- Improves cache locality by reusing buffers
- Reduces garbage collection pressure in performance-critical loops

The SHTPlan stores all necessary working arrays and can handle both complex
FFTs and real-optimized FFTs (RFFT) depending on the use case.
"""

"""
    SHTPlan

Pre-allocated working buffers and FFTW plans for zero-allocation transforms.

# Thread Safety

**WARNING:** A single `SHTPlan` instance must NOT be used from multiple threads
simultaneously. The internal Fourier buffers (`Fθk`, `Fφk`, `real_scratch`, …)
are shared mutable state — concurrent calls to `analysis!` or `synthesis!` on the
same plan will produce data races and incorrect results.

For multi-threaded use, create one `SHTPlan` per thread:
```julia
plans = [SHTPlan(cfg) for _ in 1:Threads.nthreads()]
Threads.@threads for i in 1:n
    plan = plans[Threads.threadid()]
    analysis!(plan, alm_out[i], fields[i])
end
```
"""
struct SHTPlan{FP, IP, RP, IRP}
    cfg::SHTConfig                # Configuration parameters
    # Legendre/latitude scratch, kept for compatibility with code that reads
    # these fields. The transforms themselves route through the shared
    # orchestrators in core_transforms.jl / sphtor_transforms.jl, which own
    # per-thread scratch on the config (see `_ensure_otf_scratch!`).
    P::Vector{Float64}            # Working array for Legendre polynomials P_l^m(x)
    dPdx::Vector{Float64}         # Working array for derivatives dP_l^m/dx
    dPdtheta::Vector{Float64}     # Working array for pole-safe derivatives dP_l^m/dθ
    P_over_sinth::Vector{Float64} # Working array for pole-safe P_l^m/sin(θ)
    Pb::Vector{Float64}           # Scratch buffer of length lmax+2 for normalized dθ recurrence
    G::Vector{ComplexF64}         # Temporary array for latitudinal profiles
    Fθk::Matrix{ComplexF64}       # Fourier coefficients of the θ component [latitude × longitude]
    Fφk::Matrix{ComplexF64}       # Fourier coefficients of the φ component; vector transforms hold
                                  # both components at once so one Legendre row serves S and T.
    Fθk_r::Matrix{ComplexF64}     # (nlat, nlon÷2+1) θ buffer for rfft path; 0×0 when use_rfft=false
    Fφk_r::Matrix{ComplexF64}     # (nlat, nlon÷2+1) φ buffer for rfft path; 0×0 when use_rfft=false
    real_scratch::Matrix{Float64} # (nlat, nlon) real scratch for rfft path; 0×0 when use_rfft=false
    real_scratch2::Matrix{Float64}# second real scratch (φ component); 0×0 when use_rfft=false
    fft_plan::FP                  # Pre-optimized forward FFT plan
    ifft_plan::IP                 # Pre-optimized inverse FFT plan
    rfft_plan::RP                 # Real→complex FFT plan (nothing when use_rfft=false)
    irfft_plan::IRP               # Complex→real inverse FFT plan (nothing when use_rfft=false)
    use_rfft::Bool                # Flag: true = use real FFT optimization, false = complex FFT
end

"""
    SHTPlan(cfg::SHTConfig; use_rfft=false)

Create an optimized transform plan with pre-allocated buffers and FFT plans.

This constructor performs the "planning" phase: it allocates all working memory
and optimizes FFTW plans for the specific grid configuration. The resulting
plan can then be reused for many transforms without additional allocations.

Parameters:
- cfg: SHTConfig defining the grid and spectral resolution
- use_rfft: if true, use real-FFT along φ for scalar `analysis!`/`synthesis!`
  (halves the Fourier buffer size). Vector (sphtor) transforms on the plan
  still use the complex buffer. Requires `cfg.mmax ≤ cfg.nlon÷2`.
"""
function SHTPlan(cfg::SHTConfig; use_rfft::Bool=false)
    nlat, nlon = cfg.nlat, cfg.nlon
    if use_rfft && cfg.mmax > nlon ÷ 2
        throw(ArgumentError("use_rfft=true requires mmax ≤ nlon÷2, got mmax=$(cfg.mmax), nlon=$nlon"))
    end

    # Allocate working arrays for Legendre polynomial computation
    P = Vector{Float64}(undef, cfg.lmax + 1)            # P_l^m(cos θ) values
    dPdx = Vector{Float64}(undef, cfg.lmax + 1)         # dP_l^m/d(cos θ) derivatives (legacy)
    dPdtheta = Vector{Float64}(undef, cfg.lmax + 1)     # dP_l^m/dθ pole-safe derivatives
    P_over_sinth = Vector{Float64}(undef, cfg.lmax + 1) # P_l^m/sin(θ) pole-safe
    Pb = Vector{Float64}(undef, cfg.lmax + 2)           # Scratch for normalized dθ recurrence (needs lmax+2)
    G = Vector{ComplexF64}(undef, nlat)                 # Temporary latitudinal profiles

    # Full complex FFT path — always present for vector (sphtor) transforms.
    # BOTH components get their own buffer: the vector transforms contract S and
    # T against the same Legendre row in a single (m, θ) traversal, so Vθ's and
    # Vφ's Fourier data must be live simultaneously. The previous two-pass form
    # shared one buffer and paid for the full Legendre recurrence twice, which
    # made the "optimized" planned vector transform slower than the allocating
    # `cfg`-form one.
    Fθk = Matrix{ComplexF64}(undef, nlat, nlon)
    Fφk = Matrix{ComplexF64}(undef, nlat, nlon)
    fill!(Fθk, zero(ComplexF64))
    fill!(Fφk, zero(ComplexF64))
    fft_plan = FFTW.plan_fft!(Fθk, 2)
    ifft_plan = FFTW.plan_ifft!(Fθk, 2)

    # RFFT-specific buffers and plans
    if use_rfft
        # Scalar planned rfft stores half-width Fourier data. The full complex
        # buffers remain allocated because vector planned transforms still need
        # explicit negative-m columns when not using rfft.
        real_scratch = Matrix{Float64}(undef, nlat, nlon)
        real_scratch2 = Matrix{Float64}(undef, nlat, nlon)
        fill!(real_scratch, 0.0)
        fill!(real_scratch2, 0.0)
        Fθk_r = Matrix{ComplexF64}(undef, nlat, nlon ÷ 2 + 1)
        Fφk_r = Matrix{ComplexF64}(undef, nlat, nlon ÷ 2 + 1)
        fill!(Fθk_r, zero(ComplexF64))
        fill!(Fφk_r, zero(ComplexF64))
        rfft_plan = FFTW.plan_rfft(real_scratch, 2)
        irfft_plan = FFTW.plan_irfft(Fθk_r, nlon, 2)
    else
        real_scratch = Matrix{Float64}(undef, 0, 0)
        real_scratch2 = Matrix{Float64}(undef, 0, 0)
        Fθk_r = Matrix{ComplexF64}(undef, 0, 0)
        Fφk_r = Matrix{ComplexF64}(undef, 0, 0)
        rfft_plan = nothing
        irfft_plan = nothing
    end

    # Convention conversion is a public-boundary operation. Canonical plans
    # remain allocation-free; non-canonical synthesis uses a typed temporary.
    return SHTPlan(cfg, P, dPdx, dPdtheta, P_over_sinth, Pb, G, Fθk, Fφk,
                   Fθk_r, Fφk_r, real_scratch, real_scratch2,
                   fft_plan, ifft_plan, rfft_plan, irfft_plan, use_rfft)
end

"""
    analysis_sphtor!(plan::SHTPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)

In-place vector analysis. Accumulates Slm/Tlm into preallocated outputs.
Uses a two-pass strategy over φ FFTs to avoid extra buffers.
"""
function analysis_sphtor!(plan::SHTPlan, Slm_out::AbstractMatrix, Tlm_out::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon

    size(Vt,1)==nlat && size(Vt,2)==nlon || throw(DimensionMismatch("Vt dims"))
    size(Vp,1)==nlat && size(Vp,2)==nlon || throw(DimensionMismatch("Vp dims"))
    size(Slm_out,1)==cfg.lmax+1 && size(Slm_out,2)==cfg.mmax+1 || throw(DimensionMismatch("Slm_out dims"))
    size(Tlm_out,1)==cfg.lmax+1 && size(Tlm_out,2)==cfg.mmax+1 || throw(DimensionMismatch("Tlm_out dims"))

    fill!(Slm_out, zero(eltype(Slm_out))); fill!(Tlm_out, zero(eltype(Tlm_out)))

    # Transform BOTH components up front into their own buffers, then hand them
    # to the shared orchestrator. Robert-form scaling is applied there (on the
    # Fourier bins), so the copies below are plain copies.
    if plan.use_rfft
        eltype(Vt) <: Real && eltype(Vp) <: Real ||
            throw(ArgumentError("use_rfft plan requires real-valued Vt/Vp"))
        _load_vector_real_scratch!(plan, Vt, Vp)
        mul!(plan.Fθk_r, plan.rfft_plan, plan.real_scratch)
        mul!(plan.Fφk_r, plan.rfft_plan, plan.real_scratch2)
        _analysis_sphtor_mloop!(Slm_out, Tlm_out, cfg, plan.Fθk_r, plan.Fφk_r; ltr=cfg.lmax)
    else
        _load_vector_complex_scratch!(plan, Vt, Vp)
        plan.fft_plan * plan.Fθk
        plan.fft_plan * plan.Fφk
        _analysis_sphtor_mloop!(Slm_out, Tlm_out, cfg, plan.Fθk, plan.Fφk; ltr=cfg.lmax)
    end

    _externalize_coefficients!(Slm_out, cfg)
    _externalize_coefficients!(Tlm_out, cfg)
    return Slm_out, Tlm_out
end

"""Copy `Vt`/`Vp` into the plan's real rfft scratch (Robert scaling happens in the m-loop)."""
function _load_vector_real_scratch!(plan::SHTPlan, Vt::AbstractMatrix, Vp::AbstractMatrix)
    cfg = plan.cfg
    @inbounds for i in 1:cfg.nlat, j in 1:cfg.nlon
        plan.real_scratch[i, j]  = Vt[i, j]
        plan.real_scratch2[i, j] = Vp[i, j]
    end
    return plan
end

"""Copy `Vt`/`Vp` into the plan's complex FFT scratch (Robert scaling happens in the m-loop)."""
function _load_vector_complex_scratch!(plan::SHTPlan, Vt::AbstractMatrix, Vp::AbstractMatrix)
    cfg = plan.cfg
    @inbounds for i in 1:cfg.nlat, j in 1:cfg.nlon
        plan.Fθk[i, j] = Vt[i, j]
        plan.Fφk[i, j] = Vp[i, j]
    end
    return plan
end

function synthesis_sphtor!(plan::SHTPlan, Vt_out::AbstractMatrix, Vp_out::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon

    size(Vt_out,1)==nlat && size(Vt_out,2)==nlon || throw(DimensionMismatch("Vt_out dims"))
    size(Vp_out,1)==nlat && size(Vp_out,2)==nlon || throw(DimensionMismatch("Vp_out dims"))
    size(Slm,1)==cfg.lmax+1 && size(Slm,2)==cfg.mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1)==cfg.lmax+1 && size(Tlm,2)==cfg.mmax+1 || throw(DimensionMismatch("Tlm dims"))

    Slm_int = _internal_coefficients(Slm, cfg)
    Tlm_int = _internal_coefficients(Tlm, cfg)

    if plan.use_rfft
        real_output || throw(ArgumentError("synthesis_sphtor! with use_rfft plan requires real_output=true"))
        eltype(Vt_out) <: Real && eltype(Vp_out) <: Real ||
            throw(ArgumentError("use_rfft plan requires real-valued Vt_out, Vp_out"))
        Fθ = plan.Fθk_r
        Fφ = plan.Fφk_r
    else
        Fθ = plan.Fθk
        Fφ = plan.Fφk
    end
    fill!(Fθ, zero(eltype(Fθ)))
    fill!(Fφ, zero(eltype(Fφ)))

    # Delegate to the shared orchestrator: one Legendre traversal produces both
    # components (the kernels return `(g_theta, g_phi)` from a single row), it
    # dispatches on the config's fused tables, and — being a function barrier —
    # it specialises on the concrete type of `Slm_int`, which the caller-side
    # `_internal_coefficients` leaves as a small Union. Inlining this loop here
    # instead cost both the table path and the barrier.
    # `real_output=false` on the rfft path: the half-spectrum buffer has no
    # negative-m slots and `irfft` reconstructs them implicitly.
    _synthesis_sphtor_mloop!(Fθ, Fφ, cfg, Slm_int, Tlm_int;
                             ltr=cfg.lmax, real_output=(real_output && !plan.use_rfft))

    if plan.use_rfft
        mul!(plan.real_scratch,  plan.irfft_plan, plan.Fθk_r)
        mul!(plan.real_scratch2, plan.irfft_plan, plan.Fφk_r)
        if cfg.robert_form
            @inbounds for i in 1:nlat
                sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
                for j in 1:nlon
                    plan.real_scratch[i, j]  *= sθ
                    plan.real_scratch2[i, j] *= sθ
                end
            end
        end
        @inbounds for i in 1:nlat, j in 1:nlon
            Vt_out[i, j] = plan.real_scratch[i, j]
            Vp_out[i, j] = plan.real_scratch2[i, j]
        end
        return Vt_out, Vp_out
    end

    plan.ifft_plan * plan.Fθk
    plan.ifft_plan * plan.Fφk
    if cfg.robert_form
        @inbounds for i in 1:nlat
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
            for j in 1:nlon
                plan.Fθk[i, j] *= sθ
                plan.Fφk[i, j] *= sθ
            end
        end
    end
    if real_output
        @inbounds for i in 1:nlat, j in 1:nlon
            Vt_out[i, j] = real(plan.Fθk[i, j])
            Vp_out[i, j] = real(plan.Fφk[i, j])
        end
    else
        @inbounds for i in 1:nlat, j in 1:nlon
            Vt_out[i, j] = plan.Fθk[i, j]
            Vp_out[i, j] = plan.Fφk[i, j]
        end
    end
    return Vt_out, Vp_out
end

"""
    analysis!(plan::SHTPlan, alm_out::AbstractMatrix, f::AbstractMatrix)

In-place forward scalar SHT writing coefficients into `alm_out`.
"""
function analysis!(plan::SHTPlan, alm_out::AbstractMatrix, f::AbstractMatrix)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon
    size(f,1)==nlat || throw(DimensionMismatch("f first dim must be nlat"))
    size(f,2)==nlon || throw(DimensionMismatch("f second dim must be nlon"))
    size(alm_out,1)==cfg.lmax+1 || throw(DimensionMismatch("alm rows must be lmax+1"))
    size(alm_out,2)==cfg.mmax+1 || throw(DimensionMismatch("alm cols must be mmax+1"))

    fill!(alm_out, zero(eltype(alm_out)))

    if plan.use_rfft
        eltype(f) <: Real || throw(ArgumentError("use_rfft plan requires real-valued f"))
        @inbounds for i in 1:nlat, j in 1:nlon
            plan.real_scratch[i,j] = f[i,j]
        end
        # Half-spectrum FFT — bins 0..mmax match full-FFT values for real input.
        mul!(plan.Fθk_r, plan.rfft_plan, plan.real_scratch)
        Fbuf = plan.Fθk_r
    else
        @inbounds for i in 1:nlat, j in 1:nlon
            plan.Fθk[i,j] = f[i,j]
        end
        plan.fft_plan * plan.Fθk
        Fbuf = plan.Fθk
    end

    # Delegate to the shared orchestrator. It dispatches on the config's fused
    # tables and is a function barrier, so the accumulation loop specialises on
    # the concrete buffer type. Hardwiring the on-the-fly kernel here (as this
    # used to) made the planned transform ~6x slower than the plain
    # `analysis(cfg, f)` it exists to beat whenever tables were prepared.
    _analysis_scalar_mloop!(alm_out, cfg, Fbuf)

    return _externalize_coefficients!(alm_out, cfg)
end

"""
    synthesis!(plan::SHTPlan, f_out::AbstractMatrix, alm::AbstractMatrix; real_output=true)

In-place inverse scalar SHT writing spatial field into `f_out`.
Streams m→k directly without building a (θ×m) intermediate.
"""
function synthesis!(plan::SHTPlan, f_out::AbstractMatrix, alm::AbstractMatrix; real_output::Bool=true)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon

    size(f_out,1)==nlat || throw(DimensionMismatch("f_out first dim must be nlat"))
    size(f_out,2)==nlon || throw(DimensionMismatch("f_out second dim must be nlon"))
    size(alm,1)==cfg.lmax+1 || throw(DimensionMismatch("alm rows must be lmax+1"))
    size(alm,2)==cfg.mmax+1 || throw(DimensionMismatch("alm cols must be mmax+1"))

    alm_int = _internal_coefficients(alm, cfg)

    if plan.use_rfft
        real_output || throw(ArgumentError("synthesis! with use_rfft plan requires real_output=true"))
        eltype(f_out) <: Real || throw(ArgumentError("use_rfft plan requires real-valued f_out"))
        fill!(plan.Fθk_r, zero(eltype(plan.Fθk_r)))
        # `real_output=false` here: the half-spectrum buffer has no negative-m
        # slots, and `irfft` reconstructs them implicitly.
        _synthesis_scalar_mloop!(plan.Fθk_r, cfg, alm_int; real_output=false, use_rfft=true)
        mul!(plan.real_scratch, plan.irfft_plan, plan.Fθk_r)
        @inbounds for i in 1:nlat, j in 1:nlon
            f_out[i,j] = plan.real_scratch[i,j]
        end
        return f_out
    end

    fill!(plan.Fθk, zero(eltype(plan.Fθk)))
    # Shared orchestrator: table/on-the-fly dispatch plus the Hermitian fill of
    # the negative-m bins. See `analysis!` for why delegating matters.
    _synthesis_scalar_mloop!(plan.Fθk, cfg, alm_int; real_output=real_output)
    plan.ifft_plan * plan.Fθk

    if real_output
        @inbounds for i in 1:nlat, j in 1:nlon
            f_out[i,j] = real(plan.Fθk[i,j])
        end
    else
        @inbounds for i in 1:nlat, j in 1:nlon
            f_out[i,j] = plan.Fθk[i,j]
        end
    end
    return f_out
end
