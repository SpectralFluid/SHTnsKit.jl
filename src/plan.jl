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
simultaneously. The internal buffers (`P`, `dPdtheta`, `G`, `Fθk`, etc.) are
shared mutable state — concurrent calls to `analysis!` or `synthesis!` on the
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
    P::Vector{Float64}            # Working array for Legendre polynomials P_l^m(x)
    dPdx::Vector{Float64}         # Working array for derivatives dP_l^m/dx (legacy, kept for compatibility)
    dPdtheta::Vector{Float64}     # Working array for pole-safe derivatives dP_l^m/dθ
    P_over_sinth::Vector{Float64} # Working array for pole-safe P_l^m/sin(θ)
    Pb::Vector{Float64}           # Scratch buffer of length lmax+2 for normalized dθ recurrence
    G::Vector{ComplexF64}         # Temporary array for latitudinal profiles
    Fθk::Matrix{ComplexF64}       # Fourier coefficient matrix [latitude × longitude] (complex path)
    Fθk_r::Matrix{ComplexF64}     # (nlat, nlon÷2+1) buffer for rfft path; 0×0 when use_rfft=false
    real_scratch::Matrix{Float64} # (nlat, nlon) real scratch for rfft path; 0×0 when use_rfft=false
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

    # Full complex FFT path — always present for vector (sphtor) transforms
    # which reuse Fθk as streaming m→k buffer regardless of use_rfft.
    Fθk = Matrix{ComplexF64}(undef, nlat, nlon)
    fill!(Fθk, zero(ComplexF64))
    fft_plan = FFTW.plan_fft!(Fθk, 2)
    ifft_plan = FFTW.plan_ifft!(Fθk, 2)

    # RFFT-specific buffers and plans
    if use_rfft
        # Scalar planned rfft stores half-width Fourier data. The full complex
        # buffer remains allocated because vector planned transforms still need
        # explicit negative-m columns when not using rfft.
        real_scratch = Matrix{Float64}(undef, nlat, nlon)
        fill!(real_scratch, 0.0)
        Fθk_r = Matrix{ComplexF64}(undef, nlat, nlon ÷ 2 + 1)
        fill!(Fθk_r, zero(ComplexF64))
        rfft_plan = FFTW.plan_rfft(real_scratch, 2)
        irfft_plan = FFTW.plan_irfft(Fθk_r, nlon, 2)
    else
        real_scratch = Matrix{Float64}(undef, 0, 0)
        Fθk_r = Matrix{ComplexF64}(undef, 0, 0)
        rfft_plan = nothing
        irfft_plan = nothing
    end

    # Convention conversion is a public-boundary operation. Canonical plans
    # remain allocation-free; non-canonical synthesis uses a typed temporary.
    return SHTPlan(cfg, P, dPdx, dPdtheta, P_over_sinth, Pb, G, Fθk, Fθk_r, real_scratch,
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
    
    lmax, mmax = cfg.lmax, cfg.mmax
    scaleφ = cfg.cphi
    fill!(Slm_out, zero(eltype(Slm_out))); fill!(Tlm_out, zero(eltype(Tlm_out)))

    # Two passes over (Vt, Vp): each packs the component into a real/complex
    # FFT buffer, applies Robert form if needed, transforms to k-space, then
    # adds that component's contribution to Slm/Tlm. Both buffers have bins
    # 0..mmax at positions 1..mmax+1 in the FFT output — identical indexing
    # between complex and rfft paths so the kernel needs no change.
    for pass in 1:2
        V = pass == 1 ? Vt : Vp
        # Process Vtheta and Vphi separately through the same FFT buffer. This
        # keeps the plan compact while accumulating both S and T spectra.
        if plan.use_rfft
            eltype(V) <: Real || throw(ArgumentError("use_rfft plan requires real-valued Vt/Vp"))
            @inbounds for i in 1:nlat, j in 1:nlon
                plan.real_scratch[i,j] = V[i,j]
            end
            if cfg.robert_form
                @inbounds for i in 1:nlat
                    sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
                    if sθ > 0
                        for j in 1:nlon
                            plan.real_scratch[i, j] /= sθ
                        end
                    end
                end
            end
            mul!(plan.Fθk_r, plan.rfft_plan, plan.real_scratch)
            Fbuf = plan.Fθk_r
        else
            @inbounds for i in 1:nlat, j in 1:nlon
                plan.Fθk[i,j] = V[i,j]
            end
            if cfg.robert_form
                @inbounds for i in 1:nlat
                    sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
                    if sθ > 0
                        for j in 1:nlon
                            plan.Fθk[i, j] /= sθ
                        end
                    end
                end
            end
            plan.fft_plan * plan.Fθk
            Fbuf = plan.Fθk
        end

        xv = cfg.x; wv = cfg.w  # hoist field reads out of the i/l loops (cfg is mutable, so not auto-hoisted)
        for m in 0:mmax
            col = m + 1
            for i in 1:nlat
                Plm_norm_dPdtheta_over_sinth_row!(plan.P, plan.dPdtheta, plan.P_over_sinth, xv[i], lmax, m, plan.Pb)
                fourier_coeff = Fbuf[i, col]
                quad_weight = wv[i]
                @inbounds for l in max(1,m):lmax
                    legendre_deriv = plan.dPdtheta[l+1]       # already orthonormal-normalized
                    legendre_over_sinθ = plan.P_over_sinth[l+1]
                    weight_coeff = quad_weight * scaleφ / (l*(l+1))
                    if pass == 1
                        # From Vθ: S gets +dθY*Vθ, T gets +(im*m/sinθ)*Vθ
                        Tlm_out[l+1, col] += weight_coeff * (1.0im * m * legendre_over_sinθ * fourier_coeff)
                        Slm_out[l+1, col] += weight_coeff * (fourier_coeff * legendre_deriv)
                    else
                        # From Vφ: S gets -(im*m/sinθ)*Vφ, T gets +dθY*Vφ
                        Slm_out[l+1, col] += -weight_coeff * (1.0im * m * legendre_over_sinθ * fourier_coeff)
                        Tlm_out[l+1, col] += weight_coeff * (fourier_coeff * legendre_deriv)
                    end
                end
            end
        end
    end
    
    _externalize_coefficients!(Slm_out, cfg)
    _externalize_coefficients!(Tlm_out, cfg)
    return Slm_out, Tlm_out
end

"""
    synthesis_sphtor!(plan::SHTPlan, Vt_out::AbstractMatrix, Vp_out::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output=true)

In-place vector synthesis. Streams m→k without forming (θ×m) intermediates; inverse FFT Vt then Vp.
"""
function synthesis_sphtor!(plan::SHTPlan, Vt_out::AbstractMatrix, Vp_out::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix; real_output::Bool=true)
    cfg = plan.cfg
    nlat, nlon = cfg.nlat, cfg.nlon
    
    size(Vt_out,1)==nlat && size(Vt_out,2)==nlon || throw(DimensionMismatch("Vt_out dims"))
    size(Vp_out,1)==nlat && size(Vp_out,2)==nlon || throw(DimensionMismatch("Vp_out dims"))
    size(Slm,1)==cfg.lmax+1 && size(Slm,2)==cfg.mmax+1 || throw(DimensionMismatch("Slm dims"))
    size(Tlm,1)==cfg.lmax+1 && size(Tlm,2)==cfg.mmax+1 || throw(DimensionMismatch("Tlm dims"))
    
    lmax, mmax = cfg.lmax, cfg.mmax
    inv_scaleφ = phi_inv_scale(cfg)
    
    Slm_int = _internal_coefficients(Slm, cfg)
    Tlm_int = _internal_coefficients(Tlm, cfg)
    
    # Two sibling passes: build Vt's Fourier buffer then Vp's, each with its
    # own m-loop formula. rfft path writes to the half-spectrum buffer and uses
    # irfft directly; complex path mirrors negative-m via Hermitian fill.
    if plan.use_rfft
        real_output || throw(ArgumentError("synthesis_sphtor! with use_rfft plan requires real_output=true"))
        eltype(Vt_out) <: Real && eltype(Vp_out) <: Real ||
            throw(ArgumentError("use_rfft plan requires real-valued Vt_out, Vp_out"))
    end

    for pass in 1:2
        if plan.use_rfft
            fill!(plan.Fθk_r, zero(eltype(plan.Fθk_r)))
        else
            fill!(plan.Fθk, zero(eltype(plan.Fθk)))
        end

        xv = cfg.x  # hoist field reads out of the i/l loops (cfg is mutable, so not auto-hoisted)
        for m in 0:mmax
            col = m + 1
            for i in 1:nlat
                Plm_norm_dPdtheta_over_sinth_row!(plan.P, plan.dPdtheta, plan.P_over_sinth, xv[i], lmax, m, plan.Pb)
                g = zero(ComplexF64)
                @inbounds for l in max(1,m):lmax
                    dθY = plan.dPdtheta[l+1]       # already orthonormal-normalized
                    Y_over_sθ = plan.P_over_sinth[l+1]
                    Sl = Slm_int[l+1, col]; Tl = Tlm_int[l+1, col]
                    if pass == 1
                        # Vθ = ∂S/∂θ - (im/sinθ) * T
                        g += dθY * Sl - 1.0im * m * Y_over_sθ * Tl
                    else
                        # Vφ = (im/sinθ) * S + ∂T/∂θ
                        g += 1.0im * m * Y_over_sθ * Sl + dθY * Tl
                    end
                end
                plan.G[i] = g
            end

            if plan.use_rfft
                @inbounds for i in 1:nlat
                    plan.Fθk_r[i, col] = inv_scaleφ * plan.G[i]
                end
            else
                @inbounds for i in 1:nlat
                    plan.Fθk[i, col] = inv_scaleφ * plan.G[i]
                end
                if real_output && m > 0
                    conj_index = nlon - m + 1
                    @inbounds for i in 1:nlat
                        plan.Fθk[i, conj_index] = conj(plan.Fθk[i, col])
                    end
                end
            end
        end

        V_target = pass == 1 ? Vt_out : Vp_out

        if plan.use_rfft
            mul!(plan.real_scratch, plan.irfft_plan, plan.Fθk_r)
            if cfg.robert_form
                @inbounds for i in 1:nlat
                    sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
                    for j in 1:nlon
                        plan.real_scratch[i,j] *= sθ
                    end
                end
            end
            @inbounds for i in 1:nlat, j in 1:nlon
                V_target[i,j] = plan.real_scratch[i,j]
            end
        else
            plan.ifft_plan * plan.Fθk
            if cfg.robert_form
                @inbounds for i in 1:nlat
                    sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
                    for j in 1:nlon
                        plan.Fθk[i,j] *= sθ
                    end
                end
            end
            if real_output
                @inbounds for i in 1:nlat, j in 1:nlon
                    V_target[i,j] = real(plan.Fθk[i,j])
                end
            else
                @inbounds for i in 1:nlat, j in 1:nlon
                    V_target[i,j] = plan.Fθk[i,j]
                end
            end
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

    lmax, mmax = cfg.lmax, cfg.mmax
    scaleφ = cfg.cphi
    fill!(alm_out, zero(eltype(alm_out)))

    if plan.use_rfft
        eltype(f) <: Real || throw(ArgumentError("use_rfft plan requires real-valued f"))
        @inbounds for i in 1:nlat, j in 1:nlon
            plan.real_scratch[i,j] = f[i,j]
        end
        # Half-spectrum FFT — bins 0..mmax match full-FFT values for real input.
        mul!(plan.Fθk_r, plan.rfft_plan, plan.real_scratch)
        for m in 0:mmax
            col = m + 1
            @inbounds for i in 1:nlat
                _scalar_analysis_kernel_otf!(alm_out, cfg, plan.Fθk_r, plan.P, i, col, m, lmax, scaleφ)
            end
        end
    else
        @inbounds for i in 1:nlat, j in 1:nlon
            plan.Fθk[i,j] = f[i,j]
        end
        plan.fft_plan * plan.Fθk
        for m in 0:mmax
            col = m + 1
            @inbounds for i in 1:nlat
                _scalar_analysis_kernel_otf!(alm_out, cfg, plan.Fθk, plan.P, i, col, m, lmax, scaleφ)
            end
        end
    end

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

    lmax, mmax = cfg.lmax, cfg.mmax
    inv_scaleφ = phi_inv_scale(cfg)

    if plan.use_rfft
        real_output || throw(ArgumentError("synthesis! with use_rfft plan requires real_output=true"))
        eltype(f_out) <: Real || throw(ArgumentError("use_rfft plan requires real-valued f_out"))
        fill!(plan.Fθk_r, zero(eltype(plan.Fθk_r)))
        # Fill only nonnegative m bins. Unlike the complex path below, no
        # Hermitian mirror is written because irfft_plan reconstructs it.
        for m in 0:mmax
            col = m + 1
            @inbounds for i in 1:nlat
                plan.Fθk_r[i, col] = inv_scaleφ * _scalar_synthesis_kernel_otf(cfg, alm_int, plan.P, i, col, m, lmax)
            end
        end
        # No Hermitian fill — irfft reconstructs implicitly.
        mul!(plan.real_scratch, plan.irfft_plan, plan.Fθk_r)
        @inbounds for i in 1:nlat, j in 1:nlon
            f_out[i,j] = plan.real_scratch[i,j]
        end
        return f_out
    end

    fill!(plan.Fθk, zero(eltype(plan.Fθk)))
    for m in 0:mmax
        col = m + 1
        @inbounds for i in 1:nlat
            plan.Fθk[i, col] = inv_scaleφ * _scalar_synthesis_kernel_otf(cfg, alm_int, plan.P, i, col, m, lmax)
        end
        if real_output && m > 0
            conj_index = nlon - m + 1
            @inbounds for i in 1:nlat
                plan.Fθk[i, conj_index] = conj(plan.Fθk[i, col])
            end
        end
    end
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
