#=
================================================================================
fftutils.jl - FFT Utilities with Automatic Differentiation Support
================================================================================

This file provides FFT operations along the longitude (φ) dimension for
spherical harmonic transforms, with automatic fallback to pure Julia DFT
when FFTW cannot handle certain element types.

WHY FFT FOR LONGITUDE?
----------------------
Spherical harmonics have the form:
    Y_l^m(θ,φ) = N_lm P_l^m(cos θ) exp(imφ)

The exp(imφ) dependence is a complex exponential - exactly what FFT extracts!
For a function f(θ,φ) sampled on a grid:
    f(θ,φ) = Σ_m F_m(θ) exp(imφ)

The FFT along φ efficiently computes the Fourier modes F_m(θ) in O(N log N).

AUTOMATIC DIFFERENTIATION SUPPORT
---------------------------------
FFTW is highly optimized but only works with Float64/ComplexF64. When using
automatic differentiation (ForwardDiff, Zygote, etc.), the numbers become
special types like ForwardDiff.Dual that FFTW cannot handle.

Solution: automatic fallback to pure Julia DFT implementation that works
with any numeric type supporting arithmetic operations.

PERFORMANCE
-----------
- FFTW path: O(N log N) - used for normal numeric operations
- DFT fallback: O(N²) - slower but AD-compatible
- The fallback is only used when necessary (detected automatically)

BACKEND TRACKING
----------------
The global _FFT_BACKEND[] tracks which backend was used:
    SHTnsKit.fft_phi_backend()  # Returns :fftw or :dft

DEBUGGING
---------
```julia
# Force DFT fallback for testing
ENV["SHTNSKIT_FORCE_FFTW"] = "0"

# Check which backend was used
A = rand(32, 64)
B = fft_phi(A)
@show SHTnsKit.fft_phi_backend()  # :fftw normally

# Test with AD types
using ForwardDiff
f(x) = sum(abs2.(fft_phi(x .* ones(32,64))))
ForwardDiff.gradient(f, [1.0])  # Uses DFT fallback
```

================================================================================
=#

"""
FFT Utilities with Automatic Differentiation Support

This module provides FFT operations along the longitude dimension with automatic
fallback to pure Julia DFT when FFTW cannot handle certain element types.

The primary use case is enabling automatic differentiation through SHTnsKit
transforms. FFTW cannot handle ForwardDiff.Dual numbers or other AD types,
so we provide a pure Julia DFT fallback that works with any numeric type.

Performance Notes:
- FFTW path: O(N log N) - used for normal Float64/ComplexF64 operations
- DFT fallback: O(N²) - slower but works with arbitrary element types
- Fallback is essential for gradient computations using ForwardDiff, Zygote, etc.
"""

# Precompute 2π for efficiency in DFT calculations
const _TWO_PI = 2π

# Track which backend was used most recently for φ-FFTs: :fftw or :dft
# Uses Atomic{Int} for thread safety (0=unknown, 1=fftw, 2=dft)
const _FFT_BACKEND = Threads.Atomic{Int}(0)
const _FFT_BACKEND_FFTW = 1
const _FFT_BACKEND_DFT = 2

# Small local plan cache for the unplanned API. It removes the repeated FFTW
# planning allocation cost without changing the public `SHTPlan` contract.
# Strides are part of the key because views with the same shape may need
# different FFTW plans.
#
# This is THE φ-FFT plan cache for the whole package: the serial transforms, the
# batch helpers and the distributed extension's per-rank FFTs all route through
# `fft_phi!`/`rfft_phi!`/`ifft_phi!`/`irfft_phi!` and therefore through here. The
# public `enable_fft_plan_cache!` / `disable_fft_plan_cache!` / `set_fft_plan_cache!`
# / `fft_plan_cache_enabled` knobs control THIS cache.
const _LOCAL_FFT_PLAN_CACHE =
    Dict{Tuple{Symbol,DataType,NTuple{2,Int},NTuple{2,Int},Int}, Any}()
const _LOCAL_FFT_PLAN_CACHE_LOCK = ReentrantLock()

# `SHTNSKIT_CACHE_PENCILFFTS` is the legacy spelling kept for scripts that already
# set it; `SHTNSKIT_FFT_PLAN_CACHE` is the current name.
const _FFT_PLAN_CACHE_ENABLED =
    Ref(get(ENV, "SHTNSKIT_FFT_PLAN_CACHE",
            get(ENV, "SHTNSKIT_CACHE_PENCILFFTS", "1")) != "0")

# Soft cap on distinct cached plans. Transform sizes are few and stable in
# practice, but a long-lived process that sweeps many grid shapes would otherwise
# grow this dictionary without bound. Flush wholesale when the cap is reached.
const _FFT_PLAN_CACHE_MAX =
    Ref(parse(Int, get(ENV, "SHTNSKIT_FFT_PLAN_CACHE_MAX", "64")))

"""
    fft_plan_cache_enabled() -> Bool

Whether φ-FFT plans are cached and reused across calls.
"""
fft_plan_cache_enabled() = _FFT_PLAN_CACHE_ENABLED[]

"""
    set_fft_plan_cache!(flag::Bool; clear::Bool=true) -> Bool

Enable or disable φ-FFT plan caching. When disabling with `clear=true` (the
default) the existing entries are dropped as well. Returns `flag`.
"""
function set_fft_plan_cache!(flag::Bool; clear::Bool=true)
    _FFT_PLAN_CACHE_ENABLED[] = flag
    if !flag && clear
        lock(_LOCAL_FFT_PLAN_CACHE_LOCK) do
            empty!(_LOCAL_FFT_PLAN_CACHE)
        end
    end
    return flag
end

"""
    enable_fft_plan_cache!() -> Bool

Convenience wrapper for `set_fft_plan_cache!(true)`.
"""
enable_fft_plan_cache!() = set_fft_plan_cache!(true)

"""
    disable_fft_plan_cache!(; clear::Bool=true) -> Bool

Convenience wrapper for `set_fft_plan_cache!(false; clear)`.
"""
disable_fft_plan_cache!(; clear::Bool=true) = set_fft_plan_cache!(false; clear=clear)

"""
    SHTnsKit.fft_plan_cache_max!(n::Int) -> Int

Set the maximum number of cached φ-FFT plans; `n <= 0` removes the cap.
Returns the previous value.
"""
function fft_plan_cache_max!(n::Int)
    prev = _FFT_PLAN_CACHE_MAX[]
    _FFT_PLAN_CACHE_MAX[] = n
    return prev
end

function fft_phi_backend()
    v = _FFT_BACKEND[]
    v == _FFT_BACKEND_FFTW && return :fftw
    v == _FFT_BACKEND_DFT && return :dft
    return :unknown
end

@inline function _local_fft_plan_key(kind::Symbol, A::AbstractMatrix, nlon::Int=0)
    return (kind, eltype(A), (size(A, 1), size(A, 2)), (stride(A, 1), stride(A, 2)), nlon)
end

"""Build one FFTW plan for `A`; see `_cached_local_fft_plan` for the flag rationale."""
function _build_local_fft_plan(kind::Symbol, A::AbstractMatrix, nlon::Int)
    # UNALIGNED is required, not an optimization choice. The cache key
    # covers eltype/size/strides but NOT the base pointer's alignment, so
    # a plan built for one array gets reused for another that FFTW may
    # consider differently aligned. Without UNALIGNED that reuse throws
    # `ArgumentError`, which the callers catch and answer by falling back
    # to the pure-Julia O(nlat·nlon²) DFT — an order-of-magnitude
    # slowdown with no error surfaced. Forfeiting the aligned SIMD
    # codelets is much cheaper than forfeiting the FFT. The batch helpers
    # in batch_transforms.jl pass the same flag for the same reason.
    flags = FFTW.ESTIMATE | FFTW.UNALIGNED
    return if kind === :fft
        plan_fft!(A, 2; flags)
    elseif kind === :ifft
        plan_ifft!(A, 2; flags)
    elseif kind === :rfft
        plan_rfft(A, 2; flags)
    elseif kind === :irfft
        plan_irfft(A, nlon, 2; flags)
    else
        throw(ArgumentError("unknown FFT plan kind: $kind"))
    end
end

function _cached_local_fft_plan(kind::Symbol, A::AbstractMatrix, nlon::Int=0)
    _FFT_PLAN_CACHE_ENABLED[] || return _build_local_fft_plan(kind, A, nlon)
    key = _local_fft_plan_key(kind, A, nlon)
    lock(_LOCAL_FFT_PLAN_CACHE_LOCK)
    try
        plan = get(_LOCAL_FFT_PLAN_CACHE, key, nothing)
        if plan === nothing
            plan = _build_local_fft_plan(kind, A, nlon)
            # Enforce the soft cap: flush before inserting so the fresh entry survives.
            cap = _FFT_PLAN_CACHE_MAX[]
            if cap > 0 && length(_LOCAL_FFT_PLAN_CACHE) >= cap
                empty!(_LOCAL_FFT_PLAN_CACHE)
            end
            _LOCAL_FFT_PLAN_CACHE[key] = plan
        end
        return plan
    finally
        unlock(_LOCAL_FFT_PLAN_CACHE_LOCK)
    end
end

# FFTW's in-place complex plans mutate their input, so real inputs are copied
# explicitly into the complex destination before applying the cached plan.
@inline function _copy_complex_input!(dest::AbstractMatrix{<:Complex}, A::AbstractMatrix)
    @inbounds for I in eachindex(dest, A)
        dest[I] = A[I]
    end
    return dest
end

"""
    _dft_phi(A::AbstractMatrix, dir::Int)

Pure Julia discrete Fourier transform implementation along longitude (phi direction).

This function implements the standard DFT formula manually, without relying on
FFTW. While slower than optimized FFT libraries, it works with any numeric type
including automatic differentiation types like ForwardDiff.Dual.

Parameters:
- A: Input matrix [latitude × longitude] 
- dir: Direction flag (+1 for inverse DFT, -1 for forward DFT)

The DFT formula implemented is:
Y[k] = Σⱼ A[j] * exp(dir * 2πi * k * j / N)
"""
function _dft_phi(A::AbstractMatrix, dir::Int)
    nlat, nlon = size(A)
    Y = similar(A, complex(eltype(A)))  # complex output; avoids materializing complex.(A) just for its type/shape
    
    # Compute DFT for each latitude band independently with SIMD optimization
    @inbounds for i in 1:nlat
        # For each output frequency k
        for k in 0:(nlon-1)
            s = zero(eltype(Y))  # Accumulator for this frequency
            
            # Sum over all input points j - USE REDUCTION, NOT ivdep!
            # Multiple j iterations accumulate into same 's', so this is a reduction operation
            @simd for j in 0:(nlon-1)
                # DFT kernel: exp(dir * 2πi * k * j / N)
                s += A[i, j+1] * cis(dir * _TWO_PI * k * j / nlon)
            end
            
            Y[i, k+1] = s  # Store result (converting to 1-based indexing)
        end
    end
    
    return Y
end

"""
    _as_complex(A) -> AbstractMatrix{<:Complex}

Return `A` unchanged if already complex; otherwise `complex.(A)`. Avoids a
redundant copy when callers pass complex arrays through real-valued code paths.
"""
@inline _as_complex(A::AbstractArray{<:Complex}) = A
@inline _as_complex(A::AbstractArray) = complex.(A)

"""
    ifft_phi!(dest::AbstractMatrix{<:Complex}, A::AbstractMatrix)

In-place inverse FFT into preallocated `dest` (complex). Writes the transform of
`A` into `dest`, overwriting any existing data. Uses DFT fallback when FFTW is
unavailable for the element type.
"""
function ifft_phi!(dest::AbstractMatrix{<:Complex}, A::AbstractMatrix)
    size(dest) == size(A) || throw(DimensionMismatch("dest and A must have same size"))
    _copy_complex_input!(dest, A)
    nlat, nlon = size(A)
    try
        # Reusing the plan keeps warmed in-place transforms effectively
        # allocation-free; `mul!` avoids the allocating `plan * array` path.
        plan = _cached_local_fft_plan(:ifft, dest)
        mul!(dest, plan, dest)
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
        return dest
    catch e
        if !(e isa MethodError || e isa ArgumentError || e isa InexactError)
            rethrow(e)
        end
        # DFT fallback: must preserve original row data before overwriting
        row_buf = Vector{eltype(dest)}(undef, nlon)
        @inbounds for i in 1:nlat
            # Copy row to buffer before overwriting
            for j in 1:nlon
                row_buf[j] = dest[i, j]
            end
            # Compute inverse DFT for this row
            for k in 0:(nlon-1)
                s = zero(eltype(dest))
                @simd for j in 0:(nlon-1)
                    s += row_buf[j+1] * cis(_TWO_PI * k * j / nlon)
                end
                dest[i, k+1] = s / nlon
            end
        end
        _FFT_BACKEND[] = _FFT_BACKEND_DFT
        return dest
    end
end

"""
    fft_phi(A::AbstractMatrix)

Forward FFT along the longitude dimension with automatic differentiation support.

This function first attempts to use FFTW's optimized FFT. If that fails (e.g.,
due to unsupported element types in AD), it automatically falls back to the
pure Julia DFT implementation.

The longitude dimension corresponds to the azimuthal angle φ in spherical
coordinates, hence the function name.
"""
function fft_phi(A::AbstractMatrix)
    try
        # Primary path: use optimized FFTW along dimension 2 (longitude)
        local Y = fft(A, 2)
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
        return Y
    catch e
        # Only fall back for type-related errors (FFTW doesn't support AD types).
        # Re-throw genuine bugs like DimensionMismatch, OutOfMemoryError, etc.
        if !(e isa MethodError || e isa ArgumentError || e isa InexactError)
            rethrow(e)
        end
        # Fallback path: use pure Julia DFT for AD compatibility
        # (FFTW only supports Float32/Float64/ComplexF32/ComplexF64)
        local Y = _dft_phi(A, -1)  # Forward transform uses -1 direction
        _FFT_BACKEND[] = _FFT_BACKEND_DFT
        return Y
    end
end

"""
    fft_phi!(dest::AbstractMatrix{<:Complex}, A::AbstractMatrix)

In-place forward FFT along longitude into a preallocated complex buffer `dest`
of the same size as `A`. Reduces allocations compared to `fft_phi(complex.(A))`.
Falls back to the pure-DFT path for unsupported element types.
"""
function fft_phi!(dest::AbstractMatrix{<:Complex}, A::AbstractMatrix)
    size(dest) == size(A) || throw(DimensionMismatch("dest and A must have same size"))
    _copy_complex_input!(dest, A)
    nlat, nlon = size(A)
    try
        # Reusing the plan keeps warmed in-place transforms effectively
        # allocation-free; `mul!` avoids the allocating `plan * array` path.
        plan = _cached_local_fft_plan(:fft, dest)
        mul!(dest, plan, dest)
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
        return dest
    catch e
        if !(e isa MethodError || e isa ArgumentError || e isa InexactError)
            rethrow(e)
        end
        # DFT fallback: must preserve original row data before overwriting
        row_buf = Vector{eltype(dest)}(undef, nlon)
        @inbounds for i in 1:nlat
            # Copy row to buffer before overwriting
            for j in 1:nlon
                row_buf[j] = dest[i, j]
            end
            # Compute forward DFT for this row
            for k in 0:(nlon-1)
                s = zero(eltype(dest))
                @simd for j in 0:(nlon-1)
                    s += row_buf[j+1] * cis(-_TWO_PI * k * j / nlon)
                end
                dest[i, k+1] = s
            end
        end
        _FFT_BACKEND[] = _FFT_BACKEND_DFT
        return dest
    end
end

"""
    rfft_phi(A::AbstractMatrix{<:Real}) -> Matrix{<:Complex}

Real-to-complex FFT along longitude. Returns `(nlat, nlon÷2+1)` complex array
with the positive-frequency half of the spectrum (bin k matches full FFT bin k
for `k = 0..nlon÷2`). Falls back to a DFT-then-truncate path when FFTW cannot
handle the input element type (AD).
"""
function rfft_phi(A::AbstractMatrix{<:Real})
    # Delegate to the in-place variant so the FFTW plan comes from the local
    # plan cache instead of being rebuilt on every call (bare `rfft(A, 2)`
    # re-plans each time). The fallback DFT path for AD eltypes is preserved
    # inside rfft_phi!.
    nlat, nlon = size(A)
    dest = Matrix{complex(float(eltype(A)))}(undef, nlat, nlon ÷ 2 + 1)
    return rfft_phi!(dest, A)
end

"""
    rfft_phi!(dest::AbstractMatrix{<:Complex}, A::AbstractMatrix{<:Real})

Real-to-complex FFT along longitude into a preallocated half-spectrum buffer.
`dest` must have shape `(size(A,1), size(A,2)÷2+1)`.
"""
function rfft_phi!(dest::StridedMatrix{Complex{T}}, A::StridedMatrix{T}) where {T<:Union{Float32,Float64}}
    nlat, nlon = size(A)
    size(dest) == (nlat, nlon ÷ 2 + 1) || throw(DimensionMismatch("dest shape must be (size(A,1), size(A,2)÷2+1)"))
    # Exact FFTW-supported eltypes do not need the generic try/fallback path;
    # keeping this method straight-line avoids patch-version allocation noise.
    plan = _cached_local_fft_plan(:rfft, A)
    mul!(dest, plan, A)
    _FFT_BACKEND[] = _FFT_BACKEND_FFTW
    return dest
end

function rfft_phi!(dest::AbstractMatrix{<:Complex}, A::AbstractMatrix{<:Real})
    nlat, nlon = size(A)
    size(dest) == (nlat, nlon ÷ 2 + 1) || throw(DimensionMismatch("dest shape must be (size(A,1), size(A,2)÷2+1)"))
    try
        # The real-input plan writes only nonnegative Fourier bins. Scalar and
        # vector synthesis code never fills negative bins on this path.
        plan = _cached_local_fft_plan(:rfft, A)
        mul!(dest, plan, A)
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
        return dest
    catch e
        if !(e isa MethodError || e isa ArgumentError || e isa InexactError)
            rethrow(e)
        end
        Y_full = _dft_phi(A, -1)
        @inbounds for j in 1:size(dest, 2), i in 1:nlat
            dest[i, j] = Y_full[i, j]
        end
        _FFT_BACKEND[] = _FFT_BACKEND_DFT
        return dest
    end
end

"""
    irfft_phi!(dest::AbstractMatrix{<:Real}, A::AbstractMatrix{<:Complex}, nlon::Int)

Complex-to-real inverse FFT along longitude. `A` has `nlon÷2+1` columns (the
positive-frequency half); `dest` is `(size(A,1), nlon)`. Writes real output
in place. DFT fallback reconstructs the negative-frequency half via Hermitian
symmetry then inverse-transforms.
"""
function irfft_phi!(dest::StridedMatrix{T}, A::StridedMatrix{Complex{T}}, nlon::Int) where {T<:Union{Float32,Float64}}
    size(A, 2) == nlon ÷ 2 + 1 || throw(DimensionMismatch("A must have nlon÷2+1 columns"))
    size(dest) == (size(A, 1), nlon) || throw(DimensionMismatch("dest shape must be (size(A,1), nlon)"))
    # Exact FFTW-supported eltypes do not need the generic try/fallback path;
    # keeping this method straight-line avoids patch-version allocation noise.
    plan = _cached_local_fft_plan(:irfft, A, nlon)
    mul!(dest, plan, A)
    _FFT_BACKEND[] = _FFT_BACKEND_FFTW
    return dest
end

function irfft_phi!(dest::AbstractMatrix{<:Real}, A::AbstractMatrix{<:Complex}, nlon::Int)
    size(A, 2) == nlon ÷ 2 + 1 || throw(DimensionMismatch("A must have nlon÷2+1 columns"))
    size(dest) == (size(A, 1), nlon) || throw(DimensionMismatch("dest shape must be (size(A,1), nlon)"))
    try
        # `plan_irfft` implicitly reconstructs Hermitian negative-frequency
        # bins, so callers pass only the half-spectrum buffer here.
        plan = _cached_local_fft_plan(:irfft, A, nlon)
        mul!(dest, plan, A)
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
        return dest
    catch e
        if !(e isa MethodError || e isa ArgumentError || e isa InexactError)
            rethrow(e)
        end
        nlat = size(A, 1)
        CT = eltype(A)
        Full = Matrix{CT}(undef, nlat, nlon)
        @inbounds for i in 1:nlat
            for k in 0:(nlon - 1)
                if k <= nlon ÷ 2
                    Full[i, k + 1] = A[i, k + 1]
                else
                    # Hermitian: bin k (k > nlon/2) = conj(bin nlon-k)
                    Full[i, k + 1] = conj(A[i, nlon - k + 1])
                end
            end
        end
        @inbounds for i in 1:nlat
            for j in 0:(nlon - 1)
                s = zero(CT)
                @simd for k in 0:(nlon - 1)
                    s += Full[i, k + 1] * cis(_TWO_PI * k * j / nlon)
                end
                dest[i, j + 1] = real(s) / nlon
            end
        end
        _FFT_BACKEND[] = _FFT_BACKEND_DFT
        return dest
    end
end

"""
    ifft_phi(A::AbstractMatrix)

Inverse FFT along the longitude dimension with automatic differentiation support.

Like fft_phi, this function attempts FFTW first and falls back to pure Julia
DFT if needed. The inverse transform includes the proper normalization factor.

The scaling by 1/N is required to make the forward and inverse transforms
true inverses of each other.
"""
function ifft_phi(A::AbstractMatrix)
    nlon = size(A,2)  # Number of longitude points for normalization

    try
        # Primary path: use optimized FFTW inverse FFT
        local y = ifft(A, 2)
        _FFT_BACKEND[] = _FFT_BACKEND_FFTW
        return y
    catch e
        # Only fall back for type-related errors (FFTW doesn't support AD types).
        # Re-throw genuine bugs like DimensionMismatch, OutOfMemoryError, etc.
        if !(e isa MethodError || e isa ArgumentError || e isa InexactError)
            rethrow(e)
        end
        # Fallback path: use pure Julia inverse DFT with proper scaling
        # (FFTW only supports Float32/Float64/ComplexF32/ComplexF64)
        local y = (1/nlon) * _dft_phi(A, +1)  # Inverse transform uses +1 direction
        _FFT_BACKEND[] = _FFT_BACKEND_DFT
        return y
    end
end
