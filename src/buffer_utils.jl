"""
Buffer Allocation Utilities

This module provides common buffer allocation and manipulation patterns
used throughout the SHTnsKit codebase to eliminate code duplication
and provide consistent, optimized buffer management.
"""

"""
    balanced_m_order(mmax::Int, mres::Int=1) -> Vector{Int}

Interleave the valid `m` values from `0:mres:mmax` from low and high ends so
contiguous `@threads :static` chunks receive roughly equal total work when
inner work per m is (lmax - m + 1).
"""
function balanced_m_order(mmax::Int, mres::Int=1)
    mres >= 1 || throw(ArgumentError("mres must be >= 1"))
    n = fld(mmax, mres) + 1
    order = Vector{Int}(undef, n)
    @inbounds for i in 0:(n - 1)
        im = iseven(i) ? div(i, 2) : (n - 1) - div(i, 2)
        order[i+1] = im * mres
    end
    return order
end

"""
    _use_internal_mloop_threads() -> Bool

Return whether a transform may safely start its internal static m-mode loop.
Static loops cannot be nested or started concurrently, so transforms invoked
from an outer threaded region (or from a worker task) run their m-loop serially
on that worker instead.
"""
@inline function _use_internal_mloop_threads()
    return Threads.nthreads() > 1 &&
           Threads.threadid() == 1 &&
           ccall(:jl_in_threaded_region, Cint, ()) == 0
end

"""Run a `for` loop with static threading when safe, otherwise on the caller's thread."""
macro _threads_or_serial(loop)
    loop isa Expr && loop.head === :for ||
        throw(ArgumentError("@_threads_or_serial requires a for loop"))
    return esc(quote
        if _use_internal_mloop_threads()
            Threads.@threads :static $loop
        else
            $loop
        end
    end)
end

"""
    cached_m_order(cfg::SHTConfig) -> Vector{Int}

Return the cfg's cached balanced m-ordering, building it on first use. Callers
should not mutate the returned vector.
"""
const _M_ORDER_LOCK = Threads.ReentrantLock()

@inline function cached_m_order(cfg)
    lock(_M_ORDER_LOCK)
    try
        mo = cfg._m_order
        expected_length = fld(cfg.mmax, cfg.mres) + 1
        valid = length(mo) == expected_length
        if valid
            @inbounds for m in mo
                if m < 0 || m > cfg.mmax || m % cfg.mres != 0
                    valid = false
                    break
                end
            end
        end
        if !valid
            # Build into a private vector and publish it only after complete.
            cfg._m_order = balanced_m_order(cfg.mmax, cfg.mres)
        end
        return cfg._m_order
    finally
        unlock(_M_ORDER_LOCK)
    end
end

"""
    _ensure_otf_scratch!(buffers::Vector{Vector{Float64}}, lmax::Int) -> Vector

Lazily grow `buffers` so it has `Threads.maxthreadid()` entries, each with
capacity for at least `lmax + 1` values. Caller holds `buffers` across calls to
skip per-call allocation. Safe to call repeatedly and concurrently; buffers
grow when needed and are never shrunk while another transform may be using one.
"""
const _OTF_SCRATCH_LOCK = Threads.ReentrantLock()

@inline function _ensure_otf_scratch!(buffers::Vector{Vector{Float64}}, lmax::Int)
    n = Threads.maxthreadid()
    required_length = lmax + 1

    # The lock covers both inspection and mutation. In particular, `resize!`
    # exposes unassigned slots until they are filled; an unlocked fast path can
    # otherwise observe the new length and index one of those slots.
    lock(_OTF_SCRATCH_LOCK)
    try
        cur = length(buffers)
        if cur < n
            resize!(buffers, n)
        end
        @inbounds for i in 1:n
            if !isassigned(buffers, i) || length(buffers[i]) < required_length
                buffers[i] = Vector{Float64}(undef, required_length)
            end
        end
    finally
        unlock(_OTF_SCRATCH_LOCK)
    end
    return buffers
end

"""
    warmup!(cfg::SHTConfig) -> cfg

Eagerly populate every lazily-built cache on `cfg` up to the current
`Threads.maxthreadid()`. After this call the hot-path transforms are guaranteed
to hit cached data and will not allocate to initialise scratch buffers,
normalization matrices, or per-thread OTF Legendre storage. Call once after
config construction in multi-threaded workflows to avoid first-iteration
allocation surprises.
"""
function warmup!(cfg::SHTConfig)
    _ensure_otf_scratch!(cfg._otf_scratch_P, cfg.lmax)
    _ensure_otf_scratch!(cfg._otf_scratch_dP, cfg.lmax)
    _ensure_otf_scratch!(cfg._otf_scratch_Ps, cfg.lmax)
    _ensure_otf_scratch!(cfg._otf_scratch_Pb, cfg.lmax + 1)  # lmax+2 entries for extended P̄ row
    _ensure_norm_scale_matrix!(cfg)
    cached_m_order(cfg)
    return cfg
end

"""
    allocate_spectral_pair(template1, template2) -> (buf1, buf2)

Allocate a pair of spectral coefficient buffers based on template arrays.
Commonly used for (Slm, Tlm) or (Qlm, Slm) pairs.
"""
function allocate_spectral_pair(template1::AbstractMatrix, template2::AbstractMatrix)
    return similar(template1), similar(template2)
end

"""
    allocate_spectral_triple(template1, template2, template3) -> (buf1, buf2, buf3)

Allocate three spectral coefficient buffers, commonly used for (Qlm, Slm, Tlm) triples.
"""
function allocate_spectral_triple(template1::AbstractMatrix, template2::AbstractMatrix, template3::AbstractMatrix)
    return similar(template1), similar(template2), similar(template3)
end

"""
    copy_spectral_pair(src1, src2) -> (copy1, copy2)

Create working copies of a pair of spectral coefficient arrays.
"""
function copy_spectral_pair(src1::AbstractMatrix, src2::AbstractMatrix)
    return copy(src1), copy(src2)
end

"""
    copy_spectral_triple(src1, src2, src3) -> (copy1, copy2, copy3)

Create working copies of three spectral coefficient arrays.
"""
function copy_spectral_triple(src1::AbstractMatrix, src2::AbstractMatrix, src3::AbstractMatrix)
    return copy(src1), copy(src2), copy(src3)
end

"""
    allocate_spatial_pair(template1, template2) -> (buf1, buf2)

Allocate a pair of spatial field buffers, commonly used for (Vt, Vp) components.
"""
function allocate_spatial_pair(template1::AbstractMatrix, template2::AbstractMatrix)
    return similar(template1), similar(template2)
end

"""
    allocate_spatial_triple(template1, template2, template3) -> (buf1, buf2, buf3)

Allocate three spatial field buffers for (Vr, Vt, Vp) components.
"""
function allocate_spatial_triple(template1::AbstractMatrix, template2::AbstractMatrix, template3::AbstractMatrix)
    return similar(template1), similar(template2), similar(template3)
end

"""
    zero_high_degree_modes!(arrays::Tuple, cfg::SHTConfig, ltr::Int)

Zero out high-degree modes (l > ltr) in multiple coefficient arrays simultaneously.
This is a common operation for degree-limited transforms.
"""
function zero_high_degree_modes!(arrays::Tuple, cfg::SHTConfig, ltr::Int)
    lmax, mmax = cfg.lmax, cfg.mmax

    for array in arrays
        z = zero(eltype(array))
        @inbounds for m in 0:mmax, l in (ltr+1):lmax
            if l >= m  # Only valid (l,m) combinations
                array[l+1, m+1] = z
            end
        end
    end

    return arrays
end

"""
    zero_high_degree_modes!(array::AbstractMatrix, cfg::SHTConfig, ltr::Int)

Zero out high-degree modes in a single coefficient array.
"""
function zero_high_degree_modes!(array::AbstractMatrix, cfg::SHTConfig, ltr::Int)
    return zero_high_degree_modes!((array,), cfg, ltr)[1]
end

"""
    create_zero_coefficients(cfg::SHTConfig, element_type::Type=ComplexF64) -> Matrix

Create a zero-filled coefficient matrix with standard (lmax+1, mmax+1) dimensions.
"""
function create_zero_coefficients(cfg::SHTConfig, element_type::Type=ComplexF64)
    lmax, mmax = cfg.lmax, cfg.mmax
    array = Matrix{element_type}(undef, lmax+1, mmax+1)
    fill!(array, zero(element_type))
    return array
end

"""
    scratch_spatial(cfg::SHTConfig, T::Type=Float64) -> Matrix{T}

Allocate a spatial grid buffer of size `(nlat, nlon)`.
"""
scratch_spatial(cfg::SHTConfig, T::Type=Float64) = zeros(T, cfg.nlat, cfg.nlon)

"""
    scratch_fft(cfg::SHTConfig, T::Type=ComplexF64) -> Matrix{T}

Allocate a complex buffer sized for φ-FFTs `(nlat, nlon)`. Reuse this across
`analysis!`/`synthesis!` calls via the `fft_scratch` keyword to avoid per-call
allocations.
"""
scratch_fft(cfg::SHTConfig, T::Type=ComplexF64) = zeros(T, cfg.nlat, cfg.nlon)

"""
    validate_spectral_dimensions(array::AbstractMatrix, cfg::SHTConfig, name::String="array")

Validate that a spectral coefficient array has the correct dimensions.
Throws DimensionMismatch with descriptive message if invalid.
"""
function validate_spectral_dimensions(array::AbstractMatrix, cfg::SHTConfig, name::String="array")
    lmax, mmax = cfg.lmax, cfg.mmax
    expected_rows, expected_cols = lmax+1, mmax+1
    actual_rows, actual_cols = size(array)
    
    if actual_rows != expected_rows
        throw(DimensionMismatch("$name first dim must be lmax+1=$expected_rows, got $actual_rows"))
    end
    if actual_cols != expected_cols  
        throw(DimensionMismatch("$name second dim must be mmax+1=$expected_cols, got $actual_cols"))
    end
    
    return array
end

"""
    validate_spatial_dimensions(array::AbstractMatrix, cfg::SHTConfig, name::String="array")

Validate that a spatial field array has the correct grid dimensions.
"""
function validate_spatial_dimensions(array::AbstractMatrix, cfg::SHTConfig, name::String="array")
    nlat, nlon = cfg.nlat, cfg.nlon
    actual_rows, actual_cols = size(array)
    
    if actual_rows != nlat
        throw(DimensionMismatch("$name first dim must be nlat=$nlat, got $actual_rows"))
    end
    if actual_cols != nlon
        throw(DimensionMismatch("$name second dim must be nlon=$nlon, got $actual_cols"))
    end
    
    return array
end

"""
    validate_spectral_pair_dimensions(array1, array2, cfg::SHTConfig, names::Tuple{String,String}=("array1","array2"))

Validate dimensions for a pair of spectral arrays (commonly Slm, Tlm).
"""
function validate_spectral_pair_dimensions(array1::AbstractMatrix, 
                                            array2::AbstractMatrix, 
                                            cfg::SHTConfig, 
                                            names::Tuple{String,String}=("array1","array2"))
    validate_spectral_dimensions(array1, cfg, names[1])
    validate_spectral_dimensions(array2, cfg, names[2])
    return array1, array2
end

"""
    validate_spatial_pair_dimensions(array1, array2, cfg::SHTConfig, names::Tuple{String,String}=("array1","array2"))

Validate dimensions for a pair of spatial arrays (commonly Vt, Vp).
"""
function validate_spatial_pair_dimensions(array1::AbstractMatrix, 
                                            array2::AbstractMatrix, 
                                            cfg::SHTConfig, 
                                            names::Tuple{String,String}=("array1","array2"))
    validate_spatial_dimensions(array1, cfg, names[1])
    validate_spatial_dimensions(array2, cfg, names[2])
    return array1, array2
end

"""
    validate_qst_dimensions(Qlm, Slm, Tlm, cfg::SHTConfig)

Validate dimensions for QST coefficient triple with descriptive error messages.
"""
function validate_qst_dimensions(Qlm::AbstractMatrix, Slm::AbstractMatrix, Tlm::AbstractMatrix, cfg::SHTConfig)
    validate_spectral_dimensions(Qlm, cfg, "Qlm")
    validate_spectral_dimensions(Slm, cfg, "Slm") 
    validate_spectral_dimensions(Tlm, cfg, "Tlm")
    return Qlm, Slm, Tlm
end

"""
    validate_vector_spatial_dimensions(Vr, Vt, Vp, cfg::SHTConfig)

Validate dimensions for spatial vector field triple (Vr, Vt, Vp).
"""
function validate_vector_spatial_dimensions(Vr::AbstractMatrix, Vt::AbstractMatrix, Vp::AbstractMatrix, cfg::SHTConfig)
    validate_spatial_dimensions(Vr, cfg, "Vr")
    validate_spatial_dimensions(Vt, cfg, "Vt")
    validate_spatial_dimensions(Vp, cfg, "Vp")
    return Vr, Vt, Vp
end

"""
    thread_local_legendre_buffers(lmax::Int, nthreads::Int=Threads.maxthreadid()) -> Vector{Vector{Float64}}

Create thread-local Legendre polynomial buffers to avoid allocations and race conditions.
Returns vector of buffers, one per thread ID up to maxthreadid().

Note: Uses `Threads.maxthreadid()` to handle all possible thread IDs with static scheduling,
since threadid() can return values larger than nthreads() in some threading scenarios.
"""
function thread_local_legendre_buffers(lmax::Int, nthreads::Int=Threads.maxthreadid())
    return [Vector{Float64}(undef, lmax + 1) for _ in 1:nthreads]
end
