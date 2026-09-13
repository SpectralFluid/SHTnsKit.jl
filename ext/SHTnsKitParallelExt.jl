module SHTnsKitParallelExt

#=
================================================================================
SHTnsKit Parallel Extension - MPI-based Distributed Spherical Harmonic Transforms
================================================================================

This Julia package extension provides distributed/parallel spherical harmonic transform
capabilities using MPI for inter-process communication and PencilArrays for distributed
memory management. The extension is automatically loaded when both MPI.jl and
PencilArrays.jl are available in the environment.

ARCHITECTURE OVERVIEW
---------------------
The parallel transforms use a "pencil decomposition" strategy:
1. Input data (θ,φ grid) is distributed across MPI ranks along one or both dimensions
2. FFT along longitude (φ) transforms spatial data to Fourier coefficients
3. Legendre integration along latitude (θ) produces spherical harmonic coefficients
4. MPI reductions combine partial results from all ranks

Data flow for dist_analysis (spatial → spectral):
    PencilArray(θ,φ) → [gather φ if distributed] → FFT(φ) → Legendre(θ) → Alm

Data flow for dist_synthesis (spectral → spatial):
    Alm → Legendre(θ) → IFFT(φ) → [scatter φ if distributed] → PencilArray(θ,φ)

KEY FILES
---------
- SHTnsKitParallelExt.jl: Module setup, FFT wrappers, utility functions
- ParallelTransforms.jl: Core distributed transform implementations
- ParallelPlans.jl: Pre-allocated buffer management for repeated transforms

PERFORMANCE NOTES
-----------------
1. Memory allocations: The main loop uses "function barriers" to ensure type stability
   and eliminate boxing allocations. See _analysis_loop_no_tables!() in ParallelTransforms.jl

2. FFT plans: FFTW internally caches plans for arrays of the same size/type, so
   repeated transforms on same-sized arrays are efficient after warmup.

3. MPI communication: φ-distributed data requires MPI.Allgatherv! per latitude row.
   For large problems, consider distributing along θ only to minimize communication.

DEBUGGING TIPS
--------------
1. Check extension loaded: `Base.get_extension(SHTnsKit, :SHTnsKitParallelExt) !== nothing`

2. Verify PencilArray layout:
   - `PencilArrays.range_local(pencil(arr))` shows which global indices this rank owns
   - `size(parent(arr))` shows local array dimensions
   - `PencilArrays.size_global(arr)` shows full global dimensions

3. Common issues:
   - Hanging on MPI calls: Ensure all ranks call collective operations
   - Wrong results: Check that θ/φ ranges are correctly identified

4. Enable verbose output: `ENV["SHTNSKIT_VERBOSE_STORAGE"] = "1"`

5. Allocation profiling: Use @allocated or @timed to measure memory usage:
   ```julia
   @timed Alm = SHTnsKit.dist_analysis(cfg, fθφ)
   ```

ENVIRONMENT VARIABLES
--------------------
- SHTNSKIT_FFT_PLAN_CACHE: "1" (default) to cache φ-FFT plans, "0" to disable.
  Read by SHTnsKit proper — the cache is shared with the serial transforms.
  (legacy alias: SHTNSKIT_CACHE_PENCILFFTS)
- SHTNSKIT_VERBOSE_STORAGE: "1" to print storage optimization info
================================================================================
=#

"""
    SHTnsKitParallelExt

Parallel extension module providing MPI-distributed spherical harmonic transforms.
See module-level comments for architecture overview and debugging tips.

# Module state
The extension keeps its locks and caches in direct module constants. The φ-FFT
plan cache itself lives in SHTnsKit proper (src/fftutils.jl) and is shared with
the serial transforms.
"""

using Base.Threads                       # Threads.@threads and locks/macros
import MPI                               # Bring MPI module into scope for MPI.* calls
using MPI: Allreduce, Allreduce!, Allgather, Allgatherv!, VBuffer, Comm_size, COMM_WORLD
import PencilArrays                      # Bring PencilArrays module for qualified calls
using PencilArrays: Pencil, PencilArray, ManyPencilArray  # Distributed array framework
import PencilArrays: pencil, range_local, size_local, size_global, topology, parent
using PencilFFTs                         # Distributed FFTs
using PencilFFTs: Transforms, PencilFFTPlan, allocate_input, allocate_output
using FFTW                               # For 1D FFTs on local arrays
import LinearAlgebra
import SHTnsKit                          # Core spherical harmonic functionality

# `MPI.Comm_free` is present in MPI.jl 0.20.x; newer releases expose
# `MPI.free(::Comm)` instead. Use a shim so subcommunicators are released
# deterministically on either API rather than being left to a finalizer.
@inline function _safe_comm_free(c)
    if isdefined(MPI, :Initialized)
        try
            MPI.Initialized() || return nothing
        catch
            return nothing
        end
    end
    if isdefined(MPI, :Finalized)
        try
            MPI.Finalized() && return nothing
        catch
            return nothing
        end
    end
    if isdefined(MPI, :COMM_NULL)
        try
            c == getfield(MPI, :COMM_NULL) && return nothing
        catch
        end
    end
    if isdefined(MPI, :Comm_free)
        try
            getfield(MPI, :Comm_free)(c)
        catch
            # Communicator may already be freed or auto-finalized; ignore.
        end
    elseif isdefined(MPI, :free)
        try
            getfield(MPI, :free)(c)
        catch
            # Communicator may already be freed or auto-finalized; ignore.
        end
    end
    return nothing
end

# ===== MODULE STATE =====
# The φ-FFT plan cache (and its `enable_fft_plan_cache!` / `disable_fft_plan_cache!`
# / `set_fft_plan_cache!` / `fft_plan_cache_enabled` controls) lives in
# src/fftutils.jl and is shared with the serial transforms. This extension used to
# carry a second, parallel-only cache keyed on the pencil decomposition — but the
# only function that ever consulted it, `_get_or_plan`, had no call sites, and the
# "plans" it stored were `NamedTuple` placeholders that the FFT wrappers ignored.
# Every knob pointed at it was therefore inert. It has been deleted rather than
# left in place; recover it from git history if a PencilFFTs-level cache is ever
# actually needed.

# Compat helper: `ceildiv` was added in Julia 1.11
const _ceildiv = isdefined(Base, :ceildiv) ? Base.ceildiv : (a, b) -> cld(a, b)
ceildiv(a::Integer, b::Integer) = _ceildiv(a, b)

# ===== PENCIL GRID SUGGESTION =====
@inline function _infer_comm_size(comm_or_nprocs::Any)
    comm_or_nprocs === nothing && return 1
    if comm_or_nprocs isa Integer
        return max(1, Int(comm_or_nprocs))
    elseif comm_or_nprocs isa MPI.Comm
        return MPI.Comm_size(comm_or_nprocs)
    end

    try
        return MPI.Comm_size(comm_or_nprocs)
    catch
    end

    for accessor in (:nprocs, :size, :length)
        try
            val = getproperty(comm_or_nprocs, accessor)
            val isa Integer && val > 0 && return Int(val)
        catch
        end
        try
            val = getfield(comm_or_nprocs, accessor)
            val isa Integer && val > 0 && return Int(val)
        catch
        end
    end

    return 1
end

@inline _candidate_remainder_penalty(total::Int, splits::Int) = (total % splits == 0 ? 0.0 : 0.3)

@inline function _candidate_score(nlat::Int, nlon::Int, p_theta::Int, p_phi::Int, prefer_square::Bool)
    theta_chunk = cld(nlat, p_theta)
    phi_chunk = cld(nlon, p_phi)

    chunk_penalty = (max(theta_chunk, phi_chunk) / max(1, min(theta_chunk, phi_chunk))) - 1.0
    shape_ratio = (max(p_theta, p_phi) / max(1, min(p_theta, p_phi))) - 1.0
    shape_penalty = prefer_square ? shape_ratio : 0.25 * shape_ratio

    lat_penalty = _candidate_remainder_penalty(nlat, p_theta)
    lon_penalty = _candidate_remainder_penalty(nlon, p_phi)

    thin_penalty = (theta_chunk < 2 ? 1.0 : 0.0) + (phi_chunk < 2 ? 1.0 : 0.0)

    grid_ratio = nlon == 0 ? 1.0 : float(nlat) / max(1.0, float(nlon))
    proc_ratio = p_phi == 0 ? 1.0 : float(p_theta) / max(1.0, float(p_phi))
    anisotropy_penalty = abs(proc_ratio - grid_ratio) / max(grid_ratio, 1.0)

    return chunk_penalty + shape_penalty + lat_penalty + lon_penalty + thin_penalty + 0.3 * anisotropy_penalty
end

function _suggest_pencil_grid_impl(comm_or_nprocs::Any, nlat::Integer, nlon::Integer;
                                   prefer_square::Bool=true,
                                   allow_one_dim::Bool=true)
    nlat_val = Int(nlat)
    nlon_val = Int(nlon)
    nlat_val > 0 || throw(ArgumentError("nlat must be positive"))
    nlon_val > 0 || throw(ArgumentError("nlon must be positive"))

    nprocs = _infer_comm_size(comm_or_nprocs)
    nprocs <= 1 && return (1, 1)

    best = nothing
    best_score = Inf

    limit = isqrt(nprocs)
    for p_theta in 1:limit
        nprocs % p_theta == 0 || continue
        p_phi = nprocs ÷ p_theta
        for (a, b) in ((p_theta, p_phi), (p_phi, p_theta))
            (a > 0 && b > 0) || continue
            if !allow_one_dim && min(a, b) == 1
                continue
            end
            if a > nlat_val && b > nlon_val
                continue
            end
            score = _candidate_score(nlat_val, nlon_val, a, b, prefer_square)
            if score < best_score - 1e-8
                best = (a, b)
                best_score = score
            elseif best !== nothing && abs(score - best_score) <= 1e-8
                spread = max(a, b) - min(a, b)
                best_spread = max(best...) - min(best...)
                if spread < best_spread
                    best = (a, b)
                elseif spread == best_spread && a >= b && best[1] < best[2]
                    best = (a, b)
                end
            end
        end
    end

    if best === nothing
        if allow_one_dim
            return nlon_val >= nlat_val ? (1, nprocs) : (nprocs, 1)
        else
            return (1, nprocs)
        end
    end

    return best
end

# PencilArrays 0.19 is the package's declared compatibility target.
@inline communicator(A) = PencilArrays.get_comm(A)

# Allocate PencilArray - simplified API for SHTnsKit needs
"""
    allocate(prototype::PencilArray; eltype=eltype(prototype)) -> PencilArray

Allocate a new PencilArray with the same decomposition as the prototype.
The optional `eltype` parameter allows changing the element type.

"""
function allocate(prototype::PencilArray; eltype::Type{T}=eltype(prototype)) where T
    # Get the pencil configuration from the prototype
    pen = pencil(prototype)
    # Allocate a new PencilArray with the same configuration
    return PencilArray{T}(undef, pen)
end

"""
    allocate(T::Type, pen::Pencil) -> PencilArray

Allocate a new PencilArray with the specified type and pencil configuration.
"""
function allocate(::Type{T}, pen::Pencil) where T
    return PencilArray{T}(undef, pen)
end

"""
    allocate_like(prototype::PencilArray, ::Type{T}=eltype(prototype)) -> PencilArray

Create a new PencilArray with the same shape and decomposition as prototype but potentially different type.
"""
function allocate_like(prototype::PencilArray, ::Type{T}=eltype(prototype)) where T
    pen = pencil(prototype)
    return PencilArray{T}(undef, pen)
end

"""
    zeros_like(prototype::PencilArray, ::Type{T}=eltype(prototype)) -> PencilArray

Create a zero-initialized PencilArray with the same shape and decomposition.
"""
function zeros_like(prototype::PencilArray, ::Type{T}=eltype(prototype)) where T
    arr = allocate_like(prototype, T)
    fill!(parent(arr), zero(T))
    return arr
end

@inline globalindices(A, dim) = PencilArrays.range_local(PencilArrays.pencil(A))[dim]

# ===== DISTRIBUTED FFT WRAPPERS =====
# Use FFTW for 1D FFTs along the longitude dimension (not PencilFFTs which is for multi-D)
# PencilArrays provides the distributed array framework, FFTW provides the FFTs

"""
    fft_along_dim2!(output, input)

Perform forward FFT along dimension 2 (longitude) for each row.
Works on the local data of a PencilArray.
"""
function fft_along_dim2!(output::AbstractMatrix{Complex{T}}, input::AbstractMatrix{T2}) where {T<:AbstractFloat, T2}
    nlat, nlon = size(input)
    # Use a contiguous temp buffer for FFT (avoids stride mismatch with cached plans)
    # FFTW internally caches plans for the same size/type, so this is still efficient after warmup
    temp = Vector{Complex{T}}(undef, nlon)
    @inbounds for i in 1:nlat
        # Copy with conversion to complex into contiguous buffer
        for j in 1:nlon
            temp[j] = Complex{T}(input[i, j])
        end
        # In-place FFT on contiguous buffer (FFTW caches the plan internally)
        FFTW.fft!(temp)
        # Copy result back to output
        for j in 1:nlon
            output[i, j] = temp[j]
        end
    end
    return output
end

function fft_along_dim2!(output::AbstractMatrix{Complex{T}}, input::AbstractMatrix{Complex{T}}) where {T<:AbstractFloat}
    nlat, nlon = size(input)
    # Use a contiguous temp buffer for FFT
    temp = Vector{Complex{T}}(undef, nlon)
    @inbounds for i in 1:nlat
        # Copy to contiguous buffer
        for j in 1:nlon
            temp[j] = input[i, j]
        end
        # In-place FFT
        FFTW.fft!(temp)
        # Copy result back
        for j in 1:nlon
            output[i, j] = temp[j]
        end
    end
    return output
end

"""
    ifft_along_dim2!(output, input)

Perform inverse FFT along dimension 2 (longitude) for each row.
"""
function ifft_along_dim2!(output::AbstractMatrix{T}, input::AbstractMatrix{Complex{T2}}) where {T<:AbstractFloat, T2<:AbstractFloat}
    nlat, nlon = size(input)
    # Pre-allocate contiguous temp buffer OUTSIDE the loop
    temp = Vector{Complex{T2}}(undef, nlon)
    @inbounds for i in 1:nlat
        # Copy to contiguous buffer
        for j in 1:nlon
            temp[j] = input[i, j]
        end
        # In-place IFFT (FFTW caches plans internally)
        FFTW.ifft!(temp)
        # Copy real part back to output
        for j in 1:nlon
            output[i, j] = real(temp[j])
        end
    end
    return output
end

function ifft_along_dim2!(output::AbstractMatrix{Complex{T}}, input::AbstractMatrix{Complex{T}}) where {T<:AbstractFloat}
    nlat, nlon = size(input)
    # Pre-allocate contiguous temp buffer
    temp = Vector{Complex{T}}(undef, nlon)
    @inbounds for i in 1:nlat
        # Copy to contiguous buffer
        for j in 1:nlon
            temp[j] = input[i, j]
        end
        # In-place IFFT
        FFTW.ifft!(temp)
        # Copy back to output
        for j in 1:nlon
            output[i, j] = temp[j]
        end
    end
    return output
end

# ===== OPTIMIZED DISTRIBUTED FFT USING TRANSPOSE =====
# When φ is distributed, use a single all-to-all transpose instead of per-row Allgatherv.
# This reduces the number of MPI calls from O(nlat) to O(1).

"""
    _gather_phi_rows(local_data, θ_range, φ_range, nlon, comm) -> Matrix{eltype(local_data)}

Row-subcomm Allgatherv used by both `distributed_fft_phi!` and
`distributed_rfft_phi!`. Ranks that share the θ-slab exchange their φ segments
so every rank in the row ends up with a full `(nlat_local, nlon)` matrix along
its own θ rows. Handles 1D (φ-only split, subcomm == global comm) and 2D
(pθ×pφ, subcomm has pφ ranks) decompositions uniformly.
"""
function _gather_phi_rows(local_data::AbstractMatrix,
                          θ_range::AbstractRange, φ_range::AbstractRange,
                          nlon::Int, comm)
    nlat_local = length(θ_range)
    nlon_local = length(φ_range)

    # A rank owning zero θ rows has no slab to gather, and it must NOT be coloured
    # by `first(θ_range)`: an empty range still reports `first == 1`, so it would
    # join the row of the genuine owner of global θ index 1 and trip the
    # nlat_local consistency check below (killing that row while the other rows
    # proceed, i.e. a hang). MPI_UNDEFINED — `nothing` in MPI.jl — puts it in no
    # row at all; Comm_split hands back COMM_NULL and it skips the exchange.
    θ_color = isempty(θ_range) ? nothing : Int(first(θ_range))
    row_comm = MPI.Comm_split(comm, θ_color, MPI.Comm_rank(comm))
    if θ_color === nothing
        _safe_comm_free(row_comm)
        return Matrix{eltype(local_data)}(undef, 0, nlon)
    end
    gathered_data = try
        row_nprocs = MPI.Comm_size(row_comm)

        all_nlats = MPI.Allgather(nlat_local, row_comm)
        if !all(==(nlat_local), all_nlats)
            throw(ErrorException("_gather_phi_rows: θ_range mismatch within row subcomm (nlat_local = $all_nlats). Pencil topology is inconsistent."))
        end

        all_nlons = MPI.Allgather(nlon_local, row_comm)
        φ_displs = cumsum([0; all_nlons[1:end-1]])
        sum(all_nlons) == nlon || throw(ErrorException("_gather_phi_rows: row subcomm φ segments sum to $(sum(all_nlons)), expected nlon=$nlon."))

        T = eltype(local_data)
        send_buf = Vector{T}(undef, nlat_local * nlon_local)
        recv_buf = Vector{T}(undef, nlat_local * nlon)

        idx = 1
        @inbounds for j in 1:nlon_local
            for i in 1:nlat_local
                send_buf[idx] = local_data[i, j]
                idx += 1
            end
        end

        recv_counts = [nlat_local * Int(all_nlons[r]) for r in 1:row_nprocs]
        recv_displs = cumsum([0; recv_counts[1:end-1]])
        MPI.Allgatherv!(send_buf, VBuffer(recv_buf, recv_counts, recv_displs), row_comm)

        out = Matrix{T}(undef, nlat_local, nlon)
        @inbounds for r in 1:row_nprocs
            offset = recv_displs[r]
            r_nlon = Int(all_nlons[r])
            φ_start = Int(φ_displs[r]) + 1
            idx = 1
            for j in 1:r_nlon
                φ_idx = φ_start + j - 1
                for i in 1:nlat_local
                    out[i, φ_idx] = recv_buf[offset + idx]
                    idx += 1
                end
            end
        end
        out
    finally
        _safe_comm_free(row_comm)
    end
    return gathered_data
end

"""
    distributed_fft_phi!(Fθm_out, local_data, θ_range, φ_range, nlon, comm)

Distributed FFT along φ (longitude): gather full rows across the row subcomm,
then FFT each local θ row in place. `Fθm_out[i, m+1]` holds the Fourier mode
`m` at local θ index `i` on return.
"""
function distributed_fft_phi!(Fθm_out::AbstractMatrix{Complex{T}},
                               local_data::AbstractMatrix,
                               θ_range::AbstractRange, φ_range::AbstractRange,
                               nlon::Int, comm) where {T<:AbstractFloat}
    gathered = _gather_phi_rows(local_data, θ_range, φ_range, nlon, comm)
    fft_along_dim2!(Fθm_out, gathered)
    return Fθm_out
end

"""
    distributed_rfft_phi!(Fθm_out, local_data, θ_range, φ_range, nlon, comm)

Distributed real-FFT along φ. Same row-subcomm gather as `distributed_fft_phi!`
but runs `rfft` on the gathered real row → `Fθm_out` shape `(nlat_local, nlon÷2+1)`.
Requires `eltype(local_data) <: Real`.
"""
function distributed_rfft_phi!(Fθm_out::AbstractMatrix{Complex{T}},
                                local_data::AbstractMatrix{<:Real},
                                θ_range::AbstractRange, φ_range::AbstractRange,
                                nlon::Int, comm) where {T<:AbstractFloat}
    nlat_local = length(θ_range)
    size(Fθm_out) == (nlat_local, nlon ÷ 2 + 1) ||
        throw(DimensionMismatch("Fθm_out must be (nlat_local, nlon÷2+1)"))
    gathered = _gather_phi_rows(local_data, θ_range, φ_range, nlon, comm)
    Fθm_out .= FFTW.rfft(gathered, 2)
    return Fθm_out
end

"""
    distributed_ifft_phi!(local_out, Fθm, θ_range, φ_range, nlon, comm)

Optimized distributed IFFT along φ (longitude) dimension.
Inverse of distributed_fft_phi! - performs IFFT then scatters back to distributed layout.
"""
function distributed_ifft_phi!(local_out::AbstractMatrix,
                                Fθm::AbstractMatrix{<:Complex},
                                θ_range::AbstractRange, φ_range::AbstractRange,
                                nlon::Int, comm)
    nlat_local = length(θ_range)
    nlon_local = length(φ_range)

    # Perform IFFT on complete rows
    spatial_full = Matrix{eltype(local_out)}(undef, nlat_local, nlon)
    ifft_along_dim2!(spatial_full, Fθm)

    # Extract local portion (no communication needed - just take local φ slice)
    φ_start = first(φ_range)
    @inbounds for j in 1:nlon_local
        for i in 1:nlat_local
            local_out[i, j] = spatial_full[i, φ_start + j - 1]
        end
    end

    return local_out
end

# ===== PARALLEL EXTENSION MODULES =====
# Include specialized modules for different aspects of parallel spherical harmonic transforms
include("ParallelGPU.jl")              # Vendor-neutral GPU storage/communication policy
include("ParallelDiagnostics.jl")      # Diagnostic and profiling tools for parallel operations
include("ParallelDispatch.jl")         # Function dispatch and interface definitions
include("ParallelPlans.jl")            # Distributed transform planning and setup
include("ParallelTransforms.jl")       # Core parallel transform implementations
include("ParallelOpsPencil.jl")       # Parallel differential operators using PencilArrays
include("ParallelRotationsPencil.jl") # Parallel spherical rotation operations
include("ParallelLocal.jl")            # Local (per-process) operations and utilities
include("ParallelTransposeTransforms.jl")  # Transpose-based distributed SHT (Task 2+)

# Reduction of spectral partials over θ.
# A plain Allreduce is the correct primitive here: a tuned MPI already performs
# topology-aware (hierarchical) reduction internally. This used to route through
# `adaptive_spectral_communication!` → `hierarchical_spectral_reduce!`, which wraps
# the SAME `Allreduce!` in TWO per-call `MPI.Comm_split` collectives (`tree_reduce!`
# was literally `Allreduce!`) for zero algorithmic gain — strictly slower on the hot
# analysis/sphtor path. That whole adaptive/sparse/segmented/hierarchical tree has
# been deleted rather than left unreachable; recover it from git history if a real
# use case for the sparse or segmented strategies ever appears.
function efficient_spectral_reduce!(local_data::AbstractMatrix, comm)
    MPI.Allreduce!(local_data, +, comm)
    return local_data
end

function efficient_spectral_reduce!(local_data::AbstractVector, comm)
    MPI.Allreduce!(local_data, +, comm)
    return local_data
end

# Note: Avoid forwarding Base.zeros(Pencil) to PencilArrays.zeros to prevent
# potential recursion when PencilArrays.zeros may call Base.zeros internally.

# ===== EXPORTS =====
# Export types and functions defined in this extension

# 1D Distributed spectral types and functions (from ParallelTransforms.jl)
export DistributedSpectralPlan, DistributedSpectralArray
export create_distributed_spectral_plan, create_distributed_spectral_array
export gather_to_dense, scatter_from_dense!
export dist_analysis_distributed, dist_synthesis_distributed
export distributed_spectral_reduce!
export estimate_distributed_memory_savings

# 2D Distributed spectral types and functions (from ParallelTransforms.jl)
export DistributedSpectralPlan2D, DistributedSpectralArray2D
export create_distributed_spectral_plan_2d, create_distributed_spectral_array_2d
export suggest_spectral_grid
export gather_to_dense_2d, gather_to_full_dense_2d, scatter_from_dense_2d!
export dist_analysis_distributed_2d, dist_synthesis_distributed_2d
export dist_synthesis_distributed_2d_optimized
export estimate_distributed_memory_savings_2d
export validate_2d_distribution_alignment

# Plan types (from ParallelPlans.jl)
export DistAnalysisPlan, DistPlan, DistSphtorPlan, DistQstPlan

# Utility functions
export local_size, global_size
export validate_plm_tables, estimate_plm_tables_memory
export estimate_memory_savings

end # module SHTnsKitParallelExt
