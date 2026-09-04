#=
================================================================================
loop.jl - Unified CPU/GPU Loop Abstraction for Spherical Harmonic Operations
================================================================================

This file provides a @sht_loop macro that enables unified execution across CPU
(with SIMD) and distributed arrays (PencilArrays). GPU support is added when
the GPU extension is loaded (requires KernelAbstractions).

WHY A UNIFIED LOOP ABSTRACTION?
-------------------------------
Spherical harmonic transforms involve nested loops over:
- Latitude bands (θ direction)
- Azimuthal modes (m direction)
- Degree indices (l direction)

These loops need to run efficiently on CPU, GPU, and distributed systems.
Rather than writing separate code paths, we use a macro that automatically
selects the appropriate backend based on the array type.

HOW IT WORKS
------------
1. For regular CPU Arrays: Uses @simd with @fastmath @inbounds for vectorization
2. For GPU arrays (CuArray): Uses KernelAbstractions @kernel (when GPU extension loaded)
3. For PencilArrays: Operates on local data via parent(), uses SIMD

PENCILARRAY SUPPORT
-------------------
When the first array is a PencilArray (MPI-distributed), the macro:
1. Extracts local data using parent(arr)
2. Runs SIMD loops on the local portion
3. Does NOT handle MPI communication - that's the caller's responsibility

For distributed operations that need MPI reductions, use the full dist_* API.

USAGE
-----
```julia
# Parallel loop over latitude and mode indices
@sht_loop Fφ[i_lat, m_idx] = result over (i_lat, m_idx) ∈ CartesianIndices((nlat, mmax+1))

# Simple 1D loop over coefficients
@sht_loop alm[l+1, m+1] = acc over (l, m) ∈ CartesianIndices((lmax+1, mmax+1))

# With PencilArrays (operates on local data only)
@sht_loop local_field[I] = value over I ∈ CartesianIndices(size(parent(pencil_arr)))
```

CONFIGURATION
-------------
Force SIMD-only mode (disable GPU path):
```julia
SHTnsKit.set_loop_backend("SIMD")  # Always use CPU SIMD path
SHTnsKit.set_loop_backend("auto")  # Auto-detect from array type (default)
```

Query current backend mode:
```julia
SHTnsKit.loop_backend()  # Returns "auto" or "SIMD"
```

================================================================================
=#

# Backend preference: "auto" for automatic detection, "SIMD" to force CPU
const _LOOP_BACKEND = Ref{String}("auto")

# GPU support flag - set to true when GPU extension is loaded
const _GPU_LOOP_AVAILABLE = Ref{Bool}(false)

# GPU kernel launcher callback - set by GPU extension
const _GPU_KERNEL_LAUNCHER = Ref{Any}(nothing)

"""
    loop_backend()

Return the current loop backend mode: "auto" or "SIMD".
"""
loop_backend() = _LOOP_BACKEND[]

"""
    set_loop_backend(backend::String)

Set the loop backend preference. Valid values:
- "auto": Automatically detect from array type (default) - uses GPU when arrays are on GPU
- "SIMD": Force CPU SIMD path (useful for debugging or when GPU is not desired)
"""
function set_loop_backend(backend::String)
    if !(backend in ("SIMD", "auto"))
        throw(ArgumentError("Invalid backend: \"$backend\". Use \"SIMD\" or \"auto\"."))
    end
    _LOOP_BACKEND[] = backend
    return backend
end

# Work-group size for GPU kernels (64 is a common optimal value)
const _WORKGROUP_SIZE = 64

"""
    _is_cpu_array(arr)

Check if the array is a standard CPU array (not GPU).
Uses duck typing to detect GPU arrays without requiring GPU packages.
"""
function _is_cpu_array(arr::AbstractArray)
    T = typeof(arr)
    type_name = string(T)
    # Check for common GPU array types
    return !occursin("CuArray", type_name) &&
           !occursin("ROCArray", type_name) &&
           !occursin("MtlArray", type_name) &&
           !occursin("oneArray", type_name)
end

"""
    _is_pencil_array(arr)

Check if the array is a PencilArray (distributed MPI array).
Uses duck typing to avoid hard dependency on PencilArrays.
"""
function _is_pencil_array(arr::AbstractArray)
    T = typeof(arr)
    type_name = string(nameof(T))
    return occursin("PencilArray", type_name) || occursin("ManyPencilArray", type_name)
end

"""
    _get_local_data(arr)

Get the local data from an array. For PencilArrays, this returns parent(arr).
For regular arrays, returns the array unchanged.
"""
function _get_local_data(arr::AbstractArray)
    if _is_pencil_array(arr)
        return parent(arr)
    else
        return arr
    end
end

"""
    _enable_gpu_loops!(launcher)

Called by GPU extension to enable GPU loop support.
"""
function _enable_gpu_loops!(launcher)
    _GPU_LOOP_AVAILABLE[] = true
    _GPU_KERNEL_LAUNCHER[] = launcher
end

"""
    @sht_loop <expr> over <I ∈ R>

Macro to automate fast loops using @simd when running on CPU,
or KernelAbstractions when running on GPU (requires GPU extension).

The macro extracts all symbols from the expression, creates kernel functions
for both CPU and GPU paths, and dispatches based on the array backend at runtime.

# Examples
```julia
# Loop over 2D Cartesian range
@sht_loop dest[i, j] = src[i, j] * scale over (i, j) ∈ CartesianIndices((n, m))

# Loop with accumulation pattern
@sht_loop alm[l+1, m+1] += weights[i] * Plm[i, l+1, m+1] * Fm[i, m+1] over i ∈ 1:nlat
```

# Notes
- Detects CPU vs GPU from the first array in the expression
- GPU path requires GPU extension (CUDA.jl + KernelAbstractions)
- CPU path uses @simd @fastmath @inbounds for vectorization
- Set `SHTnsKit.set_loop_backend("SIMD")` to force CPU path

Field accesses in the body (e.g. `cfg.scale`) are supported: the *expression* is
evaluated once at the call site and passed in under a hygienic name, so it can
never pick up an unrelated caller local that happens to share the field name.
"""
macro sht_loop(args...)
    ex, _, itr = args
    _, I, R = itr.args
    idx = _loop_index_symbols(I)            # loop indices are bound by the kernel, not passed in
    ops = Any[]
    _grab_ops!(ops, ex, idx)                # operand expressions, in first-use order
    isempty(ops) && throw(ArgumentError("@sht_loop: loop body references no operands"))

    # Plain variables keep their own name (the quote is `esc`aped, so the kernel
    # argument resolves to the caller's binding). A field access gets a gensym
    # instead and is passed *as the original expression*: rewriting `cfg.scale` to
    # the bare symbol `scale` would silently bind to any caller local of that name
    # — names like `w`, `x`, `scale`, `norm` are everywhere in transform code — and
    # multiply by the wrong value with no error.
    names = Symbol[op isa Symbol ? op : gensym(_field_name(op)) for op in ops]
    body = _subst_ops(ex, ops, names)

    symT = [gensym() for _ in 1:length(names)]  # Generate type parameters
    symWtypes = joinsymtype(names, symT)        # Symbols with types: [a::A, b::B, ...]

    @gensym kern_cpu dispatch_kern loop_item
    bind_index = if I isa Symbol
        :($I = $loop_item)
    else
        Expr(:block, [:( $(I.args[k]) = $loop_item[$k] ) for k in eachindex(I.args)]...)
    end

    return quote
        # CPU path: SIMD loop
        function $kern_cpu($(symWtypes...), R) where {$(symT...)}
            # `@simd` requires a single-symbol iteration variable. Bind the
            # user-facing symbol or tuple inside the loop so documented forms
            # such as `(i, j) ∈ CartesianIndices(...)` compile as well.
            @simd for $loop_item ∈ R
                $bind_index
                @fastmath @inbounds $body
            end
        end

        # Dispatch function: choose backend based on array type
        function $dispatch_kern($(symWtypes...), R) where {$(symT...)}
            first_arr = $(names[1])

            # Check for PencilArray (MPI-distributed) - always use CPU SIMD on local data
            if $_is_pencil_array(first_arr)
                # For PencilArrays, get local data and run SIMD
                # Note: Caller must handle MPI communication separately
                $kern_cpu($(names...), R)
            elseif $_LOOP_BACKEND[] == "SIMD" || $_is_cpu_array(first_arr)
                # Regular CPU array - use SIMD
                $kern_cpu($(names...), R)
            elseif $_GPU_LOOP_AVAILABLE[]
                # GPU array - use GPU extension's kernel launcher
                $_GPU_KERNEL_LAUNCHER[]($(names...), R, $loop_item -> begin
                    $bind_index
                    $body
                end)
            else
                # GPU array but extension not loaded - fall back to CPU
                @warn "GPU array detected but GPU extension not loaded. Using CPU fallback." maxlog=1
                $kern_cpu($(names...), R)
            end
        end

        # Call the dispatcher — field accesses are evaluated HERE, in the caller
        $dispatch_kern($(ops...), $R)
    end |> esc
end

# Helper functions for macro symbol extraction (matching BioFlow.jl pattern)

"""
    _loop_index_symbols(I)

Extract all symbols from a loop index pattern. Handles both single symbols
and tuple destructuring patterns like `(i, j)` or `(l, m)`.

# Examples
```julia
_loop_index_symbols(:I)           # Returns [:I]
_loop_index_symbols(:(i, j))      # Returns [:i, :j]
_loop_index_symbols(:(l, m, n))   # Returns [:l, :m, :n]
```
"""
function _loop_index_symbols(I)
    if I isa Symbol
        return [I]
    elseif I isa Expr && I.head == :tuple
        # Tuple destructuring like (i, j) - extract all symbol arguments
        return Symbol[arg for arg in I.args if arg isa Symbol]
    else
        # Unknown pattern - return empty (conservative)
        return Symbol[]
    end
end

"""
    _is_field_access(ex) -> Bool

True for a literal `a.b` getproperty node. Deliberately excludes broadcast
syntax (`f.(x)`), which shares the `:.` head but carries a tuple, not a
`QuoteNode`, in `args[2]`.
"""
_is_field_access(ex::Expr) = ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
_is_field_access(ex) = false

_field_name(ex::Expr) = Symbol(ex.args[2].value)

"""
    _grab_ops!(ops, ex, idx)

Collect the operand expressions of a loop body into `ops`, in first-use order and
without duplicates: plain variables as `Symbol`s, field accesses as the whole
`a.b` expression. Loop indices (`idx`) and callee names are skipped.
"""
function _grab_ops!(ops::Vector{Any}, ex::Expr, idx::Vector{Symbol})
    if _is_field_access(ex)
        ex in ops || push!(ops, ex)
        return ops
    end
    # Don't grab function names in calls
    start = ex.head == :call ? 2 : 1
    for a in ex.args[start:end]
        _grab_ops!(ops, a, idx)
    end
    return ops
end

function _grab_ops!(ops::Vector{Any}, ex::Symbol, idx::Vector{Symbol})
    (ex in idx || ex in ops) || push!(ops, ex)
    return ops
end

_grab_ops!(ops::Vector{Any}, ex, idx::Vector{Symbol}) = ops  # Ignore literals, etc.

"""
    _subst_ops(ex, ops, names) -> expr

Rewrite the loop body so each operand is referred to by its kernel-argument name.
Plain variables map to themselves; a field access maps to its hygienic gensym.
"""
function _subst_ops(ex::Expr, ops::Vector{Any}, names::Vector{Symbol})
    if _is_field_access(ex)
        i = findfirst(==(ex), ops)
        return i === nothing ? ex : names[i]
    end
    return Expr(ex.head, map(a -> _subst_ops(a, ops, names), ex.args)...)
end

function _subst_ops(ex::Symbol, ops::Vector{Any}, names::Vector{Symbol})
    i = findfirst(==(ex), ops)
    return i === nothing ? ex : names[i]
end

_subst_ops(ex, ops::Vector{Any}, names::Vector{Symbol}) = ex

"""
    joinsymtype(sym, symT)

Join symbols with their type parameters to create typed argument lists.
"""
joinsymtype(sym::Symbol, symT::Symbol) = Expr(:(::), sym, symT)
joinsymtype(sym, symT) = [joinsymtype(s, t) for (s, t) in zip(sym, symT)]

# CartesianIndex utilities (similar to BioFlow.jl)

"""
    CI(a...)

Shorthand constructor for CartesianIndex.
"""
@inline CI(a...) = CartesianIndex(a...)

"""
    δ(i, N::Int)
    δ(i, I::CartesianIndex{N})

Return a CartesianIndex of dimension N which is one at index i and zero elsewhere.
Useful for offsetting indices in a specific direction.

# Example
```julia
δ(1, CartesianIndex(2,3))  # Returns CartesianIndex(1,0)
```
"""
δ(i, ::Val{N}) where N = CI(ntuple(j -> j == i ? 1 : 0, N))
δ(i, I::CartesianIndex{N}) where N = δ(i, Val{N}())

"""
    inside(a; buff=1)

Return CartesianIndices range excluding `buff` layers of cells on all boundaries.
Useful for iterating over interior points while respecting boundary conditions.
"""
@inline inside(a::AbstractArray; buff=1) = CartesianIndices(
    map(ax -> first(ax)+buff:last(ax)-buff, axes(a))
)

"""
    @sht_inside <arr[I] = expr>

Convenience macro to loop over interior points of an array, excluding boundaries.
Automatically determines the loop range from the array size.

# Example
```julia
@sht_inside field[I] = 0.5 * (field_old[I+δ(1,I)] + field_old[I-δ(1,I)])
```
"""
macro sht_inside(ex)
    @assert ex.head == :(=) && ex.args[1].head == :ref
    a, I = ex.args[1].args[1:2]
    return quote
        SHTnsKit.@sht_loop $ex over $I ∈ SHTnsKit.inside($a)
    end |> esc
end

# Spherical harmonic specific loop ranges

"""
    spectral_range(lmax, mmax)

Return a CartesianIndices range for iterating over valid (l,m) coefficient pairs.
Note: Returns range for 1-based indexing of storage array, so iterate as:
    for idx ∈ spectral_range(lmax, mmax)
        l, m = idx[1] - 1, idx[2] - 1  # Convert to 0-based degree/order
        # ... work with alm[l+1, m+1]
    end
"""
spectral_range(lmax::Int, mmax::Int) = CartesianIndices((lmax+1, mmax+1))

"""
    spatial_range(nlat, nlon)

Return a CartesianIndices range for iterating over spatial grid points.
"""
spatial_range(nlat::Int, nlon::Int) = CartesianIndices((nlat, nlon))

"""
    latitude_range(nlat)

Return a range for iterating over latitude bands.
"""
latitude_range(nlat::Int) = 1:nlat

"""
    mode_range(mmax)

Return a range for iterating over azimuthal modes (0 to mmax).
Storage index: m+1 for m ∈ 0:mmax
"""
mode_range(mmax::Int) = 0:mmax

"""
    local_range(arr)

Return CartesianIndices for iterating over the local portion of an array.
For PencilArrays, this iterates over parent(arr) dimensions.
For regular arrays, this is equivalent to CartesianIndices(arr).

# Example
```julia
# With PencilArray (MPI-distributed)
@sht_loop local_data[I] = value over I ∈ local_range(pencil_arr)

# With regular array
@sht_loop field[I] = value over I ∈ local_range(regular_arr)
```
"""
function local_range(arr::AbstractArray)
    if _is_pencil_array(arr)
        return CartesianIndices(parent(arr))
    else
        return CartesianIndices(arr)
    end
end

"""
    local_size(arr)

Return the size of the local portion of an array.
For PencilArrays, returns size(parent(arr)).
For regular arrays, returns size(arr).
"""
function local_size(arr::AbstractArray)
    if _is_pencil_array(arr)
        return size(parent(arr))
    else
        return size(arr)
    end
end
