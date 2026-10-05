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
    _global_values(arr) -> AbstractArray

The global values of `arr` on every rank: `arr` itself for an ordinary array.
The parallel extension gathers a `PencilArray` (a collective call).
"""
_global_values(arr::AbstractArray) = arr

"""
    _local_block_like(arr, values) -> AbstractArray

Inverse of `_global_values`: `values` itself for an ordinary array, and
this rank's block of the replicated `values` in `arr`'s layout for a
`PencilArray`.
"""
_local_block_like(arr::AbstractArray, values::AbstractArray) = values

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

# Stencil over the interior points
@sht_loop out[I] = src[I + δ(1, I)] - src[I - δ(1, I)] over I ∈ inside(src)
```

# Notes
- Detects CPU vs GPU from the first array in the expression
- GPU path requires GPU extension (CUDA.jl + KernelAbstractions)
- CPU path uses @simd @fastmath @inbounds for vectorization. Before the loop,
  the array accesses every iteration makes are bounds-checked, so a
  `BoundsError` is raised instead of reading or writing out of range: indices
  affine in the loop index at the corners of the iteration range, others at
  every iteration. Accesses that may not run on every iteration (inside `if`,
  `?:`, `&&`, `||`, an inner loop or closure, or after a possible early exit)
  and indices that call anything but integer arithmetic are not checked; keep
  those in range yourself.
- Every iteration must write its own elements. On a GPU the iterations run
  concurrently, so an accumulation into a shared element
  (`a[k] += b[i]` over `i`) races and loses updates; reduce with array
  operations instead.
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

    # The loop runs with @inbounds, where an out-of-range access silently
    # corrupts memory, so check the accesses every iteration makes beforehand:
    # an index affine in the loop index is bounded by its values at the corners
    # of the range (`_bounds_check_points`), any other is checked everywhere.
    refs = _collect_refs!(Expr[], body, names, vcat(names, idx))
    corner_refs = filter(ref -> all(i -> _is_affine_index(i, idx), ref.args[2:end]), refs)
    point_refs = filter(ref -> !(ref in corner_refs), refs)
    checks = Expr[]
    isempty(corner_refs) || push!(checks, :(
        for $loop_item ∈ $_bounds_check_points(R)
            $bind_index
            $(map(_bounds_check, corner_refs)...)
        end))
    isempty(point_refs) || push!(checks, :(
        for $loop_item ∈ R
            $bind_index
            $(map(_bounds_check, point_refs)...)
        end))

    return quote
        # CPU path: SIMD loop
        function $kern_cpu($(symWtypes...), R) where {$(symT...)}
            $(checks...)
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
    _collect_refs!(refs, ex, arrays, allowed) -> refs

Collect the accesses `a[idx...]` to kernel operands that every iteration of a
loop body makes, inner references first so that an index read through another
array (`a[b[I]]`) is itself checked before it is used. Accesses that may not
run are left out, since a guarded access may legitimately be out of range
where its guard fails: branches of `if`, `?:`, `&&` and `||`, inner loops,
closures, comprehensions, `try` and macro calls, and everything after a
statement that can end the iteration early. So are accesses whose indices
cannot be evaluated before the loop (see `_index_is_checkable`).
"""
function _collect_refs!(refs::Vector{Expr}, ex::Expr, arrays::Vector{Symbol},
                        allowed::Vector{Symbol})
    head = ex.head
    if head in (:if, :elseif, :&&, :||)
        # Only the condition runs on every iteration.
        return _collect_refs!(refs, ex.args[1], arrays, allowed)
    elseif head in _SKIPPED_LOOP_HEADS ||
           (head === :(=) && Meta.isexpr(ex.args[1], (:call, :where)))  # f(x) = ...
        return refs
    elseif head === :block
        for statement in ex.args
            _collect_refs!(refs, statement, arrays, allowed)
            _may_end_iteration(statement) && break
        end
        return refs
    end
    for a in ex.args
        _collect_refs!(refs, a, arrays, allowed)
    end
    if head === :ref && ex.args[1] isa Symbol && ex.args[1] in arrays &&
       all(i -> _index_is_checkable(i, allowed), ex.args[2:end])
        ex in refs || push!(refs, ex)
    end
    return refs
end
_collect_refs!(refs::Vector{Expr}, ex, arrays::Vector{Symbol}, allowed::Vector{Symbol}) = refs

# Constructs whose contents may run any number of times, including never.
const _SKIPPED_LOOP_HEADS = (:for, :while, :try, :function, :->, :do, :generator,
                             :comprehension, :typed_comprehension, :macrocall, :quote)

# Statements after one of these may not run on every iteration.
function _may_end_iteration(ex::Expr)
    ex.head in (:continue, :break, :return, :macrocall) && return true
    ex.head === :call && ex.args[1] in (:throw, :error, :rethrow) && return true
    return any(_may_end_iteration, ex.args)
end
_may_end_iteration(ex) = false

"""
    _index_is_checkable(ex, allowed) -> Bool

Whether an index can be evaluated before the loop without changing what the
program does: loop indices, operands, literals, `:`, reads of operands and
the integer arithmetic and index helpers in `_INDEX_CALLS`. Any other call
might have side effects (`rand`) or be expensive, so its access is not checked.
"""
_index_is_checkable(ex::Symbol, allowed) = ex in allowed || ex === :(:)
function _index_is_checkable(ex::Expr, allowed)
    args = if ex.head === :call && ex.args[1] in _INDEX_CALLS
        ex.args[2:end]
    elseif ex.head === :ref && ex.args[1] in allowed
        ex.args[2:end]
    elseif ex.head in (:tuple, :vect)
        ex.args
    else
        return false
    end
    return all(a -> _index_is_checkable(a, allowed), args)
end
_index_is_checkable(ex, allowed) = true  # literals

const _INDEX_CALLS = (:+, :-, :*, :÷, :div, :rem, :mod, :%, :fld, :cld, :abs,
                      :min, :max, :(:), :δ, :CI, :CartesianIndex,
                      :firstindex, :lastindex, :length, :size)

"""
    _is_affine_index(ex, idx) -> Bool

Whether an index is affine in the loop indices `idx`, so that its extremes
over the iteration range are at the corners. `δ(k, I)` is a constant unit
offset. Ranges are not affine in this sense: one that is empty at a corner
checks nothing there.
"""
function _is_affine_index(ex, idx)
    _mentions(ex, idx) || return true  # loop-invariant
    ex isa Symbol && return true
    Meta.isexpr(ex, :call) || return false
    f, args = ex.args[1], ex.args[2:end]
    f === :δ && return true
    f === :* && count(a -> _mentions(a, idx), args) > 1 && return false
    return f in (:+, :-, :*, :CI, :CartesianIndex) && all(a -> _is_affine_index(a, idx), args)
end

_mentions(ex::Symbol, idx) = ex in idx
_mentions(ex::Expr, idx) = any(a -> _mentions(a, idx), ex.args)
_mentions(ex, idx) = false

# Checks an access to an array operand; other operands (a `Ref`, `Tuple` or
# `Dict`) define no `checkbounds` and are left alone.
_bounds_check(ref::Expr) = :($(ref.args[1]) isa AbstractArray &&
                             Base.checkbounds($(ref.args[1]), $(ref.args[2:end]...)))

"""
    _bounds_check_points(R)

The iterations at which `@sht_loop` checks accesses with affine indices before
running the loop with `@inbounds`: the corners of a Cartesian box or the ends
of a range, which bound every such index (stencils such as `a[I + δ(1, I)]`
included), and every element of any other iterable.
"""
_bounds_check_points(R::CartesianIndices) = isempty(R) ? CartesianIndex{ndims(R)}[] :
    vec([CartesianIndex(c) for c in Iterators.product(map(r -> (first(r), last(r)), R.indices)...)])
_bounds_check_points(R::AbstractRange) = isempty(R) ? R : (first(R), last(R))
_bounds_check_points(R) = R

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
