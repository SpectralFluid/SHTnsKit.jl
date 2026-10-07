module GPUCommon

using KernelAbstractions
using SHTnsKit

export laplacian_kernel!, operator_matrix_kernel!, packed_operator_kernel!,
       legendre_table_kernel!, scalar_analysis_kernel!,
       scalar_synthesis_kernel!, coefficient_conversion_kernel!,
       coefficient_batch_conversion_kernel!,
       real_pack_kernel!, real_unpack_kernel!,
       mode_analysis_kernel!, mode_synthesis_kernel!,
       scalar_batch_analysis_kernel!, scalar_batch_synthesis_kernel!,
       complex_packed_analysis_kernel!, complex_packed_synthesis_kernel!,
       vector_derivative_table_kernel!, vector_analysis_kernel!,
       vector_synthesis_kernel!, vector_diagonal_kernel!,
       vector_mode_analysis_kernel!, vector_mode_synthesis_kernel!,
       vector_batch_analysis_kernel!, vector_batch_synthesis_kernel!,
       vector_host_tables,
       scalar_config_signature, vector_config_signature, scalar_host_tables,
       ScalarTableCache, scalar_cache_lookup, scalar_cache_insert!,
       scalar_cache_publish!,
       scalar_cache_clear!, scalar_cache_size,
       ScalarWorkspaceCache, scalar_workspace_use!,
       scalar_workspace_clear!, scalar_workspace_size
export RotationBlockCache, rotation_cache_lookup, rotation_cache_insert!,
       rotation_cache_publish!, rotation_cache_clear!, rotation_cache_size,
       rotation_z_real_kernel!, rotation_real_kernel!, rotation_cplx_kernel!,
       launch_sht_loop!

@kernel function _sht_loop_kernel!(body, range)
    linear_index = @index(Global, Linear)
    if linear_index <= length(range)
        body(@inbounds range[linear_index])
    end
end

"""Launch a body supplied by `@sht_loop` on the first operand's KA backend."""
function launch_sht_loop!(args...)
    first_array = args[1]
    range = args[end - 1]
    body = args[end]
    backend = KernelAbstractions.get_backend(first_array)
    kernel! = _sht_loop_kernel!(backend)
    kernel!(body, range; ndrange=length(range))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
One cached table set plus its mutable-configuration signature, LRU tick and,
when known, a weak reference to the configuration it was built for.
"""
struct ScalarTableCacheEntry
    signature::UInt
    tick::UInt64
    value::Any
    owner::Union{Nothing,WeakRef}
end

"""Immutable rotation-block cache entry with LRU publication tick and size."""
struct RotationBlockCacheEntry
    tick::UInt64
    value::Any
    bytes::Int
end

"""
Thread-safe bounded cache keyed by device, precision, and rotation inputs.

Besides the entry count, the device memory held per device is bounded by
`max_bytes_per_device` (1 GiB by default): a dense block set grows like lmax³
(about 11 GB in Float64 at lmax=1023), so eight cached sets could pin the whole
device. A set larger than the budget is returned without being cached.
"""
mutable struct RotationBlockCache
    entries::Dict{Tuple,RotationBlockCacheEntry}
    tick::UInt64
    max_per_device::Int
    max_bytes_per_device::Int
    lock::ReentrantLock
end

function RotationBlockCache(max_per_device::Integer=8;
                            max_bytes_per_device::Integer=1 << 30)
    max_per_device > 0 || throw(ArgumentError("max_per_device must be positive"))
    max_bytes_per_device >= 0 ||
        throw(ArgumentError("max_bytes_per_device must be non-negative"))
    return RotationBlockCache(Dict{Tuple,RotationBlockCacheEntry}(), 0,
                              Int(max_per_device), Int(max_bytes_per_device),
                              ReentrantLock())
end

@inline _rotation_device(key::Tuple) = key[1]

"""Bytes held by the array fields of a cached value (device or host arrays)."""
function _cached_bytes(value)
    total = 0
    for name in fieldnames(typeof(value))
        field = getfield(value, name)
        field isa AbstractArray && (total += sizeof(field))
    end
    return total
end

function rotation_cache_lookup(cache::RotationBlockCache, key::Tuple)
    return lock(cache.lock) do
        entry = get(cache.entries, key, nothing)
        entry === nothing && return nothing
        cache.tick += 1
        cache.entries[key] = RotationBlockCacheEntry(cache.tick, entry.value, entry.bytes)
        return entry.value
    end
end

function rotation_cache_insert!(cache::RotationBlockCache, key::Tuple, value)
    return lock(cache.lock) do
        existing = get(cache.entries, key, nothing)
        if existing !== nothing
            cache.tick += 1
            cache.entries[key] = RotationBlockCacheEntry(
                cache.tick, existing.value, existing.bytes,
            )
            return existing.value
        end
        bytes = _cached_bytes(value)
        bytes > cache.max_bytes_per_device && return value
        device_keys = [candidate for candidate in keys(cache.entries)
                       if _rotation_device(candidate) == _rotation_device(key)]
        held = sum(candidate -> cache.entries[candidate].bytes, device_keys; init=0)
        while !isempty(device_keys) &&
              (length(device_keys) >= cache.max_per_device ||
               held + bytes > cache.max_bytes_per_device)
            oldest = argmin(candidate -> cache.entries[candidate].tick, device_keys)
            held -= cache.entries[oldest].bytes
            delete!(cache.entries, oldest)
            filter!(!=(oldest), device_keys)
        end
        cache.tick += 1
        cache.entries[key] = RotationBlockCacheEntry(cache.tick, value, bytes)
        return value
    end
end

function rotation_cache_publish!(complete, cache::RotationBlockCache,
                                 key::Tuple, value)
    complete()
    return rotation_cache_insert!(cache, key, value)
end

function rotation_cache_clear!(cache::RotationBlockCache; device=nothing)
    lock(cache.lock) do
        if device === nothing
            empty!(cache.entries)
        else
            for key in collect(keys(cache.entries))
                _rotation_device(key) == device && delete!(cache.entries, key)
            end
        end
    end
    return nothing
end

function rotation_cache_size(cache::RotationBlockCache; device=nothing)
    return lock(cache.lock) do
        device === nothing && return length(cache.entries)
        return count(key -> _rotation_device(key) == device, keys(cache.entries))
    end
end

"""Batched form of coefficient conversion without broadcast temporaries."""
@kernel function coefficient_batch_conversion_kernel!(output, input, scales,
                                                        lmax, mmax, to_internal)
    l_idx, m_idx, batch_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1 &&
       batch_idx <= size(output, 3)
        l = l_idx - 1
        m = m_idx - 1
        if l >= m
            scale = scales[l_idx, m_idx]
            output[l_idx, m_idx, batch_idx] = to_internal ?
                scale * input[l_idx, m_idx, batch_idx] :
                input[l_idx, m_idx, batch_idx] / scale
        else
            output[l_idx, m_idx, batch_idx] = zero(eltype(output))
        end
    end
end

"""
Thread-safe, per-device bounded cache for immutable scalar transform tables.

The dictionary key deliberately uses configuration identity rather than its
mutable signature. A convention/grid mutation therefore replaces the stale
entry instead of accumulating another device allocation. Values are built
outside this cache's lock by the vendor extension.

Entries inserted with an `owner` hold it weakly and are dropped at the next
cache access once it has been reclaimed. Before, a strong entry keyed only by
`objectid(cfg)` kept a dead configuration's device tables (8.6 GB in Float64
at lmax=1023) alive until LRU eviction, which never came for a quiet device.
"""
mutable struct ScalarTableCache
    entries::Dict{Tuple{Any,UInt,DataType},ScalarTableCacheEntry}
    tick::UInt64
    max_per_device::Int
    lock::ReentrantLock
end

function ScalarTableCache(max_per_device::Integer=8)
    max_per_device > 0 || throw(ArgumentError("max_per_device must be positive"))
    return ScalarTableCache(
        Dict{Tuple{Any,UInt,DataType},ScalarTableCacheEntry}(),
        0, Int(max_per_device), ReentrantLock(),
    )
end

"""Drop entries whose weakly held owner has been reclaimed (lock held)."""
function _drop_dead_owners!(cache::ScalarTableCache)
    for key in collect(keys(cache.entries))
        owner = cache.entries[key].owner
        owner !== nothing && owner.value === nothing && delete!(cache.entries, key)
    end
    return nothing
end

"""Whether `entry` was built for `owner` (an object id can be reused)."""
@inline _owned_by(entry::ScalarTableCacheEntry, owner) =
    owner === nothing || entry.owner === nothing || entry.owner.value === owner

function scalar_cache_lookup(cache::ScalarTableCache, device, identity::UInt,
                             precision::DataType, signature::UInt; owner=nothing)
    key = (device, identity, precision)
    return lock(cache.lock) do
        _drop_dead_owners!(cache)
        entry = get(cache.entries, key, nothing)
        entry === nothing && return nothing
        if entry.signature != signature || !_owned_by(entry, owner)
            delete!(cache.entries, key)
            return nothing
        end
        cache.tick += 1
        cache.entries[key] = ScalarTableCacheEntry(
            signature, cache.tick, entry.value, entry.owner,
        )
        return entry.value
    end
end

function scalar_cache_insert!(cache::ScalarTableCache, device, identity::UInt,
                              precision::DataType, signature::UInt, value;
                              owner=nothing)
    key = (device, identity, precision)
    return lock(cache.lock) do
        _drop_dead_owners!(cache)
        existing = get(cache.entries, key, nothing)
        if existing !== nothing && existing.signature == signature &&
           _owned_by(existing, owner)
            cache.tick += 1
            cache.entries[key] = ScalarTableCacheEntry(
                signature, cache.tick, existing.value, existing.owner,
            )
            return existing.value
        end

        # Replacement for the same config identity does not consume capacity.
        existing === nothing || delete!(cache.entries, key)
        device_keys = Tuple{Any,UInt,DataType}[
            candidate for candidate in keys(cache.entries) if candidate[1] == device
        ]
        if length(device_keys) >= cache.max_per_device
            oldest = argmin(candidate -> cache.entries[candidate].tick, device_keys)
            delete!(cache.entries, oldest)
        end
        cache.tick += 1
        cache.entries[key] = ScalarTableCacheEntry(
            signature, cache.tick, value,
            owner === nothing ? nothing : WeakRef(owner),
        )
        return value
    end
end

"""
    scalar_cache_publish!(complete, cache, device, identity, precision, signature, value; owner=nothing)

Wait for an asynchronous immutable-table build to complete before making the
value visible to cache readers. `complete` deliberately runs before
`scalar_cache_insert!`, outside the cache lock. The insertion retains the
cache's double-checked behavior when concurrent builders race for one key.
"""
function scalar_cache_publish!(complete, cache::ScalarTableCache, device,
                               identity::UInt, precision::DataType,
                               signature::UInt, value; owner=nothing)
    complete()
    return scalar_cache_insert!(
        cache, device, identity, precision, signature, value; owner,
    )
end

function scalar_cache_clear!(cache::ScalarTableCache; device=nothing)
    lock(cache.lock) do
        if device === nothing
            empty!(cache.entries)
        else
            for key in collect(keys(cache.entries))
                key[1] == device && delete!(cache.entries, key)
            end
        end
    end
    return nothing
end

function scalar_cache_size(cache::ScalarTableCache; device=nothing)
    return lock(cache.lock) do
        _drop_dead_owners!(cache)
        device === nothing && return length(cache.entries)
        return count(key -> key[1] == device, keys(cache.entries))
    end
end

"""One weakly-owned, independently locked GPU transform workspace."""
mutable struct ScalarWorkspaceCacheEntry
    owner::WeakRef
    signature::UInt
    tick::UInt64
    value::Any
    lock::ReentrantLock
end

"""
Bounded device workspace cache used by in-place scalar and batch transforms.

Keys contain only `objectid(owner)`, never the owner itself. The accompanying
`WeakRef` prevents a cached vendor buffer or FFT plan from keeping an SHTPlan
or SHTConfig alive. Each entry has its own lock, so unrelated plans may execute
concurrently while reuse of the same mutable FFT workspace is serialized.
"""
mutable struct ScalarWorkspaceCache
    entries::Dict{Tuple{Any,UInt,DataType,Symbol,Tuple},ScalarWorkspaceCacheEntry}
    tick::UInt64
    max_per_device::Int
    lock::ReentrantLock
end

function ScalarWorkspaceCache(max_per_device::Integer=8)
    max_per_device > 0 || throw(ArgumentError("max_per_device must be positive"))
    return ScalarWorkspaceCache(
        Dict{Tuple{Any,UInt,DataType,Symbol,Tuple},ScalarWorkspaceCacheEntry}(),
        0, Int(max_per_device), ReentrantLock(),
    )
end

function _workspace_entry!(builder, cache::ScalarWorkspaceCache, device,
                           owner, precision::DataType, kind::Symbol,
                           shape::Tuple, signature::UInt)
    key = (device, objectid(owner), precision, kind, shape)
    return lock(cache.lock) do
        # Drop dead weak owners eagerly; this also protects against object-id
        # reuse selecting buffers belonging to a reclaimed plan/config.
        for candidate in collect(keys(cache.entries))
            cache.entries[candidate].owner.value === nothing &&
                delete!(cache.entries, candidate)
        end
        entry = get(cache.entries, key, nothing)
        if entry !== nothing && entry.owner.value === owner &&
           entry.signature == signature
            cache.tick += 1
            entry.tick = cache.tick
            return entry
        end
        entry === nothing || delete!(cache.entries, key)
        device_keys = [
            candidate for candidate in keys(cache.entries)
            if candidate[1] == device
        ]
        if length(device_keys) >= cache.max_per_device
            oldest = argmin(candidate -> cache.entries[candidate].tick, device_keys)
            delete!(cache.entries, oldest)
        end
        value = builder()
        cache.tick += 1
        built = ScalarWorkspaceCacheEntry(
            WeakRef(owner), signature, cache.tick, value, ReentrantLock(),
        )
        cache.entries[key] = built
        return built
    end
end

"""Run `f(workspace)` while holding the selected workspace's per-entry lock."""
function scalar_workspace_use!(f, builder, cache::ScalarWorkspaceCache,
                               device, owner, precision::DataType,
                               kind::Symbol, shape::Tuple, signature::UInt)
    entry = _workspace_entry!(
        builder, cache, device, owner, precision, kind, shape, signature,
    )
    return lock(entry.lock) do
        f(entry.value)
    end
end

function scalar_workspace_clear!(cache::ScalarWorkspaceCache; device=nothing)
    lock(cache.lock) do
        if device === nothing
            empty!(cache.entries)
        else
            for key in collect(keys(cache.entries))
                key[1] == device && delete!(cache.entries, key)
            end
        end
    end
    return nothing
end

function scalar_workspace_size(cache::ScalarWorkspaceCache; device=nothing)
    return lock(cache.lock) do
        device === nothing && return length(cache.entries)
        return count(key -> key[1] == device, keys(cache.entries))
    end
end

"""
Fingerprint every host-owned configuration value consumed by a scalar GPU
transform. `SHTConfig` is mutable for compatibility, so object identity alone
is not a valid cache key: changing pole order, quadrature, or a convention must
select fresh device tables.
"""
function scalar_config_signature(cfg::SHTnsKit.SHTConfig)
    grid_hash = _hash_values(cfg.x, _hash_values(cfg.w, hash(length(cfg.x))))
    return hash((
        objectid(cfg), cfg.lmax, cfg.mmax, cfg.mres, cfg.nlat, cfg.nlon,
        cfg.grid_type, cfg.cphi, cfg.south_pole_first,
        cfg.norm, cfg.real_norm, cfg.cs_phase, grid_hash,
    ))
end

"""
Hash every element of `values` into `h`. Converting an array to a `Tuple` first
(as the signatures used to) allocates and specializes on its length, and
`hash(::AbstractArray)` samples only a few elements of a large array, which
would miss most in-place mutations.
"""
function _hash_values(values, h::UInt)
    for value in values
        h = hash(value, h)
    end
    return h
end

"""
Fingerprint every mutable input consumed only by vector derivative tables.

Scalar transforms do not depend on `Nlm`; keeping this separate prevents an
`Nlm` mutation from rebuilding scalar tables while still invalidating the
pole-sensitive vector cache entry for the same configuration identity. The
derivative tables read `Nlm` only for the exact-pole m = 1 limits, so the
signature covers that column; hashing all of `Nlm` on every call cost 110 ms
and 34 MB at lmax=1023.
"""
function vector_config_signature(cfg::SHTnsKit.SHTConfig)
    column = cfg.mmax >= 1 ? view(cfg.Nlm, :, 2) : view(cfg.Nlm, 1:0, 1)
    return _hash_values(column, scalar_config_signature(cfg))
end

"""
    scalar_host_tables(cfg, T) -> (nodes, weights, scales, sint)

Host setup copied once into a vendor cache: the Float64 latitude nodes the
Legendre tables are built from, and the weights, convention scales and
`sin θ` in the device precision `T`.

The nodes stay Float64 whatever `T` is. Rounding them to Float32 moved every
P̄ₗᵐ by O(l²·ε) before a Float32 recurrence compounded the error, so Float32
GPU transforms lost about three digits by lmax=255. The table kernels now run
the recurrence in the nodes' precision and round each stored entry once. The
vector kernels likewise take `sin θ` computed from the exact node rather than
`sqrt(1 - x²)` of a rounded one, which lost accuracy next to the poles.
"""
function scalar_host_tables(cfg::SHTnsKit.SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    nodes = Float64.(cfg.x)
    sint = T.(sqrt.(max.(0.0, 1 .- nodes .^ 2)))
    return nodes, T.(cfg.w), coefficient_scales(cfg, T), sint
end

"""
    coefficient_scales(cfg, T) -> Matrix{T}

Factors converting configured coefficients to the canonical normalization,
indexed `[l + 1, m + 1]`. Point evaluation needs only these, not the
`nlat × (lmax+1) × (mmax+1)` Legendre table built alongside them.
"""
function coefficient_scales(cfg::SHTnsKit.SHTConfig, ::Type{T}) where {T<:AbstractFloat}
    scales = Matrix{T}(undef, cfg.lmax + 1, cfg.mmax + 1)
    fill!(scales, one(T))
    for m in 0:cfg.mmax, l in m:cfg.lmax
        scales[l + 1, m + 1] = T(SHTnsKit.coefficient_scale_to_canonical(cfg, l, m))
    end
    return scales
end

"""
    vector_host_tables(cfg, T) -> (nodes, weights, scales, Nlm, sint)

`scalar_host_tables` plus the Float64 normalization `Nlm` that the vector
derivative tables are built from.
"""
function vector_host_tables(cfg::SHTnsKit.SHTConfig,
                            ::Type{T}) where {T<:AbstractFloat}
    nodes, weights, scales, sint = scalar_host_tables(cfg, T)
    return nodes, weights, scales, Float64.(cfg.Nlm), sint
end

# Preserve diagonal seeds that are smaller than the device format can store.
# The recurrence uses a shared power-of-two exponent for its two mantissas;
# only table writes restore the physical magnitude. Int32 selects the native
# CUDA/ROCm ldexp intrinsic without widening Float32 arithmetic.
@inline function _rescale_legendre_pair(previous2::T, previous1::T,
                                         exponent::Int32) where {T}
    magnitude = max(abs(previous2), abs(previous1))
    if magnitude > T(0x1p32)
        return previous2 * T(0x1p-64), previous1 * T(0x1p-64), exponent + Int32(64)
    elseif !iszero(magnitude) && magnitude < T(0x1p-32)
        return previous2 * T(0x1p64), previous1 * T(0x1p64), exponent - Int32(64)
    end
    return previous2, previous1, exponent
end

@inline function _legendre_table_row!(Plm, xi::T, i, m, lmax) where {T}
    sint = sqrt(max(zero(T), one(T) - xi * xi))
    pmm = inv(sqrt(T(4) * T(pi)))
    exponent = Int32(0)
    @inbounds for k in 1:m
        tk = T(k)
        pmm = -sqrt((T(2) * tk + one(T)) / (T(2) * tk)) * sint * pmm
        if !iszero(pmm) && abs(pmm) < T(0x1p-32)
            pmm *= T(0x1p64)
            exponent -= Int32(64)
        end
    end
    @inbounds Plm[i, m + 1, m + 1] = ldexp(pmm, exponent)
    if m < lmax
        pm1m = sqrt(T(2m + 3)) * xi * pmm
        @inbounds Plm[i, m + 2, m + 1] = ldexp(pm1m, exponent)
        previous2, previous1, exponent = _rescale_legendre_pair(pmm, pm1m, exponent)
        @inbounds for l in (m + 2):lmax
            tl = T(l)
            tm = T(m)
            a = sqrt(((T(2) * tl - one(T)) * (T(2) * tl + one(T))) /
                     ((tl - tm) * (tl + tm)))
            b = sqrt(((T(2) * tl + one(T)) * (tl - one(T) - tm) *
                      (tl - one(T) + tm)) /
                     ((T(2) * tl - T(3)) * (tl - tm) * (tl + tm)))
            value = a * xi * previous1 - b * previous2
            Plm[i, l + 1, m + 1] = ldexp(value, exponent)
            previous2, previous1, exponent =
                _rescale_legendre_pair(previous1, value, exponent)
        end
    end
    return nothing
end

"""Build orthonormal, Condon--Shortley associated Legendre values on device."""
@kernel function legendre_table_kernel!(Plm, x, lmax, mmax)
    i, m_idx = @index(Global, NTuple)
    if i <= length(x) && m_idx <= mmax + 1
        _legendre_table_row!(Plm, x[i], i, m_idx - 1, lmax)
    end
end

"""Store P̄ₗᵐ, dP̄ₗᵐ/dθ and P̄ₗᵐ/sinθ from a recurrence pair sharing `exponent`."""
@inline function _store_vector_entry!(Plm, dtheta, over_sin, i, l, m,
                                      xi::T, s::T, current::T, previous::T,
                                      exponent::Int32) where {T}
    beta = l == m ? zero(T) : sqrt(T((2l + 1) * (l * l - m * m)) / T(2l - 1))
    @inbounds begin
        Plm[i, l + 1, m + 1] = ldexp(current, exponent)
        dtheta[i, l + 1, m + 1] =
            ldexp((T(l) * xi * current - beta * previous) / s, exponent)
        over_sin[i, l + 1, m + 1] = ldexp(current / s, exponent)
    end
    return nothing
end

"""
`_legendre_table_row!` plus the dP̄/dθ and P̄/sinθ entries of the same row
(`s = sin θ > 0`), formed from the recurrence's own values. Reading the stored
P̄ back instead would difference entries already rounded to the table's
precision.
"""
@inline function _legendre_vector_row!(Plm, dtheta, over_sin, xi::T, s::T,
                                       i, m, lmax) where {T}
    pmm = inv(sqrt(T(4) * T(pi)))
    exponent = Int32(0)
    @inbounds for k in 1:m
        tk = T(k)
        pmm = -sqrt((T(2) * tk + one(T)) / (T(2) * tk)) * s * pmm
        if !iszero(pmm) && abs(pmm) < T(0x1p-32)
            pmm *= T(0x1p64)
            exponent -= Int32(64)
        end
    end
    _store_vector_entry!(Plm, dtheta, over_sin, i, m, m, xi, s, pmm, zero(T),
                         exponent)
    m < lmax || return nothing
    pm1m = sqrt(T(2m + 3)) * xi * pmm
    _store_vector_entry!(Plm, dtheta, over_sin, i, m + 1, m, xi, s, pm1m, pmm,
                         exponent)
    previous2, previous1, exponent = _rescale_legendre_pair(pmm, pm1m, exponent)
    @inbounds for l in (m + 2):lmax
        tl = T(l)
        tm = T(m)
        a = sqrt(((T(2) * tl - one(T)) * (T(2) * tl + one(T))) /
                 ((tl - tm) * (tl + tm)))
        b = sqrt(((T(2) * tl + one(T)) * (tl - one(T) - tm) *
                  (tl - one(T) + tm)) /
                 ((T(2) * tl - T(3)) * (tl - tm) * (tl + tm)))
        value = a * xi * previous1 - b * previous2
        _store_vector_entry!(Plm, dtheta, over_sin, i, l, m, xi, s, value,
                             previous1, exponent)
        previous2, previous1, exponent =
            _rescale_legendre_pair(previous1, value, exponent)
    end
    return nothing
end

"""
Build orthonormal P, dP/dtheta, and P/sin(theta) tables. The recurrence runs in
the precision of the nodes `x` (Float64 from `scalar_host_tables`) and each
entry is rounded once when stored in the tables' precision. The exact-pole
branch evaluates the finite m=1 limits analytically; no kernel ever forms a
singular quotient and masks it afterwards.
"""
@kernel function vector_derivative_table_kernel!(Plm, dtheta, over_sin,
                                                  x, Nlm, lmax, mmax)
    i, m_idx = @index(Global, NTuple)
    if i <= length(x) && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        T = typeof(xi)
        s = sqrt(max(zero(T), one(T) - xi * xi))
        if iszero(s)
            _legendre_table_row!(Plm, xi, i, m, lmax)
        else
            _legendre_vector_row!(Plm, dtheta, over_sin, xi, s, i, m, lmax)
        end

        @inbounds for l in 0:lmax
            if l < m
                Plm[i, l + 1, m_idx] = zero(T)
                dtheta[i, l + 1, m_idx] = zero(T)
                over_sin[i, l + 1, m_idx] = zero(T)
            elseif iszero(s)
                if m == 1
                    half_ll1 = T(l * (l + 1)) / T(2)
                    N = T(Nlm[l + 1, m_idx])
                    north = xi > zero(T)
                    dsign = north ? -one(T) : (isodd(l) ? one(T) : -one(T))
                    psign = north ? -one(T) : (iseven(l) ? one(T) : -one(T))
                    dtheta[i, l + 1, m_idx] = dsign * N * half_ll1
                    over_sin[i, l + 1, m_idx] = psign * N * half_ll1
                else
                    dtheta[i, l + 1, m_idx] = zero(T)
                    over_sin[i, l + 1, m_idx] = zero(T)
                end
            end
        end
    end
end

"""Latitude contraction for the two tangential Fourier components."""
@kernel function vector_analysis_kernel!(Sout, Tout, Ftheta, Fphi,
                                          dtheta, over_sin, weights, scales,
                                          sint, cphi, lcap, mmax, mres,
                                          robert_form)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lcap + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l <= lcap && l >= max(1, m) && m % mres == 0
            Svalue = zero(eltype(Sout))
            Tvalue = zero(eltype(Tout))
            @inbounds for i in 1:length(weights)
                s = sint[i]
                Ft = Ftheta[i, m_idx]
                Fp = Fphi[i, m_idx]
                if robert_form && !iszero(s)
                    Ft /= s
                    Fp /= s
                end
                d = dtheta[i, l_idx, m_idx]
                term = complex(zero(d), typeof(d)(m) * over_sin[i, l_idx, m_idx])
                factor = weights[i] * cphi / typeof(d)(l * (l + 1))
                Svalue += factor * (Ft * d + conj(term) * Fp)
                Tvalue += factor * (-conj(term) * Ft + d * Fp)
            end
            scale = scales[l_idx, m_idx]
            Sout[l_idx, m_idx] = Svalue / scale
            Tout[l_idx, m_idx] = Tvalue / scale
        else
            Sout[l_idx, m_idx] = zero(eltype(Sout))
            Tout[l_idx, m_idx] = zero(eltype(Tout))
        end
    end
end

"""Vector Legendre synthesis into vendor-IFFT Fourier bins."""
@kernel function vector_synthesis_kernel!(Ftheta, Fphi, Sin, Tin,
                                           dtheta, over_sin, scales, sint,
                                           inv_scale, nlon, lmax, mmax,
                                           mres, real_output, robert_form)
    i, m_idx = @index(Global, NTuple)
    if i <= size(Ftheta, 1) && m_idx <= mmax + 1
        m = m_idx - 1
        if m % mres == 0
            gt = zero(eltype(Ftheta))
            gp = zero(eltype(Fphi))
            @inbounds for l in max(1, m):lmax
                d = dtheta[i, l + 1, m_idx]
                p = over_sin[i, l + 1, m_idx]
                scale = scales[l + 1, m_idx]
                S = scale * Sin[l + 1, m_idx]
                Tv = scale * Tin[l + 1, m_idx]
                term = complex(zero(d), typeof(d)(m) * p)
                gt += d * S - term * Tv
                gp += term * S + d * Tv
            end
            if robert_form
                s = sint[i]
                gt *= s
                gp *= s
            end
            bt = inv_scale * gt
            bp = inv_scale * gp
            Ftheta[i, m_idx] = bt
            Fphi[i, m_idx] = bp
            if real_output && m > 0
                negative_idx = nlon - m + 1
                if negative_idx != m_idx
                    Ftheta[i, negative_idx] = conj(bt)
                    Fphi[i, negative_idx] = conj(bp)
                end
            end
        end
    end
end

"""Analyze one stored vector order without expanding to a dense spectrum."""
@kernel function vector_mode_analysis_kernel!(Sout, Tout, Ftheta, Fphi,
                                               dtheta, over_sin, weights,
                                               scales, sint, cphi, physical_m,
                                               lcap, robert_form)
    q_idx = @index(Global)
    l = physical_m + q_idx - 1
    if l <= lcap
        if l < max(1, physical_m)
            # Vector spherical harmonics have no l=0 coefficient.  Keep its
            # logical axisymmetric storage slot exact without evaluating the
            # undefined 1/(l(l+1)) analysis factor.
            Sout[q_idx] = zero(eltype(Sout))
            Tout[q_idx] = zero(eltype(Tout))
        else
            Svalue = zero(eltype(Sout))
            Tvalue = zero(eltype(Tout))
            @inbounds for i in 1:length(weights)
                s = sint[i]
                Ft = Ftheta[i]
                Fp = Fphi[i]
                if robert_form && !iszero(s)
                    Ft /= s
                    Fp /= s
                end
                d = dtheta[i, l + 1, physical_m + 1]
                term = complex(zero(d), typeof(d)(physical_m) *
                                over_sin[i, l + 1, physical_m + 1])
                factor = weights[i] * cphi / typeof(d)(l * (l + 1))
                Svalue += factor * (Ft * d + conj(term) * Fp)
                Tvalue += factor * (-conj(term) * Ft + d * Fp)
            end
            scale = scales[l + 1, physical_m + 1]
            Sout[q_idx] = Svalue / scale
            Tout[q_idx] = Tvalue / scale
        end
    end
end

"""Synthesize one stored vector order directly into latitude vectors."""
@kernel function vector_mode_synthesis_kernel!(Vtheta, Vphi, Sin, Tin,
                                                dtheta, over_sin, scales, sint,
                                                inv_scale, physical_m, lcap,
                                                robert_form)
    i = @index(Global)
    if i <= length(Vtheta)
        gt = zero(eltype(Vtheta))
        gp = zero(eltype(Vphi))
        @inbounds for l in max(1, physical_m):lcap
            scale = scales[l + 1, physical_m + 1]
            S = scale * Sin[l - physical_m + 1]
            Tvalue = scale * Tin[l - physical_m + 1]
            d = dtheta[i, l + 1, physical_m + 1]
            term = complex(zero(d), typeof(d)(physical_m) *
                            over_sin[i, l + 1, physical_m + 1])
            gt += d * S - term * Tvalue
            gp += term * S + d * Tvalue
        end
        if robert_form
            s = sint[i]
            gt *= s
            gp *= s
        end
        Vtheta[i] = inv_scale * gt
        Vphi[i] = inv_scale * gp
    end
end

"""Latitude contraction for vector fields in a trailing batch dimension."""
@kernel function vector_batch_analysis_kernel!(Sout, Tout, Ftheta, Fphi,
                                                dtheta, over_sin, weights,
                                                scales, sint, cphi, lmax, mmax,
                                                mres, robert_form)
    l_idx, m_idx, batch_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1 &&
       batch_idx <= size(Sout, 3)
        l = l_idx - 1
        m = m_idx - 1
        if l >= max(1, m) && m % mres == 0
            Svalue = zero(eltype(Sout))
            Tvalue = zero(eltype(Tout))
            @inbounds for i in 1:length(weights)
                s = sint[i]
                Ft = Ftheta[i, m_idx, batch_idx]
                Fp = Fphi[i, m_idx, batch_idx]
                if robert_form && !iszero(s)
                    Ft /= s
                    Fp /= s
                end
                d = dtheta[i, l_idx, m_idx]
                term = complex(zero(d), typeof(d)(m) * over_sin[i, l_idx, m_idx])
                factor = weights[i] * cphi / typeof(d)(l * (l + 1))
                Svalue += factor * (Ft * d + conj(term) * Fp)
                Tvalue += factor * (-conj(term) * Ft + d * Fp)
            end
            scale = scales[l_idx, m_idx]
            Sout[l_idx, m_idx, batch_idx] = Svalue / scale
            Tout[l_idx, m_idx, batch_idx] = Tvalue / scale
        else
            Sout[l_idx, m_idx, batch_idx] = zero(eltype(Sout))
            Tout[l_idx, m_idx, batch_idx] = zero(eltype(Tout))
        end
    end
end

"""Vector synthesis with independent fields in the trailing batch dimension."""
@kernel function vector_batch_synthesis_kernel!(Ftheta, Fphi, Sin, Tin,
                                                 dtheta, over_sin, scales, sint,
                                                 inv_scale, nlon, lmax, mmax,
                                                 mres, real_output, robert_form)
    i, m_idx, batch_idx = @index(Global, NTuple)
    if i <= size(Ftheta, 1) && m_idx <= mmax + 1 &&
       batch_idx <= size(Ftheta, 3)
        m = m_idx - 1
        if m % mres == 0
            gt = zero(eltype(Ftheta))
            gp = zero(eltype(Fphi))
            @inbounds for l in max(1, m):lmax
                scale = scales[l + 1, m_idx]
                S = scale * Sin[l + 1, m_idx, batch_idx]
                Tvalue = scale * Tin[l + 1, m_idx, batch_idx]
                d = dtheta[i, l + 1, m_idx]
                term = complex(zero(d), typeof(d)(m) * over_sin[i, l + 1, m_idx])
                gt += d * S - term * Tvalue
                gp += term * S + d * Tvalue
            end
            if robert_form
                s = sint[i]
                gt *= s
                gp *= s
            end
            bt = inv_scale * gt
            bp = inv_scale * gp
            Ftheta[i, m_idx, batch_idx] = bt
            Fphi[i, m_idx, batch_idx] = bp
            if real_output && m > 0
                negative_idx = nlon - m + 1
                if negative_idx != m_idx
                    Ftheta[i, negative_idx, batch_idx] = conj(bt)
                    Fphi[i, negative_idx, batch_idx] = conj(bp)
                end
            end
        end
    end
end

"""Apply or invert the tangential `-l(l+1)` spectral multiplier."""
@kernel function vector_diagonal_kernel!(output, input, lmax, mmax, mres,
                                         inverse)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l >= max(1, m) && m % mres == 0
            ll1 = l * (l + 1)
            output[l_idx, m_idx] = inverse ?
                -(input[l_idx, m_idx] / ll1) :
                -ll1 * input[l_idx, m_idx]
        else
            output[l_idx, m_idx] = zero(eltype(output))
        end
    end
end

"""Latitude integration after the vendor FFT; output is canonical."""
@kernel function scalar_analysis_kernel!(canonical, fourier, Plm, weights,
                                          cphi, lmax, mmax, mres, lcap)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l <= lcap && l >= m && m % mres == 0
            value = zero(eltype(canonical))
            @inbounds for i in 1:length(weights)
                value += weights[i] * Plm[i, l_idx, m_idx] * fourier[i, m_idx]
            end
            canonical[l_idx, m_idx] = cphi * value
        else
            canonical[l_idx, m_idx] = zero(eltype(canonical))
        end
    end
end

"""Convert dense coefficients entirely on device at the public boundary."""
@kernel function coefficient_conversion_kernel!(output, input, scales,
                                                  lmax, mmax, to_internal)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l >= m
            scale = scales[l_idx, m_idx]
            output[l_idx, m_idx] = to_internal ?
                scale * input[l_idx, m_idx] : input[l_idx, m_idx] / scale
        else
            output[l_idx, m_idx] = zero(eltype(output))
        end
    end
end

"""Legendre synthesis into vendor-IFFT bins from canonical coefficients."""
@kernel function scalar_synthesis_kernel!(fourier, canonical, Plm, inv_scale,
                                           nlon, lmax, mmax, mres, real_output)
    i, m_idx = @index(Global, NTuple)
    if i <= size(fourier, 1) && m_idx <= mmax + 1
        m = m_idx - 1
        if m % mres == 0
            value = zero(eltype(fourier))
            @inbounds for l in m:lmax
                value += Plm[i, l + 1, m_idx] * canonical[l + 1, m_idx]
            end
            bin = inv_scale * value
            fourier[i, m_idx] = bin
            if real_output && m > 0
                negative_idx = nlon - m + 1
                if negative_idx != m_idx
                    fourier[i, negative_idx] = conj(bin)
                end
            end
        end
    end
end

"""MPI-pencil scalar analysis for an owned contiguous band of Fourier orders."""
@kernel function distributed_scalar_analysis_kernel!(output, fourier, Plm,
                                                       weights, scales, cphi,
                                                       first_m, lmax, mmax,
                                                       mres, lcap)
    l_idx, local_m_idx, batch_idx = @index(Global, NTuple)
    m = first_m + local_m_idx - 1
    if l_idx <= size(output, 1) && local_m_idx <= size(output, 2) &&
       batch_idx <= size(output, 3)
        l = l_idx - 1
        if m <= mmax && l <= lcap && l >= m && m % mres == 0
            value = zero(eltype(output))
            @inbounds for i in 1:length(weights)
                value += weights[i] * Plm[i, l_idx, m + 1] *
                         fourier[i, local_m_idx, batch_idx]
            end
            output[l_idx, local_m_idx, batch_idx] =
                cphi * value / scales[l_idx, m + 1]
        else
            output[l_idx, local_m_idx, batch_idx] = zero(eltype(output))
        end
    end
end

"""MPI-pencil scalar synthesis for an owned contiguous band of Fourier orders."""
@kernel function distributed_scalar_synthesis_kernel!(fourier, input, Plm,
                                                        scales, inv_scale, first_m,
                                                        lmax, mmax, mres)
    i, local_m_idx, batch_idx = @index(Global, NTuple)
    m = first_m + local_m_idx - 1
    if i <= size(fourier, 1) && local_m_idx <= size(fourier, 2) &&
       batch_idx <= size(fourier, 3)
        value = zero(eltype(fourier))
        if m <= mmax && m % mres == 0
            @inbounds for l in m:lmax
                value += scales[l + 1, m + 1] * Plm[i, l + 1, m + 1] *
                         input[l + 1, local_m_idx, batch_idx]
            end
        end
        fourier[i, local_m_idx, batch_idx] = inv_scale * value
    end
end

"""MPI-pencil vector analysis for an owned contiguous Fourier-order band."""
@kernel function distributed_vector_analysis_kernel!(Sout, Tout, Ftheta,
                                                       Fphi, dtheta, over_sin,
                                                       weights, scales, sint, cphi,
                                                       first_m, lmax, mmax,
                                                       mres, robert_form)
    l_idx, local_m_idx, batch_idx = @index(Global, NTuple)
    m = first_m + local_m_idx - 1
    if l_idx <= size(Sout, 1) && local_m_idx <= size(Sout, 2) &&
       batch_idx <= size(Sout, 3)
        l = l_idx - 1
        if m <= mmax && l >= max(1, m) && m % mres == 0
            Svalue = zero(eltype(Sout))
            Tvalue = zero(eltype(Tout))
            @inbounds for i in 1:length(weights)
                s = sint[i]
                Ft = Ftheta[i, local_m_idx, batch_idx]
                Fp = Fphi[i, local_m_idx, batch_idx]
                if robert_form && !iszero(s)
                    Ft /= s
                    Fp /= s
                end
                d = dtheta[i, l_idx, m + 1]
                term = complex(zero(d), typeof(d)(m) * over_sin[i, l_idx, m + 1])
                factor = weights[i] * cphi / typeof(d)(l * (l + 1))
                Svalue += factor * (Ft * d + conj(term) * Fp)
                Tvalue += factor * (-conj(term) * Ft + d * Fp)
            end
            scale = scales[l_idx, m + 1]
            Sout[l_idx, local_m_idx, batch_idx] = Svalue / scale
            Tout[l_idx, local_m_idx, batch_idx] = Tvalue / scale
        else
            Sout[l_idx, local_m_idx, batch_idx] = zero(eltype(Sout))
            Tout[l_idx, local_m_idx, batch_idx] = zero(eltype(Tout))
        end
    end
end

"""MPI-pencil vector synthesis for an owned contiguous Fourier-order band."""
@kernel function distributed_vector_synthesis_kernel!(Ftheta, Fphi, Sin, Tin,
                                                        dtheta, over_sin, scales, sint,
                                                        inv_scale, first_m,
                                                        lmax, mmax, mres,
                                                        robert_form)
    i, local_m_idx, batch_idx = @index(Global, NTuple)
    m = first_m + local_m_idx - 1
    if i <= size(Ftheta, 1) && local_m_idx <= size(Ftheta, 2) &&
       batch_idx <= size(Ftheta, 3)
        gt = zero(eltype(Ftheta))
        gp = zero(eltype(Fphi))
        if m <= mmax && m % mres == 0
            @inbounds for l in max(1, m):lmax
                scale = scales[l + 1, m + 1]
                S = scale * Sin[l + 1, local_m_idx, batch_idx]
                Tvalue = scale * Tin[l + 1, local_m_idx, batch_idx]
                d = dtheta[i, l + 1, m + 1]
                term = complex(zero(d), typeof(d)(m) *
                               over_sin[i, l + 1, m + 1])
                gt += d * S - term * Tvalue
                gp += term * S + d * Tvalue
            end
            if robert_form
                s = sint[i]
                gt *= s
                gp *= s
            end
        end
        Ftheta[i, local_m_idx, batch_idx] = inv_scale * gt
        Fphi[i, local_m_idx, batch_idx] = inv_scale * gp
    end
end

"""Pack a dense non-negative-order spectrum, optionally truncating in degree."""
@kernel function real_pack_kernel!(packed, dense, lmax, mmax, mres, lcap)
    l_idx, im_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && im_idx <= mmax ÷ mres + 1
        l = l_idx - 1
        im = im_idx - 1
        m = im * mres
        if m <= mmax && l >= m
            base = (im * (2lmax + 2 - (im + 1) * mres)) >>> 1
            packed[base + l + 1] = l <= lcap ? dense[l_idx, m + 1] : zero(eltype(packed))
        end
    end
end

"""Expand SHTns LM storage to a dense matrix with an explicit degree cap."""
@kernel function real_unpack_kernel!(dense, packed, lmax, mmax, mres, lcap)
    l_idx, m_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1
        l = l_idx - 1
        m = m_idx - 1
        if l <= lcap && l >= m && m % mres == 0
            im = m ÷ mres
            base = (im * (2lmax + 2 - (im + 1) * mres)) >>> 1
            dense[l_idx, m_idx] = packed[base + l + 1]
        else
            dense[l_idx, m_idx] = zero(eltype(dense))
        end
    end
end

"""Analyze one physical Fourier order over an explicit degree interval."""
@kernel function mode_analysis_kernel!(output, mode, Plm, weights, scale,
                                       physical_m, lcap)
    q_idx = @index(Global)
    l = physical_m + q_idx - 1
    if l <= lcap
        value = zero(eltype(output))
        @inbounds for i in 1:length(weights)
            value += weights[i] * Plm[i, l + 1, physical_m + 1] * mode[i]
        end
        output[q_idx] = scale * value
    end
end

"""Synthesize one physical Fourier order over an explicit degree interval."""
@kernel function mode_synthesis_kernel!(mode, coefficients, Plm, scale,
                                        physical_m, lcap)
    i = @index(Global)
    if i <= size(Plm, 1)
        value = zero(eltype(mode))
        @inbounds for l in physical_m:lcap
            value += Plm[i, l + 1, physical_m + 1] *
                     coefficients[l - physical_m + 1]
        end
        mode[i] = scale * value
    end
end

"""Latitude integration for independent scalar fields in the trailing axis."""
@kernel function scalar_batch_analysis_kernel!(canonical, fourier, Plm, weights,
                                                cphi, lmax, mmax, mres)
    l_idx, m_idx, batch_idx = @index(Global, NTuple)
    if l_idx <= lmax + 1 && m_idx <= mmax + 1 &&
       batch_idx <= size(canonical, 3)
        l = l_idx - 1
        m = m_idx - 1
        if l >= m && m % mres == 0
            value = zero(eltype(canonical))
            @inbounds for i in 1:length(weights)
                value += weights[i] * Plm[i, l_idx, m_idx] *
                         fourier[i, m_idx, batch_idx]
            end
            canonical[l_idx, m_idx, batch_idx] = cphi * value
        else
            canonical[l_idx, m_idx, batch_idx] = zero(eltype(canonical))
        end
    end
end

"""Legendre synthesis for independent scalar fields in the trailing axis."""
@kernel function scalar_batch_synthesis_kernel!(fourier, canonical, Plm, inv_scale,
                                                 nlon, lmax, mmax, mres,
                                                 real_output)
    i, m_idx, batch_idx = @index(Global, NTuple)
    if i <= size(fourier, 1) && m_idx <= mmax + 1 &&
       batch_idx <= size(fourier, 3)
        m = m_idx - 1
        if m % mres == 0
            value = zero(eltype(fourier))
            @inbounds for l in m:lmax
                value += Plm[i, l + 1, m_idx] * canonical[l + 1, m_idx, batch_idx]
            end
            bin = inv_scale * value
            fourier[i, m_idx, batch_idx] = bin
            if real_output && m > 0
                negative_idx = nlon - m + 1
                if negative_idx != m_idx
                    fourier[i, negative_idx, batch_idx] = conj(bin)
                end
            end
        end
    end
end

@inline function _lm_cplx_device_index(l, m, mmax)
    return l <= mmax ? l * (l + 1) + m : mmax * (2l - mmax) + l + m
end

"""Analyze both Fourier signs directly into SHTns LM_cplx storage."""
@kernel function complex_packed_analysis_kernel!(packed, fourier, Plm, weights,
                                                 scales, cphi, nlon, lcap,
                                                 mmax, mcap)
    l_idx, signed_idx = @index(Global, NTuple)
    m = signed_idx - mcap - 1
    am = abs(m)
    l = l_idx - 1
    if l <= lcap && am <= mcap && l >= am
        column = m >= 0 ? m + 1 : nlon + m + 1
        value = zero(eltype(packed))
        @inbounds for i in 1:length(weights)
            value += weights[i] * Plm[i, l_idx, am + 1] * fourier[i, column]
        end
        packed[_lm_cplx_device_index(l, m, mmax) + 1] =
            cphi * value / scales[l_idx, am + 1]
    end
end

"""Synthesize both Fourier signs directly from SHTns LM_cplx storage."""
@kernel function complex_packed_synthesis_kernel!(fourier, packed, Plm, scales,
                                                  inv_scale, nlon, lcap,
                                                  mmax, mcap)
    i, signed_idx = @index(Global, NTuple)
    m = signed_idx - mcap - 1
    am = abs(m)
    if i <= size(fourier, 1) && am <= mcap
        value = zero(eltype(fourier))
        @inbounds for l in am:lcap
            coefficient = packed[_lm_cplx_device_index(l, m, mmax) + 1] *
                          scales[l + 1, am + 1]
            value += Plm[i, l + 1, am + 1] * coefficient
        end
        column = m >= 0 ? m + 1 : nlon + m + 1
        fourier[i, column] = inv_scale * value
    end
end

@inline function _real_packed_device_index(l, m, lmax, mres)
    im = m ÷ mres
    base = (im * (2lmax + 2 - (im + 1) * mres)) >>> 1
    return base + l
end

@inline function _rotation_lm_from_packed(k, lmax, mmax)
    cursor = 0
    for m in 0:mmax
        block = lmax - m + 1
        if k < cursor + block
            return m + (k - cursor), m
        end
        cursor += block
    end
    return 0, 0
end

@inline function _rotation_lm_from_cplx(k, lmax, mmax)
    for l in 0:lmax
        mm = min(l, mmax)
        first_index = _lm_cplx_device_index(l, -mm, mmax)
        count = 2mm + 1
        if first_index <= k < first_index + count
            return l, -mm + (k - first_index)
        end
    end
    return 0, 0
end

@inline function _rotation_block_value(values, offset, l, m, mp)
    n = 2l + 1
    return values[offset + (m + l) * n + (mp + l)]
end

"""Local diagonal Z rotation for packed real-field coefficients."""
@kernel function rotation_z_real_kernel!(output, input, angle, orders)
    k = @index(Global, Linear)
    if k <= length(input)
        output[k] = input[k] * cis(-typeof(angle)(orders[k]) * angle)
    end
end

"""Apply one general rotation to SHTns real-packed device coefficients."""
@kernel function rotation_real_kernel!(output, input, offsets, values,
                                       input_scales, output_scales,
                                       alpha, gamma, lmax, mmax)
    index = @index(Global, Linear)
    if index <= length(output)
        l, m = _rotation_lm_from_packed(index - 1, lmax, mmax)
        RT = typeof(alpha)
        offset = Int(offsets[l + 1])
        acc = zero(eltype(output))
        @inbounds for mp in -min(l, mmax):min(l, mmax)
            packed_index = _real_packed_device_index(l, abs(mp), lmax, 1) + 1
            value = mp < 0 ? conj(input[packed_index]) : input[packed_index]
            cplx_index = _lm_cplx_device_index(l, mp, mmax) + 1
            scale = input_scales[cplx_index]
            epsilon = mp < 0 && isodd(mp) ? -one(RT) : one(RT)
            value *= scale * epsilon * cis(-RT(mp) * gamma)
            acc += _rotation_block_value(values, offset, l, m, mp) * value
        end
        cplx_index = _lm_cplx_device_index(l, m, mmax) + 1
        output[index] = acc * cis(-RT(m) * alpha) * output_scales[cplx_index]
    end
end

"""Apply one general rotation to full LM_cplx device coefficients."""
@kernel function rotation_cplx_kernel!(output, input, offsets, values,
                                       input_scales, output_scales,
                                       alpha, gamma, lmax, mmax)
    index = @index(Global, Linear)
    if index <= length(output)
        l, m = _rotation_lm_from_cplx(index - 1, lmax, mmax)
        RT = typeof(alpha)
        offset = Int(offsets[l + 1])
        acc = zero(eltype(output))
        @inbounds for mp in -min(l, mmax):min(l, mmax)
            source_index = _lm_cplx_device_index(l, mp, mmax) + 1
            epsilon = mp < 0 && isodd(mp) ? -one(RT) : one(RT)
            value = input[source_index] * input_scales[source_index] * epsilon *
                    cis(-RT(mp) * gamma)
            acc += _rotation_block_value(values, offset, l, m, mp) * value
        end
        epsilon = m < 0 && isodd(m) ? -one(RT) : one(RT)
        output[index] = epsilon * acc * cis(-RT(m) * alpha) *
                        output_scales[index]
    end
end

"""
Evaluate a dense non-negative-m real spectrum at one or more longitudes.

Like the other local evaluators, it multiplies by `phi_scale`, the
`SHTnsKit._evaluator_phi_scale` factor (1 under `:dft`, 1/2π under `:quad`)
that makes a point value match the grid `synthesis` writes.
"""
@kernel function local_scalar_kernel!(output, coefficients, Plm, scales,
                                      phi0, phi_step, lmax, mmax, mres,
                                      lcap, mcap, phi_scale)
    j = @index(Global)
    if j <= length(output)
        phi = phi0 + (j - 1) * phi_step
        value = zero(eltype(output))
        @inbounds for m in 0:mcap
            if m % mres == 0
                radial = zero(eltype(coefficients))
                for l in m:lcap
                    radial += Plm[1, l + 1, m + 1] *
                              scales[l + 1, m + 1] * coefficients[l + 1, m + 1]
                end
                wave = radial * cis(m * phi)
                value += m == 0 ? real(wave) : 2real(wave)
            end
        end
        output[j] = phi_scale * value
    end
end

"""Evaluate SHTns LM_cplx storage at one or more longitudes."""
@kernel function local_complex_kernel!(output, coefficients, Plm, scales,
                                       phi0, phi_step, lmax, mmax, lcap,
                                       phi_scale)
    j = @index(Global)
    if j <= length(output)
        phi = phi0 + (j - 1) * phi_step
        value = zero(eltype(output))
        @inbounds for m in -mmax:mmax
            am = abs(m)
            radial = zero(eltype(output))
            for l in am:lcap
                index = _lm_cplx_device_index(l, m, mmax) + 1
                radial += Plm[1, l + 1, am + 1] *
                          scales[l + 1, am + 1] * coefficients[index]
            end
            value += radial * cis(m * phi)
        end
        output[j] = phi_scale * value
    end
end

"""
Evaluate packed real Q/S/T spectra at one or more longitudes. Boolean component
flags let the scalar-gradient path reuse this kernel without allocating zero
spectra on the device.
"""
@kernel function local_qst_kernel!(Vr, Vt, Vp, Q, S, Tlm,
                                   Plm, dtheta, over_sin, scales,
                                   phi0, phi_step, lmax, mmax, mres,
                                   lcap, mcap, has_q, has_s, has_t,
                                   robert_form, sinth, phi_scale)
    j = @index(Global)
    if j <= length(Vr)
        phi = phi0 + (j - 1) * phi_step
        vr = zero(eltype(Vr))
        vt = zero(eltype(Vt))
        vp = zero(eltype(Vp))
        imagunit = complex(zero(eltype(Vr)), one(eltype(Vr)))
        @inbounds for m in 0:mcap
            if m % mres == 0
                qmode = zero(eltype(Q))
                smode_t = zero(eltype(S))
                smode_p = zero(eltype(S))
                tmode_t = zero(eltype(Tlm))
                tmode_p = zero(eltype(Tlm))
                for l in m:lcap
                    index = _real_packed_device_index(l, m, lmax, mres) + 1
                    scale = scales[l + 1, m + 1]
                    has_q && (qmode += Plm[1, l + 1, m + 1] * scale * Q[index])
                    if has_s
                        coefficient = scale * S[index]
                        smode_t += dtheta[1, l + 1, m + 1] * coefficient
                        smode_p += imagunit * m * over_sin[1, l + 1, m + 1] * coefficient
                    end
                    if has_t
                        coefficient = scale * Tlm[index]
                        tmode_t -= imagunit * m * over_sin[1, l + 1, m + 1] * coefficient
                        tmode_p += dtheta[1, l + 1, m + 1] * coefficient
                    end
                end
                phase = cis(m * phi)
                if m == 0
                    vr += real(qmode)
                    vt += real(smode_t + tmode_t)
                    vp += real(smode_p + tmode_p)
                else
                    vr += 2real(qmode * phase)
                    vt += 2real((smode_t + tmode_t) * phase)
                    vp += 2real((smode_p + tmode_p) * phase)
                end
            end
        end
        Vr[j] = phi_scale * vr
        Vt[j] = phi_scale * (robert_form ? sinth * vt : vt)
        Vp[j] = phi_scale * (robert_form ? sinth * vp : vp)
    end
end

# Device-neutral kernels live here. Vendor extensions own array placement,
# FFT libraries, synchronization, device selection, and runtime inspection.
@kernel function operator_matrix_kernel!(mx, li, mi, down_ratios, up_ratios,
                                         lmax, derivative)
    k = @index(Global, Linear)
    if k <= length(li)
        l = li[k]
        m = mi[k]
        T = eltype(mx)
        down = zero(T)
        up = zero(T)
        if l > m
            down = sqrt(max(zero(T), T(l*l - m*m) /
                T((2l - 1) * (2l + 1)))) * down_ratios[k]
        end
        if l < lmax
            up = sqrt(max(zero(T), T((l + 1)^2 - m*m) /
                T((2l + 1) * (2l + 3)))) * up_ratios[k]
        end
        mx[2k - 1] = derivative ? -T(l + 1) * down : down
        mx[2k] = derivative ? T(l) * up : up
    end
end

@kernel function packed_operator_kernel!(output, input, mx, lower, upper)
    k = @index(Global, Linear)
    if k <= length(input)
        acc = zero(eltype(output))
        below = lower[k]
        above = upper[k]
        if below > 0
            acc += mx[2below] * input[below]
        end
        if above > 0
            acc += mx[2above - 1] * input[above]
        end
        output[k] = acc
    end
end

@kernel function laplacian_kernel!(output, input, lmax, mmax, mres)
    l, m = @index(Global, NTuple)
    if l <= lmax + 1 && m <= mmax + 1
        l_val = l - 1
        m_val = m - 1
        if l_val >= max(1, m_val) && m_val % mres == 0
            output[l, m] = -l_val * (l_val + 1) * input[l, m]
        else
            output[l, m] = zero(eltype(output))
        end
    end
end

end # module GPUCommon
