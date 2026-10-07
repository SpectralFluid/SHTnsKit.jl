##########
# Strict MPI communication for GPU-backed PencilArrays
##########

# This file deliberately knows nothing about CUDA or AMDGPU.  The two compound
# package extensions register the small set of vendor operations needed here.
# Mathematical code calls `allreduce!`/`exchange!` and storage helpers only.

# Weak registry entries must refer to the caller's live adapter identity.
# An immutable adapter can be boxed separately for WeakRef and disappear during
# GC even inside GC.@preserve. Const fields retain the fixed callback contract.
mutable struct ParallelGPUAdapter{FM,FA,FD,FWD,FG,FS,FH,FHD,FDH}
    const name::Symbol
    const matches::FM
    const array_type::FA
    const device::FD
    const with_device::FWD
    const gpu_aware::FG
    const synchronize::FS
    const allocate_pinned::FH
    const device_to_host!::FHD
    const host_to_device!::FDH
end

ParallelGPUAdapter(name::Symbol, matches, array_type, device, gpu_aware,
                   synchronize, allocate_pinned, device_to_host!,
                   host_to_device!) = ParallelGPUAdapter(
    name, matches, array_type, device, (f, _device) -> f(), gpu_aware,
    synchronize, allocate_pinned, device_to_host!, host_to_device!,
)

# Julia 1.11+ arrays wrap a separately collected `Memory`, and `reshape` returns
# a new wrapper over it. Host mirrors keep only reshaped wrappers, so pinning
# must follow the backing memory: a finalizer on the allocated `Vector` would
# unpin storage that is still in use once that wrapper is collected.
_pinned_storage(host::Array) =
    @static VERSION >= v"1.11.0-" ? host.ref.mem : host

const _PARALLEL_GPU_ADAPTERS = Dict{Symbol,WeakRef}()
const _PARALLEL_GPU_ADAPTER_LOCK = ReentrantLock()

function _register_parallel_gpu_adapter!(adapter::ParallelGPUAdapter)
    lock(_PARALLEL_GPU_ADAPTER_LOCK) do
        _PARALLEL_GPU_ADAPTERS[adapter.name] = WeakRef(adapter)
    end
    return adapter
end

function _parallel_gpu_adapters()
    snapshot = lock(_PARALLEL_GPU_ADAPTER_LOCK) do
        result = Pair{Symbol,Any}[]
        for (name, reference) in _PARALLEL_GPU_ADAPTERS
            value = reference.value
            value === nothing || push!(result, name => value)
        end
        result
    end
    return sort!(snapshot; by=first)
end

function _parallel_gpu_adapter(value)
    matches = Pair{Symbol,Any}[]
    for entry in _parallel_gpu_adapters()
        entry.second.matches(value) && push!(matches, entry)
    end
    isempty(matches) && return nothing
    length(matches) == 1 || throw(ArgumentError(
        "multiple MPI GPU adapters recognize $(typeof(value))",
    ))
    return only(matches).second
end

@inline _parallel_parent(value::PencilArray) = parent(value)
@inline _parallel_parent(value) = value

# Vendor device queries must inspect the allocation, not the task's current
# device. Peel the array wrappers that can sit between a PencilArray/view and
# the CUDA/ROCm allocation which owns its memory.
@inline _parallel_root_buffer(value::PencilArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::SubArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::Base.ReshapedArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::Base.ReinterpretArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::Base.LogicalIndex) =
    _parallel_root_buffer(getfield(value, :mask))
@inline _parallel_root_buffer(value::Base.PermutedDimsArray) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Adjoint) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Transpose) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Symmetric) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Hermitian) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Diagonal) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.Bidiagonal) =
    _parallel_root_buffer(getfield(value, :dv))
@inline _parallel_root_buffer(value::LinearAlgebra.Tridiagonal) =
    _parallel_root_buffer(getfield(value, :d))
@inline _parallel_root_buffer(value::LinearAlgebra.UpperTriangular) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.LowerTriangular) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.UnitUpperTriangular) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value::LinearAlgebra.UnitLowerTriangular) =
    _parallel_root_buffer(parent(value))
@inline _parallel_root_buffer(value) = value

function _parallel_array_type(prototype)
    adapter = _parallel_gpu_adapter(_parallel_parent(prototype))
    return adapter === nothing ? Array : adapter.array_type(prototype)
end

@inline _parallel_vendor_code(name::Symbol) =
    name === :cuda ? 1 : name === :amdgpu ? 2 : 3

function _parallel_array_type_code(array_type::Type)
    array_type === Array && return Int64(0)
    for entry in _parallel_gpu_adapters()
        candidate = try
            entry.second.array_type(nothing)
        catch
            nothing
        end
        array_type === candidate && return Int64(_parallel_vendor_code(entry.first))
    end
    code = UInt64(0xcbf29ce484222325)
    for byte in codeunits(string(array_type))
        code = (code ⊻ UInt64(byte)) * UInt64(0x100000001b3)
    end
    return reinterpret(Int64, code)
end

function _parallel_pencil(pen::Pencil, prototype)
    array_type = _parallel_array_type(prototype)
    return PencilArrays.typeof_array(pen) === array_type ? pen : similar(pen, array_type)
end

function _parallel_allocate(pen::Pencil, ::Type{T}, prototype,
                            extra_dims::Vararg{Integer}) where {T}
    return PencilArray{T}(undef, _parallel_pencil(pen, prototype), extra_dims...)
end

mutable struct _GPUAwarenessEntry
    comm::WeakRef
    lock::ReentrantLock
    ready::Bool
    supported::Bool
end

mutable struct _GPUStagingEntry
    root_owner::WeakRef
    logical_owner::WeakRef
    logical_id::UInt
    signature::Tuple
    buffer::Any
    lock::ReentrantLock
    tick::UInt64
    users::Int
    cacheable::Bool
    reserved_users::Int
end

@inline _parallel_logical_region(value::SubArray) =
    (typeof(value), _parallel_logical_region(parent(value)),
     Base.parentindices(value), size(value))
@inline _parallel_logical_region(value::Base.ReshapedArray) =
    (typeof(value), _parallel_logical_region(parent(value)), size(value))
@inline _parallel_logical_region(value::Base.ReinterpretArray) =
    (typeof(value), _parallel_logical_region(parent(value)),
     eltype(value), size(value))
@inline _parallel_logical_region(value::Base.PermutedDimsArray) =
    (typeof(value), _parallel_logical_region(parent(value)), size(value))
@inline _parallel_logical_region(value::LinearAlgebra.Adjoint) =
    (typeof(value), _parallel_logical_region(parent(value)), size(value))
@inline _parallel_logical_region(value::LinearAlgebra.Transpose) =
    (typeof(value), _parallel_logical_region(parent(value)), size(value))
@inline _parallel_logical_region(value::LinearAlgebra.AbstractTriangular) =
    (typeof(value), _parallel_logical_region(parent(value)), size(value))
@inline _parallel_logical_region(value::LinearAlgebra.Symmetric) =
    (typeof(value), _parallel_logical_region(parent(value)),
     getfield(value, :uplo), size(value))
@inline _parallel_logical_region(value::LinearAlgebra.Hermitian) =
    (typeof(value), _parallel_logical_region(parent(value)),
     getfield(value, :uplo), size(value))
@inline _parallel_logical_region(value::LinearAlgebra.Diagonal) =
    (typeof(value), _parallel_logical_region(getfield(value, :diag)))
@inline _parallel_logical_region(value::LinearAlgebra.Bidiagonal) =
    (typeof(value), getfield(value, :uplo),
     _parallel_logical_region(getfield(value, :dv)),
     _parallel_logical_region(getfield(value, :ev)))
@inline _parallel_logical_region(value::LinearAlgebra.Tridiagonal) =
    (typeof(value), _parallel_logical_region(getfield(value, :dl)),
     _parallel_logical_region(getfield(value, :d)),
     _parallel_logical_region(getfield(value, :du)))
@inline _parallel_logical_region(value) =
    (typeof(value), objectid(value), size(value))

mutable struct _GPUTransposeHostEntry
    plan_owner::WeakRef
    fft_plan::Any
    spatial::Any
    spectral::Any
    lock::ReentrantLock
    tick::UInt64
    users::Int
    cacheable::Bool
    reserved_users::Int
end

mutable struct _GPUPendingEntry
    event::Base.Event
    cacheable::Bool
    reserved_users::Int
end

const _GPU_AWARENESS = Dict{Tuple,_GPUAwarenessEntry}()
const _GPU_STAGING = Dict{Tuple,_GPUStagingEntry}()
const _GPU_STAGING_PENDING = Dict{Tuple,_GPUPendingEntry}()
const _GPU_STAGING_AVAILABLE = Base.Event(true)
const _GPU_TRANSPOSE_HOST = Dict{Tuple,_GPUTransposeHostEntry}()
const _GPU_TRANSPOSE_PENDING = Dict{Tuple,_GPUPendingEntry}()
const _GPU_TRANSPOSE_AVAILABLE = Base.Event(true)
const _GPU_AWARENESS_LOCK = ReentrantLock()
const _GPU_STAGING_LOCK = ReentrantLock()
const _GPU_TRANSPOSE_HOST_LOCK = ReentrantLock()
const _GPU_STAGING_LIMIT = Ref(8)
const _GPU_TRANSPOSE_HOST_LIMIT = 8
const _GPU_AWARENESS_LIMIT = 64
const _GPU_CACHE_TICK = Ref(UInt64(0))

const _GPU_DIRECT_CALLS = Threads.Atomic{Int}(0)
const _GPU_STAGED_CALLS = Threads.Atomic{Int}(0)
const _GPU_DIRECT_BYTES = Threads.Atomic{Int}(0)
const _GPU_STAGED_BYTES = Threads.Atomic{Int}(0)

@inline _communication_bytes(value) = sizeof(eltype(value)) * length(value)

function parallel_gpu_stats()
    return (
        direct_calls=_GPU_DIRECT_CALLS[],
        staged_calls=_GPU_STAGED_CALLS[],
        direct_bytes=_GPU_DIRECT_BYTES[],
        staged_bytes=_GPU_STAGED_BYTES[],
    )
end

function parallel_gpu_cache_sizes()
    awareness = lock(_GPU_AWARENESS_LOCK) do
        filter!(pair -> pair.second.comm.value !== nothing, _GPU_AWARENESS)
        length(_GPU_AWARENESS)
    end
    staging = lock(_GPU_STAGING_LOCK) do
        filter!(pair -> pair.second.root_owner.value !== nothing ||
                        pair.second.users > 0, _GPU_STAGING)
        length(_GPU_STAGING)
    end
    native_host_plans = lock(_GPU_TRANSPOSE_HOST_LOCK) do
        filter!(pair -> pair.second.plan_owner.value !== nothing ||
                        pair.second.users > 0, _GPU_TRANSPOSE_HOST)
        length(_GPU_TRANSPOSE_HOST)
    end
    return (; awareness, staging, native_host_plans)
end

function _trim_staging_cache_locked!()
    limit = _GPU_STAGING_LIMIT[]
    while length(_GPU_STAGING) + length(_GPU_STAGING_PENDING) > limit
        idle = [pair for pair in _GPU_STAGING if pair.second.users == 0]
        isempty(idle) && break
        victim = first(sort!(idle; by=pair -> pair.second.tick))
        delete!(_GPU_STAGING, first(victim))
    end
    return nothing
end

function _delete_staging_entry_locked!(entry)
    for (key, candidate) in _GPU_STAGING
        candidate === entry || continue
        delete!(_GPU_STAGING, key)
        return true
    end
    return false
end

function _release_staging_entry_locked!(entry)
    entry.users -= 1
    if entry.users == 0
        entry.cacheable || _delete_staging_entry_locked!(entry)
        _trim_staging_cache_locked!()
        notify(_GPU_STAGING_AVAILABLE)
    end
    return nothing
end

function parallel_gpu_cache_limit!(limit::Integer)
    limit >= 1 || throw(ArgumentError("MPI GPU staging cache limit must be positive"))
    return lock(_GPU_STAGING_LOCK) do
        previous = _GPU_STAGING_LIMIT[]
        _GPU_STAGING_LIMIT[] = Int(limit)
        _trim_staging_cache_locked!()
        notify(_GPU_STAGING_AVAILABLE)
        previous
    end
end

function parallel_gpu_clear_caches!()
    lock(_GPU_AWARENESS_LOCK) do
        empty!(_GPU_AWARENESS)
    end
    lock(_GPU_STAGING_LOCK) do
        # Pending builders and active users own their registry identities.
        # Clear only completed idle entries; release/publication performs the
        # eventual cleanup without allowing stale builders to overwrite peers.
        for entry in values(_GPU_STAGING)
            entry.users > 0 && (entry.cacheable = false)
        end
        filter!(pair -> pair.second.users > 0, _GPU_STAGING)
        foreach(pending -> pending.cacheable = false,
                values(_GPU_STAGING_PENDING))
        notify(_GPU_STAGING_AVAILABLE)
    end
    lock(_GPU_TRANSPOSE_HOST_LOCK) do
        for entry in values(_GPU_TRANSPOSE_HOST)
            entry.users > 0 && (entry.cacheable = false)
        end
        filter!(pair -> pair.second.users > 0, _GPU_TRANSPOSE_HOST)
        foreach(pending -> pending.cacheable = false,
                values(_GPU_TRANSPOSE_PENDING))
        notify(_GPU_TRANSPOSE_AVAILABLE)
    end
    _GPU_DIRECT_CALLS[] = 0
    _GPU_STAGED_CALLS[] = 0
    _GPU_DIRECT_BYTES[] = 0
    _GPU_STAGED_BYTES[] = 0
    return nothing
end

function _build_gpu_transpose_host_entry(adapter, plan, device, plan_owner)
    # A cache miss is local: eviction, clearing, and subcommunicator use can
    # differ across ranks. Reuse this plan's existing MPI topology so rebuilding
    # a host mirror neither creates communicators collectively nor changes the
    # communication context relative to mirrors still cached on other ranks.
    # FFT planning and pinned allocation remain outside the registry lock.
    input_pencil = similar(PencilFFTs.pencil_input(plan.fft_plan), Array)
    fft_plan = PencilFFTPlan(
        input_pencil,
        (Transforms.RFFT(), Transforms.NoTransform()),
        Transforms.eltype_input(plan.fft_plan);
        extra_dims=(plan.nlev,),
        transpose_method=plan.fft_plan.transpose_method,
    )
    spatial_template = allocate_input(fft_plan)
    spectral_template = allocate_output(fft_plan)
    spatial_parent, spectral_parent = adapter.with_device(device) do
        spatial_parent = adapter.allocate_pinned(
            eltype(spatial_template), length(parent(spatial_template)),
        )
        spectral_parent = adapter.allocate_pinned(
            eltype(spectral_template), length(parent(spectral_template)),
        )
        spatial_parent, spectral_parent
    end
    spatial = PencilArray(
        pencil(spatial_template),
        reshape(spatial_parent, size(parent(spatial_template))),
    )
    spectral = PencilArray(
        pencil(spectral_template),
        reshape(spectral_parent, size(parent(spectral_template))),
    )
    return _GPUTransposeHostEntry(
        WeakRef(plan_owner), fft_plan, spatial, spectral, ReentrantLock(), 0, 0,
        true, 0,
    )
end

function _acquire_gpu_transpose_host_entry(adapter, plan, prototype)
    device = adapter.device(_parallel_root_buffer(_parallel_parent(prototype)))
    # DistTransposePlan is immutable, so WeakRef(plan) may refer to a transient
    # box even while the logical plan remains live. Its mutable F_buf root has
    # stable identity and exactly follows the plan/workspace lifetime.
    plan_owner = _parallel_root_buffer(_parallel_parent(plan.F_buf))
    key = (objectid(plan_owner), adapter.name, device)
    while true
        found, pending, builder, wait_for_slot = lock(
                _GPU_TRANSPOSE_HOST_LOCK) do
            filter!(pair -> pair.second.plan_owner.value !== nothing ||
                            pair.second.users > 0, _GPU_TRANSPOSE_HOST)
            entry = get(_GPU_TRANSPOSE_HOST, key, nothing)
            if entry !== nothing && entry.plan_owner.value === plan_owner
                if entry.reserved_users > 0
                    entry.reserved_users -= 1
                else
                    entry.users += 1
                end
                _GPU_CACHE_TICK[] += 1
                entry.tick = _GPU_CACHE_TICK[]
                return (entry, nothing, false, false)
            end
            pending = get(_GPU_TRANSPOSE_PENDING, key, nothing)
            if pending !== nothing
                pending.reserved_users += 1
                return (nothing, pending, false, false)
            end
            if length(_GPU_TRANSPOSE_HOST) + length(_GPU_TRANSPOSE_PENDING) >=
                    _GPU_TRANSPOSE_HOST_LIMIT
                idle = [pair for pair in _GPU_TRANSPOSE_HOST
                        if pair.second.users == 0]
                isempty(idle) && return (nothing, nothing, false, true)
                victim = first(sort!(idle; by=pair -> pair.second.tick))
                delete!(_GPU_TRANSPOSE_HOST, first(victim))
            end
            pending = _GPUPendingEntry(Base.Event(), true, 0)
            _GPU_TRANSPOSE_PENDING[key] = pending
            return (nothing, pending, true, false)
        end
        found === nothing || return found
        if wait_for_slot
            wait(_GPU_TRANSPOSE_AVAILABLE)
            continue
        elseif !builder
            wait(pending.event)
            continue
        end
        candidate = try
            _build_gpu_transpose_host_entry(adapter, plan, device, plan_owner)
        catch
            lock(_GPU_TRANSPOSE_HOST_LOCK) do
                get(_GPU_TRANSPOSE_PENDING, key, nothing) === pending &&
                    delete!(_GPU_TRANSPOSE_PENDING, key)
                notify(pending.event)
                notify(_GPU_TRANSPOSE_AVAILABLE)
            end
            rethrow()
        end
        published = lock(_GPU_TRANSPOSE_HOST_LOCK) do
            if get(_GPU_TRANSPOSE_PENDING, key, nothing) === pending
                candidate.users = 1 + pending.reserved_users
                candidate.reserved_users = pending.reserved_users
                candidate.cacheable = pending.cacheable
                _GPU_CACHE_TICK[] += 1
                candidate.tick = _GPU_CACHE_TICK[]
                _GPU_TRANSPOSE_HOST[key] = candidate
                delete!(_GPU_TRANSPOSE_PENDING, key)
                notify(pending.event)
                return candidate
            end
            notify(pending.event)
            nothing
        end
        published === nothing || return published
    end
end

function _release_gpu_transpose_host_entry(entry)
    lock(_GPU_TRANSPOSE_HOST_LOCK) do
        entry.users -= 1
        if entry.users == 0
            if !entry.cacheable
                for (key, candidate) in _GPU_TRANSPOSE_HOST
                    candidate === entry || continue
                    delete!(_GPU_TRANSPOSE_HOST, key)
                    break
                end
            end
            notify(_GPU_TRANSPOSE_AVAILABLE)
        end
    end
    return nothing
end

function _with_gpu_transpose_host_entry(f, adapter, plan, prototype)
    entry = _acquire_gpu_transpose_host_entry(adapter, plan, prototype)
    try
        return lock(entry.lock) do
            f(entry)
        end
    finally
        _release_gpu_transpose_host_entry(entry)
    end
end

function _gpu_transpose_aware(adapter::ParallelGPUAdapter, plan, buffer)
    local_aware = _gpu_awareness(adapter, plan.comm, _parallel_parent(buffer))
    return MPI.Allreduce(local_aware ? 1 : 0, min, plan.comm) == 1
end

"""Run a native forward PencilFFT directly or through its cached CPU mirror."""
function _gpu_transpose_forward!(adapter::ParallelGPUAdapter, plan,
                                 device_output, device_input)
    bytes = _communication_bytes(_parallel_parent(device_input)) +
            _communication_bytes(_parallel_parent(device_output))
    if _gpu_transpose_aware(adapter, plan, device_input)
        result = _with_owner_device(adapter, device_input) do
            adapter.synchronize(_parallel_parent(device_input))
            mul!(device_output, plan.fft_plan, device_input)
            adapter.synchronize(_parallel_parent(device_output))
            device_output
        end
        Threads.atomic_add!(_GPU_DIRECT_CALLS, 1)
        Threads.atomic_add!(_GPU_DIRECT_BYTES, bytes)
        return result
    end
    result = _with_gpu_transpose_host_entry(adapter, plan, device_input) do entry
        _device_to_host_snapshot!(
            adapter, parent(entry.spatial), device_input,
        )
        mul!(entry.spectral, entry.fft_plan, entry.spatial)
        _host_to_device_snapshot!(
            adapter, device_output, parent(entry.spectral),
        )
        device_output
    end
    Threads.atomic_add!(_GPU_STAGED_CALLS, 1)
    Threads.atomic_add!(_GPU_STAGED_BYTES, bytes)
    return result
end

"""Run a native inverse PencilFFT directly or through its cached CPU mirror."""
function _gpu_transpose_inverse!(adapter::ParallelGPUAdapter, plan,
                                 device_output, device_input)
    bytes = _communication_bytes(_parallel_parent(device_input)) +
            _communication_bytes(_parallel_parent(device_output))
    if _gpu_transpose_aware(adapter, plan, device_input)
        result = _with_owner_device(adapter, device_input) do
            adapter.synchronize(_parallel_parent(device_input))
            ldiv!(device_output, plan.fft_plan, device_input)
            adapter.synchronize(_parallel_parent(device_output))
            device_output
        end
        Threads.atomic_add!(_GPU_DIRECT_CALLS, 1)
        Threads.atomic_add!(_GPU_DIRECT_BYTES, bytes)
        return result
    end
    result = _with_gpu_transpose_host_entry(adapter, plan, device_input) do entry
        _device_to_host_snapshot!(
            adapter, parent(entry.spectral), device_input,
        )
        ldiv!(entry.spatial, entry.fft_plan, entry.spectral)
        _host_to_device_snapshot!(
            adapter, device_output, parent(entry.spatial),
        )
        device_output
    end
    Threads.atomic_add!(_GPU_STAGED_CALLS, 1)
    Threads.atomic_add!(_GPU_STAGED_BYTES, bytes)
    return result
end

function _gpu_awareness(adapter::ParallelGPUAdapter, comm, buffer)
    device = adapter.device(buffer)
    key = (objectid(comm), adapter.name, device)
    entry = lock(_GPU_AWARENESS_LOCK) do
        filter!(pair -> pair.second.comm.value !== nothing, _GPU_AWARENESS)
        current = get(_GPU_AWARENESS, key, nothing)
        if current === nothing || current.comm.value !== comm
            current = _GPUAwarenessEntry(WeakRef(comm), ReentrantLock(), false, false)
            if length(_GPU_AWARENESS) >= _GPU_AWARENESS_LIMIT
                delete!(_GPU_AWARENESS, first(keys(_GPU_AWARENESS)))
            end
            _GPU_AWARENESS[key] = current
        end
        current
    end
    # Detection may call into MPI/vendor code, so it runs outside the registry
    # lock.  The entry lock ensures one completed publication per key.
    return lock(entry.lock) do
        if !entry.ready
            entry.supported = Bool(adapter.gpu_aware(comm))
            entry.ready = true
        end
        entry.supported
    end
end

function _staging_entry(adapter::ParallelGPUAdapter, comm, owner, n::Int;
                        acquire::Bool=false, T::Type=eltype(owner))
    root_owner = _parallel_root_buffer(owner)
    logical_region = _parallel_logical_region(owner)
    logical_id = hash(logical_region)
    device = adapter.device(root_owner)
    signature = (objectid(comm), adapter.name, device, T, n)
    key = (logical_region, objectid(root_owner), signature)

    while true
        found, pending, builder, wait_for_slot = lock(_GPU_STAGING_LOCK) do
            entry = get(_GPU_STAGING, key, nothing)
            if entry !== nothing && entry.logical_id == logical_id &&
               entry.root_owner.value === root_owner
                entry.logical_owner = WeakRef(owner)
                if acquire
                    if entry.reserved_users > 0
                        entry.reserved_users -= 1
                    else
                        entry.users += 1
                    end
                end
                _GPU_CACHE_TICK[] += 1
                entry.tick = _GPU_CACHE_TICK[]
                return (entry, nothing, false, false)
            end
            filter!(pair -> pair.second.root_owner.value !== nothing ||
                            pair.second.users > 0, _GPU_STAGING)
            pending = get(_GPU_STAGING_PENDING, key, nothing)
            if pending !== nothing
                acquire && (pending.reserved_users += 1)
                return (nothing, pending, false, false)
            end

            limit = _GPU_STAGING_LIMIT[]
            occupied = length(_GPU_STAGING) + length(_GPU_STAGING_PENDING)
            if occupied >= limit
                compatible = [pair for pair in _GPU_STAGING
                              if pair.second.users == 0 &&
                                 pair.second.signature[2] === adapter.name &&
                                 pair.second.signature[3] == device &&
                                 eltype(pair.second.buffer) === T &&
                                 length(pair.second.buffer) >= n]
                if !isempty(compatible)
                    victim = first(sort!(compatible; by=pair -> pair.second.tick))
                    delete!(_GPU_STAGING, first(victim))
                    reused = last(victim)
                    reused.root_owner = WeakRef(root_owner)
                    reused.logical_owner = WeakRef(owner)
                    reused.logical_id = logical_id
                    reused.signature = signature
                    reused.users = acquire ? 1 : 0
                    reused.reserved_users = 0
                    _GPU_CACHE_TICK[] += 1
                    reused.tick = _GPU_CACHE_TICK[]
                    _GPU_STAGING[key] = reused
                    return (reused, nothing, false, false)
                end
                idle = [pair for pair in _GPU_STAGING
                        if pair.second.users == 0]
                isempty(idle) && return (nothing, nothing, false, true)
                victim = first(sort!(idle; by=pair -> pair.second.tick))
                delete!(_GPU_STAGING, first(victim))
            end
            pending = _GPUPendingEntry(Base.Event(), true, 0)
            _GPU_STAGING_PENDING[key] = pending
            return (nothing, pending, true, false)
        end
        found === nothing || return found
        if wait_for_slot
            wait(_GPU_STAGING_AVAILABLE)
            continue
        elseif !builder
            wait(pending.event)
            continue
        end

        buffer = try
            adapter.with_device(device) do
                adapter.allocate_pinned(T, n)
            end
        catch
            lock(_GPU_STAGING_LOCK) do
                get(_GPU_STAGING_PENDING, key, nothing) === pending &&
                    delete!(_GPU_STAGING_PENDING, key)
                notify(pending.event)
                notify(_GPU_STAGING_AVAILABLE)
            end
            rethrow()
        end
        candidate = _GPUStagingEntry(
            WeakRef(root_owner), WeakRef(owner), logical_id, signature, buffer,
            ReentrantLock(), UInt64(0),
            (acquire ? 1 : 0) + pending.reserved_users,
            pending.cacheable, pending.reserved_users,
        )
        published = lock(_GPU_STAGING_LOCK) do
            if get(_GPU_STAGING_PENDING, key, nothing) === pending
                candidate.users = (acquire ? 1 : 0) + pending.reserved_users
                candidate.reserved_users = pending.reserved_users
                candidate.cacheable = pending.cacheable
                _GPU_CACHE_TICK[] += 1
                candidate.tick = _GPU_CACHE_TICK[]
                _GPU_STAGING[key] = candidate
                delete!(_GPU_STAGING_PENDING, key)
                notify(pending.event)
                return candidate
            end
            notify(pending.event)
            nothing
        end
        published === nothing || return published
    end
end

function _release_staging_entry(entry)
    lock(_GPU_STAGING_LOCK) do
        _release_staging_entry_locked!(entry)
    end
    return nothing
end

function _staging_entry_description(adapter::ParallelGPUAdapter, comm, owner)
    root_owner = _parallel_root_buffer(owner)
    logical_region = _parallel_logical_region(owner)
    logical_id = hash(logical_region)
    device = adapter.device(root_owner)
    T = eltype(owner)
    n = length(owner)
    signature = (objectid(comm), adapter.name, device, T, n)
    key = (logical_region, objectid(root_owner), signature)
    return (; owner, root_owner, logical_region, logical_id, device, T, n,
            signature, key)
end

"""
Atomically reserve the complete unique staging working set for one operation.

The configured limit bounds retained idle buffers, not live call requirements.
Entries beyond that limit remain discoverable while active, then are discarded
on final release. No allocation or copy occurs under the registry lock.
"""
function _staging_entries(adapter::ParallelGPUAdapter, comm, owners...)
    descriptions = map(owner -> _staging_entry_description(
        adapter, comm, owner,
    ), owners)
    unique_descriptions = Any[]
    description_by_key = Dict{Any,Any}()
    for description in descriptions
        haskey(description_by_key, description.key) && continue
        description_by_key[description.key] = description
        push!(unique_descriptions, description)
    end

    while true
        reservation = lock(_GPU_STAGING_LOCK) do
            filter!(pair -> pair.second.root_owner.value !== nothing ||
                            pair.second.users > 0, _GPU_STAGING)
            waits = unique(get(_GPU_STAGING_PENDING, description.key, nothing)
                           for description in unique_descriptions
                           if haskey(_GPU_STAGING_PENDING, description.key))
            foreach(pending -> pending.reserved_users += 1, waits)
            isempty(waits) || return (; wait=waits, acquired=Any[], builds=Any[])

            acquired = Any[]
            builds = Any[]
            # Existing requested entries become active before idle eviction.
            for description in unique_descriptions
                entry = get(_GPU_STAGING, description.key, nothing)
                if entry !== nothing &&
                   entry.logical_id == description.logical_id &&
                   entry.root_owner.value === description.root_owner
                    entry.logical_owner = WeakRef(description.owner)
                    if entry.reserved_users > 0
                        entry.reserved_users -= 1
                    else
                        entry.users += 1
                    end
                    _GPU_CACHE_TICK[] += 1
                    entry.tick = _GPU_CACHE_TICK[]
                    push!(acquired, description.key => entry)
                end
            end

            for description in unique_descriptions
                any(pair -> first(pair) == description.key, acquired) && continue
                # Prefer recycling an idle compatible retained allocation.
                compatible = [pair for pair in _GPU_STAGING
                              if pair.second.users == 0 &&
                                 pair.second.signature[2] === adapter.name &&
                                 pair.second.signature[3] == description.device &&
                                 eltype(pair.second.buffer) === description.T &&
                                 length(pair.second.buffer) >= description.n]
                if !isempty(compatible)
                    victim = first(sort!(compatible; by=pair -> pair.second.tick))
                    delete!(_GPU_STAGING, first(victim))
                    entry = last(victim)
                    entry.root_owner = WeakRef(description.root_owner)
                    entry.logical_owner = WeakRef(description.owner)
                    entry.logical_id = description.logical_id
                    entry.signature = description.signature
                    entry.users = 1
                    entry.cacheable = true
                    entry.reserved_users = 0
                    _GPU_CACHE_TICK[] += 1
                    entry.tick = _GPU_CACHE_TICK[]
                    _GPU_STAGING[description.key] = entry
                    push!(acquired, description.key => entry)
                    continue
                end

                # Evict one idle entry if doing so creates retained capacity.
                if length(_GPU_STAGING) + length(_GPU_STAGING_PENDING) >=
                        _GPU_STAGING_LIMIT[]
                    idle = [pair for pair in _GPU_STAGING
                            if pair.second.users == 0]
                    if !isempty(idle)
                        victim = first(sort!(idle; by=pair -> pair.second.tick))
                        delete!(_GPU_STAGING, first(victim))
                    end
                end
                cacheable = length(_GPU_STAGING) +
                            length(_GPU_STAGING_PENDING) < _GPU_STAGING_LIMIT[]
                pending = _GPUPendingEntry(Base.Event(), cacheable, 0)
                _GPU_STAGING_PENDING[description.key] = pending
                push!(builds, (; description, pending))
            end
            return (; wait=Any[], acquired, builds)
        end

        if !isempty(reservation.wait)
            foreach(pending -> wait(pending.event), reservation.wait)
            continue
        end

        candidates = Any[]
        try
            for build in reservation.builds
                description = build.description
                buffer = adapter.with_device(description.device) do
                    adapter.allocate_pinned(description.T, description.n)
                end
                entry = _GPUStagingEntry(
                    WeakRef(description.root_owner), WeakRef(description.owner),
                    description.logical_id, description.signature, buffer,
                    ReentrantLock(), UInt64(0),
                    1 + build.pending.reserved_users,
                    build.pending.cacheable, build.pending.reserved_users,
                )
                push!(candidates, (; build, entry))
            end
        catch
            lock(_GPU_STAGING_LOCK) do
                for build in reservation.builds
                    get(_GPU_STAGING_PENDING, build.description.key, nothing) ===
                        build.pending || continue
                    delete!(_GPU_STAGING_PENDING, build.description.key)
                    notify(build.pending.event)
                end
                for pair in reservation.acquired
                    _release_staging_entry_locked!(last(pair))
                end
                _trim_staging_cache_locked!()
                notify(_GPU_STAGING_AVAILABLE)
            end
            rethrow()
        end

        published = lock(_GPU_STAGING_LOCK) do
            entries = copy(reservation.acquired)
            valid = true
            for candidate in candidates
                build = candidate.build
                if get(_GPU_STAGING_PENDING, build.description.key, nothing) !==
                        build.pending
                    valid = false
                    break
                end
            end
            if valid
                for candidate in candidates
                    build = candidate.build
                    entry = candidate.entry
                    entry.users = 1 + build.pending.reserved_users
                    entry.reserved_users = build.pending.reserved_users
                    entry.cacheable = build.pending.cacheable
                    _GPU_CACHE_TICK[] += 1
                    entry.tick = _GPU_CACHE_TICK[]
                    _GPU_STAGING[build.description.key] = entry
                    delete!(_GPU_STAGING_PENDING, build.description.key)
                    push!(entries, build.description.key => entry)
                end
            else
                for pair in reservation.acquired
                    _release_staging_entry_locked!(last(pair))
                end
            end
            foreach(candidate -> notify(candidate.build.pending.event), candidates)
            notify(_GPU_STAGING_AVAILABLE)
            valid ? Dict(entries) : nothing
        end
        published === nothing && continue
        return map(description -> published[description.key], descriptions)
    end
end

function _with_staging(f, adapter::ParallelGPUAdapter, comm, owner,
                       ::Type{T}=eltype(owner), n::Int=length(owner)) where {T}
    entry = _staging_entry(adapter, comm, owner, n; acquire=true, T)
    try
        return lock(entry.lock) do
            f(view(entry.buffer, 1:n))
        end
    finally
        _release_staging_entry(entry)
    end
end

function _host_staging_value(adapter::ParallelGPUAdapter, comm, value,
                             reserved_entry=nothing)
    raw = _parallel_parent(value)
    adapter.matches(raw) || return (value, nothing)
    entry = reserved_entry === nothing ?
        _staging_entry(adapter, comm, raw, length(raw); acquire=true) :
        reserved_entry
    buffer = view(entry.buffer, 1:length(raw))
    host = if value isa PencilArray
        host_pen = similar(pencil(value), Array)
        PencilArray(host_pen, reshape(buffer, size(raw)))
    else
        reshape(buffer, size(raw))
    end
    return host, entry
end

function _with_entry_locks(f, entries, index::Int=1)
    index > length(entries) && return f()
    return lock(entries[index].lock) do
        _with_entry_locks(f, entries, index + 1)
    end
end

@inline function _with_owner_device(f, adapter::ParallelGPUAdapter, value)
    raw = _parallel_parent(value)
    device = adapter.device(_parallel_root_buffer(raw))
    return adapter.with_device(f, device)
end

@inline function _device_to_host_snapshot!(adapter::ParallelGPUAdapter,
                                           host, value)
    raw = _parallel_parent(value)
    return _with_owner_device(adapter, raw) do
        adapter.synchronize(raw)
        adapter.device_to_host!(host, raw)
        adapter.synchronize(raw)
        host
    end
end

@inline function _host_to_device_snapshot!(adapter::ParallelGPUAdapter,
                                           value, host)
    raw = _parallel_parent(value)
    return _with_owner_device(adapter, raw) do
        adapter.host_to_device!(raw, host)
        adapter.synchronize(raw)
        raw
    end
end

function _device_result_with_comm(adapter::ParallelGPUAdapter, comm, prototype,
                                  result, staged=(), originals=())
    for (pair, original) in zip(staged, originals)
        result === first(pair) && return original
    end
    if result isa PencilArray
        device = adapter.device(_parallel_root_buffer(prototype))
        return _with_staging(
                adapter, comm, prototype, eltype(result), length(result)) do pinned
            copyto!(pinned, vec(parent(result)))
            adapter.with_device(device) do
                array_type = adapter.array_type(prototype)
                device_pen = similar(pencil(result), array_type)
                output = PencilArray{eltype(result)}(
                    undef, device_pen, PencilArrays.extra_dims(result)...,
                )
                adapter.host_to_device!(parent(output),
                                        reshape(pinned, size(parent(result))))
                adapter.synchronize(parent(output))
                output
            end
        end
    elseif result isa AbstractArray
        device = adapter.device(_parallel_root_buffer(prototype))
        return _with_staging(
                adapter, comm, prototype, eltype(result), length(result)) do pinned
            copyto!(pinned, vec(result))
            adapter.with_device(device) do
                array_type = adapter.array_type(prototype)
                output = array_type{eltype(result)}(undef, size(result))
                adapter.host_to_device!(output, reshape(pinned, size(result)))
                adapter.synchronize(output)
                output
            end
        end
    elseif result isa Tuple
        return map(value -> _device_result_with_comm(
            adapter, comm, prototype, value, staged, originals,
        ), result)
    elseif result isa NamedTuple
        return map(value -> _device_result_with_comm(
            adapter, comm, prototype, value, staged, originals,
        ), result)
    end
    return result
end

_device_result(adapter::ParallelGPUAdapter, prototype, result,
               staged=(), originals=()) = _device_result_with_comm(
    adapter, MPI.COMM_SELF, prototype, result, staged, originals,
)

function _materialize_staged_result(result, staged)
    any(pair -> result === first(pair), staged) && return result
    if result isa PencilArray
        host_pen = similar(pencil(result), Array)
        host_parent = copy(parent(result))
        return PencilArray(host_pen, host_parent)
    elseif result isa AbstractArray
        return Array(result)
    elseif result isa Tuple
        return map(value -> _materialize_staged_result(value, staged), result)
    elseif result isa NamedTuple
        return map(value -> _materialize_staged_result(value, staged), result)
    end
    return result
end

"""
    _staged_gpu_call(adapter, operation, comm, f, values...)

Execute an existing CPU Pencil algorithm through bounded pinned snapshots.  All
GPU inputs are copied before `f` runs and every array result is restored to the
same vendor before return.  This is the only permitted host-staging boundary
for ordinary MPI mathematical entry points.
"""
function _staged_gpu_call(adapter::ParallelGPUAdapter, operation::Symbol,
                          comm, f, values...;
                          mutated::Tuple=(), validate_storage::Bool=true)
    validate_storage && _validate_parallel_storage!(
        comm, operation, values...; adapter,
    )
    all(index -> index in eachindex(values), mutated) || throw(ArgumentError(
        "$operation staged mutation index is out of bounds",
    ))
    device_values = [_parallel_parent(value) for value in values
                     if adapter.matches(_parallel_parent(value))]
    reserved = _staging_entries(adapter, comm, device_values...)
    staged = Any[]
    try
        next_reserved = 1
        for value in values
            entry = adapter.matches(_parallel_parent(value)) ?
                reserved[next_reserved] : nothing
            push!(staged, _host_staging_value(adapter, comm, value, entry))
            entry === nothing || (next_reserved += 1)
        end
    catch
        foreach(_release_staging_entry, unique(reserved))
        rethrow()
    end
    hosts = first.(staged)
    entries = unique(last(pair) for pair in staged if last(pair) !== nothing)
    sort!(entries; by=objectid)
    isempty(entries) && throw(ArgumentError(
        "$operation staged GPU call received no device-backed value",
    ))
    prototype = _parallel_parent(first(value for value in values
                                       if adapter.matches(_parallel_parent(value))))
    bytes = sum(_communication_bytes(_parallel_parent(value)) for value in values
                if adapter.matches(_parallel_parent(value)))
    cpu_result = try
        _with_entry_locks(entries) do
            for (value, pair) in zip(values, staged)
                last(pair) === nothing && continue
                _device_to_host_snapshot!(
                    adapter, _parallel_parent(first(pair)), value,
                )
            end
            computed = f(hosts...)
            for index in mutated
                pair = staged[index]
                last(pair) === nothing && continue
                _host_to_device_snapshot!(
                    adapter, values[index], _parallel_parent(first(pair)),
                )
            end
            # Results may alias a pinned input snapshot. Materialize every
            # non-bang array while its entry locks are held, then release all
            # input reservations before acquiring result staging capacity.
            _materialize_staged_result(computed, staged)
        end
    finally
        foreach(_release_staging_entry, entries)
    end
    result = _device_result_with_comm(
        adapter, comm, prototype, cpu_result, staged, values,
    )
    Threads.atomic_add!(_GPU_STAGED_CALLS, 1)
    Threads.atomic_add!(_GPU_STAGED_BYTES, bytes)
    return result
end

"""
    allreduce!(buffer, op, comm)

All-reduce `buffer` in place. GPU buffers use direct device communication only
when the registered vendor and MPI implementation report support; otherwise a
bounded reusable pinned-host stage is used.
"""
function allreduce!(buffer, op, comm;
                    adapter=_parallel_gpu_adapter(buffer),
                    collective=MPI.Allreduce!)
    adapter === nothing && return collective(buffer, op, comm)
    bytes = _communication_bytes(buffer)
    if _gpu_awareness(adapter, comm, buffer)
        result = _with_owner_device(adapter, buffer) do
            adapter.synchronize(buffer)
            value = collective(buffer, op, comm)
            adapter.synchronize(buffer)
            value
        end
        Threads.atomic_add!(_GPU_DIRECT_CALLS, 1)
        Threads.atomic_add!(_GPU_DIRECT_BYTES, bytes)
        return result
    end
    result = _with_staging(adapter, comm, buffer) do host
        _device_to_host_snapshot!(adapter, host, buffer)
        collective(host, op, comm)
        _host_to_device_snapshot!(adapter, buffer, host)
        buffer
    end
    Threads.atomic_add!(_GPU_STAGED_CALLS, 1)
    Threads.atomic_add!(_GPU_STAGED_BYTES, bytes)
    return result
end

"""
    exchange!(send, receive, comm)

Vendor-neutral two-buffer exchange used by transpose/redistribution paths.
The callback has signature `(send, receive, comm)`; callers may supply the
exact MPI collective required by their layout.
"""
function exchange!(send, receive, comm;
                   adapter=_parallel_gpu_adapter(send),
                   collective=MPI.Alltoall!)
    alias_flags = Base.mightalias(send, receive) ? UInt32(0x2000) : UInt32(0)
    _collective_validation_error(comm, alias_flags, :exchange)
    receive_adapter = _parallel_gpu_adapter(receive)
    if adapter === nothing && receive_adapter === nothing
        return collective(send, receive, comm)
    end
    _validate_parallel_storage!(comm, :exchange, send, receive; adapter)
    adapter !== nothing && adapter.matches(send) && adapter.matches(receive) &&
        (receive_adapter === nothing || adapter.name === receive_adapter.name) ||
        throw(ArgumentError(
        "MPI exchange send/receive buffers must use the same GPU vendor",
    ))
    bytes = _communication_bytes(send) + _communication_bytes(receive)
    if _gpu_awareness(adapter, comm, send)
        result = _with_owner_device(adapter, send) do
            adapter.synchronize(send)
            value = collective(send, receive, comm)
            adapter.synchronize(receive)
            value
        end
        Threads.atomic_add!(_GPU_DIRECT_CALLS, 1)
        Threads.atomic_add!(_GPU_DIRECT_BYTES, bytes)
        return result
    end
    send_entry, receive_entry = _staging_entries(
        adapter, comm, send, receive,
    )
    result = try
        send_entry === receive_entry && throw(ArgumentError(
            "MPI exchange send/receive buffers may not alias",
        ))
        # Lock in object-id order to keep simultaneous exchanges deadlock-free.
        first_entry, second_entry = objectid(send_entry) < objectid(receive_entry) ?
            (send_entry, receive_entry) : (receive_entry, send_entry)
        lock(first_entry.lock) do
            lock(second_entry.lock) do
                send_host = view(send_entry.buffer, 1:length(send))
                receive_host = view(receive_entry.buffer, 1:length(receive))
                _device_to_host_snapshot!(adapter, send_host, send)
                collective(send_host, receive_host, comm)
                _host_to_device_snapshot!(adapter, receive, receive_host)
                receive
            end
        end
    finally
        _release_staging_entry(receive_entry)
        _release_staging_entry(send_entry)
    end
    Threads.atomic_add!(_GPU_STAGED_CALLS, 1)
    Threads.atomic_add!(_GPU_STAGED_BYTES, bytes)
    return result
end

function _parallel_storage_code(value)
    adapter = _parallel_gpu_adapter(_parallel_parent(value))
    adapter === nothing && return 0
    return _parallel_vendor_code(adapter.name)
end

"""Validate vendor/residency collectively before allocation or mutation."""
function _validate_parallel_storage!(comm, operation::Symbol, values...;
                                     adapter=nothing)
    flags = UInt32(0)
    codes = map(_parallel_storage_code, values)
    local_min = isempty(codes) ? 0 : minimum(codes)
    local_max = isempty(codes) ? 0 : maximum(codes)
    local_min == local_max || (flags |= 0x20000)
    global_min = MPI.Allreduce(local_min, min, comm)
    global_max = MPI.Allreduce(local_max, max, comm)
    global_min == global_max || (flags |= 0x20000)
    device_descriptors = Any[]
    for value in values
        raw = _parallel_parent(value)
        value_adapter = adapter !== nothing && adapter.matches(raw) ? adapter :
                        _parallel_gpu_adapter(raw)
        value_adapter === nothing && continue
        push!(device_descriptors, (
            value_adapter.name,
            value_adapter.device(_parallel_root_buffer(raw)),
        ))
    end
    local_device_mismatch = any(
        descriptor -> descriptor[1] != first(device_descriptors)[1] ||
                      descriptor[2] != first(device_descriptors)[2],
        Iterators.drop(device_descriptors, 1),
    )
    MPI.Allreduce(local_device_mismatch ? 1 : 0, max, comm) == 0 ||
        (flags |= 0x40000)
    flags == 0 || throw(ArgumentError(
        "$operation collective validation failed: storage/vendor/device mismatch",
    ))
    return local_min
end

"""Shared first collective for dense compatibility synthesis entry points."""
function _validate_dense_synthesis_storage!(comm, operation::Symbol,
                                            values...)
    _validate_parallel_storage!(comm, operation, values...)
    return nothing
end

"""Validate scalar dense storage and optional negative-m storage in one order."""
function _validate_dense_scalar_synthesis_storage!(
        comm, coefficients, prototype, Aminus=nothing)
    operation = :dist_synthesis_dense
    _validate_dense_synthesis_storage!(
        comm, operation, coefficients, prototype,
    )
    minus_count = MPI.Allreduce(Aminus === nothing ? 0 : 1, +, comm)
    comm_size = MPI.Comm_size(comm)
    minus_count in (0, comm_size) || throw(ArgumentError(
        "$operation collective validation failed: rank-varying Aminus presence",
    ))
    has_minus = minus_count == comm_size
    has_minus && _validate_dense_synthesis_storage!(
        comm, operation, coefficients, Aminus, prototype,
    )
    return has_minus
end

function _dist_transpose_gpu_analysis!(adapter, plan, output, input)
    return _dist_transpose_gpu_analysis!(
        Val(adapter.name), adapter, plan, output, input,
    )
end

function _dist_transpose_gpu_synthesis!(adapter, plan, output, input)
    return _dist_transpose_gpu_synthesis!(
        Val(adapter.name), adapter, plan, output, input,
    )
end


function _dist_transpose_gpu_vector_analysis!(adapter, plan, Sout, Tout, Vt, Vp)
    return _dist_transpose_gpu_vector_analysis!(
        Val(adapter.name), adapter, plan, Sout, Tout, Vt, Vp,
    )
end

function _dist_transpose_gpu_vector_synthesis!(adapter, plan, Vt, Vp, Sin, Tin)
    return _dist_transpose_gpu_vector_synthesis!(
        Val(adapter.name), adapter, plan, Vt, Vp, Sin, Tin,
    )
end

_missing_parallel_gpu_operation(name, operation) = throw(ArgumentError(
    "MPI GPU adapter $name does not implement $operation",
))
_dist_transpose_gpu_analysis!(::Val{name}, args...) where {name} =
    _missing_parallel_gpu_operation(name, :dist_analysis)
_dist_transpose_gpu_synthesis!(::Val{name}, args...) where {name} =
    _missing_parallel_gpu_operation(name, :dist_synthesis)
_dist_transpose_gpu_vector_analysis!(::Val{name}, args...) where {name} =
    _missing_parallel_gpu_operation(name, :dist_analysis_sphtor)
_dist_transpose_gpu_vector_synthesis!(::Val{name}, args...) where {name} =
    _missing_parallel_gpu_operation(name, :dist_synthesis_sphtor)
