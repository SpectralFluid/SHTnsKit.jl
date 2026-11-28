# Strict typed device selection and vendor-neutral GPU adapter routing.

const _DEVICE_STATE = Ref{Union{Nothing,ComputeDevice}}(nothing)
const _DEVICE_STATE_LOCK = ReentrantLock()

# Extension modules own adapter objects strongly. Core retains only weak
# references, keyed by stable vendor names and traversed in sorted order.
const _GPU_ADAPTERS = Dict{Symbol,WeakRef}()
const _GPU_ADAPTERS_LOCK = ReentrantLock()

function _register_gpu_adapter!(name::Symbol, adapter)
    lock(_GPU_ADAPTERS_LOCK) do
        _GPU_ADAPTERS[name] = WeakRef(adapter)
    end
    return adapter
end

function _registered_gpu_adapters()
    adapters = lock(_GPU_ADAPTERS_LOCK) do
        snapshot = Pair{Symbol,Any}[]
        for (name, reference) in _GPU_ADAPTERS
            adapter = reference.value
            adapter === nothing || push!(snapshot, name => adapter)
        end
        snapshot
    end
    return sort!(adapters; by=first)
end

# Adapter protocol. Vendor extensions add methods without introducing package
# types or vendor symbols into core.
_gpu_adapter_functional(adapter) = false
_gpu_adapter_matches(adapter, value) = false

function _gpu_adapter_adapt(adapter, value)
    throw(BackendUnavailableError(:to_device, "the selected GPU adapter cannot place this value"))
end

# Copy device storage in a deeper wrapper than the adapter's methods accept
# into a dense array on the same device. Other values pass through unchanged.
_gpu_adapter_dense(adapter, value) = value

# An allocation identifies its physical GPU even if another device is active
# in the calling task. CPU arrays and single-device adapters need no switch.
_allocation_device(value) = nothing
function _allocation_device(a::AbstractArray)
    p = parent(a)
    return p isa AbstractArray && p !== a ? _allocation_device(p) : nothing
end

_on_allocation_device(f, ::Nothing) = f()

function _with_gpu_device(f, values...)
    device = nothing
    for value in values
        owner = _allocation_device(value)
        owner === nothing && continue
        if device === nothing
            device = owner
        elseif device != owner
            throw(ArgumentError("GPU operands must be on one device, got $device and $owner"))
        end
    end
    return _on_allocation_device(f, device)
end

function _with_gpu_operands(f, operation::Symbol, prototype, values::Tuple)
    selection = prototype
    if selection === nothing
        index = findfirst(value -> on_device(value) isa GPU, values)
        index === nothing || (selection = values[index])
    end
    adapter = _gpu_adapter(selection; operation)
    for value in values
        if on_device(value) isa GPU && !_gpu_adapter_matches(adapter, value)
            throw(ArgumentError("$operation operands and GPU prototype use different vendors"))
        end
    end
    return _with_gpu_device(prototype, values...) do
        placed = map(values) do value
            _gpu_adapter_matches(adapter, value) ? value : _gpu_adapter_adapt(adapter, value)
        end
        f(adapter, placed)
    end
end

function _gpu_adapter_analysis(adapter, cfg, field; kwargs...)
    throw(BackendUnavailableError(:analysis, "the selected GPU adapter does not yet implement scalar analysis"))
end

function _gpu_adapter_synthesis(adapter, cfg, coefficients; kwargs...)
    throw(BackendUnavailableError(:synthesis, "the selected GPU adapter does not yet implement scalar synthesis"))
end

function _gpu_adapter_clear_cache!(adapter)
    throw(BackendUnavailableError(
        :gpu_clear_cache!, "the selected GPU adapter does not expose cache clearing",
    ))
end

function _functional_gpu_adapters()
    functional, _ = _probe_gpu_adapters()
    return functional
end

function _probe_gpu_adapters()
    functional = Pair{Symbol,Any}[]
    failures = Pair{Symbol,Any}[]
    # `_registered_gpu_adapters` returns a strong snapshot. Runtime probes run
    # after its registry lock is released because vendor probes may block or
    # call back into package code.
    for entry in _registered_gpu_adapters()
        try
            Bool(_gpu_adapter_functional(entry.second)) && push!(functional, entry)
        catch err
            push!(failures, entry.first => err)
        end
    end
    return functional, failures
end

function _no_gpu_error(operation::Symbol, failures::AbstractVector=Pair{Symbol,Any}[])
    detail = "no loaded GPU adapter is functional; load CUDA.jl, AMDGPU.jl, or Metal.jl and verify its runtime"
    if !isempty(failures)
        causes = join(("$name: $(sprint(showerror, err))" for (name, err) in failures), "; ")
        detail *= "; functional probe failures: $causes"
    end
    return BackendUnavailableError(
        operation,
        detail,
    )
end

function _gpu_adapter(prototype=nothing; operation::Symbol=:gpu)
    if prototype !== nothing
        prototype isa AbstractArray && on_device(prototype) isa GPU || throw(ArgumentError(
            "GPU prototype for `$operation` must already use GPU storage, got $(typeof(prototype))",
        ))
        matches = filter(_registered_gpu_adapters()) do entry
            _gpu_adapter_matches(entry.second, prototype)
        end
        isempty(matches) && throw(BackendUnavailableError(
            operation,
            "no loaded GPU adapter recognizes prototype $(typeof(prototype))",
        ))
        length(matches) == 1 || throw(ArgumentError(
            "multiple GPU adapters recognize prototype $(typeof(prototype)) for `$operation`",
        ))
        name, adapter = only(matches)
        functional = try
            Bool(_gpu_adapter_functional(adapter))
        catch err
            throw(BackendUnavailableError(
                operation,
                "$name functional probe failed for $(typeof(prototype)): $(sprint(showerror, err))",
            ))
        end
        functional || throw(BackendUnavailableError(
            operation,
            "$name recognizes $(typeof(prototype)) but its runtime is not functional",
        ))
        return adapter
    end

    functional, failures = _probe_gpu_adapters()
    isempty(functional) && throw(_no_gpu_error(operation, failures))
    length(functional) == 1 && return only(functional).second
    names = join(first.(functional), ", ")
    throw(ArgumentError(
        "multiple functional GPU adapters are loaded ($names) for `$operation`; pass a device-array prototype",
    ))
end

"""Return the preferred compute device without silently relabelling requests."""
function get_device()
    requested = lock(_DEVICE_STATE_LOCK) do
        _DEVICE_STATE[]
    end
    if requested isa GPU
        functional, failures = _probe_gpu_adapters()
        isempty(functional) && throw(_no_gpu_error(:get_device, failures))
        return requested
    elseif requested isa CPU
        return requested
    end
    return isempty(_functional_gpu_adapters()) ? CPU() : GPU()
end

"""Select `CPU()` or `GPU()` as the preferred compute device."""
function set_device!(device::ComputeDevice)
    if device isa GPU
        functional, failures = _probe_gpu_adapters()
        isempty(functional) && throw(_no_gpu_error(:set_device!, failures))
    end
    lock(_DEVICE_STATE_LOCK) do
        _DEVICE_STATE[] = device
    end
    return device
end

"""
    to_device(device::ComputeDevice, value)
    to_device(GPU(), value, prototype)
    to_device(value)

Return `value` on `device`. `to_device(CPU(), x)` copies device storage into a
host `Array`, first gathering a wrapped array on its device, and returns host
values unchanged. `to_device(GPU(), x)` copies host data to the one functional
GPU backend and returns device arrays unchanged. With a `prototype` device
array it uses that array's backend and device. `to_device(value)` uses
[`get_device`](@ref). The value-first forms `to_device(value, device)` and
`to_device(value, device, prototype)` are equivalent.
"""
to_device(::CPU, value) = _to_cpu(value)

function to_device(::GPU, value)
    adapter = _gpu_adapter(on_device(value) isa GPU ? value : nothing; operation=:to_device)
    return _gpu_adapter_matches(adapter, value) ? value : _gpu_adapter_adapt(adapter, value)
end

function to_device(::GPU, value, prototype)
    prototype isa AbstractArray && on_device(prototype) isa GPU || throw(ArgumentError(
        "GPU placement prototype must already be a device array, got $(typeof(prototype))",
    ))
    return _with_gpu_operands(:to_device, prototype, (value,)) do _, placed
        only(placed)
    end
end

# Preserve the original value-first order and add the prototype form.
to_device(value::AbstractArray, device::ComputeDevice) = to_device(device, value)
to_device(value::AbstractArray, device::ComputeDevice, prototype) = to_device(device, value, prototype)
to_device(value) = to_device(get_device(), value)

_to_cpu(value) = value
_to_cpu(value::Array) = value
function _to_cpu(value::AbstractArray)
    # Gather wrapped device storage into a dense array on its device first, so
    # the transfer is one bulk copy instead of per-element reads.
    if on_device(value) isa GPU
        return _with_gpu_device(value) do
            dense = parent(value) === value ? value : (similar(value, eltype(value), size(value)) .= value)
            Array(dense)
        end
    end
    return Array(value)
end

"""Return `CPU()` for ordinary arrays; vendor extensions specialize this.
A wrapper reports the storage of the array it wraps."""
function on_device(x::AbstractArray)
    p = parent(x)
    return p !== x && p isa AbstractArray ? on_device(p) : CPU()
end

function _require_cpu_storage(operation::Symbol, value)
    on_device(value) isa CPU && return nothing
    throw(ArgumentError(
        "`$operation(CPU(), ...)` requires CPU storage; explicitly call to_device(CPU(), value) first",
    ))
end

function analysis(::CPU, cfg::SHTConfig, field::AbstractMatrix; kwargs...)
    _require_cpu_storage(:analysis, field)
    return analysis(cfg, field; kwargs...)
end

function synthesis(::CPU, cfg::SHTConfig, coefficients::AbstractMatrix; kwargs...)
    _require_cpu_storage(:synthesis, coefficients)
    return synthesis(cfg, coefficients; kwargs...)
end

function synthesis_cplx(::CPU, cfg::SHTConfig, coefficients::AbstractMatrix)
    _require_cpu_storage(:synthesis_cplx, coefficients)
    return synthesis_cplx(cfg, coefficients)
end

function analysis(::GPU, cfg::SHTConfig, field::AbstractMatrix; prototype=nothing, kwargs...)
    return _with_gpu_operands(:analysis, prototype, (field,)) do adapter, placed
        _gpu_adapter_analysis(adapter, cfg, only(placed); kwargs...)
    end
end

function synthesis(::GPU, cfg::SHTConfig, coefficients::AbstractMatrix; prototype=nothing, kwargs...)
    return _with_gpu_operands(:synthesis, prototype, (coefficients,)) do adapter, placed
        _gpu_adapter_synthesis(adapter, cfg, only(placed); kwargs...)
    end
end

function synthesis_cplx(::GPU, cfg::SHTConfig, coefficients::AbstractMatrix;
                        prototype=nothing)
    return synthesis(
        GPU(), cfg, coefficients; prototype, real_output=false,
    )
end

"""Clear scalar-table caches for the selected GPU vendor."""
function gpu_clear_cache!(::GPU; prototype=nothing)
    adapter = _gpu_adapter(prototype; operation=:gpu_clear_cache!)
    _gpu_adapter_clear_cache!(adapter)
    return nothing
end
