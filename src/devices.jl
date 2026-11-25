# Typed compute-device markers shared by the core and optional GPU extension.

"""Abstract supertype for compute devices supported by SHTnsKit."""
abstract type ComputeDevice end

"""CPU compute device."""
struct CPU <: ComputeDevice end

"""
GPU compute device.

`GPU()` is vendor-neutral. A CUDA, AMDGPU, or Metal backend is inferred from
an existing device array/prototype, or selected when exactly one loaded GPU
adapter is functional.
"""
struct GPU <: ComputeDevice end

"""Raised when a requested compute backend cannot perform an operation."""
struct BackendUnavailableError <: Exception
    operation::Symbol
    detail::String
end

function Base.showerror(io::IO, err::BackendUnavailableError)
    print(io, "GPU backend unavailable for `", err.operation, "`: ", err.detail)
end
