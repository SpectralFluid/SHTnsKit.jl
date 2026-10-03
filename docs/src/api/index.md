# API Reference

This page is generated from the docstrings of the public `SHTnsKit` module.
Optional extension methods appear when their dependencies are loaded in the
calling environment.

## Calling conventions

- Allocating transforms use `operation(cfg, input...)`.
- In-place transforms place outputs before inputs, for example
  `analysis!(plan, alm_out, field)` and
  `synthesis!(plan, field_out, alm)`.
- `analysis(CPU(), ...)` and `analysis(GPU(), ...)` request strict typed device
  dispatch; ordinary calls infer the backend from array storage.
- Dense coefficients have shape `(cfg.lmax + 1, cfg.mmax + 1)`. Entries with
  `l < m` are unused.
- Batch APIs add a third, field-index dimension.
- A configuration's `norm`, `cs_phase`, `real_norm`, and `robert_form` options
  apply at every public transform boundary.

See [Migrating to v2.0](../migration.md) for removed compatibility signatures
and numerical changes.

## Extension activation

| Extension | Load these packages |
|---|---|
| CUDA | `CUDA`, `GPUArrays`, `GPUArraysCore`, `KernelAbstractions` |
| AMDGPU | `AMDGPU`, `GPUArrays`, `GPUArraysCore`, `KernelAbstractions` |
| MPI | `MPI`, `PencilArrays`, `PencilFFTs` |
| MPI + CUDA | `MPI`, `PencilArrays`, `PencilFFTs`, `CUDA`, `GPUArrays`, `GPUArraysCore`, `KernelAbstractions` |
| MPI + AMDGPU | `MPI`, `PencilArrays`, `PencilFFTs`, `AMDGPU`, `GPUArrays`, `GPUArraysCore`, `KernelAbstractions` |
| MPI + AD | `ChainRulesCore`, `MPI`, `PencilArrays`, `PencilFFTs` |
| LoopVectorization | `LoopVectorization` |
| ForwardDiff | `ForwardDiff` |
| Zygote | `Zygote` |
| Advanced AD | `ChainRulesCore` |

Hardware-specific types such as the CUDA extension's `CuFFTPlan` live in their
extension module and are documented in [GPU Acceleration](../gpu.md). See
[Package Architecture](../architecture.md) for the full extension map and
shared helper files.

## Public API

```@autodocs
Modules = [SHTnsKit]
Private = false
Order = [:module, :constant, :type, :function, :macro]
```

## Configuration lifecycle

`SHTConfig` is managed by Julia's garbage collector. `destroy_config(cfg)` is a
no-op retained for API symmetry; callers do not need a `try`/`finally` teardown
pattern.

## Related guides

- [Quick Start](../quickstart.md)
- [Grid Types](../grids.md)
- [Normalization and Phase](../norms.md)
- [Advanced Usage](../advanced.md)
- [Distributed Computing](../distributed.md)
