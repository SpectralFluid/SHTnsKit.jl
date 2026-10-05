# GPU Acceleration

SHTnsKit uses the array type to select NVIDIA CUDA or AMD ROCm execution.
Transforms return arrays on the same device, so a pipeline can stay resident
without repeated host transfers.

## Install a backend

```julia
using Pkg

# NVIDIA
Pkg.add(["CUDA", "GPUArrays", "GPUArraysCore", "KernelAbstractions"])

# AMD
Pkg.add(["AMDGPU", "GPUArrays", "GPUArraysCore", "KernelAbstractions"])
```

Load `SHTnsKit`, the vendor package, and the three shared GPU packages in the
same Julia session.

## Device-resident roundtrip

For NVIDIA:

```julia
using SHTnsKit, CUDA, GPUArrays, GPUArraysCore, KernelAbstractions
CUDA.functional() || error("CUDA is not functional")

cfg = create_gauss_config(64, 66)
field = CUDA.rand(Float64, cfg.nlat, cfg.nlon)

coefficients = analysis(cfg, field)
recovered = synthesis(cfg, coefficients)

@assert coefficients isa CUDA.AnyCuArray
@assert recovered isa CUDA.AnyCuArray
```

For AMD, load `AMDGPU` instead and create the input with
`AMDGPU.ROCArray(rand(cfg.nlat, cfg.nlon))`. Scalar, vector, QST, packed, and
batch transform families follow the same storage-based dispatch.

## Reuse a plan

Repeated transforms should reuse workspace and device outputs:

```julia
plan = SHTPlan(cfg; use_rfft=true)
coefficients = CUDA.zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
recovered = CUDA.zeros(Float64, cfg.nlat, cfg.nlon)

analysis!(plan, coefficients, field)
synthesis!(plan, recovered, coefficients)
```

Output arguments come before inputs. Plans own mutable workspace, so use one
plan per concurrent task.

## Explicit placement

Ordinary calls infer the backend from their arrays. Use `analysis(CPU(), ...)`
or `analysis(GPU(), ...)` only when execution intent must be strict. `GPU()`
raises [`BackendUnavailableError`](@ref) rather than silently moving the whole
operation to the CPU.

`to_device` performs an intentional transfer and [`on_device`](@ref)
reports placement. If both CUDA and AMDGPU are loaded, pass an existing device
array as the prototype when a transfer needs to select one vendor.

## Troubleshooting

- Check `CUDA.functional()` or `AMDGPU.functional()` before allocating.
- Keep inputs, outputs, scratch, and prototypes on one vendor.
- Load every package listed for the selected backend in [Installation](installation.md).
- Avoid host/device copies inside a time-stepping loop; transfer once, compute,
  then copy back only the final result.
- Device tables are cached per configuration and precision. The scalar table
  holds `nlat × (lmax+1) × (mmax+1)` values, 8.6 GB in Float64 at
  `lmax = 1023`. The cache drops a configuration's tables after the
  configuration is garbage-collected, and `gpu_clear_cache!(GPU())` drops all
  of them.

For multi-rank arrays, first establish the CPU workflow in [Distributed
Computing](distributed.md), then use device-backed `PencilArray` storage with a
GPU-aware MPI stack.
