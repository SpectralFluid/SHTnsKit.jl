# Installation

SHTnsKit 2.0 supports Julia 1.10 and later 1.x releases on Linux, macOS, and
Windows.
The core package is pure Julia; FFTW.jl is installed automatically.

## Core package

```julia
using Pkg
Pkg.add("SHTnsKit")
```

Verify the installation with a small band-limited roundtrip:

```@example installation-core
using SHTnsKit

cfg = create_gauss_config(8, 10)
coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
coefficients[3, 2] = 0.5 - 0.25im
field = synthesis(cfg, coefficients)
recovered = analysis(cfg, field)

@assert maximum(abs, recovered - coefficients) < 1e-12
(lmax=cfg.lmax, grid=size(field))
```

Configurations are managed by Julia's garbage collector; explicit teardown is
not required.

## Optional capabilities

Install only the row your application needs. Loading those packages activates
the matching SHTnsKit extension automatically.

| Capability | Packages |
|:---|:---|
| NVIDIA GPU | `CUDA`, `GPUArrays`, `GPUArraysCore`, `KernelAbstractions` |
| AMD GPU | `AMDGPU`, `GPUArrays`, `GPUArraysCore`, `KernelAbstractions` |
| MPI distributed | `MPI`, `PencilArrays`, `PencilFFTs` |
| SIMD helpers | `LoopVectorization` |
| Forward-mode AD | `ForwardDiff` |
| Reverse-mode AD | `Zygote` |

For example:

```julia
using Pkg
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

Continue with [GPU Acceleration](gpu.md) or [Distributed
Computing](distributed.md) for a minimal working program.

## Common setup problems

**A grid constructor rejects the size.** Gauss grids require
`nlat >= lmax + 1` and `nlon >= 2*mmax + 1`. See [Grid Types](grids.md) for
the equiangular-grid constraints.

**An optional method is missing.** Load every package in its extension row in
the same Julia session. For MPI, use:

```julia
using MPI, PencilArrays, PencilFFTs, SHTnsKit
```

**Older code no longer runs.** Version 2 removed ambiguous flags, device
symbols, and compatibility plan signatures. See [Migrating to SHTnsKit
v2.0](migration.md).
