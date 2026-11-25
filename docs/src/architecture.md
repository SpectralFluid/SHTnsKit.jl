# Package Architecture

The core module implements CPU transforms. Optional Julia package extensions
add GPU backends, MPI, automatic differentiation, and SIMD kernels.

## Directory layout

```
SHTnsKit.jl/
  src/              Core module sources
  ext/              Package extensions (loaded when optional deps are present)
  docs/             Documenter.jl site you are reading now
  examples/         Runnable scripts (serial and MPI)
  benchmark/        Micro-benchmarks
  test/             Test suite
  Project.toml      Dependencies, extensions, compat bounds
```

## Core module (`src/`)

Every file is included from `SHTnsKit.jl` in dependency order.

### Configuration and grid setup

| File | Role |
|:---|:---|
| `config.jl` | [`SHTConfig`](@ref) struct, grid constructors (`create_gauss_config`, `create_regular_config`, `create_config`), padding control, south-pole-first mode |
| `legendre.jl` | Gauss-Legendre quadrature nodes and weights, associated Legendre polynomial recurrences |
| `normalization.jl` | Orthonormal, four-pi, and Schmidt semi-normalized basis scaling; Condon-Shortley phase; `real_norm` factor |
| `layout.jl` | Data layout conventions and memory organization |
| `devices.jl` | Typed `CPU()` / `GPU()` device markers for dispatch |

### Transform engine

| File | Role |
|:---|:---|
| `kernels.jl` | Inlined per-latitude Legendre accumulation kernels, shared by all transform families; table-lookup and on-the-fly variants |
| `core_transforms.jl` | `analysis` / `synthesis` for 2D scalar grids: FFT along longitude, then Legendre integration or summation along latitude |
| `transforms.jl` | Point evaluation (`synthesis_point`), packed-storage transforms, latitude-band and single-mode (`_l`, `_ml`) variants |
| `sphtor_transforms.jl` | Spheroidal/toroidal (tangential 2D) vector transforms: `analysis_sphtor`, `synthesis_sphtor`, gradient synthesis |
| `qst_transforms.jl` | Three-component QST vector transforms: `analysis_qst`, `synthesis_qst` |
| `plan.jl` | [`SHTPlan`](@ref) and its in-place transforms reuse FFT plans and scratch buffers |
| `batch_transforms.jl` | Batch (multi-field) scalar, vector, and QST transforms with a third field-index dimension |

### Utilities

| File | Role |
|:---|:---|
| `fftutils.jl` | FFT wrappers, scratch allocation (`scratch_fft`, `scratch_spatial`) |
| `buffer_utils.jl` | Common buffer allocation patterns for transforms |
| `mathutils.jl` | Small numerical helpers |
| `complex_packed.jl` | Complex-packed coefficient storage and indexing |
| `loop.jl` | Unified CPU/GPU loop abstraction (`@sht_loop`, `@sht_inside`) |
| `device_utils.jl` | Device queries (`get_device`, `to_device`, `on_device`) |
| `prettyprinting.jl` | Compact `show` methods for `SHTConfig`, `SHTPlan`, and `SHTRotation` |

### Operators and diagnostics

| File | Role |
|:---|:---|
| `operators.jl` | Spectral differential operators: Laplacian, gradient, divergence-spheroidal and vorticity-toroidal conversions |
| `rotations.jl` | Z/Y/X-axis rotations, Euler-angle `SHTRotation` objects, Wigner d-matrices |
| `energy_diagnostics.jl` | Scalar and vector energy, gradients of energy functionals |
| `spectral_diagnostics.jl` | Power spectra by degree `l` and order `m` |
| `vorticity_diagnostics.jl` | Vorticity, enstrophy, and their gradients |

### Other core files

| File | Role |
|:---|:---|
| `local.jl` | Point and latitude-circle evaluation of real and complex fields |
| `parallel_dense.jl` | Parallel dense matrix operations for CPU multi-threading |

## Extension system (`ext/`)

SHTnsKit uses Julia's package-extension mechanism. Each extension activates
automatically when its trigger packages are loaded in the same session. No
manual registration is required.

### Extension map

| Extension | Trigger packages | Purpose |
|:---|:---|:---|
| `SHTnsKitGPUExt` | CUDA, GPUArrays, GPUArraysCore, KernelAbstractions | NVIDIA GPU transforms |
| `SHTnsKitAMDGPUExt` | AMDGPU, GPUArrays, GPUArraysCore, KernelAbstractions | AMD GPU transforms |
| `SHTnsKitMetalExt` | Metal, KernelAbstractions | Single-precision transforms on Apple GPUs |
| `SHTnsKitParallelExt` | MPI, PencilArrays, PencilFFTs | MPI-distributed transforms and plans |
| `SHTnsKitParallelCUDAExt` | MPI, PencilArrays, PencilFFTs, CUDA, GPUArrays, GPUArraysCore, KernelAbstractions | GPU-backed distributed transforms (NVIDIA) |
| `SHTnsKitParallelAMDGPUExt` | MPI, PencilArrays, PencilFFTs, AMDGPU, GPUArrays, GPUArraysCore, KernelAbstractions | GPU-backed distributed transforms (AMD) |
| `SHTnsKitParallelADExt` | ChainRulesCore, MPI, PencilArrays, PencilFFTs | AD rules for distributed transforms |
| `SHTnsKitLoopVecExt` | LoopVectorization | SIMD-optimized Legendre loops (`analysis_turbo`, `synthesis_turbo`) |
| `SHTnsKitForwardDiffExt` | ForwardDiff | Forward-mode AD wrappers |
| `SHTnsKitZygoteExt` | Zygote | Reverse-mode AD wrappers |
| `SHTnsKitAdvancedADExt` | ChainRulesCore | ChainRules `rrule` definitions |

### Shared parallel helpers

The parallel extension is split across several files for maintainability:

| File | Role |
|:---|:---|
| `ParallelPlans.jl` | `DistAnalysisPlan`, `DistPlan`, `DistSphtorPlan`, `DistQstPlan`, `DistTransposePlan` |
| `ParallelTransforms.jl` | `dist_analysis`, `dist_synthesis` and their in-place variants |
| `ParallelTransposeTransforms.jl` | Transpose-based distributed SHT for m-distributed spectral storage |
| `ParallelRotationsPencil.jl` | Distributed rotation operations on PencilArrays |
| `ParallelOpsPencil.jl` | Distributed Laplacian, divergence, vorticity |
| `ParallelDiagnostics.jl` | Distributed energy and spectrum computation |
| `ParallelLocal.jl` | Distributed point and latitude-circle evaluation |
| `ParallelDispatch.jl` | Dispatch logic for distributed paths |

The CUDA, AMDGPU, and Metal extensions share kernels and cache helpers in
`GPUCommon.jl`. Distributed GPU code uses `ParallelGPUVendorFirewall.jl` to
dispatch to CUDA or AMDGPU.

### How device dispatch works

The core implements `analysis` and `synthesis` for CPU arrays. Each GPU
extension registers an adapter and adds methods for its own array types.
Passing a device array selects that backend. With host inputs and several
functional backends, pass a device array as the `prototype` to a typed
`GPU()` call to select one. [GPU Acceleration](gpu.md)

## Transform pipeline

Full-grid scalar transforms combine a longitude FFT with a latitude transform.

**Analysis (spatial to spectral):**
1. FFT along longitude: `f(theta, phi)` becomes Fourier modes `F_m(theta)`
2. Legendre integration along latitude: weighted sum of `F_m * P_l^m` yields `a_lm`

**Synthesis (spectral to spatial):**
1. Legendre summation along latitude: `a_lm * P_l^m` accumulates into `F_m(theta)`
2. Inverse FFT along longitude: Fourier modes become `f(theta, phi)`

The `kernels.jl` file contains the inlined per-latitude accumulation loops
shared by scalar, vector, and QST transforms. Table-lookup kernels use
precomputed `P_l^m` values; on-the-fly kernels recompute them via three-term
recurrence.

## Configuration lifecycle

`SHTConfig` is mutable and holds grid nodes, quadrature weights, normalization
factors, and optional Legendre tables. Precomputed tables can dominate memory
at high degree. Julia's garbage collector manages their lifetime.

Plans (`SHTPlan`, distributed plan types) hold mutable workspace including FFT
plans and scratch buffers. Create one plan per simultaneously executing task.
