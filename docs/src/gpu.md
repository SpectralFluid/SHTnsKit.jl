# GPU Acceleration

SHTnsKit uses the array type to select NVIDIA CUDA, AMD ROCm, or Apple Metal execution.
Full-grid transforms return arrays on the same device, so a pipeline can stay resident
without repeated host transfers.

## Install a backend

```julia
using Pkg

# NVIDIA
Pkg.add(["CUDA", "GPUArrays", "GPUArraysCore", "KernelAbstractions"])

# AMD
Pkg.add(["AMDGPU", "GPUArrays", "GPUArraysCore", "KernelAbstractions"])

# Apple Silicon
Pkg.add("Metal")
```

Load `SHTnsKit` and the packages listed for your GPU in the same Julia session.

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

## Supported operations

| Operation | NVIDIA CUDA and AMD ROCm | Apple Metal |
|:---|:---|:---|
| Full-grid scalar, spheroidal/toroidal, and QST transforms, including batches | Supported | Supported |
| In-place scalar and tangential-vector transforms with `SHTPlan` | Supported | Supported |
| Packed scalar `LM` and signed `LM_cplx` transforms, including degree truncation | Supported | Supported |
| Degree-truncated (`_l`), single-order (`_ml`) and axisymmetric scalar, vector and QST transforms | Supported | Supported |
| Spectral Laplacian and conversions between divergence/vorticity and vector coefficients | Supported | Supported |
| Point and latitude evaluation, rotations | Supported | Unavailable |
| Distributed transforms on GPU-backed MPI pencils | Supported with GPU-aware MPI | Unavailable |

## Apple Metal

On Apple Silicon, load `Metal` 1.11 or later in the 1.x series. The SHTnsKit
v2.0.3 tag predates this extension. To use the backend from a checkout, follow
the development installation instructions in [Installation](installation.md).

```julia
using SHTnsKit, Metal
@assert Metal.functional()
@assert Base.get_extension(SHTnsKit, :SHTnsKitMetalExt) !== nothing

cfg = create_gauss_config(8, 10)
a = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
a[3, 2] = 0.5f0 - 0.25f0im
field = synthesis(cfg, MtlArray(a))
recovered = analysis(cfg, field)

@assert field isa MtlMatrix{Float32}
@assert recovered isa MtlMatrix{ComplexF32}
@assert maximum(abs, Array(recovered) - a) < 2f-6
```

Use `Float32` for real fields and `ComplexF32` for complex fields and
coefficients. The backend computes latitude tables once at the original grid
precision and stores them in shared Metal memory. The FFTs and spherical
harmonic sums run on the GPU. Dense real packed and tangential-vector/QST
batch analyses store only the nonnegative longitude frequencies. With even
longitude counts, real synthesis in these families also uses half-width storage.

For repeated real transforms, reuse an `SHTPlan(cfg; use_rfft=true)` and the
output arrays. On an odd longitude grid, analysis alone uses half-width buffers.
The first synthesis allocates a full-width buffer per component, and later
analyses reuse its leading columns.
For in-place scalar batches on an odd grid, pass full-width `ComplexF32`
scratch as `fft_batch` with `use_rfft=false` to reuse it in both directions.

Run the Metal parity suite from the root of a source checkout on an Apple GPU.

```sh
julia --project=test/gpu/metal -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=test/gpu/metal test/gpu/metal/runtests.jl
```

## Reuse a plan

Starting with the real `field` from either roundtrip above, allocate outputs
once. `similar` preserves the array's backend and the field sets the precision.

```julia
plan = SHTPlan(cfg; use_rfft=true)
coefficients = similar(field, complex(eltype(field)), (cfg.lmax + 1, cfg.mmax + 1))
recovered = similar(field)

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
reports placement. If several GPU backends are loaded, pass an existing device
array as the prototype when a transfer needs to select one vendor.

On CUDA and ROCm, transforms run on the physical GPU that owns the input or
prototype. Device inputs, outputs and scratch for a call must share that GPU.
The caller's active device is restored when the call finishes.

Vector and QST calls whose operands mix CPU and GPU storage copy the CPU
operands to that GPU and run there. Pass device arrays directly or wrapped
once, by a view, reshape or adjoint. Transforms that return new arrays also
accept deeper wrappers, such as a view of a `PermutedDimsArray`, and copy them
into a dense array on the same device first. In-place transforms write their
outputs where they are, so they raise `BackendUnavailableError` for deeper
wrappers.

CUDA and ROCm in-place scalar batches accept dense or strided device scratch
in `fft_batch`. Dense buffers go directly to the FFT plan. For a strided view,
the transform uses its cached dense FFT buffer and copies the Fourier bins
between the buffer and the view on the GPU.

## Troubleshooting

- Check `CUDA.functional()`, `AMDGPU.functional()`, or `Metal.functional()` for your backend.
- Keep inputs, outputs, scratch, and prototypes on one vendor.
- Load every package listed for the selected backend in [Installation](installation.md).
- Avoid host/device copies inside a time-stepping loop; transfer once, compute,
  then copy back only the final result.
- Device tables are cached per configuration and precision. The scalar table
  holds `nlat × (lmax+1) × (mmax+1)` values, 8.6 GB in Float64 at
  `nlat = lmax + 1 = mmax + 1 = 1024`. A later cache access drops entries
  whose configurations have been garbage-collected. To clear the selected
  backend's cache explicitly, call `gpu_clear_cache!(GPU(); prototype=field)`.

For multi-rank arrays, first establish the CPU workflow in [Distributed
Computing](distributed.md), then use CUDA- or ROCm-backed `PencilArray` storage
with a GPU-aware MPI stack.
