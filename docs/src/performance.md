# Performance Guide

Optimize only after a correct band-limited roundtrip passes at the intended
grid size. Resolution, batch size, and data movement usually matter more than
individual kernel timings.

## 1. Reuse a plan

For repeated transforms on one grid, [`SHTPlan`](@ref) reuses FFT plans and
workspace:

```@example performance-plan
using SHTnsKit

cfg = create_gauss_config(64, 66)
plan = SHTPlan(cfg; use_rfft=true)
field = rand(cfg.nlat, cfg.nlon)
coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
recovered = zeros(cfg.nlat, cfg.nlon)

analysis!(plan, coefficients, field)
synthesis!(plan, recovered, coefficients)
nothing
```

Use `use_rfft=true` for real scalar input and output. A CPU plan owns mutable
workspace, so use one per simultaneous transform. Build a new plan if you
raise `cfg.lmax` above the degree used at construction.

## 2. Batch matching fields

When fields share a configuration, put the field index in dimension three:

```@example performance-batch
using SHTnsKit

cfg = create_gauss_config(32, 34)
fields = rand(cfg.nlat, cfg.nlon, 8)
coefficients = analysis_batch(cfg, fields; use_rfft=true)
recovered = synthesis_batch(
    cfg, coefficients; real_output=true, use_rfft=true
)
size(recovered)
```

Vector and QST batch families use the same layout.

## 3. Keep data where it is processed

- On a GPU, transfer once and keep intermediate arrays on the device.
- Under MPI, keep fields in `PencilArray` storage and choose a distributed
  spectral layout when replicated coefficients become expensive.
- On a CPU, reuse output arrays rather than allocating inside a time loop.

See [GPU Acceleration](gpu.md) and [Distributed Computing](distributed.md) for
their minimal resident workflows.

## 4. Benchmark the whole operation

Use BenchmarkTools after one warmup call:

```julia
using BenchmarkTools
analysis!(plan, coefficients, field)  # warmup
@btime analysis!($plan, $coefficients, $field)
```

For GPU timings, synchronize and include transfers only if production transfers
on every iteration. For MPI, report the maximum elapsed time across ranks and
construct plans outside the timed region.

### Measure Metal buffer allocations

With the device arrays and plan from [GPU Acceleration](gpu.md), warm up the
same operation you want to measure. Metal.jl reports new GPU buffer bytes for
the call separately from allocations on the Julia heap.

```julia
synthesis!(plan, recovered, coefficients)
gpu_bytes = Metal.@allocated synthesis!(plan, recovered, coefficients)
```

On an Apple M5 with `lmax=63`, 72 latitudes, and 129 longitudes, repeated
in-place scalar and tangential-vector synthesis each allocated zero new Metal
buffer bytes with dense outputs. The plan retains its Fourier workspace
between calls. For odd-length scalar batches with caller-owned scratch, use
a full-width complex buffer and `use_rfft=false` to avoid allocating a
temporary for the inverse.

## If more tuning is needed

- Compare the default on-the-fly Legendre recurrence with
  `prepare_plm_tables!(cfg)` when memory allows.
- Start Julia with the desired thread count and avoid oversubscribing cores with
  Julia tasks, FFTW threads, and MPI ranks at the same time.
- Prefer the simplest grid that matches the data; [Grid Types](grids.md)
  explains the sampling trade-offs.

Measure end-to-end throughput after every change. A faster transform that adds
extra conversions, transfers, or gathers may make the application slower.

## Low-allocation serial recipe

Preallocate the FFT scratch and output arrays to reuse them across calls.

```@example performance-lowalloc
using SHTnsKit

cfg = create_gauss_config(32, 34; nlon=129)
fft_scratch = scratch_fft(cfg)
alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
f = randn(cfg.nlat, cfg.nlon)

analysis!(cfg, alm, f; fft_scratch=fft_scratch)
f_out = scratch_spatial(cfg)
synthesis!(cfg, f_out, alm; fft_scratch=fft_scratch)
nothing
```

## Low-allocation distributed recipe

Reuse distributed plans with in-plan scratch:

```julia
aplan = DistAnalysisPlan(cfg, proto; use_rfft=true)
vplan = DistSphtorPlan(cfg, proto; use_rfft=true, with_spatial_scratch=true)
splan = DistPlan(cfg, proto; use_rfft=true)

dist_analysis!(aplan, Alm, fθφ)
dist_synthesis_sphtor!(vplan, Vt, Vp, S, T; real_output=true)
dist_synthesis!(splan, fθφ, Alm; real_output=true)
```

`use_rfft` trims the spectral grid for real fields. `with_spatial_scratch`
keeps a single complex `(θ, φ)` buffer inside vector/QST plans so real outputs
avoid per-call iFFT allocations.

## Precomputed Legendre tables

On fixed grids, precompute associated Legendre values for faster transforms:

```julia
enable_plm_tables!(cfg)
```

The CPU value and derivative tables together store
`2 * cfg.nlat * (cfg.lmax + 1) * (cfg.mmax + 1)` values of type `Float64`.
Call `disable_plm_tables!(cfg)` to release the configuration's references.

## Longitude FFT scaling

The default `phi_scale` is `:dft`. The environment variable
`SHTNSKIT_PHI_SCALE=dft|quad` is a low-level override. Use one setting
consistently across every transform in an application. See
[Normalization and Phase](norms.md) for its effect on fields and diagnostics.

## LoopVectorization

When `LoopVectorization` is loaded, `analysis_turbo` and `synthesis_turbo`
accelerate inner Legendre loops with SIMD:

```julia
using LoopVectorization, SHTnsKit

cfg = create_gauss_config(64, 66)
field = rand(cfg.nlat, cfg.nlon)
alm = analysis_turbo(cfg, field)
```

## Threading

Start Julia with the desired thread count and tune FFTW separately with
`FFTW.set_num_threads`. Avoid oversubscribing cores with Julia tasks, FFTW
threads, and MPI ranks simultaneously.
