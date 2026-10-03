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

Use `use_rfft=true` for real scalar input and output. A plan owns mutable
workspace; create one per simultaneously executing task.

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

Preallocate FFT scratch and output buffers to avoid per-call allocations:

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

Tables trade memory for speed and produce identical results to on-the-fly
recurrence. Use `disable_plm_tables!(cfg)` to free the memory.

## Longitude FFT scaling

The default `phi_scale` is `:dft`. The environment variable
`SHTNSKIT_PHI_SCALE=dft|quad` is a low-level override. Use one setting
consistently across every transform in an application.

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
