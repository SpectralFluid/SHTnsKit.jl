# Performance Tips

```@raw html
<div style="background: linear-gradient(135deg, #ea580c 0%, #f59e0b 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
    <h2 style="margin: 0 0 0.5rem 0; color: white; border: none;">Performance Tips</h2>
    <p style="margin: 0; opacity: 0.9;">Quick tips for reducing allocations and maximizing throughput</p>
</div>
```

This page summarizes practical tips to reduce allocations and improve locality and throughput in SHTnsKit.jl, especially for distributed (MPI + PencilArrays) use.

- Reuse plans: Construct `SHTPlan` (serial) and distributed plans (`DistAnalysisPlan`, `DistSphtorPlan`, `DistQstPlan`) once per size and reuse. Plans hold FFT plans and working buffers to avoid per-call allocations.

- Grid defaults: every constructor sets `phi_scale=:dft` (FFT scaling `nlon`), Gauss and regular alike. You can override globally with `ENV["SHTNSKIT_PHI_SCALE"]=dft|quad` or per-config by assigning `cfg.phi_scale` if you need the quadrature convention. `analysis` and `synthesis` stay mutual inverses in either mode; `:quad` only rescales the spatial field by 1/2π.

- Low-allocation serial recipe: preallocate FFT scratch and outputs.
  ```julia
  cfg = create_gauss_config(32, 34; nlon=129)
  fft_scratch = scratch_fft(cfg)
  alm = zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
  f   = randn(cfg.nlat, cfg.nlon)
  analysis!(cfg, alm, f; fft_scratch=fft_scratch)      # reuse alm + scratch
  f_out = scratch_spatial(cfg)
  synthesis!(cfg, f_out, alm; fft_scratch=fft_scratch) # reuse f_out + scratch
  ```

- Low-allocation distributed recipe: reuse plans with in-plan scratch.
  ```julia
  aplan = DistAnalysisPlan(cfg, proto; use_rfft=true)
  vplan = DistSphtorPlan(cfg, proto; use_rfft=true, with_spatial_scratch=true)
  splan = DistPlan(cfg, proto; use_rfft=true)
  dist_analysis!(aplan, Alm, fθφ)          # no per-call FFT allocs
  dist_synthesis_sphtor!(vplan, Vt, Vp, S, T; real_output=true)
  dist_synthesis!(splan, fθφ, Alm; real_output=true)
  ```
  `use_rfft` trims the spectral grid; `with_spatial_scratch` keeps a single complex (θ,φ) buffer inside the vector/QST plans so real outputs don’t allocate a fresh iFFT workspace each call.

- use_rfft (distributed plans): When available in your `PencilFFTs`, set `use_rfft=true` in distributed plans to cut the (θ,k) spectral memory and accelerate real-output paths. The code falls back to complex FFTs when real transforms are not supported.

- with_spatial_scratch (vector/QST): Enable `with_spatial_scratch=true` to keep a single complex (θ,φ) scratch in the plan. This removes per-call iFFT allocations for real outputs. Default remains off to minimize footprint.

- Precomputed Legendre tables: On fixed grids, call `enable_plm_tables!(cfg)` to precompute `plm_tables` and `dplm_tables`. They provide identical results to on-the-fly recurrences and usually reduce CPU cost.

- Threading inside rank: For large lmax, enable Julia threads and (optionally) FFTW threads. Use `set_optimal_threads!()` or tune with `set_threading!()` and `set_fft_threads()` to match your core layout.

- LoopVectorization: If available, `analysis_turbo`/`synthesis_turbo` and related helpers can accelerate inner loops. Guard with `using LoopVectorization`.

- Data locality by m: Keep Alm distributed by m throughout your pipeline to avoid dense gathers. The distributed plans in this package consume and produce m-sliced data to preserve cache locality.
