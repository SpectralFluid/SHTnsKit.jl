# SHTnsKit.jl Troubleshooting

This guide helps diagnose and fix common issues when using SHTnsKit.jl (pure Julia spherical harmonic transforms).

## Quick Check

```julia
using SHTnsKit

cfg = create_gauss_config(8, 8)
@show get_lmax(cfg), cfg.nlat, cfg.nlon

# Create bandlimited test coefficients (prevents high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
spatial = synthesis(cfg, sh)
recovered = analysis(cfg, spatial)

println("round-trip error = ", norm(sh - recovered) / norm(sh))

destroy_config(cfg)
```

If this runs and prints a small round-trip error (e.g., < 1e-10), your setup is fine.

## Common Errors

- DimensionMismatch: spatial_data size (X, Y) must be (nlat, nphi)
  - Ensure `length(sh) == cfg.nlm` and `size(spatial) == (cfg.nlat, cfg.nlon)`.

- BoundsError on (l, m) indexing
  - Use `lmidx(cfg, l, m)` and `lm_from_index(cfg, idx)`; only m ≥ 0 are stored for real basis.

- Invalid grid sizes for transforms
  - Gauss grid: `nlat > lmax` and `nphi ≥ 2*mmax + 1`
  - Regular grid: `nlat ≥ 2*lmax + 1` and `nphi ≥ 2*mmax + 1`

- Large numerical error in analysis/synthesis
  - Verify grid constraints above. Prefer Gauss grids for accuracy. Consider higher precision (`T=Float64` default).

## Performance Tips

- Enable threading and set FFTW threads
  ```julia
  summary = set_optimal_threads!()  # (threads=…, fft_threads=…)
  println(summary)
  # or fine-tune
  set_threading!(true)
  set_fft_threads(4)
  ```

- Reuse allocations with in-place APIs
  ```julia
  sh = allocate_spectral(cfg)
  spatial = allocate_spatial(cfg)
  rand!(sh)
  synthesize!(cfg, sh, spatial)
  analyze!(cfg, spatial, sh)
  ```

- Avoid allocations inside hot loops
  - Keep buffers outside loops; don’t recreate configs or arrays repeatedly.

- Prefer Gauss-Legendre grids for fewer points at similar accuracy
  - `cfg = create_gauss_config(lmax, lmax)`

## Numerical Validation Patterns

- Round-trip test (spatial → spectral → spatial)
  ```julia
  # Create bandlimited spatial field (smooth test function)
θ, φ = cfg.θ, cfg.φ
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    spatial[i,j] = 1.0 + 0.5 * cos(θ[i]) + 0.3 * sin(θ[i]) * cos(φ[j])
end
  err = transform_roundtrip_error(cfg, spatial)
  println("max abs error = ", err)
  ```

- Power spectrum sanity
  ```julia
  sh = analysis(cfg, spatial)
  p = power_spectrum(cfg, sh)
  println("total power = ", sum(p))
  ```

- Coordinate utilities
  ```julia
  θ, φ = SHTnsKit.create_coordinate_matrices(cfg)
  # or access single coords
  t1 = get_theta(cfg, 1); p1 = get_phi(cfg, 1)
  ```

## MPI and Distributed Computing Issues

### Package Version Requirements

SHTnsKit.jl's distributed extension requires:
- **MPI.jl**: v0.20 (uses `Allgatherv!` with `VBuffer` API)
- **PencilArrays.jl**: v0.19 (uses `range_local`, `size_local`, `get_comm` API)
- **PencilFFTs.jl**: v0.15

Check your versions:
```julia
using Pkg
Pkg.status(["MPI", "PencilArrays", "PencilFFTs"])
```

### Common MPI Errors

- `MethodError: no method matching communicator(::Pencil{...})`
  - **Cause**: The environment resolved PencilArrays outside the supported v0.19 range
  - **Fix**: Run `Pkg.resolve()` in the project environment
  - The supported API uses `get_comm(pen)`

- `MethodError: no method matching globalindices(::PencilArray{...})`
  - **Cause**: The same dependency mismatch
  - **Fix**: Resolve PencilArrays v0.19; its local ranges come from `range_local(pen)`

- `Permission denied @ mkdir_pid_file` or random `InexactError`
  - **Cause**: Multiple MPI processes competing for precompilation cache
  - **Fix**: Use a fresh depot or precompile in serial first:
    ```bash
    # Option 1: Fresh depot
    JULIA_DEPOT_PATH=/tmp/fresh_depot:$HOME/.julia mpiexec -n 4 julia script.jl

    # Option 2: Precompile first in serial
    julia --project -e 'using SHTnsKit, MPI, PencilArrays, PencilFFTs'
    mpiexec -n 4 julia --project script.jl
    ```

### Distributed Transform Usage

Correct pattern for PencilArrays v0.19:
```julia
using MPI, PencilArrays, SHTnsKit

MPI.Init()
cfg = create_gauss_config(32, 48; nlon=96)

# Create Pencil and PencilArray
pen = Pencil((cfg.nlat, cfg.nlon), (1,), MPI.COMM_WORLD)
fθφ_local = zeros(Float64, PencilArrays.size_local(pen)...)
fθφ = PencilArray(pen, fθφ_local)

# Get local ranges for global-to-local mapping
ranges = PencilArrays.range_local(pen)
θ_range, φ_range = ranges[1], ranges[2]

# Fill local data using enumerated ranges
for (i_local, i_global) in enumerate(θ_range)
    for (j_local, j_global) in enumerate(φ_range)
        fθφ[i_local, j_local] = sin(cfg.θ[i_global]) * cos(cfg.φ[j_global])
    end
end

# Distributed transforms
Alm = SHTnsKit.dist_analysis(cfg, fθφ; use_rfft=true)
fθφ_recovered = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true, use_rfft=true)

MPI.Finalize()
```

### Running MPI Tests

```bash
# Run the built-in MPI test suite
JULIA_DEPOT_PATH=/tmp/test_depot:$HOME/.julia mpiexec -n 2 julia --project test/test_mpi_pencil.jl
```

## Profiling and Benchmarking

- Quick timing
  ```julia
  @time synthesis(cfg, sh)
  @time analysis(cfg, spatial)
  ```

- Accurate benchmarking
  ```julia
  using BenchmarkTools
  @btime synthesize($cfg, $sh)
  @btime analyze($cfg, $spatial)
  ```

- Allocation checks
  - Use `@btime` output to monitor allocations; switch to in-place APIs if needed.

## Minimal Reproducer Template

```julia
using SHTnsKit

function reproduce()
    cfg = create_gauss_config(16, 16)
    try
        # Create bandlimited test coefficients (prevents high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
        spatial = synthesis(cfg, sh)
        rec = analysis(cfg, spatial)
        return norm(sh - rec) / max(norm(sh), eps())
    finally
        destroy_config(cfg)
    end
end

println("relative error = ", reproduce())
```

## Environment Info

```julia
using InteractiveUtils
versioninfo()  # Include this in bug reports
```

## Getting Help

- File an issue with the minimal reproducer and `versioninfo()` output:
  https://github.com/SpectralFluid/SHTnsKit.jl/issues
