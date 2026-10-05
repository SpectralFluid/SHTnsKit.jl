# Distributed Computing

Use MPI when a field, a batch, or its spectral coefficients no longer fit
comfortably on one process. For small single fields, serial plans or a GPU are
usually simpler and may be faster.

## Install

```julia
using Pkg
Pkg.add(["MPI", "PencilArrays", "PencilFFTs"])
```

MPI.jl can use its bundled MPI or a system installation. Configure the MPI.jl
backend first, then use its matching launcher.

## Scalar roundtrip

This complete program distributes latitude, transforms a band-limited field,
and checks the largest error across all ranks:

```julia
using MPI
MPI.Init()

using PencilArrays, PencilFFTs, SHTnsKit

comm = MPI.COMM_WORLD
cfg = create_gauss_config(64, 66; nlon=129)

# create_spatial_pencil decomposes latitude, the scalable spatial dimension.
pencil = create_spatial_pencil(cfg; comm)
field = PencilArray(
    pencil,
    zeros(Float64, PencilArrays.size_local(pencil)...),
)

local_field = parent(field)
ranges = PencilArrays.range_local(pencil)
for (i_local, i_global) in enumerate(ranges[1]),
    (j_local, j_global) in enumerate(ranges[2])
    x = cfg.x[i_global]
    local_field[i_local, j_local] = 1 + 0.25 * (3x^2 - 1)
end

coefficients = dist_analysis(cfg, field)
recovered = dist_synthesis(
    cfg, coefficients; prototype_θφ=field, real_output=true
)

local_error = maximum(abs, parent(recovered) - local_field)
global_error = MPI.Allreduce(local_error, MPI.MAX, comm)
global_error < 1e-10 || error("roundtrip error: $global_error")

MPI.Comm_rank(comm) == 0 && println("roundtrip error: $global_error")
MPI.Finalize()
```

Save it as `roundtrip_mpi.jl` and run:

```bash
mpiexec -n 2 julia --project=. roundtrip_mpi.jl
```

All ranks must enter distributed operations collectively and construct the
same configuration.

## Choose the spectral layout

SHTnsKit offers two useful layouts:

| Workflow | Spectral result | Choose it when… |
|:---|:---|:---|
| `dist_analysis` / `dist_synthesis` | dense coefficients replicated on every rank | global spectral access and simpler post-processing matter most |
| `DistTransposePlan` | coefficients distributed by order `m` | spectral memory or communication dominates at larger rank counts |

A transpose plan also batches a local third dimension such as radial levels:

```julia
plan = DistTransposePlan(cfg; comm=comm, nlev=8, use_rfft=true)
spatial = allocate_spatial(plan)
spectral = allocate_spectral(plan)

# Fill parent(spatial), then transform collectively.
dist_analysis!(plan, spectral, spatial)
dist_synthesis!(plan, spatial, spectral)
```

Use `dist_analysis_sphtor` / `dist_synthesis_sphtor` for tangential vectors and
the `dist_*_qst` family for three-component fields. Their planned counterparts
follow the same output-before-input convention as serial in-place transforms.

## Practical rules

- Decompose latitude with `create_spatial_pencil`; a longitude-split
  input requires expensive gathering.
- Pass a spatial prototype to allocating synthesis so the result uses the
  intended distribution and storage type.
- Reuse plans for repeated transforms and batch fields that share a grid.
- Measure the slowest rank after warmup, including the communication your real
  workflow performs.
- Keep distributed data distributed between operations; avoid gathering only
  to scatter again.
- Reverse-mode rules cover `analysis`, `synthesis` and their `sphtor`/`qst`
  forms on PencilArrays, and `zgrad_*` return gradients in the input's layout.
  A distributed output's cotangent is each rank's derivative for its own
  block, so the gradient is that of the sum of the ranks' losses.
- With reverse-mode AD, every rank must differentiate a loss that uses each
  distributed transform's output. The pullbacks communicate (for example to sum
  every rank's contribution to the gradient of replicated coefficients), and
  Zygote skips the pullback of an output that a rank's loss ignores, so a loss
  such as `rank == 0 ? sum(abs2, y) : 0.0` leaves the other ranks waiting
  forever. Let each rank's loss use its local block of distributed outputs
  (weight it by zero if it must not contribute, e.g. `0 * sum(abs2, parent(y))`),
  and compute any loss built from a replicated output identically on every rank.

Runnable repository examples:

```bash
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl
mpiexec -n 2 julia --project=. examples/parallel_fft_roundtrip.jl
```
