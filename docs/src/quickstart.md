# Quick Start

A spherical-harmonic transform moves between two representations of the same
band-limited field:

| Representation | Shape | Use |
|:---|:---:|:---|
| Spatial field | `(cfg.nlat, cfg.nlon)` | values at latitude × longitude points |
| Dense coefficients | `(cfg.lmax + 1, cfg.mmax + 1)` | complex amplitudes indexed by `(l + 1, m + 1)` |

For real spatial fields, dense storage keeps non-negative `m`; entries with
`l < m` are unused.

## Scalar roundtrip

[`synthesis`](@ref) maps coefficients to a grid. [`analysis`](@ref) maps grid
values back to coefficients:

```@example quickstart-scalar
using SHTnsKit

lmax = 32
cfg = create_gauss_config(lmax, lmax + 2)

alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
alm[3, 1] = 1.0
alm[5, 3] = 0.25 - 0.1im

field = synthesis(cfg, alm)
alm_recovered = analysis(cfg, field)

@assert maximum(abs, alm_recovered - alm) < 1e-12
(spatial=size(field), spectral=size(alm_recovered))
```

Analyzing measured or arbitrary grid data computes its representable
band-limited projection. Such data will not generally reproduce every input
grid value after synthesis; test exact roundtrips with band-limited
coefficients as above.

## Choose a grid

Start with [`create_gauss_config`](@ref). It gives accurate quadrature with the
fewest latitude samples for most transform work. Use an equiangular grid only
when your input format or numerical method requires it.

The [Grid Types](grids.md) guide shows all four sampling patterns and their
constructor constraints.

## Continue by task

- Repeated or batched transforms: [Performance Guide](performance.md)
- Tangential vector fields and QST fields: [Examples Gallery](examples/index.md)
  and [Advanced Usage](advanced.md)
- For CUDA, ROCm, or Metal arrays, see [GPU Acceleration](gpu.md).
- MPI/PencilArrays: [Distributed Computing](distributed.md)
- Non-default coefficient conventions: [Normalization and Phase](norms.md)
- Complete signatures and in-place argument order: [API Reference](api/index.md)
