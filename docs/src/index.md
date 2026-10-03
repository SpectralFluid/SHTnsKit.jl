# SHTnsKit.jl

```@raw html
<div class="hero-banner">
  <h1>SHTnsKit.jl</h1>
  <p>Fast spherical harmonic transforms for Julia &mdash; scalars, vectors, and
  spectral operators on CPUs, GPUs, and MPI clusters.</p>
  <span class="hero-version">v2.0.3</span>
</div>
```

```@raw html
<div class="feature-grid">
  <div class="feature-card">
    <h3><span class="icon">&#x1F310;</span> Full Transform Suite</h3>
    <p>Scalar, tangential-vector, and three-component QST transforms with
    Gauss&ndash;Legendre and equiangular grids.</p>
  </div>
  <div class="feature-card">
    <h3><span class="icon">&#x26A1;</span> Multi-Backend</h3>
    <p>Same API on CPU, NVIDIA CUDA, and AMD ROCm GPUs. Drop-in
    acceleration without rewriting your code.</p>
  </div>
  <div class="feature-card">
    <h3><span class="icon">&#x1F4E1;</span> MPI Distribution</h3>
    <p>Distribute latitude bands across MPI ranks with PencilArrays for
    fields that exceed single-node memory.</p>
  </div>
  <div class="feature-card">
    <h3><span class="icon">&#x1F9EE;</span> Differentiable</h3>
    <p>ForwardDiff, Zygote, and ChainRules extensions make transforms
    compatible with Julia&rsquo;s AD ecosystem.</p>
  </div>
</div>
```

## Install

```julia
using Pkg
Pkg.add("SHTnsKit")
```

See [Installation](installation.md) only if you need a GPU, MPI, or help with
setup.

## First transform

Start with known band-limited coefficients so the roundtrip has an exact answer:

```@example home-roundtrip
using SHTnsKit

cfg = create_gauss_config(16, 18)
coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
coefficients[3, 1] = 1.0             # degree l=2, order m=0
coefficients[5, 3] = 0.25 - 0.1im   # degree l=4, order m=2

field = synthesis(cfg, coefficients)
recovered = analysis(cfg, field)

@assert maximum(abs, recovered - coefficients) < 1e-12
size(field)
```

Spatial fields use `(latitude, longitude)` order. Dense coefficients use
`(l + 1, m + 1)` indexing because Julia arrays start at one.

## Choose your path

| I want to... | Read... |
|:---|:---|
| understand arrays and transform directions | [Quick Start](quickstart.md) |
| choose the right spherical sampling | [Grid Types](grids.md) |
| adapt a working scientific recipe | [Examples Gallery](examples/index.md) |
| keep transforms on an NVIDIA or AMD GPU | [GPU Acceleration](gpu.md) |
| distribute fields across MPI ranks | [Distributed Computing](distributed.md) |
| make repeated transforms faster | [Performance Guide](performance.md) |
| use packed storage, operators, rotations, or AD | [Advanced Usage](advanced.md) |
| exchange coefficients with another library | [Normalization and Phase](norms.md) |
| understand source files and extension system | [Package Architecture](architecture.md) |

The [API Reference](api/index.md) lists the complete public surface. Most users
can begin with the default Gauss–Legendre grid and orthonormal convention.
