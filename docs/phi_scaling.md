# φ-Scaling in SHTnsKit

## Overview

The `phi_scale` field in `SHTConfig` controls how the longitude (φ) dimension is scaled during FFT operations in spherical harmonic transforms. This is critical for ensuring correct round-trip accuracy.

## Scaling Modes

### `:dft` - DFT Scaling
- **Value**: `inv_scale = nlon`
- **Use cases**:
  - Gauss-Legendre grids (default)
  - Driscoll-Healy grids
- **Rationale**: These grids use exact quadrature in latitude with FFT-based longitude integration

### `:quad` - Quadrature Scaling
- **Value**: `inv_scale = nlon / (2π)`
- **Use cases**:
  - Regular/equiangular grids without special quadrature
  - Regular grids with poles (simple trapezoidal rule)
- **Rationale**: Adjusts for the φ integration measure `dφ` where ∫₀²ᵖ f dφ ≈ (2π/nlon) Σ f_j

### `:auto` - Unset
- **Behavior**: treated as `:dft`.
- Historically this keyed on `grid_type` and handed every non-Gauss grid `:quad`,
  so a regular grid built through the exported `SHTConfig(; ...)` keyword
  constructor (which defaulted to `:auto`) disagreed by 2π with the identical
  grid from `create_regular_config`, which sets `:dft` explicitly. Both
  constructors emit `:dft`, so an unset value now means `:dft` as well.

## Configuration

### In Code
```julia
# Every constructor emits :dft; there is no phi_scale keyword on them.
cfg = create_gauss_config(lmax, nlat)     # :dft
cfg = create_regular_config(lmax, nlat)   # :dft

# To opt into the quadrature convention, set it on the config:
cfg.phi_scale = :quad
```

!!! note
    An earlier version of this page showed `create_regular_config(lmax, nlat;
    phi_scale=:quad)` and claimed regular grids default to `:quad`. Neither was
    true: those constructors take no `phi_scale` keyword (that call raises) and
    both set `:dft`.

### Via Environment Variable
```bash
# Override for all grids
export SHTNSKIT_PHI_SCALE=dft
export SHTNSKIT_PHI_SCALE=quad
```

## Implementation Details

The scaling is applied in `phi_inv_scale(cfg)` which is called during synthesis:

```julia
function phi_inv_scale(cfg::SHTConfig)
    # 1. Check environment variable override
    mode = get(ENV, "SHTNSKIT_PHI_SCALE", "")
    if mode == "quad"
        return cfg.nlon / (2π)
    elseif mode == "dft"
        return cfg.nlon
    end

    # 2. Use config-specified mode
    if cfg.phi_scale === :quad
        return cfg.nlon / (2π)
    elseif cfg.phi_scale === :dft
        return cfg.nlon
    end

    # 3. Fall back to grid-type heuristic
    return cfg.grid_type == :gauss ? cfg.nlon : cfg.nlon / (2π)
end
```

## Why This Matters

Incorrect φ-scaling leads to round-trip errors:
- Analysis transforms spatial grid → spectral coefficients
- Synthesis transforms spectral coefficients → spatial grid
- Round-trip: `alm_rt = analysis(synthesis(alm))` should satisfy `alm_rt ≈ alm`

The φ-scaling factor must match the quadrature weight convention to ensure:
```
∫∫ f(θ,φ) Y_lm(θ,φ) sin(θ) dθ dφ ≈ Σᵢⱼ f[i,j] Y_lm[i,j] w[i] * φ_scale
```

## History

- Original: All grids used DFT scaling (`nlon`)
- Commit fc1d114: Regular grids changed to quadrature scaling (`nlon/(2π)`)
- Commit 2441db0: Formalized with auto-detection
- Current: Explicit `phi_scale` field for clarity and control


## Invariant

Whichever mode is selected, `analysis` and `synthesis` are mutual inverses:

```julia
analysis(cfg, synthesis(cfg, alm)) ≈ alm    # exact under :dft and :quad
```

`synthesis` scales its Fourier bins by `phi_inv_scale(cfg)` and the inverse FFT
divides by `nlon`, a net spatial factor `σ`; `analysis` carries `cphi/σ` so the
two cancel. `:quad` therefore changes the scale of the *spatial* field (by 1/2π)
without changing what a round trip returns. Point and latitude evaluators
(`synthesis_point`, `SH_to_lat`, `SHqst_to_lat`, …) apply the same `σ`, so they
always agree with the grid `synthesis` produces.

Before this was fixed, `analysis` ignored `phi_scale` entirely, so under `:quad`
a round trip returned `alm/2π` and every evaluator was 2π off from the grid.
