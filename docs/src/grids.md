## Grid Types

SHTnsKit supports several latitude grids through its configuration constructors:

- Gauss (Gaussian quadrature): Exact for integrals up to degree `2*nlat-1`.
  Use `create_gauss_config`; suggested `nlat = lmax+1` and
  `nlon ≥ 2*mmax+1`.

- Regular equiangular without poles (reg_fast/reg_dct/quick_init):
  Midpoint latitudes `θ_i = (i+0.5)π/nlat`, weights `w_i = (π/nlat) sin θ_i`.
  Fast to set up and compatible with FFT-friendly sampling. Use
  `create_regular_config(...; include_poles=false)`.

- Regular equiangular with poles (reg_poles):
  `θ_i = i π/(nlat-1)` including poles, trapezoidal weights. Use
  `create_regular_config(...; include_poles=true)`.

- Driscoll–Healy (`grid_type = :driscoll_healy`):
  `θ_j = π j/nlat` with the Driscoll–Healy quadrature weights, requires
  `nlat = 2*(lmax+1)`. **Exact**, like Gauss.

!!! warning "Regular-grid quadrature is not exact"
    The `:regular` and `:regular_poles` weights above are a plain
    midpoint/trapezoidal rule against `sin θ`, which does **not** integrate
    products of associated Legendre functions exactly. `analysis ∘ synthesis` on
    those grids is therefore not an identity, and the error shrinks only
    algebraically with `nlat` — measured relative round-trip error at `lmax = 8`:

    | grid | `nlat = lmax+2` | `2.5(lmax+1)` |
    |---|---|---|
    | `:gauss` | 1e-15 | 1e-15 |
    | `:driscoll_healy` | — | 7e-16 (at `nlat = 2(lmax+1)`) |
    | `:regular` | 0.11 | 0.005 |
    | `:regular_poles` | 0.26 | 0.018 |

    Note `create_config`'s default `nlat = lmax+2` sits at the worst end of that
    range. Use `:gauss` (or `:driscoll_healy` if you need pole samples) whenever
    round-trip accuracy matters; reserve `:regular`/`:regular_poles` for
    image-like sampling where an approximate analysis is acceptable.

`create_config` provides a common entry point through its `grid_type` keyword.
For best numerical exactness, prefer Gauss. For image-like sampling, use a
regular grid with precomputed Legendre tables.
