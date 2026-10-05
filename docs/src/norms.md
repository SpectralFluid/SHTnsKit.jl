# Normalization and Phase

SHTnsKit computes internally with orthonormal harmonics including the
Condon–Shortley phase. Public transforms convert coefficients to and from the
convention stored in [`SHTConfig`](@ref).

## Supported options

| Keyword | Values | Meaning |
|---|---|---|
| `norm` | `:orthonormal`, `:fourpi`, `:schmidt` | spherical-harmonic basis scaling |
| `cs_phase` | `true`, `false` | include or omit `(-1)^m` |
| `real_norm` | `true`, `false` | independent real-field `sqrt(2)` coefficient convention for `m > 0` |
| `robert_form` | `true`, `false` | vector-component Robert form |

The default is `norm=:orthonormal`, `cs_phase=true`, `real_norm=false`, and
`robert_form=false`.

```@example norms-config
using SHTnsKit

cfg = create_gauss_config(
    24, 26;
    norm=:fourpi,
    cs_phase=false,
    real_norm=true,
)
(cfg.norm, cfg.cs_phase, cfg.real_norm)
```

## Transform boundary

Scalar, vector, QST, packed, batch, fixed-degree/order, planned, GPU, and
distributed transforms all consume and return coefficients in the configured
convention. Diagnostics and spectral operators do the same.

```@example norms-roundtrip
using SHTnsKit

cfg = create_gauss_config(16, 18; norm=:schmidt, cs_phase=false)
alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
alm[5, 3] = 0.75 - 0.2im

field = synthesis(cfg, alm)
recovered = analysis(cfg, field)
@assert maximum(abs, recovered - alm) < 1e-12
nothing
```

This means application code should not multiply by normalization or phase
factors around public transforms. Configure both producer and consumer to the
same convention, or perform an explicit, independently specified conversion at
the data interchange boundary.

## Scale definitions

Relative to orthonormal harmonics, the implemented basis scale is:

- four-pi: `sqrt(4π)`
- Schmidt semi-normalized: `sqrt(4π / (2l + 1))`

The `real_norm` factor is independent of Schmidt scaling. Toggling
`cs_phase` changes the sign of odd-`m` basis functions and coefficients through
the corresponding `(-1)^m` factor.

## Axisymmetric analysis

`analysis_axisym` and `analysis_axisym_l` include the longitude integral even
though their input is a latitude vector. SHTnsKit 2.0 corrected the formerly
missing factor: results are `2π` times those returned by affected older
versions. Remove any manual factor that compensated for the old behavior.

## Differences from SHTns

SHTnsKit mirrors the grids, coefficient layouts and normalization options of
SHTns 3.7, with three deliberate differences in convention. Data moved between
the two libraries needs these conversions; they are exactly the ones SHTnsKit's
SHTns 3.7 parity fixtures apply.

| Quantity | Relation | SHTns → SHTnsKit |
|---|---|---|
| Toroidal coefficients `Tlm` (`*_sphtor`, `*_qst`) | opposite sign for the same field | `Tlm = -T_shtns` |
| Complex `LM_cplx` coefficients with odd `m < 0` | SHTns's complex transforms apply `(-1)^m` to negative orders | negate the odd negative orders |
| [`SH_mul_mx`](@ref) operator vectors | SHTnsKit stores how each `Y_l^m` couples to `Y_{l±1}^m`; SHTns stores what each `(l, m)` receives from `l ± 1` | see below |

The operator layouts coincide for the symmetric cosθ operator, so a
[`mul_ct_matrix`](@ref) array is the same in both libraries, but not for
[`st_dt_matrix`](@ref) or other non-symmetric operators. To use an SHTns
operator vector `mx_shtns` with `SH_mul_mx`:

```julia
mx = zeros(2 * cfg.nlm)
for k in 0:(cfg.nlm - 1)            # 0-based packed index of (l, m)
    l, m = cfg.li[k + 1], cfg.mi[k + 1]
    if l > m          # coupling of (l, m) to (l-1, m)
        mx[2k + 1] = mx_shtns[2LM_index(cfg.lmax, cfg.mres, l - 1, m) + 2]
    end
    if l < cfg.lmax   # coupling of (l, m) to (l+1, m)
        mx[2k + 2] = mx_shtns[2LM_index(cfg.lmax, cfg.mres, l + 1, m) + 1]
    end
end
```

## Interoperability checklist

- Confirm whether the other format stores a real-field `sqrt(2)` factor.
- Confirm whether the Condon–Shortley phase is present.
- Confirm whether coefficients are dense non-negative-`m`, real packed, or
  complex packed.
- Use the same `mres` interpretation.
- Exchanging data with SHTns: apply the conversions in
  [Differences from SHTns](@ref).
- Test one known `(l,m)` mode before converting a large dataset.
