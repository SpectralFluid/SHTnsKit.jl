# Changelog

## Unreleased (v2.0.0)

### Breaking changes

**Legacy configuration and device compatibility APIs were removed.** The
package now uses `create_config`, `create_gauss_config`, and
`create_regular_config` directly; the C-style configuration/memory helpers and
integer `SHT_*` flags are gone. Device selection accepts only typed `CPU()` and
`GPU()` values. The duplicate backend/config-device APIs, symbol device inputs,
and multi-GPU orchestration were also removed. Single-device CUDA transforms
remain available.

**The distributed extension now targets the declared dependency versions.**
PencilArrays 0.19's `get_comm` and `range_local` APIs are used directly. The
runtime version probes, forwarding cache-blocked/fused analysis names, and
ignored scalar-plan keywords were removed; use `dist_analysis` or
`DistAnalysisPlan(cfg, prototype; use_rfft)`.

**`dist_synthesis!` / `dist_synthesis_sphtor!` reject `real_output=false` with real output arrays.**
`real_output=false` used to return the *real* field wrapped as complex (a real
buffer re-typed), so writing it into a `PencilArray{Float64}` happened to work.
It now performs a genuine complex synthesis — summing the m ≥ 0 half without the
Hermitian mirror — which is a different function, not the same field in a wider
type. On a typical config the complex-path result has `|imag|` up to 1.34 and a
real part differing from the real field by 1.31, against a field magnitude of
2.96, so no tolerance check can bridge the two.

*Porting:* if you passed `real_output=false` with a real output array and wanted
the real field, pass `real_output=true`. If you want the true complex synthesis,
pass a complex output `PencilArray`. The error message states both.

**`analysis_axisym` / `analysis_axisym_l` now return different values.**
Both omitted the φ quadrature factor `cfg.cphi * nlon = 2π`, so every returned
coefficient was `1/2π` too small — they inverted neither `synthesis_axisym` nor
the m=0 column of the full `analysis`. They now agree with both. Anything that
compensated for the old scale downstream must drop that compensation.

**Z-axis and X-axis rotations now rotate in the documented direction; `SH_Zrotate`
and `SH_Xrotate90` return different values.**
Through v2.0.0 the package disagreed with itself about the sign of a rotation.
`SH_Zrotate` applied `R_lm = Q_lm · exp(+imα)` while the general Wigner engine
behind `shtns_rotation_apply_real` / `shtns_rotation_apply_cplx` — and therefore
`SH_Yrotate` — applied `exp(-imα)`, so `SH_Zrotate(cfg, Qlm, α, Rlm)` and
`shtns_rotation_apply_real` with `ZYZ(α, 0, 0)` were *different rotations* of the
same field. `SH_Xrotate90` had the matching defect in its Euler triple: it used
`ZYZ(π/2, π/2, -π/2)`, which is `Rx(-π/2)` — the inverse of the +90° rotation its
name promises.

Everything now uses the **active** convention: `R_lm = Q_lm · exp(-imα)`, meaning
the field is rotated by `+α` about the axis, `g(θ, φ) = f(θ, φ - α)`, so a feature
at longitude `φ₀` moves to `φ₀ + α`. This is pinned against three independent
references — a real φ shift on the FFT grid, the ZYZ engine, and the distributed
twins — and matches the old `exp(+imα)` result only at `α = 0` (at `lmax = 6` the
two differ by 1.46 in relative norm; `SH_Xrotate90`'s old and new results differ
by 1.34).

Affected: `SH_Zrotate`, `SH_Xrotate90`, and both distributed Z-rotations
(`dist_SH_Zrotate` in `src/parallel_dense.jl` and in the PencilArray extension,
which had followed `SH_Zrotate`'s old sign). `SH_Yrotate`, `SH_Yrotate90`,
`shtns_rotation_apply_real` and `shtns_rotation_apply_cplx` are **unchanged** —
they were already on the new convention, which is why the two disagreed.

*Porting:* to reproduce the old output, negate the angle —
`SH_Zrotate(cfg, Qlm, -α, Rlm)` and `dist_SH_Zrotate(cfg, Alm, -α, Rlm)`. For
`SH_Xrotate90`, apply the new one three times, or call
`shtns_rotation_apply_real` with `ZXZ(0, -π/2, 0)`. Code that mixed `SH_Zrotate`
with `SH_Yrotate` or the Euler-angle API was getting inconsistent results before
and needs no compensation now.

**Order-mixing rotations now reject `mmax < lmax` instead of truncating.**
A Wigner-d rotation through a general `β` couples `Y_l^m` to every `Y_l^{m'}` with
`|m'| ≤ l`. When storage stopped at `mmax < lmax`, the `|m'| > mmax` components
were silently dropped — measured at `lmax = 8`, that discarded **14.8 %** of the
field's energy at `mmax = 5` and **24.0 %** at `mmax = 3`, with no error. `SH_Yrotate`,
`SH_Yrotate90`, `SH_Xrotate90` and the Euler-angle API now raise an `ArgumentError`
on such a configuration. Pure Z-rotations are unaffected: `β ≡ 0` is diagonal and
`β ≡ π` is anti-diagonal (`m' = -m`), so both still work at any `mmax`.

*Porting:* use `mmax == lmax` for anything but a Z-rotation. Results that appeared
to work before were missing the truncated energy.

**`shtns_rotation_set_angle_axis` returned the wrong axis for exact half-turns.**
Its degenerate branch used the `β ≈ 0` formula at both poles. At `β ≈ π` the
observable Euler combination is `α - γ`, read off the *negated* first column, so a
180° turn about `x̂` came back as `ZYZ(0, π, 0)` — which is exactly `Ry(π)`. The
relative error against the true `Rx(π)` was 1.33, and against `Ry(π)` it was 0.
Only exact half-turns were affected; 90° and 179.99° were already correct.

**Point and latitude evaluators now honour `phi_scale`.**
`synthesis` scales its Fourier bins by `phi_inv_scale(cfg)` and the inverse FFT
divides by `nlon`, a net factor of 1 under `:dft` but `1/2π` under `:quad`.
`synthesis_point`, `synthesis_axisym`, `synthesis_axisym_l`, `SH_to_lat`,
`SH_to_lat_cplx`, `SHqst_to_point`, `SH_to_grad_point` and `SHqst_to_lat` applied
no factor at all, so under `:quad` every one of them disagreed with the grid it
claims to sample by exactly 2π. **Default `:dft` behaviour is unchanged.**

**Structural `SHTConfig` fields are no longer silently inconsistent.**
Assigning `cfg.lmax = 10` left `size(cfg.Nlm) == (7, 7)` while the transforms index
it as `(lmax+1, mmax+1)` under `@inbounds` — an out-of-bounds read of a live array.
`lmax`, `mmax` and `mres` now rebuild the derived spectral layout (`Nlm`, `nlm`,
`li`, `mi`, cached scale matrix and m-ordering) and drop the now-stale Legendre
tables; `nlat`, `nlon`, `grid_type`, `nlm`, `li`, `mi` and `nspat` raise an
`ArgumentError` pointing at the `create_*_config` constructors, because there is no
grid-type-independent way to regenerate the quadrature in place.

**The exported `SHTConfig(; ...)` keyword constructor validates its invariants.**
It previously checked nothing, so a hand-built configuration could violate
`nlon ≥ 2*mmax+1` and then silently synthesize an all-zero field for any mode it
could not resolve (and hand `use_rfft=true` a raw `BoundsError`). It now enforces
the same constraints the `create_*_config` helpers always have.

**Argument validation added where it was missing.** `analysis_sphtor_ml` /
`synthesis_sphtor_ml` returned `±Inf` coefficients for an out-of-range order or
truncation where their scalar twins already raised; `energy_scalar` /
`energy_vector` indexed under `@inbounds` with no size check and returned a
silently wrong number for a mis-sized spectrum. Both now raise.

### Fixed

- **QST transforms had no reverse-mode rule.** `synthesis_qst` and `analysis_qst`
  fell through to Zygote's source tracing and crashed inside FFTW. Both now have
  `rrule`s composed from the existing scalar and sphtor adjoints, verified against
  finite differences.
- **`rrule`s that declared fewer keyword arguments than their primal were skipped
  entirely** the moment a caller passed one, even at its default value — `analysis`
  accepted none, and `synthesis` / `synthesis_sphtor` omitted `use_rfft`. They now
  accept the primal's full keyword set; the adjoint is unchanged, since these pick
  a different FFT implementation of the same linear operator.
- **A GC finalizer called an MPI collective.** `DistributedSpectralPlan2D` attached
  a finalizer that ran `close`, which frees the plan's sub-communicators via
  `MPI_Comm_free` — a collective. Garbage collection is rank-local and
  nondeterministic, so ranks could enter it at divergent points. Cleanup is now
  explicit only; `close` documents the collective contract.
- **`im_from_lm` was bounded by `lmax` rather than `mmax`,** so on an `mmax < lmax`
  layout a past-the-end index resolved to an order the configuration does not store
  instead of raising. It now takes an optional `mmax` keyword (defaulting to `lmax`,
  preserving behaviour for full layouts).
- **The two vorticity inverse-problem gradients did not stride by `mres`,** unlike
  every other diagnostic. Latent only — `analysis` pre-zeros its output — but a
  reused gradient buffer would have produced entries for unrepresentable modes.

- **Silent precision loss in batch QST/sphtor transforms.** `analysis_qst_batch`,
  `_synthesis_qst_batch` and the sphtor batch pair derived their output element
  type from one input array instead of promoting across all of them, truncating
  double-precision components to a single-precision sibling's type (measured
  error 2.05e-8 instead of ~1e-17).
- **Invalid Driscoll-Healy configurations could be mislabeled.** Passing
  `use_dh_weights=true` without `include_poles=true` built midpoint nodes and
  ordinary weights but tagged the result `:driscoll_healy`; the invalid option
  combination is now rejected. Cloning and persistence preserve valid DH grids.
- **Pole-inclusive grids with `nlat == 1` produced an all-NaN config** (`π/0`)
  that returned NaN from every subsequent transform with no error raised. Now
  rejected with a message naming the cause.
- **`dist_SH_mul_mx!` crashed on every `mres > 1` config** — it walked all orders
  through `LM_index`, which requires multiples of `mres`.
- **`dist_SH_Yrotate` crashed on `mres > 1`.** A Y-rotation mixes orders and so
  cannot be represented in an `mres`-strided layout at all; it now says that
  up front instead of failing deep inside the rotation.
- **Device selection now uses typed `CPU()` / `GPU()` values consistently.**
  Selection, transfer, inspection, and CUDA-extension APIs previously mixed
  symbol values and two device type systems. Unsupported values now fail by
  dispatch.
- **Equivalent distributed pencils could deadlock topology detection.** A
  rank-local object-identity cache let one rank return while another entered an
  `Allreduce`. The non-planned path now performs one rank-symmetric reduction on
  every call; planned transforms retain their cached topology.
- **`copy(cfg)` defeated the Legendre-table memory reduction.** Copies now retain
  the `NP_tables === plm_tables` and `NdP_tables === dplm_tables` aliases while
  remaining independent of the source config.
- **ForwardDiff could not flow through any plan-based batch transform.**
  `SHTPlan` is FFTW-backed and cannot hold `ForwardDiff.Dual`; the batch entry
  points now route non-FFTW element types through the plan-free `cfg`-form
  transforms.
- **Cached FFT plans could silently fall back to an O(n²) DFT.** The plan cache
  keys on shape and strides but not alignment, so reuse on a differently-aligned
  matrix threw and callers quietly downgraded. Plans are now built `UNALIGNED`.
- Legendre south-pole and normalization-comment corrections carried over from the
  orthonormal refactor; eleven comments prescribed conversions the code no longer
  performs.

### Performance

- **Distributed analysis: 3 topology collectives per call → 1.** The
  `φ_is_local_all` / `θ_is_distributed` predicates share a single bitmask
  `Allreduce`. Planned transforms still compute and retain them at construction.
- **`dist_synthesis_packed_cplx` is single-pass**, down from two full distributed
  syntheses. The negative-m φ bins are filled in the same θ/m traversal as the
  positive ones, reusing one Legendre row per `(m, θ)`. It now matches the serial
  reference exactly.
- **Legendre table memory halved.** `prepare_plm_tables!` was building
  `NP_tables`/`NdP_tables` bit-for-bit identical to `plm_tables`/`dplm_tables`;
  they now alias. `estimate_table_memory` previously reported half the true
  figure, so jobs sized by it allocated twice their budget.
- Batch FFT helpers reuse the shared plan cache instead of re-planning per call.

### Internal

- `pack_lm!`/`pack_lm`/`unpack_lm!`/`unpack_lm` in `src/layout.jl` replace six
  open-coded copies of the packed↔dense `(l,m)` mapping. The `m % mres` guard had
  to be fixed three separate times across those copies.
