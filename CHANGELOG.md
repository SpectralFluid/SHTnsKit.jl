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

**`analysis` and the direct evaluators now honour `phi_scale`; an unset
`phi_scale` resolves to `:dft`.** Two halves of one convention had drifted apart.

*`analysis` ignored `phi_scale` entirely.* `synthesis` scales its Fourier bins by
`phi_inv_scale(cfg)`, but `analysis` applied a fixed `cfg.cphi`, so under `:quad`
the pair was not mutually inverse: `analysis(cfg, synthesis(cfg, alm))` came back
as `alm / 2π` exactly. Analysis now applies `cphi · nlon / phi_inv_scale(cfg)`,
which is `cphi` under `:dft` and restores the inverse property under `:quad`. The
same factor was threaded through the batch, complex-packed, planned, distributed
and adjoint analysis paths so they all agree.

*The direct evaluators applied no φ factor at all.* `synthesis_point`,
`synthesis_point_cplx`, `synthesis_axisym`, `synthesis_axisym_l`, `SH_to_lat`,
`SH_to_lat_cplx`, `SHqst_to_point`, `SH_to_grad_point`, `SHqst_to_lat` and the
PencilArray local evaluations each disagreed with the grid they claim to sample by
exactly 2π under `:quad`. They now carry the same `phi_inv_scale(cfg)/nlon` factor
`synthesis` does.

*An unset `phi_scale` (`:auto`) used to fall back to a grid-type guess* —
`grid_type == :gauss ? nlon : nlon/2π` — so a regular grid built through the
exported `SHTConfig(; …)` keyword constructor disagreed by 2π with the identical
grid from `create_regular_config`, which sets `:dft` explicitly. Every constructor
emits `:dft`, so an unset value now means `:dft` too.

**Default `:dft` behaviour is unchanged in all three cases**; only `:quad` and
hand-built `:auto` configurations move, and they move to the values that make
`analysis` and `synthesis` inverses.

**Order-mixing rotations now reject `mmax < lmax` instead of truncating.**
A Wigner-d rotation through a general `β` couples `Y_l^m` to every `Y_l^{m'}` with
`|m'| ≤ l`. When storage stopped at `mmax < lmax`, the `|m'| > mmax` components
were silently dropped — measured at `lmax = 8`, that discarded **14.8 %** of the
field's energy at `mmax = 5` and **24.0 %** at `mmax = 3`, with no error and no
warning. `SH_Yrotate`, `SH_Yrotate90`, `SH_Xrotate90` and the Euler-angle API now
raise an `ArgumentError` on such a configuration. Pure Z-rotations are unaffected:
`β ≡ 0` is diagonal and `β ≡ π` is anti-diagonal (`m' = -m`), so both still work
at any `mmax`.

*Porting:* use `mmax == lmax` for anything but a Z-rotation. Results that appeared
to work before were missing the truncated energy.

**Structural `SHTConfig` fields are no longer silently inconsistent.**
Assigning `cfg.lmax = 10` left `size(cfg.Nlm) == (7, 7)` while the transforms index
it as `(lmax+1, mmax+1)` under `@inbounds` — an out-of-bounds read of a live array.
`lmax`, `mmax` and `mres` now rebuild the derived spectral layout (`Nlm`, `nlm`,
`li`, `mi`, cached scale matrix and m-ordering) and drop the now-stale Legendre
tables; `nlat`, `nlon`, `grid_type`, `nlm`, `li`, `mi` and `nspat` raise an
`ArgumentError` pointing at the `create_*_config` constructors, because there is no
grid-type-independent way to regenerate the quadrature in place.

**The exported `SHTConfig(; …)` keyword constructor validates its invariants.**
It previously checked nothing, so a hand-built configuration could violate
`nlon ≥ 2*mmax+1` and then silently synthesize an all-zero field for any mode it
could not resolve, or hand `use_rfft=true` a raw `BoundsError`. It now enforces the
same constraints the `create_*_config` helpers always have. The exported keyword
signature is otherwise unchanged.

### Fixed

- **`analysis_turbo` / `synthesis_turbo` ignored `mres`.** Both walked a bare
  `0:mmax` instead of `0:mres:mmax`, so `analysis_turbo` populated — and
  `synthesis_turbo` consumed — coefficient columns an `mres > 1` transform has no
  storage for. The disagreement with `analysis`/`synthesis` was O(1), not
  roundoff. Both now share the core's cached `m` ordering. The turbo pair also
  nested `@threads :static`, which is illegal inside an outer threaded region;
  they now fall back to a serial loop there, using the same predicate the core
  orchestrators use.
- **Rotation pullbacks read coefficients the primal had already overwritten.**
  `SH_Yrotate`, `shtns_rotation_apply_cplx` and `shtns_rotation_apply_real`
  captured their primal *input* and read it lazily, so an in-place call
  (`Rlm === Qlm`) — or any caller reusing the buffer before the pullback ran —
  silently corrupted the angle gradient. Each rule now snapshots what it needs at
  primal time. Applies to both the ChainRules and Zygote adjoints.
- **`rrule`s for `analysis`/`synthesis`/`analysis_sphtor`/`synthesis_sphtor`
  declared fewer keyword arguments than their primals.** Passing `use_rfft` or
  `fft_scratch` — even at its default — made ChainRules skip the rule entirely and
  fall through to source tracing. The keywords select a different FFT
  implementation of the same linear operator, so the adjoints are unchanged.
- **`synthesis_qst` and `analysis_qst` had no `rrule` at all**, so
  differentiating a QST pipeline fell through to Zygote's source tracing and
  crashed inside FFTW. Each adjoint is the existing scalar and sphtor adjoints
  side by side.
- **`shtns_rotation_apply_real` reported a bare size mismatch for `mres > 1`
  configurations**, leaving the caller to reverse-engineer why. The message now
  names `mres` and states the restriction, matching `dist_SH_Yrotate`.
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

- **Removed the dead parallel FFT-plan cache.** The `SHTNSKIT_CACHE_PENCILFFTS`
  environment variable and the `fft_plan_cache_enabled` / `set_fft_plan_cache!` /
  `enable_fft_plan_cache!` / `disable_fft_plan_cache!` controls forwarded to a
  cache in the parallel extension whose only reader, `_get_or_plan`, had no call
  sites — the "plans" it stored were `NamedTuple` placeholders the FFT wrappers
  ignored, so every knob was a no-op. The cache the transforms actually use is now
  in `src/fftutils.jl`, shared by the serial and distributed paths, and the same
  four controls address it without requiring the extension to be loaded.
  `SHTNSKIT_FFT_PLAN_CACHE` is the current spelling; the old name still works.
- **`DistributedSpectralPlan2D` no longer attaches a finalizer.** `close` frees
  the plan's `l_comm` / `m_comm` sub-communicators, and `MPI_Comm_free` is
  collective; a finalizer runs at whatever point that rank's garbage collector
  fires, which is rank-local and nondeterministic. Cleanup is explicit only — call
  `close(plan)` collectively. Leaking two communicators until `MPI_Finalize` is
  strictly better than a nondeterministic collective.
- **`spatial_view(cfg, A)` is exported**, the missing bridge in the padding API:
  `allocate_padded_spatial` returns an array with `nlat_padded ≥ nlat` rows while
  every transform requires exactly `nlat`, so the padded buffer could not be passed
  to `analysis` at all. The view keeps the padded column stride, so it preserves
  what the padding is for.
- **`set_batch_size!` is documented as advisory.** `howmany` / `spec_dist` mirror
  the SHTns C batch descriptors and are stored for interoperability, but the Julia
  batch entry points take the field count from `size(fields, 3)`; the old docstring
  claimed otherwise.
- **Regular-grid quadrature exactness is documented.** Fejér and Clenshaw–Curtis
  rules with `nlat` nodes are exact only through degree `nlat - 1`, and analysis
  integrates degree `2*lmax`, so the equiangular grids need `nlat ≥ 2*lmax + 1` —
  where Gauss–Legendre needs `nlat = lmax + 1`. Below that threshold nothing warns
  and `analysis ∘ synthesis` is not an identity (7.2e-2 relative error at
  `lmax = 8, nlat = 10`). Both `create_regular_config` and `docs/src/grids.md` now
  say so with measured numbers.
- New regression coverage: planned-vs-`cfg` conformance across the tables / rfft /
  Robert-form matrix; the `mmax < lmax` rotation guard; angle-gradient survival
  under an in-place primal; evaluator `phi_scale` agreement; turbo `mres`; and a
  4-rank `test_mpi_2d_alignment.jl` for the 2D spectral-plan alignment
  preconditions, wired into CI.

- `pack_lm!`/`pack_lm`/`unpack_lm!`/`unpack_lm` in `src/layout.jl` replace six
  open-coded copies of the packed↔dense `(l,m)` mapping. The `m % mres` guard had
  to be fixed three separate times across those copies.
