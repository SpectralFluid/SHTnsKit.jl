# Migrating to SHTnsKit v2.0

SHTnsKit 2.0 removes ambiguous compatibility layers and makes execution,
normalization, and output storage explicit. Standard CPU calls such as
`create_gauss_config`, `analysis`, and `synthesis` keep their familiar shape,
but applications using legacy flags, device symbols, distributed plans, or
axisymmetric analysis need the changes below.

## Configuration and devices

- Use Julia configuration constructors and keywords. C-style helpers and
  integer `SHT_*` flags are no longer public API.
- Device selection accepts typed `CPU()` and `GPU()` values. Symbols such as
  `:cpu` and `:gpu`, duplicate backend selectors, and multi-GPU orchestration
  helpers have been removed.
- `GPU()` is strict: an unavailable backend raises
  [`BackendUnavailableError`](@ref), while ambiguous adapter selection raises
  `ArgumentError` and requires a vendor-specific prototype array. It never
  silently moves an operation to the host. The CUDA-only `gpu_analysis_safe`
  and `gpu_synthesis_safe` wrappers remain the explicit compatibility path
  when automatic CPU fallback is desired.

```julia
cpu_coefficients = analysis(CPU(), cfg, host_field)
gpu_coefficients = analysis(GPU(), cfg, device_field)
```

Ordinary `analysis(cfg, array)` and `synthesis(cfg, array)` calls infer CPU,
CUDA, AMDGPU, or distributed execution from the array type.

## Normalization and phase

Choose conventions when creating the configuration:

```julia
cfg = create_gauss_config(
    64, 66;
    norm=:fourpi,
    cs_phase=false,
    real_norm=true,
)
```

Scalar, vector, QST, packed, batch, planned, GPU, and distributed transforms
honor `cfg.norm`, `cfg.cs_phase`, and `cfg.real_norm` at their public boundary.
Remove application-side compensation factors that duplicated these conversions.

`analysis_axisym` and `analysis_axisym_l` now include the missing longitude
integral. Their coefficients are therefore `2π` times the values returned by
older releases for the same field. Remove any downstream `2π` compensation.

## Distributed plans and synthesis output

Load `MPI`, `PencilArrays`, and `PencilFFTs` to activate the distributed
extension. Construct scalar analysis plans from the spatial prototype:

```julia
aplan = DistAnalysisPlan(cfg, spatial_pencil; use_rfft=true)
dist_analysis!(aplan, coefficients, spatial_pencil)
```

Runtime compatibility probes, aliases, and ignored scalar-plan keywords were
removed. Use the allocating `dist_analysis` API or the declared plan
constructors (`DistAnalysisPlan`, `DistPlan`, `DistSphtorPlan`, `DistQstPlan`).

`dist_synthesis!` and `dist_synthesis_sphtor!` now reject
`real_output=false` when their destination PencilArrays have real element type.
Pass `real_output=true` for real storage, or allocate complex output storage for
a genuinely complex synthesis.

## Earlier validation

Invalid configurations now fail at construction or dispatch instead of later in
a transform. In particular:

- Driscoll–Healy weights require `include_poles=true`.
- Pole-inclusive grids require at least two latitude points.
- Y rotations reject `mres > 1`, because a Y rotation mixes azimuthal orders
  that an `mres`-strided layout cannot represent.
