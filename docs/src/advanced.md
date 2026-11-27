# Advanced Usage

This page points to public building blocks beyond dense scalar transforms.
Reuse and batching are covered in the [Performance Guide](performance.md).

## Packed coefficient storage

Packed transforms use a length-`cfg.nlm` vector compatible with SHTns
real-field storage:

```@example advanced-packed
using SHTnsKit

cfg = create_gauss_config(24, 26)
dense = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
dense[4, 2] = 0.75 - 0.2im
field = synthesis(cfg, dense)

packed = analysis_packed(cfg, vec(field))
field_vector = synthesis_packed(cfg, packed)

@assert length(packed) == cfg.nlm
@assert maximum(abs, reshape(field_vector, size(field)) - field) < 1e-12
nothing
```

Use [`LM_index`](@ref), `cfg.li`, and `cfg.mi` to map packed modes. The
`*_packed_cplx` family stores both signs of `m` for complex fields and
requires `mres == 1`.

## Selective transforms

Applications that need only part of a spectrum can use the `_l` and `_ml`
families instead of materializing a full unrelated result:

- `analysis_packed_l` / `synthesis_packed_l`
- `analysis_packed_ml` / `synthesis_packed_ml`
- `analysis_sphtor_l` / `synthesis_sphtor_l`
- `analysis_qst_l` / `synthesis_qst_l`
- `synthesis_point`, `synthesis_point_cplx`, `SH_to_lat`, and `SHqst_to_point`

All use the convention stored in `cfg`.

For local point and latitude evaluations with `Float32` or `Float64` inputs,
the coordinates retain their precision and the result follows the coefficient
precision. `ComplexF32` coefficients at `Float64` coordinates (any float
literal, or `cfg.x[i]`) therefore use a `Float64` Legendre basis: about a third
slower on the CPU, and FP64 kernels on CUDA and AMDGPU. Pass `Float32`
coordinates to stay in single precision. Exact coordinates such as `0`, `1//2`
or `π` take the coefficient precision. Mixing `ComplexF32` and `ComplexF64` spectra in a QST or gradient
evaluation gives `Float64` components on CPU, CUDA, and AMDGPU.

## Vector operators

Spheroidal coefficients encode divergence and toroidal coefficients encode
vorticity:

```@example advanced-vector-operators
using SHTnsKit

cfg = create_gauss_config(24, 26)
S = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
T = zeros(ComplexF64, size(S))
S[5, 3] = 0.4im
T[6, 2] = -0.3

divergence = divergence_from_spheroidal(cfg, S)
vorticity = vorticity_from_toroidal(cfg, T)

@assert maximum(abs, spheroidal_from_divergence(cfg, divergence) - S) < 1e-12
@assert maximum(abs, toroidal_from_vorticity(cfg, vorticity) - T) < 1e-12
nothing
```

The [API Reference](api/index.md) lists gradient, Laplacian, and other spectral
operators.

## Diagnostics

Energy and spectrum helpers honor the configuration's coefficient convention:

```@example advanced-diagnostics
using SHTnsKit

cfg = create_gauss_config(24, 26; norm=:fourpi, cs_phase=false)
alm = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
alm[4, 2] = 0.5 + 0.25im

energy = energy_scalar(cfg, alm)
spectrum = energy_scalar_l_spectrum(cfg, alm)
@assert sum(spectrum) ≈ energy
@assert grid_energy_scalar(cfg, synthesis(cfg, alm)) ≈ energy
(energy=energy, active_degrees=count(!iszero, spectrum))
```

With `robert_form=true`, pass the weighted components returned by
`synthesis_sphtor` directly to `grid_energy_vector`. It accounts for the
`sin(θ)` weighting when computing physical energy. If the spectrum includes
`m=1`, this requires a grid with zero pole weights or no pole samples.

## Rotations

Packed Z rotations are allocation-free when outputs are supplied:

```@example advanced-rotation
using SHTnsKit

cfg = create_gauss_config(12, 14)
input = zeros(ComplexF64, cfg.nlm)
input[LM_index(cfg.lmax, cfg.mres, 3, 2) + 1] = 1
rotated = similar(input)
restored = similar(input)

SH_Zrotate(cfg, input, 0.3, rotated)
SH_Zrotate(cfg, rotated, -0.3, restored)
@assert maximum(abs, restored - input) < 1e-12
nothing
```

Y rotations require `mres == 1` because they mix azimuthal orders. Reuse an
[`SHTRotation`](@ref) for repeated arbitrary Euler rotations.

## Axisymmetric transforms

`analysis_axisym` and `synthesis_axisym` operate on a latitude vector for
`m = 0`. Analysis includes the full longitude integral; do not apply an extra
manual `2π` correction.

## Automatic differentiation

Load the matching optional dependency, then use the documented wrappers:

```julia
using ForwardDiff, SHTnsKit

cfg = create_gauss_config(8, 10)
field = zeros(cfg.nlat, cfg.nlon)
gradient = fdgrad_scalar_energy(cfg, field)
```

Zygote wrappers include `zgrad_scalar_energy`, `zgrad_vector_energy`, and
`zgrad_enstrophy_Tlm`. For real packed coefficients and `mres == 1`,
`zgrad_rotation_angles_real` differentiates half the squared norm of the
rotated coefficients with respect to the three Euler angles, using the
normalization and phase in `cfg`.
