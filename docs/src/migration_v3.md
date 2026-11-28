# Migrating to SHTnsKit v3.0

SHTnsKit 3.0 adds the Apple Metal backend and broadens GPU and distributed
support. It also corrects several results that 2.0 got wrong and rejects some
inputs that 2.0 processed silently. Most code runs unchanged. Check the lists
below if you use vorticity loss gradients, `phi_scale = :quad`, the
`create_config` defaults, rotations on truncated layouts, or mixed precision.

## Changed results

- `grad_loss_vorticity_Tlm` and `loss_and_grad_vorticity_Tlm` return the
  Euclidean gradient, so entries with `m > 0` are twice the 2.0 values. They
  now agree with `grad_enstrophy_Tlm`, the `grad_energy_*` functions and Zygote.
- With `phi_scale = :quad`, `analysis` inverts `synthesis`, where 2.0 returned
  `alm / 2π`. The point and latitude evaluators carry the same φ factor as the
  grid they sample, and energies are scaled by 1/(2π)² relative to `:dft`. An
  unset `phi_scale` now means `:dft` on every grid.
- `create_config` defaults to `nlat = 2lmax + 1` on equiangular grids and
  `2(lmax + 1)` on Driscoll–Healy grids, instead of `lmax + 2`, whose round
  trips were 7–22% off. `create_regular_config` still defaults to `lmax + 2`.
- QST transforms compute all three components at the precision promoted across
  them. A `Float32` radial field with `Float64` tangential fields gives three
  `ComplexF64` spectra.
- On CUDA and ROCm, in-place scalar `analysis!` and `synthesis!` compute at the
  wider of the input and output precision.
- With `robert_form = true`, `grid_energy_vector` expects the sin(θ)-weighted
  components that `synthesis_sphtor` returns.

## Inputs that now raise

- Grids reject an `nlat` below the minimum for their `lmax`. On Gauss grids
  that means `lmax ≥ nlat` raises.
- Setting `cfg.lmax`, `cfg.mmax` or `cfg.mres` validates the whole layout before
  storing it, including `nlon ≥ 2mmax + 1`.
- Rotations that mix orders, with β other than 0 or π, raise on layouts with
  `mmax < lmax`, serial and distributed. 2.0 silently dropped the orders above
  `mmax`.
- Robert-form `grid_energy_vector` raises on grids with nonzero-weight pole
  nodes when the spectrum includes `m = 1`.
- Planned and in-place CPU transforms reject arrays whose axes do not start at
  1, and planned vector transforms reject one array passed as both outputs.
- CUDA's `gpu_apply_laplacian!` rejects a batch of fields, as ROCm already did.
- In-place GPU transforms raise [`BackendUnavailableError`](@ref) for device
  arrays in deeper wrappers. GPU storage never reaches the CPU loops.
- Reverse-mode AD on `PencilArray` inputs raises `ArgumentError` where 2.0
  returned a wrong adjoint, and l-split spectral batches raise on every rank.

## New in 3.0

- An Apple Metal backend for single-precision transforms. See
  [GPU Acceleration](gpu.md).
- Allocating transforms accept deeper device wrappers, such as a view of a
  `PermutedDimsArray`. CUDA and ROCm add configuration-level in-place
  transforms, and run on the GPU that owns an allocation.
- `DistTransposePlan` transforms run on CUDA- and ROCm-backed pencils, with or
  without GPU-aware MPI.
- `zgrad_rotation_angles_real` and `zgrad_rotation_angles_cplx` take a `loss`
  keyword, so they can differentiate objectives such as alignment with a target.
