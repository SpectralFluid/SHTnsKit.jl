#=
================================================================================
SHTnsKit.jl - Fast Spherical Harmonic Transforms for Julia
================================================================================

A high-performance Julia package for computing Spherical Harmonic Transforms (SHT),
essential for applications in climate modeling, geophysics, astrophysics, and
computational physics on spherical domains.

WHAT ARE SPHERICAL HARMONICS?
-----------------------------
Spherical harmonics Y_l^m(θ,φ) are orthogonal basis functions on the sphere,
analogous to Fourier series on a circle. Any square-integrable function on
the sphere can be expanded as:

    f(θ,φ) = Σ_{l=0}^{lmax} Σ_{m=-l}^{l} a_lm * Y_l^m(θ,φ)

where:
- l is the degree (total wavenumber), l ≥ 0
- m is the order (azimuthal wavenumber), |m| ≤ l
- a_lm are the spherical harmonic coefficients
- θ is colatitude (0 at north pole, π at south pole)
- φ is longitude (0 to 2π)

KEY TRANSFORMS
--------------
- Analysis:  f(θ,φ) → a_lm   (spatial to spectral)
- Synthesis: a_lm → f(θ,φ)   (spectral to spatial)

The transforms use:
1. FFT along longitude (φ) for the azimuthal expansion
2. Legendre transform along latitude (θ) using Gauss-Legendre quadrature

PACKAGE STRUCTURE
-----------------
Core files:
- SHTnsKit.jl        : Main module, exports, and file includes
- config.jl          : SHTConfig struct and configuration functions
- legendre.jl        : Legendre polynomials and Gauss-Legendre quadrature
- kernels.jl         : Shared Legendre accumulation kernels (single source of truth)
- core_transforms.jl : Basic analysis/synthesis implementations
- plan.jl            : SHTPlan for optimized repeated transforms

Vector transforms:
- sphtor_transforms.jl : Spheroidal/toroidal (2D) vector field transforms
- qst_transforms.jl    : QST (3D) vector field transforms

Utilities:
- normalization.jl   : Spherical harmonic normalization conventions
- fftutils.jl        : FFT wrappers and utilities
- rotations.jl       : Rotation of spherical harmonic coefficients

Diagnostics:
- energy_diagnostics.jl    : Energy calculations and gradients
- spectral_diagnostics.jl  : Power spectrum analysis
- vorticity_diagnostics.jl : Vorticity and enstrophy

Extensions (loaded conditionally):
- ext/SHTnsKitParallelExt.jl : MPI-distributed transforms (requires MPI.jl, PencilArrays.jl)
- ext/SHTnsKitLoopVecExt.jl  : LoopVectorization optimizations

QUICK START
-----------
```julia
using SHTnsKit

# Create configuration for lmax=64 with 96 Gauss points
cfg = create_gauss_config(64, 96)

# Create test data on the grid
f = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    θ = acos(cfg.x[i])
    φ = 2π * (j-1) / cfg.nlon
    f[i,j] = cos(θ)  # Y_1^0 pattern
end

# Transform to spectral space
alm = analysis(cfg, f)

# Transform back to spatial space
f_reconstructed = synthesis(cfg, alm)

# Check roundtrip error
println("Max error: ", maximum(abs, f - f_reconstructed))
```

NORMALIZATION CONVENTIONS
-------------------------
Default: Orthonormal with Condon-Shortley phase
- cfg.norm = :orthonormal (default), :fourpi, :schmidt
- cfg.cs_phase = true (default) includes (-1)^m factor

COEFFICIENT STORAGE
-------------------
Dense format: Matrix of size (lmax+1, mmax+1)
- alm[l+1, m+1] for l ≥ m ≥ 0
- Only non-negative m stored (negative m via conjugate symmetry for real fields)

ENVIRONMENT VARIABLES
--------------------
- SHTNSKIT_PHI_SCALE: "dft" or "quad" for φ scaling convention
- SHTNSKIT_VERBOSE_STORAGE: "1" to print storage optimization info
- SHTNSKIT_FFT_PLAN_CACHE: "0" to disable φ-FFT plan caching (serial and distributed)
  (legacy alias: SHTNSKIT_CACHE_PENCILFFTS)
- SHTNSKIT_FFT_PLAN_CACHE_MAX: cap on distinct cached plans (default 64; ≤0 = no cap)

DEBUGGING TIPS
--------------
1. Check Gauss weights: sum(cfg.w) ≈ 2.0
2. Check grid points: cfg.x contains cos(θ) values, cfg.x[1] ≈ 1 (north pole vicinity)
3. Roundtrip test: f ≈ synthesis(cfg, analysis(cfg, f)) within machine precision
4. Coefficient check: alm[1,1] is the l=0,m=0 (mean) coefficient

================================================================================
=#

module SHTnsKit

# Import required standard libraries
using LinearAlgebra  # For linear algebra operations
using FFTW          # For Fast Fourier Transform operations
using Base.Threads  # For multi-threading support

# Runtime knob for inverse-FFT φ scaling during synthesis.
# Defaults: Gauss grids use "dft" (nlon), regular/equiangular use "quad" (nlon/(2π)).
# Override via cfg.phi_scale=:dft/:quad or ENV SHTNSKIT_PHI_SCALE=dft|quad.
phi_inv_scale(nlon::Integer) = (get(ENV, "SHTNSKIT_PHI_SCALE", "dft") == "quad" ? nlon/(2π) : Float64(nlon))

# Include all module source files
include("devices.jl")                        # Typed CPU()/GPU() device markers
include("loop.jl")                           # Unified CPU/GPU loop abstraction
include("fftutils.jl")                      # FFT utility functions and helpers
include("layout.jl")                        # Data layout and memory organization
include("mathutils.jl")                      # Mathematical utility functions
include("legendre.jl")                       # Legendre polynomials and Gauss-Legendre quadrature
include("normalization.jl")                  # Spherical harmonic normalization
include("config.jl")                         # Configuration and setup functions

# Config-aware φ scaling (requires SHTConfig definition)
function phi_inv_scale(cfg::SHTConfig)
    mode = get(ENV, "SHTNSKIT_PHI_SCALE", "")
    if mode == "quad"
        return cfg.nlon / (2π)
    elseif mode == "dft"
        return Float64(cfg.nlon)
    end
    if cfg.phi_scale === :quad
        return cfg.nlon / (2π)
    else
        # `:dft` and anything unset (`:auto`). The old fallback keyed on
        # `grid_type` and handed every non-Gauss grid `nlon/2π` — so a regular
        # grid built through the exported keyword constructor (which defaulted to
        # `:auto`) disagreed by 2π with the identical grid from
        # `create_regular_config`, which sets `:dft` explicitly. Both constructors
        # emit `:dft`, so `:dft` is the right default for an unset value.
        return Float64(cfg.nlon)
    end
end

"""
    _analysis_phi_scale(cfg) -> Float64

The φ quadrature factor `analysis` must apply so that it inverts `synthesis`.

`synthesis` scales its Fourier bins by `phi_inv_scale(cfg)` and the inverse FFT
divides by `nlon`, a net spatial factor of `σ = phi_inv_scale(cfg)/nlon`. For the
pair to be mutually inverse, analysis must carry `cphi/σ`.

Under the default `:dft` mode `σ = 1` and this is just `cphi = 2π/nlon`, exactly
what analysis always used — so nothing changes for any configuration the
`create_*_config` constructors produce. Under `:quad` (`σ = 1/2π`) the old fixed
`cphi` made `analysis(synthesis(alm))` come back as `alm/2π`: the two halves of
the transform pair simply disagreed about the convention, with synthesis honouring
`phi_scale` and analysis ignoring it.
"""
@inline _analysis_phi_scale(cfg::SHTConfig) = cfg.cphi * cfg.nlon / phi_inv_scale(cfg)

"""
    _evaluator_phi_scale(cfg) -> Float64

The factor a point/latitude evaluator must apply to reproduce the value
`synthesis` writes on the grid: `phi_inv_scale(cfg)/nlon`, i.e. 1 under `:dft`
and `1/2π` under `:quad`. Omitting it made every direct evaluator disagree with
the very grid it claims to sample by a factor of 2π.
"""
@inline _evaluator_phi_scale(cfg::SHTConfig) = phi_inv_scale(cfg) / cfg.nlon

"""
    _evaluator_phi_scale(cfg, ::Type{T}) -> T

`_evaluator_phi_scale` narrowed to the evaluator's own real type.

The evaluators promise their caller the element type their coefficients carry —
`Float32` coefficients give `Float32` values, and a `Dual` stays a `Dual`. The
untyped scale is a `Float64`, so multiplying by it silently widens every
`Float32` result to `Float64`. Convert once, at the boundary.
"""
@inline _evaluator_phi_scale(cfg::SHTConfig, ::Type{T}) where {T} =
    convert(real(T), _evaluator_phi_scale(cfg))

include("buffer_utils.jl")                   # Common buffer allocation patterns
include("kernels.jl")                       # Legendre accumulation kernels
include("plan.jl")                           # Transform planning and optimization
include("core_transforms.jl")                # Core 2D grid ↔ spectral transforms
include("transforms.jl")                     # Vector and point transforms
include("complex_packed.jl")                  # Complex number packing utilities
include("qst_transforms.jl")                  # QST (3D) vector field operations
include("sphtor_transforms.jl")               # Spheroidal/toroidal (2D) vector operations
include("operators.jl")                       # Differential operators on sphere
include("rotations.jl")                       # Spherical rotation operations
include("local.jl")                           # Local (thread-local) operations
include("energy_diagnostics.jl")              # Energy calculations and gradients
include("spectral_diagnostics.jl")            # Spectral analysis and spectrum functions
include("vorticity_diagnostics.jl")           # Vorticity and enstrophy calculations
include("batch_transforms.jl")                # Batch (multi-field) transforms
include("parallel_dense.jl")                  # Parallel dense matrix operations
include("device_utils.jl")                    # GPU device utilities and management
include("prettyprinting.jl")                  # Pretty-printing of public types

# ===== CORE CONFIGURATION AND SETUP =====
export SHTConfig, create_gauss_config, create_regular_config, create_config, destroy_config  # Configuration management
export create_gauss_fly_config, set_on_the_fly!, set_use_tables!, is_on_the_fly  # On-the-fly mode
export estimate_table_memory                                          # Memory estimation
export set_south_pole_first!, set_north_pole_first!, is_south_pole_first  # South pole first mode
export create_gauss_config_spf                                        # South pole first config creation
export set_allow_padding!, disable_padding!, is_padding_enabled       # Memory padding control
export get_nlat_padded, get_spat_dist, compute_optimal_padding        # Padding queries
export allocate_padded_spatial, allocate_padded_spatial_batch         # Padded array allocation
export copy_to_padded!, copy_from_padded!, estimate_padding_overhead  # Padding utilities
export spatial_view                                                   # Padded buffer → transform-shaped view

# ===== BASIC TRANSFORMS =====
# The `*_cplx` helpers are intentionally separate from `real_output=false`
# keyword calls. A literal helper gives Julia a compile-time output mode and
# keeps inference-sensitive callers from seeing a runtime Bool return union.
export analysis, synthesis, synthesis_cplx             # Basic forward/backward transforms
export SHTPlan, analysis!, synthesis!                  # Planned (optimized) transforms

# ===== BATCH TRANSFORMS =====
export set_batch_size!, get_batch_size, reset_batch_size!  # Batch configuration
export analysis_batch, analysis_batch!                  # Batch scalar analysis
export synthesis_batch, synthesis_batch!, synthesis_batch_cplx  # Batch scalar synthesis
export analysis_sphtor_batch, synthesis_sphtor_batch, synthesis_sphtor_batch_cplx  # Batch vector transforms
export analysis_qst_batch, synthesis_qst_batch, synthesis_qst_batch_cplx           # Batch 3D vector transforms

# ===== SPATIAL ↔ SPHERICAL HARMONIC TRANSFORMS =====
export analysis_sphtor!, synthesis_sphtor!              # In-place spheroidal/toroidal transforms
export analysis_packed, synthesis_packed, analysis_packed_l, synthesis_packed_l, analysis_packed_ml, synthesis_packed_ml, synthesis_point
export analysis_axisym, synthesis_axisym, analysis_axisym_l, synthesis_axisym_l

# ===== INDEXING AND COMPLEX NUMBER UTILITIES =====
export nlm_calc, nlm_cplx_calc, LM_index, LiM_index, build_li_mi, im_from_lm, LM_cplx_index, LM_cplx
export analysis_packed_cplx, synthesis_packed_cplx                        # Complex number transforms
export analysis_packed_cplx_l, synthesis_packed_cplx_l, synthesis_point_cplx
export fft_phi_backend

# ===== BUFFER HELPERS =====
export scratch_fft, scratch_spatial

# ===== VECTOR FIELD TRANSFORMS =====
export analysis_sphtor, synthesis_sphtor, synthesis_sph, synthesis_tor, synthesis_sph_cplx, synthesis_tor_cplx
export synthesis_grad, synthesis_grad_l, synthesis_grad_ml
export divergence_from_spheroidal, divergence_from_spheroidal!, spheroidal_from_divergence, spheroidal_from_divergence!
export vorticity_from_toroidal, vorticity_from_toroidal!, toroidal_from_vorticity, toroidal_from_vorticity!
export analysis_qst, synthesis_qst, analysis_qst_cplx, synthesis_qst_cplx  # Q,S,T decomposition
export analysis_qst!, synthesis_qst!                         # Planned Q,S,T transforms

# ===== LATITUDE-BAND AND M-MODE SPECIFIC TRANSFORMS =====
# `_l` variants truncate by degree; `_ml` variants operate on one Fourier mode.
# The complex helpers mirror the full-grid API convention above.
export synthesis_sphtor_l, analysis_sphtor_l, synthesis_sph_l, synthesis_tor_l
export synthesis_sphtor_l_cplx, synthesis_sph_l_cplx, synthesis_tor_l_cplx
export analysis_sphtor_ml, synthesis_sphtor_ml, synthesis_sph_ml, synthesis_tor_ml
export analysis_qst_l, synthesis_qst_l, synthesis_qst_l_cplx, analysis_qst_ml, synthesis_qst_ml
export synthesis_sphtor_cplx, analysis_sphtor_cplx
export suggest_pencil_grid, set_fft_plan_cache!, enable_fft_plan_cache!, disable_fft_plan_cache!, fft_plan_cache_enabled

# ===== MATRIX OPERATIONS AND DIFFERENTIAL OPERATORS =====
export mul_ct_matrix, st_dt_matrix, SH_mul_mx          # Matrix multiplication utilities
export SH_to_lat, SH_to_lat_cplx, SHqst_to_lat         # Latitude-specific transforms
export SHqst_to_point, SH_to_grad_point                 # Point-specific vector transforms

# ===== ROTATIONS =====
export SH_Zrotate                                       # Z-axis rotations
export SH_Yrotate, SH_Yrotate90, SH_Xrotate90         # Y and X axis rotations
export SHTRotation, shtns_rotation_create, shtns_rotation_destroy  # Rotation objects
export shtns_rotation_set_angles_ZYZ, shtns_rotation_set_angles_ZXZ
export shtns_rotation_wigner_d_matrix, shtns_rotation_apply_cplx, shtns_rotation_apply_real, shtns_rotation_set_angle_axis

# ===== ENERGY AND DIAGNOSTICS =====
export energy_scalar, energy_vector, enstrophy, vorticity_spectral, vorticity_grid
export grid_energy_scalar, grid_energy_vector, grid_enstrophy
export energy_scalar_l_spectrum, energy_scalar_m_spectrum    # Spectral energy analysis
export energy_vector_l_spectrum, energy_vector_m_spectrum
export enstrophy_l_spectrum, enstrophy_m_spectrum
export energy_scalar_lm, energy_vector_lm, enstrophy_lm

# ===== GRADIENT COMPUTATIONS =====
export grad_energy_scalar_alm, grad_energy_vector_Slm_Tlm, grad_enstrophy_Tlm
export grad_grid_energy_scalar_field, grad_grid_energy_vector_fields, grad_grid_enstrophy_zeta
export energy_scalar_packed, grad_energy_scalar_packed
export energy_vector_packed, grad_energy_vector_packed
export loss_vorticity_grid, grad_loss_vorticity_Tlm, loss_and_grad_vorticity_Tlm

# ===== PERFORMANCE OPTIMIZATIONS =====
export prepare_plm_tables!, enable_plm_tables!, disable_plm_tables!  # Precomputed Legendre tables

# ===== UNIFIED LOOP ABSTRACTION =====
export @sht_loop, @sht_inside                                        # Unified CPU/GPU loop macros
export loop_backend, set_loop_backend                                # Loop backend configuration
export CI, δ, inside                                                 # CartesianIndex utilities
export spectral_range, spatial_range, latitude_range, mode_range     # SHT-specific loop ranges
export local_range, local_size                                       # PencilArray-aware range helpers

# ===== DEVICE MANAGEMENT =====
export ComputeDevice, CPU, GPU, BackendUnavailableError
export get_device, set_device!, to_device, on_device

# ===== EXTENSION-PROVIDED FUNCTIONS =====
# These functions are implemented in Julia package extensions and only available when
# the corresponding packages are loaded

# GPU Computing functions (SHTnsKitGPUExt extension)
export gpu_analysis, gpu_synthesis, gpu_analysis_safe, gpu_synthesis_safe  # GPU transforms
export gpu_analysis_sphtor, gpu_synthesis_sphtor         # GPU vector transforms
export gpu_apply_laplacian!                              # GPU operators
export gpu_memory_info, check_gpu_memory, gpu_clear_cache!, estimate_memory_usage  # Memory management
export get_available_gpus, set_gpu_device                # CUDA device management

# Optional LoopVectorization-powered helpers (SHTnsKitLoopVecExt extension)
export analysis_turbo, synthesis_turbo                    # Vectorized transforms
export turbo_apply_laplacian!, benchmark_turbo_vs_simd    # Performance utilities

# Automatic Differentiation wrappers (AD extensions: Zygote, ForwardDiff)
export zgrad_scalar_energy, zgrad_vector_energy, zgrad_enstrophy_Tlm      # Zygote gradients
export fdgrad_scalar_energy, fdgrad_vector_energy                         # ForwardDiff gradients
export zgrad_rotation_angles_real, zgrad_rotation_angles_cplx             # Rotation gradients

# Distributed/Parallel computing functions (SHTnsKitParallelExt extension)
export dist_analysis, dist_synthesis                      # Distributed transforms
export dist_scalar_roundtrip!, dist_vector_roundtrip!    # Distributed roundtrip tests
export DistPlan, dist_synthesis!                         # Distributed plans
export DistAnalysisPlan, dist_analysis!                  
export DistSphtorPlan, dist_analysis_sphtor, dist_synthesis_sphtor,
       dist_analysis_sphtor!, dist_synthesis_sphtor!   # Distributed vector transforms
export DistQstPlan, dist_analysis_qst, dist_synthesis_qst,
       dist_analysis_qst!, dist_synthesis_qst!            # Distributed Q,S,T transforms
export dist_SH_to_lat, dist_SH_to_point, dist_SHqst_to_point           # Distributed evaluation
export dist_analysis_packed, dist_synthesis_packed                   # Distributed packed transforms
export dist_analysis_packed_cplx, dist_synthesis_packed_cplx                      # Distributed complex transforms
export dist_SHqst_to_lat                                                # Distributed Q,S,T to latitude
export dist_SH_rotate_euler                                             # Distributed Euler rotations
export dist_spatial_divergence, dist_spatial_vorticity                  # Distributed vector invariants
export dist_scalar_laplacian, dist_scalar_laplacian!                    # Distributed scalar Laplacian
export dist_SH_Zrotate_packed, dist_SH_Yrotate_packed, dist_SH_Yrotate90_packed, dist_SH_Xrotate90_packed
export create_spectral_pencil, create_spectral_array                        # Distributed spectral array creation
export create_spatial_pencil, create_spatial_array                          # Distributed spatial (θ-decomposed) array creation
export matrix_to_spectral_pencil, spectral_pencil_to_matrix                 # Distributed spectral conversion

# ===== EXTENSION FALLBACK FUNCTIONS =====
# These provide informative error messages when extension packages are not loaded

"""
    analysis_qst!(plan::DistQstPlan, Qlm_out, Slm_out, Tlm_out, Vrθφ, Vtθφ, Vpθφ) -> (Qlm_out, Slm_out, Tlm_out)

Planned distributed analysis of a 3D vector field with radial `Vrθφ`,
colatitude `Vtθφ` and longitude `Vpθφ` components, given as `PencilArray`s.
It writes the radial `Qlm`, spheroidal `Slm` and toroidal `Tlm` coefficients
into the output matrices: dense `(lmax+1, mmax+1)` and filled identically on
every rank. Build the plan with `DistQstPlan(cfg, prototype_θφ; use_rfft=false)`
for fields laid out like `prototype_θφ`; it reuses its buffers across calls.
This is the same operation as `dist_analysis_qst!` with that plan.

Only `DistQstPlan` methods exist. They are defined when
`using MPI, PencilArrays, PencilFFTs` loads the parallel extension, and they
need CPU storage. The call is collective over the plan's communicator.
"""
function analysis_qst! end

"""
    synthesis_qst!(plan::DistQstPlan, Vrθφ_out, Vtθφ_out, Vpθφ_out, Qlm, Slm, Tlm; real_output=true) -> (Vrθφ_out, Vtθφ_out, Vpθφ_out)

Planned distributed synthesis of a 3D vector field from dense radial `Qlm`,
spheroidal `Slm` and toroidal `Tlm` coefficient matrices, which must be
identical on every rank. It writes the radial, colatitude and longitude
components into the output `PencilArray`s, laid out like the plan's prototype.
This is the same operation as `dist_synthesis_qst!` with a `DistQstPlan`.

Only `DistQstPlan` methods exist. They are defined when
`using MPI, PencilArrays, PencilFFTs` loads the parallel extension, and they
need CPU storage. The call is collective over the plan's communicator.
"""
function synthesis_qst! end

@inline _parallel_ext_module() = Base.get_extension(@__MODULE__, :SHTnsKitParallelExt)

"""Construct the parallel extension's concrete distributed scalar analysis plan."""
function DistAnalysisPlan(args...; kwargs...)
    ext = _parallel_ext_module()
    ext === nothing && error(
        "Parallel extension not loaded. Load MPI, PencilArrays, and PencilFFTs first",
    )
    return getproperty(ext, :DistAnalysisPlan)(args...; kwargs...)
end

"""Construct the parallel extension's concrete distributed scalar synthesis plan."""
function DistPlan(args...; kwargs...)
    ext = _parallel_ext_module()
    ext === nothing && error(
        "Parallel extension not loaded. Load MPI, PencilArrays, and PencilFFTs first",
    )
    return getproperty(ext, :DistPlan)(args...; kwargs...)
end

"""Construct the parallel extension's concrete distributed vector plan."""
function DistSphtorPlan(args...; kwargs...)
    ext = _parallel_ext_module()
    ext === nothing && error(
        "Parallel extension not loaded. Load MPI, PencilArrays, and PencilFFTs first",
    )
    return getproperty(ext, :DistSphtorPlan)(args...; kwargs...)
end

"""Construct the parallel extension's concrete distributed QST plan."""
function DistQstPlan(args...; kwargs...)
    ext = _parallel_ext_module()
    ext === nothing && error(
        "Parallel extension not loaded. Load MPI, PencilArrays, and PencilFFTs first",
    )
    return getproperty(ext, :DistQstPlan)(args...; kwargs...)
end

# NOTE: the φ-FFT plan cache and its `fft_plan_cache_enabled` /
# `set_fft_plan_cache!` / `enable_fft_plan_cache!` / `disable_fft_plan_cache!`
# controls live in src/fftutils.jl. They used to forward to the parallel
# extension, where the cache they addressed had no readers at all — `_get_or_plan`
# was never called from anywhere, so every one of these knobs (and the
# `SHTNSKIT_CACHE_PENCILFFTS` environment variable the distributed guide
# advertises) was a no-op. The cache that every transform actually uses, serial
# and distributed alike, is the one in fftutils.jl, so the controls now address
# that and no longer require the extension to be loaded.

# ===== PENCIL GRID SUGGESTION =====
function _suggest_pencil_grid_fallback(comm_or_nprocs::Any, nlat::Integer, nlon::Integer;
                                       prefer_square::Bool=true,
                                       allow_one_dim::Bool=true)
    return (1, 1)
end

function suggest_pencil_grid(comm_or_nprocs::Any, nlat::Integer, nlon::Integer;
                              prefer_square::Bool=true,
                              allow_one_dim::Bool=true)
    ext = _parallel_ext_module()
    impl = ext === nothing ? _suggest_pencil_grid_fallback : getproperty(ext, :_suggest_pencil_grid_impl)
    return impl(comm_or_nprocs, Int(nlat), Int(nlon);
                prefer_square=prefer_square,
                allow_one_dim=allow_one_dim)
end

"""
    suggest_pencil_grid(comm_or_nprocs, nlat, nlon; prefer_square=true, allow_one_dim=true)

Return a suggested MPI pencil decomposition `(p_theta, p_phi)` for a grid of size
`nlat x nlon`. The base package provides a `(1,1)` fallback so the function is
always defined; when `SHTnsKitParallelExt` is loaded the callback is replaced
with an MPI-aware heuristic that favours balanced 2D decompositions.
"""
suggest_pencil_grid

# GPU extension fallbacks
const _GPU_NOT_LOADED_MSG = "No loaded GPU extension implements this entry point for the supplied arguments. For portable transforms, load CUDA.jl, AMDGPU.jl, or Metal.jl and call analysis or synthesis."
"""
    gpu_analysis(cfg, f; device=GPU()) -> Matrix

Scalar analysis on the active CUDA device, with the coefficients copied back to
a host `Matrix`. They are the coefficients `analysis` gives for a `CuArray`.
`f` may be a host array (copied to the device) or a `CuArray`;
`analysis(cfg, f)` on a `CuArray` keeps the result on the device.

Requires `using CUDA` and a functional device; otherwise it raises
[`BackendUnavailableError`](@ref). `device=CPU()` is an error. Only the CUDA
extension implements this function.
"""
gpu_analysis(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)

"""
    gpu_synthesis(cfg, alm; device=GPU(), real_output=true) -> Matrix

Scalar synthesis on the active CUDA device, returned as a host `Matrix`
(complex when `real_output=false`). `alm` may be a host matrix or a `CuArray`;
`synthesis(cfg, alm)` on a `CuArray` keeps the result on the device.

Requires `using CUDA` and a functional device. `device=CPU()` is an error. Only
the CUDA extension implements this function.
"""
gpu_synthesis(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)

"""
    gpu_analysis_safe(cfg, f; device=get_device()) -> Matrix

Like [`gpu_analysis`](@ref), but it falls back to CPU `analysis` in any of
these cases:
- `device` is `CPU()`;
- CUDA is not functional;
- the estimated memory exceeds the free device memory (see
  [`estimate_memory_usage`](@ref));
- the device runs out of memory.

Requires `using CUDA`.
"""
gpu_analysis_safe(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)

"""
    gpu_synthesis_safe(cfg, alm; device=get_device(), real_output=true) -> Matrix

Like [`gpu_synthesis`](@ref), with the same CPU fallbacks as
[`gpu_analysis_safe`](@ref).

Requires `using CUDA`.
"""
gpu_synthesis_safe(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_analysis_sphtor(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_synthesis_sphtor(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
"""
    gpu_apply_laplacian!(cfg, alm; device=GPU()) -> alm

Multiply dense `(lmax+1, mmax+1)` coefficients by `-l(l+1)` in place on the
GPU, zeroing degree 0 and unused entries.
- With `using CUDA`, `alm` is a `CuArray`, or a host matrix that is copied to
  the device and back.
- With `using Metal`, `alm` is an `MtlArray` matrix.

ROCm arrays have no method.
"""
gpu_apply_laplacian!(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)

"""
    gpu_memory_info() -> (free, total)

Free and total memory of the active CUDA device, in bytes.

Requires `using CUDA` and a functional device.
"""
gpu_memory_info(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)

"""
    check_gpu_memory(required_bytes::Int) -> Bool

Return whether the active CUDA device has at least `required_bytes` free. It
returns `false` with a warning when memory is short, and `false` when CUDA is
not functional.

Requires `using CUDA`.
"""
check_gpu_memory(::Any; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_clear_cache!(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)

"""
    estimate_memory_usage(cfg, operation::Symbol) -> Int

Rough device-memory estimate, in bytes, for a CUDA transform. `operation` is
`:analysis`, `:synthesis` or `:vector`; any other value counts only the spatial
and coefficient buffers. The `gpu_*_safe` wrappers compare this estimate with
the free memory reported by [`gpu_memory_info`](@ref).

Requires `using CUDA`.
"""
estimate_memory_usage(::SHTConfig, ::Any) = error(_GPU_NOT_LOADED_MSG)

"""
    get_available_gpus() -> Vector

List the CUDA devices as named tuples `(device=GPU(), id, name)`, with 0-based
`id`. The list is empty when CUDA is not functional.

Requires `using CUDA`.
"""
get_available_gpus(args...) = error(_GPU_NOT_LOADED_MSG)

"""
    set_gpu_device(id::Int) -> Bool
    set_gpu_device(GPU(), id::Int) -> Bool

Make CUDA device `id` the active device. Ids are 0-based, as listed by
[`get_available_gpus`](@ref). Returns `false` and changes nothing when CUDA is
not functional.

Requires `using CUDA`.
"""
set_gpu_device(::Any) = error(_GPU_NOT_LOADED_MSG)

# Default fallbacks if extensions are not loaded (use broad signatures to avoid overwriting)
"""
    zgrad_scalar_energy(cfg, f) -> ∂E/∂f

Reverse-mode (Zygote) gradient of the spectral energy
`energy_scalar(cfg, analysis(cfg, f))` with respect to the spatial field `f`.
The gradient has the shape of `f`. If the parallel extension is also loaded,
a distributed `PencilArray` input gives its gradient in the same pencil.

Requires `using Zygote`.
"""
zgrad_scalar_energy(::SHTConfig, ::Any) = error("Zygote extension not loaded")

"""
    zgrad_vector_energy(cfg, Vt, Vp) -> (∂E/∂Vt, ∂E/∂Vp)

Zygote gradients of the vector energy
`energy_vector(cfg, analysis_sphtor(cfg, Vt, Vp)...)` with respect to the
colatitude and longitude components, which must have the same size.
Distributed `PencilArray` inputs work as for [`zgrad_scalar_energy`](@ref).

Requires `using Zygote`.
"""
zgrad_vector_energy(::SHTConfig, ::Any, ::Any) = error("Zygote extension not loaded")

"""
    zgrad_enstrophy_Tlm(cfg, Tlm) -> ∂Z/∂Tlm

Zygote gradient of `enstrophy(cfg, Tlm)` with respect to the toroidal
coefficients `Tlm`.

Requires `using Zygote`.
"""
zgrad_enstrophy_Tlm(::SHTConfig, ::Any) = error("Zygote extension not loaded")

"""
    fdgrad_scalar_energy(cfg, f) -> ∂E/∂f

Forward-mode (ForwardDiff) gradient of `energy_scalar(cfg, analysis(cfg, f))`
with respect to `f`, the full `(nlat, nlon)` grid. A distributed `PencilArray`
is gathered on every rank, which is a collective call, and its gradient is
returned in the same pencil. The cost grows with the number of grid points, so
prefer [`zgrad_scalar_energy`](@ref) beyond small grids.

Requires `using ForwardDiff`.
"""
fdgrad_scalar_energy(::SHTConfig, ::Any) = error("ForwardDiff extension not loaded")

"""
    fdgrad_vector_energy(cfg, Vt, Vp) -> (∂E/∂Vt, ∂E/∂Vp)

ForwardDiff gradients of `energy_vector(cfg, analysis_sphtor(cfg, Vt, Vp)...)`.
The full-grid requirement and `PencilArray` handling are as for
[`fdgrad_scalar_energy`](@ref).

Requires `using ForwardDiff`.
"""
fdgrad_vector_energy(::SHTConfig, ::Any, ::Any) = error("ForwardDiff extension not loaded")

"""
    zgrad_rotation_angles_real(cfg, Qlm::AbstractVector, α, β, γ) -> (∂L/∂α, ∂L/∂β, ∂L/∂γ)

Gradient of `L = ½‖R(α, β, γ) Qlm‖²` with respect to the ZYZ Euler angles. `R`
rotates the packed real-field coefficients `Qlm` as an `SHTRotation` does,
using `cfg`'s normalization and phase convention. Needs `cfg.mres == 1`.

Defined by the Zygote extension: requires `using Zygote`.
"""
zgrad_rotation_angles_real(::SHTConfig, ::Any, ::Any, ::Any, ::Any) = error("Zygote extension not loaded")

"""
    zgrad_rotation_angles_cplx(lmax, mmax, Zlm::AbstractVector, α, β, γ) -> (∂L/∂α, ∂L/∂β, ∂L/∂γ)

Complex counterpart of [`zgrad_rotation_angles_real`](@ref), for coefficients
in the complex packed `LM_cplx` layout with degrees up to `lmax` and orders up
to `mmax`.

Defined by the Zygote extension: requires `using Zygote`.
"""
zgrad_rotation_angles_cplx(::Any, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Zygote extension not loaded")
"""
    dist_analysis(cfg, fθφ::PencilArray; use_rfft=false, use_packed_storage=false) -> Matrix

Spherical harmonic analysis of a distributed scalar field. `fθφ` holds the
`(nlat, nlon)` grid as a `PencilArray`, best decomposed along latitude (see
[`create_spatial_array`](@ref)). Returns the dense `(lmax+1, mmax+1)`
coefficient matrix in `cfg`'s convention, replicated on every rank. With
`use_packed_storage=true` the same coefficients come packed in LM order, as a
`cfg.nlm` vector. `analysis(cfg, fθφ)` returns an m-distributed spectral
`PencilArray` instead.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `fθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref); use a [`DistTransposePlan`](@ref) for device
storage.
"""
dist_analysis(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_synthesis(cfg, Alm; prototype_θφ::PencilArray, real_output=true, use_rfft=false) -> Matrix

Synthesize a distributed scalar field from coefficients `Alm` in `cfg`'s
convention. `Alm` is either a dense `(lmax+1, mmax+1)` matrix identical on
every rank, or a spectral `PencilArray` (see [`create_spectral_array`](@ref)).
Returns this rank's block of the field as a plain `Matrix`, with the local size
of `prototype_θφ`; it is complex when `real_output=false`. For a spectral
`PencilArray`, `synthesis(cfg, Alm; prototype_θφ)` returns a `PencilArray`
instead.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `prototype_θφ`. CUDA- and ROCm-backed storage raises
[`BackendUnavailableError`](@ref); use a [`DistTransposePlan`](@ref).
"""
dist_synthesis(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_scalar_roundtrip!(cfg, fθφ::PencilArray) -> (rel_local, rel_global)

Round-trip diagnostic: analyse `fθφ` with [`dist_analysis`](@ref), synthesize
it back, and return the relative L2 error on this rank and over all ranks.
`fθφ` is not modified. The error is at round-off level only for band-limited
fields, with degrees up to `cfg.lmax`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `fθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_scalar_roundtrip!(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_vector_roundtrip!(cfg, Vtθφ::PencilArray, Vpθφ::PencilArray) -> ((rel_local, rel_global), (rel_local, rel_global))

Vector counterpart of [`dist_scalar_roundtrip!`](@ref). It runs a round trip
through [`dist_analysis_sphtor`](@ref) and [`dist_synthesis_sphtor`](@ref) and
returns the relative L2 errors of the colatitude and longitude components. The
inputs are not modified.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Vtθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_vector_roundtrip!(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_analysis_sphtor(cfg, Vtθφ::PencilArray, Vpθφ::PencilArray; use_rfft=false) -> (Slm, Tlm)

Spheroidal/toroidal analysis of a distributed tangent vector field, given by
its colatitude (`Vtθφ`) and longitude (`Vpθφ`) components on the `(nlat, nlon)`
grid. Returns dense `(lmax+1, mmax+1)` matrices `Slm` and `Tlm`, replicated on
every rank. `analysis_sphtor(cfg, Vtθφ, Vpθφ)` returns spectral `PencilArray`s
instead.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Vtθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref); use a [`DistTransposePlan`](@ref).
"""
dist_analysis_sphtor(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_synthesis_sphtor(cfg, Slm, Tlm; prototype_θφ::PencilArray, real_output=true, use_rfft=false) -> (Vt, Vp)

Synthesize the colatitude and longitude components of a tangent vector field
from spheroidal and toroidal coefficients. The coefficients are dense matrices
identical on every rank, or spectral `PencilArray`s. Returns this rank's block
of each component as a plain `Matrix`, with the local size of `prototype_θφ`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `prototype_θφ`. CUDA- and ROCm-backed storage raises
[`BackendUnavailableError`](@ref); use a [`DistTransposePlan`](@ref).
"""
dist_synthesis_sphtor(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_analysis_qst(cfg, Vrθφ::PencilArray, Vtθφ::PencilArray, Vpθφ::PencilArray; use_rfft=false) -> (Qlm, Slm, Tlm)

Analysis of a distributed 3D vector field. The radial component `Vrθφ` gives
the coefficients `Qlm`; the colatitude and longitude components give the
spheroidal `Slm` and toroidal `Tlm` coefficients. Returns dense
`(lmax+1, mmax+1)` matrices, replicated on every rank.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Vrθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref); use a [`DistTransposePlan`](@ref).
"""
dist_analysis_qst(::SHTConfig, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_synthesis_qst(cfg, Qlm, Slm, Tlm; prototype_θφ::PencilArray, real_output=true, use_rfft=false) -> (Vr, Vt, Vp)

Synthesize the radial, colatitude and longitude components of a 3D vector
field from `Qlm`, `Slm` and `Tlm`. The coefficients are dense matrices
identical on every rank, or spectral `PencilArray`s. Returns this rank's block
of each component as a plain `Matrix`, with the local size of `prototype_θφ`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `prototype_θφ`. CUDA- and ROCm-backed storage raises
[`BackendUnavailableError`](@ref); use a [`DistTransposePlan`](@ref).
"""
dist_synthesis_qst(::SHTConfig, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
"""
    dist_analysis!(plan::DistAnalysisPlan, Alm_out::AbstractMatrix, fθφ::PencilArray) -> Alm_out
    dist_analysis!(plan::DistTransposePlan, Alm::PencilArray, f::PencilArray) -> Alm

Planned distributed scalar analysis.

- `DistAnalysisPlan(cfg, prototype_θφ; use_rfft=false)` serves fields laid out
  like `prototype_θφ`. Each call writes the dense coefficients, replicated on
  every rank, into `Alm_out` and reuses the plan's buffers. It needs CPU
  storage.
- With a [`DistTransposePlan`](@ref), `f` and `Alm` must come from
  [`allocate_spatial`](@ref) and [`allocate_spectral`](@ref) of the same plan,
  and the coefficients stay m-distributed. This is the form that runs on CUDA
  and ROCm storage.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the plan's communicator.
"""
dist_analysis!(::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_synthesis!(plan::DistPlan, fθφ_out::PencilArray, Alm::PencilArray; real_output=true) -> fθφ_out
    dist_synthesis!(plan::DistTransposePlan, f::PencilArray, Alm::PencilArray) -> f

Planned distributed scalar synthesis.

- `DistPlan(cfg, prototype_θφ; use_rfft=false)` synthesizes from the spectral
  `PencilArray` `Alm` into `fθφ_out`, which is laid out like `prototype_θφ`. It
  needs CPU storage.
- With a [`DistTransposePlan`](@ref), `Alm` and `f` must come from
  [`allocate_spectral`](@ref) and [`allocate_spatial`](@ref) of the same plan.
  This is the form that runs on CUDA and ROCm storage.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the plan's communicator.
"""
dist_synthesis!(::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_analysis_sphtor!(plan::DistSphtorPlan, Slm_out, Tlm_out, Vtθφ, Vpθφ) -> (Slm_out, Tlm_out)
    dist_analysis_sphtor!(plan::DistTransposePlan, Slm, Tlm, Vt, Vp) -> (Slm, Tlm)

Planned spheroidal/toroidal analysis of a distributed tangent vector field.

- `DistSphtorPlan(cfg, prototype_θφ; use_rfft=false)` writes dense
  coefficients, replicated on every rank, into the matrices `Slm_out` and
  `Tlm_out`. It needs CPU storage.
- With a [`DistTransposePlan`](@ref) built with `with_vector=true`, all four
  arrays come from [`allocate_spatial`](@ref) and [`allocate_spectral`](@ref)
  of the plan. The outputs must be distinct arrays; the read-only inputs may
  be the same array. This is the form that runs on CUDA and ROCm storage.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the plan's communicator.
"""
dist_analysis_sphtor!(::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_synthesis_sphtor!(plan::DistSphtorPlan, Vtθφ_out, Vpθφ_out, Slm, Tlm; real_output=true) -> (Vtθφ_out, Vpθφ_out)
    dist_synthesis_sphtor!(plan::DistTransposePlan, Vt, Vp, Slm, Tlm) -> (Vt, Vp)

Planned synthesis of a distributed tangent vector field from spheroidal and
toroidal coefficients.

- `DistSphtorPlan(cfg, prototype_θφ; use_rfft=false)` takes dense `Slm` and
  `Tlm` matrices, identical on every rank, and writes the components into the
  `PencilArray`s `Vtθφ_out` and `Vpθφ_out`. It needs CPU storage.
- With a [`DistTransposePlan`](@ref) built with `with_vector=true`, all four
  arrays come from [`allocate_spatial`](@ref) and [`allocate_spectral`](@ref)
  of the plan. The outputs must be distinct arrays; `Slm` and `Tlm` may be the
  same array. This is the form that runs on CUDA and ROCm storage.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the plan's communicator.
"""
dist_synthesis_sphtor!(::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis_sphtor!(::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_analysis_qst!(plan::DistQstPlan, Qlm_out, Slm_out, Tlm_out, Vrθφ, Vtθφ, Vpθφ) -> (Qlm_out, Slm_out, Tlm_out)
    dist_analysis_qst!(plan::DistTransposePlan, Qlm, Slm, Tlm, Vr, Vt, Vp) -> (Qlm, Slm, Tlm)

Planned analysis of a distributed 3D vector field into radial `Qlm`,
spheroidal `Slm` and toroidal `Tlm` coefficients.

- `DistQstPlan(cfg, prototype_θφ; use_rfft=false)` writes dense coefficients,
  replicated on every rank, into the three output matrices. It needs CPU
  storage. [`analysis_qst!`](@ref) is the same operation.
- With a [`DistTransposePlan`](@ref) built with `with_vector=true`, all six
  arrays come from [`allocate_spatial`](@ref) and [`allocate_spectral`](@ref)
  of the plan and must be distinct. This is the form that runs on CUDA and
  ROCm storage.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the plan's communicator.
"""
dist_analysis_qst!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_synthesis_qst!(plan::DistQstPlan, Vrθφ_out, Vtθφ_out, Vpθφ_out, Qlm, Slm, Tlm; real_output=true) -> (Vrθφ_out, Vtθφ_out, Vpθφ_out)
    dist_synthesis_qst!(plan::DistTransposePlan, Vr, Vt, Vp, Qlm, Slm, Tlm) -> (Vr, Vt, Vp)

Planned synthesis of a distributed 3D vector field from `Qlm`, `Slm` and
`Tlm`.

- `DistQstPlan(cfg, prototype_θφ; use_rfft=false)` takes dense coefficient
  matrices, identical on every rank, and writes the three components into the
  output `PencilArray`s. It needs CPU storage. [`synthesis_qst!`](@ref) is the
  same operation.
- With a [`DistTransposePlan`](@ref) built with `with_vector=true`, all six
  arrays come from [`allocate_spatial`](@ref) and [`allocate_spectral`](@ref)
  of the plan and must be distinct. This is the form that runs on CUDA and
  ROCm storage.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the plan's communicator.
"""
dist_synthesis_qst!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
"""
    dist_SH_to_lat(cfg, Alm::PencilArray, cost; nphi=cfg.nlon, ltr=cfg.lmax, mtr=cfg.mmax, real_output=true) -> Vector

Evaluate the field with spectral `PencilArray` coefficients `Alm`, in `cfg`'s
convention, at `nphi` equally spaced longitudes on the latitude where
`cos(θ) = cost`. The expansion is truncated to degrees `≤ ltr` and orders
`≤ mtr`. Every rank receives the full vector.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Alm`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_SH_to_lat(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_SH_to_point(cfg, Alm::PencilArray, cost, phi) -> Real

Evaluate a real field, given by spectral `PencilArray` coefficients in `cfg`'s
convention, at the point where `cos(θ) = cost` and the longitude is `phi`.
Every rank receives the value.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Alm`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_SH_to_point(::SHTConfig, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")

"""
    dist_SHqst_to_point(cfg, Q::PencilArray, S::PencilArray, T::PencilArray, cost, phi) -> (vr, vt, vp)

Evaluate a 3D vector field at one point, where `cos(θ) = cost` and the
longitude is `phi`. The field is given by radial `Q`, spheroidal `S` and
toroidal `T` spectral `PencilArray`s. Every rank receives the radial,
colatitude and longitude components.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Q`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_SHqst_to_point(::SHTConfig, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")

"""
    dist_analysis_packed(cfg, fθφ::PencilArray; ltr=cfg.lmax, use_rfft=false) -> Vector

Distributed analysis into the packed LM layout of `analysis_packed`: a
`cfg.nlm` vector, replicated on every rank. Degrees above `ltr` are zero.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `fθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_analysis_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_synthesis_packed(cfg, Qlm::AbstractVector; prototype_θφ::PencilArray, real_output=true, ltr=cfg.lmax) -> Matrix

Synthesize a distributed field from packed LM-layout coefficients of length
`cfg.nlm`, which must be identical on every rank. Degrees above `ltr` are
ignored. Returns this rank's block as a plain `Matrix`, with the local size of
`prototype_θφ`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `prototype_θφ`. CUDA- and ROCm-backed storage raises
[`BackendUnavailableError`](@ref).
"""
dist_synthesis_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_analysis_packed_cplx(cfg, z::PencilArray) -> Vector

Analysis of a complex-valued distributed field into the complex packed layout
(`LM_cplx_index` order, both signs of m), replicated on every rank. Needs
`cfg.mres == 1`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `z`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_analysis_packed_cplx(::SHTConfig, ::Any) = error("Parallel extension not loaded")

"""
    dist_synthesis_packed_cplx(cfg, alm::AbstractVector; prototype_θφ::PencilArray) -> Matrix

Inverse of [`dist_analysis_packed_cplx`](@ref): synthesize a complex field
from coefficients in the complex packed layout, which must be identical on
every rank. Returns this rank's complex block as a plain `Matrix`, with the
local size of `prototype_θφ`. Needs `cfg.mres == 1`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `prototype_θφ`. CUDA- and ROCm-backed storage raises
[`BackendUnavailableError`](@ref).
"""
dist_synthesis_packed_cplx(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_SHqst_to_lat(cfg, Q::PencilArray, S::PencilArray, T::PencilArray, cost; nphi=cfg.nlon, ltr=cfg.lmax, mtr=cfg.mmax) -> (Vr, Vt, Vp)

Evaluate a 3D vector field at `nphi` equally spaced longitudes on the latitude
where `cos(θ) = cost`. The field is given by radial `Q`, spheroidal `S` and
toroidal `T` spectral `PencilArray`s. Every rank receives the three component
vectors.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Q`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_SHqst_to_lat(::SHTConfig, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
"""
    dist_SH_rotate_euler(cfg, input::PencilArray, α, β, γ, output::PencilArray) -> output

Rotate distributed spectral coefficients by the ZYZ Euler angles: `α` about z,
then `β` about y, then `γ` about z. `input` and `output` are spectral
`PencilArray`s with the same layout. Needs `cfg.mres == 1`. Unless `β` is a
multiple of π it also needs `cfg.mmax == cfg.lmax`, because other angles mix
in orders above `mmax`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `input`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_SH_rotate_euler(::SHTConfig, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")

"""
    dist_SH_Zrotate_packed(cfg, Qlm::AbstractVector{<:Complex}, α; prototype_lm::PencilArray) -> Vector

Rotate packed LM-layout coefficients by `α` about the z axis. `Qlm` has length
`cfg.nlm` and is replicated on every rank, and every rank gets a new rotated
vector. `prototype_lm` is any `PencilArray` on the communicator to use; it
supplies the communicator for the collective argument checks.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `prototype_lm`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_SH_Zrotate_packed(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_SH_Yrotate_packed(cfg, Qlm::AbstractVector{<:Complex}, β; prototype_lm::PencilArray) -> Vector

Rotate replicated packed LM-layout coefficients by `β` about the y axis; the
arguments are as for [`dist_SH_Zrotate_packed`](@ref). Needs `cfg.mres == 1`,
and also `cfg.mmax == cfg.lmax` unless `β` is a multiple of π.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `prototype_lm`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_SH_Yrotate_packed(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_SH_Yrotate90_packed(cfg, Qlm::AbstractVector{<:Complex}; prototype_lm::PencilArray) -> Vector

[`dist_SH_Yrotate_packed`](@ref) with `β = π/2`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `prototype_lm`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_SH_Yrotate90_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_SH_Xrotate90_packed(cfg, Qlm::AbstractVector{<:Complex}; prototype_lm::PencilArray) -> Vector

Rotate replicated packed LM-layout coefficients by 90° about the x axis; the
arguments are as for [`dist_SH_Zrotate_packed`](@ref). Needs `cfg.mres == 1`
and `cfg.mmax == cfg.lmax`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `prototype_lm`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_SH_Xrotate90_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_spatial_divergence(cfg, Vtθφ::PencilArray, Vpθφ::PencilArray; prototype_θφ=Vtθφ, use_rfft=false, real_output=true) -> Matrix

Horizontal divergence ∇·V of a distributed tangent vector field, computed
spectrally from its spheroidal part. Returns this rank's block as a plain
`Matrix`, with the local size of `prototype_θφ`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Vtθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_spatial_divergence(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_spatial_vorticity(cfg, Vtθφ::PencilArray, Vpθφ::PencilArray; prototype_θφ=Vtθφ, use_rfft=false, real_output=true) -> Matrix

Radial vorticity r̂·(∇×V) of a distributed tangent vector field, computed
spectrally from its toroidal part. Returns this rank's block as a plain
`Matrix`, with the local size of `prototype_θφ`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Vtθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_spatial_vorticity(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_scalar_laplacian(cfg, fθφ::PencilArray; prototype_θφ=fθφ, use_rfft=false, real_output=true) -> Matrix

Surface Laplacian of a distributed scalar field: analyse, scale degree `l` by
`-l(l+1)`, and synthesize. Returns this rank's block as a plain `Matrix`, with
the local size of `prototype_θφ`. To write into a `PencilArray`, use
[`dist_scalar_laplacian!`](@ref).

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `fθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_scalar_laplacian(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")

"""
    dist_scalar_laplacian!(cfg, outθφ::PencilArray, inθφ::PencilArray; use_rfft=false, real_output=true) -> outθφ

Write the surface Laplacian of `inθφ` into `outθφ`. `outθφ` is a `PencilArray`
with the same layout, and its element type must match the output: real when
`real_output=true`.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `inθφ`. CUDA- and ROCm-backed pencils raise
[`BackendUnavailableError`](@ref).
"""
dist_scalar_laplacian!(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_apply_laplacian!(::SHTConfig, ::Any) = error("Parallel extension not loaded")
dist_SH_mul_mx!(::SHTConfig, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
# Note: These stubs use untyped first argument to be less specific than extension methods.
# Extensions with typed (cfg::SHTConfig) signature will take precedence in dispatch.
"""
    create_spectral_pencil(cfg; comm=MPI.COMM_WORLD) -> Pencil

Return the `Pencil` for distributed spectral coefficients. Its global size is
`(lmax+1, mmax+1)`, decomposed along m (dimension 2), so each rank holds every
degree of its orders.

Requires `using MPI, PencilArrays, PencilFFTs`. Call it on every rank of
`comm`.
"""
create_spectral_pencil(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

"""
    create_spectral_array(cfg; comm=MPI.COMM_WORLD) -> PencilArray{ComplexF64}

Allocate an uninitialized spectral `PencilArray` on
[`create_spectral_pencil`](@ref).

Requires `using MPI, PencilArrays, PencilFFTs`. Call it on every rank of
`comm`.
"""
create_spectral_array(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

"""
    create_spatial_pencil(cfg; comm=MPI.COMM_WORLD) -> Pencil

Return the `Pencil` for distributed spatial fields. Its global size is
`(nlat, nlon)`, decomposed along latitude (dimension 1). Prefer it to
`Pencil((nlat, nlon), comm)`, which splits longitude; that decomposition does
not scale for these transforms.

Requires `using MPI, PencilArrays, PencilFFTs`. Call it on every rank of
`comm`.
"""
create_spatial_pencil(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

"""
    create_spatial_array(cfg; comm=MPI.COMM_WORLD) -> PencilArray{Float64}

Allocate a zero-initialized spatial `PencilArray` on
[`create_spatial_pencil`](@ref).

Requires `using MPI, PencilArrays, PencilFFTs`. Call it on every rank of
`comm`.
"""
create_spatial_array(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

"""
    matrix_to_spectral_pencil(cfg, Alm::AbstractMatrix; comm=MPI.COMM_WORLD) -> PencilArray

Distribute a dense `(lmax+1, mmax+1)` coefficient matrix, which must be
identical on every rank, into a spectral `PencilArray` on
[`create_spectral_pencil`](@ref). Each rank keeps its own m-columns.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
`comm`.
"""
matrix_to_spectral_pencil(cfg, Alm; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

"""
    spectral_pencil_to_matrix(cfg, Alm::PencilArray) -> Matrix

Gather a spectral `PencilArray` into the dense `(lmax+1, mmax+1)` matrix, on
every rank.

Requires `using MPI, PencilArrays, PencilFFTs`. The call is collective over
the communicator of `Alm`.
"""
spectral_pencil_to_matrix(cfg, Alm_p; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

# ===== TRANSPOSE-BASED DISTRIBUTED SHT (Task 2+) =====
export DistTransposePlan, allocate_spatial, allocate_spectral
"""
    DistTransposePlan(cfg; comm=MPI.COMM_WORLD, nlev=1, with_vector=true, array_type=Array, real_type=Float64, prototype=nothing)

Plan for transpose-based distributed transforms. This is the scalable
distributed path, and the only one that runs on CUDA or ROCm storage. A
PencilFFTs real FFT along longitude, followed by a latitude/order transpose,
gives each rank complete latitudes for its orders. The Legendre stage then
needs no further communication.

Allocate operands with [`allocate_spatial`](@ref) and
[`allocate_spectral`](@ref), then pass them to `dist_analysis!`,
`dist_synthesis!`, `dist_analysis_sphtor!`, `dist_synthesis_sphtor!`,
`dist_analysis_qst!` or `dist_synthesis_qst!`. Arrays from another plan are
rejected, even if their sizes match.

- `nlev` stacks independent fields along a trailing dimension.
- `with_vector=true` builds the tables the vector and QST transforms need.
- `array_type` selects the storage, for example `CuArray` or `ROCArray`.
  `prototype` takes the storage and precision from an existing `PencilArray`.
- `real_type` is `Float64` or `Float32`.

Configurations with `robert_form` are not supported. Requires
`using MPI, PencilArrays, PencilFFTs`, plus CUDA.jl or AMDGPU.jl for a device
plan. Construction and every transform are collective over `comm`.

```julia
plan = DistTransposePlan(cfg; comm=MPI.COMM_WORLD, nlev=2)
f = allocate_spatial(plan)      # fill this rank's block, parent(f)
alm = allocate_spectral(plan)
dist_analysis!(plan, alm, f)
dist_synthesis!(plan, f, alm)
```
"""
DistTransposePlan(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

"""
    allocate_spatial(plan::DistTransposePlan) -> PencilArray

Allocate an uninitialized real spatial `PencilArray` for `plan`. Its global
size is `(nlon, nlat, nlev)`: longitude comes first, unlike the
`(nlat, nlon)` grids of the `cfg`-based functions. Latitude is distributed, and
the storage follows the plan's `array_type`.

Requires `using MPI, PencilArrays, PencilFFTs`.
"""
allocate_spatial(plan) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

"""
    allocate_spectral(plan::DistTransposePlan) -> PencilArray

Allocate an uninitialized complex spectral `PencilArray` for `plan`. Its global
size is `(lmax+1, nlon÷2+1, nlev)`, with the second (order) dimension
distributed. That dimension indexes the real-FFT bins `m = 0:nlon÷2`. Only bins
with `m ≤ mmax` and `m % mres == 0` hold coefficients: orders `plan.m_local`,
stored in the local columns `plan.m_slots`. Analysis zeroes the other columns
and synthesis ignores them.

Requires `using MPI, PencilArrays, PencilFFTs`.
"""
allocate_spectral(plan) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

# ===== PARALLEL ROTATION FUNCTIONS =====
# Parallel rotations fallbacks (PencilArray-based)
# Non-bang (out-of-place) and in-place rotation variants
function dist_SH_Zrotate(::SHTConfig, ::Any, ::Any); error("Parallel extension not loaded"); end          # Out-of-place Z rotation
function dist_SH_Zrotate(::SHTConfig, ::Any, ::Any, ::Any); error("Parallel extension not loaded"); end   # In-place Z rotation
function dist_SH_Yrotate_allgatherm!(::SHTConfig, ::Any, ::Any, ::Any); error("Parallel extension not loaded"); end    # Y rotation with full gather
function dist_SH_Yrotate_truncgatherm!(::SHTConfig, ::Any, ::Any, ::Any); error("Parallel extension not loaded"); end # Y rotation with truncated gather
function dist_SH_Yrotate(::SHTConfig, ::Any, ::Any, ::Any); error("Parallel extension not loaded"); end              # General Y rotation
function dist_SH_Yrotate90(::SHTConfig, ::Any, ::Any); error("Parallel extension not loaded"); end                   # 90° Y rotation
function dist_SH_Xrotate90(::SHTConfig, ::Any, ::Any); error("Parallel extension not loaded"); end                   # 90° X rotation

# ===== LOOPVECTORIZATION EXTENSION FALLBACKS =====
# LoopVectorization extension fallbacks (keep methods broad so extensions can specialize)
"""
    analysis_turbo(cfg, f::AbstractMatrix) -> Matrix

[`analysis`](@ref) with LoopVectorization-optimized CPU loops. The arguments
and result are the same.

Requires `using LoopVectorization`.
"""
analysis_turbo(::Any, ::Any) = error("LoopVectorization extension not loaded")                    # Vectorized analysis

"""
    synthesis_turbo(cfg, alm::AbstractMatrix; real_output=true) -> Matrix

[`synthesis`](@ref) with LoopVectorization-optimized CPU loops. The arguments
and result are the same.

Requires `using LoopVectorization`.
"""
synthesis_turbo(::Any, ::Any; real_output::Bool=true) = error("LoopVectorization extension not loaded")  # Vectorized synthesis

"""
    turbo_apply_laplacian!(cfg, alm) -> alm

Multiply coefficients by `-l(l+1)` in place. `alm` is a dense
`(lmax+1, mmax+1)` matrix, whose orders not stored under `cfg.mres` are
zeroed, or a packed `cfg.nlm` vector.

Requires `using LoopVectorization`.
"""
turbo_apply_laplacian!(::Any, ::Any) = error("LoopVectorization extension not loaded")            # Vectorized Laplacian

"""
    benchmark_turbo_vs_simd(cfg; trials=3) -> NamedTuple

Time `analysis` and `synthesis` against [`analysis_turbo`](@ref) and
[`synthesis_turbo`](@ref) on random data, keeping the best of `trials` runs (in
seconds). Returns the four timings, plus `analysis_speedup`,
`synthesis_speedup`, and `speedup`, which repeats the analysis speedup.

Requires `using LoopVectorization`.
"""
benchmark_turbo_vs_simd(::Any; kwargs...) = error("LoopVectorization extension not loaded")      # Performance comparison

end # module SHTnsKit
