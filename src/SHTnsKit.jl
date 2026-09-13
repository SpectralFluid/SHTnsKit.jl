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
include("shtns37_contract.jl")               # Executable SHTns 3.7 parity inventory
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

# ===== SHTNS 3.7 CAPABILITY CONTRACT =====
export SHTns37Capability, SHTNS37_BACKENDS, SHTNS37_CAPABILITIES
export shtns37_capabilities

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

function analysis_qst! end
function synthesis_qst! end

@inline _parallel_ext_module() = Base.get_extension(@__MODULE__, :SHTnsKitParallelExt)

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
const _GPU_NOT_LOADED_MSG = "GPU extension not loaded. Install and load CUDA.jl or AMDGPU.jl with GPUArrays and KernelAbstractions"
gpu_analysis(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_synthesis(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_analysis_safe(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_synthesis_safe(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_analysis_sphtor(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_synthesis_sphtor(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_apply_laplacian!(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_memory_info(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
check_gpu_memory(::Any; kwargs...) = error(_GPU_NOT_LOADED_MSG)
gpu_clear_cache!(args...; kwargs...) = error(_GPU_NOT_LOADED_MSG)
estimate_memory_usage(::SHTConfig, ::Any) = error(_GPU_NOT_LOADED_MSG)
get_available_gpus(args...) = error(_GPU_NOT_LOADED_MSG)
set_gpu_device(::Any) = error(_GPU_NOT_LOADED_MSG)

# Default fallbacks if extensions are not loaded (use broad signatures to avoid overwriting)
zgrad_scalar_energy(::SHTConfig, ::Any) = error("Zygote extension not loaded")
zgrad_vector_energy(::SHTConfig, ::Any, ::Any) = error("Zygote extension not loaded")
zgrad_enstrophy_Tlm(::SHTConfig, ::Any) = error("Zygote extension not loaded")
fdgrad_scalar_energy(::SHTConfig, ::Any) = error("ForwardDiff extension not loaded")
fdgrad_vector_energy(::SHTConfig, ::Any, ::Any) = error("ForwardDiff extension not loaded")
zgrad_rotation_angles_real(::SHTConfig, ::Any, ::Any, ::Any, ::Any) = error("Zygote extension not loaded")
zgrad_rotation_angles_cplx(::Any, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Zygote extension not loaded")
dist_analysis(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_scalar_roundtrip!(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_vector_roundtrip!(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_analysis_sphtor(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis_sphtor(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_analysis_qst(::SHTConfig, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis_qst(::SHTConfig, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_analysis!(::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis!(::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_analysis_sphtor!(::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis_sphtor!(::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis_sphtor!(::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_analysis_qst!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis_qst!(::Any, ::Any, ::Any, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_to_lat(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_to_point(::SHTConfig, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
dist_SHqst_to_point(::SHTConfig, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
dist_analysis_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_synthesis_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_analysis_packed_cplx(::SHTConfig, ::Any) = error("Parallel extension not loaded")
dist_synthesis_packed_cplx(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SHqst_to_lat(::SHTConfig, ::Any, ::Any, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_rotate_euler(::SHTConfig, ::Any, ::Any, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
dist_SH_Zrotate_packed(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_Yrotate_packed(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_Yrotate90_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_SH_Xrotate90_packed(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spatial_divergence(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_spatial_vorticity(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_scalar_laplacian(::SHTConfig, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_scalar_laplacian!(::SHTConfig, ::Any, ::Any; kwargs...) = error("Parallel extension not loaded")
dist_apply_laplacian!(::SHTConfig, ::Any) = error("Parallel extension not loaded")
dist_SH_mul_mx!(::SHTConfig, ::Any, ::Any, ::Any) = error("Parallel extension not loaded")
# Note: These stubs use untyped first argument to be less specific than extension methods.
# Extensions with typed (cfg::SHTConfig) signature will take precedence in dispatch.
create_spectral_pencil(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")
create_spectral_array(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")
create_spatial_pencil(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")
create_spatial_array(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")
matrix_to_spectral_pencil(cfg, Alm; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")
spectral_pencil_to_matrix(cfg, Alm_p; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")

# ===== TRANSPOSE-BASED DISTRIBUTED SHT (Task 2+) =====
export DistTransposePlan, allocate_spatial, allocate_spectral
DistTransposePlan(cfg; kwargs...) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")
allocate_spatial(plan) = error("Parallel extension not loaded. Add MPI, PencilArrays, PencilFFTs.")
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
analysis_turbo(::Any, ::Any) = error("LoopVectorization extension not loaded")                    # Vectorized analysis
synthesis_turbo(::Any, ::Any; real_output::Bool=true) = error("LoopVectorization extension not loaded")  # Vectorized synthesis
turbo_apply_laplacian!(::Any, ::Any) = error("LoopVectorization extension not loaded")            # Vectorized Laplacian
benchmark_turbo_vs_simd(::Any; kwargs...) = error("LoopVectorization extension not loaded")      # Performance comparison

end # module SHTnsKit
