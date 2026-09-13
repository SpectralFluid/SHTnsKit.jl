module SHTnsKitGPUExt

using SHTnsKit
using KernelAbstractions, GPUArrays, GPUArraysCore
using LinearAlgebra, FFTW

# Import CUDA
using CUDA
using CUDA.CUFFT

# Import functions from SHTnsKit to extend them
import SHTnsKit: gpu_analysis, gpu_synthesis, gpu_analysis_safe, gpu_synthesis_safe,
                 gpu_analysis_sphtor, gpu_synthesis_sphtor,
                 gpu_apply_laplacian!,
                 gpu_memory_info, check_gpu_memory, gpu_clear_cache!,
                 estimate_memory_usage, get_available_gpus, set_gpu_device

# Import device utilities functions to override
import SHTnsKit: on_device, _notify_cuda_loaded!
import SHTnsKit: _enable_gpu_loops!

# ============================================================================
# CUDA Backend Integration with device_utils.jl
# ============================================================================

@kernel function _sht_loop_kernel!(body, range)
    linear_index = @index(Global, Linear)
    if linear_index <= length(range)
        body(@inbounds range[linear_index])
    end
end

"""Launch a body supplied by `@sht_loop` on the first operand's KA backend."""
function _launch_sht_loop!(args...)
    first_array = args[1]
    range = args[end - 1]
    body = args[end]
    backend = KernelAbstractions.get_backend(first_array)
    kernel! = _sht_loop_kernel!(backend)
    kernel!(body, range; ndrange=length(range))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# Notify the main module that CUDA is available
function __init__()
    _enable_gpu_loops!(_launch_sht_loop!)
    if CUDA.functional()
        _notify_cuda_loaded!()
    end
end

"""
    _to_gpu_impl(arr::AbstractArray)

Transfer array to CUDA GPU. Overrides the stub in device_utils.jl.
"""
function _to_gpu_impl(arr::AbstractArray)
    if !CUDA.functional()
        error("CUDA is not functional")
    end
    return CuArray(arr)
end

# Avoid double conversion
_to_gpu_impl(arr::CuArray) = arr

"""
    on_device(arr::CuArray) -> ComputeDevice

Returns `GPU()` for CUDA arrays.
"""
on_device(::CuArray) = SHTnsKit.GPU()

@inline _is_cpu_device(::SHTnsKit.CPU) = true
@inline _is_cpu_device(::SHTnsKit.GPU) = false

"""
    get_available_gpus()

Returns a list of available CUDA GPU devices with their IDs and names.
"""
function get_available_gpus()
    gpus = []
    if CUDA.functional()
        for i = 0:(CUDA.ndevices()-1)
            push!(gpus, (device=SHTnsKit.GPU(), id=i, name=CUDA.name(CUDA.CuDevice(i))))
        end
    end
    return gpus
end

"""
    set_gpu_device(device_id::Int)

Set the active CUDA GPU device by ID.
"""
function set_gpu_device(device_id::Int)
    if CUDA.functional()
        CUDA.device!(device_id)
        return true
    end
    return false
end

set_gpu_device(::SHTnsKit.GPU, device_id::Int) = set_gpu_device(device_id)

# ============================================================================
# cuFFT-based FFT operations with pre-planned transforms
# ============================================================================

"""
    CuFFTPlan

Pre-planned cuFFT operations for efficient repeated transforms.
"""
struct CuFFTPlan{ForwardPlan,InversePlan,Buffer}
    forward_plan::ForwardPlan
    inverse_plan::InversePlan
    buffer::Buffer
    nlat::Int
    nlon::Int
end

"""
    create_cufft_plan(nlat::Int, nlon::Int)

Create pre-planned cuFFT operations for a grid of size (nlat, nlon).
Forward and inverse plans are created for transforms along the longitude dimension.
"""
function create_cufft_plan(nlat::Int, nlon::Int)
    # Allocate buffer for FFT operations
    buffer = CUDA.zeros(ComplexF64, nlat, nlon)

    # Create forward FFT plan along dimension 2 (longitude)
    forward_plan = CUFFT.plan_fft!(buffer, 2)

    # Create inverse FFT plan along dimension 2 (longitude)
    inverse_plan = CUFFT.plan_ifft!(buffer, 2)

    return CuFFTPlan(forward_plan, inverse_plan, buffer, nlat, nlon)
end

"""
    gpu_fft!(plan::CuFFTPlan, data::CuArray)

Perform in-place forward FFT using pre-planned cuFFT.
"""
function gpu_fft!(plan::CuFFTPlan, data::CuArray)
    mul!(data, plan.forward_plan, data)
    return data
end

"""
    gpu_ifft!(plan::CuFFTPlan, data::CuArray)

Perform in-place inverse FFT using pre-planned cuFFT.
"""
function gpu_ifft!(plan::CuFFTPlan, data::CuArray)
    mul!(data, plan.inverse_plan, data)
    return data
end

"""
    gpu_fft!(data::CuArray, dims)

Perform FFT on CUDA array along specified dimensions (without pre-planning).
"""
function gpu_fft!(data::CuArray, dims)
    plan = CUFFT.plan_fft!(data, dims)
    plan * data  # In-place plan: mul! is called internally, modifying data
    return data
end

"""
    gpu_ifft!(data::CuArray, dims)

Perform inverse FFT on CUDA array along specified dimensions (without pre-planning).
"""
function gpu_ifft!(data::CuArray, dims)
    plan = CUFFT.plan_ifft!(data, dims)
    plan * data
    return data
end

"""
    gpu_rfft(data::CuArray, dims)

Perform real-to-complex FFT on CUDA array.
"""
function gpu_rfft(data::CuArray{<:Real}, dims)
    return CUFFT.rfft(data, dims)
end

"""
    gpu_irfft(data::CuArray, n, dims)

Perform complex-to-real inverse FFT on CUDA array.
"""
function gpu_irfft(data::CuArray, n, dims)
    return CUFFT.irfft(data, n, dims)
end

# ============================================================================
# GPU-accelerated core operations using KernelAbstractions
# ============================================================================

@kernel function legendre_associated_kernel!(Plm, x, lmax, mmax)
    """
    GPU kernel for computing ORTHONORMAL normalized associated Legendre functions P̄_l^m(x).
    Parallelized over (latitude, m) pairs for maximum GPU utilization.

    Computes P̄_l^m = Nlm·P_l^m directly via a bounded recurrence (|P̄| ≲ 1 at all l,m),
    fixing overflow for lmax ≥ ~151 that afflicted the old unnormalized approach.
    The downstream analysis/synthesis kernels must NOT multiply by Nlm (it is folded in).

    Recurrence (Condon–Shortley phase included, orthonormal convention):
      P̄_0^0 = 1/sqrt(4π)   (INV_SQRT_4PI)
      P̄_m^m = −sqrt((2m+1)/(2m)) · sinθ · P̄_{m−1}^{m−1}   (sectoral step)
      P̄_{m+1}^m = sqrt(2m+3) · x · P̄_m^m
      P̄_l^m = a·x·P̄_{l−1}^m − b·P̄_{l−2}^m   (l ≥ m+2)
        a = sqrt(((2l−1)(2l+1)) / ((l−m)(l+m)))
        b = sqrt(((2l+1)(l−1−m)(l−1+m)) / ((2l−3)(l−m)(l+m)))
    """
    i, m_idx = @index(Global, NTuple)
    nlat = length(x)
    if i <= nlat && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        sint = sqrt(max(0.0, 1.0 - xi*xi))  # sin(θ)

        # ── Sectoral term P̄_m^m ────────────────────────────────────────────────
        # Start from P̄_0^0 = 1/sqrt(4π) and apply the sectoral recurrence m times.
        # Each step: P̄_k^k = −sqrt((2k+1)/(2k)) · sinθ · P̄_{k−1}^{k−1}
        const_inv_sqrt4pi = 0.28209479177387814  # 1/sqrt(4π)
        pmm = const_inv_sqrt4pi
        @inbounds for k = 1:m
            pmm = -sqrt((2.0*k + 1.0) / (2.0*k)) * sint * pmm
        end
        Plm[i, m+1, m_idx] = pmm

        # ── P̄_{m+1}^m ──────────────────────────────────────────────────────────
        if m < lmax
            pm1m = sqrt(2.0*m + 3.0) * xi * pmm
            Plm[i, m+2, m_idx] = pm1m

            # ── Three-term recurrence for l ≥ m+2 ──────────────────────────────
            plm_prev2 = pmm    # P̄_{l−2}^m
            plm_prev1 = pm1m   # P̄_{l−1}^m
            @inbounds for l = m+2:lmax
                fl  = Float64(l)
                fm  = Float64(m)
                a = sqrt(((2.0*fl - 1.0) * (2.0*fl + 1.0)) /
                         ((fl - fm) * (fl + fm)))
                b = sqrt(((2.0*fl + 1.0) * (fl - 1.0 - fm) * (fl - 1.0 + fm)) /
                         ((2.0*fl - 3.0) * (fl - fm) * (fl + fm)))
                plm = a * xi * plm_prev1 - b * plm_prev2
                Plm[i, l+1, m_idx] = plm
                plm_prev2 = plm_prev1
                plm_prev1 = plm
            end
        end
    end
end

@kernel function legendre_and_derivative_kernel!(Plm, dPlm, x, lmax, mmax)
    """
    GPU kernel for computing ORTHONORMAL normalized P̄_l^m(x) AND their
    derivatives dP̄_l^m/dx. Required for vector spherical harmonic transforms.

    Computes P̄_l^m via the same bounded normalized sectoral recurrence as
    legendre_associated_kernel! (no overflow at high lmax).

    For dP̄_l^m/dx the standard formula adapted to the normalized functions is:
      (x²−1) · dP̄_l^m/dx = l·x·P̄_l^m − sqrt((l²−m²)(2l+1)/(2l−1)) · P̄_{l−1}^m
    At poles (x²−1 ≈ 0) the derivative is set to zero (pole handling is done by
    the pole closed-form in the downstream synthesis kernel via Nlm separately).

    The downstream kernels must NOT multiply by Nlm (it is folded into P̄ and dP̄).
    """
    i, m_idx = @index(Global, NTuple)
    nlat = length(x)
    if i <= nlat && m_idx <= mmax + 1
        m = m_idx - 1
        xi = x[i]
        sint = sqrt(max(0.0, 1.0 - xi*xi))
        x2m1 = xi*xi - 1.0
        # Guard against x = ±1 (poles)
        inv_x2m1 = abs(x2m1) < 1e-14 ? 0.0 : 1.0 / x2m1

        # ── Sectoral P̄_m^m (same as legendre_associated_kernel!) ────────────
        const_inv_sqrt4pi = 0.28209479177387814
        pmm = const_inv_sqrt4pi
        @inbounds for k = 1:m
            pmm = -sqrt((2.0*k + 1.0) / (2.0*k)) * sint * pmm
        end
        Plm[i, m+1, m_idx] = pmm

        # dP̄_m^m/dx = m·x·P̄_m^m / (x²−1)   [0 for m=0]
        if m == 0
            dPlm[i, 1, 1] = 0.0
        else
            dPlm[i, m+1, m_idx] = Float64(m) * xi * pmm * inv_x2m1
        end

        if m < lmax
            # ── P̄_{m+1}^m ───────────────────────────────────────────────────
            pm1m = sqrt(2.0*m + 3.0) * xi * pmm
            Plm[i, m+2, m_idx] = pm1m

            # dP̄_{m+1}^m/dx: l=m+1 case.
            # Numerator: (m+1)·x·P̄_{m+1}^m − sqrt(((m+1)²−m²)(2(m+1)+1)/(2(m+1)−1)) · P̄_m^m
            # = (m+1)·x·P̄_{m+1}^m − sqrt((2m+1)(2m+3)/(2m+1)) · P̄_m^m
            # = (m+1)·x·P̄_{m+1}^m − sqrt(2m+3) · P̄_m^m
            dPlm[i, m+2, m_idx] = (Float64(m+1) * xi * pm1m -
                                    sqrt(2.0*m + 3.0) * pmm) * inv_x2m1

            # ── Three-term recurrence for l ≥ m+2 ──────────────────────────
            plm_prev2 = pmm    # P̄_{l−2}^m = P̄_m^m
            plm_prev1 = pm1m   # P̄_{l−1}^m = P̄_{m+1}^m
            @inbounds for l = m+2:lmax
                fl = Float64(l)
                fm = Float64(m)
                a = sqrt(((2.0*fl - 1.0) * (2.0*fl + 1.0)) /
                         ((fl - fm) * (fl + fm)))
                b = sqrt(((2.0*fl + 1.0) * (fl - 1.0 - fm) * (fl - 1.0 + fm)) /
                         ((2.0*fl - 3.0) * (fl - fm) * (fl + fm)))
                plm = a * xi * plm_prev1 - b * plm_prev2
                Plm[i, l+1, m_idx] = plm
                # dP̄_l^m/dx = [l·x·P̄_l^m − sqrt((l²−m²)(2l+1)/(2l−1))·P̄_{l−1}^m] / (x²−1)
                scale_lm = sqrt((fl*fl - fm*fm) * (2.0*fl + 1.0) / (2.0*fl - 1.0))
                dPlm[i, l+1, m_idx] = (fl * xi * plm - scale_lm * plm_prev1) * inv_x2m1
                plm_prev2 = plm_prev1
                plm_prev1 = plm
            end
        end
    end
end

"""
    gpu_analysis(cfg::SHTConfig, spatial_data; device=get_device())

GPU-accelerated spherical harmonic analysis transform using cuFFT.

Implements: a_lm = ∫∫ f(θ,φ) Y_l^m*(θ,φ) sin(θ) dθ dφ
1. FFT along φ (dimension 2) to extract Fourier modes
2. Gauss-Legendre integration along θ (dimension 1) with P_l^m weights

Fully parallelized: all (l,m) coefficients computed in a single kernel launch.

"""
function gpu_analysis(cfg::SHTConfig, spatial_data; device=get_device())
    if _is_cpu_device(device)
        return SHTnsKit.analysis(cfg, spatial_data)
    end

    # Validate input dimensions
    nlat, nlon = cfg.nlat, cfg.nlon
    size(spatial_data, 1) == nlat || throw(DimensionMismatch("spatial_data must have $nlat rows (nlat), got $(size(spatial_data, 1))"))
    size(spatial_data, 2) == nlon || throw(DimensionMismatch("spatial_data must have $nlon columns (nlon), got $(size(spatial_data, 2))"))

    # Transfer input data to GPU
    gpu_data = CuArray(ComplexF64.(spatial_data))

    # Allocate GPU arrays
    coeffs = CUDA.zeros(ComplexF64, cfg.lmax+1, cfg.mmax+1)
    Plm = CUDA.zeros(Float64, cfg.nlat, cfg.lmax+1, cfg.mmax+1)
    weights = CuArray(cfg.w)
    x_values = CuArray(cfg.x)  # cos(θ) values at Gauss points

    # Step 1: Precompute ORTHONORMAL normalized Legendre functions P̄_l^m on GPU.
    # P̄ = Nlm·P_l^m is bounded (|P̄| ≲ 1) at all lmax — no overflow.
    # Normalization Nlm is folded into P̄; downstream kernels must NOT multiply by Nlm.
    backend = CUDABackend()
    legendre_kernel! = legendre_associated_kernel!(backend)
    legendre_kernel!(Plm, x_values, cfg.lmax, cfg.mmax; ndrange=(cfg.nlat, cfg.mmax+1))
    CUDA.synchronize()

    # Step 2: FFT along φ direction (dimension 2) using cuFFT
    # After FFT: gpu_data[:, m+1] contains the m-th Fourier mode for m = 0, 1, ..., nlon-1
    gpu_fft!(gpu_data, 2)

    # Scaling factor for φ integration (matches CPU: cfg.cphi = 2π/nlon)
    scaleφ = cfg.cphi

    # Step 3: Fully parallel Legendre integration - ALL (l,m) pairs in one kernel.
    # Each thread computes one a_lm coefficient.
    # Plm already holds P̄_l^m (orthonormal-normalized); no separate Nlm factor needed.
    @kernel function analysis_kernel!(coeffs, Fφ, Plm, weights, nlat, nlon, lmax, mmax, mres, scale)
        l_idx, m_idx = @index(Global, NTuple)
        if l_idx <= lmax + 1 && m_idx <= mmax + 1
            l = l_idx - 1
            m = m_idx - 1
            # Only compute triangular entries for configured azimuthal orders.
            if l >= m && m <= nlon ÷ 2 && m % mres == 0
                result = ComplexF64(0, 0)
                @inbounds for i_lat = 1:nlat
                    # Gauss-Legendre quadrature: weight * P̄_l^m * Fourier_mode
                    # Fourier mode m is in column m+1; P̄ already includes Nlm.
                    result += weights[i_lat] * Plm[i_lat, l_idx, m_idx] * Fφ[i_lat, m_idx]
                end
                coeffs[l_idx, m_idx] = result * scale
            end
        end
    end

    analysis_k! = analysis_kernel!(backend)
    analysis_k!(coeffs, gpu_data, Plm, weights,
                cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres, scaleφ;
                ndrange=(cfg.lmax+1, cfg.mmax+1))
    CUDA.synchronize()

    # Transfer result back to CPU - coefficients are always complex
    Qlm = Array(coeffs)
    return SHTnsKit._externalize_coefficients!(Qlm, cfg)
end

"""
    gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)

GPU-accelerated spherical harmonic synthesis transform using cuFFT.

Implements: f(θ,φ) = Σ_l Σ_m a_lm Y_l^m(θ,φ)
1. Legendre summation along θ: F_m(θ) = Σ_l a_lm * P_l^m(cos θ) * N_lm
2. Inverse FFT along φ (dimension 2) to reconstruct spatial field

Fully parallelized: all (θ,m) Fourier modes computed in a single kernel launch.
"""
function gpu_synthesis(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
    if _is_cpu_device(device)
        return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
    end

    # Validate input dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(coeffs, 1) == lmax + 1 || throw(DimensionMismatch("coeffs must have $(lmax+1) rows (lmax+1), got $(size(coeffs, 1))"))
    size(coeffs, 2) == mmax + 1 || throw(DimensionMismatch("coeffs must have $(mmax+1) columns (mmax+1), got $(size(coeffs, 2))"))

    coeffs_int = SHTnsKit._internal_coefficients(coeffs, cfg)

    # Transfer coefficients to GPU
    gpu_coeffs = CuArray(ComplexF64.(coeffs_int))

    # Allocate GPU arrays
    Plm = CUDA.zeros(Float64, cfg.nlat, cfg.lmax+1, cfg.mmax+1)
    x_values = CuArray(cfg.x)  # cos(θ) values at Gauss points

    backend = CUDABackend()

    # Step 1: Precompute ORTHONORMAL normalized Legendre functions P̄_l^m on GPU.
    # P̄ = Nlm·P_l^m is bounded (|P̄| ≲ 1) at all lmax — no overflow.
    # Normalization Nlm is folded into P̄; downstream kernel must NOT multiply by Nlm.
    legendre_kernel! = legendre_associated_kernel!(backend)
    legendre_kernel!(Plm, x_values, cfg.lmax, cfg.mmax; ndrange=(cfg.nlat, cfg.mmax+1))
    CUDA.synchronize()

    # Step 2: Fully parallel Legendre summation - ALL (θ, m) pairs in one kernel.
    # Each thread computes F_m(θ_i) = Σ_l a_lm * P̄_l^m(cos θ_i) for one (lat, m).
    # Plm already holds P̄_l^m; no separate Nlm factor needed.
    fourier_modes = CUDA.zeros(ComplexF64, cfg.nlat, cfg.nlon)

    @kernel function synthesis_kernel!(Fφ, coeffs, Plm, nlat, nlon, lmax, mmax, mres, do_hermitian)
        i_lat, m_idx = @index(Global, NTuple)
        if i_lat <= nlat && m_idx <= mmax + 1 && (m_idx - 1) % mres == 0
            m = m_idx - 1
            # Compute F_m(θ_i) = Σ_l a_lm * P̄_l^m(cos θ_i)
            result = ComplexF64(0, 0)
            @inbounds for l = m:lmax
                l_idx = l + 1
                result += coeffs[l_idx, m_idx] * Plm[i_lat, l_idx, m_idx]
            end

            # Place in Fourier mode slots for IFFT
            # FFT convention: [0, 1, 2, ..., N/2, -N/2+1, ..., -1]
            if m == 0
                Fφ[i_lat, 1] = result
            elseif m <= nlon ÷ 2
                Fφ[i_lat, m + 1] = result
                # Hermitian symmetry for real output: F_{-m} = conj(F_m).
                # Skip when the negative-m slot coincides with the positive slot
                # (Nyquist mode m == nlon/2 for even nlon) — else conj(result)
                # would clobber the just-written result.
                if do_hermitian && m > 0
                    neg_m_idx = nlon - m + 1
                    if neg_m_idx >= 1 && neg_m_idx <= nlon && neg_m_idx != m + 1
                        Fφ[i_lat, neg_m_idx] = conj(result)
                    end
                end
            end
        end
    end

    synthesis_k! = synthesis_kernel!(backend)
    synthesis_k!(fourier_modes, gpu_coeffs, Plm,
                 cfg.nlat, cfg.nlon, cfg.lmax, cfg.mmax, cfg.mres, real_output;
                 ndrange=(cfg.nlat, cfg.mmax+1))
    CUDA.synchronize()

    # Step 3: Inverse FFT along φ direction (dimension 2) using cuFFT
    gpu_ifft!(fourier_modes, 2)

    # Apply inverse φ scaling (matches CPU: phi_inv_scale(cfg))
    # For Gauss grids: nlon; for regular grids: nlon/(2π)
    inv_scaleφ = SHTnsKit.phi_inv_scale(cfg)
    fourier_modes .*= inv_scaleφ

    # Transfer result back to CPU
    result = Array(fourier_modes)

    if real_output
        return real(result)
    else
        return result
    end
end

# ============================================================================
# Vector field GPU operations using proper spectral method
# ============================================================================

# Pole limits for the sphtor kernels — device twins of `_dPdtheta_at_pole` /
# `_P_over_sinth_at_pole` in src/kernels.jl.
#
# At an exact pole node (sinθ == 0, i.e. a pole-inclusive regular or
# Driscoll-Healy grid) the tables give dP̄/dθ = -sinθ·dP̄/dx = 0 and P̄/sinθ = 0
# because `inv_sθ` is guarded to 0. Both are wrong: the true limits are finite
# and non-zero for m = 1, so without this branch the GPU silently dropped the
# entire m = 1 contribution from the two pole rows and disagreed with the CPU
# for the same cfg. `(-1)^k` is written as an `isodd`/`iseven` select to stay
# allocation- and intrinsic-free inside a kernel.
const _GPU_POLE_TOL = SHTnsKit.POLE_TOLERANCE_FACTOR * eps(Float64)

@inline function _gpu_dPdtheta_at_pole(l::Int, m::Int, x::Float64, N::Float64)
    m == 1 || return 0.0
    ll1 = 0.5 * Float64(l * (l + 1))
    s = x > 0 ? -1.0 : (isodd(l) ? 1.0 : -1.0)   # north: -1;  south: (-1)^(l+1)
    return s * N * ll1
end

@inline function _gpu_P_over_sinth_at_pole(l::Int, m::Int, x::Float64, N::Float64)
    m == 1 || return 0.0
    ll1 = 0.5 * Float64(l * (l + 1))
    s = x > 0 ? -1.0 : (iseven(l) ? 1.0 : -1.0)  # north: -1;  south: (-1)^l
    return s * N * ll1
end

"""
    gpu_analysis_sphtor(cfg::SHTConfig, vθ, vφ; device=get_device())

GPU-accelerated spheroidal-toroidal decomposition of vector fields using proper spectral method.

Uses the adjoint of the synthesis formula with Gauss-Legendre quadrature:
    S_lm = Σ_i w_i * scaleφ / (l(l+1)) * (F_θ * ∂Y_l^m/∂θ + conj(im·m/sinθ·Y_l^m) * F_φ)
    T_lm = Σ_i w_i * scaleφ / (l(l+1)) * (-conj(im·m/sinθ·Y_l^m) * F_θ + ∂Y_l^m/∂θ * F_φ)

Where F_θ, F_φ are Fourier modes of Vθ, Vφ and w_i are quadrature weights.
All computation stays on GPU for maximum performance.
"""
function gpu_analysis_sphtor(cfg::SHTConfig, vθ, vφ; device=get_device())
    if _is_cpu_device(device)
        return SHTnsKit.analysis_sphtor(cfg, vθ, vφ)
    end

    nlat, nlon = cfg.nlat, cfg.nlon
    lmax, mmax = cfg.lmax, cfg.mmax
    size(vθ) == (nlat, nlon) || throw(DimensionMismatch(
        "vθ must have shape ($nlat, $nlon), got $(size(vθ))"))
    size(vφ) == (nlat, nlon) || throw(DimensionMismatch(
        "vφ must have shape ($nlat, $nlon), got $(size(vφ))"))

    backend = CUDABackend()

    # Transfer input to GPU and compute FFT along φ
    gpu_vθ = CuArray(ComplexF64.(vθ))
    gpu_vφ = CuArray(ComplexF64.(vφ))
    gpu_fft!(gpu_vθ, 2)
    gpu_fft!(gpu_vφ, 2)

    # Transfer config data to GPU. `Nlm` is only read on the pole branch below,
    # where the P̄/dP̄ tables collapse to 0 and the closed-form limits need N.
    x_values = CuArray(cfg.x)
    weights = CuArray(cfg.w)
    Nlm_values = CuArray(cfg.Nlm)
    scaleφ = cfg.cphi
    robert_form = cfg.robert_form

    # Compute ORTHONORMAL normalized P̄_l^m AND their dP̄/dx on GPU.
    # Both arrays already include Nlm — downstream kernels must NOT multiply by Nlm.
    Plm = CUDA.zeros(Float64, nlat, lmax+1, mmax+1)
    dPlm = CUDA.zeros(Float64, nlat, lmax+1, mmax+1)
    legendre_deriv_kernel! = legendre_and_derivative_kernel!(backend)
    legendre_deriv_kernel!(Plm, dPlm, x_values, lmax, mmax; ndrange=(nlat, mmax+1))
    CUDA.synchronize()

    # Phase 1: Compute per-latitude weighted contributions for each (l, m)
    # This produces intermediate arrays of shape (nlat, lmax+1, mmax+1)
    # Each thread handles one (latitude, l, m) triplet
    S_contrib = CUDA.zeros(ComplexF64, nlat, lmax+1, mmax+1)
    T_contrib = CUDA.zeros(ComplexF64, nlat, lmax+1, mmax+1)

    @kernel function vector_analysis_contrib_kernel!(S_out, T_out, Fθ, Fφ, Plm, dPlm,
                                                      x_vals, w_vals, Nlm_vals, scale,
                                                      nlat, lmax, mmax, mres, do_robert)
        i_lat, l_idx, m_idx = @index(Global, NTuple)

        if i_lat <= nlat && l_idx <= lmax + 1 && m_idx <= mmax + 1
            l = l_idx - 1
            m = m_idx - 1

            # Only compute valid pairs for configured azimuthal orders.
            if l >= max(1, m) && m % mres == 0
                x = x_vals[i_lat]
                sθ = sqrt(max(0.0, 1.0 - x * x))
                is_pole = sθ < _GPU_POLE_TOL
                inv_sθ = is_pole ? 0.0 : 1.0 / sθ
                wi = w_vals[i_lat]

                # Get Fourier modes for this latitude and m
                Fθ_val = Fθ[i_lat, m_idx]
                Fφ_val = Fφ[i_lat, m_idx]

                # Robert-form handling: input is sin(θ)*V, divide by sin(θ)
                if do_robert && !is_pole
                    Fθ_val /= sθ
                    Fφ_val /= sθ
                end

                # Get ORTHONORMAL normalized Legendre values (Nlm already folded in)
                P = Plm[i_lat, l_idx, m_idx]   # P̄_l^m
                dP = dPlm[i_lat, l_idx, m_idx]  # dP̄_l^m/dx

                # ∂Ȳ_l^m/∂θ = -sinθ * dP̄_l^m/dx, and Ȳ/sinθ = P̄ * inv_sθ — both
                # collapse to 0 at an exact pole node, so substitute the closed-form
                # limits there (mirrors `_sphtor_analysis_kernel!` on the CPU).
                dθY = -sθ * dP
                Y_over_s = P * inv_sθ
                if is_pole
                    N = Nlm_vals[l_idx, m_idx]
                    dθY = _gpu_dPdtheta_at_pole(l, m, x, N)
                    Y_over_s = _gpu_P_over_sinth_at_pole(l, m, x, N)
                end

                # Compute coefficient and term
                coeff = wi * scale / (l * (l + 1))
                term_re = 0.0
                term_im = m * Y_over_s

                # Adjoint of synthesis formulas:
                # S_lm += coeff * (Fθ * dθY + conj(term) * Fφ)
                # T_lm += coeff * (-conj(term) * Fθ + dθY * Fφ)

                # conj(term) = (term_re, -term_im) = (0, -m*Ȳ/sinθ)
                Fθ_re = real(Fθ_val)
                Fθ_im = imag(Fθ_val)
                Fφ_re = real(Fφ_val)
                Fφ_im = imag(Fφ_val)

                # Fθ * dθY (dθY is real)
                s1_re = Fθ_re * dθY
                s1_im = Fθ_im * dθY

                # conj(term) * Fφ = (0 - (-term_im)*Fφ_im, 0*Fφ_im + (-term_im)*Fφ_re)
                #                 = (term_im * Fφ_im, -term_im * Fφ_re)
                s2_re = term_im * Fφ_im
                s2_im = -term_im * Fφ_re

                S_out[i_lat, l_idx, m_idx] = coeff * ComplexF64(s1_re + s2_re, s1_im + s2_im)

                # -conj(term) * Fθ = -(term_im * Fθ_im, -term_im * Fθ_re)
                #                  = (-term_im * Fθ_im, term_im * Fθ_re)
                t1_re = -term_im * Fθ_im
                t1_im = term_im * Fθ_re

                # dθY * Fφ (dθY is real)
                t2_re = dθY * Fφ_re
                t2_im = dθY * Fφ_im

                T_out[i_lat, l_idx, m_idx] = coeff * ComplexF64(t1_re + t2_re, t1_im + t2_im)
            end
        end
    end

    contrib_kernel! = vector_analysis_contrib_kernel!(backend)
    contrib_kernel!(S_contrib, T_contrib, gpu_vθ, gpu_vφ, Plm, dPlm,
                    x_values, weights, Nlm_values, scaleφ,
                    nlat, lmax, mmax, cfg.mres, robert_form;
                    ndrange=(nlat, lmax+1, mmax+1))
    CUDA.synchronize()

    # Phase 2: Sum over latitude dimension to get final coefficients
    # Use CUDA reduction: sum along dimension 1
    Slm_gpu = dropdims(sum(S_contrib, dims=1), dims=1)
    Tlm_gpu = dropdims(sum(T_contrib, dims=1), dims=1)

    # Transfer results back to CPU
    Slm = Array(Slm_gpu)
    Tlm = Array(Tlm_gpu)

    SHTnsKit._externalize_coefficients!(Slm, cfg)
    SHTnsKit._externalize_coefficients!(Tlm, cfg)
    return Slm, Tlm
end

"""
    gpu_synthesis_sphtor(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=get_device(), real_output=true)

GPU-accelerated synthesis of spheroidal-toroidal vector field components using proper spectral method.

Uses the spectral formula:
    V_θ = ∂S/∂θ - (1/sinθ) ∂T/∂φ = Σ_{l,m} [∂Y_l^m/∂θ * S_lm - im/sinθ * Y_l^m * T_lm]
    V_φ = (1/sinθ) ∂S/∂φ + ∂T/∂θ = Σ_{l,m} [im/sinθ * Y_l^m * S_lm + ∂Y_l^m/∂θ * T_lm]

Where ∂Y_l^m/∂θ = -sinθ * N_lm * dP_l^m/dx (x = cosθ)
"""
function gpu_synthesis_sphtor(cfg::SHTConfig, sph_coeffs, tor_coeffs; device=get_device(), real_output=true)
    if _is_cpu_device(device)
        return SHTnsKit.synthesis_sphtor(cfg, sph_coeffs, tor_coeffs; real_output=real_output)
    end

    nlat, nlon = cfg.nlat, cfg.nlon
    lmax, mmax = cfg.lmax, cfg.mmax
    coeff_shape = (lmax + 1, mmax + 1)
    size(sph_coeffs) == coeff_shape || throw(DimensionMismatch(
        "sph_coeffs must have shape $coeff_shape, got $(size(sph_coeffs))"))
    size(tor_coeffs) == coeff_shape || throw(DimensionMismatch(
        "tor_coeffs must have shape $coeff_shape, got $(size(tor_coeffs))"))

    backend = CUDABackend()

    Slm_int = SHTnsKit._internal_coefficients(sph_coeffs, cfg)
    Tlm_int = SHTnsKit._internal_coefficients(tor_coeffs, cfg)

    # Transfer coefficients to GPU. Nlm is folded into P̄/dP̄ for the interior;
    # it is still needed for the pole-limit closed forms, where those tables are 0.
    gpu_Slm = CuArray(ComplexF64.(Slm_int))
    gpu_Tlm = CuArray(ComplexF64.(Tlm_int))
    x_values = CuArray(cfg.x)
    Nlm_values = CuArray(cfg.Nlm)

    # Compute ORTHONORMAL normalized P̄_l^m AND dP̄/dx on GPU.
    # Both arrays include Nlm — downstream kernel must NOT multiply by Nlm.
    Plm = CUDA.zeros(Float64, nlat, lmax+1, mmax+1)
    dPlm = CUDA.zeros(Float64, nlat, lmax+1, mmax+1)
    legendre_deriv_kernel! = legendre_and_derivative_kernel!(backend)
    legendre_deriv_kernel!(Plm, dPlm, x_values, lmax, mmax; ndrange=(nlat, mmax+1))
    CUDA.synchronize()

    # Fourier coefficients for vector components
    Fθ = CUDA.zeros(ComplexF64, nlat, nlon)
    Fφ = CUDA.zeros(ComplexF64, nlat, nlon)

    # sin(θ) values for each latitude
    sintheta = CuArray(cfg.st)

    # Scale factor for inverse FFT - use phi_inv_scale to match CPU (not 1/cphi!)
    inv_scaleφ = SHTnsKit.phi_inv_scale(cfg)

    # Kernel for spectral vector synthesis - compute Fourier modes for each (latitude, m).
    # Plm holds P̄_l^m and dPlm holds dP̄_l^m/dx (both orthonormal-normalized, no Nlm factor).
    @kernel function vector_spectral_synthesis_kernel!(Ftheta, Fphi, Slm, Tlm, Plm, dPlm, sintheta, x_vals, Nlm_vals, nlat, nlon, lmax, mmax, mres, inv_scale)
        i, m_idx = @index(Global, NTuple)
        if i <= nlat && m_idx <= mmax + 1 && (m_idx - 1) % mres == 0
            m = m_idx - 1
            sθ = sintheta[i]
            x = x_vals[i]
            is_pole = abs(sθ) < _GPU_POLE_TOL
            inv_sθ = is_pole ? 0.0 : 1.0 / sθ

            gθ = ComplexF64(0, 0)
            gφ = ComplexF64(0, 0)

            @inbounds for l = m:lmax
                l_idx = l + 1
                # P̄_l^m and dP̄_l^m/dx already include Nlm — no separate N factor.
                Y    = Plm[i, l_idx, m_idx]   # P̄_l^m
                dP   = dPlm[i, l_idx, m_idx]  # dP̄_l^m/dx

                # ∂Ȳ_l^m/∂θ = -sinθ * dP̄_l^m/dx and Ȳ/sinθ = P̄ * inv_sθ. At an
                # exact pole node both are 0 (inv_sθ is guarded), which drops the
                # whole m=1 term; substitute the closed-form limits as the CPU
                # `_sphtor_synthesis_kernel` does.
                dYdθ     = -sθ * dP
                Y_over_s = Y * inv_sθ
                if is_pole
                    N = Nlm_vals[l_idx, m_idx]
                    dYdθ     = _gpu_dPdtheta_at_pole(l, m, x, N)
                    Y_over_s = _gpu_P_over_sinth_at_pole(l, m, x, N)
                end

                Sl = Slm[l_idx, m_idx]
                Tl = Tlm[l_idx, m_idx]

                # V_θ = ∂S/∂θ - (im/sinθ) * T
                gθ += dYdθ * Sl - ComplexF64(0, m) * Y_over_s * Tl
                # V_φ = (im/sinθ) * S + ∂T/∂θ
                gφ += ComplexF64(0, m) * Y_over_s * Sl + dYdθ * Tl
            end

            # Store in Fourier coefficient array
            Ftheta[i, m_idx] = inv_scale * gθ
            Fphi[i, m_idx] = inv_scale * gφ
        end
    end

    synth_kernel! = vector_spectral_synthesis_kernel!(backend)
    synth_kernel!(Fθ, Fφ, gpu_Slm, gpu_Tlm, Plm, dPlm, sintheta, x_values, Nlm_values, nlat, nlon, lmax, mmax, cfg.mres, inv_scaleφ; ndrange=(nlat, mmax+1))
    CUDA.synchronize()

    # Apply Hermitian symmetry for real output
    if real_output
        @kernel function hermitian_symmetry_kernel!(F, nlat, nlon, mmax)
            i, m_idx = @index(Global, NTuple)
            if i <= nlat && m_idx <= mmax + 1
                m = m_idx - 1
                if m > 0 && m <= nlon ÷ 2
                    conj_idx = nlon - m + 1
                    if conj_idx >= 1 && conj_idx <= nlon
                        F[i, conj_idx] = conj(F[i, m_idx])
                    end
                end
            end
        end
        herm_kernel! = hermitian_symmetry_kernel!(backend)
        herm_kernel!(Fθ, nlat, nlon, mmax; ndrange=(nlat, mmax+1))
        herm_kernel!(Fφ, nlat, nlon, mmax; ndrange=(nlat, mmax+1))
        CUDA.synchronize()
    end

    # Inverse FFT along φ
    gpu_ifft!(Fθ, 2)
    gpu_ifft!(Fφ, 2)

    result_vθ = Array(Fθ)
    result_vφ = Array(Fφ)

    # Apply Robert-form scaling if configured (multiply by sin(θ) after IFFT)
    if cfg.robert_form
        for i in 1:nlat
            sθ = sqrt(max(0.0, 1 - cfg.x[i]^2))
            result_vθ[i, :] .*= sθ
            result_vφ[i, :] .*= sθ
        end
    end

    if real_output
        return real(result_vθ), real(result_vφ)
    else
        return result_vθ, result_vφ
    end
end

# ============================================================================
# Laplacian operator
# ============================================================================

@kernel function laplacian_kernel!(output, input, lmax, mmax)
    l, m = @index(Global, NTuple)
    if l <= lmax + 1 && m <= mmax + 1
        l_val = l - 1
        m_val = m - 1
        if l_val >= m_val
            output[l, m] = -l_val * (l_val + 1) * input[l, m]
        end
    end
end

"""
    gpu_apply_laplacian!(cfg::SHTConfig, coeffs; device=get_device())

GPU-accelerated Laplacian operator in spectral space.
"""
function gpu_apply_laplacian!(cfg::SHTConfig, coeffs; device=get_device())
    if _is_cpu_device(device)
        return SHTnsKit.apply_laplacian!(cfg, coeffs)
    end

    # Validate input dimensions
    lmax, mmax = cfg.lmax, cfg.mmax
    size(coeffs, 1) == lmax + 1 || throw(DimensionMismatch("coeffs must have $(lmax+1) rows (lmax+1), got $(size(coeffs, 1))"))
    size(coeffs, 2) == mmax + 1 || throw(DimensionMismatch("coeffs must have $(mmax+1) columns (mmax+1), got $(size(coeffs, 2))"))

    gpu_coeffs = CuArray(coeffs)
    # Zero-initialize output to handle l < m entries (which should be zero)
    output = CUDA.zeros(eltype(gpu_coeffs), size(gpu_coeffs))

    backend = CUDABackend()
    kernel! = laplacian_kernel!(backend)
    kernel!(output, gpu_coeffs, lmax, mmax; ndrange=(lmax+1, mmax+1))
    CUDA.synchronize()

    coeffs .= Array(output)
    return coeffs
end

# ============================================================================
# Memory utilities
# ============================================================================

"""
    gpu_memory_info()

Get memory information for the active CUDA device.
Returns a named tuple with `free` and `total` fields (in bytes).
"""
function gpu_memory_info()
    if CUDA.functional()
        mem = CUDA.MemoryInfo()
        return (free=mem.free_bytes, total=mem.total_bytes)
    else
        return (free=Sys.free_memory(), total=Sys.total_memory())
    end
end

"""
    check_gpu_memory(required_bytes::Int)

Check if sufficient GPU memory is available.
"""
function check_gpu_memory(required_bytes::Int)
    try
        mem_info = gpu_memory_info()
        if mem_info.free < required_bytes
            @warn "Insufficient memory: need $(required_bytes÷(1024^3)) GB, have $(mem_info.free÷(1024^3)) GB available"
            return false
        end
        return true
    catch e
        @warn "Could not check memory availability: $e"
        return true
    end
end

"""
    gpu_clear_cache!()

Clear the active CUDA device's memory cache.
"""
function gpu_clear_cache!()
    if CUDA.functional()
        try
            CUDA.reclaim()
            @info "CUDA memory cache cleared"
        catch e
            @warn "Failed to clear CUDA cache: $e"
        end
    end
    return nothing
end

"""
    estimate_memory_usage(cfg::SHTConfig, operation::Symbol)

Estimate memory usage for GPU operations.
"""
function estimate_memory_usage(cfg::SHTConfig, operation::Symbol)
    spatial_size = cfg.nlat * cfg.nlon * 16  # ComplexF64 = 16 bytes
    coeff_size = (cfg.lmax + 1) * (cfg.mmax + 1) * 16
    legendre_size = cfg.nlat * (cfg.lmax + 1) * (cfg.mmax + 1) * 8

    if operation == :analysis
        return spatial_size + coeff_size + legendre_size + spatial_size
    elseif operation == :synthesis
        return coeff_size + spatial_size + legendre_size + spatial_size
    elseif operation == :vector
        # Plm and dPlm are real (2×), while S_contrib and T_contrib are
        # ComplexF64 (4× the real tensor's bytes): 6× in total.
        return 2 * spatial_size + 2 * coeff_size + 6 * legendre_size + 2 * spatial_size
    else
        return spatial_size + coeff_size
    end
end

"""
    gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device())

Memory-safe GPU analysis with automatic fallback to CPU.
"""
function gpu_analysis_safe(cfg::SHTConfig, spatial_data; device=get_device())
    if _is_cpu_device(device)
        return SHTnsKit.analysis(cfg, spatial_data)
    end

    required_memory = estimate_memory_usage(cfg, :analysis)
    if !check_gpu_memory(required_memory)
        @info "Falling back to CPU due to memory constraints"
        return SHTnsKit.analysis(cfg, spatial_data)
    end

    try
        return gpu_analysis(cfg, spatial_data; device=device)
    catch e
        if isa(e, CUDA.OutOfGPUMemoryError) || contains(string(e), "memory")
            @warn "GPU out of memory, falling back to CPU: $e"
            return SHTnsKit.analysis(cfg, spatial_data)
        else
            rethrow(e)
        end
    end
end

"""
    gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)

Memory-safe GPU synthesis with automatic fallback to CPU.
"""
function gpu_synthesis_safe(cfg::SHTConfig, coeffs; device=get_device(), real_output=true)
    if _is_cpu_device(device)
        return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
    end

    required_memory = estimate_memory_usage(cfg, :synthesis)
    if !check_gpu_memory(required_memory)
        @info "Falling back to CPU due to memory constraints"
        return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
    end

    try
        return gpu_synthesis(cfg, coeffs; device=device, real_output=real_output)
    catch e
        if isa(e, CUDA.OutOfGPUMemoryError) || contains(string(e), "memory")
            @warn "GPU out of memory, falling back to CPU: $e"
            return SHTnsKit.synthesis(cfg, coeffs; real_output=real_output)
        else
            rethrow(e)
        end
    end
end

# Types defined by this extension.
export CuFFTPlan, create_cufft_plan, gpu_fft!, gpu_ifft!

end # module SHTnsKitGPUExt
