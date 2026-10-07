"""
ForwardDiff Extension for Automatic Differentiation Support

This extension provides automatic differentiation capabilities for SHTnsKit using 
ForwardDiff.jl. It enables gradient computation through spherical harmonic transforms, 
which is essential for optimization problems in spherical geometry.

Key Features:
- Automatic gradient computation for scalar and vector energy functionals
- Support for both regular matrices and distributed arrays
- Seamless integration with ForwardDiff's dual number arithmetic
- Compatible with optimization workflows in geophysical modeling

Mathematical Foundation:
The extension computes gradients of energy functionals like:
- Scalar energy: E = 0.5 ∫ |f(θ,φ)|² dΩ  
- Vector energy: E = 0.5 ∫ |∇×V|² + |∇·V|² dΩ

These are fundamental quantities in fluid dynamics and field theory.
"""
module SHTnsKitForwardDiffExt

using ForwardDiff
using SHTnsKit

# ===== SCALAR FIELD GRADIENT COMPUTATION =====

"""
    fdgrad_scalar_energy(cfg, f) -> ∂E/∂f

ForwardDiff gradient of scalar energy E = 0.5 ∫ |f|^2 under spectral transform.

This function computes the functional derivative of the scalar energy with respect
to the input field f. The energy is computed in spectral space after spherical
harmonic analysis, making this useful for spectral optimization problems.

Parameters:
- cfg: SHTnsKit configuration defining the transform parameters
- f: Input scalar field matrix [nlat × nlon]

Returns:
- Gradient matrix of same size as f, representing ∂E/∂f at each point
"""
function SHTnsKit.fdgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)
    # Dual numbers cannot pass through the distributed kernels, so a PencilArray
    # is gathered on every rank (a collective call), differentiated as a whole
    # grid, and its gradient returned in the input's own pencil. Every rank pays
    # for a full serial gradient: suitable for small grids; zgrad_* scales.
    values = SHTnsKit._global_values(f)
    values === f ||
        return SHTnsKit._local_block_like(f, SHTnsKit.fdgrad_scalar_energy(cfg, values))
    nlat, nlon = size(f)
    # NOTE: AbstractMatrix ≡ AbstractArray{T,2}, so this method (not the generic
    # AbstractArray overload below) is what Julia selects for ANY 2-D input,
    # including a 2-D PencilArray. The guard must live HERE or it never fires.
    (nlat == cfg.nlat && nlon == cfg.nlon) || throw(DimensionMismatch(
        "fdgrad_scalar_energy expects the full $(cfg.nlat)×$(cfg.nlon) grid; got $(nlat)×$(nlon) " *
        "(a distributed PencilArray gives LOCAL data — gather to the global grid first)."))

    # Define the energy functional as a function of flattened field
    loss(x) = SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, reshape(x, nlat, nlon)))
    
    # Use ForwardDiff to compute gradient via dual numbers
    g = ForwardDiff.gradient(loss, vec(f))
    return reshape(g, nlat, nlon)
end

# ===== DISTRIBUTED ARRAY SUPPORT =====
# Generic distributed/array wrappers (avoid hard dependency on PencilArrays)
# These methods work with any AbstractArray type, including distributed arrays

"""
    fdgrad_scalar_energy(cfg, fθφ::AbstractArray) -> ∂E/∂fθφ

ForwardDiff gradient for any array type, including distributed PencilArrays.

Dual numbers cannot pass through the distributed kernels, so a PencilArray is
gathered on every rank (a collective call), differentiated as a whole grid, and
its gradient returned as a PencilArray in the input's pencil. Each rank pays
for a full serial gradient, so this suits small grids; `zgrad_scalar_energy`
uses the distributed transforms directly.
"""
function SHTnsKit.fdgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, fθφ::AbstractArray)
    values = SHTnsKit._global_values(fθφ)
    values === fθφ ||
        return SHTnsKit._local_block_like(fθφ, SHTnsKit.fdgrad_scalar_energy(cfg, values))
    # Materialize to a plain Matrix; the size guard below rejects anything that
    # is not the whole grid.
    fθφ_mat = fθφ isa Matrix ? fθφ : Matrix(fθφ)
    nlat = size(fθφ_mat, 1); nlon = size(fθφ_mat, 2)
    (nlat == cfg.nlat && nlon == cfg.nlon) || throw(DimensionMismatch(
        "fdgrad_scalar_energy expects the full $(cfg.nlat)×$(cfg.nlon) grid; got $(nlat)×$(nlon) " *
        "(a distributed PencilArray gives LOCAL data — gather to the global grid first)."))

    # Define energy loss function for flattened array
    function loss_flat(z)
        xloc = reshape(z, nlat, nlon)
        return SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, xloc))
    end

    # Compute gradient
    g = ForwardDiff.gradient(loss_flat, vec(fθφ_mat))
    gl = reshape(g, nlat, nlon)

    # Copy result back to distributed array format if supported
    if typeof(fθφ) <: AbstractMatrix
        return gl
    end
    # For distributed arrays (e.g., PencilArrays), try to preserve the container type
    gout = similar(fθφ, eltype(gl))
    copyto!(gout, gl)
    return gout
end

# ===== VECTOR FIELD GRADIENT COMPUTATION =====

"""
    fdgrad_vector_energy(cfg, Vtθφ, Vpθφ) -> (∂E/∂Vt, ∂E/∂Vp)

ForwardDiff gradient of vector field energy for distributed arrays.

This function computes gradients of the vector energy functional with respect
to both theta and phi components of a vector field. The vector energy typically
involves kinetic energy, enstrophy, or other quadratic functionals.

The implementation concatenates both vector components into a single state vector,
computes the energy gradient, then splits the result back into component gradients.

Parameters:
- cfg: SHTnsKit configuration
- Vtθφ: Theta component of vector field (distributed array)
- Vpθφ: Phi component of vector field (distributed array)

Returns:
- Tuple of gradient arrays (∂E/∂Vt, ∂E/∂Vp) with same types as inputs
"""
function SHTnsKit.fdgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vtθφ::AbstractArray, Vpθφ::AbstractArray)
    # Validate dimensions match
    size(Vtθφ) == size(Vpθφ) || throw(DimensionMismatch("Vt and Vp must have the same dimensions"))
    Vt_values = SHTnsKit._global_values(Vtθφ)
    if Vt_values !== Vtθφ
        gVt, gVp = SHTnsKit.fdgrad_vector_energy(
            cfg, Vt_values, SHTnsKit._global_values(Vpθφ))
        return SHTnsKit._local_block_like(Vtθφ, gVt), SHTnsKit._local_block_like(Vpθφ, gVp)
    end
    nlat = length(axes(Vtθφ, 1)); nlon = length(axes(Vtθφ, 2))
    (nlat == cfg.nlat && nlon == cfg.nlon) || throw(DimensionMismatch(
        "fdgrad_vector_energy expects the full $(cfg.nlat)×$(cfg.nlon) grid; got $(nlat)×$(nlon) " *
        "(a distributed PencilArray gives LOCAL data — gather to the global grid first)."))

    # Define vector energy functional for combined state vector [Vt; Vp]
    function loss_flat(z)
        Xt = reshape(view(z, 1:nlat*nlon), nlat, nlon)           # Extract Vt component
        Xp = reshape(view(z, nlat*nlon+1:2*nlat*nlon), nlat, nlon) # Extract Vp component
        Slm, Tlm = SHTnsKit.analysis_sphtor(cfg, Xt, Xp)      # Spheroidal/toroidal analysis
        return SHTnsKit.energy_vector(cfg, Slm, Tlm)            # Compute vector energy
    end
    
    # Create combined state vector and compute gradient
    # For non-Matrix types (e.g., PencilArrays), Array() gives local data only.
    Vt_mat = Vtθφ isa Matrix ? Vtθφ : Matrix(Vtθφ)
    Vp_mat = Vpθφ isa Matrix ? Vpθφ : Matrix(Vpθφ)
    z0 = vcat(vec(Vt_mat), vec(Vp_mat))                        # Concatenate components
    g = ForwardDiff.gradient(loss_flat, z0)                    # Compute full gradient
    
    # Split gradient back into component gradients (make copies for consistent behavior)
    gVt = reshape(g[1:nlat*nlon], nlat, nlon)                  # ∂E/∂Vt component
    gVp = reshape(g[nlat*nlon+1:2*nlat*nlon], nlat, nlon)      # ∂E/∂Vp component

    # Copy back to distributed array format if supported
    if typeof(Vtθφ) <: AbstractMatrix
        return gVt, gVp
    end
    # For distributed arrays (e.g., PencilArrays), preserve the container type
    GVt = similar(Vtθφ, eltype(gVt)); GVp = similar(Vpθφ, eltype(gVp))
    copyto!(GVt, gVt); copyto!(GVp, gVp)
    return GVt, GVp
end

"""
    fdgrad_vector_energy(cfg, Vt, Vp) -> (∂E/∂Vt, ∂E/∂Vp)

ForwardDiff gradient of vector field energy for regular matrices.

This is the standard matrix version of the vector energy gradient computation.
It works directly with AbstractMatrix types without the distributed array
overhead, making it more efficient for small to medium-sized problems.

The algorithm is identical to the distributed version but avoids the Array()
conversion since the inputs are already local matrices.

Parameters:
- cfg: SHTnsKit configuration
- Vt: Theta component matrix [nlat × nlon]
- Vp: Phi component matrix [nlat × nlon]

Returns:
- Tuple of gradient matrices (∂E/∂Vt, ∂E/∂Vp)
"""
function SHTnsKit.fdgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    # Validate dimensions match
    size(Vt) == size(Vp) || throw(DimensionMismatch("Vt and Vp must have the same dimensions"))
    # This AbstractMatrix method is selected for every 2-D input, including a
    # 2-D PencilArray: gather it (see `fdgrad_scalar_energy`).
    Vt_values = SHTnsKit._global_values(Vt)
    if Vt_values !== Vt
        gVt, gVp = SHTnsKit.fdgrad_vector_energy(
            cfg, Vt_values, SHTnsKit._global_values(Vp))
        return SHTnsKit._local_block_like(Vt, gVt), SHTnsKit._local_block_like(Vp, gVp)
    end
    nlat, nlon = size(Vt)
    (nlat == cfg.nlat && nlon == cfg.nlon) || throw(DimensionMismatch(
        "fdgrad_vector_energy expects the full $(cfg.nlat)×$(cfg.nlon) grid; got $(nlat)×$(nlon) " *
        "(a distributed PencilArray gives LOCAL data — gather to the global grid first)."))

    # Define vector energy functional for matrix inputs
    function loss_flat(z)
        Xt = reshape(view(z, 1:nlat*nlon), nlat, nlon)           # Extract Vt component
        Xp = reshape(view(z, nlat*nlon+1:2*nlat*nlon), nlat, nlon) # Extract Vp component
        Slm, Tlm = SHTnsKit.analysis_sphtor(cfg, Xt, Xp)      # Transform to spectral
        return SHTnsKit.energy_vector(cfg, Slm, Tlm)            # Compute energy
    end
    
    # Compute gradient directly on matrix data
    z0 = vcat(vec(Vt), vec(Vp))                                 # Concatenate vector components
    g = ForwardDiff.gradient(loss_flat, z0)                    # Compute gradient
    
    # Split result into component gradients (copies for safety)
    gVt = reshape(g[1:nlat*nlon], nlat, nlon)                  # ∂E/∂Vt
    gVp = reshape(g[nlat*nlon+1:2*nlat*nlon], nlat, nlon)      # ∂E/∂Vp
    return gVt, gVp
end

end # module
