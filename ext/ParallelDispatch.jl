##########
# Unified dispatch helpers for PencilArray inputs
##########

"""
    create_spectral_pencil(cfg; comm=MPI.COMM_WORLD)

Create a Pencil configuration for distributed spectral coefficients.
The spectral array has dimensions (lmax+1, mmax+1) and is distributed
along the m dimension (dimension 2) for optimal SHT performance.
"""
function SHTnsKit.create_spectral_pencil(cfg::SHTnsKit.SHTConfig; comm=MPI.COMM_WORLD)
    # Distribute along m (dimension 2) - each rank owns all l values for its m subset
    # This is optimal because each m-column is independent in Legendre transforms
    return _cached_pencil((cfg.lmax + 1, cfg.mmax + 1), (2,), comm)
end

"""
    create_spectral_array(cfg; comm=MPI.COMM_WORLD)

Create an uninitialized distributed PencilArray for spectral coefficients.
"""
function SHTnsKit.create_spectral_array(cfg::SHTnsKit.SHTConfig; comm=MPI.COMM_WORLD)
    pen = SHTnsKit.create_spectral_pencil(cfg; comm)
    return PencilArray{ComplexF64}(undef, pen)
end

"""
    create_spatial_pencil(cfg; comm=MPI.COMM_WORLD)

Create a Pencil for distributed spatial data `(nlat, nlon)` decomposed along
**latitude (θ, dimension 1)** — the scaling-friendly axis for SHTnsKit.

Prefer this over the bare `Pencil((nlat, nlon), comm)`: PencilArrays splits the
*last* dimension by default, which decomposes longitude (φ). The φ-distributed
analysis path `Allgatherv!`s the full longitude onto every rank and then
replicates the Legendre transform, so it does NOT scale (it usually gets slower
with more ranks). Latitude decomposition instead divides the Legendre work per
rank. Ordinary `analysis(cfg, pencil)` reduces only each destination rank's
owned m-columns; pass `return_pencil=false` explicitly for the replicated dense
compatibility result.
"""
function SHTnsKit.create_spatial_pencil(cfg::SHTnsKit.SHTConfig; comm=MPI.COMM_WORLD)
    # Decompose dimension 1 (θ / latitude). The (1,) tuple selects which global
    # dimension to split; omitting it would default to the last dim (φ).
    return _cached_pencil((cfg.nlat, cfg.nlon), (1,), comm)
end

"""
    create_spatial_array(cfg; comm=MPI.COMM_WORLD)

Create a zero-initialized distributed `PencilArray{Float64}` for spatial data,
decomposed along latitude (θ). See [`create_spatial_pencil`](@ref).
"""
function SHTnsKit.create_spatial_array(cfg::SHTnsKit.SHTConfig; comm=MPI.COMM_WORLD)
    pen = SHTnsKit.create_spatial_pencil(cfg; comm)
    arr = PencilArray{Float64}(undef, pen)
    fill!(parent(arr), 0.0)
    return arr
end

"""
    matrix_to_spectral_pencil(cfg, Alm::AbstractMatrix; comm=MPI.COMM_WORLD)

Convert a dense spectral coefficient matrix to a distributed PencilArray.
Each rank receives its local portion of the m-distributed array.
"""
function SHTnsKit.matrix_to_spectral_pencil(cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix; comm=MPI.COMM_WORLD)
    _validate_parallel_storage!(comm, :matrix_to_spectral_pencil, Alm)
    _record_pencil_scalar_stat!(:full_matrix_helper_calls, 1)
    pen = SHTnsKit.create_spectral_pencil(cfg; comm)
    known_comm = PencilArrays.get_comm(pen)
    _validate_explicit_comm!(known_comm, comm, :matrix_to_spectral_pencil)
    _validate_cfg_replicated(cfg, known_comm)
    _validate_spectral_pencil_plan!(cfg, pen, known_comm, :matrix_to_spectral_pencil)
    _validate_dense_spectral_matrix!(cfg, Alm, known_comm, :matrix_to_spectral_pencil)

    Alm_p = PencilArray{eltype(Alm)}(undef, pen)

    # Copy only the local portion
    lloc = axes(Alm_p, 1)
    mloc = axes(Alm_p, 2)
    gl_l = collect(Int, globalindices(Alm_p, 1))
    gl_m = collect(Int, globalindices(Alm_p, 2))

    for (jj, jm) in enumerate(mloc)
        mglob = gl_m[jj]
        for (ii, il) in enumerate(lloc)
            lglob = gl_l[ii]
            Alm_p[il, jm] = Alm[lglob, mglob]
        end
    end

    return Alm_p
end

"""
    spectral_pencil_to_matrix(cfg, Alm_p::PencilArray; comm=nothing)

Gather a distributed spectral PencilArray to a dense matrix on all ranks.
"""
function SHTnsKit.spectral_pencil_to_matrix(cfg::SHTnsKit.SHTConfig, Alm_p::PencilArray; comm=nothing)
    _record_pencil_scalar_stat!(:full_matrix_helper_calls, 1)
    known_comm = communicator(Alm_p)
    _validate_parallel_storage!(
        known_comm, :spectral_pencil_to_matrix, Alm_p,
    )
    _validate_cfg_replicated(cfg, known_comm)
    _validate_explicit_comm!(known_comm, comm, :spectral_pencil_to_matrix)
    _validate_scalar_pencil!(
        cfg, Alm_p, (cfg.lmax + 1, cfg.mmax + 1),
        :spectral_pencil_to_matrix;
        comm=known_comm, require_full_first_dim=true,
        required_decomposition=(2,), require_complex_input=true,
    )

    Alm = zeros(eltype(Alm_p), cfg.lmax + 1, cfg.mmax + 1)

    # Copy local portion
    lloc = axes(Alm_p, 1)
    mloc = axes(Alm_p, 2)
    gl_l = collect(Int, globalindices(Alm_p, 1))
    gl_m = collect(Int, globalindices(Alm_p, 2))

    for (jj, jm) in enumerate(mloc)
        mglob = gl_m[jj]
        for (ii, il) in enumerate(lloc)
            lglob = gl_l[ii]
            Alm[lglob, mglob] = Alm_p[il, jm]
        end
    end

    # Allreduce to combine contributions from all ranks
    # Always use the Pencil's communicator. An explicitly supplied congruent
    # communicator is accepted above, but may be a different duplicate on each
    # rank; using the known communicator keeps collective context ordering safe.
    MPI.Allreduce!(Alm, +, known_comm)

    return Alm
end

##########
# Scalar transform dispatch
##########

function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig, Alm::PencilArray;
                            prototype_θφ::PencilArray, real_output::Bool=true,
                            use_rfft::Bool=false,
                            comm=communicator(prototype_θφ))
    comm = _validate_public_comm_anchor!(
        communicator(prototype_θφ), comm, :synthesis, Alm, prototype_θφ,
    )
    local_result = SHTnsKit.dist_synthesis(
        cfg, Alm; prototype_θφ, real_output, use_rfft, comm,
    )
    result = PencilArray{eltype(local_result)}(undef, pencil(prototype_θφ))
    copyto!(parent(result), local_result)
    return result
end

function SHTnsKit.synthesis_cplx(cfg::SHTnsKit.SHTConfig, Alm::PencilArray;
                                 prototype_θφ::PencilArray)
    return SHTnsKit.synthesis(
        cfg, Alm; prototype_θφ, real_output=false,
    )
end

"""
Analyze a distributed field. The ordinary/default result is a spectral
`PencilArray`; `return_pencil=false` preserves the replicated dense
compatibility result from `dist_analysis`.
"""
function SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                           use_rfft::Bool=false, return_pencil::Bool=true)
    comm = communicator(fθφ)
    _validate_parallel_storage!(comm, :analysis, fθφ)
    return_pencil = _validate_collective_bool_option!(
        comm, return_pencil, :analysis, UInt32(0x1000),
    )
    if return_pencil
        return dist_analysis_pencil(cfg, fθφ; use_rfft, comm)
    else
        # Dense return is the preserved compatibility path.
        return SHTnsKit.dist_analysis(cfg, fθφ; use_rfft, comm)
    end
end

##########
# Vector/QST dispatch for PencilArrays
##########

function SHTnsKit.analysis_sphtor(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArray, Vpθφ::PencilArray;
                                   use_tables=cfg.use_plm_tables,
                                   use_rfft::Bool=false,
                                   return_pencil::Bool=true,
                                   comm=communicator(Vtθφ))
    comm = _validate_public_comm_anchor!(
        communicator(Vtθφ), comm, :analysis_sphtor, Vtθφ, Vpθφ,
    )
    return_pencil = _validate_collective_bool_option!(
        comm, return_pencil, :analysis_sphtor, UInt32(0x1000),
    )
    if return_pencil
        return dist_analysis_sphtor_pencil(
            cfg, Vtθφ, Vpθφ; use_rfft, comm,
        )
    else
        return SHTnsKit.dist_analysis_sphtor(
            cfg, Vtθφ, Vpθφ; use_tables, use_rfft, comm,
        )
    end
end

function SHTnsKit.analysis_sphtor_cplx(cfg::SHTnsKit.SHTConfig,
                                        Vtθφ::PencilArray,
                                        Vpθφ::PencilArray;
                                        return_pencil::Bool=true)
    return SHTnsKit.analysis_sphtor(
        cfg, Vtθφ, Vpθφ; return_pencil, use_rfft=false,
    )
end

function SHTnsKit.synthesis_sphtor(cfg::SHTnsKit.SHTConfig,
                                    Slm::PencilArray, Tlm::PencilArray;
                                    prototype_θφ::PencilArray,
                                    real_output::Bool=true,
                                    use_rfft::Bool=false,
                                    comm=communicator(prototype_θφ))
    comm = _validate_public_comm_anchor!(
        communicator(prototype_θφ), comm, :synthesis_sphtor,
        Slm, Tlm, prototype_θφ,
    )
    Vt_local, Vp_local = dist_synthesis_sphtor_pencil(
        cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft, comm,
    )
    Vt = PencilArray{eltype(Vt_local)}(undef, pencil(prototype_θφ))
    Vp = PencilArray{eltype(Vp_local)}(undef, pencil(prototype_θφ))
    copyto!(parent(Vt), Vt_local); copyto!(parent(Vp), Vp_local)
    return Vt, Vp
end


function SHTnsKit.synthesis_sphtor_cplx(cfg::SHTnsKit.SHTConfig,
                                         Slm::PencilArray,
                                         Tlm::PencilArray;
                                         prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_sphtor(
        cfg, Slm, Tlm; prototype_θφ, real_output=false,
    )
end

function _zero_spectral_pencil_like(reference::PencilArray)
    result = PencilArray{eltype(reference)}(undef, pencil(reference))
    fill!(parent(result), zero(eltype(result)))
    return result
end

function SHTnsKit.synthesis_sph(cfg::SHTnsKit.SHTConfig,
                                Slm::PencilArray;
                                prototype_θφ::PencilArray,
                                real_output::Bool=true)
    return SHTnsKit.synthesis_sphtor(
        cfg, Slm, _zero_spectral_pencil_like(Slm);
        prototype_θφ, real_output,
    )
end


function SHTnsKit.synthesis_sph_cplx(cfg::SHTnsKit.SHTConfig,
                                     Slm::PencilArray;
                                     prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_sph(
        cfg, Slm; prototype_θφ, real_output=false,
    )
end

function SHTnsKit.synthesis_tor(cfg::SHTnsKit.SHTConfig,
                                Tlm::PencilArray;
                                prototype_θφ::PencilArray,
                                real_output::Bool=true)
    return SHTnsKit.synthesis_sphtor(
        cfg, _zero_spectral_pencil_like(Tlm), Tlm;
        prototype_θφ, real_output,
    )
end


function SHTnsKit.synthesis_tor_cplx(cfg::SHTnsKit.SHTConfig,
                                     Tlm::PencilArray;
                                     prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_tor(
        cfg, Tlm; prototype_θφ, real_output=false,
    )
end

function SHTnsKit.analysis_sphtor_l(cfg::SHTnsKit.SHTConfig,
                                    Vt::PencilArray, Vp::PencilArray,
                                    ltr::Integer)
    return dist_analysis_sphtor_pencil(cfg, Vt, Vp; ltr)
end

function SHTnsKit.synthesis_sphtor_l(cfg::SHTnsKit.SHTConfig,
                                     S::PencilArray, Tlm::PencilArray,
                                     ltr::Integer;
                                     prototype_θφ::PencilArray,
                                     real_output::Bool=true,
                                     use_rfft::Bool=false)
    local_vt, local_vp = dist_synthesis_sphtor_pencil(
        cfg, S, Tlm; prototype_θφ, real_output, use_rfft, ltr,
    )
    outputs = map((local_vt, local_vp)) do local_value
        result = PencilArray{eltype(local_value)}(undef, pencil(prototype_θφ))
        copyto!(parent(result), local_value)
        result
    end
    return outputs
end

function SHTnsKit.synthesis_sphtor_l_cplx(cfg::SHTnsKit.SHTConfig,
                                          S::PencilArray, Tlm::PencilArray,
                                          ltr::Integer;
                                          prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_sphtor_l(
        cfg, S, Tlm, ltr; prototype_θφ, real_output=false,
    )
end

function SHTnsKit.synthesis_sph_l(cfg::SHTnsKit.SHTConfig,
                                  S::PencilArray, ltr::Integer;
                                  prototype_θφ::PencilArray,
                                  real_output::Bool=true)
    return SHTnsKit.synthesis_sphtor_l(
        cfg, S, _zero_spectral_pencil_like(S), ltr;
        prototype_θφ, real_output,
    )
end

function SHTnsKit.synthesis_sph_l_cplx(cfg::SHTnsKit.SHTConfig,
                                       S::PencilArray, ltr::Integer;
                                       prototype_θφ::PencilArray)
    comm = communicator(prototype_θφ)
    _validate_parallel_storage!(
        comm, :synthesis_sph_l_cplx, S, prototype_θφ,
    )
    return SHTnsKit.synthesis_sph_l(
        cfg, S, ltr; prototype_θφ, real_output=false,
    )
end

function SHTnsKit.synthesis_tor_l(cfg::SHTnsKit.SHTConfig,
                                  Tlm::PencilArray, ltr::Integer;
                                  prototype_θφ::PencilArray,
                                  real_output::Bool=true)
    return SHTnsKit.synthesis_sphtor_l(
        cfg, _zero_spectral_pencil_like(Tlm), Tlm, ltr;
        prototype_θφ, real_output,
    )
end

function SHTnsKit.synthesis_tor_l_cplx(cfg::SHTnsKit.SHTConfig,
                                       Tlm::PencilArray, ltr::Integer;
                                       prototype_θφ::PencilArray)
    comm = communicator(prototype_θφ)
    _validate_parallel_storage!(
        comm, :synthesis_tor_l_cplx, Tlm, prototype_θφ,
    )
    return SHTnsKit.synthesis_tor_l(
        cfg, Tlm, ltr; prototype_θφ, real_output=false,
    )
end

SHTnsKit.synthesis_grad(cfg::SHTnsKit.SHTConfig, S::PencilArray;
                        prototype_θφ::PencilArray,
                        real_output::Bool=true) =
    SHTnsKit.synthesis_sph(cfg, S; prototype_θφ, real_output)
SHTnsKit.synthesis_grad_l(cfg::SHTnsKit.SHTConfig, S::PencilArray,
                          ltr::Integer; prototype_θφ::PencilArray,
                          real_output::Bool=true) =
    SHTnsKit.synthesis_sph_l(cfg, S, ltr; prototype_θφ, real_output)

SHTnsKit.analysis_sphtor_ml(cfg::SHTnsKit.SHTConfig, im::Integer,
                            Vt::PencilArray, Vp::PencilArray,
                            ltr::Integer) =
    _analysis_sphtor_mode_pencil(cfg, im, Vt, Vp, ltr)
SHTnsKit.synthesis_sphtor_ml(cfg::SHTnsKit.SHTConfig, im::Integer,
                             S::PencilArray, Tlm::PencilArray,
                             ltr::Integer) =
    _synthesis_sphtor_mode_pencil(cfg, im, S, Tlm, ltr)
SHTnsKit.synthesis_sph_ml(cfg::SHTnsKit.SHTConfig, im::Integer,
                          S::PencilArray, ltr::Integer) =
    SHTnsKit.synthesis_sphtor_ml(
        cfg, im, S, _zero_spectral_pencil_like(S), ltr,
    )
SHTnsKit.synthesis_tor_ml(cfg::SHTnsKit.SHTConfig, im::Integer,
                          Tlm::PencilArray, ltr::Integer) =
    SHTnsKit.synthesis_sphtor_ml(
        cfg, im, _zero_spectral_pencil_like(Tlm), Tlm, ltr,
    )
SHTnsKit.synthesis_grad_ml(cfg::SHTnsKit.SHTConfig, im::Integer,
                           S::PencilArray, ltr::Integer) =
    SHTnsKit.synthesis_sph_ml(cfg, im, S, ltr)

function _pencil_vector_diagonal!(output::PencilArray,
                                  cfg::SHTnsKit.SHTConfig,
                                  input::PencilArray;
                                  inverse::Bool,
                                  operation::Symbol)
    comm = communicator(input)
    _validate_qst_pencil_communicators!(
        comm, (input, output), operation,
    )
    _validate_cfg_replicated(cfg, comm)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    _validate_scalar_pencil!(
        cfg, input, expected, operation; comm, peer=output,
        require_full_first_dim=true, required_decomposition=(2,),
    )
    _validate_scalar_pencil!(
        cfg, output, expected, operation; comm, peer=input,
        require_full_first_dim=true, required_decomposition=(2,),
    )
    _validate_identical_pencil_layout!(input, output, operation; comm)
    flags = eltype(input) === eltype(output) ? UInt32(0) : UInt32(0x0004)
    _collective_validation_error(comm, flags, operation)

    l_globals = collect(Int, globalindices(input, 1))
    m_globals = collect(Int, globalindices(input, 2))
    source = parent(input); destination = parent(output)
    @inbounds for (local_m, m_index) in pairs(m_globals),
                  (local_l, l_index) in pairs(l_globals)
        l = l_index - 1; m = m_index - 1
        if l >= max(1, m) && m % cfg.mres == 0
            ll1 = l * (l + 1)
            destination[local_l, local_m] = inverse ?
                -(source[local_l, local_m] / ll1) :
                -ll1 * source[local_l, local_m]
        else
            destination[local_l, local_m] = zero(eltype(destination))
        end
    end
    return output
end

function _pencil_vector_diagonal(cfg::SHTnsKit.SHTConfig,
                                 input::PencilArray;
                                 inverse::Bool,
                                 operation::Symbol)
    output = PencilArray{eltype(input)}(undef, pencil(input))
    return _pencil_vector_diagonal!(
        output, cfg, input; inverse, operation,
    )
end

SHTnsKit.divergence_from_spheroidal(cfg::SHTnsKit.SHTConfig,
                                    input::PencilArray) =
    _pencil_vector_diagonal(
        cfg, input; inverse=false, operation=:divergence_from_spheroidal,
    )
SHTnsKit.divergence_from_spheroidal!(cfg::SHTnsKit.SHTConfig,
                                     output::PencilArray,
                                     input::PencilArray) =
    _pencil_vector_diagonal!(
        output, cfg, input;
        inverse=false, operation=:divergence_from_spheroidal!,
    )
SHTnsKit.spheroidal_from_divergence(cfg::SHTnsKit.SHTConfig,
                                    input::PencilArray) =
    _pencil_vector_diagonal(
        cfg, input; inverse=true, operation=:spheroidal_from_divergence,
    )
SHTnsKit.spheroidal_from_divergence!(cfg::SHTnsKit.SHTConfig,
                                     output::PencilArray,
                                     input::PencilArray) =
    _pencil_vector_diagonal!(
        output, cfg, input;
        inverse=true, operation=:spheroidal_from_divergence!,
    )
SHTnsKit.vorticity_from_toroidal(cfg::SHTnsKit.SHTConfig,
                                 input::PencilArray) =
    _pencil_vector_diagonal(
        cfg, input; inverse=false, operation=:vorticity_from_toroidal,
    )
SHTnsKit.vorticity_from_toroidal!(cfg::SHTnsKit.SHTConfig,
                                  output::PencilArray,
                                  input::PencilArray) =
    _pencil_vector_diagonal!(
        output, cfg, input;
        inverse=false, operation=:vorticity_from_toroidal!,
    )
SHTnsKit.toroidal_from_vorticity(cfg::SHTnsKit.SHTConfig,
                                 input::PencilArray) =
    _pencil_vector_diagonal(
        cfg, input; inverse=true, operation=:toroidal_from_vorticity,
    )
SHTnsKit.toroidal_from_vorticity!(cfg::SHTnsKit.SHTConfig,
                                  output::PencilArray,
                                  input::PencilArray) =
    _pencil_vector_diagonal!(
        output, cfg, input;
        inverse=true, operation=:toroidal_from_vorticity!,
    )

function SHTnsKit.analysis_qst(cfg::SHTnsKit.SHTConfig,
                               Vrθφ::PencilArray,
                               Vtθφ::PencilArray,
                               Vpθφ::PencilArray;
                               use_rfft::Bool=false,
                               return_pencil::Bool=true)
    comm = communicator(Vrθφ)
    _validate_parallel_storage!(
        comm, :analysis_qst, Vrθφ, Vtθφ, Vpθφ,
    )
    comm = _validate_qst_spatial_inputs!(
        cfg, Vrθφ, Vtθφ, Vpθφ; use_rfft,
    )
    return_pencil = _validate_collective_bool_option!(
        comm, return_pencil, :analysis_qst, UInt32(0x1000),
    )
    if return_pencil
        return dist_analysis_pencil(cfg, Vrθφ; use_rfft, comm),
               dist_analysis_sphtor_pencil(
                   cfg, Vtθφ, Vpθφ; use_rfft, comm,
               )...
    end
    return SHTnsKit.dist_analysis_qst(
        cfg, Vrθφ, Vtθφ, Vpθφ; use_rfft,
    )
end

function SHTnsKit.analysis_qst_cplx(cfg::SHTnsKit.SHTConfig,
                                    Vrθφ::PencilArray,
                                    Vtθφ::PencilArray,
                                    Vpθφ::PencilArray;
                                    return_pencil::Bool=true)
    return SHTnsKit.analysis_qst(
        cfg, Vrθφ, Vtθφ, Vpθφ; return_pencil,
    )
end

function SHTnsKit.synthesis_qst(cfg::SHTnsKit.SHTConfig,
                                Qlm::PencilArray,
                                Slm::PencilArray,
                                Tlm::PencilArray;
                                prototype_θφ::PencilArray,
                                real_output::Bool=true,
                                use_rfft::Bool=false,
                                comm=communicator(prototype_θφ))
    comm = _validate_public_comm_anchor!(
        communicator(prototype_θφ), comm, :synthesis_qst,
        Qlm, Slm, Tlm, prototype_θφ,
    )
    _validate_qst_synthesis_inputs!(
        cfg, Qlm, Slm, Tlm, prototype_θφ; real_output, use_rfft, comm,
    )
    Vr_local = SHTnsKit.dist_synthesis(
        cfg, Qlm; prototype_θφ, real_output, use_rfft, comm,
    )
    Vt_local, Vp_local = dist_synthesis_sphtor_pencil(
        cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft, comm,
    )
    outputs = map((Vr_local, Vt_local, Vp_local)) do local_value
        result = PencilArray{eltype(local_value)}(undef, pencil(prototype_θφ))
        copyto!(parent(result), local_value)
        result
    end
    return outputs
end

function SHTnsKit.synthesis_qst_cplx(cfg::SHTnsKit.SHTConfig,
                                     Qlm::PencilArray,
                                     Slm::PencilArray,
                                     Tlm::PencilArray;
                                     prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_qst(
        cfg, Qlm, Slm, Tlm; prototype_θφ, real_output=false,
    )
end

function SHTnsKit.analysis_qst_l(cfg::SHTnsKit.SHTConfig,
                                 Vr::PencilArray, Vt::PencilArray,
                                 Vp::PencilArray, ltr::Integer)
    comm = _validate_qst_spatial_inputs!(
        cfg, Vr, Vt, Vp; use_rfft=false,
    )
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :analysis_qst_l)
    return dist_analysis_pencil(cfg, Vr; ltr=lcap, comm),
           dist_analysis_sphtor_pencil(
               cfg, Vt, Vp; ltr=lcap, comm,
           )...
end

function SHTnsKit.synthesis_qst_l(cfg::SHTnsKit.SHTConfig,
                                  Q::PencilArray, S::PencilArray,
                                  Tlm::PencilArray, ltr::Integer;
                                  prototype_θφ::PencilArray,
                                  real_output::Bool=true,
                                  use_rfft::Bool=false)
    comm = _validate_qst_synthesis_inputs!(
        cfg, Q, S, Tlm, prototype_θφ; real_output, use_rfft,
    )
    lcap = _collective_truncation(comm, ltr, cfg.lmax, :synthesis_qst_l)
    radial_local = SHTnsKit.dist_synthesis(
        cfg, Q; prototype_θφ, real_output, use_rfft, ltr=lcap, comm,
    )
    tangential_local = dist_synthesis_sphtor_pencil(
        cfg, S, Tlm; prototype_θφ, real_output, use_rfft, ltr=lcap, comm,
    )
    return map((radial_local, tangential_local...)) do local_value
        result = PencilArray{eltype(local_value)}(undef, pencil(prototype_θφ))
        copyto!(parent(result), local_value)
        result
    end
end

function SHTnsKit.synthesis_qst_l_cplx(cfg::SHTnsKit.SHTConfig,
                                       Q::PencilArray, S::PencilArray,
                                       Tlm::PencilArray, ltr::Integer;
                                       prototype_θφ::PencilArray)
    return SHTnsKit.synthesis_qst_l(
        cfg, Q, S, Tlm, ltr; prototype_θφ, real_output=false,
    )
end

function SHTnsKit.analysis_qst_ml(cfg::SHTnsKit.SHTConfig,
                                  stored_im::Integer, Vr::PencilArray,
                                  Vt::PencilArray, Vp::PencilArray,
                                  ltr::Integer)
    comm = communicator(Vr)
    _validate_qst_pencil_communicators!(
        comm, (Vr, Vt, Vp), :analysis_qst_ml,
    )
    _validate_cfg_replicated(cfg, comm)
    stored, _, lcap = _collective_fixed_order(
        comm, cfg, stored_im, ltr, :analysis_qst_ml,
    )
    _validate_mode_pencils!(comm, (Vr, Vt, Vp), cfg.nlat, :analysis_qst_ml)
    Q = SHTnsKit.analysis_packed_ml(cfg, stored, Vr, lcap)
    S, Tlm = _analysis_sphtor_mode_pencil(
        cfg, stored, Vt, Vp, lcap; comm,
    )
    return Q, S, Tlm
end

function SHTnsKit.synthesis_qst_ml(cfg::SHTnsKit.SHTConfig,
                                   stored_im::Integer, Q::PencilArray,
                                   S::PencilArray, Tlm::PencilArray,
                                   ltr::Integer)
    comm = communicator(Q)
    _validate_qst_pencil_communicators!(
        comm, (Q, S, Tlm), :synthesis_qst_ml,
    )
    _validate_cfg_replicated(cfg, comm)
    stored, physical_m, lcap = _collective_fixed_order(
        comm, cfg, stored_im, ltr, :synthesis_qst_ml,
    )
    active_length = lcap - physical_m + 1
    _validate_mode_pencils!(
        comm, (Q, S, Tlm), active_length, :synthesis_qst_ml,
    )
    Vr = SHTnsKit.synthesis_packed_ml(cfg, stored, Q, lcap)
    Vt, Vp = _synthesis_sphtor_mode_pencil(
        cfg, stored, S, Tlm, lcap; comm,
    )
    return Vr, Vt, Vp
end

function _validate_pencil_batch!(cfg::SHTnsKit.SHTConfig, values::Tuple,
                                 kind::Symbol, operation::Symbol;
                                 comm=communicator(first(values)))
    _validate_qst_pencil_communicators!(comm, values, operation)
    _validate_cfg_replicated(cfg, comm)
    prefix = kind === :spatial ? (cfg.nlat, cfg.nlon) :
                                (cfg.lmax + 1, cfg.mmax + 1)
    reference = first(values)
    global_shape = size_global(reference)
    nfields = length(global_shape) == 3 ? global_shape[3] : 0
    flags = length(global_shape) == 3 && global_shape[1:2] == prefix &&
            nfields > 0 ? UInt32(0) : UInt32(0x0001)
    for value in values
        size_global(value) == global_shape || (flags |= 0x0001)
        # Batch fields are copied from physical parent slices into unpermuted
        # pencils, so their original permutation must be rejected here.
        PencilArrays.permutation(value) isa PencilArrays.NoPermutation ||
            (flags |= 0x0002)
        ndims(parent(value)) == 3 && size(parent(value), 3) == nfields ||
            (flags |= 0x0002)
        # Each field is rebuilt on the batch's own topology (see
        # `_batch_field_pencil`); a topology that decomposes all three
        # dimensions has no two-dimensional counterpart.
        length(PencilArrays.decomposition(pencil(value))) <= 2 || (flags |= 0x0002)
        eltype(value) === eltype(reference) || (flags |= 0x0004)
    end
    for value in Iterators.drop(values, 1)
        _validate_identical_pencil_layout!(reference, value, operation; comm)
    end
    kind === :spatial && !(eltype(reference) <: Real) && (flags |= 0x0400)
    kind === :spectral && !(eltype(reference) <: Complex) && (flags |= 0x0004)
    _collective_validation_error(comm, flags, operation)
    return comm, nfields
end

"""
    _batch_field_pencil(batch, canonical, comm) -> Pencil

The two-dimensional pencil that one field of `batch` is copied into, chosen so
that the field's local block is exactly the batch's local `(:, :, k)` slice.

When every rank's slice already coincides with the `canonical` pencil (the
default θ-split spatial or m-split spectral layout), that pencil is used, as the
per-field transforms expect. Otherwise the field takes the batch's own topology
and decomposition restricted to the first two dimensions. A topology axis that
decomposes the batch dimension has a single process (`_validate_pencil_batch!`
requires the batch dimension to be local), so it is placed on the field
dimension left free.

Copying the slice into the canonical pencil regardless scrambled every field of
a batch with another decomposition, including PencilArrays' own default for
three dimensions. Collective over `comm`.
"""
function _batch_field_pencil(batch::PencilArray, canonical, comm)
    p = pencil(batch)
    same_block = range_local(p)[1:2] == range_local(canonical)
    MPI.Allreduce(same_block, &, comm) && return canonical
    ndims(p) == 2 && return p        # a 2-D pencil carrying an extra batch dimension
    decomposition = PencilArrays.decomposition(p)
    free = [d for d in (1, 2) if !(d in decomposition)]
    field_decomposition = map(d -> d == 3 ? popfirst!(free) : d, decomposition)
    return Pencil(PencilArrays.topology(p), size_global(p)[1:2], field_decomposition)
end

function _pencil_batch_field(batch::PencilArray, field_index::Int, field_pencil)
    field = PencilArray{eltype(batch)}(undef, field_pencil)
    copyto!(parent(field), @view(parent(batch)[:, :, field_index]))
    return field
end

function _pencil_batch_output(cfg::SHTnsKit.SHTConfig, ::Type{T}, nfields::Int,
                              kind::Symbol, comm) where {T}
    global_shape = kind === :spatial ? (cfg.nlat, cfg.nlon, nfields) :
                                       (cfg.lmax + 1, cfg.mmax + 1, nfields)
    # Same partition as the per-field results, which use the default θ-split
    # spatial and m-split spectral pencils of `comm`.
    decomposition = kind === :spatial ? (1,) : (2,)
    return PencilArray{T}(undef, _cached_pencil(global_shape, decomposition, comm))
end

function SHTnsKit.analysis_sphtor_batch(cfg::SHTnsKit.SHTConfig,
                                        Vt::PencilArray, Vp::PencilArray;
                                        comm=communicator(Vt))
    comm = _validate_public_comm_anchor!(
        communicator(Vt), comm, :analysis_sphtor_batch, Vt, Vp,
    )
    comm, nfields = _validate_pencil_batch!(
        cfg, (Vt, Vp), :spatial, :analysis_sphtor_batch; comm,
    )
    CT = Complex{float(eltype(Vt))}
    S = _pencil_batch_output(cfg, CT, nfields, :spectral, comm)
    Tlm = similar(S)
    field_pencil = _batch_field_pencil(
        Vt, SHTnsKit.create_spatial_pencil(cfg; comm), comm,
    )
    for field_index in 1:nfields
        St, Tt = SHTnsKit.analysis_sphtor(
            cfg,
            _pencil_batch_field(Vt, field_index, field_pencil),
            _pencil_batch_field(Vp, field_index, field_pencil);
            comm,
        )
        copyto!(@view(parent(S)[:, :, field_index]), parent(St))
        copyto!(@view(parent(Tlm)[:, :, field_index]), parent(Tt))
    end
    return S, Tlm
end

function SHTnsKit.synthesis_sphtor_batch(cfg::SHTnsKit.SHTConfig,
                                         S::PencilArray, Tlm::PencilArray;
                                         real_output::Bool=true,
                                         comm=communicator(S))
    comm = _validate_public_comm_anchor!(
        communicator(S), comm, :synthesis_sphtor_batch, S, Tlm,
    )
    comm, nfields = _validate_pencil_batch!(
        cfg, (S, Tlm), :spectral, :synthesis_sphtor_batch; comm,
    )
    CT = eltype(S); RT = typeof(real(zero(CT))); OT = real_output ? RT : CT
    Vt = _pencil_batch_output(cfg, OT, nfields, :spatial, comm)
    Vp = similar(Vt)
    prototype = PencilArray{OT}(undef, SHTnsKit.create_spatial_pencil(cfg; comm))
    field_pencil = _batch_field_pencil(
        S, SHTnsKit.create_spectral_pencil(cfg; comm), comm,
    )
    for field_index in 1:nfields
        vt, vp = SHTnsKit.synthesis_sphtor(
            cfg,
            _pencil_batch_field(S, field_index, field_pencil),
            _pencil_batch_field(Tlm, field_index, field_pencil);
            prototype_θφ=prototype, real_output,
        )
        copyto!(@view(parent(Vt)[:, :, field_index]), parent(vt))
        copyto!(@view(parent(Vp)[:, :, field_index]), parent(vp))
    end
    return Vt, Vp
end

SHTnsKit.synthesis_sphtor_batch_cplx(cfg::SHTnsKit.SHTConfig,
                                     S::PencilArray, Tlm::PencilArray) =
    SHTnsKit.synthesis_sphtor_batch(cfg, S, Tlm; real_output=false)

function SHTnsKit.analysis_qst_batch(cfg::SHTnsKit.SHTConfig,
                                     Vr::PencilArray, Vt::PencilArray,
                                     Vp::PencilArray)
    comm, nfields = _validate_pencil_batch!(
        cfg, (Vr, Vt, Vp), :spatial, :analysis_qst_batch,
    )
    Q = begin
        output = _pencil_batch_output(
            cfg, Complex{float(eltype(Vr))}, nfields, :spectral, comm,
        )
        field_pencil = _batch_field_pencil(
            Vr, SHTnsKit.create_spatial_pencil(cfg; comm), comm,
        )
        for k in 1:nfields
            q = dist_analysis_pencil(
                cfg, _pencil_batch_field(Vr, k, field_pencil),
            )
            copyto!(@view(parent(output)[:, :, k]), parent(q))
        end
        output
    end
    S, Tlm = SHTnsKit.analysis_sphtor_batch(cfg, Vt, Vp; comm)
    return Q, S, Tlm
end

function SHTnsKit.synthesis_qst_batch(cfg::SHTnsKit.SHTConfig,
                                      Q::PencilArray, S::PencilArray,
                                      Tlm::PencilArray;
                                      real_output::Bool=true)
    comm, nfields = _validate_pencil_batch!(
        cfg, (Q, S, Tlm), :spectral, :synthesis_qst_batch,
    )
    CT = eltype(Q); RT = typeof(real(zero(CT))); OT = real_output ? RT : CT
    Vr = _pencil_batch_output(cfg, OT, nfields, :spatial, comm)
    prototype = PencilArray{OT}(undef, SHTnsKit.create_spatial_pencil(cfg; comm))
    field_pencil = _batch_field_pencil(
        Q, SHTnsKit.create_spectral_pencil(cfg; comm), comm,
    )
    for k in 1:nfields
        vr = SHTnsKit.dist_synthesis(
            cfg, _pencil_batch_field(Q, k, field_pencil);
            prototype_θφ=prototype, real_output,
        )
        copyto!(@view(parent(Vr)[:, :, k]), vr)
    end
    Vt, Vp = SHTnsKit.synthesis_sphtor_batch(
        cfg, S, Tlm; real_output, comm,
    )
    return Vr, Vt, Vp
end

SHTnsKit.synthesis_qst_batch_cplx(cfg::SHTnsKit.SHTConfig,
                                  Q::PencilArray, S::PencilArray,
                                  Tlm::PencilArray) =
    SHTnsKit.synthesis_qst_batch(cfg, Q, S, Tlm; real_output=false)
