##########
# Vendor-owned PencilArray dispatch firewall
##########

# This file is included inside each compound MPI+GPU extension after it defines
# `VendorArray`, `VendorArrayType`, and `VENDOR_NAME`.  Methods added here are
# therefore owned by the compound extension and are more specific than the
# CPU/MPI `PencilArray` methods in `SHTnsKitParallelExt`.

# Keep PencilArray's bounds (and Pencil's buffer bound) so the alias is a
# subtype of `PencilArray`; otherwise no method below is more specific than
# its generic counterpart and vendor-backed calls are ambiguous. `VendorArray`
# cannot bound the data parameter, since some of its wrappers (`Adjoint`,
# `LogicalIndex`) need not have rank N. PencilArrays requires every parent to
# unwrap to `typeof_array(pencil)`, so the pencil's buffer type names the vendor.
const VendorPencilArray = PencilArrays.PencilArray{
    T,N,A,Np,E,P,
} where {T,N,A<:AbstractArray{T,N},Np,E,
         P<:PencilArrays.Pencil{<:Any,<:Any,<:Any,<:VendorArrayType{UInt8,1}}}

@inline _vendor_parent(value::VendorPencilArray) = PencilArrays.parent(value)
@inline _vendor_parent(value) = value

@inline function _vendor_comm(values...)
    for value in values
        value isa VendorPencilArray && return ParallelExt.communicator(value)
    end
    throw(ArgumentError("MPI GPU call requires a vendor-backed PencilArray"))
end

@inline _vendor_adapter(value::VendorPencilArray) =
    ParallelExt._parallel_gpu_adapter(PencilArrays.parent(value))

function _stage_vendor_call(operation::Symbol, f, values...;
                            mutated::Tuple=(), comm=_vendor_comm(values...))
    # Ordinary CPU and vendor methods can be selected on different MPI ranks
    # when residency itself is rank-varying. Participate in the same fixed-order
    # storage preflight as the generic entry point before reporting that the
    # all-vendor operation has no native implementation. A local early throw
    # here would otherwise strand CPU ranks in their first collective.
    ParallelExt._validate_parallel_storage!(comm, operation, values...)
    return _ordinary_vendor_backend_unavailable(operation)
end

# Do-block form: `_stage_vendor_call(operation, values...) do host ... end`.
_stage_vendor_call(f::Function, operation::Symbol, values...; kwargs...) =
    _stage_vendor_call(operation, f, values...; kwargs...)

function _stage_vendor_call_with_adapter(
        adapter, comm, operation::Symbol, f, values...; mutated::Tuple=(),
        validate_storage::Bool=true)
    validate_storage && ParallelExt._validate_parallel_storage!(
        comm, operation, values...; adapter,
    )
    return _ordinary_vendor_backend_unavailable(operation)
end

@noinline function _ordinary_vendor_backend_unavailable(operation::Symbol)
    throw(SHTnsKit.BackendUnavailableError(
        operation,
        "distributed GPU execution is not device-native for this API; " *
        "use a DistTransposePlan native transform or CPU storage",
    ))
end

# Explicit internal hook for policy/cache tests.  Ordinary public mathematical
# dispatch never calls this helper: full-field host staging is permitted only
# inside the MPI collective implementation in `ParallelGPU.jl`.
function _internal_staged_vendor_call_with_adapter(
        adapter, comm, operation::Symbol, f, values...; mutated::Tuple=(),
        validate_storage::Bool=true)
    validate_storage && ParallelExt._validate_parallel_storage!(
        comm, operation, values...; adapter,
    )
    return ParallelExt._staged_gpu_call(
        adapter, operation, comm, f, values...;
        mutated, validate_storage=false,
    )
end

# The ordinary full-grid APIs already have device-native transpose-plan entry
# points in the parallel extension and the vendor-specific kernels below this
# include.  All other compound APIs fail at this single boundary before a
# generic CPU PencilArray method can copy or index device storage.

function _distributed_spectral_reduce_vendor(
        plan::ParallelExt.DistributedSpectralPlan,
        local_contrib, result)
    return _stage_vendor_call(
        :distributed_spectral_reduce!,
        (host_local, host_result) ->
            ParallelExt.distributed_spectral_reduce!(
                plan, host_local, host_result,
            ),
        local_contrib, result; mutated=(2,), comm=plan.comm,
    )
end

function ParallelExt.distributed_spectral_reduce!(
        plan::ParallelExt.DistributedSpectralPlan,
        local_contrib::VendorArray, result::AbstractMatrix)
    return _distributed_spectral_reduce_vendor(plan, local_contrib, result)
end

function ParallelExt.distributed_spectral_reduce!(
        plan::ParallelExt.DistributedSpectralPlan,
        local_contrib::AbstractMatrix, result::VendorArray)
    return _distributed_spectral_reduce_vendor(plan, local_contrib, result)
end

function ParallelExt.distributed_spectral_reduce!(
        plan::ParallelExt.DistributedSpectralPlan,
        local_contrib::VendorArray, result::VendorArray)
    return _distributed_spectral_reduce_vendor(plan, local_contrib, result)
end

function ParallelExt.dist_analysis_distributed(
        cfg::SHTnsKit.SHTConfig, field::VendorPencilArray;
        plan::ParallelExt.DistributedSpectralPlan, kwargs...)
    return _stage_vendor_call(
        :dist_analysis_distributed,
        host -> ParallelExt.dist_analysis_distributed(
            cfg, host; plan, kwargs...,
        ),
        field; comm=plan.comm,
    )
end


function ParallelExt.dist_analysis_distributed_2d(
        cfg::SHTnsKit.SHTConfig, field::VendorPencilArray;
        plan::ParallelExt.DistributedSpectralPlan2D, kwargs...)
    return _stage_vendor_call(
        :dist_analysis_distributed_2d,
        host -> ParallelExt.dist_analysis_distributed_2d(
            cfg, host; plan, kwargs...,
        ),
        field; comm=plan.comm,
    )
end

function SHTnsKit.matrix_to_spectral_pencil(
        cfg::SHTnsKit.SHTConfig, coefficients::VendorArray;
        comm=MPI.COMM_WORLD)
    return _stage_vendor_call(
        :matrix_to_spectral_pencil,
        host -> SHTnsKit.matrix_to_spectral_pencil(cfg, host; comm),
        coefficients; comm,
    )
end

function SHTnsKit.spectral_pencil_to_matrix(
        cfg::SHTnsKit.SHTConfig, coefficients::VendorPencilArray;
        comm=nothing)
    trusted_comm = ParallelExt.communicator(coefficients)
    return _stage_vendor_call(
        :spectral_pencil_to_matrix,
        host -> SHTnsKit.spectral_pencil_to_matrix(cfg, host; comm),
        coefficients; comm=trusted_comm,
    )
end

function SHTnsKit.analysis(cfg::SHTnsKit.SHTConfig,
                           field::VendorPencilArray; kwargs...)
    return _stage_vendor_call(:analysis, field) do host
        SHTnsKit.analysis(cfg, host; kwargs...)
    end
end

function SHTnsKit.synthesis(cfg::SHTnsKit.SHTConfig,
                            coefficients::VendorPencilArray;
                            prototype_θφ::PencilArrays.PencilArray, kwargs...)
    return _stage_vendor_call(
        :synthesis,
        (host_coefficients, host_prototype) -> SHTnsKit.synthesis(
            cfg, host_coefficients; prototype_θφ=host_prototype, kwargs...,
        ),
        coefficients, prototype_θφ;
        comm=ParallelExt.communicator(prototype_θφ),
    )
end

function SHTnsKit.synthesis_cplx(cfg::SHTnsKit.SHTConfig,
                                 coefficients::VendorPencilArray;
                                 prototype_θφ::PencilArrays.PencilArray)
    return _stage_vendor_call(
        :synthesis_cplx,
        (host_coefficients, host_prototype) -> SHTnsKit.synthesis_cplx(
            cfg, host_coefficients; prototype_θφ=host_prototype,
        ), coefficients, prototype_θφ;
        comm=ParallelExt.communicator(prototype_θφ),
    )
end

function SHTnsKit.analysis_sphtor(cfg::SHTnsKit.SHTConfig,
                                  Vt::VendorPencilArray,
                                  Vp::PencilArrays.PencilArray; kwargs...)
    return _stage_vendor_call(
        :analysis_sphtor,
        (host_vt, host_vp) -> SHTnsKit.analysis_sphtor(
            cfg, host_vt, host_vp; kwargs...,
        ),
        Vt, Vp,
    )
end

function SHTnsKit.synthesis_sphtor(cfg::SHTnsKit.SHTConfig,
                                   S::VendorPencilArray,
                                   T::PencilArrays.PencilArray;
                                   prototype_θφ::PencilArrays.PencilArray, kwargs...)
    return _stage_vendor_call(
        :synthesis_sphtor,
        (host_s, host_t, host_prototype) -> SHTnsKit.synthesis_sphtor(
            cfg, host_s, host_t; prototype_θφ=host_prototype, kwargs...,
        ),
        S, T, prototype_θφ;
        comm=ParallelExt.communicator(prototype_θφ),
    )
end

function SHTnsKit.analysis_qst(cfg::SHTnsKit.SHTConfig,
                               Vr::VendorPencilArray,
                               Vt::PencilArrays.PencilArray,
                               Vp::PencilArrays.PencilArray; kwargs...)
    return _stage_vendor_call(
        :analysis_qst,
        (host_vr, host_vt, host_vp) -> SHTnsKit.analysis_qst(
            cfg, host_vr, host_vt, host_vp; kwargs...,
        ),
        Vr, Vt, Vp,
    )
end

function SHTnsKit.synthesis_qst(cfg::SHTnsKit.SHTConfig,
                                Q::VendorPencilArray,
                                S::PencilArrays.PencilArray,
                                T::PencilArrays.PencilArray;
                                prototype_θφ::PencilArrays.PencilArray, kwargs...)
    return _stage_vendor_call(
        :synthesis_qst,
        (host_q, host_s, host_t, host_prototype) -> SHTnsKit.synthesis_qst(
            cfg, host_q, host_s, host_t;
            prototype_θφ=host_prototype, kwargs...,
        ),
        Q, S, T, prototype_θφ;
        comm=ParallelExt.communicator(prototype_θφ),
    )
end

# Same-shape local spectral APIs are owned here so device storage cannot fall
# through to CPU indexing. They remain unavailable until native kernels exist.
for name in (
        :divergence_from_spheroidal, :spheroidal_from_divergence,
        :vorticity_from_toroidal, :toroidal_from_vorticity,
        :energy_scalar, :energy_scalar_l_spectrum,
        :energy_scalar_m_spectrum, :grid_energy_scalar, :grid_enstrophy,
    )
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host; kwargs...)
        end
    end
end

for name in (
        :divergence_from_spheroidal!, :spheroidal_from_divergence!,
        :vorticity_from_toroidal!, :toroidal_from_vorticity!,
    )
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  output::VendorPencilArray,
                                  input::PencilArrays.PencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_output, host_input) -> SHTnsKit.$name(
                cfg, host_output, host_input,
            ),
            output, input; mutated=(1,),
            comm=ParallelExt.communicator(input),
        )
    end
end

function SHTnsKit.dist_apply_laplacian!(cfg::SHTnsKit.SHTConfig,
                                        coefficients::VendorPencilArray)
    return _stage_vendor_call(
        :dist_apply_laplacian!,
        host -> SHTnsKit.dist_apply_laplacian!(cfg, host), coefficients;
        mutated=(1,),
    )
end

# Native transpose-plan entry points.  These stay entirely device-resident:
# PencilFFTs owns the distributed FFT/transpose and the compound extension owns
# the vendor Legendre kernels selected by these helpers.
function SHTnsKit.dist_analysis!(plan::ParallelExt.DistTransposePlan,
                                 output::VendorPencilArray,
                                 input::PencilArrays.PencilArray)
    ParallelExt._validate_transpose_call!(
        plan, :dist_analysis_transpose; spatial=(input,), spectral=(output,),
    )
    return ParallelExt._dist_transpose_gpu_analysis!(
        _vendor_adapter(input), plan, output, input,
    )
end

function SHTnsKit.dist_synthesis!(plan::ParallelExt.DistTransposePlan,
                                  output::VendorPencilArray,
                                  input::PencilArrays.PencilArray)
    ParallelExt._validate_transpose_call!(
        plan, :dist_synthesis_transpose; spatial=(output,), spectral=(input,),
    )
    return ParallelExt._dist_transpose_gpu_synthesis!(
        _vendor_adapter(output), plan, output, input,
    )
end

# The vector transposes use the CPU methods' operand rules, including the
# collective `with_vector` check: outputs must be distinct, while the read-only
# inputs may alias (one field passed as both components).
function SHTnsKit.dist_analysis_sphtor!(plan::ParallelExt.DistTransposePlan,
                                        S::VendorPencilArray,
                                        T::PencilArrays.PencilArray,
                                        Vt::PencilArrays.PencilArray,
                                        Vp::PencilArrays.PencilArray)
    ParallelExt._require_transpose_operands(
        plan, :dist_analysis_sphtor_transpose,
        (Vt, Vp), (S, T), (), (S, T), true,
    )
    return ParallelExt._dist_transpose_gpu_vector_analysis!(
        _vendor_adapter(Vt), plan, S, T, Vt, Vp,
    )
end

function SHTnsKit.dist_synthesis_sphtor!(plan::ParallelExt.DistTransposePlan,
                                         Vt::VendorPencilArray,
                                         Vp::PencilArrays.PencilArray,
                                         S::PencilArrays.PencilArray,
                                         T::PencilArrays.PencilArray)
    ParallelExt._require_transpose_operands(
        plan, :dist_synthesis_sphtor_transpose,
        (Vt, Vp), (S, T), (Vt, Vp), (), true,
    )
    return ParallelExt._dist_transpose_gpu_vector_synthesis!(
        _vendor_adapter(Vt), plan, Vt, Vp, S, T,
    )
end

function SHTnsKit.dist_analysis_qst!(plan::ParallelExt.DistTransposePlan,
                                     Q::VendorPencilArray,
                                     S::PencilArrays.PencilArray,
                                     T::PencilArrays.PencilArray,
                                     Vr::PencilArrays.PencilArray,
                                     Vt::PencilArrays.PencilArray,
                                     Vp::PencilArrays.PencilArray)
    ParallelExt._validate_transpose_qst_call!(
        plan, :dist_analysis_qst_transpose;
        spatial=(Vr, Vt, Vp), spectral=(Q, S, T),
    )
    SHTnsKit.dist_analysis!(plan, Q, Vr)
    SHTnsKit.dist_analysis_sphtor!(plan, S, T, Vt, Vp)
    return Q, S, T
end

function SHTnsKit.dist_synthesis_qst!(plan::ParallelExt.DistTransposePlan,
                                      Vr::VendorPencilArray,
                                      Vt::PencilArrays.PencilArray,
                                      Vp::PencilArrays.PencilArray,
                                      Q::PencilArrays.PencilArray,
                                      S::PencilArrays.PencilArray,
                                      T::PencilArrays.PencilArray)
    ParallelExt._validate_transpose_qst_call!(
        plan, :dist_synthesis_qst_transpose;
        spatial=(Vr, Vt, Vp), spectral=(Q, S, T),
    )
    SHTnsKit.dist_synthesis!(plan, Vr, Q)
    SHTnsKit.dist_synthesis_sphtor!(plan, Vt, Vp, S, T)
    return Vr, Vt, Vp
end

# Complex aliases and degree/order-restricted transforms.
for name in (:analysis_sphtor_cplx,)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  Vt::VendorPencilArray,
                                  Vp::PencilArrays.PencilArray; kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_vt, host_vp) -> SHTnsKit.$name(
                cfg, host_vt, host_vp; kwargs...,
            ), Vt, Vp,
        )
    end
end

for name in (:synthesis_sphtor_cplx,)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  S::VendorPencilArray,
                                  T::PencilArrays.PencilArray;
                                  prototype_θφ::PencilArrays.PencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_s, host_t, host_prototype) -> SHTnsKit.$name(
                cfg, host_s, host_t;
                prototype_θφ=host_prototype, kwargs...,
            ), S, T, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
        )
    end
end

for name in (:synthesis_sph, :synthesis_sph_cplx,
             :synthesis_tor, :synthesis_tor_cplx, :synthesis_grad)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorPencilArray;
                                  prototype_θφ::PencilArrays.PencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_coefficients, host_prototype) -> SHTnsKit.$name(
                cfg, host_coefficients;
                prototype_θφ=host_prototype, kwargs...,
            ), coefficients, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
        )
    end
end

function SHTnsKit.analysis_sphtor_l(cfg::SHTnsKit.SHTConfig,
                                    Vt::VendorPencilArray,
                                    Vp::PencilArrays.PencilArray, ltr::Integer)
    return _stage_vendor_call(
        :analysis_sphtor_l,
        (host_vt, host_vp) -> SHTnsKit.analysis_sphtor_l(
            cfg, host_vt, host_vp, ltr,
        ), Vt, Vp,
    )
end

for name in (:synthesis_sphtor_l, :synthesis_sphtor_l_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  S::VendorPencilArray,
                                  T::PencilArrays.PencilArray, ltr::Integer;
                                  prototype_θφ::PencilArrays.PencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_s, host_t, host_prototype) -> SHTnsKit.$name(
                cfg, host_s, host_t, ltr;
                prototype_θφ=host_prototype, kwargs...,
            ), S, T, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
        )
    end
end

for name in (:synthesis_sph_l, :synthesis_tor_l, :synthesis_grad_l)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorPencilArray,
                                  ltr::Integer;
                                  prototype_θφ::PencilArrays.PencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_coefficients, host_prototype) -> SHTnsKit.$name(
                cfg, host_coefficients, ltr;
                prototype_θφ=host_prototype, kwargs...,
            ), coefficients, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
        )
    end
end

for (complex_name, real_name) in (
    (:synthesis_sph_l_cplx, :synthesis_sph_l),
    (:synthesis_tor_l_cplx, :synthesis_tor_l),
)
    @eval function SHTnsKit.$complex_name(
        cfg::SHTnsKit.SHTConfig, coefficients::VendorPencilArray,
        ltr::Integer; prototype_θφ::PencilArrays.PencilArray,
    )
        return _stage_vendor_call(
            $(QuoteNode(complex_name)),
            (host_coefficients, host_prototype) -> SHTnsKit.$real_name(
                cfg, host_coefficients, ltr;
                prototype_θφ=host_prototype, real_output=false,
            ), coefficients, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
        )
    end
end

function SHTnsKit.analysis_sphtor_ml(cfg::SHTnsKit.SHTConfig,
                                     stored_im::Integer,
                                     Vt::VendorPencilArray,
                                     Vp::PencilArrays.PencilArray, ltr::Integer)
    return _stage_vendor_call(
        :analysis_sphtor_ml,
        (host_vt, host_vp) -> SHTnsKit.analysis_sphtor_ml(
            cfg, stored_im, host_vt, host_vp, ltr,
        ), Vt, Vp,
    )
end

function SHTnsKit.synthesis_sphtor_ml(cfg::SHTnsKit.SHTConfig,
                                      stored_im::Integer,
                                      S::VendorPencilArray,
                                      T::PencilArrays.PencilArray, ltr::Integer)
    return _stage_vendor_call(
        :synthesis_sphtor_ml,
        (host_s, host_t) -> SHTnsKit.synthesis_sphtor_ml(
            cfg, stored_im, host_s, host_t, ltr,
        ), S, T,
    )
end

for name in (:synthesis_sph_ml, :synthesis_tor_ml, :synthesis_grad_ml)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  stored_im::Integer,
                                  coefficients::VendorPencilArray,
                                  ltr::Integer)
        return _stage_vendor_call($(QuoteNode(name)), coefficients) do host
            SHTnsKit.$name(cfg, stored_im, host, ltr)
        end
    end
end

function SHTnsKit.analysis_qst_cplx(cfg::SHTnsKit.SHTConfig,
                                    Vr::VendorPencilArray,
                                    Vt::PencilArrays.PencilArray,
                                    Vp::PencilArrays.PencilArray; kwargs...)
    return _stage_vendor_call(
        :analysis_qst_cplx,
        (host_vr, host_vt, host_vp) -> SHTnsKit.analysis_qst_cplx(
            cfg, host_vr, host_vt, host_vp; kwargs...,
        ), Vr, Vt, Vp,
    )
end

function SHTnsKit.analysis_qst_l(cfg::SHTnsKit.SHTConfig,
                                 Vr::VendorPencilArray,
                                 Vt::PencilArrays.PencilArray,
                                 Vp::PencilArrays.PencilArray, ltr::Integer)
    return _stage_vendor_call(
        :analysis_qst_l,
        (host_vr, host_vt, host_vp) -> SHTnsKit.analysis_qst_l(
            cfg, host_vr, host_vt, host_vp, ltr,
        ), Vr, Vt, Vp,
    )
end

function SHTnsKit.synthesis_qst_cplx(cfg::SHTnsKit.SHTConfig,
                                     Q::VendorPencilArray,
                                     S::PencilArrays.PencilArray,
                                     T::PencilArrays.PencilArray;
                                     prototype_θφ::PencilArrays.PencilArray)
    return _stage_vendor_call(
        :synthesis_qst_cplx,
        (host_q, host_s, host_t, host_prototype) ->
            SHTnsKit.synthesis_qst_cplx(
                cfg, host_q, host_s, host_t;
                prototype_θφ=host_prototype,
            ), Q, S, T, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
    )
end


for name in (:synthesis_qst_l, :synthesis_qst_l_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  Q::VendorPencilArray,
                                  S::PencilArrays.PencilArray,
                                  T::PencilArrays.PencilArray, ltr::Integer;
                                  prototype_θφ::PencilArrays.PencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_q, host_s, host_t, host_prototype) -> SHTnsKit.$name(
                cfg, host_q, host_s, host_t, ltr;
                prototype_θφ=host_prototype, kwargs...,
            ), Q, S, T, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
        )
    end
end

for name in (:analysis_qst_ml, :synthesis_qst_ml)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  stored_im::Integer,
                                  Q::VendorPencilArray,
                                  S::PencilArrays.PencilArray,
                                  T::PencilArrays.PencilArray, ltr::Integer)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_q, host_s, host_t) -> SHTnsKit.$name(
                cfg, stored_im, host_q, host_s, host_t, ltr,
            ), Q, S, T,
        )
    end
end

# Packed, axisymmetric, batch, and local-evaluation families.
for name in (:analysis_packed, :analysis_packed_cplx, :analysis_axisym)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host; kwargs...)
        end
    end
end

for name in (:analysis_packed_l, :analysis_packed_cplx_l, :analysis_axisym_l)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray, ltr::Integer;
                                  kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host, ltr; kwargs...)
        end
    end
end

for name in (:synthesis_packed, :synthesis_packed_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray;
                                  prototype_θφ::PencilArrays.PencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_prototype) -> SHTnsKit.$name(
                cfg, host_input;
                prototype_θφ=host_prototype, kwargs...,
            ), input, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
        )
    end
end

for name in (:synthesis_packed_l, :synthesis_packed_cplx_l)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray, ltr::Integer;
                                  prototype_θφ::PencilArrays.PencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_prototype) -> SHTnsKit.$name(
                cfg, host_input, ltr;
                prototype_θφ=host_prototype, kwargs...,
            ), input, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
        )
    end
end

function SHTnsKit.synthesis_axisym(cfg::SHTnsKit.SHTConfig,
                                   input::VendorPencilArray)
    return _stage_vendor_call(:synthesis_axisym, input) do host
        SHTnsKit.synthesis_axisym(cfg, host)
    end
end

function SHTnsKit.synthesis_axisym_l(cfg::SHTnsKit.SHTConfig,
                                     input::VendorPencilArray, ltr::Integer)
    return _stage_vendor_call(:synthesis_axisym_l, input) do host
        SHTnsKit.synthesis_axisym_l(cfg, host, ltr)
    end
end

function SHTnsKit.analysis_packed_ml(cfg::SHTnsKit.SHTConfig,
                                     stored_im::Int,
                                     field::VendorPencilArray, ltr::Integer)
    return _stage_vendor_call(:analysis_packed_ml, field) do host
        SHTnsKit.analysis_packed_ml(cfg, stored_im, host, ltr)
    end
end

function SHTnsKit.synthesis_packed_ml(cfg::SHTnsKit.SHTConfig,
                                      stored_im::Int,
                                      coefficients::VendorPencilArray,
                                      ltr::Integer)
    return _stage_vendor_call(:synthesis_packed_ml, coefficients) do host
        SHTnsKit.synthesis_packed_ml(cfg, stored_im, host, ltr)
    end
end

for name in (:analysis_batch,)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host; kwargs...)
        end
    end
end

function SHTnsKit.analysis_batch!(cfg::SHTnsKit.SHTConfig,
                                   output::VendorPencilArray,
                                   input::PencilArrays.PencilArray; kwargs...)
    return _stage_vendor_call(
        :analysis_batch!,
        (host_output, host_input) -> SHTnsKit.analysis_batch!(
            cfg, host_output, host_input; kwargs...,
        ), output, input; mutated=(1,),
        comm=ParallelExt.communicator(input),
    )
end

for name in (:synthesis_batch, :synthesis_batch_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray;
                                  prototype_θφ::PencilArrays.PencilArray, kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_prototype) -> SHTnsKit.$name(
                cfg, host_input; prototype_θφ=host_prototype, kwargs...,
            ), input, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
        )
    end
end

function SHTnsKit.synthesis_batch!(cfg::SHTnsKit.SHTConfig,
                                    output::VendorPencilArray,
                                    input::PencilArrays.PencilArray;
                                    prototype_θφ::PencilArrays.PencilArray=output,
                                    kwargs...)
    return _stage_vendor_call(
        :synthesis_batch!,
        (host_output, host_input, host_prototype) ->
            SHTnsKit.synthesis_batch!(
                cfg, host_output, host_input;
                prototype_θφ=host_prototype, kwargs...,
            ),
        output, input, prototype_θφ; mutated=(1,),
        comm=ParallelExt.communicator(output),
    )
end

for name in (:analysis_sphtor_batch, :synthesis_sphtor_batch,
             :synthesis_sphtor_batch_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  first::VendorPencilArray,
                                  second::PencilArrays.PencilArray; kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_first, host_second) -> SHTnsKit.$name(
                cfg, host_first, host_second; kwargs...,
            ), first, second,
        )
    end
end

for name in (:analysis_qst_batch, :synthesis_qst_batch,
             :synthesis_qst_batch_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  first::VendorPencilArray,
                                  second::PencilArrays.PencilArray,
                                  third::PencilArrays.PencilArray; kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_first, host_second, host_third) -> SHTnsKit.$name(
                cfg, host_first, host_second, host_third; kwargs...,
            ), first, second, third,
        )
    end
end

for name in (:synthesis_point, :synthesis_point_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorPencilArray,
                                  cost::Real, phi::Real)
        return _stage_vendor_call($(QuoteNode(name)), coefficients) do host
            SHTnsKit.$name(cfg, host, cost, phi)
        end
    end
end

for name in (:SH_to_lat, :SH_to_lat_cplx)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorPencilArray,
                                  cost::Real; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), coefficients) do host
            SHTnsKit.$name(cfg, host, cost; kwargs...)
        end
    end
end

for name in (:SHqst_to_point, :SH_to_grad_point)
    # SH_to_grad_point is covered by the two-input prefix of this family.
    if name === :SHqst_to_point
        @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                      Q::VendorPencilArray,
                                      S::PencilArrays.PencilArray,
                                      T::PencilArrays.PencilArray,
                                      cost::Real, phi::Real)
            return _stage_vendor_call(
                $(QuoteNode(name)),
                (host_q, host_s, host_t) -> SHTnsKit.$name(
                    cfg, host_q, host_s, host_t, cost, phi,
                ), Q, S, T,
            )
        end
    end
end

function SHTnsKit.SH_to_grad_point(cfg::SHTnsKit.SHTConfig,
                                   Dr::VendorPencilArray,
                                   S::PencilArrays.PencilArray,
                                   cost::Real, phi::Real)
    return _stage_vendor_call(
        :SH_to_grad_point,
        (host_dr, host_s) -> SHTnsKit.SH_to_grad_point(
            cfg, host_dr, host_s, cost, phi,
        ), Dr, S,
    )
end

function SHTnsKit.SHqst_to_lat(cfg::SHTnsKit.SHTConfig,
                               Q::VendorPencilArray,
                               S::PencilArrays.PencilArray,
                               T::PencilArrays.PencilArray, cost::Real; kwargs...)
    return _stage_vendor_call(
        :SHqst_to_lat,
        (host_q, host_s, host_t) -> SHTnsKit.SHqst_to_lat(
            cfg, host_q, host_s, host_t, cost; kwargs...,
        ), Q, S, T,
    )
end

# Preserved `dist_*` local-evaluation names. These must be intercepted before
# `ParallelLocal.jl` indexes a GPU-backed PencilArray.
function SHTnsKit.dist_SH_to_lat(cfg::SHTnsKit.SHTConfig,
                                 coefficients::VendorPencilArray,
                                 cost::Real; kwargs...)
    return _stage_vendor_call(:dist_SH_to_lat, coefficients) do host
        SHTnsKit.dist_SH_to_lat(cfg, host, cost; kwargs...)
    end
end


function SHTnsKit.dist_SH_to_point(cfg::SHTnsKit.SHTConfig,
                                   coefficients::VendorPencilArray,
                                   cost::Real, phi::Real)
    return _stage_vendor_call(:dist_SH_to_point, coefficients) do host
        SHTnsKit.dist_SH_to_point(cfg, host, cost, phi)
    end
end


function SHTnsKit.dist_SHqst_to_point(cfg::SHTnsKit.SHTConfig,
                                      Q::VendorPencilArray,
                                      S::PencilArrays.PencilArray,
                                      T::PencilArrays.PencilArray,
                                      cost::Real, phi::Real)
    return _stage_vendor_call(
        :dist_SHqst_to_point,
        (host_q, host_s, host_t) -> SHTnsKit.dist_SHqst_to_point(
            cfg, host_q, host_s, host_t, cost, phi,
        ), Q, S, T,
    )
end


function SHTnsKit.dist_SHqst_to_lat(cfg::SHTnsKit.SHTConfig,
                                    Q::VendorPencilArray,
                                    S::PencilArrays.PencilArray,
                                    T::PencilArrays.PencilArray,
                                    cost::Real; kwargs...)
    return _stage_vendor_call(
        :dist_SHqst_to_lat,
        (host_q, host_s, host_t) -> SHTnsKit.dist_SHqst_to_lat(
            cfg, host_q, host_s, host_t, cost; kwargs...,
        ), Q, S, T,
    )
end


function SHTnsKit.dist_analysis_packed(cfg::SHTnsKit.SHTConfig,
                                       field::VendorPencilArray; kwargs...)
    return _stage_vendor_call(:dist_analysis_packed, field) do host
        SHTnsKit.dist_analysis_packed(cfg, host; kwargs...)
    end
end


function SHTnsKit.dist_synthesis_packed(
        cfg::SHTnsKit.SHTConfig, coefficients::VendorArray;
        prototype_θφ::PencilArrays.PencilArray, kwargs...)
    return _stage_vendor_call(
        :dist_synthesis_packed,
        (host_coefficients, host_prototype) ->
            SHTnsKit.dist_synthesis_packed(
                cfg, host_coefficients;
                prototype_θφ=host_prototype, kwargs...,
            ), coefficients, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
    )
end


function SHTnsKit.dist_analysis_packed_cplx(
        cfg::SHTnsKit.SHTConfig, field::VendorPencilArray)
    return _stage_vendor_call(:dist_analysis_packed_cplx, field) do host
        SHTnsKit.dist_analysis_packed_cplx(cfg, host)
    end
end


function SHTnsKit.dist_synthesis_packed_cplx(
        cfg::SHTnsKit.SHTConfig, coefficients::VendorArray;
        prototype_θφ::PencilArrays.PencilArray)
    return _stage_vendor_call(
        :dist_synthesis_packed_cplx,
        (host_coefficients, host_prototype) ->
            SHTnsKit.dist_synthesis_packed_cplx(
                cfg, host_coefficients; prototype_θφ=host_prototype,
            ), coefficients, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
    )
end

# Preserved `dist_*` aliases use the same early-error boundary for cfg-form calls.
function SHTnsKit.dist_analysis(cfg::SHTnsKit.SHTConfig,
                                field::VendorPencilArray; kwargs...)
    return _stage_vendor_call(:dist_analysis, field) do host
        SHTnsKit.dist_analysis(cfg, host; kwargs...)
    end
end

function SHTnsKit.dist_synthesis(cfg::SHTnsKit.SHTConfig,
                                 coefficients::VendorPencilArray;
                                 prototype_θφ::PencilArrays.PencilArray, kwargs...)
    return _stage_vendor_call(
        :dist_synthesis,
        (host_coefficients, host_prototype) -> SHTnsKit.dist_synthesis(
            cfg, host_coefficients;
            prototype_θφ=host_prototype, kwargs...,
        ), coefficients, prototype_θφ;
        comm=ParallelExt.communicator(prototype_θφ),
    )
end

# Dense compatibility analyses return replicated vendor matrices.  These
# overloads are deliberately limited to a vendor coefficient array together
# with a distributed Pencil prototype: serial GPU transforms have no such
# prototype and distributed Pencil coefficients keep their more-specific
# VendorPencilArray methods above.
function SHTnsKit.dist_synthesis(
        cfg::SHTnsKit.SHTConfig, coefficients::VendorArray;
        prototype_θφ::PencilArrays.PencilArray, Aminus=nothing, kwargs...)
    comm = ParallelExt.communicator(prototype_θφ)
    ParallelExt._validate_dense_scalar_synthesis_storage!(
        comm, coefficients, prototype_θφ, Aminus,
    )
    adapter = ParallelExt._parallel_gpu_adapter(coefficients)
    return _dist_synthesis_dense_vendor(
        adapter, comm, cfg, coefficients, prototype_θφ;
        Aminus, storage_prevalidated=true, kwargs...,
    )
end

function _dist_synthesis_dense_vendor(
        adapter, comm, cfg, coefficients, prototype_θφ;
        Aminus=nothing, storage_prevalidated::Bool=false, kwargs...)
    has_minus = storage_prevalidated ? Aminus !== nothing :
        ParallelExt._validate_dense_scalar_synthesis_storage!(
            comm, coefficients, prototype_θφ, Aminus,
        )
    if !has_minus
        return _stage_vendor_call_with_adapter(
            adapter, comm,
            :dist_synthesis_dense,
            (host_coefficients, host_prototype) -> SHTnsKit.dist_synthesis(
                cfg, host_coefficients;
                prototype_θφ=host_prototype, Aminus=nothing, kwargs...,
            ), coefficients, prototype_θφ; validate_storage=false,
        )
    end
    return _stage_vendor_call_with_adapter(
        adapter, comm,
        :dist_synthesis_dense,
        (host_coefficients, host_aminus, host_prototype) ->
            SHTnsKit.dist_synthesis(
                cfg, host_coefficients;
                prototype_θφ=host_prototype, Aminus=host_aminus, kwargs...,
        ), coefficients, Aminus, prototype_θφ; validate_storage=false,
    )
end

function SHTnsKit.dist_analysis_sphtor(cfg::SHTnsKit.SHTConfig,
                                       Vt::VendorPencilArray,
                                       Vp::PencilArrays.PencilArray; kwargs...)
    return _stage_vendor_call(
        :dist_analysis_sphtor,
        (host_vt, host_vp) -> SHTnsKit.dist_analysis_sphtor(
            cfg, host_vt, host_vp; kwargs...,
        ), Vt, Vp,
    )
end

function SHTnsKit.dist_synthesis_sphtor(cfg::SHTnsKit.SHTConfig,
                                        S::VendorPencilArray,
                                        T::PencilArrays.PencilArray;
                                        prototype_θφ::PencilArrays.PencilArray,
                                        kwargs...)
    return _stage_vendor_call(
        :dist_synthesis_sphtor,
        (host_s, host_t, host_prototype) ->
            SHTnsKit.dist_synthesis_sphtor(
                cfg, host_s, host_t;
                prototype_θφ=host_prototype, kwargs...,
            ), S, T, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
    )
end

function SHTnsKit.dist_synthesis_sphtor(
        cfg::SHTnsKit.SHTConfig, S::VendorArray, T::VendorArray;
        prototype_θφ::PencilArrays.PencilArray, kwargs...)
    comm = ParallelExt.communicator(prototype_θφ)
    ParallelExt._validate_dense_synthesis_storage!(
        comm, :dist_synthesis_sphtor_dense, S, T, prototype_θφ,
    )
    adapter = ParallelExt._parallel_gpu_adapter(S)
    return _dist_synthesis_sphtor_dense_vendor(
        adapter, comm, cfg, S, T, prototype_θφ;
        storage_prevalidated=true, kwargs...,
    )
end


function _dist_synthesis_sphtor_dense_vendor(
        adapter, comm, cfg, S, T, prototype_θφ;
        storage_prevalidated::Bool=false, kwargs...)
    storage_prevalidated || ParallelExt._validate_dense_synthesis_storage!(
        comm, :dist_synthesis_sphtor_dense, S, T, prototype_θφ,
    )
    return _stage_vendor_call_with_adapter(
        adapter, comm,
        :dist_synthesis_sphtor_dense,
        (host_s, host_t, host_prototype) ->
            SHTnsKit.dist_synthesis_sphtor(
                cfg, host_s, host_t;
                prototype_θφ=host_prototype, kwargs...,
            ), S, T, prototype_θφ; validate_storage=false,
    )
end

function SHTnsKit.dist_analysis_qst(cfg::SHTnsKit.SHTConfig,
                                    Vr::VendorPencilArray,
                                    Vt::PencilArrays.PencilArray,
                                    Vp::PencilArrays.PencilArray; kwargs...)
    return _stage_vendor_call(
        :dist_analysis_qst,
        (host_vr, host_vt, host_vp) -> SHTnsKit.dist_analysis_qst(
            cfg, host_vr, host_vt, host_vp; kwargs...,
        ), Vr, Vt, Vp,
    )
end

function SHTnsKit.dist_synthesis_qst(cfg::SHTnsKit.SHTConfig,
                                     Q::VendorPencilArray,
                                     S::PencilArrays.PencilArray,
                                     T::PencilArrays.PencilArray;
                                     prototype_θφ::PencilArrays.PencilArray,
                                     kwargs...)
    return _stage_vendor_call(
        :dist_synthesis_qst,
        (host_q, host_s, host_t, host_prototype) ->
            SHTnsKit.dist_synthesis_qst(
                cfg, host_q, host_s, host_t;
                prototype_θφ=host_prototype, kwargs...,
            ), Q, S, T, prototype_θφ;
            comm=ParallelExt.communicator(prototype_θφ),
    )
end

function SHTnsKit.dist_synthesis_qst(
        cfg::SHTnsKit.SHTConfig, Q::VendorArray, S::VendorArray,
        T::VendorArray; prototype_θφ::PencilArrays.PencilArray, kwargs...)
    comm = ParallelExt.communicator(prototype_θφ)
    ParallelExt._validate_dense_synthesis_storage!(
        comm, :dist_synthesis_qst_dense, Q, S, T, prototype_θφ,
    )
    adapter = ParallelExt._parallel_gpu_adapter(Q)
    return _dist_synthesis_qst_dense_vendor(
        adapter, comm, cfg, Q, S, T, prototype_θφ;
        storage_prevalidated=true, kwargs...,
    )
end


function _dist_synthesis_qst_dense_vendor(
        adapter, comm, cfg, Q, S, T, prototype_θφ;
        storage_prevalidated::Bool=false, kwargs...)
    storage_prevalidated || ParallelExt._validate_dense_synthesis_storage!(
        comm, :dist_synthesis_qst_dense, Q, S, T, prototype_θφ,
    )
    return _stage_vendor_call_with_adapter(
        adapter, comm,
        :dist_synthesis_qst_dense,
        (host_q, host_s, host_t, host_prototype) ->
            SHTnsKit.dist_synthesis_qst(
                cfg, host_q, host_s, host_t;
                prototype_θφ=host_prototype, kwargs...,
            ), Q, S, T, prototype_θφ; validate_storage=false,
    )
end

function SHTnsKit.dist_scalar_roundtrip!(cfg::SHTnsKit.SHTConfig,
                                         field::VendorPencilArray)
    return _stage_vendor_call(:dist_scalar_roundtrip!, field) do host
        SHTnsKit.dist_scalar_roundtrip!(cfg, host)
    end
end

function SHTnsKit.dist_vector_roundtrip!(cfg::SHTnsKit.SHTConfig,
                                         Vt::VendorPencilArray,
                                         Vp::PencilArrays.PencilArray)
    return _stage_vendor_call(
        :dist_vector_roundtrip!,
        (host_vt, host_vp) -> SHTnsKit.dist_vector_roundtrip!(
            cfg, host_vt, host_vp,
        ), Vt, Vp,
    )
end

# Local operators, spatial composites, rotations, and diagnostics.
function SHTnsKit.SH_mul_mx(::SHTnsKit.CPU, cfg::SHTnsKit.SHTConfig,
                            mx::AbstractVector{<:Real},
                            input::VendorPencilArray,
                            output::PencilArrays.PencilArray)
    return _stage_vendor_call(
        :SH_mul_mx,
        (host_input, host_output) -> SHTnsKit.SH_mul_mx(
            SHTnsKit.CPU(), cfg, mx, host_input, host_output,
        ), input, output; mutated=(2,),
    )
end

SHTnsKit.SH_mul_mx(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real},
                   input::VendorPencilArray,
                   output::PencilArrays.PencilArray) =
    SHTnsKit.SH_mul_mx(SHTnsKit.CPU(), cfg, mx, input, output)

SHTnsKit.dist_SH_mul_mx!(cfg::SHTnsKit.SHTConfig,
                         mx::AbstractVector{<:Real},
                         input::VendorPencilArray,
                         output::PencilArrays.PencilArray) =
    SHTnsKit.SH_mul_mx(SHTnsKit.CPU(), cfg, mx, input, output)

for name in (:dist_spatial_divergence, :dist_spatial_vorticity)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  first::VendorPencilArray,
                                  second::PencilArrays.PencilArray; kwargs...)
        prototype = get(kwargs, :prototype_θφ, first)
        forwarded = Base.structdiff((; kwargs...), (; prototype_θφ=prototype))
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_first, host_second, host_prototype) -> SHTnsKit.$name(
                cfg, host_first, host_second;
                prototype_θφ=host_prototype, forwarded...,
            ), first, second, prototype,
        )
    end
end

function SHTnsKit.dist_scalar_laplacian(cfg::SHTnsKit.SHTConfig,
                                        input::VendorPencilArray; kwargs...)
    prototype = get(kwargs, :prototype_θφ, input)
    forwarded = Base.structdiff((; kwargs...), (; prototype_θφ=prototype))
    return _stage_vendor_call(
        :dist_scalar_laplacian,
        (host_input, host_prototype) -> SHTnsKit.dist_scalar_laplacian(
            cfg, host_input;
            prototype_θφ=host_prototype, forwarded...,
        ), input, prototype,
    )
end

function SHTnsKit.dist_scalar_laplacian!(cfg::SHTnsKit.SHTConfig,
                                         output::VendorPencilArray,
                                         input::PencilArrays.PencilArray; kwargs...)
    return _stage_vendor_call(
        :dist_scalar_laplacian!,
        (host_output, host_input) -> SHTnsKit.dist_scalar_laplacian!(
            cfg, host_output, host_input; kwargs...,
        ), output, input; mutated=(1,),
        comm=ParallelExt.communicator(input),
    )
end

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray, angle::Real)
    return _stage_vendor_call(:dist_SH_Zrotate, input) do host
        SHTnsKit.dist_SH_Zrotate(cfg, host, angle)
    end
end

for name in (:dist_SH_Zrotate, :dist_SH_Yrotate,
             :dist_SH_Yrotate_allgatherm!,
             :dist_SH_Yrotate_truncgatherm!)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray, angle::Real,
                                  output::PencilArrays.PencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_output) -> SHTnsKit.$name(
                cfg, host_input, angle, host_output,
            ), input, output; mutated=(2,),
        )
    end
end

for name in (:dist_SH_Yrotate90, :dist_SH_Xrotate90)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray,
                                  output::PencilArrays.PencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_input, host_output) -> SHTnsKit.$name(
                cfg, host_input, host_output,
            ), input, output; mutated=(2,),
        )
    end
end

function SHTnsKit.dist_SH_rotate_euler(cfg::SHTnsKit.SHTConfig,
                                       input::VendorPencilArray,
                                       alpha::Real, beta::Real, gamma::Real,
                                       output::PencilArrays.PencilArray)
    return _stage_vendor_call(
        :dist_SH_rotate_euler,
        (host_input, host_output) -> SHTnsKit.dist_SH_rotate_euler(
            cfg, host_input, alpha, beta, gamma, host_output,
        ), input, output; mutated=(2,),
    )
end

for name in (:dist_SH_Zrotate_packed, :dist_SH_Yrotate_packed)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorArray, angle::Real;
                                  prototype_lm::PencilArrays.PencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_coefficients, host_prototype) -> SHTnsKit.$name(
                cfg, host_coefficients, angle; prototype_lm=host_prototype,
            ), coefficients, prototype_lm;
            comm=ParallelExt.communicator(prototype_lm),
        )
    end
end

for name in (:dist_SH_Yrotate90_packed, :dist_SH_Xrotate90_packed)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  coefficients::VendorArray;
                                  prototype_lm::PencilArrays.PencilArray)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_coefficients, host_prototype) -> SHTnsKit.$name(
                cfg, host_coefficients; prototype_lm=host_prototype,
            ), coefficients, prototype_lm;
            comm=ParallelExt.communicator(prototype_lm),
        )
    end
end

for name in (:enstrophy, :enstrophy_l_spectrum, :enstrophy_m_spectrum)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  input::VendorPencilArray; kwargs...)
        return _stage_vendor_call($(QuoteNode(name)), input) do host
            SHTnsKit.$name(cfg, host; kwargs...)
        end
    end
end

for name in (:energy_vector, :energy_vector_l_spectrum, :energy_vector_m_spectrum,
             :grid_energy_vector)
    @eval function SHTnsKit.$name(cfg::SHTnsKit.SHTConfig,
                                  first::VendorPencilArray,
                                  second::PencilArrays.PencilArray; kwargs...)
        return _stage_vendor_call(
            $(QuoteNode(name)),
            (host_first, host_second) -> SHTnsKit.$name(
                cfg, host_first, host_second; kwargs...,
            ), first, second,
        )
    end
end
