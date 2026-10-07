module SHTnsKitParallelADExt

#=
================================================================================
SHTnsKitParallelADExt — ChainRules rrules for distributed transforms
================================================================================

Loads only when ChainRulesCore + MPI + PencilArrays + PencilFFTs are all
present. Provides backward-pass rules for `dist_analysis` and `dist_synthesis`
so Zygote/ChainRules-based AD pipelines get accurate gradients through the
distributed spatial↔spectral path without falling back to source-level tracing
of MPI collectives.

Math summary
------------
Forward `dist_analysis(cfg, fθφ)` produces a fully-reduced `Alm` that is
identical on every rank (enforced by the contract upheld in `dist_synthesis`).
Its adjoint operator maps an `Alm̄` (also replicated) to a spatial cotangent
`f̄θφ` localized per-rank — this is exactly the local `_adjoint_analysis`
already implemented in `SHTnsKitAdvancedADExt` restricted to the rank's θ
slab. No inter-rank communication is needed for the backward pass: every
rank's θ rows are independent in the adjoint.

Forward `dist_synthesis(cfg, Alm; prototype_θφ)` maps a replicated `Alm` to a
distributed spatial field. Its adjoint maps a distributed spatial cotangent
`f̄_local` to a replicated `Ālm` — an Allreduce across ranks sums per-rank
contributions, which matches the adjoint of the implicit "broadcast Alm"
operation on the forward side.

SPMD contract for reverse mode
------------------------------
The pullbacks below run collectives (the Allreduce above, and the collective
validation of replicated cotangents), so every rank must run every pullback.
Zygote never runs the pullback of an output that a rank's loss does not use: a
ChainRules pullback is skipped for a `nothing` cotangent, and pruned reverse
statements are not executed at all. A loss like `rank == 0 ? sum(abs2, y) : 0.0`
therefore leaves the ranks that did use `y` blocked in the first collective; no
rule here can detect that, because the skipping rank never enters it. Each
rank's loss must use its local block of every distributed output (zero weight
is fine) and must be identical on every rank for replicated outputs.
docs/src/distributed.md states the same contract for users.
================================================================================
=#

using ChainRulesCore
using MPI
using PencilArrays
using PencilArrays: PencilArray
using SHTnsKit
using FFTW

# Distributed reverse rules in this extension use CPU FFTW/Legendre adjoints.
# Runtime storage classification accepts CPU wrappers (views, shared arrays,
# and custom host arrays) while preventing a vendor PencilArray from reaching
# an implicit host conversion. GPU-backed distributed AD requires a
# vendor-native compound rule; until one is available, fail at the storage
# boundary before running the forward transform.

@inline function _require_host_pencil(operation::Symbol, value::PencilArray,
                                      comm=communicator(value))
    local_ok = try
        SHTnsKit.on_device(parent(value)) isa SHTnsKit.CPU
    catch
        false
    end
    MPI.Allreduce(local_ok, &, comm) && return value
    throw(SHTnsKit.BackendUnavailableError(
        operation,
        "distributed reverse-mode AD for GPU-backed PencilArray storage requires a vendor-native compound rule",
    ))
end

@inline function _materialize_host_coefficient(operation::Symbol, value, cfg)
    value isa ChainRulesCore.AbstractZero &&
        return zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    SHTnsKit.on_device(value) isa SHTnsKit.CPU || throw(
        SHTnsKit.BackendUnavailableError(
            operation,
            "distributed reverse-mode AD coefficient cotangents must remain on the CPU for a host-backed PencilArray",
        ),
    )
    return ComplexF64.(value)
end

# ----- PencilArrays helpers -------------------------------------------------
# `communicator` and `globalindices` are internal helpers of the sibling
# SHTnsKitParallelExt module and are NOT exported by PencilArrays (verified
# across 0.19.8–0.19.11). This is a SEPARATE extension module, so without local
# definitions every rrule below throws `UndefVarError` the moment it fires.
# These mirror the primary 0.19 API used by the main extension.
@inline communicator(A) = PencilArrays.get_comm(A)

@inline globalindices(A, dim) = PencilArrays.range_local(PencilArrays.pencil(A))[dim]

# ----- helpers ---------------------------------------------------------------

"""
    _phi_window(φ_globals, nlon_local, cfg_nlon) -> (φ_is_local, φ_window)

The rank's global φ slice as a range, or `nothing` when φ is replicated.

A rank can legitimately own ZERO φ columns — a pencil with more partitions than
the dimension has points, e.g. `nlon = 4` on 5 ranks. The raw
`first(φ_globals)` throws `BoundsError` there, and because these rrules run
inside a collective region (the pullbacks `MPI.Allreduce` below) that kills one
rank while the others block forever, so the job hangs instead of failing. An
empty window is what the zero-pad path already wants: nothing is copied into
`f̄_full`, so this rank contributes an all-zero partial to the Allreduce.

Mirrors `_owned_range` in ext/ParallelTransforms.jl — duplicated because this is
a separate extension module and cannot see it.
"""
@inline function _phi_window(φ_globals, nlon_local::Int, cfg_nlon::Int)
    φ_is_local = (nlon_local == cfg_nlon)
    φ_is_local && return true, nothing
    isempty(φ_globals) && return false, 1:0
    φ_start = Int(first(φ_globals))
    return false, φ_start:(φ_start + nlon_local - 1)
end


# The rank-local adjoint is just the parametrized `SHTnsKit._adjoint_analysis`
# called with a restricted θ subset and optional φ-window.
@inline function _local_adjoint_analysis(cfg::SHTnsKit.SHTConfig, Alm̄,
                                          θ_globals::AbstractVector{<:Integer},
                                          φ_window)
    return SHTnsKit._adjoint_analysis(cfg, Alm̄; θ_globals=θ_globals, φ_window=φ_window)
end

"""Materialize and collectively verify one logically replicated cotangent."""
function _replicated_coeff_cotangent(cfg::SHTnsKit.SHTConfig, ȳ, comm;
                                     packed::Bool=false,
                                     operation::Symbol=:dist_analysis_pullback)
    ȳ = ChainRulesCore.unthunk(ȳ)
    is_zero = ȳ isa ChainRulesCore.AbstractZero

    # Every rank must take the same validation collectives. In particular, a
    # rank-local AbstractZero must not skip an Allreduce taken by nonzero peers:
    # that would mismatch this validation with the later Bcast and deadlock.
    local_host_ok = is_zero || try
        SHTnsKit.on_device(ȳ) isa SHTnsKit.CPU
    catch
        false
    end
    MPI.Allreduce(local_host_ok, &, comm) || throw(
        SHTnsKit.BackendUnavailableError(
            operation,
            "distributed reverse-mode AD coefficient cotangents must remain on the CPU for a host-backed PencilArray",
        ),
    )

    local_shape_ok = is_zero || try
        packed ? length(ȳ) == cfg.nlm :
                 (ȳ isa AbstractMatrix &&
                  size(ȳ) == (cfg.lmax + 1, cfg.mmax + 1))
    catch
        false
    end
    MPI.Allreduce(local_shape_ok, &, comm) || throw(DimensionMismatch(
        packed ? "distributed packed coefficient cotangent must have length $(cfg.nlm)" :
                 "distributed coefficient cotangent must have size $((cfg.lmax + 1, cfg.mmax + 1))",
    ))

    Alm̄ = nothing
    local_materialization_ok = true
    if is_zero
        Alm̄ = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    else
        Alm̄ = try
            converted = ComplexF64.(ȳ)
            packed ? SHTnsKit.unpack_lm(cfg, converted) : converted
        catch
            local_materialization_ok = false
            nothing
        end
    end
    MPI.Allreduce(local_materialization_ok, &, comm) || throw(ArgumentError(
        "distributed coefficient cotangent elements must be convertible to ComplexF64",
    ))

    if MPI.Comm_size(comm) > 1
        root_value = similar(Alm̄)
        if MPI.Comm_rank(comm) == 0
            copyto!(root_value, Alm̄)
        else
            fill!(root_value, zero(eltype(root_value)))
        end
        MPI.Bcast!(root_value, comm; root=0)
        # `==` is false for NaN, so a NaN cotangent that every rank shares
        # would be rejected as rank-varying; treat matching NaNs as equal.
        same = all(((a, b),) -> a == b || isequal(a, b), zip(Alm̄, root_value))
        MPI.Allreduce(same, &, comm) || throw(ArgumentError(
            "the cotangent of replicated distributed-analysis output must be identical on every rank; rank-varying partial cotangents are unsupported",
        ))
    end
    return Alm̄
end

"""
The cotangent a PencilArray output actually received.

A loss that reads `parent(y)` (or `y.data`) returns a structural tangent of the
PencilArray whose `data` field holds the rank-local block; Zygote marks a field
the loss never touched `nothing`.
"""
function _structural_pencil_data(ȳ)
    (ȳ isa ChainRulesCore.Tangent || ȳ isa NamedTuple) || return ȳ
    data = hasproperty(ȳ, :data) ? ChainRulesCore.unthunk(getproperty(ȳ, :data)) : nothing
    return data === nothing ? ChainRulesCore.ZeroTangent() : data
end

@inline _cotangent_eltype_code(::Type{Float32}) = 1
@inline _cotangent_eltype_code(::Type{Float64}) = 2
@inline _cotangent_eltype_code(::Type{ComplexF32}) = 3
@inline _cotangent_eltype_code(::Type{ComplexF64}) = 4
@inline _cotangent_eltype_code(::Type) = 0

"""
Collectively validate and materialize the rank-local block of a cotangent for a
distributed output laid out like `prototype` (a spatial field, or a spectral
pencil).
"""
function _local_spatial_cotangent(ȳ, prototype::PencilArray, comm;
                                   zero_eltype::Type,
                                   operation::Symbol)
    ȳ = _structural_pencil_data(ChainRulesCore.unthunk(ȳ))
    is_zero = ȳ isa ChainRulesCore.AbstractZero
    is_pencil = ȳ isa PencilArray
    local_flags = UInt32(0)
    local_value = nothing
    materialized = nothing

    # The adjoint result is subsequently reduced by MPI, so even otherwise
    # valid rank-local cotangents must be converted to one common element type.
    target_code = _cotangent_eltype_code(zero_eltype)
    root_target_code = MPI.bcast(target_code, 0, comm)
    (target_code != root_target_code || target_code == 0) &&
        (local_flags |= 0x0020)

    if !is_zero
        if is_pencil
            local_value = try
                parent(ȳ)
            catch
                local_flags |= 0x0001
                nothing
            end
        elseif ȳ isa AbstractMatrix
            local_value = ȳ
        else
            local_flags |= 0x0001
        end

        if local_value !== nothing
            host_ok = try
                SHTnsKit.on_device(local_value) isa SHTnsKit.CPU
            catch
                false
            end
            host_ok || (local_flags |= 0x0002)

            size_ok = try
                size(local_value) == size(parent(prototype))
            catch
                false
            end
            size_ok || (local_flags |= 0x0004)

            eltype_ok = try
                float(eltype(local_value))
                true
            catch
                false
            end
            eltype_ok || (local_flags |= 0x0010)

            if local_flags == 0
                materialized = try
                    eltype(local_value) === zero_eltype ? local_value :
                        Matrix{zero_eltype}(local_value)
                catch
                    local_flags |= 0x0010
                    nothing
                end
            end
        end

        if is_pencil
            layout_ok = try
                reference_pen = PencilArrays.pencil(prototype)
                candidate_pen = PencilArrays.pencil(ȳ)
                candidate_comm = communicator(ȳ)
                MPI.Comm_size(candidate_comm) == MPI.Comm_size(comm) &&
                    MPI.Comm_compare(candidate_comm, comm) in
                        (MPI.IDENT, MPI.CONGRUENT) &&
                    PencilArrays.size_global(ȳ) ==
                        PencilArrays.size_global(prototype) &&
                    PencilArrays.decomposition(candidate_pen) ==
                        PencilArrays.decomposition(reference_pen) &&
                    size(PencilArrays.topology(candidate_pen)) ==
                        size(PencilArrays.topology(reference_pen)) &&
                    PencilArrays.range_local(candidate_pen) ==
                        PencilArrays.range_local(reference_pen) &&
                    PencilArrays.permutation(ȳ) ==
                        PencilArrays.permutation(prototype)
            catch
                false
            end
            layout_ok || (local_flags |= 0x0008)
        end
    end

    # A single bitmask reduction gives every rank the same verdict before any
    # rank touches the cotangent's shape, eltype, or storage in the adjoint.
    flags = MPI.Allreduce(local_flags, |, comm)
    flags == 0 || begin
        flags & 0x0001 != 0 && throw(ArgumentError(
            "$operation spatial cotangent must be an AbstractMatrix, " *
            "PencilArray, or AbstractZero",
        ))
        flags & 0x0002 != 0 && throw(SHTnsKit.BackendUnavailableError(
            operation,
            "distributed reverse-mode AD spatial cotangents must remain on the CPU",
        ))
        flags & 0x0008 != 0 && throw(ArgumentError(
            "$operation PencilArray cotangent must match the spatial output layout and communicator",
        ))
        flags & 0x0004 != 0 && throw(DimensionMismatch(
            "$operation spatial cotangent must have rank-local size $(size(parent(prototype)))",
        ))
        throw(ArgumentError(
            "$operation spatial cotangent must have a floating-point-compatible element type",
        ))
    end

    return is_zero ? zeros(zero_eltype, size(parent(prototype))) : materialized
end

"""Collectively unpack the `n` component cotangents of a tuple-valued transform."""
function _cotangent_components(ȳ, n::Int, comm, operation::Symbol)
    ȳ = ChainRulesCore.unthunk(ȳ)
    components = ntuple(_ -> ChainRulesCore.ZeroTangent(), n)
    local_ok = true
    if !(ȳ isa ChainRulesCore.AbstractZero)
        try
            components = ntuple(n) do k
                component = ChainRulesCore.unthunk(ȳ[k])
                component === nothing ? ChainRulesCore.ZeroTangent() : component
            end
        catch
            local_ok = false
        end
    end
    MPI.Allreduce(local_ok, &, comm) || throw(ArgumentError(
        "$operation cotangent must contain $n components",
    ))
    return components
end

_cotangent_pair(ȳ, comm, operation::Symbol) = _cotangent_components(ȳ, 2, comm, operation)

function _require_ad_communicator_match(spectral::PencilArray,
                                        spatial::PencilArray)
    # Reduce the verdict on the spatial prototype's communicator. Throwing on
    # only the rank whose spectral operand uses COMM_SELF would strand peers.
    comm = communicator(spatial)
    local_ok = try
        MPI.Comm_compare(communicator(spectral), comm) in
            (MPI.IDENT, MPI.CONGRUENT)
    catch
        false
    end
    MPI.Allreduce(local_ok, &, comm) || throw(ArgumentError(
        "spectral and spatial PencilArrays must use communicators with the same process group and rank order",
    ))
    return nothing
end

"""Scatter a dense coefficient cotangent into a primal spectral pencil."""
function _scatter_spectral_tangent(primal::PencilArray, dense::AbstractMatrix)
    lr = collect(globalindices(primal, 1))
    mr = collect(globalindices(primal, 2))
    raw = Matrix{eltype(dense)}(undef, length(lr), length(mr))
    @inbounds for (jj, gm) in enumerate(mr), (ii, gl) in enumerate(lr)
        raw[ii, jj] = dense[gl, gm]
    end
    local_parent = ProjectTo(parent(primal))(raw)
    return PencilArray(PencilArrays.pencil(primal), local_parent)
end

"""Wrap a rank-local spatial adjoint block in the primal's pencil."""
_spatial_tangent(primal::PencilArray, values::AbstractMatrix) =
    PencilArray(PencilArrays.pencil(primal), ProjectTo(parent(primal))(values))

"""This rank's θ rows and φ window within a distributed spatial layout."""
function _spatial_geometry(prototype::PencilArray, cfg::SHTnsKit.SHTConfig)
    θ_globals = collect(globalindices(prototype, 1))
    φ_globals = collect(globalindices(prototype, 2))
    φ_is_local, φ_window = _phi_window(φ_globals, length(φ_globals), cfg.nlon)
    return θ_globals, φ_is_local, φ_window
end

"""
Widen a rank-local spatial cotangent block to the full longitude width.

The forward inverse FFT runs over all of φ before the rank's window is cut
out, so the adjoint zero-pads that window back; FFT linearity makes the sum
over ranks of the padded windows the transform of the full field.
"""
function _zero_padded_phi(values::AbstractMatrix, nθ_local::Int, nlon::Int,
                          φ_is_local::Bool, φ_window)
    full = zeros(float(eltype(values)), nθ_local, nlon)
    if φ_is_local
        full .= values
    else
        @views full[:, φ_window] .= values
    end
    return full
end

"""Zero the cotangent of degrees a truncated (`ltr < lmax`) synthesis ignores."""
function _truncate_degrees!(Ā::AbstractMatrix, ltr::Integer)
    ltr + 1 < size(Ā, 1) && fill!(view(Ā, (ltr + 2):size(Ā, 1), :), zero(eltype(Ā)))
    return Ā
end

"""
Replicated coefficient cotangent of a distributed scalar synthesis.

The adjoint of synthesis is `_adjoint_synthesis` (no quadrature weights or
cphi; those belong to analysis), applied to this rank's θ slab of the
cotangent and summed across ranks. Using `dist_analysis` instead would inject
the Gauss weights `w[θ]·cphi`.
"""
function _synthesis_coefficient_cotangent(cfg::SHTnsKit.SHTConfig, ȳ,
                                          prototype_θφ::PencilArray, comm;
                                          real_output::Bool, zero_eltype::Type,
                                          operation::Symbol)
    θ_globals, φ_is_local, φ_window = _spatial_geometry(prototype_θφ, cfg)
    ȳ_loc = _local_spatial_cotangent(ȳ, prototype_θφ, comm; zero_eltype, operation)
    f̄_full = _zero_padded_phi(ȳ_loc, length(θ_globals), cfg.nlon,
                               φ_is_local, φ_window)
    Ā_partial = SHTnsKit._adjoint_synthesis(cfg, f̄_full; θ_globals, real_output)
    return MPI.Allreduce(Ā_partial, +, comm)
end

"""
Replicated `(S̄, T̄)` cotangents of a distributed vector synthesis.

As in the scalar case, `_adjoint_synthesis_sphtor` carries none of the
`w[θ]·scaleφ/(l(l+1))` weights of vector analysis.
"""
function _synthesis_sphtor_coefficient_cotangents(cfg::SHTnsKit.SHTConfig, ȳ,
                                                  prototype_θφ::PencilArray, comm;
                                                  real_output::Bool, zero_eltypes,
                                                  operation::Symbol)
    V̄t, V̄p = _cotangent_pair(ȳ, comm, operation)
    θ_globals, φ_is_local, φ_window = _spatial_geometry(prototype_θφ, cfg)
    nθ_local = length(θ_globals)
    V̄t_loc = _local_spatial_cotangent(V̄t, prototype_θφ, comm;
                                      zero_eltype=zero_eltypes[1], operation)
    V̄p_loc = _local_spatial_cotangent(V̄p, prototype_θφ, comm;
                                      zero_eltype=zero_eltypes[2], operation)
    S̄, T̄ = SHTnsKit._adjoint_synthesis_sphtor(
        cfg,
        _zero_padded_phi(V̄t_loc, nθ_local, cfg.nlon, φ_is_local, φ_window),
        _zero_padded_phi(V̄p_loc, nθ_local, cfg.nlon, φ_is_local, φ_window);
        θ_globals, real_output,
    )
    # One Allreduce over the stacked pair, copied back into the matrices the
    # adjoint allocated: `combined[1:n]` would allocate two more full-size
    # arrays, and a reshaped view of `combined` is not a buffer a downstream
    # distributed pullback can hand to `MPI.Allreduce!`.
    n = length(S̄)
    combined = MPI.Allreduce!(vcat(vec(S̄), vec(T̄)), +, comm)
    copyto!(S̄, 1, combined, 1, n)
    copyto!(T̄, 1, combined, n + 1, length(combined) - n)
    return S̄, T̄
end

"""
Dense coefficient cotangent of an m-distributed spectral output.

Each rank holds the derivative of the loss with respect to its own block —
from a per-rank partial loss, or from a replicated loss whose distributed
reductions (such as the spectral-pencil energies below) return block-local
derivatives — so the dense cotangent is the sum of the blocks placed at their
global `(l, m)` indices.
"""
function _distributed_coeff_cotangent(cfg::SHTnsKit.SHTConfig, ȳ,
                                      output::PencilArray, comm;
                                      operation::Symbol)
    block = _local_spatial_cotangent(ȳ, output, comm;
                                     zero_eltype=ComplexF64, operation)
    dense = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    placed = PencilArrays.global_view(PencilArray(PencilArrays.pencil(output), block))
    for index in CartesianIndices(placed)
        dense[index] = placed[index]
    end
    return MPI.Allreduce!(dense, +, comm)
end

"""
Dense coefficient cotangents for the outputs of an analysis: summed blocks for
spectral PencilArrays, verified replicas for the dense `return_pencil=false`
results.
"""
function _analysis_coeff_cotangents(cfg::SHTnsKit.SHTConfig, ȳ, outputs::Tuple,
                                    comm, operation::Symbol)
    components = _cotangent_components(ȳ, length(outputs), comm, operation)
    return map(components, outputs) do component, output
        output isa PencilArray ?
            _distributed_coeff_cotangent(cfg, component, output, comm; operation) :
            _replicated_coeff_cotangent(cfg, component, comm; operation)
    end
end

# ----- dist_analysis rrule ---------------------------------------------------

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_analysis),
                              cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                              use_tables=cfg.use_plm_tables,
                              use_rfft::Bool=false,
                              use_packed_storage::Bool=false,
                              comm=communicator(fθφ))
    known_comm = communicator(fθφ)
    _require_host_pencil(:dist_analysis_pullback, fθφ, known_comm)
    y = SHTnsKit.dist_analysis(cfg, fθφ;
                               use_tables, use_rfft, use_packed_storage, comm)
    θ_globals, _, φ_window = _spatial_geometry(fθφ, cfg)

    function dist_analysis_pullback(ȳ)
        Alm̄ = _replicated_coeff_cotangent(
            cfg, ȳ, known_comm; packed=use_packed_storage,
            operation=:dist_analysis_pullback)
        f̄ = _local_adjoint_analysis(cfg, Alm̄, θ_globals, φ_window)
        # Wrap in fθφ's pencil so downstream gradients stay distributed.
        return NoTangent(), NoTangent(), _spatial_tangent(fθφ, f̄)
    end
    return y, dist_analysis_pullback
end

# ----- dist_synthesis rrule --------------------------------------------------
# `Aminus` (explicit negative orders of a complex synthesis) is a keyword, so
# it receives no tangent; the output is linear in `Alm` and `Aminus`
# separately, so the `Alm` cotangent is unaffected by it.

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis),
                              cfg::SHTnsKit.SHTConfig, Alm::PencilArray;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false,
                              Aminus=nothing,
                              ltr::Integer=cfg.lmax,
                              comm=communicator(prototype_θφ))
    known_comm = communicator(prototype_θφ)
    _require_host_pencil(:dist_synthesis_pullback, Alm, known_comm)
    _require_host_pencil(:dist_synthesis_pullback, prototype_θφ, known_comm)
    _require_ad_communicator_match(Alm, prototype_θφ)
    y = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output, use_rfft,
                                Aminus, ltr, comm)

    function dist_synthesis_pencil_pullback(ȳ)
        Ā = _synthesis_coefficient_cotangent(
            cfg, ȳ, prototype_θφ, known_comm; real_output,
            zero_eltype=eltype(y), operation=:dist_synthesis_pullback)
        _truncate_degrees!(Ā, ltr)
        return NoTangent(), NoTangent(), _scatter_spectral_tangent(Alm, Ā)
    end
    return y, dist_synthesis_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis),
                              cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false,
                              Aminus=nothing,
                              comm=communicator(prototype_θφ))
    known_comm = communicator(prototype_θφ)
    _require_host_pencil(:dist_synthesis_pullback, prototype_θφ, known_comm)
    y = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output, use_rfft,
                                Aminus, comm)
    project_Alm = ProjectTo(Alm)

    function dist_synthesis_pullback(ȳ)
        Ā = _synthesis_coefficient_cotangent(
            cfg, ȳ, prototype_θφ, known_comm; real_output,
            zero_eltype=eltype(y), operation=:dist_synthesis_pullback)
        return NoTangent(), NoTangent(), project_Alm(Ā)
    end
    return y, dist_synthesis_pullback
end

# ----- dist_analysis_sphtor rrule -------------------------------------------
# Adjoint: analogous to scalar dist_analysis. (Slm̄, Tlm̄) arrive replicated;
# each rank reconstructs its own (V̄t, V̄p) θ rows × φ window locally using
# the shared `_adjoint_analysis_sphtor` primitive, no inter-rank comms needed.

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_analysis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Vtθφ::PencilArray, Vpθφ::PencilArray;
                              kwargs...)
    comm = communicator(Vtθφ)
    _require_host_pencil(:dist_analysis_sphtor_pullback, Vtθφ, comm)
    _require_host_pencil(:dist_analysis_sphtor_pullback, Vpθφ, comm)
    y = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ; kwargs...)
    θ_globals, _, φ_window = _spatial_geometry(Vtθφ, cfg)

    function dist_analysis_sphtor_pullback(ȳ)
        # Unthunk each component and materialise only after the host-storage
        # guard. This preserves CPU AD without making a hidden vendor→host copy.
        S̄, T̄ = _analysis_coeff_cotangents(
            cfg, ȳ, y, comm, :dist_analysis_sphtor_pullback)
        V̄t, V̄p = SHTnsKit._adjoint_analysis_sphtor(
            cfg, S̄, T̄; θ_globals, φ_window)
        return NoTangent(), NoTangent(),
               _spatial_tangent(Vtθφ, V̄t), _spatial_tangent(Vpθφ, V̄p)
    end
    return y, dist_analysis_sphtor_pullback
end

# ----- dist_synthesis_sphtor rrule ------------------------------------------

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Slm::PencilArray, Tlm::PencilArray;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false,
                              comm=communicator(prototype_θφ))
    known_comm = communicator(prototype_θφ)
    _require_host_pencil(:dist_synthesis_sphtor_pullback, Slm, known_comm)
    _require_host_pencil(:dist_synthesis_sphtor_pullback, Tlm, known_comm)
    _require_host_pencil(
        :dist_synthesis_sphtor_pullback, prototype_θφ, known_comm,
    )
    _require_ad_communicator_match(Slm, prototype_θφ)
    _require_ad_communicator_match(Tlm, prototype_θφ)
    y = SHTnsKit.dist_synthesis_sphtor(
        cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft, comm)

    function dist_synthesis_sphtor_pencil_pullback(ȳ)
        S̄, T̄ = _synthesis_sphtor_coefficient_cotangents(
            cfg, ȳ, prototype_θφ, known_comm; real_output,
            zero_eltypes=(eltype(y[1]), eltype(y[2])),
            operation=:dist_synthesis_sphtor_pullback)
        return NoTangent(), NoTangent(),
               _scatter_spectral_tangent(Slm, S̄), _scatter_spectral_tangent(Tlm, T̄)
    end
    return y, dist_synthesis_sphtor_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Slm::AbstractMatrix, Tlm::AbstractMatrix;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    comm = communicator(prototype_θφ)
    _require_host_pencil(
        :dist_synthesis_sphtor_pullback, prototype_θφ, comm,
    )
    y = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm;
                                        prototype_θφ=prototype_θφ,
                                        real_output=real_output,
                                        use_rfft=use_rfft)
    project_Slm = ProjectTo(Slm)
    project_Tlm = ProjectTo(Tlm)

    function dist_synthesis_sphtor_pullback(ȳ)
        S̄, T̄ = _synthesis_sphtor_coefficient_cotangents(
            cfg, ȳ, prototype_θφ, comm; real_output,
            zero_eltypes=(eltype(y[1]), eltype(y[2])),
            operation=:dist_synthesis_sphtor_pullback)
        return NoTangent(), NoTangent(), project_Slm(S̄), project_Tlm(T̄)
    end
    return y, dist_synthesis_sphtor_pullback
end

# ----- public transforms on PencilArrays -------------------------------------
# `analysis`, `synthesis` and their vector/QST forms dispatch PencilArray
# arguments to the distributed kernels, but SHTnsKitAdvancedADExt's rules take
# untyped arrays: without these more specific methods a PencilArray reached the
# serial adjoint (a MethodError, or rank-local blocks treated as the whole
# grid). Each rule runs the public primal, so validation and results are
# unchanged, accepts all of its keywords, and reuses the adjoints above.
#
# Cotangents of distributed outputs hold each rank's derivative for its own
# block and are summed across ranks; cotangents of replicated outputs
# (`return_pencil=false`) must be identical on every rank. Gradients with
# respect to keyword arguments (`prototype_θφ`, `Aminus`) are not returned.

function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis), cfg::SHTnsKit.SHTConfig,
                              fθφ::PencilArray; use_rfft::Bool=false,
                              return_pencil::Bool=true)
    comm = communicator(fθφ)
    _require_host_pencil(:analysis_pullback, fθφ, comm)
    y = SHTnsKit.analysis(cfg, fθφ; use_rfft, return_pencil)
    θ_globals, _, φ_window = _spatial_geometry(fθφ, cfg)

    function analysis_pencil_pullback(ȳ)
        Alm̄ = only(_analysis_coeff_cotangents(
            cfg, (ȳ,), (y,), comm, :analysis_pullback))
        f̄ = _local_adjoint_analysis(cfg, Alm̄, θ_globals, φ_window)
        return NoTangent(), NoTangent(), _spatial_tangent(fθφ, f̄)
    end
    return y, analysis_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis), cfg::SHTnsKit.SHTConfig,
                              Alm::PencilArray; prototype_θφ::PencilArray,
                              real_output::Bool=true, use_rfft::Bool=false,
                              comm=communicator(prototype_θφ))
    known_comm = communicator(prototype_θφ)
    _require_host_pencil(:synthesis_pullback, Alm, known_comm)
    _require_host_pencil(:synthesis_pullback, prototype_θφ, known_comm)
    _require_ad_communicator_match(Alm, prototype_θφ)
    y = SHTnsKit.synthesis(cfg, Alm; prototype_θφ, real_output, use_rfft, comm)

    function synthesis_pencil_pullback(ȳ)
        Ā = _synthesis_coefficient_cotangent(
            cfg, ȳ, prototype_θφ, known_comm; real_output,
            zero_eltype=eltype(y), operation=:synthesis_pullback)
        return NoTangent(), NoTangent(), _scatter_spectral_tangent(Alm, Ā)
    end
    return y, synthesis_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Vtθφ::PencilArray, Vpθφ::PencilArray;
                              use_tables=cfg.use_plm_tables,
                              use_rfft::Bool=false, return_pencil::Bool=true,
                              comm=communicator(Vtθφ))
    known_comm = communicator(Vtθφ)
    _require_host_pencil(:analysis_sphtor_pullback, Vtθφ, known_comm)
    _require_host_pencil(:analysis_sphtor_pullback, Vpθφ, known_comm)
    y = SHTnsKit.analysis_sphtor(cfg, Vtθφ, Vpθφ;
                                 use_tables, use_rfft, return_pencil, comm)
    θ_globals, _, φ_window = _spatial_geometry(Vtθφ, cfg)

    function analysis_sphtor_pencil_pullback(ȳ)
        S̄, T̄ = _analysis_coeff_cotangents(
            cfg, ȳ, y, known_comm, :analysis_sphtor_pullback)
        V̄t, V̄p = SHTnsKit._adjoint_analysis_sphtor(
            cfg, S̄, T̄; θ_globals, φ_window)
        return NoTangent(), NoTangent(),
               _spatial_tangent(Vtθφ, V̄t), _spatial_tangent(Vpθφ, V̄p)
    end
    return y, analysis_sphtor_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Slm::PencilArray, Tlm::PencilArray;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true, use_rfft::Bool=false,
                              comm=communicator(prototype_θφ))
    known_comm = communicator(prototype_θφ)
    for value in (Slm, Tlm, prototype_θφ)
        _require_host_pencil(:synthesis_sphtor_pullback, value, known_comm)
    end
    _require_ad_communicator_match(Slm, prototype_θφ)
    _require_ad_communicator_match(Tlm, prototype_θφ)
    y = SHTnsKit.synthesis_sphtor(cfg, Slm, Tlm;
                                  prototype_θφ, real_output, use_rfft, comm)

    function synthesis_sphtor_pencil_pullback(ȳ)
        S̄, T̄ = _synthesis_sphtor_coefficient_cotangents(
            cfg, ȳ, prototype_θφ, known_comm; real_output,
            zero_eltypes=(eltype(y[1]), eltype(y[2])),
            operation=:synthesis_sphtor_pullback)
        return NoTangent(), NoTangent(),
               _scatter_spectral_tangent(Slm, S̄), _scatter_spectral_tangent(Tlm, T̄)
    end
    return y, synthesis_sphtor_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.analysis_qst), cfg::SHTnsKit.SHTConfig,
                              Vrθφ::PencilArray, Vtθφ::PencilArray,
                              Vpθφ::PencilArray; use_rfft::Bool=false,
                              return_pencil::Bool=true)
    comm = communicator(Vrθφ)
    for value in (Vrθφ, Vtθφ, Vpθφ)
        _require_host_pencil(:analysis_qst_pullback, value, comm)
    end
    y = SHTnsKit.analysis_qst(cfg, Vrθφ, Vtθφ, Vpθφ; use_rfft, return_pencil)
    θ_globals, _, φ_window = _spatial_geometry(Vrθφ, cfg)

    function analysis_qst_pencil_pullback(ȳ)
        Q̄, S̄, T̄ = _analysis_coeff_cotangents(
            cfg, ȳ, y, comm, :analysis_qst_pullback)
        V̄r = _local_adjoint_analysis(cfg, Q̄, θ_globals, φ_window)
        V̄t, V̄p = SHTnsKit._adjoint_analysis_sphtor(
            cfg, S̄, T̄; θ_globals, φ_window)
        return NoTangent(), NoTangent(), _spatial_tangent(Vrθφ, V̄r),
               _spatial_tangent(Vtθφ, V̄t), _spatial_tangent(Vpθφ, V̄p)
    end
    return y, analysis_qst_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.synthesis_qst), cfg::SHTnsKit.SHTConfig,
                              Qlm::PencilArray, Slm::PencilArray, Tlm::PencilArray;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true, use_rfft::Bool=false,
                              comm=communicator(prototype_θφ))
    known_comm = communicator(prototype_θφ)
    for value in (Qlm, Slm, Tlm, prototype_θφ)
        _require_host_pencil(:synthesis_qst_pullback, value, known_comm)
    end
    for value in (Qlm, Slm, Tlm)
        _require_ad_communicator_match(value, prototype_θφ)
    end
    y = SHTnsKit.synthesis_qst(cfg, Qlm, Slm, Tlm;
                               prototype_θφ, real_output, use_rfft, comm)

    function synthesis_qst_pencil_pullback(ȳ)
        V̄r, V̄t, V̄p = _cotangent_components(
            ȳ, 3, known_comm, :synthesis_qst_pullback)
        Q̄ = _synthesis_coefficient_cotangent(
            cfg, V̄r, prototype_θφ, known_comm; real_output,
            zero_eltype=eltype(y[1]), operation=:synthesis_qst_pullback)
        S̄, T̄ = _synthesis_sphtor_coefficient_cotangents(
            cfg, (V̄t, V̄p), prototype_θφ, known_comm; real_output,
            zero_eltypes=(eltype(y[2]), eltype(y[3])),
            operation=:synthesis_qst_pullback)
        return NoTangent(), NoTangent(), _scatter_spectral_tangent(Qlm, Q̄),
               _scatter_spectral_tangent(Slm, S̄), _scatter_spectral_tangent(Tlm, T̄)
    end
    return y, synthesis_qst_pencil_pullback
end

# ----- energy diagnostics of spectral PencilArrays ----------------------------
# These sum each rank's block and Allreduce, so every rank gets the replicated
# global value. Their pullbacks need no communication: each rank returns the
# derivative for its own block, which `_distributed_coeff_cotangent` then sums.
# Zygote cannot trace the Allreduce itself.

"""
Rank-local gradient block of a spectral-PencilArray energy. Entry `(l, m)` is
`Ē · wₘ · metric(l, m) · (l(l+1))^power · A[l, m]` on the stored triangle
(`l ≥ max(lmin, m)`, `m % mres == 0`) and zero elsewhere — the same weights as
the primal sum.
"""
function _energy_gradient_block(cfg::SHTnsKit.SHTConfig, A::PencilArray, Ē;
                                real_field::Bool, power::Int, lmin::Int)
    scale_matrix = SHTnsKit._diagnostic_scale_matrix(cfg)
    values = parent(A)
    block = zero(values)
    gl_l = collect(Int, globalindices(A, 1))
    gl_m = collect(Int, globalindices(A, 2))
    @inbounds for (jj, gm) in enumerate(gl_m)
        m = gm - 1
        m % cfg.mres == 0 || continue
        w = (real_field && m > 0) ? 2.0 : 1.0
        for (ii, gl) in enumerate(gl_l)
            l = gl - 1
            l >= max(lmin, m) || continue
            weight = w * SHTnsKit._convention_metric(scale_matrix, l, m) *
                     (l * (l + 1))^power
            block[ii, jj] = Ē * weight * values[ii, jj]
        end
    end
    return PencilArray(PencilArrays.pencil(A), block)
end

"""
The cotangent of a replicated scalar output (an energy), `0.0` for a zero
tangent. Like `_replicated_coeff_cotangent`, it must be identical on every
rank: blocks scaled by rank-varying cotangents would not form one gradient.
"""
function _replicated_scalar_cotangent(Ē, comm, operation::Symbol)
    Ē = ChainRulesCore.unthunk(Ē)
    value = Ē isa ChainRulesCore.AbstractZero ? 0.0 : Ē
    local_ok = value isa Real
    MPI.Allreduce(local_ok, &, comm) || throw(ArgumentError(
        "$operation cotangent must be a real number",
    ))
    reference = Ref(Float64(value))
    MPI.Bcast!(reference, comm; root=0)
    same = Float64(value) == reference[] || isequal(Float64(value), reference[])
    MPI.Allreduce(same, &, comm) || throw(ArgumentError(
        "the cotangent of a replicated distributed energy must be identical on every rank; rank-varying partial cotangents are unsupported",
    ))
    return value
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.energy_scalar), cfg::SHTnsKit.SHTConfig,
                              Alm::PencilArray; real_field::Bool=true)
    _require_host_pencil(:energy_scalar_pullback, Alm)
    E = SHTnsKit.energy_scalar(cfg, Alm; real_field)
    A0 = copy(Alm)
    function energy_scalar_pencil_pullback(Ē)
        Ē = _replicated_scalar_cotangent(Ē, communicator(A0), :energy_scalar_pullback)
        iszero(Ē) && return NoTangent(), NoTangent(), ZeroTangent()
        return NoTangent(), NoTangent(),
               _energy_gradient_block(cfg, A0, Ē; real_field, power=0, lmin=0)
    end
    return E, energy_scalar_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.energy_vector), cfg::SHTnsKit.SHTConfig,
                              Slm::PencilArray, Tlm::PencilArray;
                              real_field::Bool=true)
    _require_host_pencil(:energy_vector_pullback, Slm)
    _require_host_pencil(:energy_vector_pullback, Tlm, communicator(Slm))
    E = SHTnsKit.energy_vector(cfg, Slm, Tlm; real_field)
    S0 = copy(Slm); T0 = copy(Tlm)
    function energy_vector_pencil_pullback(Ē)
        Ē = _replicated_scalar_cotangent(Ē, communicator(S0), :energy_vector_pullback)
        iszero(Ē) && return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
        return NoTangent(), NoTangent(),
               _energy_gradient_block(cfg, S0, Ē; real_field, power=1, lmin=1),
               _energy_gradient_block(cfg, T0, Ē; real_field, power=1, lmin=1)
    end
    return E, energy_vector_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.enstrophy), cfg::SHTnsKit.SHTConfig,
                              Tlm::PencilArray; real_field::Bool=true)
    _require_host_pencil(:enstrophy_pullback, Tlm)
    Z = SHTnsKit.enstrophy(cfg, Tlm; real_field)
    T0 = copy(Tlm)
    function enstrophy_pencil_pullback(Z̄)
        Z̄ = _replicated_scalar_cotangent(Z̄, communicator(T0), :enstrophy_pullback)
        iszero(Z̄) && return NoTangent(), NoTangent(), ZeroTangent()
        return NoTangent(), NoTangent(),
               _energy_gradient_block(cfg, T0, Z̄; real_field, power=2, lmin=1)
    end
    return Z, enstrophy_pencil_pullback
end

# ----- PencilArray methods without a distributed adjoint ----------------------
# These have PencilArray methods, and SHTnsKitAdvancedADExt's untyped serial
# rules used to capture them and run a serial adjoint on rank-local blocks.
# Fail with a clear message instead.

for transform in (:analysis_batch, :synthesis_batch, :analysis_packed,
                  :synthesis_packed, :analysis_packed_cplx, :synthesis_packed_cplx)
    @eval function ChainRulesCore.rrule(::typeof(SHTnsKit.$transform),
                                        ::SHTnsKit.SHTConfig, ::PencilArray;
                                        kwargs...)
        throw(ArgumentError(string(
            "reverse-mode AD through distributed `", $(QuoteNode(transform)),
            "` is not supported; differentiate the per-field `analysis`/`synthesis` ",
            "(or their sphtor/qst forms), which have distributed rules")))
    end
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.SH_mul_mx), ::SHTnsKit.SHTConfig,
                              mx, ::PencilArray, Rlm)
    throw(ArgumentError(
        "reverse-mode AD through distributed `SH_mul_mx` is not supported"))
end

end # module
