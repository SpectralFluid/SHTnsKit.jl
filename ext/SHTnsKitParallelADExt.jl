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
================================================================================
=#

using ChainRulesCore
using MPI
using PencilArrays
using PencilArrays: PencilArray
using SHTnsKit
using FFTW

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

"""Materialize and verify the cotangent of one logically replicated result."""
function _replicated_coeff_cotangent(cfg::SHTnsKit.SHTConfig, ȳ, comm;
                                     packed::Bool=false)
    ȳ = ChainRulesCore.unthunk(ȳ)
    Alm̄ = if ȳ isa ChainRulesCore.AbstractZero
        zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    elseif packed
        SHTnsKit.unpack_lm(cfg, ȳ)
    else
        Matrix{ComplexF64}(ȳ)
    end
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    shape_ok = size(Alm̄) == expected
    MPI.Allreduce(shape_ok, &, comm) ||
        throw(DimensionMismatch("distributed coefficient cotangent must have size $expected"))

    if MPI.Comm_size(comm) > 1
        root_value = similar(Alm̄)
        if MPI.Comm_rank(comm) == 0
            copyto!(root_value, Alm̄)
        else
            fill!(root_value, zero(eltype(root_value)))
        end
        MPI.Bcast!(root_value, comm; root=0)
        same = Alm̄ == root_value
        MPI.Allreduce(same, &, comm) || throw(ArgumentError(
            "the cotangent of replicated distributed-analysis output must be " *
            "identical on every rank; rank-varying partial cotangents are unsupported",
        ))
    end
    return Alm̄
end

function _require_ad_communicator_match(spectral::PencilArray,
                                        spatial::PencilArray)
    # Reduce the verdict on the spatial prototype's communicator.  Throwing
    # only on a rank whose spectral operand uses COMM_SELF would leave peers
    # entering the primal transform's world-communicator collectives.
    comm = communicator(spatial)
    local_ok = MPI.Comm_compare(communicator(spectral), comm) != MPI.UNEQUAL
    MPI.Allreduce(local_ok, &, comm) || throw(ArgumentError(
        "spectral and spatial PencilArrays must use communicators with the same process group",
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
    return PencilArray(primal.pencil, local_parent)
end

# ----- dist_analysis rrule ---------------------------------------------------

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_analysis),
                              cfg::SHTnsKit.SHTConfig, fθφ::PencilArray;
                              use_tables=cfg.use_plm_tables,
                              use_rfft::Bool=false,
                              use_packed_storage::Bool=false)
    y = SHTnsKit.dist_analysis(cfg, fθφ;
                                use_tables, use_rfft, use_packed_storage)
    comm = communicator(fθφ)
    θ_globals = collect(globalindices(fθφ, 1))
    φ_globals = collect(globalindices(fθφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)
    project_f_parent = ProjectTo(parent(fθφ))

    function dist_analysis_pullback(ȳ)
        Alm̄ = _replicated_coeff_cotangent(
            cfg, ȳ, comm; packed=use_packed_storage)
        f̄_parent = _local_adjoint_analysis(cfg, Alm̄, θ_globals, φ_window)
        # Wrap in a PencilArray sharing fθφ's pencil so downstream grads stay distributed.
        f̄ = PencilArray(fθφ.pencil, project_f_parent(f̄_parent))
        return NoTangent(), NoTangent(), f̄
    end
    return y, dist_analysis_pullback
end

# ----- dist_synthesis rrule --------------------------------------------------

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis),
                              cfg::SHTnsKit.SHTConfig, Alm::PencilArray;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    _require_ad_communicator_match(Alm, prototype_θφ)
    y = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output, use_rfft)
    comm = communicator(prototype_θφ)
    θ_globals = collect(globalindices(prototype_θφ, 1))
    φ_globals = collect(globalindices(prototype_θφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)

    function dist_synthesis_pencil_pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        nθ_local = length(θ_globals)
        ȳ_loc = ȳ isa ChainRulesCore.AbstractZero ?
                 zeros(real_output ? Float64 : ComplexF64, nθ_local, nlon_local) :
                 (ȳ isa PencilArray ? parent(ȳ) : ȳ)
        f̄_full = zeros(float(eltype(ȳ_loc)), nθ_local, cfg.nlon)
        if φ_is_local
            f̄_full .= ȳ_loc
        else
            @views f̄_full[:, φ_window] .= ȳ_loc
        end
        Āpartial = SHTnsKit._adjoint_synthesis(
            cfg, f̄_full; θ_globals=θ_globals, real_output=real_output)
        Ādense = MPI.Allreduce(Āpartial, +, comm)
        Ā = _scatter_spectral_tangent(Alm, Ādense)
        return NoTangent(), NoTangent(), Ā
    end
    return y, dist_synthesis_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis),
                              cfg::SHTnsKit.SHTConfig, Alm::AbstractMatrix;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    y = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ, real_output, use_rfft)
    project_Alm = ProjectTo(Alm)
    comm = communicator(prototype_θφ)
    θ_globals = collect(globalindices(prototype_θφ, 1))
    φ_globals = collect(globalindices(prototype_θφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)

    function dist_synthesis_pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        # Adjoint of synthesis is `_adjoint_synthesis` (NO quadrature weights /
        # cphi — those belong to analysis), applied on this rank's θ slab over a
        # FULL-nlon-width cotangent, then Allreduce-summed across ranks to build
        # the replicated Ālm. Using `dist_analysis` here would wrongly inject the
        # Gauss weights `w[θ]·cphi`. The forward ifft is over the full φ width and
        # only then sliced, so we zero-pad the local φ window back to full nlon;
        # FFT linearity makes Σ_ranks fft(padded window) = fft(full field).
        nθ_local = length(θ_globals)
        ȳ_loc = ȳ isa ChainRulesCore.AbstractZero ?
                 zeros(real_output ? Float64 : ComplexF64, nθ_local, nlon_local) :
                 (ȳ isa PencilArray ? parent(ȳ) : ȳ)
        ET = float(eltype(ȳ_loc))  # real for real_output, complex otherwise
        f̄_full = zeros(ET, nθ_local, cfg.nlon)
        if φ_is_local
            f̄_full .= ȳ_loc
        else
            @views f̄_full[:, φ_window] .= ȳ_loc
        end
        Ālm_partial = SHTnsKit._adjoint_synthesis(cfg, f̄_full;
                                                  θ_globals=θ_globals,
                                                  real_output=real_output)
        Ālm = project_Alm(MPI.Allreduce(Ālm_partial, +, comm))
        return NoTangent(), NoTangent(), Ālm
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
    y = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ; kwargs...)
    comm = communicator(Vtθφ)
    θ_globals = collect(globalindices(Vtθφ, 1))
    φ_globals = collect(globalindices(Vtθφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)
    project_Vt_parent = ProjectTo(parent(Vtθφ))
    project_Vp_parent = ProjectTo(parent(Vpθφ))

    function dist_analysis_sphtor_pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        # unthunk EACH component — a Tuple/Tangent of Thunks would otherwise reach
        # Matrix{ComplexF64}(::Thunk) below and error (matches the synthesis twin).
        Slm̄, Tlm̄ = ChainRulesCore.unthunk(ȳ[1]), ChainRulesCore.unthunk(ȳ[2])
        # Both outputs are one logical replicated value. Materialize zero slots
        # and reject ambiguous rank-varying partial cotangents collectively.
        S̄in = _replicated_coeff_cotangent(cfg, Slm̄, comm)
        T̄in = _replicated_coeff_cotangent(cfg, Tlm̄, comm)
        V̄t_parent, V̄p_parent = SHTnsKit._adjoint_analysis_sphtor(
            cfg, S̄in, T̄in;
            θ_globals=θ_globals, φ_window=φ_window)
        # Keep the shared adjoint complex-linear, then independently project
        # each local parent buffer into its primal's tangent space.  This drops
        # the imaginary component only for real-valued PencilArray primals.
        V̄t = PencilArray(Vtθφ.pencil, project_Vt_parent(V̄t_parent))
        V̄p = PencilArray(Vpθφ.pencil, project_Vp_parent(V̄p_parent))
        return NoTangent(), NoTangent(), V̄t, V̄p
    end
    return y, dist_analysis_sphtor_pullback
end

# ----- dist_synthesis_sphtor rrule ------------------------------------------
# Adjoint: analogous to scalar. dist_analysis_sphtor on the spatial cotangents
# performs the per-rank analysis + Allreduce to produce replicated (Ālm_S, Ālm_T).

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Slm::PencilArray, Tlm::PencilArray;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    _require_ad_communicator_match(Slm, prototype_θφ)
    _require_ad_communicator_match(Tlm, prototype_θφ)
    y = SHTnsKit.dist_synthesis_sphtor(
        cfg, Slm, Tlm; prototype_θφ, real_output, use_rfft)
    comm = communicator(prototype_θφ)
    θ_globals = collect(globalindices(prototype_θφ, 1))
    φ_globals = collect(globalindices(prototype_θφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)

    function dist_synthesis_sphtor_pencil_pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        V̄t = ChainRulesCore.unthunk(ȳ[1])
        V̄p = ChainRulesCore.unthunk(ȳ[2])
        nθ_local = length(θ_globals)
        _local_or_zero(A) = A isa ChainRulesCore.AbstractZero ?
            zeros(Float64, nθ_local, nlon_local) :
            (A isa PencilArray ? parent(A) : A)
        V̄t_loc = _local_or_zero(V̄t)
        V̄p_loc = _local_or_zero(V̄p)
        V̄t_full = zeros(float(eltype(V̄t_loc)), nθ_local, cfg.nlon)
        V̄p_full = zeros(float(eltype(V̄p_loc)), nθ_local, cfg.nlon)
        if φ_is_local
            V̄t_full .= V̄t_loc
            V̄p_full .= V̄p_loc
        else
            @views V̄t_full[:, φ_window] .= V̄t_loc
            @views V̄p_full[:, φ_window] .= V̄p_loc
        end
        S̄partial, T̄partial = SHTnsKit._adjoint_synthesis_sphtor(
            cfg, V̄t_full, V̄p_full;
            θ_globals=θ_globals, real_output=real_output)
        n = length(S̄partial)
        combined = MPI.Allreduce!(vcat(vec(S̄partial), vec(T̄partial)), +, comm)
        copyto!(S̄partial, 1, combined, 1, n)
        copyto!(T̄partial, 1, combined, n + 1, length(combined) - n)
        S̄ = _scatter_spectral_tangent(Slm, S̄partial)
        T̄ = _scatter_spectral_tangent(Tlm, T̄partial)
        return NoTangent(), NoTangent(), S̄, T̄
    end
    return y, dist_synthesis_sphtor_pencil_pullback
end

function ChainRulesCore.rrule(::typeof(SHTnsKit.dist_synthesis_sphtor),
                              cfg::SHTnsKit.SHTConfig,
                              Slm::AbstractMatrix, Tlm::AbstractMatrix;
                              prototype_θφ::PencilArray,
                              real_output::Bool=true,
                              use_rfft::Bool=false)
    y = SHTnsKit.dist_synthesis_sphtor(cfg, Slm, Tlm;
                                        prototype_θφ=prototype_θφ,
                                        real_output=real_output,
                                        use_rfft=use_rfft)
    project_Slm = ProjectTo(Slm)
    project_Tlm = ProjectTo(Tlm)
    comm = communicator(prototype_θφ)
    θ_globals = collect(globalindices(prototype_θφ, 1))
    φ_globals = collect(globalindices(prototype_θφ, 2))
    nlon_local = length(φ_globals)
    φ_is_local, φ_window = _phi_window(φ_globals, nlon_local, cfg.nlon)

    function dist_synthesis_sphtor_pullback(ȳ)
        ȳ = ChainRulesCore.unthunk(ȳ)
        # unthunk EACH component (a Tangent tuple of thunks would otherwise slip through)
        V̄t = ChainRulesCore.unthunk(ȳ[1])
        V̄p = ChainRulesCore.unthunk(ȳ[2])
        # Adjoint of vector synthesis is `_adjoint_synthesis_sphtor` (no quadrature
        # weights), applied per-rank on the local θ slab over a full-nlon-width
        # (zero-padded) cotangent, then Allreduce-summed. Previously this called
        # `dist_analysis_sphtor`, which injects Gauss weights `w[θ]·scaleφ/(l(l+1))`
        # that the synthesis adjoint must NOT carry.
        # Zero spatial cotangents likewise: materialise before touching eltype.
        nθl = length(θ_globals)
        _mzs(A) = A isa ChainRulesCore.AbstractZero ? zeros(Float64, nθl, nlon_local) :
                  (A isa PencilArray ? parent(A) : A)
        V̄t_loc = _mzs(V̄t)
        V̄p_loc = _mzs(V̄p)
        nθ_local = length(θ_globals)
        ETt = float(eltype(V̄t_loc)); ETp = float(eltype(V̄p_loc))
        V̄t_full = zeros(ETt, nθ_local, cfg.nlon)
        V̄p_full = zeros(ETp, nθ_local, cfg.nlon)
        if φ_is_local
            V̄t_full .= V̄t_loc
            V̄p_full .= V̄p_loc
        else
            @views V̄t_full[:, φ_window] .= V̄t_loc
            @views V̄p_full[:, φ_window] .= V̄p_loc
        end
        S̄p, T̄p = SHTnsKit._adjoint_synthesis_sphtor(cfg, V̄t_full, V̄p_full;
                                                    θ_globals=θ_globals,
                                                    real_output=real_output)
        # One batched Allreduce over stacked (S̄,T̄) instead of two round-trips.
        n = length(S̄p)
        combined = MPI.Allreduce!(vcat(vec(S̄p), vec(T̄p)), +, comm)
        # Scatter back into the matrices `_adjoint_synthesis_sphtor` already
        # allocated. `combined[1:n]` would allocate two more full-size arrays on
        # top of the vcat (~12 MB per backward pass at lmax=511), and returning
        # `reshape(view(combined, …))` avoids that but hands back a ReshapedArray
        # rather than a Matrix — a downstream distributed pullback that feeds the
        # tangent straight to `MPI.Allreduce!` then fails buffer conversion.
        # copyto! reuses S̄p/T̄p, so this is both allocation-free and an Array.
        copyto!(S̄p, 1, combined, 1, n)
        copyto!(T̄p, 1, combined, n + 1, length(combined) - n)
        return NoTangent(), NoTangent(), project_Slm(S̄p), project_Tlm(T̄p)
    end
    return y, dist_synthesis_sphtor_pullback
end

end # module
