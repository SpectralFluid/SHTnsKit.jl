##########
# Minimal plan structs to keep API stable
##########

"""Fingerprint every configuration field that can affect a transform."""
function _cfg_fingerprint(cfg::SHTnsKit.SHTConfig)
    fingerprint = hash((
        cfg.lmax, cfg.mmax, cfg.mres, cfg.nlat, cfg.nlon, cfg.grid_type,
        cfg.nlm, cfg.nspat, cfg.phi_scale, cfg.on_the_fly,
        cfg.howmany, cfg.spec_dist, cfg.south_pole_first,
        cfg.allow_padding, cfg.nlat_padded, cfg.spat_dist,
        cfg.norm, cfg.cs_phase, cfg.real_norm, cfg.robert_form,
        cfg.cphi, cfg.use_plm_tables,
    ))
    for values in (cfg.li, cfg.mi, cfg.θ, cfg.φ, cfg.x, cfg.w, cfg.st, cfg.Nlm)
        # Julia's array hash samples large arrays. A change to any grid node
        # or weight must invalidate plans that cached a copy of it.
        fingerprint = hash(size(values), fingerprint)
        for value in values
            fingerprint = hash(value, fingerprint)
        end
    end
    # The PLM arrays are derived caches.  Their enabled state and shapes affect
    # dispatch, while hashing all values would make every cached-plan call
    # unnecessarily O(lmax²*nlat).
    for tables in (
            cfg.plm_tables, cfg.dplm_tables, cfg.NP_tables, cfg.NdP_tables)
        fingerprint = hash(length(tables), fingerprint)
        for table in tables
            fingerprint = hash(size(table), fingerprint)
        end
    end
    return fingerprint
end

"""Collectively require a configuration to be identical on every rank."""
function _validate_cfg_replicated(cfg::SHTnsKit.SHTConfig, comm)
    MPI.Comm_size(comm) > 1 || return nothing
    sig = _cfg_fingerprint(cfg)
    root_sig = MPI.bcast(sig, 0, comm)
    # Decide the throw COLLECTIVELY: a lone throw on the mismatched rank(s) would
    # leave the matching ranks (incl. rank 0, which always matches) proceeding into
    # the plan's later collectives → hang. Allreduce the mismatch so all ranks
    # raise together and no rank is left waiting.
    n_mismatch = MPI.Allreduce(sig != root_sig ? 1 : 0, +, comm)
    if n_mismatch != 0
        throw(ArgumentError("SHTConfig diverges across ranks ($(n_mismatch) mismatched). " *
                            "All ranks must construct cfg with identical parameters."))
    end
    return nothing
end

@inline function _communicators_congruent(a::MPI.Comm, b::MPI.Comm)
    comparison = MPI.Comm_compare(a, b)
    return comparison == MPI.IDENT || comparison == MPI.CONGRUENT
end

function _validate_prototype_communicator(comm::MPI.Comm, prototype::PencilArray,
                                          operation::AbstractString)
    local_ok = try
        _communicators_congruent(comm, communicator(prototype))
    catch
        false
    end
    MPI.Allreduce(local_ok, &, comm) || throw(ArgumentError(
        "$operation collective validation failed: communicator mismatch " *
        "(the plan and spatial prototype must use congruent MPI communicators)",
    ))
    return nothing
end

function _validate_spatial_shape(cfg::SHTnsKit.SHTConfig, prototype::PencilArray,
                                 comm::MPI.Comm, operation::AbstractString)
    expected = (cfg.nlat, cfg.nlon)
    actual = Tuple(PencilArrays.size_global(prototype))
    mismatches = MPI.Allreduce(actual == expected ? 0 : 1, +, comm)
    mismatches == 0 || throw(DimensionMismatch(
        "$operation requires a spatial PencilArray with global shape $expected " *
        "on every rank ($mismatches mismatched)",
    ))
    return nothing
end

"""Reject parent-storage permutations unsupported by the cfg-form kernels."""
function _require_unpermuted_pencil(A::PencilArray, operation::AbstractString;
                                    comm=communicator(A))
    local_ok = try
        PencilArrays.permutation(A) isa PencilArrays.NoPermutation
    catch
        false
    end
    MPI.Allreduce(local_ok, &, comm) || throw(ArgumentError(
        "$operation does not support permuted PencilArray parent storage; " *
        "construct the pencil with NoPermutation()",
    ))
    return nothing
end

function _validate_cfg_spatial_prototype(cfg::SHTnsKit.SHTConfig,
                                         prototype::PencilArray,
                                         operation::AbstractString;
                                         comm=communicator(prototype))
    _validate_cfg_replicated(cfg, comm)
    _validate_prototype_communicator(comm, prototype, operation)
    _validate_spatial_shape(cfg, prototype, comm, operation)
    _require_unpermuted_pencil(prototype, operation; comm)
    return nothing
end

function _validate_spatial_pencil_against_prototype(
        cfg::SHTnsKit.SHTConfig, expected::PencilArray, actual::PencilArray,
        operation::AbstractString; comm=communicator(expected))
    _validate_prototype_communicator(comm, expected, operation)
    _validate_prototype_communicator(comm, actual, operation)
    _validate_spatial_shape(cfg, actual, comm, operation)
    expected_ranges = PencilArrays.range_local(pencil(expected))
    actual_ranges = PencilArrays.range_local(pencil(actual))
    local_ok = expected_ranges == actual_ranges &&
               size(parent(expected)) == size(parent(actual))
    mismatches = MPI.Allreduce(local_ok ? 0 : 1, +, comm)
    mismatches == 0 || throw(DimensionMismatch(
        "$operation requires the same rank-local spatial ranges as the plan " *
        "prototype on every rank ($mismatches mismatched)",
    ))
    _require_unpermuted_pencil(actual, operation; comm)
    return nothing
end

function _validate_spectral_pencil(
        cfg::SHTnsKit.SHTConfig, spectral::PencilArray,
        spatial_prototype::PencilArray, operation::AbstractString;
        validate_context::Bool=true,
        comm=communicator(spatial_prototype))
    validate_context && _validate_cfg_spatial_prototype(
        cfg, spatial_prototype, operation; comm,
    )
    _validate_prototype_communicator(comm, spectral, operation)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    actual = Tuple(PencilArrays.size_global(spectral))
    mismatches = MPI.Allreduce(actual == expected ? 0 : 1, +, comm)
    mismatches == 0 || throw(DimensionMismatch(
        "$operation requires spectral PencilArrays with global shape $expected " *
        "on every rank ($mismatches mismatched)",
    ))
    _require_unpermuted_pencil(spectral, operation; comm)
    return nothing
end

function _validate_matching_pencil_layout(reference::PencilArray,
                                          actual::PencilArray,
                                          operation::AbstractString;
                                          comm=communicator(reference))
    return _validate_pencil_layout_description!(
        pencil(reference), size_global(reference), size(parent(reference)),
        actual, Symbol(operation); comm,
    )
end

function _validate_replicated_call_signature(
        comm::MPI.Comm, operation::AbstractString, signature)
    MPI.Comm_size(comm) > 1 || return nothing
    local_sig = hash(signature)
    root_sig = MPI.bcast(local_sig, 0, comm)
    nbad = MPI.Allreduce(local_sig == root_sig ? 0 : 1, +, comm)
    nbad == 0 || throw(ArgumentError(
        "$operation requires identical replicated inputs and options on every rank",
    ))
    return nothing
end

function _validate_cached_plan_cfg(
        cfg::SHTnsKit.SHTConfig, cfg_fingerprint::UInt, comm::MPI.Comm,
        operation::AbstractString)
    local_ok = _cfg_fingerprint(cfg) == cfg_fingerprint
    MPI.Allreduce(local_ok, &, comm) || throw(ArgumentError(
        "$operation cannot reuse a plan after its SHTConfig has changed; rebuild the plan",
    ))
    return nothing
end

function _validate_dense_spectral_shapes(
        cfg::SHTnsKit.SHTConfig, comm::MPI.Comm,
        operation::AbstractString, arrays::Tuple)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    local_ok = all(A -> A === nothing || size(A) == expected, arrays)
    MPI.Allreduce(local_ok, &, comm) || throw(DimensionMismatch(
        "$operation requires every dense spectral matrix to have shape $expected",
    ))
    return nothing
end

"""Validate every semantically active coefficient of replicated dense spectra."""
function _validate_replicated_dense_spectra(
        cfg::SHTnsKit.SHTConfig, comm::MPI.Comm,
        operation::AbstractString, arrays::Tuple;
        options::Tuple=(),
        domains::Tuple=ntuple(
            _ -> (minimum_l=0, include_m0=true), length(arrays),
        ))
    length(domains) == length(arrays) || throw(ArgumentError(
        "one active coefficient domain is required for each dense spectrum",
    ))
    array_signatures = map(arrays, domains) do A, domain
        A === nothing && return nothing
        content_hash = hash((eltype(A), axes(A), domain))
        @inbounds for m in 0:cfg.mres:cfg.mmax
            !domain.include_m0 && m == 0 && continue
            for l in max(m, domain.minimum_l):cfg.lmax
                content_hash = hash(A[l + 1, m + 1], content_hash)
            end
        end
        return (eltype(A), axes(A), content_hash)
    end
    _validate_replicated_call_signature(
        comm, operation, (options, array_signatures),
    )
    return nothing
end

function _validate_distributed_plan_preflight(cfg::SHTnsKit.SHTConfig, plan,
                                              prototype::PencilArray,
                                              operation::AbstractString)
    comm = plan.comm
    is_2d = hasproperty(plan, :p_l)
    plan_signature = if is_2d
        context = getproperty(plan, :scratch_context)
        scratch = getproperty(plan, :scratch)
        (
            true, getproperty(plan, :instance_id),
            plan.lmax, plan.mmax, plan.mres, plan.nprocs,
            getproperty(plan, :p_l), getproperty(plan, :p_m),
            getproperty(plan, :with_scratch), scratch !== nothing,
            context !== nothing, getproperty(plan, :closed),
        )
    else
        (
            false, plan.lmax, plan.mmax, plan.mres, plan.nprocs,
            plan.recv_counts, plan.recv_displs,
        )
    end
    # This must be the first collective. In particular, plan construction owns
    # derived communicator contexts in the 2-D case, so its replicated instance
    # identity must agree before any rank is allowed to touch those contexts.
    _validate_replicated_call_signature(comm, operation, plan_signature)

    if is_2d
        nprocs = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        p_l = getproperty(plan, :p_l)
        p_m = getproperty(plan, :p_m)
        flags = UInt32(0)
        getproperty(plan, :closed) && (flags |= 0x0002)
        plan.nprocs == nprocs && plan.rank == rank || (flags |= 0x0002)
        p_l > 0 && p_m > 0 && p_l * p_m == nprocs || (flags |= 0x0002)
        getproperty(plan, :l_rank) == rank % max(p_l, 1) &&
            getproperty(plan, :m_rank) == rank ÷ max(p_l, 1) ||
            (flags |= 0x0002)
        getproperty(plan, :local_nlm) ==
            length(getproperty(plan, :local_lm_indices)) || (flags |= 0x0002)
        length(getproperty(plan, :l_recv_counts)) == p_l &&
            length(getproperty(plan, :l_recv_displs)) == p_l ||
            (flags |= 0x0002)
        getproperty(plan, :with_scratch) ==
            (getproperty(plan, :scratch) !== nothing &&
             getproperty(plan, :scratch_context) !== nothing) ||
            (flags |= 0x0002)
        !_comm_is_null(getproperty(plan, :l_comm)) &&
            !_comm_is_null(getproperty(plan, :m_comm)) || (flags |= 0x0002)
        _collective_validation_error(comm, flags, Symbol(operation))

        # Safe only after the parent-communicator verdict above has established
        # that every rank owns open subcommunicators and the same process grid.
        subgroup_ok = MPI.Comm_size(getproperty(plan, :l_comm)) == p_l &&
            MPI.Comm_rank(getproperty(plan, :l_comm)) ==
                getproperty(plan, :l_rank) &&
            MPI.Comm_size(getproperty(plan, :m_comm)) == p_m &&
            MPI.Comm_rank(getproperty(plan, :m_comm)) ==
                getproperty(plan, :m_rank)
        _collective_validation_error(
            comm, subgroup_ok ? UInt32(0) : UInt32(0x0002),
            Symbol(operation),
        )
    end
    _validate_cfg_replicated(cfg, comm)
    expected = (plan.lmax, plan.mmax, plan.mres)
    actual = (cfg.lmax, cfg.mmax, cfg.mres)
    mismatches = MPI.Allreduce(actual == expected ? 0 : 1, +, comm)
    mismatches == 0 || throw(ArgumentError(
        "$operation configuration does not match the distributed plan on " *
        "every rank ($mismatches mismatched)",
    ))
    _validate_prototype_communicator(comm, prototype, operation)
    _validate_spatial_shape(cfg, prototype, comm, operation)
    _require_unpermuted_pencil(prototype, operation; comm)
    if hasproperty(plan, :scratch_context)
        context = getproperty(plan, :scratch_context)
        if context !== nothing
            cfg_ok = _cfg_fingerprint(cfg) == context.cfg_fingerprint
            MPI.Allreduce(cfg_ok, &, comm) || throw(ArgumentError(
                "$operation configuration differs from the one used to build the scratch plan",
            ))
            actual_ranges = PencilArrays.range_local(pencil(prototype))
            layout_ok = actual_ranges == context.spatial_ranges &&
                        Tuple(size(parent(prototype))) == context.spatial_parent_size
            MPI.Allreduce(layout_ok, &, comm) || throw(DimensionMismatch(
                "$operation requires the exact rank-local spatial layout used to build the scratch plan",
            ))
        end
    end
    return nothing
end

struct DistAnalysisPlan{CT<:Complex}
    cfg::SHTnsKit.SHTConfig
    cfg_fingerprint::UInt
    prototype_θφ::PencilArray
    use_rfft::Bool
    # φ-distributed prototypes need the longitude gather; dist_analysis! falls
    # back to the allocating standard path for them (that layout anti-scales
    # and already warns).
    fallback_standard::Bool
    # Per-call scratch, sized once from cfg + the prototype's local θ slab so
    # dist_analysis! runs allocation-free after warmup.
    θ_globals::Vector{Int}
    weights_cache::Vector{Float64}
    x_cache::Vector{Float64}
    P::Vector{Float64}
    Fθm::Matrix{CT}
    Alm_work::Matrix{CT}
    θ_is_distributed::Bool
    # θ-column subcomm for the partial-sum reduction (Comm_split once here
    # instead of every call). Equals the full communicator when θ is not
    # distributed or for the fallback path; freed by MPI_Finalize with the
    # plan's lifetime (plans are long-lived by design).
    reduce_comm::MPI.Comm
end

@inline _plan_real_type(::Type{Float32}) = Float32
@inline _plan_real_type(::Type{ComplexF32}) = Float32
@inline _plan_real_type(::Type{Float64}) = Float64
@inline _plan_real_type(::Type{ComplexF64}) = Float64
@inline _plan_real_type(::Type) = nothing

function _validate_plan_prototype_precision(prototype_θφ::PencilArray,
                                            comm, operation::Symbol)
    candidate = _plan_real_type(eltype(prototype_θφ))
    code = candidate === Float32 ? 1 : candidate === Float64 ? 2 : 0
    min_code = MPI.Allreduce(code, min, comm)
    max_code = MPI.Allreduce(code, max, comm)
    if code == 0 || min_code != max_code
        throw(ArgumentError(
            "$operation requires one replicated Float32/64 or " *
            "ComplexF32/64 prototype precision on every rank",
        ))
    end
    return candidate
end

function _require_cpu_plan_storage!(comm, operation::Symbol, values...)
    storage_code = _validate_parallel_storage!(comm, operation, values...)
    storage_code == 0 || throw(SHTnsKit.BackendUnavailableError(
        operation,
        "this reusable distributed plan owns CPU scratch storage; " *
        "construct and execute it with CPU-backed PencilArrays",
    ))
    return nothing
end

function DistAnalysisPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    comm = communicator(prototype_θφ)
    _require_cpu_plan_storage!(comm, :DistAnalysisPlan, prototype_θφ)
    _validate_cfg_spatial_prototype(cfg, prototype_θφ, "DistAnalysisPlan")
    _validate_replicated_call_signature(comm, "DistAnalysisPlan", (use_rfft,))
    cfg_fingerprint = _cfg_fingerprint(cfg)
    RT = _validate_plan_prototype_precision(
        prototype_θφ, comm, :DistAnalysisPlan,
    )
    θ_globals = collect(Int, globalindices(prototype_θφ, 1))
    nθ_local = length(θ_globals)
    nlon_local = size(parent(prototype_θφ), 2)
    # Reduced, like θ_is_distributed and φ_is_local_all: this selects which BRANCH
    # `dist_analysis!` takes, and the two branches enter different full-comm
    # collectives. Per-rank, a pencil with more φ-partitions than columns sends
    # the owner into the planned Allreduce and the empty ranks into
    # `dist_analysis_standard`'s own Allreduce — they never pair, and the job hangs.
    fallback_standard = MPI.Allreduce(nlon_local != cfg.nlon, |, comm)
    weights_cache = Float64[cfg.w[i] for i in θ_globals]
    x_cache = Float64[cfg.x[i] for i in θ_globals]
    P = Vector{Float64}(undef, cfg.lmax + 1)
    nbins = use_rfft ? (cfg.nlon ÷ 2 + 1) : cfg.nlon
    Fθm = Matrix{Complex{RT}}(undef, nθ_local, nbins)
    Alm_work = Matrix{Complex{RT}}(undef, cfg.lmax + 1, cfg.mmax + 1)
    # Reduced, not per-rank. The consumers (`dist_analysis!`,
    # `dist_analysis_sphtor!`) guard a full-comm `MPI.Allreduce!` with this flag,
    # so a topology where one rank owns every latitude and the rest own none
    # (nlat=1 over ≥2 θ-partitions) would have the owner skip while the empty
    # ranks block forever. Computed once at plan construction, not per call.
    θ_is_distributed = MPI.Allreduce(nθ_local < cfg.nlat, |, comm)
    # No Comm_split: this branch requires `!fallback_standard`, i.e. the rank owns
    # the COMPLETE φ range, so every rank's φ-colour would be 1 and the
    # split just duplicates `comm` — at the cost of a synchronizing collective
    # and a live communicator per plan, reclaimed only when GC runs the
    # finalizer. Code that rebuilds a plan per shell or timestep can exhaust the
    # MPI communicator pool that way.
    reduce_comm = comm
    return DistAnalysisPlan(cfg, cfg_fingerprint, prototype_θφ, use_rfft, fallback_standard,
                            θ_globals, weights_cache, x_cache, P, Fθm, Alm_work,
                            θ_is_distributed, reduce_comm)
end

struct DistPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
end

function DistPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    comm = communicator(prototype_θφ)
    _require_cpu_plan_storage!(comm, :DistPlan, prototype_θφ)
    _validate_cfg_spatial_prototype(cfg, prototype_θφ, "DistPlan")
    _validate_replicated_call_signature(comm, "DistPlan", (use_rfft,))
    _validate_plan_prototype_precision(prototype_θφ, comm, :DistPlan)
    return DistPlan(cfg, prototype_θφ, use_rfft)
end

struct DistSphtorPlan{CT<:Complex}
    cfg::SHTnsKit.SHTConfig
    cfg_fingerprint::UInt
    prototype_θφ::PencilArray
    use_rfft::Bool
    with_spatial_scratch::Bool
    spatial_scratch::Union{Nothing, NamedTuple}
    # --- analysis scratch (always allocated; see DistAnalysisPlan) ---
    fallback_standard::Bool
    θ_globals::Vector{Int}
    x_cache::Vector{Float64}
    sθ_cache::Vector{Float64}
    inv_sθ_cache::Vector{Float64}
    weights_cache::Vector{Float64}
    P::Vector{Float64}
    dPdtheta::Vector{Float64}
    P_over_sth::Vector{Float64}
    Pbuf::Vector{Float64}
    Ftθm::Matrix{CT}
    Fpθm::Matrix{CT}
    Slm_work::Matrix{CT}
    Tlm_work::Matrix{CT}
    θ_is_distributed::Bool
    reduce_comm::MPI.Comm
end

function DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    comm = communicator(prototype_θφ)
    _require_cpu_plan_storage!(comm, :DistSphtorPlan, prototype_θφ)
    _validate_cfg_spatial_prototype(cfg, prototype_θφ, "DistSphtorPlan")
    _validate_replicated_call_signature(
        comm, "DistSphtorPlan", (with_spatial_scratch, use_rfft),
    )
    cfg_fingerprint = _cfg_fingerprint(cfg)
    RT = _validate_plan_prototype_precision(
        prototype_θφ, comm, :DistSphtorPlan,
    )
    CT = Complex{RT}
    θ_globals = collect(Int, globalindices(prototype_θφ, 1))
    nθ_local = length(θ_globals)
    nlon = cfg.nlon
    lmax = cfg.lmax
    scratch = if with_spatial_scratch
        # Pre-allocate all scratch buffers needed for synthesis
        (
            Fθ = Matrix{CT}(undef, nθ_local, nlon),   # Fourier coeffs for Vθ
            Fφ = Matrix{CT}(undef, nθ_local, nlon),   # Fourier coeffs for Vφ
            Vtθ = Matrix{RT}(undef, nθ_local, nlon),  # Real output for Vθ
            Vpθ = Matrix{RT}(undef, nθ_local, nlon),  # Real output for Vφ
            P = Vector{Float64}(undef, lmax + 1),
            dPdtheta = Vector{Float64}(undef, lmax + 1),
            P_over_sth = Vector{Float64}(undef, lmax + 1),
            Pbuf = Vector{Float64}(undef, lmax + 2),
        )
    else
        nothing
    end
    nlon_local = size(parent(prototype_θφ), 2)
    # Reduced — see DistAnalysisPlan: a per-rank value sends different ranks
    # into different branches, each entering its own full-comm collective.
    fallback_standard = MPI.Allreduce(nlon_local != nlon, |, comm)
    x_cache = Vector{Float64}(undef, nθ_local)
    sθ_cache = Vector{Float64}(undef, nθ_local)
    inv_sθ_cache = Vector{Float64}(undef, nθ_local)
    weights_cache = Vector{Float64}(undef, nθ_local)
    for (ii, iglob) in enumerate(θ_globals)
        x = cfg.x[iglob]
        sθ = sqrt(max(0.0, 1 - x * x))
        x_cache[ii] = x
        sθ_cache[ii] = sθ
        inv_sθ_cache[ii] = sθ == 0 ? 0.0 : 1.0 / sθ
        weights_cache[ii] = cfg.w[iglob]
    end
    nbins = use_rfft ? (nlon ÷ 2 + 1) : nlon
    Ftθm = Matrix{CT}(undef, nθ_local, nbins)
    Fpθm = Matrix{CT}(undef, nθ_local, nbins)
    Slm_work = Matrix{CT}(undef, lmax + 1, cfg.mmax + 1)
    Tlm_work = Matrix{CT}(undef, lmax + 1, cfg.mmax + 1)
    # Reduced, not per-rank. The consumers (`dist_analysis!`,
    # `dist_analysis_sphtor!`) guard a full-comm `MPI.Allreduce!` with this flag,
    # so a topology where one rank owns every latitude and the rest own none
    # (nlat=1 over ≥2 θ-partitions) would have the owner skip while the empty
    # ranks block forever. Computed once at plan construction, not per call.
    θ_is_distributed = MPI.Allreduce(nθ_local < cfg.nlat, |, comm)
    reduce_comm = comm   # see DistAnalysisPlan: the split was provably a no-op here
    return DistSphtorPlan(cfg, cfg_fingerprint, prototype_θφ, use_rfft, with_spatial_scratch, scratch,
                          fallback_standard, θ_globals, x_cache, sθ_cache, inv_sθ_cache,
                          weights_cache,
                          Vector{Float64}(undef, lmax + 1), Vector{Float64}(undef, lmax + 1),
                          Vector{Float64}(undef, lmax + 1), Vector{Float64}(undef, lmax + 2),
                          Ftθm, Fpθm, Slm_work, Tlm_work, θ_is_distributed, reduce_comm)
end

struct DistQstPlan
    cfg::SHTnsKit.SHTConfig
    prototype_θφ::PencilArray
    use_rfft::Bool
    # QST analysis = scalar (radial) + sphtor (tangential); delegate to the
    # planned sub-transforms so all scratch lives in the sub-plans.
    scalar_plan::DistAnalysisPlan
    sphtor_plan::DistSphtorPlan
end

function DistQstPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    scalar_plan = DistAnalysisPlan(cfg, prototype_θφ; use_rfft)
    sphtor_plan = DistSphtorPlan(cfg, prototype_θφ; with_spatial_scratch, use_rfft)
    return DistQstPlan(cfg, prototype_θφ, use_rfft, scalar_plan, sphtor_plan)
end
