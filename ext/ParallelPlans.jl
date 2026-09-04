##########
# Minimal plan structs to keep API stable
##########

"""
    _cfg_fingerprint(cfg)

Fingerprint every configuration field that can affect a distributed transform.
"""
function _cfg_fingerprint(cfg::SHTnsKit.SHTConfig)
    # Hash array *contents*, not merely their dimensions. Grid nodes, weights,
    # and normalization tables are just as result-affecting as lmax/mmax. The
    # precomputed PLM table contents are derived from these values, so their
    # dimensions/enabled state are sufficient here and avoid an O(lmax²*nlat)
    # pass whenever a plan is constructed. Hash the arrays incrementally: one
    # giant heterogeneous tuple forces several kilobytes of boxing per call,
    # which is particularly costly in cached-plan preflight.
    fingerprint = hash((
        cfg.lmax, cfg.mmax, cfg.mres, cfg.nlat, cfg.nlon, cfg.grid_type,
        cfg.nlm, cfg.nspat, cfg.phi_scale, cfg.on_the_fly,
        cfg.howmany, cfg.spec_dist, cfg.south_pole_first,
        cfg.allow_padding, cfg.nlat_padded, cfg.spat_dist,
        cfg.norm, cfg.cs_phase, cfg.real_norm, cfg.robert_form,
        cfg.cphi, cfg.use_plm_tables,
    ))
    for values in (cfg.li, cfg.mi, cfg.θ, cfg.φ, cfg.x, cfg.w, cfg.st, cfg.Nlm)
        fingerprint = hash(values, fingerprint)
    end
    for tables in (
            cfg.plm_tables, cfg.dplm_tables, cfg.NP_tables, cfg.NdP_tables)
        fingerprint = hash(length(tables), fingerprint)
        for table in tables
            fingerprint = hash(size(table), fingerprint)
        end
    end
    return fingerprint
end

"""
    _validate_cfg_replicated(cfg, comm)

Rank 0 broadcasts the configuration fingerprint. Each rank compares it
collectively and all ranks throw together on a mismatch, before entering
transform collectives.
"""
function _validate_cfg_replicated(cfg::SHTnsKit.SHTConfig, comm)
    MPI.Comm_size(comm) > 1 || return
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
    return
end

"""Return true only for communicators with the same process group and rank order."""
@inline function _communicators_congruent(a::MPI.Comm, b::MPI.Comm)
    comparison = MPI.Comm_compare(a, b)
    return comparison == MPI.IDENT || comparison == MPI.CONGRUENT
end

"""
    _validate_prototype_communicator(comm, prototype, operation)

Collectively reject a spatial prototype whose communicator is not congruent to
the plan communicator. The reduction itself uses only `comm`, so validation
does not enter a collective on a mismatched prototype communicator.
"""
function _validate_prototype_communicator(comm::MPI.Comm, prototype::PencilArray,
                                          operation::AbstractString)
    local_ok = _communicators_congruent(comm, communicator(prototype))
    all_ok = MPI.Allreduce(local_ok, &, comm)
    all_ok || throw(ArgumentError(
        "$operation requires the plan and spatial prototype to use congruent MPI communicators",
    ))
    return nothing
end

function _validate_spatial_shape(cfg::SHTnsKit.SHTConfig, prototype::PencilArray,
                                 comm::MPI.Comm, operation::AbstractString)
    expected = (cfg.nlat, cfg.nlon)
    actual = Tuple(PencilArrays.size_global(prototype))
    all_ok = MPI.Allreduce(actual == expected, &, comm)
    all_ok || throw(DimensionMismatch(
        "$operation requires a spatial PencilArray with global shape $expected; got $actual",
    ))
    return nothing
end

"""Validate a cfg-form spatial prototype before caching or indexing its parent."""
function _validate_cfg_spatial_prototype(cfg::SHTnsKit.SHTConfig,
                                         prototype::PencilArray,
                                         operation::AbstractString)
    comm = communicator(prototype)
    _validate_cfg_replicated(cfg, comm)
    _validate_spatial_shape(cfg, prototype, comm, operation)
    _require_unpermuted_pencil(prototype, operation)
    return nothing
end

"""
Validate that a call-time spatial array has the communicator, global shape, and
exact rank-local index ranges used to construct a cached plan.
"""
function _validate_spatial_pencil_against_prototype(
        cfg::SHTnsKit.SHTConfig, expected::PencilArray, actual::PencilArray,
        operation::AbstractString)
    comm = communicator(expected)
    _validate_prototype_communicator(comm, actual, operation)
    _validate_spatial_shape(cfg, actual, comm, operation)

    expected_ranges = PencilArrays.range_local(pencil(expected))
    actual_ranges = PencilArrays.range_local(pencil(actual))
    local_ok = expected_ranges == actual_ranges &&
               size(parent(expected)) == size(parent(actual))
    all_ok = MPI.Allreduce(local_ok, &, comm)
    all_ok || throw(DimensionMismatch(
        "$operation requires the same rank-local spatial ranges as the plan prototype; " *
        "expected $expected_ranges, got $actual_ranges",
    ))
    _require_unpermuted_pencil(actual, operation)
    return nothing
end

"""Validate a spectral PencilArray before its gather collective."""
function _validate_spectral_pencil(
        cfg::SHTnsKit.SHTConfig, spectral::PencilArray,
        spatial_prototype::PencilArray, operation::AbstractString;
        validate_context::Bool=true)
    comm = communicator(spatial_prototype)
    if validate_context
        _validate_cfg_spatial_prototype(cfg, spatial_prototype, operation)
    end
    _validate_prototype_communicator(comm, spectral, operation)

    expected = (cfg.lmax + 1, cfg.mmax + 1)
    actual = Tuple(PencilArrays.size_global(spectral))
    all_ok = MPI.Allreduce(actual == expected, &, comm)
    all_ok || throw(DimensionMismatch(
        "$operation requires spectral PencilArrays with global shape $expected; got $actual",
    ))
    _require_unpermuted_pencil(spectral, operation)
    return nothing
end

"""Collectively require two PencilArrays to have identical rank-local ranges."""
function _validate_matching_pencil_layout(reference::PencilArray,
                                          actual::PencilArray,
                                          operation::AbstractString)
    comm = communicator(reference)
    _validate_prototype_communicator(comm, actual, operation)
    expected_ranges = PencilArrays.range_local(pencil(reference))
    actual_ranges = PencilArrays.range_local(pencil(actual))
    local_ok = expected_ranges == actual_ranges &&
               size(parent(reference)) == size(parent(actual))
    all_ok = MPI.Allreduce(local_ok, &, comm)
    all_ok || throw(DimensionMismatch(
        "$operation requires matching rank-local PencilArray ranges; " *
        "expected $expected_ranges, got $actual_ranges",
    ))
    return nothing
end

"""Collectively require a rank-local call signature to be identical."""
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

"""Collectively reject a configuration changed after a cached plan was built."""
function _validate_cached_plan_cfg(
        cfg::SHTnsKit.SHTConfig, cfg_fingerprint::UInt, comm::MPI.Comm,
        operation::AbstractString)
    local_ok = _cfg_fingerprint(cfg) == cfg_fingerprint
    all_ok = MPI.Allreduce(local_ok, &, comm)
    all_ok || throw(ArgumentError(
        "$operation cannot reuse a plan after its SHTConfig has changed; rebuild the plan",
    ))
    return nothing
end

"""Collectively validate dense spectral matrix shapes before later collectives."""
function _validate_dense_spectral_shapes(
        cfg::SHTnsKit.SHTConfig, comm::MPI.Comm,
        operation::AbstractString, arrays::Tuple)
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    local_ok = all(A -> A === nothing || size(A) == expected, arrays)
    all_ok = MPI.Allreduce(local_ok, &, comm)
    all_ok || throw(DimensionMismatch(
        "$operation requires every dense spectral matrix to have shape $expected",
    ))
    return nothing
end

"""
Validate every coefficient that the configured transform can read from a
replicated dense spectrum, not a bounded sample.  Storage below the triangular
`l >= m` domain, columns excluded by `mres`, and any caller-declared inactive
degrees or `m = 0` column are deliberately ignored: callers may leave those
semantically unused entries uninitialised.  A synthesis computes rank-local
spatial slabs without a final reduction, so one differing active coefficient
would otherwise create one globally inconsistent field while still returning
successfully.
"""
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


"""
    _validate_distributed_plan_preflight(cfg, plan, prototype, operation)

Validate the plan lifecycle, replicated configuration, spectral
dimensions/stride, spatial prototype communicator, and parent-storage order
before a distributed spectral transform performs its first data collective.
"""
function _validate_distributed_plan_preflight(cfg::SHTnsKit.SHTConfig, plan,
                                              prototype::PencilArray,
                                              operation::AbstractString)
    if hasproperty(plan, :closed)
        all_open = MPI.Allreduce(!getproperty(plan, :closed), &, plan.comm)
        all_open || throw(ArgumentError(
            "$operation requires an open distributed spectral plan",
        ))
    end
    _validate_cfg_replicated(cfg, plan.comm)

    expected = (plan.lmax, plan.mmax, plan.mres)
    actual = (cfg.lmax, cfg.mmax, cfg.mres)
    plan_matches = MPI.Allreduce(actual == expected, &, plan.comm)
    plan_matches || throw(ArgumentError(
        "$operation configuration (lmax, mmax, mres)=$actual does not match plan $expected",
    ))

    _validate_prototype_communicator(plan.comm, prototype, operation)
    _validate_spatial_shape(cfg, prototype, plan.comm, operation)
    _require_unpermuted_pencil(prototype, operation)

    # Scratch-backed 2-D plans cache quadrature values and exact local spatial
    # ranges at construction. Same-sized but numerically different configs, or
    # a different decomposition of the same global grid, cannot safely reuse
    # those buffers.
    if hasproperty(plan, :scratch_context)
        context = getproperty(plan, :scratch_context)
        if context !== nothing
            cfg_ok = _cfg_fingerprint(cfg) == context.cfg_fingerprint
            MPI.Allreduce(cfg_ok, &, plan.comm) || throw(ArgumentError(
                "$operation configuration differs from the one used to build the scratch plan",
            ))

            actual_ranges = PencilArrays.range_local(pencil(prototype))
            layout_ok = actual_ranges == context.spatial_ranges &&
                        Tuple(size(parent(prototype))) == context.spatial_parent_size
            MPI.Allreduce(layout_ok, &, plan.comm) || throw(DimensionMismatch(
                "$operation requires the exact rank-local spatial layout used to build the scratch plan",
            ))
        end
    end
    return nothing
end

struct DistAnalysisPlan
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
    Fθm::Matrix{ComplexF64}
    Alm_work::Matrix{ComplexF64}
    θ_is_distributed::Bool
    # θ-column subcomm for the partial-sum reduction (Comm_split once here
    # instead of every call). Equals the full communicator when θ is not
    # distributed or for the fallback path; freed by MPI_Finalize with the
    # plan's lifetime (plans are long-lived by design).
    reduce_comm::MPI.Comm
end

function DistAnalysisPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; use_rfft::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    comm = communicator(prototype_θφ)
    _validate_cfg_spatial_prototype(cfg, prototype_θφ, "DistAnalysisPlan")
    _validate_replicated_call_signature(
        comm, "DistAnalysisPlan", (use_rfft,),
    )
    cfg_fingerprint = _cfg_fingerprint(cfg)
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
    Fθm = Matrix{ComplexF64}(undef, nθ_local, nbins)
    Alm_work = Matrix{ComplexF64}(undef, cfg.lmax + 1, cfg.mmax + 1)
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
    _validate_cfg_spatial_prototype(cfg, prototype_θφ, "DistPlan")
    _validate_replicated_call_signature(
        communicator(prototype_θφ), "DistPlan", (use_rfft,),
    )
    return DistPlan(cfg, prototype_θφ, use_rfft)
end

const _SphtorScratch = NamedTuple{(:Fθ, :Fφ, :Vtθ, :Vpθ, :P, :dPdx),
                                   Tuple{Matrix{ComplexF64}, Matrix{ComplexF64},
                                         Matrix{Float64}, Matrix{Float64},
                                         Vector{Float64}, Vector{Float64}}}

struct DistSphtorPlan
    cfg::SHTnsKit.SHTConfig
    cfg_fingerprint::UInt
    prototype_θφ::PencilArray
    use_rfft::Bool
    with_spatial_scratch::Bool
    spatial_scratch::Union{Nothing, _SphtorScratch}
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
    Ftθm::Matrix{ComplexF64}
    Fpθm::Matrix{ComplexF64}
    Slm_work::Matrix{ComplexF64}
    Tlm_work::Matrix{ComplexF64}
    θ_is_distributed::Bool
    reduce_comm::MPI.Comm
end

function DistSphtorPlan(cfg::SHTnsKit.SHTConfig, prototype_θφ::PencilArray; with_spatial_scratch::Bool=false, use_rfft::Bool=false)
    # use_rfft=true is wired through dist_analysis_standard and dist_synthesis
    # for real inputs/outputs. Case A (φ replicated) uses FFTW.rfft directly;
    # Case B (φ split) uses a row-subcomm gather + FFTW.rfft via
    # distributed_rfft_phi!. Complex-valued callers still use the complex FFT.
    comm = communicator(prototype_θφ)
    _validate_cfg_spatial_prototype(cfg, prototype_θφ, "DistSphtorPlan")
    _validate_replicated_call_signature(
        comm, "DistSphtorPlan", (with_spatial_scratch, use_rfft),
    )
    cfg_fingerprint = _cfg_fingerprint(cfg)
    θ_globals = collect(Int, globalindices(prototype_θφ, 1))
    nθ_local = length(θ_globals)
    nlon = cfg.nlon
    lmax = cfg.lmax
    scratch = if with_spatial_scratch
        # Pre-allocate all scratch buffers needed for synthesis
        (
            Fθ = Matrix{ComplexF64}(undef, nθ_local, nlon),   # Fourier coeffs for Vθ
            Fφ = Matrix{ComplexF64}(undef, nθ_local, nlon),   # Fourier coeffs for Vφ
            Vtθ = Matrix{Float64}(undef, nθ_local, nlon),     # Real output for Vθ
            Vpθ = Matrix{Float64}(undef, nθ_local, nlon),     # Real output for Vφ
            P = Vector{Float64}(undef, lmax + 1),             # Legendre polynomial buffer
            dPdx = Vector{Float64}(undef, lmax + 1),          # Legendre derivative buffer
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
    Ftθm = Matrix{ComplexF64}(undef, nθ_local, nbins)
    Fpθm = Matrix{ComplexF64}(undef, nθ_local, nbins)
    Slm_work = Matrix{ComplexF64}(undef, lmax + 1, cfg.mmax + 1)
    Tlm_work = Matrix{ComplexF64}(undef, lmax + 1, cfg.mmax + 1)
    # Reduced, not per-rank. The consumers (`dist_analysis!`,
    # `dist_analysis_sphtor!`) guard a full-comm `MPI.Allreduce!` with this flag,
    # so a topology where one rank owns every latitude and the rest own none
    # (nlat=1 over ≥2 θ-partitions) would have the owner skip while the empty
    # ranks block forever. Computed once at plan construction, not per call.
    θ_is_distributed = MPI.Allreduce(nθ_local < cfg.nlat, |, comm)
    reduce_comm = comm   # see DistAnalysisPlan: the split was provably a no-op here
    return DistSphtorPlan(cfg, cfg_fingerprint, prototype_θφ,
                          use_rfft, with_spatial_scratch, scratch,
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
