##########
# PencilArray operators
##########

"""
    dist_apply_laplacian!(cfg, Alm_pencil::PencilArray)

In-place multiply by -l(l+1) for distributed Alm with dims (:l,:m). No communication.
"""
function SHTnsKit.dist_apply_laplacian!(cfg::SHTnsKit.SHTConfig, Alm_pencil::PencilArray)
    # Scalar-indexing the PencilArray avoids `.*=` on a row slice, which tries
    # to construct a differently sized `similar` PencilArray and throws
    # `DimensionMismatch`.
    gsize = PencilArrays.size_global(Alm_pencil)
    gsize[1] == cfg.lmax + 1 && gsize[2] >= cfg.mmax + 1 ||
        throw(DimensionMismatch("spectral pencil must have global size " *
                                "($(cfg.lmax + 1), >=$(cfg.mmax + 1)); got $gsize"))
    lloc = axes(Alm_pencil, 1); gl_l = collect(Int, globalindices(Alm_pencil, 1))
    mloc = axes(Alm_pencil, 2); gl_m = collect(Int, globalindices(Alm_pencil, 2))
    @inbounds for (ii, il) in enumerate(lloc)
        lval = gl_l[ii] - 1
        factor = -(lval * (lval + 1))
        for (jj, jm) in enumerate(mloc)
            mval = gl_m[jj] - 1
            (mval <= cfg.mmax && mval % cfg.mres == 0 && lval >= mval) || continue
            Alm_pencil[il, jm] *= factor
        end
    end
    return Alm_pencil
end

"""
    dist_SH_mul_mx!(cfg, mx, Alm_pencil::PencilArray, R_pencil::PencilArray)

Apply 3-diagonal operator to distributed Alm pencils using per-m Allgatherv of l-columns.
Forward pass: R[l,m] = mx[2*lm_prev+2]*Q[l-1,m] + mx[2*lm_next+1]*Q[l+1,m]
where lm_prev = LM_index(l-1,m) and lm_next = LM_index(l+1,m).
"""
function SHTnsKit.dist_SH_mul_mx!(cfg::SHTnsKit.SHTConfig, mx::AbstractVector{<:Real}, Alm_pencil::PencilArray, R_pencil::PencilArray)
    lmax, mmax, mres = cfg.lmax, cfg.mmax, cfg.mres
    comm = communicator(Alm_pencil)
    _validate_prototype_communicator(comm, R_pencil, "dist_SH_mul_mx!")
    local_alias = parent(Alm_pencil) === parent(R_pencil) ||
                  Base.mightalias(parent(Alm_pencil), parent(R_pencil))
    global_alias = MPI.Allreduce(local_alias, |, comm)
    global_alias && throw(ArgumentError(
        "dist_SH_mul_mx! is out-of-place; input and output PencilArrays must not alias"))
    length(mx) == 2cfg.nlm ||
        throw(DimensionMismatch("mx length must be 2*nlm=$(2cfg.nlm); got $(length(mx))"))
    in_size = PencilArrays.size_global(Alm_pencil)
    out_size = PencilArrays.size_global(R_pencil)
    in_size[1] == lmax + 1 && in_size[2] >= mmax + 1 ||
        throw(DimensionMismatch("input spectral pencil must have global size " *
                                "($(lmax + 1), >=$(mmax + 1)); got $in_size"))
    out_size == in_size ||
        throw(DimensionMismatch("input/output spectral pencils must have matching global sizes; " *
                                "got $in_size and $out_size"))
    for dim in 1:2
        globalindices(Alm_pencil, dim) == globalindices(R_pencil, dim) ||
            throw(DimensionMismatch("input/output spectral pencils must have identical local ranges"))
    end
    size(parent(Alm_pencil)) == size(parent(R_pencil)) ||
        throw(DimensionMismatch("input/output spectral pencils must have identical local storage"))
    lloc = axes(Alm_pencil, 1); mloc = axes(Alm_pencil, 2)
    gl_l = collect(Int, globalindices(Alm_pencil, 1))
    gl_m = collect(Int, globalindices(Alm_pencil, 2))
    # This kernel reads the l±1 neighbours of every local coefficient out of a
    # full l-column, so it is only valid when the l dimension is NOT distributed
    # (the usual case: PencilArrays splits the last dim, m). That was asserted in
    # a comment only — on an l-decomposed pencil the unowned entries of
    # `col_full` stayed uninitialized and the operator silently returned garbage.
    # Check it instead: the condition is rank-local and identical in form on
    # every rank, so a violating pencil throws everywhere rather than deadlocking.
    if length(gl_l) != lmax + 1
        throw(ArgumentError("dist_SH_mul_mx! requires the l dimension to be local " *
                            "(rank owns $(length(gl_l)) of $(lmax+1) degrees); " *
                            "decompose the spectral pencil along m only"))
    end
    CT = promote_type(eltype(Alm_pencil), eltype(R_pencil), complex(eltype(mx)))  # AD/Float32-safe
    col_full = zeros(CT, lmax + 1)
    fill!(R_pencil, zero(eltype(R_pencil)))
    for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        # Columns this operator does not compute must still be DEFINED: the
        # caller's `R_pencil` is typically freshly `undef`-allocated, so a bare
        # `continue` would leave them holding garbage rather than the zero the
        # operator implies. Zero, then skip.
        #   * m > mmax          — outside the spectral band (dealiased grids)
        #   * m % mres != 0     — not a stored order; `LM_index` throws on these
        (mval <= mmax && mval % mres == 0) || continue
        # Extract the full l-column from local data
        @inbounds for (ii, il) in enumerate(lloc)
            col_full[gl_l[ii]] = Alm_pencil[il, jm]
        end
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            lval >= mval || continue
            acc = zero(CT)
            # Contribution from lower neighbor Y_{l-1}^m (uses mx[2*lm_prev + 2])
            if lval > mval && lval > 0
                lm_prev = SHTnsKit.LM_index(lmax, mres, lval - 1, mval)
                c_from_below = mx[2*lm_prev + 2]  # b_{l-1}^m coefficient
                acc += c_from_below * col_full[lval]  # col_full[lval] = Q[l-1,m]
            end
            # Contribution from upper neighbor Y_{l+1}^m (uses mx[2*lm_next + 1])
            if lval < lmax && lval + 1 >= mval
                lm_next = SHTnsKit.LM_index(lmax, mres, lval + 1, mval)
                c_from_above = mx[2*lm_next + 1]  # a_{l+1}^m coefficient
                acc += c_from_above * col_full[lval + 2]  # col_full[lval+2] = Q[l+1,m]
            end
            R_pencil[il, jm] = acc
        end
    end
    return R_pencil
end

"""
    dist_spatial_divergence(cfg, Vtθφ, Vpθφ; prototype_θφ=Vtθφ, use_rfft=false, real_output=true)

Compute ∇·V for a distributed horizontal vector field using spectral decomposition.
"""
function SHTnsKit.dist_spatial_divergence(cfg::SHTnsKit.SHTConfig,
                                          Vtθφ::PencilArray, Vpθφ::PencilArray;
                                          prototype_θφ::PencilArray=Vtθφ,
                                          use_rfft::Bool=false,
                                          real_output::Bool=true)
    Slm, _ = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ; use_rfft)
    δlm = SHTnsKit.divergence_from_spheroidal(cfg, Slm)
    return SHTnsKit.dist_synthesis(cfg, δlm; prototype_θφ=prototype_θφ,
                                   real_output=real_output, use_rfft=use_rfft)
end

"""
    dist_spatial_vorticity(cfg, Vtθφ, Vpθφ; prototype_θφ=Vtθφ, use_rfft=false, real_output=true)

Compute vertical vorticity (∇×V)·r̂ for a distributed horizontal vector field.
"""
function SHTnsKit.dist_spatial_vorticity(cfg::SHTnsKit.SHTConfig,
                                         Vtθφ::PencilArray, Vpθφ::PencilArray;
                                         prototype_θφ::PencilArray=Vtθφ,
                                         use_rfft::Bool=false,
                                         real_output::Bool=true)
    _, Tlm = SHTnsKit.dist_analysis_sphtor(cfg, Vtθφ, Vpθφ; use_rfft)
    ζlm = SHTnsKit.vorticity_from_toroidal(cfg, Tlm)
    return SHTnsKit.dist_synthesis(cfg, ζlm; prototype_θφ=prototype_θφ,
                                   real_output=real_output, use_rfft=use_rfft)
end

"""
    dist_scalar_laplacian(cfg, fθφ; prototype_θφ=fθφ, use_rfft=false, real_output=true)

Apply spherical Laplacian to a distributed scalar field by transforming to spectral
space, scaling by −l(l+1), and synthesizing back.
"""
function SHTnsKit.dist_scalar_laplacian(cfg::SHTnsKit.SHTConfig,
                                        fθφ::PencilArray;
                                        prototype_θφ::PencilArray=fθφ,
                                        use_rfft::Bool=false,
                                        real_output::Bool=true)
    Alm = SHTnsKit.dist_analysis(cfg, fθφ; use_rfft)
    SHTnsKit.dist_apply_laplacian!(cfg, Alm)
    return SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=prototype_θφ,
                                   real_output=real_output, use_rfft=use_rfft)
end

"""
    dist_scalar_laplacian!(cfg, outθφ, inθφ; use_rfft=false, real_output=true)

In-place version that writes the Laplacian of `inθφ` into `outθφ`.
"""
function SHTnsKit.dist_scalar_laplacian!(cfg::SHTnsKit.SHTConfig,
                                         outθφ::PencilArray,
                                         inθφ::PencilArray;
                                         use_rfft::Bool=false,
                                         real_output::Bool=true)
    result = SHTnsKit.dist_scalar_laplacian(cfg, inθφ; prototype_θφ=outθφ,
                                            use_rfft=use_rfft, real_output=real_output)
    copyto!(outθφ, result)
    return outθφ
end
