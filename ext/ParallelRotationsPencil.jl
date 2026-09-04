##########
# PencilArray rotations
##########

"""Validate input/output spectral layouts before rotation writes or collectives."""
function _require_matching_rotation_pencils(cfg::SHTnsKit.SHTConfig,
                                            Alm_pencil::PencilArray,
                                            R_pencil::PencilArray)
    comm = communicator(Alm_pencil)
    _validate_cfg_replicated(cfg, comm)
    gl_l = collect(Int, globalindices(Alm_pencil, 1))
    gl_m = collect(Int, globalindices(Alm_pencil, 2))
    out_l = collect(Int, globalindices(R_pencil, 1))
    out_m = collect(Int, globalindices(R_pencil, 2))
    expected = (cfg.lmax + 1, cfg.mmax + 1)
    comm_ok = MPI.Comm_compare(comm, communicator(R_pencil)) != MPI.UNEQUAL
    local_ok = PencilArrays.size_global(Alm_pencil) == expected &&
               PencilArrays.size_global(R_pencil) == expected &&
               PencilArrays.permutation(Alm_pencil) isa PencilArrays.NoPermutation &&
               PencilArrays.permutation(R_pencil) isa PencilArrays.NoPermutation &&
               gl_l == out_l && gl_m == out_m &&
               ndims(parent(Alm_pencil)) == 2 && ndims(parent(R_pencil)) == 2 &&
               size(parent(Alm_pencil)) == (length(gl_l), length(gl_m)) &&
               size(parent(R_pencil)) == size(parent(Alm_pencil)) && comm_ok
    # A rotation produces one logical distributed spectrum, so even the
    # communication-free Z kernel needs one rank-symmetric layout verdict.
    all_ok = MPI.Allreduce(local_ok, &, comm)
    all_ok || throw(DimensionMismatch(
        "rotation input/output must be unpermuted spectral pencils with global " *
        "size $expected, identical local ranges, and congruent communicators",
    ))
    return gl_l, gl_m
end

"""Reject decompositions that would enter row-wise collectives with different l rows."""
function _require_m_distributed_spectral_pencil(cfg::SHTnsKit.SHTConfig,
                                                Alm_pencil::PencilArray,
                                                R_pencil::PencilArray)
    gl_l, gl_m = _require_matching_rotation_pencils(cfg, Alm_pencil, R_pencil)
    cfg.mres == 1 || throw(ArgumentError(
        "distributed Y rotation requires mres==1 (got mres=$(cfg.mres)); " *
        "a Y rotation mixes orders and cannot be represented in an mres-strided layout",
    ))
    expected = cfg.lmax + 1
    full_l_rows = !isempty(gl_l) && length(gl_l) == expected &&
                  first(gl_l) == 1 && last(gl_l) == expected
    all_full_l_rows = MPI.Allreduce(full_l_rows, &, communicator(Alm_pencil))
    all_full_l_rows || throw(ArgumentError(
        "distributed Y rotation requires a spectral pencil decomposed only along m; " *
        "each rank must own all $expected l rows",
    ))
    return gl_l, gl_m
end

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig,
                            Alm_pencil::PencilArray, alpha::Real)
    # Return a new PencilArray; do not mutate the input
    R_pencil = similar(Alm_pencil)
    return SHTnsKit.dist_SH_Zrotate(cfg, Alm_pencil, alpha, R_pencil)
end

function SHTnsKit.dist_SH_Zrotate(cfg::SHTnsKit.SHTConfig,
                            Alm_pencil::PencilArray, alpha::Real,
                            R_pencil::PencilArray)
    gl_l, gl_m = _require_matching_rotation_pencils(cfg, Alm_pencil, R_pencil)
    _validate_replicated_call_signature(
        communicator(Alm_pencil), "dist_SH_Zrotate", (alpha,),
    )
    A_local = parent(Alm_pencil)
    R_local = parent(R_pencil)
    fill!(R_local, zero(eltype(R_local)))
    for (jj, gm) in enumerate(gl_m)
        mval = gm - 1
        mval % cfg.mres == 0 || continue
        phase = cis(-mval * alpha)
        @inbounds for (ii, gl) in enumerate(gl_l)
            gl - 1 >= mval || continue
            R_local[ii, jj] = phase * A_local[ii, jj]
        end
    end
    return R_pencil
end

function SHTnsKit.dist_SH_Yrotate_allgatherm!(cfg::SHTnsKit.SHTConfig, 
                                            Alm_pencil::PencilArray, 
                                            beta::Real, 
                                            R_pencil::PencilArray)

    lmax, mmax = cfg.lmax, cfg.mmax
    
    comm = communicator(Alm_pencil)

    lloc = axes(Alm_pencil, 1)
    mloc = axes(Alm_pencil, 2)
    
    gl_l, gl_m = _require_m_distributed_spectral_pencil(cfg, Alm_pencil, R_pencil)
    _validate_replicated_call_signature(
        comm, "dist_SH_Yrotate_allgatherm!", (beta,),
    )

    nm_local = length(mloc)
    counts_m = Allgather(nm_local, comm)
    sum(counts_m) == mmax + 1 || throw(ArgumentError(
        "distributed Y rotation requires a non-overlapping partition of m=0:$mmax",
    ))
    displs_m = cumsum([0; counts_m[1:end-1]])
    a_full = Vector{ComplexF64}(undef, mmax + 1)
    scales = SHTnsKit._ensure_norm_scale_matrix!(cfg)
    A_local = parent(Alm_pencil)
    R_local = parent(R_pencil)

    # Hoisted scratch buffers reused across l-rows.
    a_local = Vector{ComplexF64}(undef, nm_local)
    max_n2 = 2*lmax + 1
    b_buf = Vector{ComplexF64}(undef, max_n2)
    c_buf = Vector{ComplexF64}(undef, max_n2)
    D_buf = Matrix{Float64}(undef, max_n2, max_n2)             # reused Wigner-d buffer (was alloc'd per l-row)
    D_work = similar(D_buf)

    for (ii, il) in enumerate(lloc)
        lval = gl_l[ii] - 1
        # Index the validated, unpermuted parent directly. PencilArray's view
        # construction indexes the first element of every dimension and throws
        # when this rank owns zero m columns, even though a zero-count
        # Allgatherv is valid and required for the other ranks to progress.
        copyto!(a_local, view(A_local, ii, :))
        Allgatherv!(a_local, VBuffer(a_full, counts_m), comm)
        mm = min(lval, mmax)
        n2 = 2*lval + 1
        b = view(b_buf, 1:n2); fill!(b, 0.0 + 0.0im)
        if lval >= 0
            b[0 + lval + 1] = scales[lval+1, 1] * a_full[1]
        end

        for m in 1:mm
            a_int = scales[lval+1, m+1] * a_full[m+1]
            b[m + lval + 1] = a_int
            b[-m + lval + 1] = (-1.0)^m * conj(a_int)
        end

        dl = view(D_buf, 1:n2, 1:n2)
        dwork = view(D_work, 1:n2, 1:n2)
        SHTnsKit.wigner_d_matrix!(dl, lval, Float64(beta), dwork)
        c = view(c_buf, 1:n2)
        @inbounds for mi in -lval:lval
            acc = 0.0 + 0.0im
            for mp in -lval:lval
                acc += dl[mi + lval + 1, mp + lval + 1] * b[mp + lval + 1]
            end
            c[mi + lval + 1] = acc
        end

        for (jj, jm) in enumerate(mloc)
            mval = gl_m[jj] - 1
            if mval <= lval
                R_local[ii, jj] = c[mval + lval + 1] / scales[lval+1, mval+1]
            else
                R_local[ii, jj] = 0.0 + 0.0im
            end
        end
    end
    
    return R_pencil
end

"""
    dist_SH_Yrotate_truncgatherm!(cfg, Alm_pencil, beta, R_pencil)

Allgather only m-columns with m ≤ l for each l-row, reducing communication for small l.
"""
function SHTnsKit.dist_SH_Yrotate_truncgatherm!(cfg::SHTnsKit.SHTConfig,
                                               Alm_pencil::PencilArray,
                                               beta::Real,
                                               R_pencil::PencilArray)
    # Materialize concrete-typed inputs here and run the row loop behind a
    # function barrier: keep the concrete index vectors out of the row loop to
    # avoid boxing and unnecessary per-call allocations.
    comm = communicator(Alm_pencil)
    gl_l, gl_m = _require_m_distributed_spectral_pencil(cfg, Alm_pencil, R_pencil)
    _validate_replicated_call_signature(
        comm, "dist_SH_Yrotate_truncgatherm!", (beta,),
    )
    counts_m = collect(Int, Allgather(length(gl_m), comm))
    sum(counts_m) == cfg.mmax + 1 || throw(ArgumentError(
        "distributed Y rotation requires a non-overlapping partition of m=0:$(cfg.mmax)",
    ))
    _yrotate_truncgather_rows!(cfg, parent(Alm_pencil), parent(R_pencil),
                               gl_l, gl_m, counts_m,
                               MPI.Comm_size(comm), MPI.Comm_rank(comm),
                               float(beta), comm)
    return R_pencil
end

function _yrotate_truncgather_rows!(cfg::SHTnsKit.SHTConfig,
                                    A_local::AbstractMatrix{<:Complex},
                                    R_local::AbstractMatrix{<:Complex},
                                    gl_l::Vector{Int}, gl_m::Vector{Int},
                                    counts_m::Vector{Int}, nranks::Int, myrank::Int,
                                    beta::Real, comm)
    lmax, mmax = cfg.lmax, cfg.mmax
    # Single exchange of m-ownership counts (done by the caller). The global m
    # order is the ascending rank-block concatenation (the same invariant
    # dist_SH_Zrotate's Allgatherv relies on), so each l-row's truncated
    # per-rank counts are computable locally — one Allgatherv per row instead
    # of three collectives.
    nm_local = length(gl_m)
    counts_l = Vector{Int}(undef, nranks)

    # Hoisted scratch reused across l-rows.
    max_n2 = 2*lmax + 1
    a_local = Vector{ComplexF64}(undef, nm_local)
    a_full = Vector{ComplexF64}(undef, mmax + 1)
    b_buf = Vector{ComplexF64}(undef, max_n2)
    c_buf = Vector{ComplexF64}(undef, max_n2)
    D_buf = Matrix{Float64}(undef, max_n2, max_n2)
    D_work = similar(D_buf)
    scales = SHTnsKit._ensure_norm_scale_matrix!(cfg)

    for ii in 1:size(A_local, 1)
        il = ii
        lval = gl_l[ii] - 1
        mm = min(lval, mmax)
        # Clip each rank's contiguous m block at mm+1 gathered elements total.
        off = 0
        @inbounds for r in 1:nranks
            counts_l[r] = max(0, min(off + counts_m[r], mm + 1) - off)
            off += counts_m[r]
        end
        count_local = counts_l[myrank + 1]
        # Owned m values ascend, so the first count_local columns are m ≤ mm.
        @inbounds for k in 1:count_local
            a_local[k] = A_local[il, k]
        end
        # Gathered blocks land in m order: a_full[m+1] = A[l, m] for m = 0:mm.
        Allgatherv!(view(a_local, 1:count_local), VBuffer(a_full, counts_l), comm)
        # Build symmetric b of size 2l+1 from positive m part
        n2 = 2*lval + 1
        b = view(b_buf, 1:n2); fill!(b, 0.0 + 0.0im)
        if lval >= 0
            b[0 + lval + 1] = (mm >= 0 ? scales[lval+1, 1] * a_full[1] : 0.0 + 0.0im)
        end
        for m in 1:mm
            a_int = scales[lval+1, m+1] * a_full[m+1]
            b[m + lval + 1] = a_int
            b[-m + lval + 1] = (-1.0)^m * conj(a_int)
        end
        # d-matrix multiply (Wigner-d built into the hoisted buffer)
        dl = view(D_buf, 1:n2, 1:n2)
        dwork = view(D_work, 1:n2, 1:n2)
        SHTnsKit.wigner_d_matrix!(dl, lval, Float64(beta), dwork)
        c = view(c_buf, 1:n2)
        @inbounds for mi in -lval:lval
            acc = 0.0 + 0.0im
            for mp in -lval:lval
                acc += dl[mi + lval + 1, mp + lval + 1] * b[mp + lval + 1]
            end
            c[mi + lval + 1] = acc
        end
        # Write back local columns
        @inbounds for jj in 1:nm_local
            mval = gl_m[jj] - 1
            if mval <= lval
                R_local[il, jj] = c[mval + lval + 1] / scales[lval+1, mval+1]
            else
                R_local[il, jj] = 0.0 + 0.0im
            end
        end
    end
    return nothing
end
function SHTnsKit.dist_SH_Yrotate(cfg::SHTnsKit.SHTConfig,
                                  Alm_pencil::PencilArray,
                                  beta::Real,
                                  R_pencil::PencilArray)
    # Truncated gather reduces bandwidth for small l
    return SHTnsKit.dist_SH_Yrotate_truncgatherm!(cfg, Alm_pencil, beta, R_pencil)
end

"""
    dist_SH_Yrotate90(cfg, Alm_pencil::PencilArray, R_pencil::PencilArray)

Rotate distributed Alm by +90° around Y in Pencil layout.
"""
function SHTnsKit.dist_SH_Yrotate90(cfg::SHTnsKit.SHTConfig,
                                    Alm_pencil::PencilArray,
                                    R_pencil::PencilArray)
    return SHTnsKit.dist_SH_Yrotate(cfg, Alm_pencil, π/2, R_pencil)
end

"""
    dist_SH_Xrotate90(cfg, Alm_pencil::PencilArray, R_pencil::PencilArray)

Rotate distributed Alm by +90° around X using the ZYZ Euler matrix
`Rz(-π/2) * Ry(π/2) * Rz(π/2)`.
"""
function SHTnsKit.dist_SH_Xrotate90(cfg::SHTnsKit.SHTConfig,
                                    Alm_pencil::PencilArray,
                                    R_pencil::PencilArray)
    return SHTnsKit.dist_SH_rotate_euler(cfg, Alm_pencil, -π/2, π/2, π/2, R_pencil)
end

##########
# Composite Euler rotation on PencilArrays: Rz(α) * Ry(β) * Rz(γ)
##########

function SHTnsKit.dist_SH_rotate_euler(cfg::SHTnsKit.SHTConfig,
                                       Alm_pencil::PencilArray,
                                       α::Real, β::Real, γ::Real,
                                       R_pencil::PencilArray)
    # Temp buffer with same layout
    tmp1 = similar(Alm_pencil)
    tmp2 = similar(Alm_pencil)
    # Matrix products act right-to-left on coefficient vectors.  Apply γ first
    # so the composed operator matches the serial rotation engine's
    # diag(exp(-imα)) * d(β) * diag(exp(-imγ)) convention.
    SHTnsKit.dist_SH_Zrotate(cfg, Alm_pencil, γ, tmp1)
    # Y(β): requires allgather over m
    SHTnsKit.dist_SH_Yrotate_allgatherm!(cfg, tmp1, β, tmp2)
    # Z(α)
    SHTnsKit.dist_SH_Zrotate(cfg, tmp2, α, R_pencil)
    return R_pencil
end

##########
# Convenience wrappers: packed Qlm vectors rotated via distributed Pencil operations
##########

"""
    dist_SH_Zrotate_packed(cfg, Qlm::AbstractVector{<:Complex}, α; prototype_lm::PencilArray) -> Rlm::Vector

Rotate packed real-field Qlm around Z by α using distributed Pencil operations.
"""
function SHTnsKit.dist_SH_Zrotate_packed(cfg::SHTnsKit.SHTConfig,
                                         Qlm::AbstractVector{<:Complex}, α::Real;
                                         prototype_lm::PencilArray)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be $(cfg.nlm)"))
    # Z-rotation is diagonal in m; no communication needed
    Rlm = similar(Qlm)
    @inbounds for k in eachindex(Qlm)
        m = cfg.mi[k]
        Rlm[k] = cis(-m * α) * Qlm[k]
    end
    return Rlm
end

"""
    dist_SH_Yrotate_packed(cfg, Qlm, β; prototype_lm) -> Rlm
"""
function SHTnsKit.dist_SH_Yrotate_packed(cfg::SHTnsKit.SHTConfig,
                                         Qlm::AbstractVector{<:Complex}, β::Real;
                                         prototype_lm::PencilArray)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be $(cfg.nlm)"))
    cfg.mres == 1 || throw(ArgumentError(
        "dist_SH_Yrotate_packed requires mres==1 (got mres=$(cfg.mres)); " *
        "a Y rotation mixes orders and cannot be represented in an mres-strided layout",
    ))
    lmax, mmax = cfg.lmax, cfg.mmax
    Alm = zeros(ComplexF64, lmax+1, mmax+1)
    @inbounds for m in 0:mmax, l in m:lmax
        Alm[l+1, m+1] = Qlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1]
    end
    # Create PencilArrays using the communicator from prototype_lm
    comm = communicator(prototype_lm)
    Alm_p = SHTnsKit.matrix_to_spectral_pencil(cfg, Alm; comm)
    R_p = PencilArray{ComplexF64}(undef, pencil(Alm_p))
    SHTnsKit.dist_SH_Yrotate(cfg, Alm_p, β, R_p)
    Rlm_mat = zeros(ComplexF64, lmax+1, mmax+1)
    lloc = axes(R_p, 1); mloc = axes(R_p, 2)
    gl_l = globalindices(R_p, 1)
    gl_m = globalindices(R_p, 2)
    for (ii, il) in enumerate(lloc), (jj, jm) in enumerate(mloc)
        Rlm_mat[gl_l[ii], gl_m[jj]] = R_p[il, jm]
    end
    MPI.Allreduce!(Rlm_mat, +, communicator(R_p))
    Rlm = similar(Qlm)
    @inbounds for m in 0:mmax, l in m:lmax
        Rlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1] = Rlm_mat[l+1, m+1]
    end
    return Rlm
end

"""
    dist_SH_Yrotate90_packed(cfg, Qlm; prototype_lm) -> Rlm
"""
function SHTnsKit.dist_SH_Yrotate90_packed(cfg::SHTnsKit.SHTConfig,
                                           Qlm::AbstractVector{<:Complex};
                                           prototype_lm::PencilArray)
    return SHTnsKit.dist_SH_Yrotate_packed(cfg, Qlm, π/2; prototype_lm)
end

"""
    dist_SH_Xrotate90_packed(cfg, Qlm; prototype_lm) -> Rlm
"""
function SHTnsKit.dist_SH_Xrotate90_packed(cfg::SHTnsKit.SHTConfig,
                                           Qlm::AbstractVector{<:Complex};
                                           prototype_lm::PencilArray)
    length(Qlm) == cfg.nlm || throw(DimensionMismatch("Qlm length must be $(cfg.nlm)"))
    cfg.mres == 1 || throw(ArgumentError(
        "dist_SH_Xrotate90_packed requires mres==1 (got mres=$(cfg.mres)); " *
        "an X rotation mixes orders and cannot be represented in an mres-strided layout",
    ))
    lmax, mmax = cfg.lmax, cfg.mmax
    Alm = zeros(ComplexF64, lmax+1, mmax+1)
    @inbounds for m in 0:mmax, l in m:lmax
        Alm[l+1, m+1] = Qlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1]
    end
    # Create PencilArrays using the communicator from prototype_lm
    comm = communicator(prototype_lm)
    Alm_p = SHTnsKit.matrix_to_spectral_pencil(cfg, Alm; comm)
    R_p = PencilArray{ComplexF64}(undef, pencil(Alm_p))
    SHTnsKit.dist_SH_rotate_euler(cfg, Alm_p, -π/2, π/2, π/2, R_p)
    Rlm_mat = zeros(ComplexF64, lmax+1, mmax+1)
    lloc = axes(R_p, 1); mloc = axes(R_p, 2)
    gl_l = globalindices(R_p, 1)
    gl_m = globalindices(R_p, 2)
    for (ii, il) in enumerate(lloc), (jj, jm) in enumerate(mloc)
        Rlm_mat[gl_l[ii], gl_m[jj]] = R_p[il, jm]
    end
    MPI.Allreduce!(Rlm_mat, +, communicator(R_p))
    Rlm = similar(Qlm)
    @inbounds for m in 0:mmax, l in m:lmax
        Rlm[SHTnsKit.LM_index(lmax, cfg.mres, l, m) + 1] = Rlm_mat[l+1, m+1]
    end
    return Rlm
end
