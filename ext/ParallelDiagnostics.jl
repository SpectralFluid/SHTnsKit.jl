##########
# Pencil-aware diagnostics (MPI reductions)
##########

@inline function _diagnostic_expected_size(cfg::SHTnsKit.SHTConfig, domain::Symbol)
    domain === :spectral && return (cfg.lmax + 1, cfg.mmax + 1)
    domain === :spatial && return (cfg.nlat, cfg.nlon)
    throw(ArgumentError("unknown diagnostic domain: $domain"))
end

@inline _diagnostic_unpermuted(A::PencilArray) =
    PencilArrays.permutation(A) == PencilArrays.NoPermutation()

@inline function _diagnostic_comm_compatible(A::PencilArray, B::PencilArray)
    cmp = MPI.Comm_compare(communicator(A), communicator(B))
    return cmp == MPI.IDENT || cmp == MPI.CONGRUENT
end

"""
Validate a single distributed diagnostic operand collectively.

The collective verdict is important: if only one rank has a malformed local
layout, every rank must throw before any later diagnostic reduction.  Diagnostics
currently require logical and memory dimension order to agree; accepting a
permuted parent would make future parent-array fast paths silently reinterpret
the `(l,m)` or `(θ,φ)` axes.
"""
function _require_diagnostic_array(cfg::SHTnsKit.SHTConfig, A::PencilArray,
                                   domain::Symbol, name::AbstractString;
                                   options::Tuple=())
    comm = communicator(A)
    _validate_cfg_replicated(cfg, comm)
    _validate_replicated_call_signature(
        comm, "$name diagnostic", (options, eltype(A)))
    expected = _diagnostic_expected_size(cfg, domain)
    local_ok = ndims(A) == 2 &&
               size_global(A) == expected &&
               _diagnostic_unpermuted(A)
    all_ok = MPI.Allreduce(local_ok, &, comm)
    all_ok || throw(ArgumentError(
        "$name must be an unpermuted distributed array with global size $expected"))
    return nothing
end


"""Validate matching distributed diagnostic operands before paired indexing."""
function _require_diagnostic_pair(cfg::SHTnsKit.SHTConfig,
                                  A::PencilArray, B::PencilArray,
                                  domain::Symbol,
                                  names::Tuple{<:AbstractString,<:AbstractString};
                                  options::Tuple=())
    comm = communicator(A)
    _validate_cfg_replicated(cfg, comm)
    _validate_replicated_call_signature(
        comm, "$(names[1])/$(names[2]) diagnostic",
        (options, eltype(A), eltype(B)))
    expected = _diagnostic_expected_size(cfg, domain)
    same_ranges = all(dim -> globalindices(A, dim) == globalindices(B, dim), 1:2)
    local_ok = ndims(A) == 2 && ndims(B) == 2 &&
               size_global(A) == expected && size_global(B) == expected &&
               _diagnostic_unpermuted(A) && _diagnostic_unpermuted(B) &&
               _diagnostic_comm_compatible(A, B) && same_ranges
    all_ok = MPI.Allreduce(local_ok, &, comm)
    all_ok || throw(ArgumentError(
        "$(names[1]) and $(names[2]) must be unpermuted distributed arrays " *
        "with global size $expected, matching local ranges, and congruent communicators"))
    return nothing
end

function SHTnsKit.energy_scalar(cfg::SHTnsKit.SHTConfig, Alm::PencilArray; real_field::Bool=true)
    _require_diagnostic_array(cfg, Alm, :spectral, "Alm"; options=(real_field,))
    lloc = axes(Alm, 1); mloc = axes(Alm, 2)
    gl_l = collect(Int, globalindices(Alm, 1))
    gl_m = collect(Int, globalindices(Alm, 2))
    RT = promote_type(Float64, real(float(eltype(Alm))))
    e_local = zero(RT)
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval % cfg.mres == 0 || continue
        w = (real_field && mval > 0) ? 2.0 : 1.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= mval
                metric = SHTnsKit._convention_metric(cfg, lval, mval)
                e_local += w * metric * abs2(Alm[il, jm])
            end
        end
    end
    e = Allreduce(e_local, +, communicator(Alm))
    return 0.5 * e
end

function SHTnsKit.energy_vector(cfg::SHTnsKit.SHTConfig,
                                Slm::PencilArray, Tlm::PencilArray;
                                real_field::Bool=true)
    _require_diagnostic_pair(cfg, Slm, Tlm, :spectral, ("Slm", "Tlm");
                             options=(real_field,))
    lloc = axes(Slm, 1); mloc = axes(Slm, 2)
    gl_l = collect(Int, globalindices(Slm, 1))
    gl_m = collect(Int, globalindices(Slm, 2))
    RT = promote_type(Float64, real(float(eltype(Slm))), real(float(eltype(Tlm))))
    e_local = zero(RT)
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval % cfg.mres == 0 || continue
        w = (real_field && mval > 0) ? 2.0 : 1.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval * (lval + 1)
                metric = SHTnsKit._convention_metric(cfg, lval, mval)
                e_local += w * metric * L2 *
                           (abs2(Slm[il, jm]) + abs2(Tlm[il, jm]))
            end
        end
    end
    e = Allreduce(e_local, +, communicator(Slm))
    return 0.5 * e
end

function SHTnsKit.enstrophy(cfg::SHTnsKit.SHTConfig, Tlm::PencilArray;
                            real_field::Bool=true)
    _require_diagnostic_array(cfg, Tlm, :spectral, "Tlm"; options=(real_field,))
    lloc = axes(Tlm, 1); mloc = axes(Tlm, 2)
    gl_l = collect(Int, globalindices(Tlm, 1))
    gl_m = collect(Int, globalindices(Tlm, 2))
    RT = promote_type(Float64, real(float(eltype(Tlm))))
    z_local = zero(RT)
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval % cfg.mres == 0 || continue
        w = (real_field && mval > 0) ? 2.0 : 1.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval * (lval + 1)
                metric = SHTnsKit._convention_metric(cfg, lval, mval)
                z_local += w * metric * (L2^2) * abs2(Tlm[il, jm])
            end
        end
    end
    z = Allreduce(z_local, +, communicator(Tlm))
    return 0.5 * z
end

function SHTnsKit.energy_scalar_l_spectrum(cfg::SHTnsKit.SHTConfig, Alm::PencilArray; real_field::Bool=true)
    _require_diagnostic_array(cfg, Alm, :spectral, "Alm"; options=(real_field,))
    lmax = cfg.lmax
    RT = real(float(eltype(Alm)))
    E = zeros(RT, lmax + 1)
    lloc = axes(Alm, 1); mloc = axes(Alm, 2)
    gl_l = collect(Int, globalindices(Alm, 1))
    gl_m = collect(Int, globalindices(Alm, 2))
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval % cfg.mres == 0 || continue
        wm = (real_field && mval > 0) ? 2.0 : 1.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= mval
                metric = SHTnsKit._convention_metric(cfg, lval, mval)
                E[lval + 1] += wm * metric * abs2(Alm[il, jm])
            end
        end
    end
    MPI.Allreduce!(E, +, communicator(Alm))
    return 0.5 * E
end

function SHTnsKit.energy_scalar_m_spectrum(cfg::SHTnsKit.SHTConfig, Alm::PencilArray; real_field::Bool=true)
    _require_diagnostic_array(cfg, Alm, :spectral, "Alm"; options=(real_field,))
    mmax = cfg.mmax
    RT = real(float(eltype(Alm)))
    E = zeros(RT, mmax + 1)
    mloc = axes(Alm, 2)
    gl_m = collect(Int, globalindices(Alm, 2))
    lloc = axes(Alm, 1)
    gl_l = collect(Int, globalindices(Alm, 1))
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval % cfg.mres == 0 || continue
        s = zero(RT)
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= mval
                metric = SHTnsKit._convention_metric(cfg, lval, mval)
                s += metric * abs2(Alm[il, jm])
            end
        end
        wm = (real_field && mval > 0) ? 2.0 : 1.0
        E[mval + 1] += wm * s
    end
    MPI.Allreduce!(E, +, communicator(Alm))
    return 0.5 * E
end

function SHTnsKit.energy_vector_l_spectrum(cfg::SHTnsKit.SHTConfig, Slm::PencilArray, Tlm::PencilArray; real_field::Bool=true)
    _require_diagnostic_pair(cfg, Slm, Tlm, :spectral, ("Slm", "Tlm");
                             options=(real_field,))
    lmax = cfg.lmax
    RT = real(float(promote_type(eltype(Slm), eltype(Tlm))))
    E = zeros(RT, lmax + 1)
    lloc = axes(Slm, 1); mloc = axes(Slm, 2)
    gl_l = collect(Int, globalindices(Slm, 1))
    gl_m = collect(Int, globalindices(Slm, 2))
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval % cfg.mres == 0 || continue
        wm = (real_field && mval > 0) ? 2.0 : 1.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval*(lval+1)
                metric = SHTnsKit._convention_metric(cfg, lval, mval)
                E[lval + 1] += wm * metric * L2 *
                               (abs2(Slm[il, jm]) + abs2(Tlm[il, jm]))
            end
        end
    end
    MPI.Allreduce!(E, +, communicator(Slm))
    return 0.5 * E
end

function SHTnsKit.energy_vector_m_spectrum(cfg::SHTnsKit.SHTConfig, Slm::PencilArray, Tlm::PencilArray; real_field::Bool=true)
    _require_diagnostic_pair(cfg, Slm, Tlm, :spectral, ("Slm", "Tlm");
                             options=(real_field,))
    mmax = cfg.mmax
    RT = real(float(promote_type(eltype(Slm), eltype(Tlm))))
    E = zeros(RT, mmax + 1)
    lloc = axes(Slm, 1); mloc = axes(Slm, 2)
    gl_l = collect(Int, globalindices(Slm, 1))
    gl_m = collect(Int, globalindices(Slm, 2))
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval % cfg.mres == 0 || continue
        s = zero(RT)
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval*(lval+1)
                metric = SHTnsKit._convention_metric(cfg, lval, mval)
                s += metric * L2 * (abs2(Slm[il, jm]) + abs2(Tlm[il, jm]))
            end
        end
        wm = (real_field && mval > 0) ? 2.0 : 1.0
        E[mval + 1] += wm * s
    end
    MPI.Allreduce!(E, +, communicator(Slm))
    return 0.5 * E
end

function SHTnsKit.enstrophy_l_spectrum(cfg::SHTnsKit.SHTConfig, Tlm::PencilArray; real_field::Bool=true)
    _require_diagnostic_array(cfg, Tlm, :spectral, "Tlm"; options=(real_field,))
    lmax = cfg.lmax
    RT = real(float(eltype(Tlm)))
    Z = zeros(RT, lmax + 1)
    lloc = axes(Tlm, 1); mloc = axes(Tlm, 2)
    gl_l = collect(Int, globalindices(Tlm, 1))
    gl_m = collect(Int, globalindices(Tlm, 2))
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval % cfg.mres == 0 || continue
        wm = (real_field && mval > 0) ? 2.0 : 1.0
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval*(lval+1)
                metric = SHTnsKit._convention_metric(cfg, lval, mval)
                Z[lval + 1] += wm * metric * (L2^2) * abs2(Tlm[il, jm])
            end
        end
    end
    MPI.Allreduce!(Z, +, communicator(Tlm))
    return 0.5 * Z
end

function SHTnsKit.enstrophy_m_spectrum(cfg::SHTnsKit.SHTConfig, Tlm::PencilArray; real_field::Bool=true)
    _require_diagnostic_array(cfg, Tlm, :spectral, "Tlm"; options=(real_field,))
    mmax = cfg.mmax
    RT = real(float(eltype(Tlm)))
    Z = zeros(RT, mmax + 1)
    lloc = axes(Tlm, 1); mloc = axes(Tlm, 2)
    gl_l = collect(Int, globalindices(Tlm, 1))
    gl_m = collect(Int, globalindices(Tlm, 2))
    @inbounds for (jj, jm) in enumerate(mloc)
        mval = gl_m[jj] - 1
        mval % cfg.mres == 0 || continue
        s = zero(RT)
        for (ii, il) in enumerate(lloc)
            lval = gl_l[ii] - 1
            if lval >= max(1, mval)
                L2 = lval*(lval+1)
                metric = SHTnsKit._convention_metric(cfg, lval, mval)
                s += metric * (L2^2) * abs2(Tlm[il, jm])
            end
        end
        wm = (real_field && mval > 0) ? 2.0 : 1.0
        Z[mval + 1] += wm * s
    end
    MPI.Allreduce!(Z, +, communicator(Tlm))
    return 0.5 * Z
end

function SHTnsKit.grid_energy_scalar(cfg::SHTnsKit.SHTConfig, fθφ::PencilArray)
    _require_diagnostic_array(cfg, fθφ, :spatial, "fθφ")
    θloc = axes(fθφ, 1)
    gl_θ = collect(Int, globalindices(fθφ, 1))
    φscale = 2π / cfg.nlon
    e_local = 0.0
    @inbounds for (ii, iθ) in enumerate(θloc)
        iglobθ = gl_θ[ii]
        wi = cfg.w[iglobθ]
        for j in axes(fθφ, 2)
            e_local += wi * abs2(fθφ[iθ, j])
        end
    end
    e = Allreduce(e_local, +, communicator(fθφ))
    return 0.5 * (φscale * e)
end

function SHTnsKit.grid_energy_vector(cfg::SHTnsKit.SHTConfig, Vtθφ::PencilArray, Vpθφ::PencilArray)
    _require_diagnostic_pair(cfg, Vtθφ, Vpθφ, :spatial, ("Vtθφ", "Vpθφ"))
    θloc = axes(Vtθφ, 1)
    gl_θ = collect(Int, globalindices(Vtθφ, 1))
    φscale = 2π / cfg.nlon
    e_local = 0.0
    @inbounds for (ii, iθ) in enumerate(θloc)
        iglobθ = gl_θ[ii]
        wi = cfg.w[iglobθ]
        for j in axes(Vtθφ, 2)
            e_local += wi * (abs2(Vtθφ[iθ, j]) + abs2(Vpθφ[iθ, j]))
        end
    end
    e = Allreduce(e_local, +, communicator(Vtθφ))
    return 0.5 * (φscale * e)
end

function SHTnsKit.grid_enstrophy(cfg::SHTnsKit.SHTConfig, ζθφ::PencilArray)
    _require_diagnostic_array(cfg, ζθφ, :spatial, "ζθφ")
    θloc = axes(ζθφ, 1)
    gl_θ = collect(Int, globalindices(ζθφ, 1))
    φscale = 2π / cfg.nlon
    z_local = 0.0
    @inbounds for (ii, iθ) in enumerate(θloc)
        iglobθ = gl_θ[ii]
        wi = cfg.w[iglobθ]
        for j in axes(ζθφ, 2)
            z_local += wi * abs2(ζθφ[iθ, j])
        end
    end
    z = Allreduce(z_local, +, communicator(ζθφ))
    return 0.5 * (φscale * z)
end
