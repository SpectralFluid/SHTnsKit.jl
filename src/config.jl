#=
================================================================================
config.jl - SHTConfig Structure and Configuration Functions
================================================================================

This file defines the core SHTConfig struct that holds all parameters and
precomputed data for spherical harmonic transforms. Understanding this struct
is essential for debugging and optimizing SHT computations.

KEY CONCEPTS
------------
1. Grid Setup:
   - nlat: Number of Gauss-Legendre points in latitude (θ direction)
   - nlon: Number of equispaced points in longitude (φ direction)
   - x[i] = cos(θ[i]): Gauss-Legendre nodes, ordered from +1 (north) to -1 (south)
   - w[i]: Gauss-Legendre weights, sum(w) ≈ 2.0

2. Spectral Setup:
   - lmax: Maximum degree l (total wavenumber), determines angular resolution
   - mmax: Maximum order m (azimuthal wavenumber), usually equals lmax
   - mres: M-resolution (usually 1), allows skipping m values

3. Accuracy Requirements:
   - nlat ≥ lmax + 1: Required for exact Gauss-Legendre integration
   - nlon ≥ 2*mmax + 1: Required for Nyquist sampling of m modes

4. Normalization:
   - Nlm[l+1, m+1]: Normalization factor for Y_l^m
   - cphi = 2π/nlon: Longitude spacing factor for FFT

COMMON DEBUGGING CHECKS
-----------------------
```julia
# Verify Gauss weights sum correctly
@assert abs(sum(cfg.w) - 2.0) < 1e-10 "Gauss weights should sum to 2"

# Check grid ordering (north to south by default)
@assert cfg.x[1] > cfg.x[end] "x should decrease (north to south)"
@assert cfg.x[1] ≈ cos(cfg.θ[1]) "x = cos(θ)"

# Verify dimensions match
@assert size(cfg.Nlm) == (cfg.lmax+1, cfg.mmax+1)
@assert length(cfg.x) == cfg.nlat
@assert length(cfg.φ) == cfg.nlon
```

PERFORMANCE MODES
-----------------
1. On-the-fly (cfg.on_the_fly = true):
   - Computes Legendre polynomials during each transform
   - Lower memory usage, slightly slower
   - Good for large lmax or memory-constrained systems

2. Table lookup (cfg.use_plm_tables = true):
   - Precomputes all P_l^m values in cfg.plm_tables
   - Higher memory usage, faster transforms
   - Good for repeated transforms at same configuration

CREATING CONFIGURATIONS
----------------------
- create_gauss_config(lmax, nlat): Standard Gauss-Legendre grid
- create_regular_config(lmax, nlat): Equispaced latitude grid
- create_gauss_config_spf(lmax, nlat): South-pole-first ordering

================================================================================
=#

"""
    _default_nlon(lmax::Int) -> Int

Compute a default nlon that satisfies the Nyquist requirement (≥ 2*lmax+1) and
is efficient for FFT. FFTW performs best with sizes whose prime factors are small
(2, 3, 5). We round up 2*lmax+1 to the next even number, which guarantees at
least one factor of 2 and avoids the worst-case odd/prime sizes.
"""
function _default_nlon(lmax::Int)
    min_nlon = 2 * lmax + 1
    # Round up to the next even number for better FFT performance
    return min_nlon + (min_nlon & 1)
end

"""
Configuration for Spherical Harmonic Transforms.

This struct contains all parameters and precomputed data needed for efficient
spherical harmonic transforms. It encapsulates both the mathematical parameters
(degrees, grid sizes) and computational optimizations (precomputed tables).

Fields
- `lmax, mmax`: maximum degree and order for spherical harmonics
- `mres`: resolution parameter for m-modes (typically 1)
- `nlat, nlon`: grid size in latitude (Gauss–Legendre) and longitude (equiangular)
- `grid_type`: :gauss, :regular, or :regular_poles
- `θ, φ`: polar and azimuth angle arrays for the computational grid
- `x, w`: Gauss–Legendre nodes and weights for numerical integration (x = cos(θ))
- `Nlm`: normalization factors matrix indexed as (l+1, m+1)
- `cphi`: longitude step size (2π / nlon) for FFT operations

# Thread Safety

**WARNING:** SHTConfig objects should be treated as **immutable during parallel operations**.
Do not modify config fields (`use_plm_tables`, `plm_tables`, `on_the_fly`, etc.) while
transforms are executing in `@threads` blocks. Create a new config or complete all
parallel work before modifying. Violating this can cause race conditions.

Safe pattern:
```julia
cfg = create_gauss_config(32, 34)
prepare_plm_tables!(cfg)  # OK: modify before parallel work

# Now cfg is "frozen" for parallel use
@threads for i in 1:100
    alm = analysis(cfg, fields[i])  # OK: read-only access
end

# After parallel work, modifications are safe again
set_on_the_fly!(cfg)
```
"""
# ============================================================================
# ROLE-TYPED SUB-STRUCTS
# ============================================================================
# SHTConfig previously held ~35 flat fields mixing grid geometry,
# normalization state, precomputed tables, and lazy scratch buffers. These
# sub-structs group them by concern so reading the main struct no longer
# requires scanning 40+ lines. All old flat-field access (cfg.x, cfg.Nlm,
# cfg.NP_tables, cfg.use_plm_tables, ...) is preserved via getproperty /
# setproperty! forwarding in the next section — no call-site changes needed.

"""Grid geometry: coordinates, quadrature nodes/weights, azimuthal spacing."""
mutable struct SHTGrid
    θ::Vector{Float64}
    φ::Vector{Float64}
    x::Vector{Float64}
    w::Vector{Float64}
    st::Vector{Float64}
    cphi::Float64
end

"""Normalization + phase convention. `scale_matrix` is lazily built; it is held
in a `Ref` so the lazy fill mutates the Ref's contents rather than rebinding a
struct field — this keeps the field compatible with an immutable config."""
Base.@kwdef mutable struct SHTNorm
    Nlm::Matrix{Float64}
    norm::Symbol
    cs_phase::Bool
    real_norm::Bool
    robert_form::Bool
    scale_matrix::Base.RefValue{Matrix{Float64}} = Ref(Matrix{Float64}(undef, 0, 0))
end

"""Precomputed Legendre tables (raw and Nlm-fused)."""
Base.@kwdef mutable struct SHTTables
    enabled::Bool = false
    plm::Vector{Matrix{Float64}} = Matrix{Float64}[]
    dplm::Vector{Matrix{Float64}} = Matrix{Float64}[]
    NP::Vector{Matrix{Float64}} = Matrix{Float64}[]
    NdP::Vector{Matrix{Float64}} = Matrix{Float64}[]
end

"""Lazy per-cfg scratch buffers for OTF Legendre eval + m-order cache."""
Base.@kwdef mutable struct SHTScratch
    otf_P::Vector{Vector{Float64}} = Vector{Float64}[]
    otf_dP::Vector{Vector{Float64}} = Vector{Float64}[]
    otf_Ps::Vector{Vector{Float64}} = Vector{Float64}[]
    otf_Pb::Vector{Vector{Float64}} = Vector{Float64}[]   # extended P̄ buffer (length lmax+2) for norm dθ kernels
    m_order::Vector{Int} = Int[]
end

mutable struct SHTConfig
    # Core spherical harmonic parameters
    lmax::Int
    mmax::Int
    mres::Int
    nlat::Int
    nlon::Int
    grid_type::Symbol

    # SHTns-compatible indexing helpers
    nlm::Int
    li::Vector{Int}
    mi::Vector{Int}
    nspat::Int

    phi_scale::Symbol
    on_the_fly::Bool

    # Batch transform configuration
    howmany::Int
    spec_dist::Int

    # Data ordering
    south_pole_first::Bool

    # Memory padding
    allow_padding::Bool
    nlat_padded::Int
    spat_dist::Int

    # Grouped sub-struct fields (under private-looking names so the old flat
    # field names are free for property forwarding).
    _grid::SHTGrid
    _norm::SHTNorm
    _tables::SHTTables
    _scratch::SHTScratch
end

"""
    SHTConfig(; lmax, mmax, mres, nlat, nlon, θ, φ, x, w, Nlm, cphi, st,
              norm, cs_phase, real_norm, robert_form,
              nlm, li, mi, nspat,
              grid_type=:gauss, phi_scale=:auto, on_the_fly=false,
              use_plm_tables=false, plm_tables=[], dplm_tables=[],
              NP_tables=[], NdP_tables=[], norm_scale_matrix=Matrix{Float64}(undef,0,0),
              _otf_scratch_P=[], _otf_scratch_dP=[], _otf_scratch_Ps=[], _otf_scratch_Pb=[], _m_order=[],
              howmany=1, spec_dist=0, south_pole_first=false,
              allow_padding=false, nlat_padded=0, spat_dist=0) -> SHTConfig

Flat-kwargs constructor that packages the grid/normalization/tables/scratch
kwargs into their respective sub-structs. Keeps the call-site syntax used by
`create_gauss_config`, `create_regular_config`, `copy(::SHTConfig)`, etc.
identical to the pre-restructure behaviour.
"""
function SHTConfig(;
    lmax::Integer, mmax::Integer, mres::Integer, nlat::Integer, nlon::Integer,
    θ::AbstractVector, φ::AbstractVector, x::AbstractVector, w::AbstractVector,
    Nlm::AbstractMatrix, cphi::Real, st::AbstractVector,
    nlm::Integer, li::AbstractVector{<:Integer}, mi::AbstractVector{<:Integer}, nspat::Integer,
    norm::Symbol, cs_phase::Bool, real_norm::Bool, robert_form::Bool,
    grid_type::Symbol = :gauss,
    phi_scale::Symbol = :auto,
    on_the_fly::Bool = false,
    use_plm_tables::Bool = false,
    plm_tables::AbstractVector = Matrix{Float64}[],
    dplm_tables::AbstractVector = Matrix{Float64}[],
    NP_tables::AbstractVector = Matrix{Float64}[],
    NdP_tables::AbstractVector = Matrix{Float64}[],
    norm_scale_matrix::AbstractMatrix = Matrix{Float64}(undef, 0, 0),
    _otf_scratch_P::AbstractVector = Vector{Float64}[],
    _otf_scratch_dP::AbstractVector = Vector{Float64}[],
    _otf_scratch_Ps::AbstractVector = Vector{Float64}[],
    _otf_scratch_Pb::AbstractVector = Vector{Float64}[],
    _m_order::AbstractVector{<:Integer} = Int[],
    howmany::Integer = 1,
    spec_dist::Integer = 0,
    south_pole_first::Bool = false,
    allow_padding::Bool = false,
    nlat_padded::Integer = 0,
    spat_dist::Integer = 0,
)
    # Concretize vectors/matrices so SHTGrid/SHTNorm/SHTTables fields always
    # carry the exact declared types. Lets callers pass ranges, views, etc.
    grid = SHTGrid(collect(Float64, θ), collect(Float64, φ),
                   collect(Float64, x), collect(Float64, w),
                   collect(Float64, st), Float64(cphi))
    norm_group = SHTNorm(; Nlm=convert(Matrix{Float64}, Nlm),
                          norm=norm, cs_phase=cs_phase,
                          real_norm=real_norm, robert_form=robert_form,
                          scale_matrix=Ref(convert(Matrix{Float64}, norm_scale_matrix)))
    plm = convert(Vector{Matrix{Float64}}, collect(plm_tables))
    dplm = convert(Vector{Matrix{Float64}}, collect(dplm_tables))
    NP = NP_tables === plm_tables ? plm : convert(Vector{Matrix{Float64}}, collect(NP_tables))
    NdP = NdP_tables === dplm_tables ? dplm : convert(Vector{Matrix{Float64}}, collect(NdP_tables))
    tables = SHTTables(; enabled=use_plm_tables, plm, dplm, NP, NdP)
    scratch = SHTScratch(;
        otf_P=convert(Vector{Vector{Float64}}, collect(_otf_scratch_P)),
        otf_dP=convert(Vector{Vector{Float64}}, collect(_otf_scratch_dP)),
        otf_Ps=convert(Vector{Vector{Float64}}, collect(_otf_scratch_Ps)),
        otf_Pb=convert(Vector{Vector{Float64}}, collect(_otf_scratch_Pb)),
        m_order=collect(Int, _m_order))
    return SHTConfig(Int(lmax), Int(mmax), Int(mres), Int(nlat), Int(nlon), grid_type,
                     Int(nlm), collect(Int, li), collect(Int, mi), Int(nspat),
                     phi_scale, on_the_fly,
                     Int(howmany), Int(spec_dist), south_pole_first,
                     allow_padding, Int(nlat_padded), Int(spat_dist),
                     grid, norm_group, tables, scratch)
end

# ============================================================================
# PROPERTY FORWARDING (flat field names -> grouped sub-structs)
# ============================================================================

@inline function Base.getproperty(cfg::SHTConfig, name::Symbol)
    # ----- grid fields -----
    if name === :θ || name === :φ || name === :x || name === :w || name === :st || name === :cphi
        return getproperty(getfield(cfg, :_grid), name)
    # ----- legacy grid aliases (advertised by propertynames): wlat→w, ct→x, sintheta→st -----
    elseif name === :wlat
        return getproperty(getfield(cfg, :_grid), :w)
    elseif name === :ct
        return getproperty(getfield(cfg, :_grid), :x)
    elseif name === :sintheta
        return getproperty(getfield(cfg, :_grid), :st)
    # ----- normalization fields -----
    elseif name === :Nlm
        return getfield(cfg, :_norm).Nlm
    elseif name === :norm
        return getfield(cfg, :_norm).norm
    elseif name === :cs_phase
        return getfield(cfg, :_norm).cs_phase
    elseif name === :real_norm
        return getfield(cfg, :_norm).real_norm
    elseif name === :robert_form
        return getfield(cfg, :_norm).robert_form
    elseif name === :norm_scale_matrix
        return getfield(cfg, :_norm).scale_matrix[]  # deref Ref so callers still see a Matrix
    # ----- tables fields -----
    elseif name === :use_plm_tables
        return getfield(cfg, :_tables).enabled
    elseif name === :plm_tables
        return getfield(cfg, :_tables).plm
    elseif name === :dplm_tables
        return getfield(cfg, :_tables).dplm
    elseif name === :NP_tables
        return getfield(cfg, :_tables).NP
    elseif name === :NdP_tables
        return getfield(cfg, :_tables).NdP
    # ----- scratch fields -----
    elseif name === :_otf_scratch_P
        return getfield(cfg, :_scratch).otf_P
    elseif name === :_otf_scratch_dP
        return getfield(cfg, :_scratch).otf_dP
    elseif name === :_otf_scratch_Ps
        return getfield(cfg, :_scratch).otf_Ps
    elseif name === :_otf_scratch_Pb
        return getfield(cfg, :_scratch).otf_Pb
    elseif name === :_m_order
        return getfield(cfg, :_scratch).m_order
    # ----- sub-struct direct access (new API) -----
    elseif name === :grid
        return getfield(cfg, :_grid)
    # Note: `cfg.norm` returns the Symbol (legacy). Use `cfg.normalization`
    # for the sub-struct to avoid name collision.
    elseif name === :normalization
        return getfield(cfg, :_norm)
    elseif name === :tables
        return getfield(cfg, :_tables)
    elseif name === :scratch
        return getfield(cfg, :_scratch)
    else
        return getfield(cfg, name)
    end
end

function Base.setproperty!(cfg::SHTConfig, name::Symbol, val)
    if name === :θ || name === :φ || name === :x || name === :w || name === :st || name === :cphi
        return setproperty!(getfield(cfg, :_grid), name, val)
    elseif name === :wlat
        return setproperty!(getfield(cfg, :_grid), :w, val)
    elseif name === :ct
        return setproperty!(getfield(cfg, :_grid), :x, val)
    elseif name === :sintheta
        return setproperty!(getfield(cfg, :_grid), :st, val)
    elseif name === :Nlm
        return setfield!(getfield(cfg, :_norm), :Nlm, val)
    elseif name === :norm
        setfield!(getfield(cfg, :_norm), :norm, val)
        # invalidate cached norm-scale matrix: it is keyed on size only, so a
        # norm change on a same-size config would otherwise reuse stale factors.
        getfield(cfg, :_norm).scale_matrix[] = Matrix{Float64}(undef, 0, 0)
        return val
    elseif name === :cs_phase
        setfield!(getfield(cfg, :_norm), :cs_phase, val)
        getfield(cfg, :_norm).scale_matrix[] = Matrix{Float64}(undef, 0, 0)  # invalidate cached scale
        return val
    elseif name === :real_norm
        setfield!(getfield(cfg, :_norm), :real_norm, val)
        getfield(cfg, :_norm).scale_matrix[] = Matrix{Float64}(undef, 0, 0)
        return val
    elseif name === :robert_form
        return setfield!(getfield(cfg, :_norm), :robert_form, val)
    elseif name === :norm_scale_matrix
        getfield(cfg, :_norm).scale_matrix[] = val  # mutate Ref contents, not rebind the field
        return val
    elseif name === :use_plm_tables
        return setfield!(getfield(cfg, :_tables), :enabled, val)
    elseif name === :plm_tables
        return setfield!(getfield(cfg, :_tables), :plm, val)
    elseif name === :dplm_tables
        return setfield!(getfield(cfg, :_tables), :dplm, val)
    elseif name === :NP_tables
        return setfield!(getfield(cfg, :_tables), :NP, val)
    elseif name === :NdP_tables
        return setfield!(getfield(cfg, :_tables), :NdP, val)
    elseif name === :_otf_scratch_P
        return setfield!(getfield(cfg, :_scratch), :otf_P, val)
    elseif name === :_otf_scratch_dP
        return setfield!(getfield(cfg, :_scratch), :otf_dP, val)
    elseif name === :_otf_scratch_Ps
        return setfield!(getfield(cfg, :_scratch), :otf_Ps, val)
    elseif name === :_otf_scratch_Pb
        return setfield!(getfield(cfg, :_scratch), :otf_Pb, val)
    elseif name === :_m_order
        return setfield!(getfield(cfg, :_scratch), :m_order, val)
    else
        return setfield!(cfg, name, val)
    end
end

function Base.propertynames(::SHTConfig, private::Bool=false)
    return (fieldnames(SHTConfig)...,
            :θ, :φ, :x, :w, :st, :cphi,
            :Nlm, :norm, :cs_phase, :real_norm, :robert_form, :norm_scale_matrix,
            :use_plm_tables, :plm_tables, :dplm_tables, :NP_tables, :NdP_tables,
            :_otf_scratch_P, :_otf_scratch_dP, :_otf_scratch_Ps, :_otf_scratch_Pb, :_m_order,
            :grid, :normalization, :tables, :scratch,
            :wlat, :ct, :sintheta)
end

# ============================================================================
# LOGICAL GROUPING ACCESSORS
# ============================================================================
# SHTConfig accumulates ~35 fields across grid, normalization, tables, and
# scratch concerns. A full role-typed restructure (struct SHTGrid; ...;
# mutable struct SHTConfig; grid::SHTGrid; end) would break every call site.
# These accessors give the same conceptual grouping for documentation and new
# code without breaking the flat-field API everything else depends on.

"""
    grid_view(cfg::SHTConfig) -> NamedTuple

Grid-related fields: grid type, θ/φ coordinates, Gauss nodes/weights, sin(θ),
azimuthal spacing. Use for read-only access when writing new code that wants
an explicit "grid" abstraction.
"""
@inline grid_view(cfg::SHTConfig) = (
    type = cfg.grid_type,
    θ = cfg.θ, φ = cfg.φ, x = cfg.x, w = cfg.w, st = cfg.st,
    nlat = cfg.nlat, nlon = cfg.nlon, cphi = cfg.cphi,
)

"""
    norm_view(cfg::SHTConfig) -> NamedTuple

Normalization / phase convention fields.
"""
@inline norm_view(cfg::SHTConfig) = (
    Nlm = cfg.Nlm, norm = cfg.norm, cs_phase = cfg.cs_phase,
    real_norm = cfg.real_norm, robert_form = cfg.robert_form,
    phi_scale = cfg.phi_scale,
)

"""
    tables_view(cfg::SHTConfig) -> NamedTuple

Precomputed table caches (plm, dplm, Nlm-fused NP/NdP) plus the toggle flag.
"""
@inline tables_view(cfg::SHTConfig) = (
    enabled = cfg.use_plm_tables,
    plm = cfg.plm_tables, dplm = cfg.dplm_tables,
    NP = cfg.NP_tables, NdP = cfg.NdP_tables,
)

"""
    scratch_view(cfg::SHTConfig) -> NamedTuple

Lazily-built scratch buffers: per-thread on-the-fly Legendre storage,
norm-scale matrix, balanced m ordering.
"""
@inline scratch_view(cfg::SHTConfig) = (
    otf_P = cfg._otf_scratch_P, otf_dP = cfg._otf_scratch_dP, otf_Ps = cfg._otf_scratch_Ps,
    otf_Pb = cfg._otf_scratch_Pb,
    norm_scale = cfg.norm_scale_matrix, m_order = cfg._m_order,
)

"""
    has_fused_scalar_tables(cfg) -> Bool

Returns true iff the scalar `NP_tables` cache is populated AND the user hasn't
disabled table lookup. Collapses the `cfg.use_plm_tables && length(cfg.NP_tables) == mmax+1`
idiom into one call so the two fields can't silently disagree.
"""
@inline function has_fused_scalar_tables(cfg::SHTConfig)
    return cfg.use_plm_tables && length(cfg.NP_tables) == cfg.mmax + 1
end

"""
    has_fused_vector_tables(cfg) -> Bool

Returns true iff both `NP_tables` and `NdP_tables` are populated (vector/sphtor
kernels need the derivative table too).
"""
@inline function has_fused_vector_tables(cfg::SHTConfig)
    return cfg.use_plm_tables &&
           length(cfg.NP_tables) == cfg.mmax + 1 &&
           length(cfg.NdP_tables) == cfg.mmax + 1
end

"""
    create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, nlon::Int=_default_nlon(lmax)) -> SHTConfig

Create a Gauss–Legendre based SHT configuration. Constraints:
- `nlat ≥ lmax+1` for exactness up to `lmax` in θ integration.
- `nlon ≥ 2*mmax+1` to resolve azimuthal orders up to `mmax`.
"""
function create_gauss_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1, nlon::Int=_default_nlon(lmax), norm::Symbol=:orthonormal, cs_phase::Bool=true, real_norm::Bool=false, robert_form::Bool=false)
    # Validate input parameters to ensure mathematical accuracy requirements
    lmax ≥ 0 || throw(ArgumentError("lmax must be ≥ 0"))
    mmax ≥ 0 || throw(ArgumentError("mmax must be ≥ 0"))
    mmax ≤ lmax || throw(ArgumentError("mmax must be ≤ lmax"))
    mres ≥ 1 || throw(ArgumentError("mres must be ≥ 1"))
    nlat ≥ lmax + 1 || throw(ArgumentError("nlat must be ≥ lmax+1 for Gauss–Legendre accuracy"))
    nlon ≥ (2*mmax + 1) || throw(ArgumentError("nlon must be ≥ 2*mmax+1"))

    # Build the computational grid using Gauss-Legendre quadrature
    θ, φ, x, w = thetaphi_from_nodes(nlat, nlon)
    reverse!(θ); reverse!(x); reverse!(w)  # Store latitude in north-pole-first order.

    # Compute normalization factors for spherical harmonics
    Nlm = Nlm_table(lmax, mmax)  # currently orthonormal; future: adjust per norm/cs_phase
    
    # Calculate indexing helpers for efficient (l,m) mode access
    nlm = nlm_calc(lmax, mmax, mres)              # Total number of spectral modes
    li, mi = build_li_mi(lmax, mmax, mres)        # Degree and order index arrays
    
    # Precompute sin(θ) for vector transforms
    st = sin.(θ)

    # Construct and return the complete configuration
    return SHTConfig(; lmax, mmax, mres, nlat, nlon, grid_type=:gauss, θ, φ, x, w, Nlm,
                     cphi = 2π / nlon, nlm, li, mi, nspat = nlat*nlon,
                     st, norm, cs_phase, real_norm, robert_form, phi_scale=:dft)
end

"""
    create_gauss_fly_config(lmax::Int, nlat::Int; kwargs...) -> SHTConfig

Create a Gauss–Legendre configuration with on-the-fly Legendre polynomial computation.

This mode computes Legendre polynomials during each transform instead of using
precomputed tables. Benefits:
- **Lower memory usage**: No storage for P_l^m tables (saves O(lmax² × nlat) memory)
- **Better for large lmax**: When lmax is large, tables become memory-intensive
- **Good cache behavior**: On modern CPUs, on-the-fly computation can be faster
  due to better cache utilization

Trade-offs:
- May be slower for repeated transforms with small lmax
- No speedup from table precomputation

This mirrors the `sht_gauss_fly` mode in SHTns C library.

# Arguments
Same as `create_gauss_config`.

# Example
```julia
# For large lmax, on-the-fly mode saves significant memory
cfg = create_gauss_fly_config(1024, 1026)

# Performs transform computing Legendre polynomials on-the-fly
alm = analysis(cfg, field)
```

See also: [`create_gauss_config`](@ref), [`set_on_the_fly!`](@ref), [`set_use_tables!`](@ref)
"""
function create_gauss_fly_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1,
                                  nlon::Int=_default_nlon(lmax), norm::Symbol=:orthonormal,
                                  cs_phase::Bool=true, real_norm::Bool=false,
                                  robert_form::Bool=false)
    # Create base Gauss config
    cfg = create_gauss_config(lmax, nlat; mmax=mmax, mres=mres, nlon=nlon,
                              norm=norm, cs_phase=cs_phase, real_norm=real_norm,
                              robert_form=robert_form)

    # Force on-the-fly mode
    cfg.on_the_fly = true
    cfg.use_plm_tables = false
    cfg.plm_tables = Matrix{Float64}[]
    cfg.dplm_tables = Matrix{Float64}[]
    cfg.NP_tables = Matrix{Float64}[]
    cfg.NdP_tables = Matrix{Float64}[]

    return cfg
end

"""
    set_on_the_fly!(cfg::SHTConfig)

Enable on-the-fly Legendre polynomial computation mode. This disables precomputed
tables and computes P_l^m(x) during each transform.

Benefits:
- Lower memory usage (no table storage)
- Better for large lmax values
- Can be faster on modern CPUs due to cache effects

See also: [`set_use_tables!`](@ref), [`create_gauss_fly_config`](@ref)
"""
function set_on_the_fly!(cfg::SHTConfig)
    cfg.on_the_fly = true
    cfg.use_plm_tables = false
    cfg.plm_tables = Matrix{Float64}[]
    cfg.dplm_tables = Matrix{Float64}[]
    cfg.NP_tables = Matrix{Float64}[]
    cfg.NdP_tables = Matrix{Float64}[]
    return cfg
end

"""
    set_use_tables!(cfg::SHTConfig)

Enable precomputed Legendre polynomial tables. This precomputes P_l^m(x_i) for
all grid points and stores them for reuse.

Benefits:
- Faster for repeated transforms with same grid
- No redundant computation of Legendre polynomials

Trade-offs:
- Higher memory usage: O(lmax × mmax × nlat)
- Initial table computation overhead

See also: [`set_on_the_fly!`](@ref), [`prepare_plm_tables!`](@ref)
"""
function set_use_tables!(cfg::SHTConfig)
    cfg.on_the_fly = false
    if !cfg.use_plm_tables || isempty(cfg.plm_tables)
        prepare_plm_tables!(cfg)
    end
    return cfg
end

"""
    is_on_the_fly(cfg::SHTConfig) -> Bool

Check if configuration is set to on-the-fly Legendre computation mode.
"""
is_on_the_fly(cfg::SHTConfig) = cfg.on_the_fly || !cfg.use_plm_tables

"""
    estimate_table_memory(lmax::Int, mmax::Int, nlat::Int) -> Int

Estimate memory usage (in bytes) for precomputed Legendre polynomial tables.

Returns the approximate memory needed for both P_l^m and dP_l^m/dx tables.
"""
function estimate_table_memory(lmax::Int, mmax::Int, nlat::Int)
    # Each table has (lmax+1) × nlat Float64 values per m
    # We have (mmax+1) such tables, for both P and dP
    bytes_per_table = (lmax + 1) * nlat * sizeof(Float64)
    num_tables = (mmax + 1) * 2  # P and dP tables
    return bytes_per_table * num_tables
end

"""
    estimate_table_memory(cfg::SHTConfig) -> Int

Estimate memory usage for Legendre tables based on configuration.
"""
estimate_table_memory(cfg::SHTConfig) = estimate_table_memory(cfg.lmax, cfg.mmax, cfg.nlat)

# ============================================================================
# SOUTH POLE FIRST MODE FUNCTIONS
# ============================================================================

"""
    set_south_pole_first!(cfg::SHTConfig)

Enable south-pole-first latitude ordering. In this mode, latitude data starts
at the south pole (θ=π) instead of the default north pole (θ=0).

This reverses the internal grid arrays (θ, x, w, st) and recalculates
any precomputed Legendre tables if necessary.

# Example
```julia
cfg = create_gauss_config(32, 34)
set_south_pole_first!(cfg)  # Data now starts at south pole
```

See also: [`set_north_pole_first!`](@ref), [`is_south_pole_first`](@ref)
"""
function set_south_pole_first!(cfg::SHTConfig)
    if cfg.south_pole_first
        return cfg  # Already in south-pole-first mode
    end

    # Reverse latitude-dependent arrays in-place (avoids allocations)
    # Note: cfg.wlat/ct/sintheta are property aliases for w/x/st, so reversing
    # the canonical fields automatically updates the aliases.
    reverse!(cfg.θ)
    reverse!(cfg.w)
    reverse!(cfg.x)
    reverse!(cfg.st)

    # Mark as south-pole-first
    cfg.south_pole_first = true

    # Recompute Legendre tables if they were precomputed
    if cfg.use_plm_tables && !isempty(cfg.plm_tables)
        prepare_plm_tables!(cfg)
    end

    return cfg
end

"""
    set_north_pole_first!(cfg::SHTConfig)

Enable north-pole-first latitude ordering (the default). In this mode, latitude
data starts at the north pole (θ=0).

This reverses the internal grid arrays if the configuration was previously in
south-pole-first mode, and recalculates any precomputed Legendre tables.

# Example
```julia
cfg = create_gauss_config(32, 34)
set_south_pole_first!(cfg)  # Data starts at south pole
set_north_pole_first!(cfg)  # Back to default (north pole first)
```

See also: [`set_south_pole_first!`](@ref), [`is_south_pole_first`](@ref)
"""
function set_north_pole_first!(cfg::SHTConfig)
    if !cfg.south_pole_first
        return cfg  # Already in north-pole-first mode
    end

    # Reverse latitude-dependent arrays in-place (avoids allocations)
    reverse!(cfg.θ)
    reverse!(cfg.w)
    reverse!(cfg.x)
    reverse!(cfg.st)

    # Mark as north-pole-first
    cfg.south_pole_first = false

    # Recompute Legendre tables if they were precomputed
    if cfg.use_plm_tables && !isempty(cfg.plm_tables)
        prepare_plm_tables!(cfg)
    end

    return cfg
end

"""
    is_south_pole_first(cfg::SHTConfig) -> Bool

Check if the configuration uses south-pole-first latitude ordering.

Returns `true` if latitude data starts at the south pole (θ=π),
`false` if it starts at the north pole (θ=0, the default).
"""
is_south_pole_first(cfg::SHTConfig) = cfg.south_pole_first

"""
    create_gauss_config_spf(lmax::Int, nlat::Int; kwargs...) -> SHTConfig

Create a Gauss-Legendre configuration with south-pole-first latitude ordering.

This is equivalent to calling `create_gauss_config` followed by `set_south_pole_first!`.
All keyword arguments are passed to `create_gauss_config`.

# Example
```julia
# Create a south-pole-first configuration
cfg = create_gauss_config_spf(32, 34)
is_south_pole_first(cfg)  # true
```

See also: [`create_gauss_config`](@ref), [`set_south_pole_first!`](@ref)
"""
function create_gauss_config_spf(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1,
                                  nlon::Int=_default_nlon(lmax), norm::Symbol=:orthonormal,
                                  cs_phase::Bool=true, real_norm::Bool=false,
                                  robert_form::Bool=false)
    cfg = create_gauss_config(lmax, nlat; mmax=mmax, mres=mres, nlon=nlon,
                              norm=norm, cs_phase=cs_phase, real_norm=real_norm,
                              robert_form=robert_form)
    set_south_pole_first!(cfg)
    return cfg
end

# ============================================================================
# MEMORY PADDING FUNCTIONS
# ============================================================================

"""
    compute_optimal_padding(nlat::Int, nlon::Int; sizeof_real::Int=8) -> Int

Compute the optimal padding to add between latitude lines to avoid cache bank conflicts.
This mimics the SHTns C library's padding strategy.

# Arguments
- `nlat`: Number of latitude points
- `nlon`: Number of longitude points (phi)
- `sizeof_real`: Size of real number in bytes (default 8 for Float64)

# Returns
The number of extra elements to add per latitude line (padding amount).

# Details
The padding strategy aims to:
1. Align memory on 32-byte or 64-byte boundaries
2. Avoid 8KB cache bank conflicts
3. Avoid L2 cache bank/channel conflicts (especially important for AMD GPUs)

This is most beneficial for:
- Large transforms where cache effects matter
- Multi-threaded transforms
- GPU computing
"""
function compute_optimal_padding(nlat::Int, nlon::Int; sizeof_real::Int=8)
    if nlon <= 1
        return 0  # No padding needed for 1D or scalar
    end

    stride_bytes = nlat * sizeof_real
    min_pad_bytes = 8 * sizeof_real  # Minimum 8 elements padding

    # Compute alignment-based padding
    # Check if full data fits in L2 cache (~8MB typical)
    # Use 64-byte alignment for L2-resident data, 32-byte otherwise
    estimated_total = nlon * (stride_bytes + min_pad_bytes)
    alignment = estimated_total < 8192 * 1024 ? 64 : 32

    # Compute padding needed for alignment
    alignment_remainder = stride_bytes % alignment
    alignment_pad = alignment_remainder == 0 ? 0 : (alignment - alignment_remainder)

    # Use the larger of minimum padding and alignment padding
    pad_bytes = max(min_pad_bytes, alignment_pad)

    # Avoid multiples of 8KB (cache bank conflicts)
    if (stride_bytes + pad_bytes) % 8192 == 0
        pad_bytes += 64
    end

    # Avoid 128-byte alignment issues (L2 cache bank conflicts)
    if (stride_bytes + pad_bytes) % 128 == 0
        pad_bytes += alignment == 64 ? 64 : 32
    end

    # Convert to number of elements
    return div(pad_bytes, sizeof_real)
end

"""
    set_allow_padding!(cfg::SHTConfig; auto_compute::Bool=true)

Enable memory padding for cache optimization. When enabled, spatial arrays
can have extra padding between latitude lines to avoid cache bank conflicts.

This can improve performance by 1-50% depending on the problem size and hardware,
especially for multi-threaded and GPU transforms.

# Arguments
- `cfg`: SHTConfig to modify
- `auto_compute`: If true, automatically compute optimal padding. If false,
  you must set `cfg.nlat_padded` manually.

# Example
```julia
cfg = create_gauss_config(64, 66)
set_allow_padding!(cfg)
println("Padded nlat: \$(cfg.nlat_padded)")  # May be > 66
```

See also: [`disable_padding!`](@ref), [`is_padding_enabled`](@ref), [`get_nlat_padded`](@ref)
"""
function set_allow_padding!(cfg::SHTConfig; auto_compute::Bool=true)
    cfg.allow_padding = true

    if auto_compute
        pad = compute_optimal_padding(cfg.nlat, cfg.nlon)
        cfg.nlat_padded = cfg.nlat + pad
    elseif cfg.nlat_padded == 0
        cfg.nlat_padded = cfg.nlat
    end

    # Compute spatial distance for batch transforms
    cfg.spat_dist = cfg.nlat_padded * cfg.nlon

    return cfg
end

"""
    disable_padding!(cfg::SHTConfig)

Disable memory padding. Spatial arrays will use contiguous memory without
extra padding between latitude lines.

See also: [`set_allow_padding!`](@ref), [`is_padding_enabled`](@ref)
"""
function disable_padding!(cfg::SHTConfig)
    cfg.allow_padding = false
    cfg.nlat_padded = 0
    cfg.spat_dist = 0
    return cfg
end

"""
    is_padding_enabled(cfg::SHTConfig) -> Bool

Check if memory padding is enabled for the configuration.
"""
is_padding_enabled(cfg::SHTConfig) = cfg.allow_padding

"""
    get_nlat_padded(cfg::SHTConfig) -> Int

Get the padded latitude dimension. Returns `cfg.nlat` if padding is disabled,
or `cfg.nlat_padded` if padding is enabled.

This is the stride between successive phi (longitude) values in padded spatial arrays.
"""
function get_nlat_padded(cfg::SHTConfig)
    if cfg.allow_padding && cfg.nlat_padded > 0
        return cfg.nlat_padded
    else
        return cfg.nlat
    end
end

"""
    get_spat_dist(cfg::SHTConfig) -> Int

Get the distance between spatial arrays in batch mode. Returns `cfg.nspat` if
padding is disabled, or the padded spatial distance if enabled.
"""
function get_spat_dist(cfg::SHTConfig)
    if cfg.allow_padding && cfg.spat_dist > 0
        return cfg.spat_dist
    else
        return cfg.nspat
    end
end

"""
    allocate_padded_spatial(cfg::SHTConfig, T::Type=Float64) -> Array{T}

Allocate a spatial array with proper padding if enabled.

Returns a 2D array of size (nlat_padded, nlon) where nlat_padded >= nlat.
The first nlat rows contain the actual data; remaining rows are padding.

# Example
```julia
cfg = create_gauss_config(64, 66)
set_allow_padding!(cfg)
field = allocate_padded_spatial(cfg)
# field has size (nlat_padded, nlon), use field[1:cfg.nlat, :] for data
```
"""
function allocate_padded_spatial(cfg::SHTConfig, T::Type=Float64)
    nlat_p = get_nlat_padded(cfg)
    return zeros(T, nlat_p, cfg.nlon)
end

"""
    allocate_padded_spatial_batch(cfg::SHTConfig, nfields::Int, T::Type=Float64) -> Array{T}

Allocate spatial arrays for batch transforms with proper padding if enabled.

Returns a 3D array of size (nlat_padded, nlon, nfields).
"""
function allocate_padded_spatial_batch(cfg::SHTConfig, nfields::Int, T::Type=Float64)
    nlat_p = get_nlat_padded(cfg)
    return zeros(T, nlat_p, cfg.nlon, nfields)
end

"""
    copy_to_padded!(dest::AbstractMatrix, src::AbstractMatrix, cfg::SHTConfig)

Copy data from a contiguous spatial array to a padded spatial array.
"""
function copy_to_padded!(dest::AbstractMatrix, src::AbstractMatrix, cfg::SHTConfig)
    size(src, 1) == cfg.nlat || throw(DimensionMismatch("src first dim must be nlat=$(cfg.nlat)"))
    size(src, 2) == cfg.nlon || throw(DimensionMismatch("src second dim must be nlon=$(cfg.nlon)"))
    size(dest, 1) >= cfg.nlat || throw(DimensionMismatch("dest first dim must be ≥ nlat=$(cfg.nlat)"))
    size(dest, 2) == cfg.nlon || throw(DimensionMismatch("dest second dim must be nlon=$(cfg.nlon)"))

    dest[1:cfg.nlat, :] .= src
    return dest
end

"""
    copy_from_padded!(dest::AbstractMatrix, src::AbstractMatrix, cfg::SHTConfig)

Copy data from a padded spatial array to a contiguous spatial array.
"""
function copy_from_padded!(dest::AbstractMatrix, src::AbstractMatrix, cfg::SHTConfig)
    size(dest, 1) == cfg.nlat || throw(DimensionMismatch("dest first dim must be nlat=$(cfg.nlat)"))
    size(dest, 2) == cfg.nlon || throw(DimensionMismatch("dest second dim must be nlon=$(cfg.nlon)"))
    size(src, 1) >= cfg.nlat || throw(DimensionMismatch("src first dim must be ≥ nlat=$(cfg.nlat)"))
    size(src, 2) == cfg.nlon || throw(DimensionMismatch("src second dim must be nlon=$(cfg.nlon)"))

    dest .= src[1:cfg.nlat, :]
    return dest
end

"""
    estimate_padding_overhead(cfg::SHTConfig) -> Float64

Estimate the memory overhead from padding as a percentage.

Returns 0.0 if padding is disabled.
"""
function estimate_padding_overhead(cfg::SHTConfig)
    if !cfg.allow_padding || cfg.nlat_padded <= cfg.nlat
        return 0.0
    end
    return 100.0 * (cfg.nlat_padded - cfg.nlat) / cfg.nlat
end

"""
    create_regular_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1,
                          nlon::Int=_default_nlon(lmax), norm::Symbol=:orthonormal,
                          cs_phase::Bool=true, real_norm::Bool=false,
                          robert_form::Bool=false, include_poles::Bool=false,
                          precompute_plm::Bool=true, use_dh_weights::Bool=false) -> SHTConfig

Create an equiangular (regular) grid configuration. Regular grids use simple
`θ = (i+0.5)π/nlat` nodes by default; set `include_poles=true` to place nodes
directly on the poles. By default associated Legendre tables are precomputed,
which mirrors SHTns' regular-grid behaviour and improves performance.

# Driscoll-Healy Quadrature

Set `use_dh_weights=true` to use Driscoll-Healy quadrature for exact spherical
harmonic transforms. This requires:
- `include_poles=true`
- `nlat = 2*(lmax+1)` for exactness up to degree lmax
- `nlat` must be even

The DH grid uses θ = πj/n for j=0,...,n-1 (includes north pole, excludes south pole)
and provides exact quadrature via specially computed weights.

Reference: Driscoll & Healy (1994), "Computing Fourier transforms and convolutions
on the 2-sphere", Adv. Appl. Math., 15, 202-250.
"""
function create_regular_config(lmax::Int, nlat::Int; mmax::Int=lmax, mres::Int=1,
                               nlon::Int=_default_nlon(lmax), norm::Symbol=:orthonormal,
                               cs_phase::Bool=true, real_norm::Bool=false,
                               robert_form::Bool=false, include_poles::Bool=false,
                               precompute_plm::Bool=true, use_dh_weights::Bool=false)
    lmax ≥ 0 || throw(ArgumentError("lmax must be ≥ 0"))
    mmax ≥ 0 || throw(ArgumentError("mmax must be ≥ 0"))
    mmax ≤ lmax || throw(ArgumentError("mmax must be ≤ lmax"))
    mres ≥ 1 || throw(ArgumentError("mres must be ≥ 1"))
    nlon ≥ (2*mmax + 1) || throw(ArgumentError("nlon must be ≥ 2*mmax+1"))
    use_dh_weights && !include_poles && throw(ArgumentError(
        "Driscoll-Healy weights require include_poles=true",
    ))
    # Regular grids benefit from a slight oversampling in latitude for accuracy
    min_nlat = include_poles ? (lmax + 1) : (lmax + 2)
    nlat ≥ min_nlat || throw(ArgumentError("nlat must be ≥ $(min_nlat) for regular grids"))

    θ = zeros(Float64, nlat)
    w = zeros(Float64, nlat)
    x = zeros(Float64, nlat)
    if include_poles
        # Check if we should use DH weights
        if use_dh_weights
            # Validate DH requirements
            iseven(nlat) || throw(ArgumentError("DH weights require even nlat"))
            nlat == 2*(lmax + 1) || @warn "DH weights are exact when nlat=2*(lmax+1)=$(2*(lmax+1)), got nlat=$nlat"

            # Use Driscoll-Healy grid: θ = π*j/n for j=0,...,n-1
            # This includes north pole (j=0, θ=0) but not south pole (j=n would give θ=π)
            # The last point is at θ = π*(n-1)/n, just before the south pole
            w = driscoll_healy_weights(nlat)
            for i in 0:(nlat-1)
                θi = π * i / nlat
                θ[i+1] = θi
                x[i+1] = cos(θi)
            end
        else
            # Use trapezoidal rule with both poles
            # Poles (θ=0 and θ=π) get half-weight per the trapezoidal rule
            # A pole-inclusive grid needs at least the two poles: `nlat == 1` makes
            # `h = π/0 = Inf` and `θ[1] = 0*Inf = NaN`, and since the generic
            # `nlat ≥ lmax+1` check passes for lmax=0 the whole config would come
            # back all-NaN and every later transform would return NaN with no
            # exception. Fail here instead, where the cause is visible.
            nlat >= 2 || throw(ArgumentError("pole-inclusive grids need nlat ≥ 2 (got nlat=$nlat); " *
                                             "use grid_type=:regular for a single-latitude grid"))
            h = π / (nlat - 1)
            for i in 0:(nlat-1)
                θi = i * h
                θ[i+1] = θi
                x[i+1] = cos(θi)
                w[i+1] = h * sin(θi)
            end
            # Apply trapezoidal endpoint correction: first and last points get half-weight.
            # At exact poles sin(θ)=0 so w is already 0, but for near-pole points and
            # correctness of the quadrature rule, enforce the half-weight convention.
            w[1] *= 0.5
            w[nlat] *= 0.5
        end
    else
        for i in 0:(nlat-1)
            θi = (i + 0.5) * (π / nlat)
            θ[i+1] = θi
            w[i+1] = (π / nlat) * sin(θi)
            x[i+1] = cos(θi)
        end
    end
    φ = (2π / nlon) .* collect(0:(nlon-1))

    # Compute normalization factors and indexing helpers
    Nlm = Nlm_table(lmax, mmax)
    nlm = nlm_calc(lmax, mmax, mres)
    li, mi = build_li_mi(lmax, mmax, mres)
    st = sin.(θ)

    # Determine grid type and phi_scale based on configuration
    grid_type = if use_dh_weights
        :driscoll_healy
    elseif include_poles
        :regular_poles
    else
        :regular
    end

    # Regular/equiangular grids use the same DFT scaling as Gauss grids
    phi_scale = :dft

    cfg = SHTConfig(; lmax, mmax, mres, nlat, nlon, grid_type,
                    θ, φ, x, w, Nlm,
                    cphi = 2π / nlon, nlm, li, mi, nspat = nlat*nlon,
                    st, norm, cs_phase, real_norm, robert_form, phi_scale)

    if precompute_plm
        prepare_plm_tables!(cfg)
    end
    return cfg
end

"""
    create_config(lmax::Int; mmax=lmax, mres=1, nlat=lmax+2, nlon=max(2*lmax+1,4),
                   norm::Symbol=:orthonormal, cs_phase::Bool=true,
                   real_norm::Bool=false, robert_form::Bool=false,
                   grid_type::Symbol=:gauss) -> SHTConfig

Compatibility wrapper for configuration creation used in some docs/snippets.
Supports Gauss–Legendre (`grid_type = :gauss`), regular equiangular
(`grid_type = :regular` or `:regular_poles`), and Driscoll-Healy
(`grid_type = :driscoll_healy`) grids, forwarding to the appropriate creator.
`nlat`/`nlon` defaults are adjusted to satisfy accuracy constraints for the
chosen grid.
"""
function create_config(lmax::Int; mmax::Int=lmax, mres::Int=1, nlat::Int=lmax+2,
                       nlon::Int=_default_nlon(lmax), norm::Symbol=:orthonormal,
                       cs_phase::Bool=true, real_norm::Bool=false,
                       robert_form::Bool=false, grid_type::Symbol=:gauss,
                       include_poles::Bool=false, precompute_plm::Bool=grid_type != :gauss)
    # Make args robust to underspecified values from older docs/snippets
    pole_grid = grid_type === :regular_poles || grid_type === :driscoll_healy
    min_lat = grid_type === :gauss ? (lmax + 1) : (pole_grid ? (lmax + 1) : (lmax + 2))
    include_poles_eff = include_poles || pole_grid
    nlat_eff = max(nlat, min_lat)               # Regular grids benefit from a small oversampling
    nlon_eff = max(nlon, 2*mmax + 1)            # Azimuthal resolution requires ≥ 2*mmax+1
    if grid_type == :gauss
        return create_gauss_config(lmax, nlat_eff; mmax=mmax, mres=mres, nlon=nlon_eff,
                                   norm=norm, cs_phase=cs_phase,
                                   real_norm=real_norm, robert_form=robert_form)
    elseif grid_type === :regular || grid_type === :regular_poles || grid_type === :driscoll_healy
        return create_regular_config(lmax, nlat_eff; mmax=mmax, mres=mres, nlon=nlon_eff,
                                     norm=norm, cs_phase=cs_phase, real_norm=real_norm,
                                     robert_form=robert_form, include_poles=include_poles_eff,
                                     precompute_plm=precompute_plm,
                                     use_dh_weights=grid_type === :driscoll_healy)
    else
        throw(ArgumentError(
            "unsupported grid_type=$(grid_type); choose :gauss, :regular, " *
            ":regular_poles, or :driscoll_healy",
        ))
    end
end

"""
    create_config(::Type{T}, lmax::Int, nlat::Int, mres::Int=1; kwargs...) -> SHTConfig

Compatibility method that ignores the element type `T` and calls `create_config`.
Provided to match older example signatures like `create_config(Float64, 20, 22, 1)`.
"""
function create_config(::Type{T}, lmax::Int, nlat::Int, mres::Int=1; kwargs...) where {T}
    return create_config(lmax; nlat=nlat, mres=mres, kwargs...)
end

"""
    prepare_plm_tables!(cfg::SHTConfig)

Precompute associated Legendre tables P_l^m(x_i) for all i and m, stored as
`cfg.plm_tables[m+1][l+1, i]`. Enables faster scalar transforms on regular grids.
"""
function prepare_plm_tables!(cfg::SHTConfig)
    lmax, mmax = cfg.lmax, cfg.mmax
    nlat = cfg.nlat

    # The derivative tables use the convention NdP = -(dP̄/dθ)/sinθ, i.e. they divide
    # by sinθ so the kernels can multiply it back (`dθY = -sinθ·NdP`). Value tables
    # (plm/NP) have no such division and are correct everywhere, so table use stays
    # enabled on pole-inclusive regular/DH grids. Only the derivative division must
    # be guarded: at an EXACT pole node (x = ±1 ⇒ sinθ = 0) `-dP̄/dθ / 0` is NaN.
    # Setting the pole entry to 0 is kernel-consistent — the kernel's ×sinθ makes
    # `dθY = -0·NdP = 0` at the pole regardless of NdP — and avoids NaN propagation
    # (`-0·NaN = NaN`). (Gauss nodes never hit poles, so this only affects DH/regular
    # pole grids, and only their vector-transform derivative rows.)

    # Allocate storage for Legendre polynomial tables
    # Each m-order gets its own matrix: (degree+1) × (latitude points)
    tables = [zeros(Float64, lmax + 1, nlat) for _ in 0:mmax]    # P_l^m values
    dtables = [zeros(Float64, lmax + 1, nlat) for _ in 0:mmax]  # dP_l^m/dx derivatives
    
    # Working arrays for computing one row at a time
    P = Vector{Float64}(undef, lmax + 1)      # P̄_l^m(x) normalized values
    dPdtheta = Vector{Float64}(undef, lmax + 1)  # dP̄_l^m/dθ normalized θ-derivatives

    # Compute tables for each azimuthal order m using bounded normalized recurrence
    # plm_tables[m+1][l+1, i]  = P̄_l^m(x_i)   (orthonormal; no overflow at high lmax)
    # dplm_tables[m+1][l+1, i] = -(dP̄_l^m/dθ)/sinθ_i  (matches NdP convention used by batch kernels)
    # Gauss nodes never hit poles, so sinθ > 0 for all i.
    for m in 0:mmax
        tbl = tables[m+1]    # Access using 1-based indexing
        dtbl = dtables[m+1]

        # Compute normalized Legendre polynomials at each latitude point
        for i in 1:nlat
            s_i = sqrt(max(0.0, 1.0 - cfg.x[i]^2))
            Plm_norm_and_dPdtheta_row!(P, dPdtheta, cfg.x[i], lmax, m)

            # Store normalized values — no Nlm multiply needed (P̄ already = Nlm * rawP)
            @inbounds @views tbl[:, i] .= P         # P̄_l^m(x_i) for l=0:lmax
            inv_s = s_i == 0 ? 0.0 : 1.0 / s_i  # pole guard: avoid 0/0→NaN (see header note)
            @inbounds for l in 0:lmax
                dtbl[l+1, i] = -dPdtheta[l+1] * inv_s  # -(dP̄/dθ)/sinθ (matches NdP convention)
            end
        end
    end
    
    # The "pre-fused Nlm" NP/NdP tables are, by construction, the tables already
    # built above: `Plm_norm_row!` returns P̄ = Nlm·rawP, which is exactly what
    # `tbl` holds, and the NdP convention -(dP̄/dθ)/sinθ is exactly what `dtbl`
    # holds. Rebuilding them from the same recurrences produced bit-for-bit
    # identical arrays at double the build time and double the resident memory,
    # while `estimate_table_memory` (which counts (mmax+1)*2 tables) reported
    # half the true figure — so a job sized by the estimator allocated twice its
    # budget and was OOM-killed. Alias instead.
    #
    # INVARIANT: these four must stay value-identical. If the NP or NdP
    # convention ever diverges from the plm/dplm one, build them separately again
    # rather than aliasing — and fix `estimate_table_memory` to match.
    #
    # Deliberately NOT copies: these are build-once, read-only caches (nothing
    # mutates a table after `prepare_plm_tables!` returns).
    NP_tables = tables
    NdP_tables = dtables

    # Enable table usage and store in configuration
    cfg.plm_tables = tables
    cfg.dplm_tables = dtables
    cfg.NP_tables = NP_tables
    cfg.NdP_tables = NdP_tables
    cfg.use_plm_tables = true
    return cfg
end

"""
    enable_plm_tables!(cfg::SHTConfig)

Alias for `prepare_plm_tables!`.
"""
enable_plm_tables!(cfg::SHTConfig) = prepare_plm_tables!(cfg)

"""
    disable_plm_tables!(cfg::SHTConfig)

Disable use of precomputed Legendre tables.
"""
function disable_plm_tables!(cfg::SHTConfig)
    cfg.use_plm_tables = false
    cfg.plm_tables = Matrix{Float64}[]
    cfg.dplm_tables = Matrix{Float64}[]
    cfg.NP_tables = Matrix{Float64}[]
    cfg.NdP_tables = Matrix{Float64}[]
    return cfg
end

"""
    destroy_config(cfg::SHTConfig)

No-op placeholder for API symmetry with libraries that require explicit teardown.
"""
destroy_config(::SHTConfig) = nothing

"""
    Base.copy(cfg::SHTConfig) -> SHTConfig

Create an independent copy of an SHTConfig with all arrays deep-copied.

Since SHTConfig is a mutable struct with shared Vector fields, a simple
assignment `cfg2 = cfg` creates an alias — mutating one modifies the other.
Use `copy(cfg)` to get a fully independent copy that can be safely modified
(e.g., `set_south_pole_first!`) without affecting the original.
"""
function Base.copy(cfg::SHTConfig)
    plm_tables = [copy(t) for t in cfg.plm_tables]
    dplm_tables = [copy(t) for t in cfg.dplm_tables]
    NP_tables = cfg.NP_tables === cfg.plm_tables ? plm_tables : [copy(t) for t in cfg.NP_tables]
    NdP_tables = cfg.NdP_tables === cfg.dplm_tables ? dplm_tables : [copy(t) for t in cfg.NdP_tables]

    return SHTConfig(;
        lmax = cfg.lmax, mmax = cfg.mmax, mres = cfg.mres,
        nlat = cfg.nlat, nlon = cfg.nlon, grid_type = cfg.grid_type,
        θ = copy(cfg.θ), φ = copy(cfg.φ), x = copy(cfg.x), w = copy(cfg.w),
        Nlm = copy(cfg.Nlm), cphi = cfg.cphi,
        nlm = cfg.nlm, li = copy(cfg.li), mi = copy(cfg.mi), nspat = cfg.nspat,
        st = copy(cfg.st),
        norm = cfg.norm, cs_phase = cfg.cs_phase, real_norm = cfg.real_norm,
        robert_form = cfg.robert_form, phi_scale = cfg.phi_scale,
        use_plm_tables = cfg.use_plm_tables, on_the_fly = cfg.on_the_fly,
        plm_tables = plm_tables,
        dplm_tables = dplm_tables,
        NP_tables = NP_tables,
        NdP_tables = NdP_tables,
        howmany = cfg.howmany, spec_dist = cfg.spec_dist,
        south_pole_first = cfg.south_pole_first,
        allow_padding = cfg.allow_padding,
        nlat_padded = cfg.nlat_padded,
        spat_dist = cfg.spat_dist,
    )
end
