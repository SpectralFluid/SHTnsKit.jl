#=
================================================================================
prettyprinting.jl - Pretty-printing of SHTnsKit objects
================================================================================

`show` methods that print a compact, tree-structured summary instead of the
default field dump. Two forms are provided for each public type:

- `show(io, x)`                  — one-line compact form (used by `print`/`string`)
- `show(io, MIME"text/plain", x)` — multi-line summary (REPL / `display`)

Arrays are never printed in full; only their extent or aggregate statistics
are shown.
================================================================================
=#

# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

_grid_name(grid_type::Symbol) =
    grid_type === :gauss          ? "Gauss-Legendre" :
    grid_type === :regular        ? "regular (Fejér)" :
    grid_type === :regular_poles  ? "regular (pole-inclusive)" :
    grid_type === :driscoll_healy ? "Driscoll-Healy" :
    string(grid_type)

_mem_string(bytes::Real) =
    bytes < 1024   ? "$(round(Int, bytes)) B" :
    bytes < 1024^2 ? "$(round(bytes / 1024; digits=1)) KiB" :
    bytes < 1024^3 ? "$(round(bytes / 1024^2; digits=1)) MiB" :
                     "$(round(bytes / 1024^3; digits=2)) GiB"

_θ_range(cfg) = "$(round(first(cfg.θ); sigdigits=4)) … $(round(last(cfg.θ); sigdigits=4))"
_φ_range(cfg) = "$(round(first(cfg.φ); sigdigits=4)) … $(round(last(cfg.φ); sigdigits=4))"

# ---------------------------------------------------------------------------
# SHTConfig
# ---------------------------------------------------------------------------

function Base.show(io::IO, cfg::SHTConfig)
    print(io, "SHTConfig($(_grid_name(cfg.grid_type)), lmax=$(cfg.lmax), mmax=$(cfg.mmax), $(cfg.nlat)×$(cfg.nlon))")
end

function Base.show(io::IO, ::MIME"text/plain", cfg::SHTConfig)
    name = _grid_name(cfg.grid_type)
    pole = cfg.south_pole_first ? "south-pole-first" : "north-pole-first"
    mode = is_on_the_fly(cfg) ? "on-the-fly" :
           "tables ($(_mem_string(estimate_table_memory(cfg))))"
    padding = is_padding_enabled(cfg) ?
              "enabled (nlat_padded = $(get_nlat_padded(cfg)))" : "disabled"

    println(io, "SHTConfig{$name} with lmax=$(cfg.lmax), mmax=$(cfg.mmax), mres=$(cfg.mres)")
    println(io, "├── grid: $(cfg.nlat) × $(cfg.nlon) ($pole)")
    println(io, "│   ├── θ: $(_θ_range(cfg))")
    println(io, "│   └── φ: $(_φ_range(cfg)), Δφ = $(round(cfg.cphi; sigdigits=4))")
    println(io, "├── quadrature: Σw = $(round(sum(cfg.w); digits=12))")
    println(io, "├── normalization: :$(cfg.norm), Condon-Shortley = $(cfg.cs_phase), real_norm = $(cfg.real_norm)")
    println(io, "├── Legendre mode: $mode")
    println(io, "├── spectral modes: nlm = $(cfg.nlm)")
    println(io, "├── φ scaling: :$(cfg.phi_scale)")
    println(io, "└── padding: $padding")
end

# ---------------------------------------------------------------------------
# SHTPlan
# ---------------------------------------------------------------------------

function Base.show(io::IO, plan::SHTPlan)
    cfg = plan.cfg
    print(io, "SHTPlan($(_grid_name(cfg.grid_type)), lmax=$(cfg.lmax), mmax=$(cfg.mmax), rfft=$(plan.use_rfft))")
end

function Base.show(io::IO, ::MIME"text/plain", plan::SHTPlan)
    cfg = plan.cfg
    fft_plans = plan.use_rfft ? "fft, ifft, rfft, irfft" : "fft, ifft"
    println(io, "SHTPlan{$(_grid_name(cfg.grid_type))} for lmax=$(cfg.lmax), mmax=$(cfg.mmax)")
    println(io, "├── rfft: $(plan.use_rfft)")
    println(io, "├── Legendre buffers: P, dPdx, dPdtheta, P_over_sinth (length = $(length(plan.P)))")
    if plan.use_rfft
        println(io, "├── Fourier buffers: complex $(size(plan.Fθk, 1)) × $(size(plan.Fθk, 2)), rfft $(size(plan.Fθk_r, 1)) × $(size(plan.Fθk_r, 2)) $(eltype(plan.Fθk))")
    else
        println(io, "├── Fourier buffer: $(size(plan.Fθk, 1)) × $(size(plan.Fθk, 2)) $(eltype(plan.Fθk))")
    end
    println(io, "└── FFT plans: $fft_plans")
end

# ---------------------------------------------------------------------------
# SHTRotation
# ---------------------------------------------------------------------------

function Base.show(io::IO, rot::SHTRotation)
    print(io, "SHTRotation($(rot.conv), lmax=$(rot.lmax), mmax=$(rot.mmax), α=$(round(rot.α; digits=3)), β=$(round(rot.β; digits=3)), γ=$(round(rot.γ; digits=3)))")
end

function Base.show(io::IO, ::MIME"text/plain", rot::SHTRotation)
    println(io, "SHTRotation{$(rot.conv)} for lmax=$(rot.lmax), mmax=$(rot.mmax)")
    println(io, "├── Euler angles (rad): α = $(rot.α), β = $(rot.β), γ = $(rot.γ)")
    println(io, "├── normalization: :$(rot.norm), Condon-Shortley = $(rot.cs_phase), real_norm = $(rot.real_norm)")
    println(io, "└── reverse_outer: $(rot.reverse_outer)")
end
