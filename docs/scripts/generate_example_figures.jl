module ExampleFigures

using CairoMakie
using SHTnsKit

export generate_example_figures, example_figure_names

const PALETTE = (
    primary    = "#1E40AF",   # deep blue
    secondary  = "#7C3AED",   # violet
    accent     = "#0891B2",   # teal
    warm       = "#DC2626",   # red
    highlight  = "#F59E0B",   # amber
    muted      = "#64748B",   # slate
    bg         = "#FFFFFF",
    grid       = "#E2E8F0",   # faint slate
    text       = "#1E293B",   # near-black
    bar_grad   = ["#1E40AF", "#3B82F6", "#60A5FA"],
)

function _docs_theme()
    Theme(
        fontsize = 14,
        Axis = (
            xlabelsize     = 13,
            ylabelsize     = 13,
            xticklabelsize = 11,
            yticklabelsize = 11,
            titlesize      = 15,
            titlefont      = :bold,
            titlealign     = :left,
            xgridvisible   = false,
            ygridvisible   = false,
            topspinevisible   = false,
            rightspinevisible = false,
            spinewidth     = 1.2,
            xtickwidth     = 1.0,
            ytickwidth     = 1.0,
        ),
        Colorbar = (
            labelsize    = 11,
            ticklabelsize = 10,
            width        = 12,
            spinewidth   = 0.8,
        ),
    )
end

"""Symmetric color limits so diverging colormaps center on zero."""
function _symmetric_lim(field)
    m = maximum(abs, field)
    m == 0 && (m = 1.0)
    return (-m, m)
end

_extrema(field, symmetric) =
    symmetric ? _symmetric_lim(field) : extrema(field)

"""Cell edges from cell centers for CairoMakie heatmap."""
function _cell_edges(centers)
    n = length(centers)
    edges = Vector{eltype(centers)}(undef, n + 1)
    edges[1] = centers[1] - (centers[2] - centers[1]) / 2
    for i in 2:n
        edges[i] = (centers[i-1] + centers[i]) / 2
    end
    edges[n+1] = centers[n] + (centers[n] - centers[n-1]) / 2
    return edges
end

"""Add a 2-D field heatmap panel."""
function _field_axis(fig, pos, cfg, field; title, colorbar_label,
                     colormap=:viridis, symmetric=false)
    ax = Axis(
        fig[pos...];
        xlabel = "longitude φ (rad)",
        ylabel = "colatitude θ (rad)",
        title  = title,
        yreversed = true,
    )
    hm = heatmap!(
        ax, _cell_edges(cfg.φ), _cell_edges(cfg.θ), permutedims(field);
        colormap   = colormap,
        colorrange = _extrema(field, symmetric),
        rasterize  = true,
    )
    Colorbar(
        fig[pos[1], pos[2] + 1], hm;
        vertical = true,
        height   = Relative(0.85),
        label    = colorbar_label,
    )
    return ax, hm
end


# ============================================================================
# Example 1: Scalar roundtrip
# ============================================================================

function scalar_roundtrip_figure()
    cfg = create_gauss_config(16, 18; nlon=33)
    temperature = [
        273.15 + 30 * (1 - cfg.x[i]^2)
        for i in 1:cfg.nlat, _ in 1:cfg.nlon
    ]

    fig = Figure(size=(720, 360), backgroundcolor=:transparent)
    ax, hm = _field_axis(
        fig, (1, 1), cfg, temperature;
        title          = "Band-limited temperature field",
        colorbar_label = "temperature (K)",
        colormap       = :inferno,
    )
    return fig
end


# ============================================================================
# Example 2: Power spectrum
# ============================================================================

function power_spectrum_figure()
    cfg = create_gauss_config(32, 34; nlon=65)
    coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    coefficients[3, 1] = 2.0
    coefficients[7, 4] = 0.5 - 0.1im
    power = energy_scalar_l_spectrum(cfg, analysis(cfg, synthesis(cfg, coefficients)))
    degrees = collect(0:cfg.lmax)

    fig = Figure(size=(720, 400), backgroundcolor=:transparent)
    ax = Axis(
        fig[1, 1];
        xlabel = "spherical-harmonic degree l",
        ylabel = "spectral energy",
        title  = "Energy by spherical-harmonic degree",
        xticks = 0:2:cfg.lmax,
        ygridvisible = true,
        ygridcolor   = PALETTE.grid,
        ygridwidth   = 0.6,
        ygridstyle   = :dash,
    )

    colors = [p > 1e-20 ? PALETTE.primary : PALETTE.grid for p in power]
    barplot!(ax, degrees, power; color=colors, gap=0.15, strokewidth=0.5,
             strokecolor=PALETTE.primary)

    text!(ax, 2, power[3]; text="l = 2", fontsize=11, align=(:center, :bottom),
          offset=(0, 6), color=PALETTE.warm)
    text!(ax, 6, power[7]; text="l = 6", fontsize=11, align=(:center, :bottom),
          offset=(0, 6), color=PALETTE.warm)

    ylims!(ax, 0, nothing)
    return fig
end


# ============================================================================
# Example 3: Vector decomposition
# ============================================================================

function vector_decomposition_figure()
    cfg = create_gauss_config(64, 66; nlon=129)
    Vθ = zeros(cfg.nlat, cfg.nlon)
    Vφ = zeros(cfg.nlat, cfg.nlon)
    for i in 1:cfg.nlat, j in 1:cfg.nlon
        θ, φ = cfg.θ[i], cfg.φ[j]
        Vθ[i, j] = 5 * cos(θ) * cos(φ)
        Vφ[i, j] = -5 * sin(φ) + 20 * sin(θ)
    end

    fig = Figure(size=(1080, 380), backgroundcolor=:transparent)

    _field_axis(fig, (1, 1), cfg, Vθ;
                title="Meridional component Vθ", colorbar_label="Vθ",
                colormap=:RdBu, symmetric=true)
    _field_axis(fig, (1, 3), cfg, Vφ;
                title="Zonal component Vφ", colorbar_label="Vφ",
                colormap=:RdBu, symmetric=true)

    colgap!(fig.layout, 2, 30)
    return fig
end


# ============================================================================
# Example 4: Stream function
# ============================================================================

function stream_function_figure()
    cfg = create_gauss_config(24, 26; nlon=49)
    vorticity_coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    vorticity_coefficients[5, 3] = 1.0 - 0.25im
    stream_coefficients = similar(vorticity_coefficients)
    fill!(stream_coefficients, 0)
    for l in 1:cfg.lmax, m in 0:min(l, cfg.mmax)
        stream_coefficients[l + 1, m + 1] =
            -vorticity_coefficients[l + 1, m + 1] / (l * (l + 1))
    end
    stream_function = synthesis(cfg, stream_coefficients)
    Vθ, Vφ = synthesis_sphtor(cfg, zero(stream_coefficients), stream_coefficients)

    fig = Figure(size=(1080, 380), backgroundcolor=:transparent)

    _field_axis(fig, (1, 1), cfg, stream_function;
                title="Stream function ψ", colorbar_label="ψ",
                colormap=:PRGn, symmetric=true)

    ax = Axis(
        fig[1, 3];
        xlabel    = "longitude φ (rad)",
        ylabel    = "colatitude θ (rad)",
        title     = "Tangential velocity (Vθ, Vφ)",
        yreversed = true,
    )

    bg_hm = heatmap!(
        ax, _cell_edges(cfg.φ), _cell_edges(cfg.θ),
        permutedims(hypot.(Vθ, Vφ));
        colormap  = Makie.Reverse(:Greys),
        colorrange = (0, maximum(hypot.(Vθ, Vφ))),
        rasterize = true,
        alpha     = 0.25,
    )

    stride = 2
    qθ = cfg.θ[1:stride:end]
    qφ = cfg.φ[1:stride:end]
    Vθq = Vθ[1:stride:end, 1:stride:end]
    Vφq = Vφ[1:stride:end, 1:stride:end]
    xs = vec([qφ[j] for _ in eachindex(qθ), j in eachindex(qφ)])
    ys = vec([qθ[i] for i in eachindex(qθ), _ in eachindex(qφ)])
    us = vec(Vφq)
    vs = vec(Vθq)
    speed = hypot.(us, vs)
    max_speed = maximum(speed)
    max_speed == 0 && (max_speed = 1.0)
    scale = 0.12 / max_speed

    arrows2d!(ax, xs, ys, us .* scale, vs .* scale;
              color=speed, colormap=:plasma,
              tipwidth=4, tiplength=6,
              shaftwidth=1.2)

    Colorbar(fig[1, 4], bg_hm; label="|V|", height=Relative(0.85))

    colgap!(fig.layout, 2, 30)
    return fig
end


# ============================================================================
# Example 5: Rotation
# ============================================================================

function rotation_figure()
    cfg = create_gauss_config(32, 34; nlon=65)
    input = zeros(ComplexF64, cfg.nlm)
    input[LM_index(cfg.lmax, cfg.mres, 3, 2) + 1] = 1.0

    rotation = SHTRotation(cfg.lmax, cfg.mmax)
    shtns_rotation_set_angles_ZYZ(rotation, π / 4, π / 6, π / 8)
    rotated = similar(input)
    shtns_rotation_apply_real(rotation, input, rotated)

    original = reshape(synthesis_packed(cfg, input), cfg.nlat, cfg.nlon)
    rotated_field = reshape(synthesis_packed(cfg, rotated), cfg.nlat, cfg.nlon)

    clim = _symmetric_lim(vcat(vec(original), vec(rotated_field)))

    fig = Figure(size=(1080, 380), backgroundcolor=:transparent)

    ax1 = Axis(fig[1, 1]; xlabel="longitude φ (rad)", ylabel="colatitude θ (rad)",
               title="Original field (l=3, m=2)", yreversed=true)
    hm1 = heatmap!(ax1, _cell_edges(cfg.φ), _cell_edges(cfg.θ), permutedims(original);
                    colormap=:BrBG, colorrange=clim, rasterize=true)
    Colorbar(fig[1, 2], hm1; label="value", height=Relative(0.85))

    ax2 = Axis(fig[1, 3]; xlabel="longitude φ (rad)", ylabel="colatitude θ (rad)",
               title="After ZYZ rotation (π/4, π/6, π/8)", yreversed=true)
    hm2 = heatmap!(ax2, _cell_edges(cfg.φ), _cell_edges(cfg.θ), permutedims(rotated_field);
                    colormap=:BrBG, colorrange=clim, rasterize=true)
    Colorbar(fig[1, 4], hm2; label="value", height=Relative(0.85))

    colgap!(fig.layout, 2, 30)
    return fig
end


# ============================================================================
# SVG metadata injection
# ============================================================================

"""Inject accessibility title/description into a CairoMakie-generated SVG."""
function _add_svg_metadata!(path::AbstractString, title::AbstractString, description::AbstractString)
    svg = read(path, String)
    svg_range = findfirst("<svg", svg)
    svg_range === nothing && error("CairoMakie did not produce an SVG root element")
    tag_end = findnext('>', svg, first(svg_range))
    tag_end === nothing && error("CairoMakie produced an incomplete SVG root element")

    opening_tag = String(SubString(svg, firstindex(svg), tag_end))
    opening_tag = replace(
        opening_tag,
        "<svg" => "<svg role=\"img\" aria-labelledby=\"example-figure-title example-figure-description\"";
        count=1,
    )
    metadata = """

<title id="example-figure-title">$title</title>
<desc id="example-figure-description">$description</desc>"""
    remainder_start = nextind(svg, tag_end)
    write(path, string(opening_tag, metadata, SubString(svg, remainder_start)))
    return path
end


# ============================================================================
# Figure registry
# ============================================================================

const _FIGURES = (
    (
        filename    = "example-scalar-roundtrip.svg",
        build       = scalar_roundtrip_figure,
        title       = "SHTnsKit scalar roundtrip example",
        description = "Heatmap of a band-limited temperature pattern on a Gauss–Legendre grid, warmest at the equator.",
    ),
    (
        filename    = "example-power-spectrum.svg",
        build       = power_spectrum_figure,
        title       = "SHTnsKit power-spectrum example",
        description = "Bar chart of spectral energy by spherical-harmonic degree with peaks at degrees two and six.",
    ),
    (
        filename    = "example-vector-decomposition.svg",
        build       = vector_decomposition_figure,
        title       = "SHTnsKit vector-decomposition example",
        description = "Side-by-side diverging heatmaps of meridional and zonal tangential vector-field components.",
    ),
    (
        filename    = "example-stream-function.svg",
        build       = stream_function_figure,
        title       = "SHTnsKit stream-function example",
        description = "Stream function heatmap alongside a quiver plot of its tangential velocity field.",
    ),
    (
        filename    = "example-rotation.svg",
        build       = rotation_figure,
        title       = "SHTnsKit rotation example",
        description = "A scalar field before and after a ZYZ Euler rotation, sharing a common color scale.",
    ),
)

example_figure_names() = Tuple(f.filename for f in _FIGURES)

"""Generate every example figure into `output_dir` and return the written paths."""
function generate_example_figures(output_dir::AbstractString)
    written = String[]
    with_theme(_docs_theme()) do
        for figure in _FIGURES
            path = joinpath(output_dir, figure.filename)
            save(path, figure.build())
            _add_svg_metadata!(path, figure.title, figure.description)
            push!(written, path)
        end
    end
    return written
end

end # module ExampleFigures
