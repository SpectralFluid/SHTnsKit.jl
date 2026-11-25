using Documenter
using SHTnsKit

println("Building SHTnsKit.jl documentation...")

#####
##### Documentation configuration
#####

# HTML format configuration
repository = "SpectralFluid/SHTnsKit.jl"
format = Documenter.HTML(
    # Keep relative figure paths identical in the preview and on GitHub Pages.
    prettyurls = true,
    canonical = "https://spectralfluid.github.io/SHTnsKit.jl/stable/",
    assets = [
        "assets/custom.css",
        "assets/extra.js",
    ],
    analytics = "",
    collapselevel = 2,
    sidebar_sitename = true,
    edit_link = "main",
    repolink = "https://github.com/$repository",
    size_threshold = 200 * 1024^2,  # 200 MiB
    size_threshold_warn = 10 * 1024^2   # 10 MiB warning
)

# Documentation pages structure
pages = Any[
    "Home" => "index.md",
    "Getting Started" => Any[
        "Installation" => "installation.md",
        "Quick Start" => "quickstart.md"
    ],
    "User Guide" => Any[
        "Grid Types" => "grids.md",
        "Examples Gallery" => "examples/index.md",
        "GPU Acceleration" => "gpu.md",
        "Distributed Computing" => "distributed.md",
        "Performance Guide" => "performance.md",
        "Advanced Usage" => "advanced.md"
    ],
    "Reference" => Any[
        "Normalization and Phase" => "norms.md",
        "Package Architecture" => "architecture.md",
        "API Reference" => "api/index.md",
        "Migrating to v2.0" => "migration.md"
    ]
]

#####
##### Build documentation
#####

println("Generating documentation with Documenter.jl...")

makedocs(;
    modules = [SHTnsKit],
    authors = "SHTnsKit.jl contributors",
    repo = "https://github.com/$repository/blob/{commit}{path}#{line}",
    sitename = "SHTnsKit.jl",
    format = format,
    pages = pages,
    clean = true,
    doctest = true,
    linkcheck = false,  # Set to true for link checking (slower)
    checkdocs = :exports,
    warnonly = [:missing_docs],
    draft = false
)

#####
##### Deploy documentation
#####

# Only deploy on CI
if get(ENV, "CI", "false") == "true"
    println("Deploying documentation...")
    
    deploydocs(;
        repo = "github.com/$repository",
        devbranch = "main",
        target = "build",
        deps = nothing,
        make = nothing,
        # Deploy both stable and dev docs since releases exist
        versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
        forcepush = false,
        deploy_config = Documenter.GitHubActions(),
        push_preview = true
    )
else
    println("Skipping deployment (not running in CI)")
    println("Documentation built successfully!")
    println("Preview from the repository root with `python3 -m http.server --directory docs/build`")
end
