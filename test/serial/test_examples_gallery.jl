# Execute the code users copy from the web documentation. Plain `julia` fences
# are not run by Documenter, so this test prevents the gallery from drifting
# away from the public API while still allowing the MPI snippets to run under
# their dedicated multi-process workflow.

using Test

const EXAMPLES_GALLERY = normpath(joinpath(
    @__DIR__, "..", "..", "docs", "src", "examples", "index.md"
))
const DOCS_ROOT = normpath(joinpath(@__DIR__, "..", "..", "docs"))

function gallery_julia_blocks(path::AbstractString=EXAMPLES_GALLERY)
    markdown = read(path, String)
    return [m.captures[1] for m in eachmatch(r"```julia\n(.*?)\n```"s, markdown)]
end

@testset "Web documentation structure" begin
    makefile = read(joinpath(DOCS_ROOT, "make.jl"), String)
    grids = read(joinpath(DOCS_ROOT, "src", "grids.md"), String)
    gallery = read(EXAMPLES_GALLERY, String)

    @test occursin("<picture>", grids)
    @test occursin("grid-patterns.svg", grids)
    @test occursin("grid-patterns-stacked.svg", grids)
    @test occursin("alt=\"Four globes comparing", grids)

    @test !occursin("Literate.markdown", makefile)
    @test !occursin("Generated Examples", makefile)
    @test !occursin("Performance Tips", makefile)
    @test !occursin("SHTns 3.7 Parity", makefile)

    @test length(gallery_julia_blocks()) == 6
    @test !occursin("using MPI", gallery)
    @test occursin("@repl inspect", gallery)
end

@testset "Examples Gallery figures" begin
    gallery = read(EXAMPLES_GALLERY, String)
    for name in (
        "example-scalar-roundtrip.svg",
        "example-power-spectrum.svg",
        "example-vector-decomposition.svg",
        "example-stream-function.svg",
        "example-rotation.svg",
    )
        @test occursin(name, gallery)
        @test isfile(joinpath(DOCS_ROOT, "src", "assets", name))
    end
end

@testset "Examples Gallery serial snippets" begin
    blocks = gallery_julia_blocks()
    @test length(blocks) == 6

    for (index, code) in enumerate(blocks)
        occursin("using MPI", code) && continue
        occursin("BenchmarkTools", code) &&
            isnothing(Base.find_package("BenchmarkTools")) && continue

        @testset "Julia block $index" begin
            example_module = Module(Symbol("GalleryExample", index))
            redirect_stdout(devnull) do
                Base.include_string(example_module, code, "$(EXAMPLES_GALLERY):block-$index")
            end

            # Accuracy examples should demonstrate an actual roundtrip, not merely
            # print a large projection error while exiting successfully.
            for error_name in (:error, :velocity_error, :max_error)
                isdefined(example_module, error_name) || continue
                value = getfield(example_module, error_name)
                value isa Real || continue
                @test abs(value) < 1e-9
            end

            if isdefined(example_module, :orig_power) && isdefined(example_module, :rot_power)
                @test isapprox(
                    getfield(example_module, :orig_power),
                    getfield(example_module, :rot_power);
                    rtol=1e-10,
                )
            end
        end
    end
end
