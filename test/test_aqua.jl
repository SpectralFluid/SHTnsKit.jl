# Aqua.jl Quality Assurance Tests for SHTnsKit.jl
#
# This test suite uses Aqua.jl to perform automated quality assurance checks
# including ambiguity detection, unbound type parameters, undefined exports, etc.

using Test
using Aqua
using SHTnsKit

@testset "Aqua Quality Assurance" begin
    Aqua.test_all(
        SHTnsKit;
        ambiguities=false,  # Disable ambiguity tests (can be strict for complex packages)
        # Aqua flags weak deps (MPI, PencilArrays, PencilFFTs) as stale because they're
        # not directly imported in the main module — they're used by package extensions.
        stale_deps=(ignore=[:MPI, :PencilArrays, :PencilFFTs],),
        # Disable deps_compat check entirely - stdlib packages don't need compat entries
        deps_compat=false,
        piracies=false,  # Disable piracy tests for packages with many extensions
        persistent_tasks=false  # FFTW.jl creates background timer tasks for wisdom management
    )
end
