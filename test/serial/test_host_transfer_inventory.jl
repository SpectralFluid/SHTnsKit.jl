using Test
using TOML
using SHTnsKit

include(joinpath(@__DIR__, "..", "support", "host_transfer_inventory.jl"))
using .HostTransferInventory

@testset "Host transfer inventory" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    audit_path = joinpath(root, "test", "fixtures", "compatibility",
                          "host_transfer_allowlist.toml")
    @test isfile(audit_path)
    transfer_fixture = TOML.parsefile(audit_path)
    scanned = scan_host_transfer_occurrences(root)
    allowed = transfer_fixture["entry"]
    @test transfer_fixture["audit"]["entry_count"] == length(scanned)
    scanned_keys = Set(transfer_occurrence_key.(scanned))
    allowed_keys = Set(entry["key"] for entry in allowed)
    @test length(allowed_keys) == length(allowed)
    @test isempty(setdiff(scanned_keys, allowed_keys))
    @test isempty(setdiff(allowed_keys, scanned_keys))
    @test all(entry -> entry["path"] in (occurrence.path for occurrence in scanned), allowed)
    @test all(entry -> occursin(r"^[0-9a-f]{64}$", entry["snippet_sha256"]), allowed)
    @test all(entry -> !isempty(entry["classification"]), allowed)
    @test all(entry -> !isempty(entry["reason"]), allowed)
    @test count(entry -> entry["token"] == "allowscalar", allowed) == 0
    similar_array_entries = filter(entry -> entry["token"] == "similar_array", allowed)
    @test length(similar_array_entries) == 3
    @test all(entry -> entry["path"] == "ext/ParallelGPU.jl", similar_array_entries)
    @test all(entry -> entry["classification"] == "bounded_pinned_mpi_staging",
              similar_array_entries)
    @test Set(entry["classification"] for entry in allowed) == Set((
        "bounded_pinned_mpi_staging", "cpu_only", "legacy_host_result",
        "metadata_or_storage_preserving", "small_setup_table",
        "explicit_cpu_or_fallback", "unreachable_early_error_callback",
    ))
    @test all(required -> any(entry -> entry["classification"] == required, allowed), (
        "metadata_or_storage_preserving", "small_setup_table", "cpu_only",
        "bounded_pinned_mpi_staging", "legacy_host_result",
        "unreachable_early_error_callback",
    ))
    for occurrence in scanned
        entry = only(filter(entry -> entry["key"] == transfer_occurrence_key(occurrence), allowed))
        @test entry["path"] == occurrence.path
        @test entry["token"] == occurrence.token
        @test entry["snippet_sha256"] == occurrence.snippet_sha256
        @test entry["same_snippet_ordinal"] == occurrence.same_snippet_ordinal
        expected_review = classify_transfer_occurrence(occurrence)
        @test expected_review !== nothing
        @test entry["classification"] == expected_review.classification
        @test entry["reason"] == expected_review.reason
    end

    cuda_source = read(joinpath(root, "ext", "SHTnsKitGPUExt.jl"), String)
    amd_source = read(joinpath(root, "ext", "SHTnsKitAMDGPUExt.jl"), String)
    @test !occursin("allowscalar", cuda_source)
    @test !occursin("allowscalar", amd_source)
    @test occursin("Historical host-buffer compatibility stays explicit and isolated", cuda_source)
    @test occursin("_staged_gpu_call", read(joinpath(root, "ext", "ParallelGPU.jl"), String))

    parallel_ad_source = read(joinpath(root, "ext", "SHTnsKitParallelADExt.jl"), String)
    @test occursin("on_device(parent(value))", parallel_ad_source)
    @test !occursin("HostPencilArray", parallel_ad_source)
    @test !occursin("Matrix{ComplexF64}(A)", parallel_ad_source)
    @test occursin("_require_host_pencil", parallel_ad_source)
    @test occursin("BackendUnavailableError", parallel_ad_source)
    scalar_synthesis_guard = findfirst(
        r"_require_host_pencil\(:dist_synthesis_pullback,\s*prototype_θφ,\s*comm\)",
        parallel_ad_source,
    )
    scalar_synthesis_forward = findfirst("y = SHTnsKit.dist_synthesis(", parallel_ad_source)
    vector_synthesis_guard = findfirst(
        r"_require_host_pencil\(\s*:dist_synthesis_sphtor_pullback,\s*prototype_θφ,\s*comm,?\s*\)",
        parallel_ad_source,
    )
    vector_synthesis_forward = findfirst(
        "y = SHTnsKit.dist_synthesis_sphtor(", parallel_ad_source,
    )
    @test scalar_synthesis_guard !== nothing
    @test scalar_synthesis_forward !== nothing
    @test vector_synthesis_guard !== nothing
    @test vector_synthesis_forward !== nothing
    if scalar_synthesis_guard !== nothing && scalar_synthesis_forward !== nothing
        @test first(scalar_synthesis_guard) < first(scalar_synthesis_forward)
    end
    if vector_synthesis_guard !== nothing && vector_synthesis_forward !== nothing
        @test first(vector_synthesis_guard) < first(vector_synthesis_forward)
    end

    parallel_runner = read(joinpath(root, "test", "parallel", "runtests.jl"), String)
    @test occursin("include(\"test_parallel_ad_storage.jl\")", parallel_runner)
end
