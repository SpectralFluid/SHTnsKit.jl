module HostTransferInventory

using SHA

export scan_host_transfer_occurrences, transfer_occurrence_key,
       transfer_scope_files, classify_transfer_occurrence

const TRANSFER_PATTERNS = (
    :to_device_cpu => r"to_device\s*\(\s*CPU\s*\(\s*\)\s*,",
    :to_cpu => r"\b_to_cpu\b",
    :similar_array => r"\bsimilar\s*\([^,\n]+,\s*Array\s*\)",
    :host_view => r"\bunsafe_wrap\s*\(\s*Array\b",
    :typed_array => r"\bArray\s*\{[^\n]*?\}\s*\(",
    :array => r"\bArray\s*\(",
    :matrix => r"\bMatrix\s*\{[^\n]*?\}\s*\(",
    :vector => r"\bVector\s*\{[^\n]*?\}\s*\(",
    :collect => r"\bcollect\s*\(",
    :allowscalar => r"\ballowscalar\b",
    :fallback => r"(?i)\bfallback\b",
    :cpu_constructor => r"\bCPU\s*\(",
    :copyto => r"\bcopyto!\s*\(",
    :copy_bang => r"\bcopy!\s*\(",
    :copy => r"\bcopy\s*\(",
    :parent => r"\bparent\s*\(",
)

"Files covered by the final host-transfer audit."
function transfer_scope_files(root::AbstractString)
    files = String[joinpath(root, "src", "device_utils.jl")]
    append!(files, sort(filter(
        path -> endswith(path, ".jl"),
        readdir(joinpath(root, "ext"); join=true),
    )))
    return files
end

@inline _normalise_snippet(line::AbstractString) =
    replace(strip(line), r"\s+" => " ")

"Stable identity independent of absolute checkout location."
function transfer_occurrence_key(entry)
    return join((entry.path, entry.token, entry.snippet_sha256,
                 string(entry.same_snippet_ordinal)), "|")
end

"""
Scan every selected spelling once. Longer/specific spellings win over their
substrings (`to_device(CPU(), ...)` is not also a `CPU()` occurrence;
`copyto!` is not also `copy`).
"""
function scan_host_transfer_occurrences(root::AbstractString)
    occurrences = NamedTuple[]
    duplicate_ordinals = Dict{Tuple{String,String,String},Int}()
    for file in transfer_scope_files(root)
        relative = replace(relpath(file, root), '\\' => '/')
        for (line_number, line) in enumerate(eachline(file))
            occupied = UnitRange{Int}[]
            for (token, pattern) in TRANSFER_PATTERNS
                for matched in eachmatch(pattern, line)
                    range = matched.offset:(matched.offset + ncodeunits(matched.match) - 1)
                    any(other -> !isdisjoint(range, other), occupied) && continue
                    push!(occupied, range)
                    snippet = _normalise_snippet(line)
                    digest = bytes2hex(sha256(snippet))
                    duplicate_key = (relative, String(token), digest)
                    ordinal = get(duplicate_ordinals, duplicate_key, 0) + 1
                    duplicate_ordinals[duplicate_key] = ordinal
                    push!(occurrences, (
                        path=relative,
                        line=line_number,
                        token=String(token),
                        snippet=snippet,
                        snippet_sha256=digest,
                        same_snippet_ordinal=ordinal,
                    ))
                end
            end
        end
    end
    sort!(occurrences; by=transfer_occurrence_key)
    return occurrences
end

@inline function _bounded_staging_occurrence(path, token, snippet)
    lower = lowercase(snippet)
    if path == "ext/ParallelGPU.jl"
        return token == "similar_array" ||
               occursin("pinned", lower) || occursin("device_to_host!", snippet) ||
               occursin("host_to_device!", snippet) || occursin("host_parent", snippet) ||
               occursin("Array(result)", snippet)
    elseif path in ("ext/SHTnsKitParallelCUDAExt.jl",
                    "ext/SHTnsKitParallelAMDGPUExt.jl")
        return token in ("vector", "copyto")
    elseif path == "ext/ParallelGPUVendorFirewall.jl"
        return false
    end
    return occursin("_stage", snippet) || occursin("pinned", lower)
end

@inline function _small_setup_table_occurrence(path, token, snippet)
    token in ("matrix", "vector", "array", "typed_array") || return false
    lower = lowercase(snippet)
    return path == "ext/GPUCommon.jl" ||
           path == "ext/ParallelTransposeTransforms.jl" ||
           occursin("cache", lower) || occursin("table", lower)
end

"""Assign the required explicit review category and an actionable review reason."""
function classify_transfer_occurrence(entry)
    path, token, snippet = entry.path, entry.token, entry.snippet
    if token == "allowscalar"
        return nothing
    elseif token == "host_view" && path == "ext/SHTnsKitMetalExt.jl" && occursin("tables", snippet)
        return (
            classification="metadata_or_storage_preserving",
            reason="the CPU initializes fresh shared Metal basis tables in place before publication",
        )
    elseif path == "ext/ParallelGPUVendorFirewall.jl" &&
           token == "cpu_constructor"
        return (
            classification="unreachable_early_error_callback",
            reason="legacy callback text is guarded by the compound early-error boundary; ordinary MPI GPU dispatch must never execute or transfer through it",
        )
    elseif occursin("gpu_analysis_safe", snippet) ||
           occursin("gpu_synthesis_safe", snippet) ||
           occursin("Historical host-buffer compatibility", snippet) ||
           occursin("return Array(_cuda", snippet)
        return (
            classification="legacy_host_result",
            reason="legacy host-result API boundary; retain only while compatibility tests assert this documented return-storage contract",
        )
    elseif _bounded_staging_occurrence(path, token, snippet)
        return (
            classification="bounded_pinned_mpi_staging",
            reason="explicit pinned MPI staging boundary; keep the transfer bounded, synchronized, and restored to the originating vendor before return",
        )
    elseif _small_setup_table_occurrence(path, token, snippet)
        return (
            classification="small_setup_table",
            reason="small setup/cache table allocated on the host; do not use this classification for spatial or spectral result fields",
        )
    elseif occursin("AdvancedADExt", path) || occursin("ForwardDiffExt", path) ||
           occursin("ZygoteExt", path) || occursin("ParallelADExt", path)
        return (
            classification="cpu_only",
            reason="CPU-only automatic-differentiation storage; vendor PencilArray inputs must be rejected before forward work or materialization",
        )
    elseif token in ("collect", "parent", "copy", "copyto", "copy_bang")
        return (
            classification="metadata_or_storage_preserving",
            reason="metadata collection or storage-preserving parent/copy operation; verify that the destination retains the source storage family",
        )
    elseif token in ("matrix", "vector", "array", "typed_array")
        return (
            classification="cpu_only",
            reason="explicit host workspace in a CPU-only algorithm; never route a vendor-backed mathematical result through this allocation",
        )
    elseif token in ("cpu_constructor", "to_cpu", "to_device_cpu", "fallback")
        return (
            classification="explicit_cpu_or_fallback",
            reason="explicit CPU dispatch/conversion or fallback marker; callers must opt into this storage transition and strict GPU paths must reject it",
        )
    end
    return nothing
end

end
