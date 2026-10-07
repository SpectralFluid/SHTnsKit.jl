using Test
using TOML
using SHTnsKit

include(joinpath(@__DIR__, "..", "support", "compatibility_inventory.jl"))
using .CompatibilityInventory

const _BASELINE_COMPATIBILITY_FIXTURE = joinpath(
    @__DIR__, "..", "fixtures", "compatibility", "e2ce9027.toml"
)
const _COMPATIBILITY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function _compatible_declared_type(baseline::AbstractString, current::AbstractString)
    baseline_type = resolve_type_expression(baseline)
    current_type = resolve_type_expression(current)
    if baseline_type !== nothing && current_type !== nothing
        return baseline_type <: current_type
    end
    return baseline == current
end

function _has_baseline_method(method)
    isdefined(SHTnsKit, Symbol(method.name)) || return false
    function_object = getfield(SHTnsKit, Symbol(method.name))
    types = [resolve_type_expression(argument.type, method.where)
             for argument in method.positional]
    any(isnothing, types) && return false

    required = count(argument -> argument.kind == "required", method.positional)
    vararg = !isempty(method.positional) && last(method.positional).kind == "vararg"
    fixed = length(method.positional) - (vararg ? 1 : 0)
    arities = collect(required:fixed)
    vararg && push!(arities, fixed + 1)
    keyword_names = Tuple(Symbol(argument.name) for argument in method.keywords
                          if argument.kind != "keyword_vararg")
    return all(arities) do arity
        signature_types = if vararg && arity > fixed
            vcat(types[1:fixed], types[end])
        else
            types[1:arity]
        end
        hasmethod(function_object, Tuple{signature_types...}, keyword_names)
    end
end

function _run_export_family_probe(name::String, kind::String)
    symbol = Symbol(name)
    if kind == "method_family"
        # Each declaration in this family is checked structurally and with
        # `hasmethod` below; this record prevents family-level omissions.
        @test isdefined(SHTnsKit, symbol)
        @test getfield(SHTnsKit, symbol) isa Function ||
              getfield(SHTnsKit, symbol) isa Type
    elseif kind == "type"
        @test isdefined(SHTnsKit, symbol)
        @test getfield(SHTnsKit, symbol) isa Type
        if name == "ComputeDevice"
            @test isabstracttype(ComputeDevice)
            @test CPU <: ComputeDevice
            @test GPU <: ComputeDevice
        elseif name == "CPU"
            @test CPU() isa ComputeDevice
        elseif name == "GPU"
            @test GPU() isa ComputeDevice
        elseif name == "SHTConfig"
            @test create_gauss_config(2, 3) isa SHTConfig
        elseif name == "SHTPlan"
            @test SHTPlan(create_gauss_config(2, 3)) isa SHTPlan
        elseif name == "SHTRotation"
            @test SHTRotation(2, 2) isa SHTRotation
        else
            error("unprobed baseline type family: $name")
        end
    elseif kind == "macro"
        @test isdefined(SHTnsKit, symbol)
        source = collect(1.0:5.0)
        output = zeros(5)
        if name == "@sht_loop"
            @sht_loop output[I] = source[I] over I ∈ CartesianIndices(output)
            @test output == source
        elseif name == "@sht_inside"
            @sht_inside output[I] = source[I]
            @test output == [0.0, 2.0, 3.0, 4.0, 0.0]
        else
            error("unprobed baseline macro family: $name")
        end
    elseif kind == "generated_or_extension"
        @test name in ("DistAnalysisPlan", "DistPlan", "DistQstPlan", "DistSphtorPlan")
        extension_source = read(joinpath(_COMPATIBILITY_ROOT, "ext", "ParallelPlans.jl"), String)
        mpi_runner = read(joinpath(_COMPATIBILITY_ROOT, "test", "parity", "runtests_mpi.jl"), String)
        @test occursin(Regex("struct\\s+" * name * "(?:\\{|\\s)"), extension_source)
        @test occursin(name * "(", extension_source)
        @test occursin(name * "(", mpi_runner)
    else
        error("unknown baseline export classification: $kind")
    end
end

function _compatibility_coefficients(cfg)
    Q = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    S = zeros(ComplexF64, size(Q))
    T = zeros(ComplexF64, size(Q))
    Q[2, 1] = 0.75
    Q[3, 2] = 0.125 - 0.25im
    S[2, 1] = -0.5
    S[3, 2] = 0.2 + 0.1im
    T[2, 1] = 0.375
    T[3, 2] = -0.15 + 0.05im
    return Q, S, T
end

function _run_baseline_probe(id::String)
    if id == "gauss_configuration_defaults"
        @test applicable(create_gauss_config, 3, 4)
        cfg = create_gauss_config(3, 4)
        @test (cfg.mmax, cfg.mres, cfg.nlon) == (3, 1, 8)
        @test (cfg.norm, cfg.cs_phase, cfg.real_norm, cfg.robert_form) ==
              (:orthonormal, true, false, false)
    elseif id == "regular_configuration_defaults"
        @test applicable(create_regular_config, 3, 5)
        cfg = create_regular_config(3, 5)
        @test (cfg.mmax, cfg.mres, cfg.nlon, cfg.grid_type) == (3, 1, 8, :regular)
        @test !cfg.real_norm
        @test !cfg.robert_form
    elseif id == "scalar_transform_keywords"
        cfg = create_gauss_config(3, 4)
        Q, _, _ = _compatibility_coefficients(cfg)
        @test applicable(synthesis, cfg, Q)
        field = synthesis(cfg, Q)
        explicit = synthesis(cfg, Q; real_output=true, fft_scratch=nothing, use_rfft=false)
        @test field isa Matrix{Float64}
        @test field ≈ explicit
        @test applicable(analysis, cfg, field)
        @test analysis(cfg, field) ≈
              analysis(cfg, field; fft_scratch=nothing, use_rfft=false)
        @test synthesis(cfg, Q; real_output=false) isa Matrix{ComplexF64}
    elseif id == "vector_transform_tuple_order"
        cfg = create_gauss_config(3, 4)
        _, S, T = _compatibility_coefficients(cfg)
        @test applicable(synthesis_sphtor, cfg, S, T)
        vector_field = synthesis_sphtor(cfg, S, T)
        @test vector_field isa Tuple && length(vector_field) == 2
        sph = synthesis_sphtor(cfg, S, zero(T))
        tor = synthesis_sphtor(cfg, zero(S), T)
        @test vector_field[1] ≈ sph[1] + tor[1]
        @test vector_field[2] ≈ sph[2] + tor[2]
        recovered = analysis_sphtor(cfg, vector_field...)
        @test recovered isa Tuple && length(recovered) == 2
        @test recovered[1] ≈ analysis_sphtor(cfg, sph...)[1]
        @test recovered[2] ≈ analysis_sphtor(cfg, tor...)[2]
    elseif id == "qst_transform_tuple_order"
        cfg = create_gauss_config(3, 4)
        Q, S, T = _compatibility_coefficients(cfg)
        @test applicable(synthesis_qst, cfg, Q, S, T)
        qst = synthesis_qst(cfg, Q, S, T)
        @test qst isa Tuple && length(qst) == 3
        @test qst[1] ≈ synthesis(cfg, Q)
        sphtor = synthesis_sphtor(cfg, S, T)
        @test qst[2] ≈ sphtor[1]
        @test qst[3] ≈ sphtor[2]
        recovered = analysis_qst(cfg, qst...)
        @test recovered isa Tuple && length(recovered) == 3
        @test recovered[1] ≈ analysis(cfg, qst[1])
        vector_recovered = analysis_sphtor(cfg, qst[2], qst[3])
        @test recovered[2] ≈ vector_recovered[1]
        @test recovered[3] ≈ vector_recovered[2]
    elseif id == "batch_transform_keywords"
        cfg = create_gauss_config(3, 4)
        Q, _, _ = _compatibility_coefficients(cfg)
        coefficients = cat(Q, 2Q; dims=3)
        @test applicable(synthesis_batch, cfg, coefficients)
        fields = synthesis_batch(cfg, coefficients)
        @test fields isa Array{Float64,3}
        @test fields ≈ synthesis_batch(
            cfg, coefficients; real_output=true, use_rfft=false,
        )
        @test applicable(analysis_batch, cfg, fields)
        @test analysis_batch(cfg, fields) ≈
              analysis_batch(cfg, fields; use_rfft=false)
    elseif id == "packed_transform_signatures"
        cfg = create_gauss_config(3, 4)
        Q, _, _ = _compatibility_coefficients(cfg)
        packed = zeros(ComplexF64, cfg.nlm)
        for m in 0:cfg.mmax, l in m:cfg.lmax
            packed[LM_index(cfg.lmax, cfg.mres, l, m) + 1] = Q[l + 1, m + 1]
        end
        @test applicable(synthesis_packed, cfg, packed)
        field = synthesis_packed(cfg, packed)
        @test applicable(analysis_packed, cfg, vec(field))
        @test length(analysis_packed(cfg, vec(field))) == cfg.nlm
        @test applicable(synthesis_point, cfg, Q, 0.25, 0.5)
        @test synthesis_point(cfg, Q, 0.25, 0.5) isa Number
    elseif id == "rotation_mutation_contract"
        rotation = shtns_rotation_create(3, 3, 0)
        @test applicable(shtns_rotation_set_angles_ZYZ, rotation, 0.0, 0.0, 0.0)
        @test shtns_rotation_set_angles_ZYZ(rotation, 0.0, 0.0, 0.0) === nothing
        input = zeros(ComplexF64, nlm_calc(3, 3, 1))
        input[1] = 1
        output = similar(input)
        @test applicable(shtns_rotation_apply_real, rotation, input, output)
        @test shtns_rotation_apply_real(rotation, input, output) === output
        @test output ≈ input
    elseif id == "device_value_first_order"
        values = [1.0, 2.0]
        @test applicable(to_device, values, CPU())
        @test set_device!(CPU()) isa CPU
        @test to_device(values, CPU()) === values
        @test to_device(values) === values
        @test on_device(values) isa CPU
    else
        error("unknown cleanup-baseline compatibility probe: $id")
    end
end

@testset "Cleanup-baseline public API compatibility" begin
    @test isfile(_BASELINE_COMPATIBILITY_FIXTURE)

    if isfile(_BASELINE_COMPATIBILITY_FIXTURE)
        fixture = TOML.parsefile(_BASELINE_COMPATIBILITY_FIXTURE)
        baseline_exports = Set(Symbol.(fixture["exports"]))
        final_exports = Set(names(SHTnsKit; all=false, imported=false))

        @test fixture["baseline"]["commit"] == "e2ce9027"
        @test fixture["baseline"]["commit_full"] ==
              "e2ce9027c94a40b29c6582bc0616d22e26a94ad7"
        @test fixture["baseline"]["source_blob"] ==
              "f5ec0b731b891260a4c44f94951ddf118086f44b"
        @test fixture["baseline"]["export_count"] == length(baseline_exports)
        @test baseline_exports ⊆ final_exports
        # An exported name the core never defines reaches `using SHTnsKit`
        # callers as an UndefVarError rather than the missing-extension hint.
        @test isempty(filter(name -> !isdefined(SHTnsKit, name), collect(final_exports)))
        if Base.get_extension(SHTnsKit, :SHTnsKitParallelExt) === nothing
            cfg = create_gauss_config(2, 3)
            for constructor in (DistAnalysisPlan, DistPlan, DistSphtorPlan, DistQstPlan)
                @test_throws "Parallel extension not loaded" constructor(cfg, zeros(3, 5))
            end
        end

        # A names-only export snapshot does not prove that callers retain the
        # positional, typed, and keyword entry points they compiled against.
        @test haskey(fixture["baseline"], "method_count")
        @test haskey(fixture, "method")
        @test haskey(fixture, "runtime_probe")

        # Normal tests must work in source tarballs and shallow checkouts. The
        # immutable baseline inventory is committed in this fixture; only the
        # separately-invoked generator consults Git history.
        baseline_inventory = withenv("PATH" => "/nonexistent") do
            fixture_inventory(fixture)
        end
        current_inventory = inventory_sources(source_pairs(_COMPATIBILITY_ROOT), fixture["exports"])
        @test isempty(current_inventory.parse_errors)
        @test occursin(r"^[0-9a-f]{64}$", baseline_inventory.source_digest)
        @test fixture["baseline"]["method_count"] == 247 == length(baseline_inventory.methods)

        baseline_tuple_contracts = filter(
            method -> !isempty(method.tuple_arities), baseline_inventory.methods,
        )
        @test length(baseline_tuple_contracts) == 31
        @test haskey(fixture["baseline"], "tuple_arity_method_count")
        @test fixture["baseline"]["tuple_arity_method_count"] == 31
        @test all(entry -> haskey(entry, "tuple_arities"), fixture["method"])
        @test haskey(fixture["baseline"], "tuple_component_method_count")
        @test fixture["baseline"]["tuple_component_method_count"] == 31
        @test all(entry -> haskey(entry, "tuple_component_signatures"), fixture["method"])

        fixture_methods = baseline_inventory.methods
        @test length(fixture_methods) == length(baseline_inventory.methods)
        @test sort(method_to_fixture.(fixture_methods); by=entry -> entry["fingerprint"]) ==
              sort(fixture["method"]; by=entry -> entry["fingerprint"])
        baseline_tuple_arities = Dict(
            method_fingerprint(method) => method.tuple_arities
            for method in fixture_methods
        )
        fixture_tuple_arities = Dict(
            method_fingerprint(method) => method.tuple_arities
            for method in fixture_methods
        )
        @test fixture_tuple_arities == baseline_tuple_arities
        @test count(method -> !isempty(method.tuple_arities), fixture_methods) == 31
        @test count(method -> !isempty(method.tuple_component_signatures),
                    fixture_methods) == 31
        @test sum(method -> length(method.tuple_component_signatures), fixture_methods) == 34

        tuple_contract = first(baseline_tuple_contracts)
        incompatible_tuple_contract = merge(
            tuple_contract,
            (; tuple_arities=[maximum(tuple_contract.tuple_arities) + 1]),
        )
        @test !method_compatible(tuple_contract, incompatible_tuple_contract)

        ordered_tuple_contract = first(filter(baseline_tuple_contracts) do method
            length(method.tuple_component_signatures) == 1 &&
                length(only(method.tuple_component_signatures)) >= 2 &&
                only(method.tuple_component_signatures) !=
                    reverse(only(method.tuple_component_signatures))
        end)
        swapped_tuple_contract = merge(
            ordered_tuple_contract,
            (; tuple_component_signatures=[
                reverse(only(ordered_tuple_contract.tuple_component_signatures)),
            ]),
        )
        @test !method_compatible(ordered_tuple_contract, swapped_tuple_contract)

        constrained_contract = merge(first(fixture_methods), (; where=["T <: Real"]))
        widened_constraint = merge(constrained_contract, (; where=["T <: Number"]))
        narrowed_constraint = merge(constrained_contract, (; where=["T <: Integer"]))
        @test method_compatible(constrained_contract, widened_constraint)
        @test !method_compatible(constrained_contract, narrowed_constraint)

        multiple_constraints = merge(
            first(fixture_methods),
            (; where=["T <: Real", "S <: Integer"]),
        )
        reordered_widened_constraints = merge(
            multiple_constraints,
            (; where=["S <: Real", "T <: Number"]),
        )
        narrowed_multiple_constraints = merge(
            multiple_constraints,
            (; where=["S <: Signed", "T <: Integer"]),
        )
        @test method_compatible(multiple_constraints, reordered_widened_constraints)
        @test !method_compatible(multiple_constraints, narrowed_multiple_constraints)

        compatibility_misses = String[]
        for baseline_method in fixture_methods
            any(current_inventory.methods) do current_method
                method_compatible(
                    baseline_method, current_method;
                    type_compatible=_compatible_declared_type,
                )
            end || push!(compatibility_misses, method_fingerprint(baseline_method))
        end
        @test isempty(compatibility_misses)

        unresolved = filter(fixture_methods) do method
            any(argument -> resolve_type_expression(argument.type, method.where) === nothing,
                method.positional)
        end
        @test isempty(unresolved)
        @test all(_has_baseline_method, fixture_methods)

        runtime_probes = fixture["runtime_probe"]
        @test length(runtime_probes) == length(baseline_exports) == 248
        @test length(unique(probe["name"] for probe in runtime_probes)) == 248
        classified = Dict(probe.name => probe.kind for probe in baseline_inventory.probes)
        expected_proof = Dict(
            "method_family" => "static_structural_and_hasmethod",
            "type" => "explicit_core_family_probe",
            "macro" => "explicit_core_family_probe",
            "generated_or_extension" => "extension_source_and_mpi_runner_probe",
        )
        for probe in runtime_probes
            @test get(classified, probe["name"], nothing) == probe["classification"]
            @test probe["proof"] == expected_proof[probe["classification"]]
            _run_export_family_probe(probe["name"], probe["classification"])
        end
        @test Set(keys(classified)) == Set(String.(fixture["exports"]))
        @test count(==("method_family"), values(classified)) == 236
        @test count(==("type"), values(classified)) == 6
        @test count(==("macro"), values(classified)) == 2
        @test count(==("generated_or_extension"), values(classified)) == 4

        probes = fixture["probe"]
        expected_probe_ids = Set((
            "gauss_configuration_defaults",
            "regular_configuration_defaults",
            "scalar_transform_keywords",
            "vector_transform_tuple_order",
            "qst_transform_tuple_order",
            "batch_transform_keywords",
            "packed_transform_signatures",
            "rotation_mutation_contract",
            "device_value_first_order",
        ))
        @test Set(probe["id"] for probe in probes) == expected_probe_ids
        for probe in probes
            @test haskey(probe, "signature")
            @test haskey(probe, "defaults")
            @test haskey(probe, "return_contract")
            _run_baseline_probe(probe["id"])
        end
    end
end
