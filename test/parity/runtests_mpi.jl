using Test
using MPI
using PencilArrays
using PencilFFTs
using SHTnsKit

MPI.Init()

include("scalar_full.jl")
include("scalar_variants.jl")
include("sphtor_full.jl")
include("qst_full.jl")
include("vector_variants.jl")
include("local_evaluation.jl")
include("operators.jl")
include("rotations.jl")
include("shtns37_fixtures.jl")

struct MPIScalarAdapter <: ScalarParityAdapter
    comm
end

struct MPIVectorAdapter <: VectorParityAdapter
    comm
end

struct MPIQSTAdapter <: QSTParityAdapter
    comm
end

function vector_place(adapter::MPIVectorAdapter, cfg::SHTConfig,
                      value::AbstractMatrix, kind::Symbol)
    if kind === :spectral
        return matrix_to_spectral_pencil(cfg, value; comm=adapter.comm)
    end
    pen = create_spatial_pencil(cfg; comm=adapter.comm)
    placed = PencilArray{eltype(value)}(undef, pen)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        parent(placed)[i, j] = value[iglobal, jglobal]
    end
    return placed
end

vector_collect(::MPIVectorAdapter, value::PencilArray, cfg::SHTConfig) =
    size_global(value) == (cfg.lmax + 1, cfg.mmax + 1) ?
        spectral_pencil_to_matrix(cfg, value) : _collect_spatial(value, cfg)
vector_resident(::MPIVectorAdapter, value) = @test value isa PencilArray
vector_analysis(::MPIVectorAdapter, cfg, Vt, Vp; use_rfft=false) =
    analysis_sphtor(cfg, Vt, Vp; use_rfft)
vector_analysis_cplx(::MPIVectorAdapter, cfg, Vt, Vp) =
    analysis_sphtor_cplx(cfg, Vt, Vp)
vector_synthesis(::MPIVectorAdapter, cfg, S, T, prototype;
                 real_output=true, use_rfft=false) =
    synthesis_sphtor(cfg, S, T; prototype_θφ=prototype, real_output, use_rfft)
vector_synthesis_cplx(::MPIVectorAdapter, cfg, S, T, prototype) =
    synthesis_sphtor_cplx(cfg, S, T; prototype_θφ=prototype)
vector_sph(::MPIVectorAdapter, cfg, S, prototype; real_output=true) =
    synthesis_sph(cfg, S; prototype_θφ=prototype, real_output)
vector_sph_cplx(::MPIVectorAdapter, cfg, S, prototype) =
    synthesis_sph_cplx(cfg, S; prototype_θφ=prototype)
vector_tor(::MPIVectorAdapter, cfg, T, prototype; real_output=true) =
    synthesis_tor(cfg, T; prototype_θφ=prototype, real_output)
vector_tor_cplx(::MPIVectorAdapter, cfg, T, prototype) =
    synthesis_tor_cplx(cfg, T; prototype_θφ=prototype)

qst_place(adapter::MPIQSTAdapter, cfg, value, kind) =
    vector_place(MPIVectorAdapter(adapter.comm), cfg, value, kind)
qst_collect(::MPIQSTAdapter, value::PencilArray, cfg) =
    size_global(value) == (cfg.lmax + 1, cfg.mmax + 1) ?
        spectral_pencil_to_matrix(cfg, value) : _collect_spatial(value, cfg)
qst_resident(::MPIQSTAdapter, value) = @test value isa PencilArray
qst_analysis(::MPIQSTAdapter, cfg, Vr, Vt, Vp; use_rfft=false) =
    analysis_qst(cfg, Vr, Vt, Vp; use_rfft)
qst_analysis_cplx(::MPIQSTAdapter, cfg, Vr, Vt, Vp) =
    analysis_qst_cplx(cfg, Vr, Vt, Vp)
qst_synthesis(::MPIQSTAdapter, cfg, Q, S, Tlm, prototype;
              real_output=true, use_rfft=false) =
    synthesis_qst(
        cfg, Q, S, Tlm; prototype_θφ=prototype, real_output, use_rfft,
    )
qst_synthesis_cplx(::MPIQSTAdapter, cfg, Q, S, Tlm, prototype) =
    synthesis_qst_cplx(cfg, Q, S, Tlm; prototype_θφ=prototype)

function place(adapter::MPIScalarAdapter, cfg::SHTConfig, value::AbstractMatrix, kind::Symbol)
    if kind === :spectral
        return matrix_to_spectral_pencil(cfg, value; comm=adapter.comm)
    end
    pen = create_spatial_pencil(cfg; comm=adapter.comm)
    placed = PencilArray{eltype(value)}(undef, pen)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        parent(placed)[i, j] = value[iglobal, jglobal]
    end
    return placed
end

collect_result(::MPIScalarAdapter, value::PencilArray, cfg::SHTConfig) =
    size_global(value) == (cfg.lmax + 1, cfg.mmax + 1) ?
        spectral_pencil_to_matrix(cfg, value) : _collect_spatial(value, cfg)

function _collect_spatial(value::PencilArray, cfg::SHTConfig)
    result = zeros(eltype(value), cfg.nlat, cfg.nlon)
    pen = pencil(value)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        result[iglobal, jglobal] = parent(value)[i, j]
    end
    MPI.Allreduce!(result, +, PencilArrays.get_comm(value))
    return result
end

function _collect_distributed_vector(value::PencilArray)
    global_size = size_global(value)
    global_size[2] == 1 || error("expected a singleton-column PencilArray")
    result = zeros(eltype(value), global_size[1])
    globals = collect(Int, PencilArrays.range_local(pencil(value))[1])
    @inbounds for (i, iglobal) in pairs(globals)
        result[iglobal] = parent(value)[i, 1]
    end
    MPI.Allreduce!(result, +, PencilArrays.get_comm(value))
    return result
end

function _place_distributed_vector(values::AbstractVector, comm)
    pen = Pencil((length(values), 1), (1,), comm)
    result = PencilArray{eltype(values)}(undef, pen)
    globals = collect(Int, PencilArrays.range_local(pen)[1])
    @inbounds for (i, iglobal) in pairs(globals)
        parent(result)[i, 1] = values[iglobal]
    end
    return result
end

function _place_distributed_batch(cfg::SHTConfig, values::AbstractArray{T,3},
                                  kind::Symbol, comm) where {T}
    decomposition = kind === :spatial ? (1,) : (2,)
    result = PencilArray{T}(undef, Pencil(size(values), decomposition, comm))
    ranges = map(d -> collect(Int, PencilArrays.range_local(result)[d]), 1:3)
    @inbounds for (k, kg) in pairs(ranges[3]), (j, jg) in pairs(ranges[2]),
                  (i, ig) in pairs(ranges[1])
        parent(result)[i, j, k] = values[ig, jg, kg]
    end
    return result
end

function _collect_distributed_batch(value::PencilArray, comm)
    result = zeros(eltype(value), size_global(value))
    ranges = map(d -> collect(Int, PencilArrays.range_local(value)[d]), 1:3)
    @inbounds for (k, kg) in pairs(ranges[3]), (j, jg) in pairs(ranges[2]),
                  (i, ig) in pairs(ranges[1])
        result[ig, jg, kg] = parent(value)[i, j, k]
    end
    MPI.Allreduce!(result, +, comm)
    return result
end

analysis_call(::MPIScalarAdapter, cfg, field; use_rfft=false) =
    analysis(cfg, field; use_rfft)
synthesis_call(::MPIScalarAdapter, cfg, coefficients, prototype;
               real_output, use_rfft=false) =
    synthesis(cfg, coefficients; prototype_θφ=prototype, real_output, use_rfft)
synthesis_cplx_call(::MPIScalarAdapter, cfg, coefficients, prototype) =
    synthesis_cplx(cfg, coefficients; prototype_θφ=prototype)
assert_resident(::MPIScalarAdapter, value) = @test value isa PencilArray

function test_phi_split_complex_float32(adapter::MPIScalarAdapter)
    cfg = _scalar_config(:gauss, 3, 8)
    coefficients = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
    coefficients[1, 1] = ComplexF32(0.25, -0.15)
    coefficients[3, 3] = ComplexF32(-0.2, 0.1)
    field = _direct_scalar_sum(cfg, coefficients; real_output=false)

    pen = Pencil((cfg.nlat, cfg.nlon), adapter.comm)
    distributed = PencilArray{ComplexF32}(undef, pen)
    theta = collect(PencilArrays.range_local(pen)[1])
    phi = collect(PencilArrays.range_local(pen)[2])
    for (j, jglobal) in pairs(phi), (i, iglobal) in pairs(theta)
        parent(distributed)[i, j] = field[iglobal, jglobal]
    end

    analyzed = analysis(cfg, distributed; return_pencil=true)
    @test analyzed isa PencilArray
    analyzed_host = spectral_pencil_to_matrix(cfg, analyzed)
    @test eltype(analyzed_host) === ComplexF32
    @test analyzed_host ≈ analysis(CPU(), cfg, field) atol=2f-4 rtol=2f-4

    reconstructed = synthesis(
        cfg, analyzed; prototype_θφ=distributed, real_output=false,
    )
    @test reconstructed isa PencilArray
    reconstructed_host = _collect_spatial(reconstructed, cfg)
    @test eltype(reconstructed_host) === ComplexF32
    @test reconstructed_host ≈ field atol=2f-4 rtol=2f-4
    return nothing
end

function test_pencil_native_path(adapter::MPIScalarAdapter)
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    cfg = _scalar_config(:gauss, 5, 10)
    coefficients, field = _closed_form_low_order(cfg, Float64)
    spatial = place(adapter, cfg, field, :spatial)
    spectral = place(adapter, cfg, coefficients, :spectral)

    extension._reset_pencil_scalar_stats!()
    analyzed = analysis(cfg, spatial)
    after_analysis = extension._pencil_scalar_stats()
    @test after_analysis.full_matrix_helper_calls == 0
    max_spectral_local = MPI.Allreduce(length(parent(analyzed)), max, adapter.comm)
    @test after_analysis.analysis_max_message_elements <= max_spectral_local

    extension._reset_pencil_scalar_stats!()
    reconstructed = synthesis(cfg, spectral; prototype_θφ=spatial)
    after_synthesis = extension._pencil_scalar_stats()
    @test after_synthesis.full_matrix_helper_calls == 0
    max_theta_local = MPI.Allreduce(size(parent(spatial), 1), max, adapter.comm)
    @test after_synthesis.synthesis_max_message_elements <= max_theta_local * cfg.nlon
    @test _collect_spatial(reconstructed, cfg) ≈ field atol=3e-12 rtol=3e-12

    dense_compat = analysis(cfg, spatial; return_pencil=false)
    @test dense_compat isa Matrix
    @test dense_compat ≈ coefficients atol=3e-12 rtol=3e-12
    local_compat = dist_synthesis(cfg, dense_compat; prototype_θφ=spatial)
    compat = PencilArray{eltype(local_compat)}(undef, pencil(spatial))
    copyto!(parent(compat), local_compat)
    @test _collect_spatial(compat, cfg) ≈ field atol=3e-12 rtol=3e-12

    pen2 = try
        Pencil((cfg.nlat, cfg.nlon), (1, 2), adapter.comm)
    catch
        nothing
    end
    if pen2 !== nothing
        ranges = PencilArrays.range_local(pen2)
        both_split = MPI.Allreduce(
            length(ranges[1]) < cfg.nlat && length(ranges[2]) < cfg.nlon,
            &, adapter.comm,
        )
        if both_split
            spatial2 = PencilArray{Float64}(undef, pen2)
            for (j, jglobal) in pairs(ranges[2]), (i, iglobal) in pairs(ranges[1])
                parent(spatial2)[i, j] = field[iglobal, jglobal]
            end
            extension._reset_pencil_scalar_stats!()
            spectral2 = analysis(cfg, spatial2)
            reconstructed2 = synthesis(cfg, spectral2; prototype_θφ=spatial2)
            @test extension._pencil_scalar_stats().full_matrix_helper_calls == 0
            @test _collect_spatial(reconstructed2, cfg) ≈ field atol=3e-12 rtol=3e-12
        else
            @test true
        end
    else
        @test true
    end
    return nothing
end

function _all_ranks_catch(call, comm; message_contains=nothing)
    message = try
        call()
        ""
    catch err
        "$(typeof(err)): $(sprint(showerror, err))"
    end
    reference = MPI.bcast(message, 0, comm)
    caught_same = !isempty(message) && message == reference &&
        (message_contains === nothing || occursin(message_contains, message))
    total = MPI.Allreduce(caught_same ? 1 : 0, +, comm)
    MPI.Barrier(comm)
    return total == MPI.Comm_size(comm)
end

function test_collective_validation(adapter::MPIScalarAdapter)
    cfg = _scalar_config(:gauss, 3, 8)
    _, field = _closed_form_low_order(cfg, Float64)
    spatial = place(adapter, cfg, field, :spatial)
    coefficients = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    spectral = place(adapter, cfg, coefficients, :spectral)
    @test spectral_pencil_to_matrix(cfg, spectral; comm=adapter.comm) == coefficients

    malformed_pen = Pencil((cfg.lmax + 2, cfg.mmax + 1), adapter.comm)
    malformed = PencilArray{ComplexF64}(undef, malformed_pen)
    fill!(parent(malformed), 0)
    @test _all_ranks_catch(adapter.comm) do
        synthesis(cfg, malformed; prototype_θφ=spatial)
    end

    rank = MPI.Comm_rank(adapter.comm)
    malformed_matrix = rank == 0 ? coefficients :
        zeros(ComplexF64, cfg.lmax + 2, cfg.mmax + 1)
    @test _all_ranks_catch(adapter.comm) do
        matrix_to_spectral_pencil(cfg, malformed_matrix; comm=adapter.comm)
    end

    rank_varying_matrix = copy(coefficients)
    rank_varying_matrix[1, 1] = rank
    @test _all_ranks_catch(adapter.comm) do
        matrix_to_spectral_pencil(cfg, rank_varying_matrix; comm=adapter.comm)
    end

    helper_shape = rank == 0 ?
        (cfg.lmax + 1, cfg.mmax + 1) : (cfg.lmax + 2, cfg.mmax + 1)
    helper_malformed_pen = Pencil(helper_shape, adapter.comm)
    helper_malformed = PencilArray{ComplexF64}(undef, helper_malformed_pen)
    fill!(parent(helper_malformed), 0)
    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, helper_malformed)
    end

    varying_precision_matrix = rank == 0 ?
        zeros(ComplexF32, size(coefficients)) : coefficients
    @test _all_ranks_catch(adapter.comm) do
        matrix_to_spectral_pencil(cfg, varying_precision_matrix; comm=adapter.comm)
    end

    varying_precision_spectral = if rank == 0
        PencilArray{ComplexF32}(undef, pencil(spectral))
    else
        spectral
    end
    fill!(parent(varying_precision_spectral), 0)
    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, varying_precision_spectral)
    end

    wrong_decomposition_pen = Pencil(
        (cfg.lmax + 1, cfg.mmax + 1), (1,), adapter.comm,
    )
    wrong_decomposition = PencilArray{ComplexF64}(undef, wrong_decomposition_pen)
    fill!(parent(wrong_decomposition), 0)
    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, wrong_decomposition)
    end

    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, spectral; comm=MPI.COMM_SELF)
    end

    optional_comm = rank == 0 ? nothing : adapter.comm
    @test _all_ranks_catch(adapter.comm) do
        spectral_pencil_to_matrix(cfg, spectral; comm=optional_comm)
    end

    rank_varying = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    rank_varying[1, 1] = rank
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis(cfg, rank_varying; prototype_θφ=spatial)
    end

    optional_minus = rank == 0 ? copy(coefficients) : nothing
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis(
            cfg, coefficients; prototype_θφ=spatial,
            real_output=false, Aminus=optional_minus,
        )
    end

    malformed_spatial_pen = Pencil((cfg.nlat + 1, cfg.nlon), (1,), adapter.comm)
    malformed_spatial = PencilArray{Float64}(undef, malformed_spatial_pen)
    fill!(parent(malformed_spatial), 0)
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis(cfg, coefficients; prototype_θφ=malformed_spatial)
    end
    return nothing
end

adapter = MPIScalarAdapter(MPI.COMM_WORLD)
vector_adapter = MPIVectorAdapter(MPI.COMM_WORLD)
qst_adapter = MPIQSTAdapter(MPI.COMM_WORLD)
if isempty(ARGS) || (!("sphtor_full" in ARGS) && !("qst_full" in ARGS) &&
                     !("operators" in ARGS) && !("rotations" in ARGS) &&
                     !("subgroup_firewall" in ARGS))
# Exercise every mathematical axis without taking their full Cartesian product:
# each Pencil construction owns an MPI Cartesian communicator, and a giant
# product needlessly exhausts MPI implementations with a 2048-context limit.
run_scalar_full_parity(
    adapter;
    grid_kinds=_SCALAR_GRID_KINDS,
    precisions=(Float32, Float64),
    mres_values=(1, 2),
    norms=(:orthonormal,),
    real_norm_values=(false,),
    cs_phase_values=(true,),
    pole_orders=(false,),
)
run_scalar_full_parity(
    adapter;
    grid_kinds=(:gauss,),
    precisions=(Float64,),
    mres_values=(1,),
    norms=(:orthonormal, :fourpi, :schmidt),
    real_norm_values=(false, true),
    cs_phase_values=(false, true),
    pole_orders=(false, true),
)
@testset "scalar full-grid parity φ-split complex Float32" begin
    test_phi_split_complex_float32(adapter)
end
@testset "Pencil-native scalar path" begin
    test_pencil_native_path(adapter)
end
@testset "collective scalar validation" begin
    test_collective_validation(adapter)
end

@testset "distributed scalar variant parity" begin
    cfg = create_gauss_config(
        5, 8; nlon=14, mres=2, norm=:schmidt,
        real_norm=true, cs_phase=false,
    )
    coefficients = _variant_coefficients(cfg, Float64)
    packed = SHTnsKit.pack_lm(cfg, coefficients)
    field = reshape(synthesis_packed(cfg, packed), cfg.nlat, cfg.nlon)
    spatial = place(adapter, cfg, field, :spatial)

    @testset "same-name packed and truncated Pencil APIs" begin
        packed_pencil = analysis_packed(cfg, spatial)
        @test packed_pencil isa PencilArray
        @test size_global(packed_pencil) == (cfg.nlm, 1)
        @test _collect_distributed_vector(packed_pencil) ≈ packed atol=3e-11 rtol=3e-11
        packed_field = synthesis_packed(
            cfg, packed_pencil; prototype_θφ=spatial,
        )
        @test packed_field isa PencilArray
        @test _collect_spatial(packed_field, cfg) ≈ field atol=3e-11 rtol=3e-11

        ltr_same_name = 3
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        extension._reset_pencil_scalar_stats!()
        truncated_pencil = analysis_packed_l(cfg, spatial, ltr_same_name)
        @test truncated_pencil isa PencilArray
        @test _collect_distributed_vector(truncated_pencil) ≈
              analysis_packed_l(cfg, vec(field), ltr_same_name) atol=3e-11 rtol=3e-11
        spectral_local = PencilArrays.range_local(
            create_spectral_pencil(cfg; comm=adapter.comm),
        )
        local_active_count = count((
            (l + 1 in spectral_local[1]) && (m + 1 in spectral_local[2])
            for m in 0:cfg.mres:min(cfg.mmax, ltr_same_name)
            for l in m:ltr_same_name
        ))
        packed_analysis_stats = extension._pencil_scalar_stats()
        @test packed_analysis_stats.analysis_packed_sent_elements == local_active_count
        @test packed_analysis_stats.analysis_packed_max_message_elements <= local_active_count
        noisy_pencil = _place_distributed_vector(copy(packed), adapter.comm)
        noisy_globals = collect(Int, PencilArrays.range_local(pencil(noisy_pencil))[1])
        for (i, packed_index) in pairs(noisy_globals)
            for m in 0:cfg.mres:cfg.mmax, l in max(m, ltr_same_name + 1):cfg.lmax
                if packed_index == LM_index(cfg.lmax, cfg.mres, l, m) + 1
                    parent(noisy_pencil)[i, 1] = 100 + 20im
                end
            end
        end
        extension._reset_pencil_scalar_stats!()
        low_field = synthesis_packed_l(
            cfg, noisy_pencil, ltr_same_name; prototype_θφ=spatial,
        )
        @test low_field isa PencilArray
        @test vec(_collect_spatial(low_field, cfg)) ≈
              synthesis_packed_l(cfg, packed, ltr_same_name) atol=3e-11 rtol=3e-11
        local_synthesis_active = count(noisy_globals) do packed_index
            any(packed_index == LM_index(cfg.lmax, cfg.mres, l, m) + 1
                for m in 0:cfg.mres:min(cfg.mmax, ltr_same_name)
                for l in m:ltr_same_name)
        end
        packed_synthesis_stats = extension._pencil_scalar_stats()
        @test packed_synthesis_stats.synthesis_packed_sent_elements ==
              local_synthesis_active
        @test packed_synthesis_stats.synthesis_packed_max_message_elements <=
              local_synthesis_active

        # Exercise the same packed owner maps with a genuinely two-dimensional
        # spatial decomposition and both supported precisions.
        packed_2d_pen = Pencil(
            (cfg.nlat, cfg.nlon), (1, 2), adapter.comm,
        )
        packed_2d_ranges = PencilArrays.range_local(packed_2d_pen)
        for T in (Float32, Float64)
            tol = T === Float32 ? 3f-4 : 5e-11
            spatial_2d = PencilArray{T}(undef, packed_2d_pen)
            for (j, jglobal) in pairs(packed_2d_ranges[2]),
                (i, iglobal) in pairs(packed_2d_ranges[1])
                parent(spatial_2d)[i, j] = T(field[iglobal, jglobal])
            end
            extension._reset_pencil_scalar_stats!()
            packed_2d = analysis_packed_l(cfg, spatial_2d, ltr_same_name)
            @test _collect_distributed_vector(packed_2d) ≈
                  analysis_packed_l(cfg, T.(vec(field)), ltr_same_name) atol=tol rtol=tol
            stats_2d_analysis = extension._pencil_scalar_stats()
            @test stats_2d_analysis.analysis_packed_sent_elements == local_active_count
            @test stats_2d_analysis.analysis_packed_max_message_elements <=
                  local_active_count
            extension._reset_pencil_scalar_stats!()
            reconstructed_2d = synthesis_packed_l(
                cfg, packed_2d, ltr_same_name; prototype_θφ=spatial_2d,
            )
            @test _collect_spatial(reconstructed_2d, cfg) ≈
                  reshape(
                      synthesis_packed_l(
                          cfg, analysis_packed_l(cfg, T.(vec(field)), ltr_same_name),
                          ltr_same_name,
                      ),
                      cfg.nlat, cfg.nlon,
                  ) atol=tol rtol=tol
            packed_2d_globals = collect(
                Int, PencilArrays.range_local(pencil(packed_2d))[1],
            )
            local_packed_2d = count(packed_2d_globals) do packed_index
                any(packed_index == LM_index(cfg.lmax, cfg.mres, l, m) + 1
                    for m in 0:cfg.mres:min(cfg.mmax, ltr_same_name)
                    for l in m:ltr_same_name)
            end
            stats_2d_synthesis = extension._pencil_scalar_stats()
            @test stats_2d_synthesis.synthesis_packed_sent_elements == local_packed_2d
            @test stats_2d_synthesis.synthesis_packed_max_message_elements <=
                  local_packed_2d
        end
    end

    @testset "same-name complex packed degree limits" begin
        for T in (Float32, Float64)
            tol = T === Float32 ? 4f-4 : 5e-11
            complex_cfg = create_gauss_config(
                5, 8; nlon=14, norm=:schmidt,
                real_norm=true, cs_phase=false,
            )
            complex_coefficients = zeros(
                Complex{T},
                nlm_cplx_calc(complex_cfg.lmax, complex_cfg.mmax, 1),
            )
            for l in 0:complex_cfg.lmax,
                m in -min(l, complex_cfg.mmax):min(l, complex_cfg.mmax)
                complex_coefficients[
                    LM_cplx_index(complex_cfg.lmax, complex_cfg.mmax, l, m) + 1
                ] = Complex{T}(T(0.03 * (l + 1) - 0.01m), T(0.02m - 0.01l))
            end
            complex_field = synthesis_packed_cplx(complex_cfg, complex_coefficients)
            complex_spatial = place(
                adapter, complex_cfg, complex_field, :spatial,
            )
            ltr_complex = 3
            truncated = copy(complex_coefficients)
            noisy = copy(complex_coefficients)
            for l in (ltr_complex + 1):complex_cfg.lmax,
                m in -min(l, complex_cfg.mmax):min(l, complex_cfg.mmax)
                index = LM_cplx_index(
                    complex_cfg.lmax, complex_cfg.mmax, l, m,
                ) + 1
                truncated[index] = 0
                noisy[index] = Complex{T}(T(90 + l), T(-70 + m))
            end

            extension._reset_pencil_scalar_stats!()
            analyzed = analysis_packed_cplx_l(
                complex_cfg, complex_spatial, ltr_complex,
            )
            @test analyzed isa PencilArray
            @test _collect_distributed_vector(analyzed) ≈ truncated atol=tol rtol=tol
            complex_spectral_local = PencilArrays.range_local(
                create_spectral_pencil(complex_cfg; comm=adapter.comm),
            )
            local_active = count((
                (l + 1 in complex_spectral_local[1]) &&
                (abs(m) + 1 in complex_spectral_local[2])
                for l in 0:ltr_complex
                for m in -min(l, complex_cfg.mmax):min(l, complex_cfg.mmax)
            ))
            complex_analysis_stats = extension._pencil_scalar_stats()
            @test complex_analysis_stats.analysis_packed_sent_elements == local_active
            @test complex_analysis_stats.analysis_packed_max_message_elements <= local_active

            noisy_pencil = _place_distributed_vector(noisy, adapter.comm)
            extension._reset_pencil_scalar_stats!()
            reconstructed = synthesis_packed_cplx_l(
                complex_cfg, noisy_pencil, ltr_complex;
                prototype_θφ=complex_spatial,
            )
            @test reconstructed isa PencilArray
            @test _collect_spatial(reconstructed, complex_cfg) ≈
                  synthesis_packed_cplx(complex_cfg, truncated) atol=tol rtol=tol
            noisy_complex_globals = collect(
                Int, PencilArrays.range_local(pencil(noisy_pencil))[1],
            )
            local_unpack_active = count(noisy_complex_globals) do packed_index
                any(packed_index == LM_cplx_index(
                        complex_cfg.lmax, complex_cfg.mmax, l, m,
                    ) + 1
                    for l in 0:ltr_complex
                    for m in -min(l, complex_cfg.mmax):min(l, complex_cfg.mmax))
            end
            complex_synthesis_stats = extension._pencil_scalar_stats()
            @test complex_synthesis_stats.synthesis_packed_sent_elements ==
                  local_unpack_active
            @test complex_synthesis_stats.synthesis_packed_max_message_elements <=
                  local_unpack_active
            @test complex_synthesis_stats.synthesis_packed_max_message_elements <
                  2nlm_cplx_calc(complex_cfg.lmax, complex_cfg.mmax, 1)

            # At ltr=0 the sole active entry is (l,m)=(0,0), so an exact
            # complex-packed unpack sends one value, not a padded ±m pair.
            for edge_ltr in (0, complex_cfg.lmax)
                extension._reset_pencil_scalar_stats!()
                edge_reconstructed = synthesis_packed_cplx_l(
                    complex_cfg, noisy_pencil, edge_ltr;
                    prototype_θφ=complex_spatial,
                )
                @test _collect_spatial(edge_reconstructed, complex_cfg) ≈
                      synthesis_packed_cplx_l(
                          complex_cfg, noisy, edge_ltr,
                      ) atol=tol rtol=tol
                edge_local_payload = count(noisy_complex_globals) do packed_index
                    any(packed_index == LM_cplx_index(
                            complex_cfg.lmax, complex_cfg.mmax, l, m,
                        ) + 1
                        for l in 0:edge_ltr
                        for m in -min(l, complex_cfg.mmax):min(l, complex_cfg.mmax))
                end
                edge_stats = extension._pencil_scalar_stats()
                @test edge_stats.synthesis_packed_sent_elements == edge_local_payload
                @test edge_stats.synthesis_packed_max_message_elements <=
                      edge_local_payload
            end

            complex_2d_pen = Pencil(
                (complex_cfg.nlat, complex_cfg.nlon), (1, 2), adapter.comm,
            )
            complex_2d = PencilArray{Complex{T}}(undef, complex_2d_pen)
            complex_2d_ranges = PencilArrays.range_local(complex_2d_pen)
            for (j, jglobal) in pairs(complex_2d_ranges[2]),
                (i, iglobal) in pairs(complex_2d_ranges[1])
                parent(complex_2d)[i, j] = complex_field[iglobal, jglobal]
            end
            extension._reset_pencil_scalar_stats!()
            analyzed_2d = analysis_packed_cplx_l(
                complex_cfg, complex_2d, ltr_complex,
            )
            @test _collect_distributed_vector(analyzed_2d) ≈ truncated atol=tol rtol=tol
            complex_2d_analysis_stats = extension._pencil_scalar_stats()
            @test complex_2d_analysis_stats.analysis_packed_sent_elements == local_active
            @test complex_2d_analysis_stats.analysis_packed_max_message_elements <=
                  local_active
            analyzed_2d_globals = collect(
                Int, PencilArrays.range_local(pencil(analyzed_2d))[1],
            )
            local_complex_2d_unpack = count(analyzed_2d_globals) do packed_index
                any(packed_index == LM_cplx_index(
                        complex_cfg.lmax, complex_cfg.mmax, l, m,
                    ) + 1
                    for l in 0:ltr_complex
                    for m in -min(l, complex_cfg.mmax):min(l, complex_cfg.mmax))
            end
            extension._reset_pencil_scalar_stats!()
            reconstructed_2d = synthesis_packed_cplx_l(
                complex_cfg, analyzed_2d, ltr_complex; prototype_θφ=complex_2d,
            )
            @test _collect_spatial(reconstructed_2d, complex_cfg) ≈
                  synthesis_packed_cplx(complex_cfg, truncated) atol=tol rtol=tol
            complex_2d_synthesis_stats = extension._pencil_scalar_stats()
            @test complex_2d_synthesis_stats.synthesis_packed_sent_elements ==
                  local_complex_2d_unpack
            @test complex_2d_synthesis_stats.synthesis_packed_max_message_elements <=
                  local_complex_2d_unpack

            rank = MPI.Comm_rank(adapter.comm)
            varying_ltr = rank == 0 ? ltr_complex : ltr_complex - 1
            @test _all_ranks_catch(adapter.comm) do
                analysis_packed_cplx_l(complex_cfg, complex_spatial, varying_ltr)
            end
        end
    end

    @testset "same-name axisymmetric and fixed-order Pencil APIs" begin
        axis_coefficients = ComplexF32[0.2, -0.1, 0.05, 0.03, -0.02, 0.01]
        axis_field = synthesis_axisym(cfg, axis_coefficients)
        axis_spatial = _place_distributed_vector(axis_field, adapter.comm)
        axis_analyzed = analysis_axisym(cfg, axis_spatial)
        @test axis_analyzed isa PencilArray
        @test _collect_distributed_vector(axis_analyzed) ≈ axis_coefficients atol=3f-5 rtol=3f-5
        axis_reconstructed = synthesis_axisym(cfg, axis_analyzed)
        @test axis_reconstructed isa PencilArray
        @test _collect_distributed_vector(axis_reconstructed) ≈ axis_field atol=3f-5 rtol=3f-5

        axis_l = analysis_axisym_l(cfg, axis_spatial, 3)
        @test size_global(axis_l) == (4, 1)
        @test _collect_distributed_vector(axis_l) ≈ axis_coefficients[1:4] atol=3f-5 rtol=3f-5
        axis_low = synthesis_axisym_l(cfg, axis_l, 3)
        @test _collect_distributed_vector(axis_low) ≈
              synthesis_axisym_l(cfg, axis_coefficients, 3) atol=3f-5 rtol=3f-5
        noisy_axis_full = copy(axis_analyzed)
        noisy_axis_globals = collect(
            Int, PencilArrays.range_local(pencil(noisy_axis_full))[1],
        )
        for (local_index, global_index) in pairs(noisy_axis_globals)
            global_index > 4 || continue
            parent(noisy_axis_full)[local_index, 1] = ComplexF32(90, -40)
        end
        axis_full_prefix = synthesis_axisym_l(cfg, noisy_axis_full, 3)
        @test _collect_distributed_vector(axis_full_prefix) ≈
              synthesis_axisym_l(cfg, axis_coefficients, 3) atol=3f-5 rtol=3f-5

        for im in 0:(cfg.mmax ÷ cfg.mres)
            m = im * cfg.mres
            mode_coefficients = ComplexF32[
                ComplexF32(0.03f0 * (l + 1), m == 0 ? 0 : -0.02f0 * (l + 1))
                for l in m:cfg.lmax
            ]
            mode_field = synthesis_packed_ml(cfg, im, mode_coefficients, cfg.lmax)
            mode_spatial = _place_distributed_vector(mode_field, adapter.comm)
            mode_analyzed = analysis_packed_ml(
                cfg, im, mode_spatial, cfg.lmax,
            )
            @test mode_analyzed isa PencilArray
            @test _collect_distributed_vector(mode_analyzed) ≈
                  mode_coefficients atol=3f-5 rtol=3f-5
            mode_reconstructed = synthesis_packed_ml(
                cfg, im, mode_analyzed, cfg.lmax,
            )
            @test mode_reconstructed isa PencilArray
            @test _collect_distributed_vector(mode_reconstructed) ≈
                  mode_field atol=3f-5 rtol=3f-5
        end
    end

    @testset "same-name batch Pencil APIs" begin
        for T in (Float32, Float64), nfields in (1, 2, 5)
            tol = T === Float32 ? 3f-4 : 5e-11
            coefficients_t = Complex{T}.(coefficients)
            field_t = T.(field)
            fields = PencilArray{T}(undef, pencil(spatial), nfields)
            for k in 1:nfields
                @views parent(fields)[:, :, k] .= T(k) .* parent(spatial)
            end
            analyzed = analysis_batch(cfg, fields)
            @test analyzed isa PencilArray
            @test size_global(analyzed) ==
                  (cfg.lmax + 1, cfg.mmax + 1, nfields)
            analyzed_dense = zeros(Complex{T}, cfg.lmax + 1, cfg.mmax + 1, nfields)
            ranges = PencilArrays.range_local(pencil(analyzed))
            lglobals = collect(Int, ranges[1])
            mglobals = collect(Int, ranges[2])
            for k in 1:nfields, (j, mg) in pairs(mglobals), (i, lg) in pairs(lglobals)
                analyzed_dense[lg, mg, k] = parent(analyzed)[i, j, k]
            end
            MPI.Allreduce!(analyzed_dense, +, adapter.comm)
            for k in 1:nfields
                @test analyzed_dense[:, :, k] ≈ T(k) .* coefficients_t atol=tol rtol=tol
            end
            analyzed_inplace = PencilArray{Complex{T}}(
                undef, pencil(analyzed), nfields,
            )
            @test analysis_batch!(cfg, analyzed_inplace, fields) === analyzed_inplace
            @test parent(analyzed_inplace) ≈ parent(analyzed) atol=tol rtol=tol
            reconstructed = synthesis_batch(
                cfg, analyzed; prototype_θφ=fields,
            )
            @test reconstructed isa PencilArray
            @test size_global(reconstructed) == (cfg.nlat, cfg.nlon, nfields)
            for k in 1:nfields
                slice = PencilArray{T}(undef, pencil(spatial))
                @views parent(slice) .= parent(reconstructed)[:, :, k]
                @test _collect_spatial(slice, cfg) ≈ T(k) .* field_t atol=tol rtol=tol
            end
            reconstructed_inplace = PencilArray{T}(
                undef, pencil(fields), nfields,
            )
            @test synthesis_batch!(
                cfg, reconstructed_inplace, analyzed;
                prototype_θφ=fields,
            ) === reconstructed_inplace
            @test parent(reconstructed_inplace) ≈
                  parent(reconstructed) atol=tol rtol=tol

            complex_batch = synthesis_batch_cplx(
                cfg, analyzed; prototype_θφ=fields,
            )
            @test complex_batch isa PencilArray
            complex_reference = synthesis_batch_cplx(cfg, analyzed_dense)
            for k in 1:nfields
                slice = PencilArray{Complex{T}}(undef, pencil(spatial))
                @views parent(slice) .= parent(complex_batch)[:, :, k]
                @test _collect_spatial(slice, cfg) ≈
                      complex_reference[:, :, k] atol=tol rtol=tol
            end
        end
    end

    @testset "distributed variant plans and collective validation" begin
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        rank = MPI.Comm_rank(adapter.comm)
        plan = extension.DistAnalysisPlan(cfg, spatial)
        planned = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        dist_analysis!(plan, planned, spatial)
        @test planned ≈ coefficients atol=3e-11 rtol=3e-11

        # Equal local lengths do not make different Pencil ownership layouts
        # interchangeable. These 1D and 2D decompositions both own 16 spatial
        # values per rank (and 4 spectral values), but their global ranges and
        # process topologies differ.
        layout_cfg = create_gauss_config(3, 8; nlon=8)
        layout_1d = Pencil((layout_cfg.nlat, layout_cfg.nlon), (1,), adapter.comm)
        layout_2d = Pencil((layout_cfg.nlat, layout_cfg.nlon), (1, 2), adapter.comm)
        layout_field = PencilArray{Float64}(undef, layout_1d)
        wrong_layout_field = PencilArray{Float64}(undef, layout_2d)
        fill!(parent(layout_field), 0.25)
        fill!(parent(wrong_layout_field), 0.25)
        @test length(parent(layout_field)) == length(parent(wrong_layout_field))

        layout_analysis_plan = extension.DistAnalysisPlan(layout_cfg, layout_field)
        layout_dense = fill(ComplexF64(9, -2), layout_cfg.lmax + 1,
                            layout_cfg.mmax + 1)
        @test _all_ranks_catch(adapter.comm) do
            analysis!(layout_analysis_plan, layout_dense, wrong_layout_field)
        end
        @test all(==(ComplexF64(9, -2)), layout_dense)

        # Dense plan outputs are replicated, so rank-varying element types must
        # be rejected collectively before fallback/FFT/reduction work. A local
        # check would let the valid rank enter transform collectives alone.
        wrong_analysis_type = rank == 0 ?
            fill(ComplexF64(12), layout_cfg.lmax + 1, layout_cfg.mmax + 1) :
            fill(Float64(12), layout_cfg.lmax + 1, layout_cfg.mmax + 1)
        @test _all_ranks_catch(adapter.comm) do
            analysis!(layout_analysis_plan, wrong_analysis_type, layout_field)
        end
        @test all(==(eltype(wrong_analysis_type)(12)), wrong_analysis_type)
        MPI.Barrier(adapter.comm)
        wrong_analysis_shape = rank == 0 ?
            fill(ComplexF64(14), layout_cfg.lmax + 1, layout_cfg.mmax + 1) :
            fill(ComplexF64(14), layout_cfg.lmax + 2, layout_cfg.mmax + 1)
        @test _all_ranks_catch(adapter.comm) do
            analysis!(layout_analysis_plan, wrong_analysis_shape, layout_field)
        end
        @test all(==(ComplexF64(14)), wrong_analysis_shape)
        MPI.Barrier(adapter.comm)

        varying_analysis_kind = if rank == 0
            input = PencilArray{Float64}(undef, layout_1d)
            fill!(parent(input), 0.25)
            input
        else
            input = PencilArray{ComplexF64}(undef, layout_1d)
            fill!(parent(input), ComplexF64(0.25, 0.125))
            input
        end
        fill!(layout_dense, ComplexF64(20, -3))
        @test _all_ranks_catch(adapter.comm) do
            analysis!(layout_analysis_plan, layout_dense, varying_analysis_kind)
        end
        @test all(==(ComplexF64(20, -3)), layout_dense)
        MPI.Barrier(adapter.comm)

        varying_analysis_precision = if rank == 0
            input = PencilArray{Float64}(undef, layout_1d)
            fill!(parent(input), 0.25)
            input
        else
            input = PencilArray{Float32}(undef, layout_1d)
            fill!(parent(input), 0.25f0)
            input
        end
        fill!(layout_dense, ComplexF64(21, -4))
        @test _all_ranks_catch(adapter.comm) do
            analysis!(
                layout_analysis_plan, layout_dense, varying_analysis_precision,
            )
        end
        @test all(==(ComplexF64(21, -4)), layout_dense)
        MPI.Barrier(adapter.comm)
        layout_rfft_plan = extension.DistAnalysisPlan(
            layout_cfg, layout_field; use_rfft=true,
        )
        fill!(layout_dense, ComplexF64(22, -5))
        @test _all_ranks_catch(adapter.comm) do
            analysis!(layout_rfft_plan, layout_dense, varying_analysis_kind)
        end
        @test all(==(ComplexF64(22, -5)), layout_dense)
        MPI.Barrier(adapter.comm)

        layout_spectral = analysis(layout_cfg, layout_field)
        layout_synthesis_plan = extension.DistPlan(layout_cfg, layout_field)
        wrong_layout_output = PencilArray{Float64}(undef, layout_2d)
        fill!(parent(wrong_layout_output), 7.0)
        @test _all_ranks_catch(adapter.comm) do
            synthesis!(layout_synthesis_plan, wrong_layout_output, layout_spectral)
        end
        @test all(==(7.0), parent(wrong_layout_output))

        self_spectral = PencilArray{ComplexF64}(
            undef,
            Pencil(
                (layout_cfg.lmax + 1, layout_cfg.mmax + 1), (2,),
                MPI.COMM_SELF,
            ),
        )
        fill!(parent(self_spectral), ComplexF64(0.5, -0.25))
        varying_coefficient_comm = rank == 0 ? self_spectral : layout_spectral
        comm_check_output = PencilArray{Float64}(undef, layout_1d)
        @test _all_ranks_catch(adapter.comm) do
            extension._validate_synthesis_plan_output!(
                layout_synthesis_plan, comm_check_output,
                varying_coefficient_comm, true,
                :test_synthesis_plan_coefficient_comm,
            )
        end
        MPI.Barrier(adapter.comm)
        fill!(parent(comm_check_output), 23)
        @test _all_ranks_catch(adapter.comm) do
            synthesis!(
                layout_synthesis_plan, comm_check_output,
                varying_coefficient_comm,
            )
        end
        @test all(==(23.0), parent(comm_check_output))
        MPI.Barrier(adapter.comm)

        wrong_synthesis_type = if rank == 0
            output = PencilArray{Float64}(undef, layout_1d)
            fill!(parent(output), 13)
            output
        else
            output = PencilArray{ComplexF64}(undef, layout_1d)
            fill!(parent(output), 13)
            output
        end
        @test _all_ranks_catch(adapter.comm) do
            synthesis!(layout_synthesis_plan, wrong_synthesis_type, layout_spectral)
        end
        @test all(==(eltype(wrong_synthesis_type)(13)), parent(wrong_synthesis_type))
        MPI.Barrier(adapter.comm)
        alternate_wrong_shape = PencilArray{Float64}(
            undef,
            Pencil((layout_cfg.nlat + 1, layout_cfg.nlon), (1,), adapter.comm),
        )
        correct_shape_output = PencilArray{Float64}(undef, layout_1d)
        wrong_synthesis_shape = rank == 0 ? correct_shape_output :
                                                    alternate_wrong_shape
        fill!(parent(wrong_synthesis_shape), 15)
        @test _all_ranks_catch(adapter.comm) do
            synthesis!(layout_synthesis_plan, wrong_synthesis_shape, layout_spectral)
        end
        @test all(==(15.0), parent(wrong_synthesis_shape))
        MPI.Barrier(adapter.comm)

        @test _all_ranks_catch(adapter.comm) do
            analysis(layout_cfg, layout_field; use_rfft=rank == 0)
        end
        MPI.Barrier(adapter.comm)
        @test _all_ranks_catch(adapter.comm) do
            extension.DistAnalysisPlan(
                layout_cfg, layout_field; use_rfft=rank == 0,
            )
        end
        MPI.Barrier(adapter.comm)
        @test _all_ranks_catch(adapter.comm) do
            synthesis(
                layout_cfg, layout_spectral; prototype_θφ=layout_field,
                real_output=rank == 0,
            )
        end
        MPI.Barrier(adapter.comm)
        @test _all_ranks_catch(adapter.comm) do
            extension.DistPlan(
                layout_cfg, layout_field; use_rfft=rank == 0,
            )
        end
        MPI.Barrier(adapter.comm)

        divergent_cfg = deepcopy(layout_cfg)
        if rank == 0
            divergent_cfg.cphi *= 1.25
            divergent_cfg.x[1] += 0.125
        end
        @test _all_ranks_catch(adapter.comm) do
            extension._validate_cfg_replicated(divergent_cfg, adapter.comm)
        end
        MPI.Barrier(adapter.comm)
        divergent_packed_size = deepcopy(layout_cfg)
        # Deliberately corrupt internal metadata: the public setter rejects nlm
        # changes before this test can exercise collective divergence detection.
        rank == 0 && setfield!(divergent_packed_size, :nlm, divergent_packed_size.nlm + 1)
        @test _all_ranks_catch(adapter.comm) do
            extension._validate_cfg_replicated(
                divergent_packed_size, adapter.comm,
            )
        end
        MPI.Barrier(adapter.comm)

        oversized_counts = zeros(Int, MPI.Comm_size(adapter.comm))
        rank == 0 && (oversized_counts[1] = Int(typemax(Cint)) + 1)
        @test _all_ranks_catch(adapter.comm) do
            extension._checked_owner_exchange_counts(
                oversized_counts, adapter.comm, :test_oversized_counts,
            )
        end
        cumulative_counts = fill(
            Int(typemax(Cint)) ÷ MPI.Comm_size(adapter.comm) + 1,
            MPI.Comm_size(adapter.comm),
        )
        @test _all_ranks_catch(adapter.comm) do
            extension._checked_owner_exchange_counts(
                cumulative_counts, adapter.comm, :test_cumulative_counts,
            )
        end
        MPI.Barrier(adapter.comm)

        layout_fields = PencilArray{Float64}(undef, layout_1d, 2)
        fill!(parent(layout_fields), 0.25)
        wrong_spectral_pen = Pencil(
            (layout_cfg.lmax + 1, layout_cfg.mmax + 1), (1, 2), adapter.comm,
        )
        wrong_batch_spectral = PencilArray{ComplexF64}(
            undef, wrong_spectral_pen, 2,
        )
        fill!(parent(wrong_batch_spectral), ComplexF64(6, 1))
        standard_batch_spectral = PencilArray{ComplexF64}(
            undef, create_spectral_pencil(layout_cfg; comm=adapter.comm), 2,
        )
        fill!(parent(standard_batch_spectral), 0)
        @test length(parent(wrong_batch_spectral)) ==
              length(parent(standard_batch_spectral))
        @test _all_ranks_catch(adapter.comm) do
            analysis_batch!(layout_cfg, wrong_batch_spectral, layout_fields)
        end
        @test all(==(ComplexF64(6, 1)), parent(wrong_batch_spectral))

        rank_varying_batch_output = if rank == 0
            output = PencilArray{ComplexF64}(
                undef, create_spectral_pencil(layout_cfg; comm=adapter.comm), 2,
            )
            fill!(parent(output), 16)
            output
        else
            output = PencilArray{Float64}(
                undef, create_spectral_pencil(layout_cfg; comm=adapter.comm), 2,
            )
            fill!(parent(output), 16)
            output
        end
        @test _all_ranks_catch(adapter.comm) do
            analysis_batch!(layout_cfg, rank_varying_batch_output, layout_fields)
        end
        @test all(
            ==(eltype(rank_varying_batch_output)(16)),
            parent(rank_varying_batch_output),
        )
        MPI.Barrier(adapter.comm)

        subgroup = MPI.Comm_split(adapter.comm, rank % 2, rank)
        subgroup_output = PencilArray{ComplexF64}(
            undef,
            Pencil(
                (layout_cfg.lmax + 1, layout_cfg.mmax + 1), (2,), subgroup,
            ),
            2,
        )
        fill!(parent(subgroup_output), 17)
        @test _all_ranks_catch(adapter.comm) do
            analysis_batch!(layout_cfg, subgroup_output, layout_fields)
        end
        @test all(==(ComplexF64(17)), parent(subgroup_output))
        MPI.Barrier(adapter.comm)

        wrong_batch_spatial = PencilArray{Float64}(undef, layout_2d, 2)
        fill!(parent(wrong_batch_spatial), 5.0)
        @test length(parent(wrong_batch_spatial)) == length(parent(layout_fields))
        @test _all_ranks_catch(adapter.comm) do
            synthesis_batch!(
                layout_cfg, wrong_batch_spatial, standard_batch_spectral;
                prototype_θφ=layout_fields,
            )
        end
        @test all(==(5.0), parent(wrong_batch_spatial))
        MPI.Barrier(adapter.comm)
        fill!(planned, 0)
        @test analysis!(plan, planned, spatial) === planned
        @test planned ≈ coefficients atol=3e-11 rtol=3e-11

        spectral = analysis(cfg, spatial)
        synthesis_plan = extension.DistPlan(cfg, spatial)
        planned_spatial = similar(spatial)
        @test synthesis!(synthesis_plan, planned_spatial, spectral) === planned_spatial
        @test _collect_spatial(planned_spatial, cfg) ≈ field atol=3e-11 rtol=3e-11
        for m in 0:cfg.mmax
            m % cfg.mres == 0 && continue
            @test iszero(maximum(abs, @view planned[:, m + 1]))
        end

        for T in (Float32, Float64), use_rfft in (false, true)
            plan_cfg = create_gauss_config(
                4, 7; nlon=12, mres=2, norm=:schmidt,
                real_norm=true, cs_phase=false,
            )
            plan_coefficients = Complex{T}.(_variant_coefficients(plan_cfg, T))
            plan_field = T.(synthesis(plan_cfg, plan_coefficients))
            plan_pencil = Pencil(
                (plan_cfg.nlat, plan_cfg.nlon), (1,), adapter.comm,
            )
            plan_field_distributed = PencilArray{T}(undef, plan_pencil)
            plan_ranges = PencilArrays.range_local(plan_pencil)
            for (j, jglobal) in pairs(plan_ranges[2]),
                (i, iglobal) in pairs(plan_ranges[1])
                parent(plan_field_distributed)[i, j] =
                    plan_field[iglobal, jglobal]
            end
            analysis_plan = extension.DistAnalysisPlan(
                plan_cfg, plan_field_distributed; use_rfft,
            )
            plan_output = zeros(
                Complex{T}, plan_cfg.lmax + 1, plan_cfg.mmax + 1,
            )
            @test analysis!(
                analysis_plan, plan_output, plan_field_distributed,
            ) === plan_output
            tolerance = T === Float32 ? 3f-4 : 5e-11
            @test plan_output ≈ analysis(
                plan_cfg, plan_field,
            ) atol=tolerance rtol=tolerance
            @test eltype(plan_output) === Complex{T}

            plan_spectral = matrix_to_spectral_pencil(
                plan_cfg, plan_coefficients; comm=adapter.comm,
            )
            synthesis_plan_t = extension.DistPlan(
                plan_cfg, plan_field_distributed; use_rfft,
            )
            plan_spatial_output = PencilArray{T}(undef, plan_pencil)
            @test synthesis!(
                synthesis_plan_t, plan_spatial_output, plan_spectral,
            ) === plan_spatial_output
            @test _collect_spatial(plan_spatial_output, plan_cfg) ≈
                  plan_field atol=tolerance rtol=tolerance
        end

        complex_spatial = PencilArray{ComplexF64}(undef, pencil(spatial))
        parent(complex_spatial) .= complex.(parent(spatial), 0.1)
        @test _all_ranks_catch(adapter.comm) do
            analysis_packed(cfg, complex_spatial)
        end

        overflowing_ltr = rank == 0 ? big(typemax(Int)) + 1 : big(3)
        @test _all_ranks_catch(adapter.comm) do
            analysis_packed_l(cfg, spatial, overflowing_ltr)
        end
        axis_field = synthesis_axisym(cfg, coefficients[:, 1])
        axis_spatial = _place_distributed_vector(axis_field, adapter.comm)
        @test _all_ranks_catch(adapter.comm) do
            analysis_axisym_l(cfg, axis_spatial, overflowing_ltr)
        end
        varying_axis_ltr = rank == 0 ? 3 : 2
        @test _all_ranks_catch(adapter.comm) do
            analysis_axisym_l(cfg, axis_spatial, varying_axis_ltr)
        end

        mode_field = synthesis_packed_ml(cfg, 1, coefficients[3:end, 3], cfg.lmax)
        mode_spatial = _place_distributed_vector(mode_field, adapter.comm)
        varying_im = rank == 0 ? 1 : 2
        @test _all_ranks_catch(adapter.comm) do
            analysis_packed_ml(cfg, varying_im, mode_spatial, cfg.lmax)
        end

        varying_fields = PencilArray{Float64}(
            undef, pencil(spatial), rank == 0 ? 1 : 2,
        )
        fill!(parent(varying_fields), 0)
        @test _all_ranks_catch(adapter.comm) do
            analysis_batch(cfg, varying_fields)
        end


        empty_fields = PencilArray{Float64}(undef, pencil(spatial), 0)
        @test _all_ranks_catch(adapter.comm) do
            analysis_batch(cfg, empty_fields)
        end
        empty_coefficients = PencilArray{ComplexF64}(
            undef, create_spectral_pencil(cfg; comm=adapter.comm), 0,
        )
        empty_analysis_output = similar(empty_coefficients)
        empty_synthesis_output = PencilArray{Float64}(
            undef, pencil(spatial), 0,
        )
        @test _all_ranks_catch(adapter.comm) do
            analysis_batch!(cfg, empty_analysis_output, empty_fields)
        end
        @test _all_ranks_catch(adapter.comm) do
            synthesis_batch(
                cfg, empty_coefficients; prototype_θφ=empty_fields,
            )
        end
        @test _all_ranks_catch(adapter.comm) do
            synthesis_batch_cplx(
                cfg, empty_coefficients; prototype_θφ=empty_fields,
            )
        end
        @test _all_ranks_catch(adapter.comm) do
            synthesis_batch!(
                cfg, empty_synthesis_output, empty_coefficients;
                prototype_θφ=empty_fields,
            )
        end

        malformed_pen = Pencil(
            (cfg.nlat + 1, cfg.nlon), (1,), adapter.comm,
        )
        malformed_real = PencilArray{Float64}(undef, malformed_pen)
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(adapter.comm) do
            synthesis_packed(
                cfg, _place_distributed_vector(packed, adapter.comm);
                prototype_θφ=malformed_real,
            )
        end
        MPI.Barrier(adapter.comm)
        malformed_stats = extension._pencil_scalar_stats()
        @test malformed_stats.synthesis_packed_max_message_elements == 0
        @test malformed_stats.synthesis_max_message_elements == 0

        complex_cfg = create_gauss_config(4, 7; nlon=12)
        complex_coefficients = zeros(
            ComplexF32,
            nlm_cplx_calc(complex_cfg.lmax, complex_cfg.mmax, 1),
        )
        complex_coefficients[
            LM_cplx_index(complex_cfg.lmax, complex_cfg.mmax, 2, -1) + 1
        ] = 0.2f0 - 0.1f0im
        distributed_complex = _place_distributed_vector(
            complex_coefficients, adapter.comm,
        )
        malformed_complex_pen = Pencil(
            (complex_cfg.nlat + 1, complex_cfg.nlon), (1,), adapter.comm,
        )
        malformed_complex = PencilArray{ComplexF32}(
            undef, malformed_complex_pen,
        )
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(adapter.comm) do
            synthesis_packed_cplx(
                complex_cfg, distributed_complex;
                prototype_θφ=malformed_complex,
            )
        end
        MPI.Barrier(adapter.comm)
        malformed_complex_stats = extension._pencil_scalar_stats()
        @test malformed_complex_stats.synthesis_packed_max_message_elements == 0
        @test malformed_complex_stats.synthesis_max_message_elements == 0

        wrong_complex_prototype = PencilArray{Float32}(
            undef,
            create_spatial_pencil(complex_cfg; comm=adapter.comm),
        )
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(adapter.comm) do
            synthesis_packed_cplx(
                complex_cfg, distributed_complex;
                prototype_θφ=wrong_complex_prototype,
            )
        end
        MPI.Barrier(adapter.comm)
        wrong_type_stats = extension._pencil_scalar_stats()
        @test wrong_type_stats.synthesis_packed_max_message_elements == 0
        @test wrong_type_stats.synthesis_max_message_elements == 0
    end

    @test dist_analysis_packed(cfg, spatial) ≈ packed atol=3e-11 rtol=3e-11
    local_full = dist_synthesis_packed(cfg, packed; prototype_θφ=spatial)
    full_pencil = PencilArray{eltype(local_full)}(undef, pencil(spatial))
    copyto!(parent(full_pencil), local_full)
    @test _collect_spatial(full_pencil, cfg) ≈ field atol=3e-11 rtol=3e-11

    ltr = 3
    extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
    extension._reset_pencil_scalar_stats!()
    truncated = dist_analysis_packed(cfg, spatial; ltr)
    expected = analysis_packed_l(cfg, vec(field), ltr)
    @test truncated ≈ expected atol=3e-11 rtol=3e-11
    stats = extension._pencil_scalar_stats()
    spectral_pen = create_spectral_pencil(cfg; comm=adapter.comm)
    max_active_owned_orders = MPI.Allreduce(
        count(PencilArrays.range_local(spectral_pen)[2]) do m_index
            m = m_index - 1
            m <= ltr && m % cfg.mres == 0
        end,
        max, adapter.comm,
    )
    @test stats.full_matrix_helper_calls == 0
    @test stats.analysis_max_message_elements <=
          (ltr + 1) * max_active_owned_orders
    active_packed_count = sum(
        ltr - m + 1 for m in 0:cfg.mres:min(cfg.mmax, ltr)
    )
    @test stats.analysis_packed_max_message_elements == active_packed_count
    @test stats.analysis_packed_max_message_elements < cfg.nlm
    for m in 0:cfg.mmax
        m % cfg.mres == 0 || continue
        for l in max(m, ltr + 1):cfg.lmax
            @test iszero(truncated[LM_index(cfg.lmax, cfg.mres, l, m) + 1])
        end
    end

    high_noise = copy(packed)
    for m in 0:cfg.mmax
        m % cfg.mres == 0 || continue
        for l in max(m, ltr + 1):cfg.lmax
            high_noise[LM_index(cfg.lmax, cfg.mres, l, m) + 1] = 100 + 20im
        end
    end
    local_truncated = dist_synthesis_packed(
        cfg, high_noise; prototype_θφ=spatial, ltr,
    )
    truncated_pencil = PencilArray{eltype(local_truncated)}(undef, pencil(spatial))
    copyto!(parent(truncated_pencil), local_truncated)
    @test vec(_collect_spatial(truncated_pencil, cfg)) ≈
          synthesis_packed_l(cfg, high_noise, ltr) atol=3e-11 rtol=3e-11

    @test _all_ranks_catch(adapter.comm) do
        dist_analysis_packed(cfg, spatial; ltr=cfg.lmax + 1)
    end
    rank_ltr = MPI.Comm_rank(adapter.comm) == 0 ? ltr : ltr - 1
    @test _all_ranks_catch(adapter.comm) do
        dist_analysis_packed(cfg, spatial; ltr=rank_ltr)
    end
    invalid_synthesis_ltr = MPI.Comm_rank(adapter.comm) == 0 ? cfg.lmax + 1 : ltr
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis_packed(
            cfg, packed; prototype_θφ=spatial, ltr=invalid_synthesis_ltr,
        )
    end

    complex_cfg = create_gauss_config(3, 6; nlon=10)
    complex_coefficients = zeros(
        ComplexF64, nlm_cplx_calc(complex_cfg.lmax, complex_cfg.mmax, 1),
    )
    complex_coefficients[LM_cplx_index(3, 3, 2, -1) + 1] = 0.2 - 0.1im
    complex_coefficients[LM_cplx_index(3, 3, 3, 2) + 1] = -0.08 + 0.04im
    complex_field = synthesis_packed_cplx(complex_cfg, complex_coefficients)
    complex_spatial = place(adapter, complex_cfg, complex_field, :spatial)
    @test dist_analysis_packed_cplx(complex_cfg, complex_spatial) ≈
          complex_coefficients atol=4e-11 rtol=4e-11
    local_complex = dist_synthesis_packed_cplx(
        complex_cfg, complex_coefficients; prototype_θφ=complex_spatial,
    )
    complex_output = PencilArray{eltype(local_complex)}(undef, pencil(complex_spatial))
    copyto!(parent(complex_output), local_complex)
    @test _collect_spatial(complex_output, complex_cfg) ≈
          complex_field atol=4e-11 rtol=4e-11

    complex_coefficients32 = ComplexF32.(complex_coefficients)
    complex_field32 = synthesis_packed_cplx(complex_cfg, complex_coefficients32)
    complex_spatial32 = place(adapter, complex_cfg, complex_field32, :spatial)
    analyzed32 = dist_analysis_packed_cplx(complex_cfg, complex_spatial32)
    @test eltype(analyzed32) === ComplexF32
    @test analyzed32 ≈ complex_coefficients32 atol=3f-5 rtol=3f-5
    local_complex32 = dist_synthesis_packed_cplx(
        complex_cfg, complex_coefficients32; prototype_θφ=complex_spatial32,
    )
    @test eltype(local_complex32) === ComplexF32

    distributed_complex = analysis_packed_cplx(complex_cfg, complex_spatial32)
    @test distributed_complex isa PencilArray
    @test eltype(distributed_complex) === ComplexF32
    @test _collect_distributed_vector(distributed_complex) ≈
          complex_coefficients32 atol=3f-5 rtol=3f-5
    reconstructed_complex = synthesis_packed_cplx(
        complex_cfg, distributed_complex; prototype_θφ=complex_spatial32,
    )
    @test reconstructed_complex isa PencilArray
    @test _collect_spatial(reconstructed_complex, complex_cfg) ≈
          complex_field32 atol=3f-5 rtol=3f-5

    rank_coefficients = MPI.Comm_rank(adapter.comm) == 0 ?
        complex_coefficients[1:(end - 1)] : complex_coefficients
    @test _all_ranks_catch(adapter.comm) do
        dist_synthesis_packed_cplx(
            complex_cfg, rank_coefficients; prototype_θφ=complex_spatial,
        )
    end
    rank_complex_cfg = MPI.Comm_rank(adapter.comm) == 0 ?
        create_gauss_config(3, 6; nlon=10, mres=2) : complex_cfg
    @test _all_ranks_catch(adapter.comm) do
        dist_analysis_packed_cplx(rank_complex_cfg, complex_spatial)
    end
end
end # scalar/full-variant runner selection

if isempty(ARGS) || "sphtor_full" in ARGS
    run_sphtor_full_parity(
        vector_adapter;
        grid_kinds=_VECTOR_GRID_KINDS,
        precisions=(Float32, Float64),
        mres_values=(1, 2),
        norms=(:orthonormal,), real_norm_values=(false,),
        cs_phase_values=(true,), robert_values=(false, true),
        pole_orders=(false,),
    )
    run_sphtor_full_parity(
        vector_adapter;
        grid_kinds=(:gauss,), precisions=(Float64,), mres_values=(1,),
        norms=(:orthonormal, :fourpi, :schmidt),
        real_norm_values=(false, true), cs_phase_values=(false, true),
        robert_values=(false,), pole_orders=(false, true),
    )

    @testset "Pencil-native vector path and collective validation" begin
        cfg = _vector_config(:gauss, 3, 8; norm=:schmidt,
                             real_norm=true, cs_phase=false)
        S, Tlm = _vector_modes(cfg, Float32)
        Vt, Vp = _direct_low_vector(cfg, S, Tlm; real_output=true)
        Vtd = vector_place(vector_adapter, cfg, Vt, :spatial)
        Vpd = vector_place(vector_adapter, cfg, Vp, :spatial)
        Sd = vector_place(vector_adapter, cfg,
                          _external_vector_coefficients(cfg, S), :spectral)
        Td = vector_place(vector_adapter, cfg,
                          _external_vector_coefficients(cfg, Tlm), :spectral)
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        extension._reset_pencil_scalar_stats!()
        Sa, Ta = analysis_sphtor(cfg, Vtd, Vpd)
        @test Sa isa PencilArray
        @test Ta isa PencilArray
        @test eltype(Sa) === ComplexF32
        @test extension._pencil_scalar_stats().full_matrix_helper_calls == 0
        extension._reset_pencil_scalar_stats!()
        Vr = synthesis_sphtor(cfg, Sd, Td; prototype_θφ=Vtd)
        @test Vr[1] isa PencilArray
        @test Vr[2] isa PencilArray
        @test eltype(Vr[1]) === Float32
        @test extension._pencil_scalar_stats().full_matrix_helper_calls == 0

        @test isdefined(SHTnsKit, :dist_analysis_sphtor)
        @test isdefined(SHTnsKit, :dist_synthesis_sphtor)
        @test isdefined(SHTnsKit, :DistSphtorPlan)
        dense_S, dense_T = dist_analysis_sphtor(cfg, Vtd, Vpd)
        @test dense_S isa Matrix{ComplexF32}
        @test dense_T isa Matrix{ComplexF32}
        @test dense_S ≈ _external_vector_coefficients(cfg, S) atol=4f-4 rtol=4f-4
        @test dense_T ≈ _external_vector_coefficients(cfg, Tlm) atol=4f-4 rtol=4f-4
        dense_Vt, dense_Vp = dist_synthesis_sphtor(
            cfg, dense_S, dense_T; prototype_θφ=Vtd,
        )
        @test dense_Vt isa Matrix{Float32}
        @test dense_Vp isa Matrix{Float32}
        dense_Vtd = PencilArray{Float32}(undef, pencil(Vtd))
        dense_Vpd = PencilArray{Float32}(undef, pencil(Vpd))
        copyto!(parent(dense_Vtd), dense_Vt)
        copyto!(parent(dense_Vpd), dense_Vp)
        @test _collect_spatial(dense_Vtd, cfg) ≈ Vt atol=4f-4 rtol=4f-4
        @test _collect_spatial(dense_Vpd, cfg) ≈ Vp atol=4f-4 rtol=4f-4

        plan = SHTnsKit.DistSphtorPlan(cfg, Vtd)
        @test eltype(plan.Ftθm) === ComplexF32
        @test eltype(plan.Fpθm) === ComplexF32
        @test eltype(plan.Slm_work) === ComplexF32
        @test eltype(plan.Tlm_work) === ComplexF32
        plan_S = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
        plan_T = similar(plan_S)
        analysis_sphtor!(plan, plan_S, plan_T, Vtd, Vpd)
        @test plan_S ≈ dense_S atol=4f-4 rtol=4f-4
        @test plan_T ≈ dense_T atol=4f-4 rtol=4f-4
        compat_S = similar(plan_S); compat_T = similar(plan_T)
        dist_analysis_sphtor!(plan, compat_S, compat_T, Vtd, Vpd)
        @test compat_S == plan_S
        @test compat_T == plan_T
        plan_Vt = PencilArray{Float32}(undef, pencil(Vtd))
        plan_Vp = PencilArray{Float32}(undef, pencil(Vpd))
        synthesis_sphtor!(
            plan, plan_Vt, plan_Vp, plan_S, plan_T; real_output=true,
        )
        @test _collect_spatial(plan_Vt, cfg) ≈ Vt atol=4f-4 rtol=4f-4
        @test _collect_spatial(plan_Vp, cfg) ≈ Vp atol=4f-4 rtol=4f-4
        compat_Vt = PencilArray{Float32}(undef, pencil(Vtd))
        compat_Vp = PencilArray{Float32}(undef, pencil(Vpd))
        dist_synthesis_sphtor!(
            plan, compat_Vt, compat_Vp, plan_S, plan_T; real_output=true,
        )
        @test parent(compat_Vt) == parent(plan_Vt)
        @test parent(compat_Vp) == parent(plan_Vp)

        scratch_plan = SHTnsKit.DistSphtorPlan(
            cfg, Vtd; with_spatial_scratch=true,
        )
        @test eltype(scratch_plan.spatial_scratch.Fθ) === ComplexF32
        @test eltype(scratch_plan.spatial_scratch.Fφ) === ComplexF32
        @test eltype(scratch_plan.spatial_scratch.Vtθ) === Float32
        @test eltype(scratch_plan.spatial_scratch.Vpθ) === Float32
        scratch_Vt = PencilArray{Float32}(undef, pencil(Vtd))
        scratch_Vp = PencilArray{Float32}(undef, pencil(Vpd))
        dist_synthesis_sphtor!(
            scratch_plan, scratch_Vt, scratch_Vp, dense_S, dense_T;
            real_output=true,
        )
        @test _collect_spatial(scratch_Vt, cfg) ≈ Vt atol=4f-4 rtol=4f-4
        @test _collect_spatial(scratch_Vp, cfg) ≈ Vp atol=4f-4 rtol=4f-4

        mres_cfg = _vector_config(
            :gauss, 3, 8; mres=2, norm=:schmidt,
            real_norm=true, cs_phase=false,
        )
        mres_plan = SHTnsKit.DistSphtorPlan(
            mres_cfg, Vtd; with_spatial_scratch=true,
        )
        unsupported = zeros(ComplexF32, mres_cfg.lmax + 1, mres_cfg.mmax + 1)
        unsupported[2, 2] = 0.3f0 - 0.2f0im
        zero_modes = zero(unsupported)
        dist_synthesis_sphtor!(
            mres_plan, scratch_Vt, scratch_Vp, unsupported, zero_modes;
            real_output=true,
        )
        @test all(iszero, _collect_spatial(scratch_Vt, mres_cfg))
        @test all(iszero, _collect_spatial(scratch_Vp, mres_cfg))

        rank_bad_S = MPI.Comm_rank(vector_adapter.comm) == 0 ?
            @view(dense_S[1:(end - 1), :]) : dense_S
        @test _all_ranks_catch(vector_adapter.comm) do
            dist_synthesis_sphtor!(
                scratch_plan, scratch_Vt, scratch_Vp, rank_bad_S, dense_T;
                real_output=true,
            )
        end
        MPI.Barrier(vector_adapter.comm)

        rank = MPI.Comm_rank(vector_adapter.comm)
        # Construct the layout collectively on every rank, then vary only the
        # local element type.  Constructing different Pencil shapes on
        # different ranks would itself enter mismatched communicator
        # collectives before the transform validation has a chance to run.
        bad = rank == 0 ?
            PencilArray{Float64}(undef, pencil(Vpd)) :
            PencilArray{Float32}(undef, pencil(Vpd))
        fill!(parent(bad), 0)
        @test _all_ranks_catch(vector_adapter.comm) do
            analysis_sphtor(cfg, Vtd, bad)
        end

        @test _all_ranks_catch(
            vector_adapter.comm;
            message_contains="invalid or rank-varying use_tables",
        ) do
            dist_analysis_sphtor(cfg, Vtd, Vpd; use_tables=:invalid)
        end
        MPI.Barrier(vector_adapter.comm)

        rank_tables = iszero(rank % 2)
        @test _all_ranks_catch(
            vector_adapter.comm;
            message_contains="invalid or rank-varying use_tables",
        ) do
            dist_analysis_sphtor(cfg, Vtd, Vpd; use_tables=rank_tables)
        end
        MPI.Barrier(vector_adapter.comm)

        rank_return_pencil = iszero(rank % 2)
        @test _all_ranks_catch(
            vector_adapter.comm;
            message_contains="rank-varying return_pencil",
        ) do
            analysis_sphtor(
                cfg, Vtd, Vpd; return_pencil=rank_return_pencil,
            )
        end
        MPI.Barrier(vector_adapter.comm)
    end
end


if isempty(ARGS) || "qst_full" in ARGS
    run_qst_full_parity(
        qst_adapter;
        grid_kinds=_VECTOR_GRID_KINDS,
        precisions=(Float32, Float64),
        mres_values=(1, 2),
        norms=(:orthonormal,), real_norm_values=(false,),
        cs_phase_values=(true,), robert_values=(false, true),
        pole_orders=(false,),
    )
    run_qst_full_parity(
        qst_adapter;
        grid_kinds=(:gauss,), precisions=(Float64,), mres_values=(1,),
        norms=(:orthonormal, :fourpi, :schmidt),
        real_norm_values=(false, true), cs_phase_values=(false, true),
        robert_values=(false,), pole_orders=(false, true),
    )

    @testset "Pencil-native QST path and compatibility" begin
        cfg = _vector_config(
            :gauss, 3, 8; norm=:schmidt, real_norm=true, cs_phase=false,
        )
        Qcan, Scan, Tcan = _qst_modes(cfg, Float32)
        Q = _qst_external(cfg, Qcan)
        S = _qst_external(cfg, Scan)
        Tlm = _qst_external(cfg, Tcan)
        Vr, Vt, Vp = _qst_references(
            cfg, Q, Scan, Tcan; real_output=true,
        )
        Vrd = qst_place(qst_adapter, cfg, Vr, :spatial)
        Vtd = qst_place(qst_adapter, cfg, Vt, :spatial)
        Vpd = qst_place(qst_adapter, cfg, Vp, :spatial)
        Qd = qst_place(qst_adapter, cfg, Q, :spectral)
        Sd = qst_place(qst_adapter, cfg, S, :spectral)
        Td = qst_place(qst_adapter, cfg, Tlm, :spectral)
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)

        extension._reset_pencil_scalar_stats!()
        analyzed = analysis_qst(cfg, Vrd, Vtd, Vpd)
        @test all(value -> value isa PencilArray, analyzed)
        @test all(value -> eltype(value) === ComplexF32, analyzed)
        @test extension._pencil_scalar_stats().full_matrix_helper_calls == 0
        extension._reset_pencil_scalar_stats!()
        synthesized = synthesis_qst(
            cfg, Qd, Sd, Td; prototype_θφ=Vrd,
        )
        @test all(value -> value isa PencilArray, synthesized)
        @test all(value -> eltype(value) === Float32, synthesized)
        @test extension._pencil_scalar_stats().full_matrix_helper_calls == 0

        dense = dist_analysis_qst(cfg, Vrd, Vtd, Vpd)
        @test dense[1] isa Matrix{ComplexF32}
        @test dense[1] ≈ Q atol=4f-4 rtol=4f-4
        @test dense[2] ≈ S atol=4f-4 rtol=4f-4
        @test dense[3] ≈ Tlm atol=4f-4 rtol=4f-4
        local_fields = dist_synthesis_qst(
            cfg, dense...; prototype_θφ=Vrd,
        )
        @test all(value -> value isa AbstractMatrix, local_fields)

        @test isdefined(SHTnsKit, :DistQstPlan)
        plan = SHTnsKit.DistQstPlan(
            cfg, Vrd; with_spatial_scratch=true,
        )
        @test eltype(plan.scalar_plan.Alm_work) === ComplexF32
        @test eltype(plan.sphtor_plan.Slm_work) === ComplexF32
        Qout = zeros(ComplexF32, size(Q)); Sout = similar(Qout); Tout = similar(Qout)
        analysis_qst!(plan, Qout, Sout, Tout, Vrd, Vtd, Vpd)
        @test Qout ≈ Q atol=4f-4 rtol=4f-4
        @test Sout ≈ S atol=4f-4 rtol=4f-4
        @test Tout ≈ Tlm atol=4f-4 rtol=4f-4
        Qcompat = similar(Qout); Scompat = similar(Sout); Tcompat = similar(Tout)
        dist_analysis_qst!(
            plan, Qcompat, Scompat, Tcompat, Vrd, Vtd, Vpd,
        )
        @test (Qcompat, Scompat, Tcompat) == (Qout, Sout, Tout)

        Vrout = PencilArray{Float32}(undef, pencil(Vrd))
        Vtout = PencilArray{Float32}(undef, pencil(Vtd))
        Vpout = PencilArray{Float32}(undef, pencil(Vpd))
        synthesis_qst!(
            plan, Vrout, Vtout, Vpout, Qout, Sout, Tout;
            real_output=true,
        )
        @test _collect_spatial(Vrout, cfg) ≈ Vr atol=4f-4 rtol=4f-4
        @test _collect_spatial(Vtout, cfg) ≈ Vt atol=4f-4 rtol=4f-4
        @test _collect_spatial(Vpout, cfg) ≈ Vp atol=4f-4 rtol=4f-4
        Vrcompat = similar(Vrout); Vtcompat = similar(Vtout); Vpcompat = similar(Vpout)
        dist_synthesis_qst!(
            plan, Vrcompat, Vtcompat, Vpcompat, Qout, Sout, Tout;
            real_output=true,
        )
        @test parent(Vrcompat) == parent(Vrout)
        @test parent(Vtcompat) == parent(Vtout)
        @test parent(Vpcompat) == parent(Vpout)

        rank = MPI.Comm_rank(qst_adapter.comm)
        self_spatial = PencilArray{Float32}(
            undef,
            Pencil((cfg.nlat, cfg.nlon), (1,), MPI.COMM_SELF),
        )
        fill!(parent(self_spatial), 0)
        rank_bad_vt = rank == 0 ? self_spatial : Vtd
        @test _all_ranks_catch(
            qst_adapter.comm; message_contains="communicator mismatch",
        ) do
            analysis_qst(cfg, Vrd, rank_bad_vt, Vpd)
        end
        MPI.Barrier(qst_adapter.comm)

        self_spectral = PencilArray{ComplexF32}(
            undef,
            Pencil(
                (cfg.lmax + 1, cfg.mmax + 1), (2,), MPI.COMM_SELF,
            ),
        )
        fill!(parent(self_spectral), 0)
        rank_bad_s = rank == 0 ? self_spectral : Sd
        @test _all_ranks_catch(
            qst_adapter.comm; message_contains="communicator mismatch",
        ) do
            synthesis_qst(
                cfg, Qd, rank_bad_s, Td; prototype_θφ=Vrd,
            )
        end
        MPI.Barrier(qst_adapter.comm)

        for analyze! in (analysis_qst!, dist_analysis_qst!), bad_component in 1:3
            fill!(Qout, ComplexF32(31, -7))
            fill!(Sout, ComplexF32(31, -7))
            fill!(Tout, ComplexF32(31, -7))
            inputs = ntuple(3) do component
                rank == 0 && component == bad_component ? self_spatial :
                    (Vrd, Vtd, Vpd)[component]
            end
            @test _all_ranks_catch(
                qst_adapter.comm; message_contains="communicator mismatch",
            ) do
                analyze!(plan, Qout, Sout, Tout, inputs...)
            end
            @test all(==(ComplexF32(31, -7)), Qout)
            @test all(==(ComplexF32(31, -7)), Sout)
            @test all(==(ComplexF32(31, -7)), Tout)
            MPI.Barrier(qst_adapter.comm)
        end

        rank_bad_vr_out = rank == 0 ? self_spatial : Vrout
        fill!(parent(Vtout), 32)
        fill!(parent(Vpout), 32)
        @test _all_ranks_catch(
            qst_adapter.comm; message_contains="communicator mismatch",
        ) do
            synthesis_qst!(
                plan, rank_bad_vr_out, Vtout, Vpout,
                dense[1], dense[2], dense[3]; real_output=true,
            )
        end
        @test all(==(32f0), parent(Vtout))
        @test all(==(32f0), parent(Vpout))
        MPI.Barrier(qst_adapter.comm)

        bad_vp = rank == 0 ?
            PencilArray{Float64}(undef, pencil(Vpd)) :
            PencilArray{Float32}(undef, pencil(Vpd))
        fill!(parent(bad_vp), 0)
        @test _all_ranks_catch(qst_adapter.comm) do
            analysis_qst(cfg, Vrd, Vtd, bad_vp)
        end
        rank_real_output = iszero(rank % 2)
        @test _all_ranks_catch(qst_adapter.comm) do
            synthesis_qst(
                cfg, Qd, Sd, Td; prototype_θφ=Vrd,
                real_output=rank_real_output,
            )
        end
        MPI.Barrier(qst_adapter.comm)
    end
end

if isempty(ARGS) || "vector_variants" in ARGS
    @testset "Pencil-native vector/QST variants" begin
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        for (name, signature) in (
            (:analysis_sphtor_l, Tuple{SHTConfig,PencilArray,PencilArray,Int}),
            (:synthesis_sphtor_l, Tuple{SHTConfig,PencilArray,PencilArray,Int}),
            (:analysis_sphtor_ml, Tuple{SHTConfig,Int,PencilArray,PencilArray,Int}),
            (:synthesis_sphtor_ml, Tuple{SHTConfig,Int,PencilArray,PencilArray,Int}),
            (:synthesis_grad_l, Tuple{SHTConfig,PencilArray,Int}),
            (:synthesis_grad_ml, Tuple{SHTConfig,Int,PencilArray,Int}),
            (:analysis_qst_l, Tuple{SHTConfig,PencilArray,PencilArray,PencilArray,Int}),
            (:synthesis_qst_l, Tuple{SHTConfig,PencilArray,PencilArray,PencilArray,Int}),
            (:analysis_qst_ml, Tuple{SHTConfig,Int,PencilArray,PencilArray,PencilArray,Int}),
            (:synthesis_qst_ml, Tuple{SHTConfig,Int,PencilArray,PencilArray,PencilArray,Int}),
            (:analysis_sphtor_batch, Tuple{SHTConfig,PencilArray,PencilArray}),
            (:synthesis_sphtor_batch, Tuple{SHTConfig,PencilArray,PencilArray}),
            (:analysis_qst_batch, Tuple{SHTConfig,PencilArray,PencilArray,PencilArray}),
            (:synthesis_qst_batch, Tuple{SHTConfig,PencilArray,PencilArray,PencilArray}),
        )
            @test hasmethod(getproperty(SHTnsKit, name), signature)
            hasmethod(getproperty(SHTnsKit, name), signature) &&
                @test which(getproperty(SHTnsKit, name), signature).module === extension
        end

        cfg = _variant_cfg(Float32; mres=2, norm=:schmidt,
                           real_norm=true, cs_phase=false)
        ltr = 5
        S = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
        Tlm = zero(S); Q = zero(S)
        S[3, 1] = 0.12f0
        S[5, 5] = 0.04f0 - 0.02f0im
        Tlm[4, 3] = ComplexF32(-0.03, 0.01)
        Q[2, 1] = 0.08f0
        S[7, 1] = 91f0; Tlm[7, 1] = -72f0; Q[7, 1] = 63f0
        expected_v = synthesis_sphtor_l(CPU(), cfg, S, Tlm, ltr)
        expected_q = synthesis_qst_l(CPU(), cfg, Q, S, Tlm, ltr)
        prototype = vector_place(vector_adapter, cfg, expected_v[1], :spatial)
        Sd = vector_place(vector_adapter, cfg, S, :spectral)
        Td = vector_place(vector_adapter, cfg, Tlm, :spectral)
        Qd = vector_place(vector_adapter, cfg, Q, :spectral)
        extension._reset_pencil_scalar_stats!()
        got_v = synthesis_sphtor_l(cfg, Sd, Td, ltr; prototype_θφ=prototype)
        synthesis_stats = extension._pencil_scalar_stats()
        active_positive_bins = length(0:cfg.mres:min(cfg.mmax, ltr))
        active_fourier_bins = 2active_positive_bins - 1
        max_theta_slab = MPI.Allreduce(
            size(parent(prototype), 1), max, MPI.COMM_WORLD,
        )
        @test synthesis_stats.vector_synthesis_max_message_elements <=
              max_theta_slab * active_fourier_bins
        @test synthesis_stats.vector_synthesis_max_message_elements <
              max_theta_slab * cfg.nlon
        got_q = synthesis_qst_l(cfg, Qd, Sd, Td, ltr; prototype_θφ=prototype)
        for (got, expected) in zip(got_v, expected_v)
            @test got isa PencilArray
            @test _collect_spatial(got, cfg) ≈ expected atol=4f-4 rtol=4f-4
        end
        for (got, expected) in zip(got_q, expected_q)
            @test got isa PencilArray
            @test _collect_spatial(got, cfg) ≈ expected atol=4f-4 rtol=4f-4
        end
        extension._reset_pencil_scalar_stats!()
        analyzed_v = analysis_sphtor_l(cfg, got_v..., ltr)
        analysis_stats = extension._pencil_scalar_stats()
        active_lm = sum(ltr - max(1, m) + 1
                        for m in 0:cfg.mres:min(cfg.mmax, ltr))
        @test analysis_stats.vector_analysis_max_message_elements <= active_lm
        @test analysis_stats.vector_analysis_sent_elements == 2active_lm
        analyzed_q = analysis_qst_l(cfg, got_q..., ltr)
        for (got, expected) in zip(analyzed_v, (S, Tlm))
            host = spectral_pencil_to_matrix(cfg, got)
            @test host[1:(ltr + 1), :] ≈ expected[1:(ltr + 1), :] atol=5f-4 rtol=5f-4
            @test all(iszero, host[(ltr + 2):end, :])
        end
        for (got, expected) in zip(analyzed_q, (Q, S, Tlm))
            host = spectral_pencil_to_matrix(cfg, got)
            @test host[1:(ltr + 1), :] ≈ expected[1:(ltr + 1), :] atol=5f-4 rtol=5f-4
        end

        stored_im = 2; physical_m = stored_im * cfg.mres
        Sm = ComplexF32.(S[(physical_m + 1):(ltr + 1), physical_m + 1])
        Tm = ComplexF32.(Tlm[(physical_m + 1):(ltr + 1), physical_m + 1])
        Qm = ComplexF32.(Q[(physical_m + 1):(ltr + 1), physical_m + 1])
        Sm .= ComplexF32(0.04, -0.01); Tm .= ComplexF32(-0.02, 0.015)
        Qm .= ComplexF32(0.03, 0.02)
        expected_m = synthesis_sphtor_ml(CPU(), cfg, stored_im, Sm, Tm, ltr)
        expected_qm = synthesis_qst_ml(CPU(), cfg, stored_im, Qm, Sm, Tm, ltr)
        Smd = _place_distributed_vector(Sm, MPI.COMM_WORLD)
        Tmd = _place_distributed_vector(Tm, MPI.COMM_WORLD)
        Qmd = _place_distributed_vector(Qm, MPI.COMM_WORLD)
        extension._reset_pencil_scalar_stats!()
        got_m = synthesis_sphtor_ml(cfg, stored_im, Smd, Tmd, ltr)
        mode_stats = extension._pencil_scalar_stats()
        max_owned_theta = MPI.Allreduce(
            size(parent(got_m[1]), 1), max, MPI.COMM_WORLD,
        )
        @test mode_stats.vector_mode_synthesis_max_message_elements ==
              max_owned_theta
        @test mode_stats.vector_mode_synthesis_sent_elements == 2cfg.nlat
        @test mode_stats.vector_mode_max_message_elements == max_owned_theta
        @test mode_stats.vector_mode_sent_elements == 2cfg.nlat
        extension._reset_pencil_scalar_stats!()
        got_qm = synthesis_qst_ml(cfg, stored_im, Qmd, Smd, Tmd, ltr)
        qst_synthesis_stats = extension._pencil_scalar_stats()
        @test qst_synthesis_stats.vector_mode_synthesis_max_message_elements ==
              max_owned_theta
        @test qst_synthesis_stats.vector_mode_synthesis_sent_elements == 2cfg.nlat
        @test qst_synthesis_stats.scalar_mode_synthesis_max_message_elements ==
              max_owned_theta
        @test qst_synthesis_stats.scalar_mode_synthesis_sent_elements == cfg.nlat
        @test _collect_distributed_vector(got_m[1]) ≈ expected_m[1] atol=5f-4 rtol=5f-4
        @test _collect_distributed_vector(got_m[2]) ≈ expected_m[2] atol=5f-4 rtol=5f-4
        @test _collect_distributed_vector(got_qm[1]) ≈ expected_qm[1] atol=5f-4 rtol=5f-4
        got_qm32 = synthesis_qst_ml(
            cfg, Int32(stored_im), Qmd, Smd, Tmd, ltr,
        )
        for (got, expected) in zip(got_qm32, expected_qm)
            @test _collect_distributed_vector(got) ≈ expected atol=5f-4 rtol=5f-4
        end
        extension._reset_pencil_scalar_stats!()
        back_m = analysis_sphtor_ml(cfg, stored_im, got_m..., ltr)
        mode_analysis_stats = extension._pencil_scalar_stats()
        max_owned_coefficients = MPI.Allreduce(
            size(parent(Smd), 1), max, MPI.COMM_WORLD,
        )
        @test mode_analysis_stats.vector_mode_analysis_max_message_elements ==
              max_owned_coefficients
        @test mode_analysis_stats.vector_mode_analysis_sent_elements == 2length(Sm)
        @test mode_analysis_stats.vector_mode_max_message_elements ==
              max_owned_coefficients
        @test mode_analysis_stats.vector_mode_sent_elements == 2length(Sm)
        if MPI.Comm_size(MPI.COMM_WORLD) >= 4
            min_owned_coefficients = MPI.Allreduce(
                size(parent(Smd), 1), min, MPI.COMM_WORLD,
            )
            @test min_owned_coefficients == 0
        end
        extension._reset_pencil_scalar_stats!()
        back_qm = analysis_qst_ml(cfg, stored_im, got_qm..., ltr)
        qst_analysis_stats = extension._pencil_scalar_stats()
        @test qst_analysis_stats.vector_mode_analysis_max_message_elements ==
              max_owned_coefficients
        @test qst_analysis_stats.vector_mode_analysis_sent_elements == 2length(Sm)
        @test qst_analysis_stats.scalar_mode_analysis_max_message_elements ==
              max_owned_coefficients
        @test qst_analysis_stats.scalar_mode_analysis_sent_elements == length(Qm)
        @test _collect_distributed_vector(back_m[1]) ≈ Sm atol=5f-4 rtol=5f-4
        @test _collect_distributed_vector(back_m[2]) ≈ Tm atol=5f-4 rtol=5f-4
        @test _collect_distributed_vector(back_qm[1]) ≈ Qm atol=5f-4 rtol=5f-4
        expected_grad = synthesis_grad_ml(CPU(), cfg, stored_im, Sm, ltr)
        extension._reset_pencil_scalar_stats!()
        got_grad = synthesis_grad_ml(cfg, stored_im, Smd, ltr)
        gradient_stats = extension._pencil_scalar_stats()
        @test gradient_stats.vector_mode_synthesis_max_message_elements ==
              max_owned_theta
        @test gradient_stats.vector_mode_synthesis_sent_elements == 2cfg.nlat
        @test _collect_distributed_vector(got_grad[1]) ≈ expected_grad[1] atol=5f-4 rtol=5f-4
        @test _collect_distributed_vector(got_grad[2]) ≈ expected_grad[2] atol=5f-4 rtol=5f-4

        cfg64 = _variant_cfg(Float64; mres=2, norm=:schmidt,
                             real_norm=true, cs_phase=false)
        Qm64 = ComplexF64.(Qm); Sm64 = ComplexF64.(Sm); Tm64 = ComplexF64.(Tm)
        expected_qm64 = synthesis_qst_ml(
            CPU(), cfg64, stored_im, Qm64, Sm64, Tm64, ltr,
        )
        Qmd64 = _place_distributed_vector(Qm64, MPI.COMM_WORLD)
        Smd64 = _place_distributed_vector(Sm64, MPI.COMM_WORLD)
        Tmd64 = _place_distributed_vector(Tm64, MPI.COMM_WORLD)
        got_qm64 = synthesis_qst_ml(
            cfg64, stored_im, Qmd64, Smd64, Tmd64, ltr,
        )
        for (got, expected) in zip(got_qm64, expected_qm64)
            @test _collect_distributed_vector(got) ≈ expected atol=2e-11 rtol=2e-11
        end
        back_qm64 = analysis_qst_ml(cfg64, stored_im, got_qm64..., ltr)
        for (got, expected) in zip(back_qm64, (Qm64, Sm64, Tm64))
            @test _collect_distributed_vector(got) ≈ expected atol=2e-11 rtol=2e-11
        end

        for nfields in (1, 2, 5)
            Qbatch = repeat(reshape(Q, size(Q)..., 1), 1, 1, nfields)
            Sbatch = repeat(reshape(S, size(S)..., 1), 1, 1, nfields)
            Tbatch = repeat(reshape(Tlm, size(Tlm)..., 1), 1, 1, nfields)
            for k in 1:nfields
                Qbatch[:, :, k] .*= k
                Sbatch[:, :, k] .*= k
                Tbatch[:, :, k] .*= k
            end
            cpu_fields = synthesis_qst_batch(CPU(), cfg, Qbatch, Sbatch, Tbatch)
            Qbd = _place_distributed_batch(cfg, Qbatch, :spectral, MPI.COMM_WORLD)
            Sbd = _place_distributed_batch(cfg, Sbatch, :spectral, MPI.COMM_WORLD)
            Tbd = _place_distributed_batch(cfg, Tbatch, :spectral, MPI.COMM_WORLD)
            distributed_fields = synthesis_qst_batch(cfg, Qbd, Sbd, Tbd)
            for (got, expected) in zip(distributed_fields, cpu_fields)
                @test got isa PencilArray
                @test size_global(got)[3] == nfields
                @test _collect_distributed_batch(got, MPI.COMM_WORLD) ≈
                      expected atol=5f-4 rtol=5f-4
            end
            analyzed_batch = analysis_qst_batch(cfg, distributed_fields...)
            for (got, expected) in zip(analyzed_batch, (Qbatch, Sbatch, Tbatch))
                @test got isa PencilArray
                @test _collect_distributed_batch(got, MPI.COMM_WORLD) ≈
                      expected atol=6f-4 rtol=6f-4
            end
        end

        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        one_batch = repeat(reshape(expected_v[1], cfg.nlat, cfg.nlon, 1), 1, 1, 2)
        good_batch = _place_distributed_batch(
            cfg, Float32.(one_batch), :spatial, MPI.COMM_WORLD,
        )
        float64_batch = _place_distributed_batch(
            cfg, Float64.(one_batch), :spatial, MPI.COMM_WORLD,
        )
        bad_batch = rank == 0 ? float64_batch : good_batch
        good_sentinel = copy(parent(good_batch))
        bad_sentinel = copy(parent(bad_batch))
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(MPI.COMM_WORLD) do
            analysis_sphtor_batch(cfg, good_batch, bad_batch)
        end
        @test parent(good_batch) == good_sentinel
        @test parent(bad_batch) == bad_sentinel
        rejected_stats = extension._pencil_scalar_stats()
        @test rejected_stats.vector_analysis_sent_elements == 0
        @test rejected_stats.vector_synthesis_sent_elements == 0
        MPI.Barrier(MPI.COMM_WORLD)

        rank_ltr = rank == 0 ? ltr - 1 : ltr
        spectral_sentinel = copy(parent(Sd))
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(
            MPI.COMM_WORLD; message_contains="degree truncation",
        ) do
            synthesis_sphtor_l(
                cfg, Sd, Td, rank_ltr; prototype_θφ=prototype,
            )
        end
        @test parent(Sd) == spectral_sentinel
        @test extension._pencil_scalar_stats().vector_synthesis_sent_elements == 0
        MPI.Barrier(MPI.COMM_WORLD)

        rank_im = rank == 0 ? stored_im - 1 : stored_im
        mode_sentinel = copy(parent(Smd))
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(
            MPI.COMM_WORLD; message_contains="degree truncation",
        ) do
            synthesis_sphtor_ml(cfg, rank_im, Smd, Tmd, ltr)
        end
        @test parent(Smd) == mode_sentinel
        @test extension._pencil_scalar_stats().vector_mode_sent_elements == 0
        MPI.Barrier(MPI.COMM_WORLD)

        q_mode_sentinel = copy(parent(Qmd))
        s_mode_sentinel = copy(parent(Smd))
        t_mode_sentinel = copy(parent(Tmd))
        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(
            MPI.COMM_WORLD; message_contains="degree truncation",
        ) do
            synthesis_qst_ml(cfg, rank_im, Qmd, Smd, Tmd, ltr)
        end
        @test parent(Qmd) == q_mode_sentinel
        @test parent(Smd) == s_mode_sentinel
        @test parent(Tmd) == t_mode_sentinel
        rejected_mode_stats = extension._pencil_scalar_stats()
        @test rejected_mode_stats.scalar_mode_synthesis_sent_elements == 0
        @test rejected_mode_stats.vector_mode_synthesis_sent_elements == 0
        @test rejected_mode_stats.vector_mode_sent_elements == 0
        MPI.Barrier(MPI.COMM_WORLD)

        extension._reset_pencil_scalar_stats!()
        @test _all_ranks_catch(
            MPI.COMM_WORLD; message_contains="degree truncation",
        ) do
            synthesis_qst_ml(
                cfg, big(typemax(Int)) + 1, Qmd, Smd, Tmd, ltr,
            )
        end
        overflow_mode_stats = extension._pencil_scalar_stats()
        @test overflow_mode_stats.scalar_mode_synthesis_sent_elements == 0
        @test overflow_mode_stats.vector_mode_synthesis_sent_elements == 0
        MPI.Barrier(MPI.COMM_WORLD)

        empty_spatial = _place_distributed_batch(
            cfg, zeros(Float32, cfg.nlat, cfg.nlon, 0), :spatial,
            MPI.COMM_WORLD,
        )
        @test _all_ranks_catch(
            MPI.COMM_WORLD; message_contains="global shape mismatch",
        ) do
            analysis_sphtor_batch(cfg, empty_spatial, empty_spatial)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

if isempty(ARGS) || "local_evaluation" in ARGS
    @testset "MPI local evaluation parity" begin
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        for T in (Float32, Float64), mres in (1, 2), convention in (
            (norm=:orthonormal, real_norm=false, cs_phase=true),
            (norm=:schmidt, real_norm=true, cs_phase=false),
        ), robert_form in (false, true)
            cfg = _local_config(
                :regular_poles, T; mres, convention..., robert_form,
            )
            Qcan, Scan, Tcan, Drcan = _local_canonical_modes(cfg, T)
            Q = _local_external(cfg, Qcan)
            S = _local_external(cfg, Scan)
            Tlm = _local_external(cfg, Tcan)
            Dr = _local_external(cfg, Drcan)
            if mres == 2
                Q[2, 2] = Complex{T}(T(91), T(-37))
                S[2, 2] = Complex{T}(T(-42), T(18))
            end
            Qp = matrix_to_spectral_pencil(cfg, Q; comm)
            Sp = matrix_to_spectral_pencil(cfg, S; comm)
            Tp = matrix_to_spectral_pencil(cfg, Tlm; comm)
            Drp = matrix_to_spectral_pencil(cfg, Dr; comm)
            tol = _local_tol(T)
            for (cost, phi) in ((T(0.37), T(0.61)), (one(T), T(2pi)),
                                (-one(T), T(-2pi)))
                @test synthesis_point(cfg, Qp, cost, phi) ≈
                      _local_direct_scalar(Qcan, cost, phi) atol=tol.atol rtol=tol.rtol
                qst = SHqst_to_point(cfg, Qp, Sp, Tp, cost, phi)
                @test all(value -> value isa T, qst)
                @test collect(qst) ≈
                      collect(_local_direct_qst(cfg, Qcan, Scan, Tcan, cost, phi)) atol=tol.atol rtol=tol.rtol
                grad = SH_to_grad_point(cfg, Drp, Sp, cost, phi)
                @test collect(grad) ≈
                      collect(_local_direct_qst(cfg, Drcan, Scan, zero(Tcan), cost, phi)) atol=tol.atol rtol=tol.rtol
            end
            nphi = 7
            cost = T(-0.28)
            lat = SH_to_lat(cfg, Qp, cost; nphi)
            @test eltype(lat) === T
            @test lat ≈ [_local_direct_scalar(Qcan, cost, T(2pi * j / nphi))
                         for j in 0:(nphi - 1)] atol=tol.atol rtol=tol.rtol
            qst_lat = SHqst_to_lat(cfg, Qp, Sp, Tp, cost; nphi)
            refs = [_local_direct_qst(
                        cfg, Qcan, Scan, Tcan, cost, T(2pi * j / nphi),
                    ) for j in 0:(nphi - 1)]
            for component in 1:3
                @test eltype(qst_lat[component]) === T
                @test qst_lat[component] ≈ getindex.(refs, component) atol=tol.atol rtol=tol.rtol
            end
        end

        cfg = _local_config(
            :gauss, Float64; norm=:fourpi, real_norm=true, cs_phase=false,
        )
        canonical, external = _local_complex_modes(cfg, Float64)
        distributed = _place_distributed_vector(external, comm)
        @test synthesis_point_cplx(cfg, distributed, -0.28, 0.77) ≈
              _local_direct_complex(canonical, -0.28, 0.77) atol=3e-13 rtol=3e-13
        complex_lat = SH_to_lat_cplx(cfg, distributed, -0.28; nphi=7)
        @test eltype(complex_lat) === ComplexF64
        @test complex_lat ≈ [_local_direct_complex(canonical, -0.28, 2pi*j/7)
                             for j in 0:6] atol=3e-13 rtol=3e-13

        # Four ranks over three spectral columns exercises an empty local owner.
        Qcan, Scan, Tcan, Drcan = _local_canonical_modes(cfg, Float64)
        Qp = matrix_to_spectral_pencil(cfg, _local_external(cfg, Qcan); comm)
        Sp = matrix_to_spectral_pencil(cfg, _local_external(cfg, Scan); comm)
        Tp = matrix_to_spectral_pencil(cfg, _local_external(cfg, Tcan); comm)
        Drp = matrix_to_spectral_pencil(cfg, _local_external(cfg, Drcan); comm)
        minimum_owned = MPI.Allreduce(length(parent(Qp)), min, comm)
        MPI.Comm_size(comm) < 4 || @test minimum_owned == 0

        extension._reset_local_evaluation_stats!()
        got = SHqst_to_lat(cfg, Qp, Sp, Tp, 0.2; nphi=5)
        stats = extension._local_evaluation_stats()
        @test stats.payload_reductions == 1
        @test stats.payload_elements == 15
        @test stats.max_payload_elements == 15
        @test all(length(value) == 5 for value in got)

        extension._reset_local_evaluation_stats!()
        rank_cost = rank == 0 ? 0.3 : 0.2
        @test _all_ranks_catch(comm; message_contains="rank-varying cost") do
            synthesis_point(cfg, Qp, rank_cost, 0.7)
        end
        @test extension._local_evaluation_stats().payload_reductions == 0
        MPI.Barrier(comm)

        extension._reset_local_evaluation_stats!()
        rank_nphi = rank == 0 ? 6 : 5
        @test _all_ranks_catch(comm; message_contains="rank-varying nphi") do
            SH_to_lat(cfg, Qp, 0.2; nphi=rank_nphi)
        end
        @test extension._local_evaluation_stats().payload_reductions == 0
        MPI.Barrier(comm)

        S32 = matrix_to_spectral_pencil(
            cfg, ComplexF32.(_local_external(cfg, Scan)); comm,
        )
        badS = rank == 0 ? S32 : Sp
        extension._reset_local_evaluation_stats!()
        @test _all_ranks_catch(comm; message_contains="precision") do
            SHqst_to_point(cfg, Qp, badS, Tp, 0.2, 0.7)
        end
        @test extension._local_evaluation_stats().payload_reductions == 0
        MPI.Barrier(comm)

        # The first input is the communicator anchor.  Exercise collective
        # rejection with a rank-varying peer while every rank enters the same
        # trusted communicator; a rank-varying anchor is two distinct calls
        # and cannot be validated by a collective without using WORLD.
        Sself = matrix_to_spectral_pencil(
            cfg, _local_external(cfg, Scan); comm=MPI.COMM_SELF,
        )
        Tself = matrix_to_spectral_pencil(
            cfg, _local_external(cfg, Tcan); comm=MPI.COMM_SELF,
        )
        badScomm = rank == 0 ? Sself : Sp
        badT = rank == 0 ? Tself : Tp
        for call in (
            () -> SHqst_to_point(cfg, Qp, badScomm, badT, 0.2, 0.7),
            () -> SHqst_to_lat(cfg, Qp, badScomm, badT, 0.2; nphi=5),
            () -> SH_to_grad_point(cfg, Drp, badScomm, 0.2, 0.7),
        )
            extension._reset_local_evaluation_stats!()
            @test _all_ranks_catch(comm; message_contains="communicator mismatch") do
                call()
            end
            @test extension._local_evaluation_stats().payload_reductions == 0
            MPI.Barrier(comm)
        end

    end
end

if isempty(ARGS) || "operators" in ARGS
    @testset "MPI spectral operator parity" begin
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        cfg = create_gauss_config(
            6, 9; mmax=2, nlon=16, mres=2, norm=:schmidt,
            real_norm=true, cs_phase=false,
        )
        dense = reshape(
            ComplexF32.(1:((cfg.lmax + 1) * (cfg.mmax + 1))),
            cfg.lmax + 1, cfg.mmax + 1,
        )
        source = matrix_to_spectral_pencil(cfg, dense; comm)
        output = PencilArray{ComplexF32}(undef, pencil(source))
        fill!(parent(output), ComplexF32(91, -17))

        extension._reset_operator_stats!()
        divergence = divergence_from_spheroidal(cfg, source)
        @test divergence isa PencilArray
        @test spectral_pencil_to_matrix(cfg, divergence) ==
              divergence_from_spheroidal(cfg, dense)
        @test extension._operator_stats().diagonal_payload_sent_elements == 0
        MPI.Barrier(comm)

        # The existing compatibility Laplacian is the same diagonal operator,
        # remains local-only, and filters l=0/l<m/nonstored m exactly.
        lap = matrix_to_spectral_pencil(cfg, dense; comm)
        extension._reset_operator_stats!()
        @test SHTnsKit.dist_apply_laplacian!(cfg, lap) === lap
        @test spectral_pencil_to_matrix(cfg, lap) ==
              divergence_from_spheroidal(cfg, dense)
        @test extension._operator_stats().diagonal_payload_sent_elements == 0

        mx = zeros(Float32, 2cfg.nlm)
        mul_ct_matrix(cfg, mx)
        reference_packed = zeros(ComplexF32, cfg.nlm)
        SH_mul_mx(cfg, mx, SHTnsKit.pack_lm(cfg, dense), reference_packed)
        reference = SHTnsKit.unpack_lm(cfg, reference_packed)
        extension._reset_operator_stats!()
        @test SH_mul_mx(SHTnsKit.CPU(), cfg, mx, source, output) === output
        @test spectral_pencil_to_matrix(cfg, output) == reference
        @test extension._operator_stats().narrow_payload_sent_elements == 0
        fill!(parent(output), ComplexF32(91, -17))
        @test SHTnsKit.dist_SH_mul_mx!(cfg, mx, source, output) === output
        @test spectral_pencil_to_matrix(cfg, output) == reference

        # Replicated coefficients must be compared elementwise.  These rank-0
        # perturbations deliberately preserve the two historical weighted
        # checksums and the extrema, so summary-only validation cannot detect
        # them.
        colliding_mx = ones(Float32, 2cfg.nlm)
        colliding_mx[1] = -100
        colliding_mx[end] = 100
        if rank == 0
            colliding_mx[2] += 6 / 128
            colliding_mx[3] -= 8 / 128
            colliding_mx[4] += 3 / 128
        end
        fill!(parent(output), ComplexF32(27, -5))
        sentinel = copy(parent(output))
        extension._reset_operator_stats!()
        @test _all_ranks_catch(
            comm; message_contains="rank-varying operator matrix",
        ) do
            SH_mul_mx(cfg, colliding_mx, source, output)
        end
        @test parent(output) == sentinel
        @test extension._operator_stats().narrow_payload_sent_elements == 0
        MPI.Barrier(comm)

        # Four ranks over three m columns includes a zero-element owner and is
        # still a valid local no-payload execution.
        minimum_owned = MPI.Allreduce(length(parent(source)), min, comm)
        MPI.Comm_size(comm) < 4 || @test minimum_owned == 0

        # Rank-varying malformed layouts are rejected collectively on the
        # trusted communicator before output mutation or payload traffic.
        bad_pen = Pencil((cfg.lmax + 2, cfg.mmax + 1), (2,), comm)
        malformed = PencilArray{ComplexF32}(undef, bad_pen)
        fill!(parent(malformed), ComplexF32(3, -2))
        fill!(parent(malformed), ComplexF32(27, -5))
        candidate_output = rank == 0 ? malformed : output
        fill!(parent(candidate_output), ComplexF32(27, -5))
        sentinel = copy(parent(candidate_output))
        extension._reset_operator_stats!()
        @test _all_ranks_catch(comm; message_contains="global shape mismatch") do
            SH_mul_mx(cfg, mx, source, candidate_output)
        end
        @test parent(candidate_output) == sentinel
        @test extension._operator_stats().narrow_payload_sent_elements == 0
        MPI.Barrier(comm)

        self_output = matrix_to_spectral_pencil(cfg, dense; comm=MPI.COMM_SELF)
        bad_comm_output = rank == 0 ? self_output : output
        extension._reset_operator_stats!()
        @test _all_ranks_catch(comm; message_contains="communicator mismatch") do
            divergence_from_spheroidal!(cfg, bad_comm_output, source)
        end
        @test extension._operator_stats().diagonal_payload_sent_elements == 0
        MPI.Barrier(comm)
    end
end

if isempty(ARGS) || "rotations" in ARGS
    @testset "distributed rotation parity" begin
        comm = MPI.COMM_WORLD
        extension = Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)
        rank = MPI.Comm_rank(comm)
        for RT in (Float32, Float64)
            tol = RT === Float32 ? RT(1e-4) : RT(3e-12)
            cfg = create_gauss_config(
                4, 7; nlon=12, norm=:schmidt,
                real_norm=true, cs_phase=false,
            )
            dense = zeros(Complex{RT}, cfg.lmax + 1, cfg.mmax + 1)
            rng = MersenneTwister(0xD071 + sizeof(RT))
            for m in 0:cfg.mmax, l in m:cfg.lmax
                dense[l + 1, m + 1] = randn(rng, Complex{RT})
            end
            dense[:, 1] .= real.(dense[:, 1])
            source = matrix_to_spectral_pencil(cfg, dense; comm)
            output = PencilArray{Complex{RT}}(undef, pencil(source))

            extension._reset_rotation_stats!()
            @test SHTnsKit.dist_SH_Zrotate(cfg, source, RT(0.27), output) === output
            expected_z = zeros(Complex{RT}, size(dense))
            SHTnsKit.dist_SH_Zrotate(cfg, dense, RT(0.27), expected_z)
            for m in 0:cfg.mmax, l in m:cfg.lmax
                @test expected_z[l + 1, m + 1] ≈
                      dense[l + 1, m + 1] * cis(-RT(m) * RT(0.27)) atol=tol rtol=tol
            end
            @test spectral_pencil_to_matrix(cfg, output) ≈ expected_z atol=tol rtol=tol
            @test extension._rotation_stats().z_payload_sent_elements == 0

            extension._reset_rotation_stats!()
            @test SHTnsKit.dist_SH_Yrotate(cfg, source, RT(-0.36), output) === output
            expected_y = zeros(Complex{RT}, size(dense))
            SHTnsKit.dist_SH_Yrotate(cfg, dense, RT(-0.36), expected_y)
            @test spectral_pencil_to_matrix(cfg, output) ≈ expected_y atol=tol rtol=tol
            stats = extension._rotation_stats()
            @test stats.general_payload_sent_elements > 0
            @test stats.general_max_message_elements <= cfg.lmax + 1

            y90 = similar(output); x90 = similar(output)
            SHTnsKit.dist_SH_Yrotate90(cfg, source, y90)
            SHTnsKit.dist_SH_Xrotate90(cfg, source, x90)
            expected_y90 = zeros(Complex{RT}, size(dense))
            expected_x90 = zeros(Complex{RT}, size(dense))
            packed_dense = SHTnsKit.pack_lm(cfg, dense)
            qy = similar(packed_dense); qx = similar(qy)
            SH_Yrotate90(CPU(), cfg, packed_dense, qy)
            SH_Xrotate90(CPU(), cfg, packed_dense, qx)
            SHTnsKit.unpack_lm!(expected_y90, cfg, qy)
            SHTnsKit.unpack_lm!(expected_x90, cfg, qx)
            @test spectral_pencil_to_matrix(cfg, y90) ≈ expected_y90 atol=tol rtol=tol
            @test spectral_pencil_to_matrix(cfg, x90) ≈ expected_x90 atol=tol rtol=tol

            euler = similar(output)
            SHTnsKit.dist_SH_rotate_euler(
                cfg, source, RT(0.13), RT(-0.28), RT(0.41), euler,
            )
            t1 = zeros(Complex{RT}, size(dense))
            t2 = zeros(Complex{RT}, size(dense))
            expected_euler = zeros(Complex{RT}, size(dense))
            SHTnsKit.dist_SH_Zrotate(cfg, dense, RT(0.13), t1)
            SHTnsKit.dist_SH_Yrotate(cfg, t1, RT(-0.28), t2)
            SHTnsKit.dist_SH_Zrotate(cfg, t2, RT(0.41), expected_euler)
            @test spectral_pencil_to_matrix(cfg, euler) ≈ expected_euler atol=tol rtol=tol

            packed = SHTnsKit.pack_lm(cfg, dense)
            @test SHTnsKit.dist_SH_Zrotate_packed(
                cfg, packed, RT(0.27); prototype_lm=source,
            ) ≈ SHTnsKit.pack_lm(cfg, expected_z) atol=tol rtol=tol
            @test SHTnsKit.dist_SH_Yrotate_packed(
                cfg, packed, RT(-0.36); prototype_lm=source,
            ) ≈ SHTnsKit.pack_lm(cfg, expected_y) atol=tol rtol=tol
            @test SHTnsKit.dist_SH_Yrotate90_packed(
                cfg, packed; prototype_lm=source,
            ) ≈ qy atol=tol rtol=tol
            @test SHTnsKit.dist_SH_Xrotate90_packed(
                cfg, packed; prototype_lm=source,
            ) ≈ qx atol=tol rtol=tol
        end

        cfg = create_gauss_config(4, 7; nlon=12)
        dense = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
        dense[3, 2] = 0.4 - 0.2im
        source = matrix_to_spectral_pencil(cfg, dense; comm)
        output = PencilArray{ComplexF64}(undef, pencil(source))
        sentinel = ComplexF64(91, -17)
        fill!(parent(output), sentinel)
        extension._reset_rotation_stats!()
        varying_angle = rank == 0 ? 0.2 : 0.3
        @test _all_ranks_catch(comm; message_contains="rotation inputs") do
            SHTnsKit.dist_SH_Zrotate(cfg, source, varying_angle, output)
        end
        @test all(==(sentinel), parent(output))
        @test extension._rotation_stats().z_payload_sent_elements == 0
        MPI.Barrier(comm)

        cfg2 = create_gauss_config(4, 7; nlon=12, mres=2)
        source2 = matrix_to_spectral_pencil(
            cfg2, zeros(ComplexF64, cfg2.lmax + 1, cfg2.mmax + 1); comm,
        )
        output2 = PencilArray{ComplexF64}(undef, pencil(source2))
        fill!(parent(output2), sentinel)
        extension._reset_rotation_stats!()
        @test _all_ranks_catch(comm; message_contains="mres==1") do
            SHTnsKit.dist_SH_Yrotate(cfg2, source2, 0.2, output2)
        end
        @test all(==(sentinel), parent(output2))
        @test extension._rotation_stats().general_payload_sent_elements == 0
        MPI.Barrier(comm)

        wrong_type = PencilArray{ComplexF32}(undef, pencil(source))
        fill!(parent(wrong_type), ComplexF32(91, -17))
        @test _all_ranks_catch(comm; message_contains="precision") do
            SHTnsKit.dist_SH_Yrotate(cfg, source, 0.2, wrong_type)
        end
        @test all(==(ComplexF32(91, -17)), parent(wrong_type))
        MPI.Barrier(comm)

        varying_precision = rank == 0 ? ComplexF32.(SHTnsKit.pack_lm(cfg, dense)) :
                            SHTnsKit.pack_lm(cfg, dense)
        @test _all_ranks_catch(comm; message_contains="precision") do
            SHTnsKit.dist_SH_Zrotate_packed(
                cfg, varying_precision, 0.2; prototype_lm=source,
            )
        end
        MPI.Barrier(comm)
    end

    if MPI.Comm_size(MPI.COMM_WORLD) == 4
        @testset "rotation subgroup communicator contract" begin
            world = MPI.COMM_WORLD
            world_rank = MPI.Comm_rank(world)
            color = world_rank ÷ 2
            subgroup = MPI.Comm_split(world, color, world_rank)
            subgroup_rank = MPI.Comm_rank(subgroup)
            cfg = create_gauss_config(4, 7; nlon=12)
            dense = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
            rng = MersenneTwister(0x5B60 + color)
            for m in 0:cfg.mmax, l in m:cfg.lmax
                dense[l + 1, m + 1] = randn(rng, ComplexF64)
            end
            dense[:, 1] .= real.(dense[:, 1])
            source = matrix_to_spectral_pencil(cfg, dense; comm=subgroup)
            output = similar(source)

            # The first input is the trusted communicator anchor. Both disjoint
            # subgroups execute independently; no WORLD collective is permitted.
            expected_z = zeros(ComplexF64, size(dense))
            SHTnsKit.dist_SH_Zrotate(cfg, dense, 0.17, expected_z)
            SHTnsKit.dist_SH_Zrotate(cfg, source, 0.17, output)
            @test spectral_pencil_to_matrix(cfg, output) ≈ expected_z atol=3e-12 rtol=3e-12
            MPI.Barrier(subgroup)

            expected_y = zeros(ComplexF64, size(dense))
            SHTnsKit.dist_SH_Yrotate(cfg, dense, -0.29, expected_y)
            SHTnsKit.dist_SH_Yrotate(cfg, source, -0.29, output)
            @test spectral_pencil_to_matrix(cfg, output) ≈ expected_y atol=3e-12 rtol=3e-12
            MPI.Barrier(subgroup)

            euler = similar(source)
            SHTnsKit.dist_SH_rotate_euler(cfg, source, 0.11, -0.23, 0.31, euler)
            first = zeros(ComplexF64, size(dense))
            second = zeros(ComplexF64, size(dense))
            expected_euler = zeros(ComplexF64, size(dense))
            SHTnsKit.dist_SH_Zrotate(cfg, dense, 0.11, first)
            SHTnsKit.dist_SH_Yrotate(cfg, first, -0.23, second)
            SHTnsKit.dist_SH_Zrotate(cfg, second, 0.31, expected_euler)
            @test spectral_pencil_to_matrix(cfg, euler) ≈
                  expected_euler atol=3e-12 rtol=3e-12
            MPI.Barrier(subgroup)

            packed = SHTnsKit.pack_lm(cfg, dense)
            expected_packed_y = similar(packed)
            expected_packed_y90 = similar(packed)
            expected_packed_x90 = similar(packed)
            SH_Yrotate(CPU(), cfg, packed, -0.29, expected_packed_y)
            SH_Yrotate90(CPU(), cfg, packed, expected_packed_y90)
            SH_Xrotate90(CPU(), cfg, packed, expected_packed_x90)
            @test SHTnsKit.dist_SH_Zrotate_packed(
                cfg, packed, 0.17; prototype_lm=source,
            ) ≈ SHTnsKit.pack_lm(cfg, expected_z) atol=3e-12 rtol=3e-12
            @test SHTnsKit.dist_SH_Yrotate_packed(
                cfg, packed, -0.29; prototype_lm=source,
            ) ≈ expected_packed_y atol=3e-12 rtol=3e-12
            @test SHTnsKit.dist_SH_Yrotate90_packed(
                cfg, packed; prototype_lm=source,
            ) ≈ expected_packed_y90 atol=3e-12 rtol=3e-12
            @test SHTnsKit.dist_SH_Xrotate90_packed(
                cfg, packed; prototype_lm=source,
            ) ≈ expected_packed_x90 atol=3e-12 rtol=3e-12
            MPI.Barrier(subgroup)

            sentinel = ComplexF64(73, -19)
            good_output = similar(source)
            fill!(parent(good_output), sentinel)
            wrong_output = if subgroup_rank == 0
                candidate = PencilArray{ComplexF64}(
                    undef,
                    Pencil(
                        (cfg.lmax + 1, cfg.mmax + 1), (2,), MPI.COMM_SELF,
                    ),
                )
                fill!(parent(candidate), sentinel)
                candidate
            else
                good_output
            end
            @test _all_ranks_catch(subgroup; message_contains="communicator mismatch") do
                SHTnsKit.dist_SH_Yrotate(cfg, source, -0.29, wrong_output)
            end
            @test all(==(sentinel), parent(wrong_output))
            MPI.Barrier(subgroup)
            Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)._safe_comm_free(subgroup)
        end
    end
end

if isempty(ARGS) || "subgroup_firewall" in ARGS
    if MPI.Comm_size(MPI.COMM_WORLD) == 4
        @testset "staged-family subgroup communicator contract" begin
            world = MPI.COMM_WORLD
            world_rank = MPI.Comm_rank(world)
            color = world_rank ÷ 2
            subgroup = MPI.Comm_split(world, color, world_rank)
            cfg = create_gauss_config(
                4, 7; nlon=12,
                norm=color == 0 ? :orthonormal : :schmidt,
                real_norm=color == 1,
                cs_phase=color == 0,
            )
            dense = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
            dense[2, 1] = 0.12 + 0im
            dense[3, 2] = ComplexF64(0.04 * (color + 1), -0.03)
            spectral = matrix_to_spectral_pencil(cfg, dense; comm=subgroup)

            # Local point/latitude and vector-local paths must reduce only
            # within the Pencil's subgroup even though the two groups carry
            # intentionally different replicated cfg conventions.
            point = synthesis_point(cfg, spectral, 0.21, 0.37)
            @test point ≈ synthesis_point(cfg, dense, 0.21, 0.37) atol=3e-12
            latitude = SH_to_lat(cfg, spectral, -0.19; nphi=7)
            packed = SHTnsKit.pack_lm(cfg, dense)
            @test latitude ≈ SH_to_lat(cfg, packed, -0.19; nphi=7) atol=3e-12
            qst_point = SHqst_to_point(
                cfg, spectral, spectral, spectral, 0.11, -0.29,
            )
            expected_qst = SHqst_to_point(
                cfg, packed, packed, packed, 0.11, -0.29,
            )
            @test collect(qst_point) ≈ collect(expected_qst) atol=3e-12
            MPI.Barrier(subgroup)

            # Diagonal and neighbouring-degree operators use the first input
            # as their trusted communicator and preserve invalid-call payloads.
            diagonal = divergence_from_spheroidal(cfg, spectral)
            @test spectral_pencil_to_matrix(cfg, diagonal) ≈
                  divergence_from_spheroidal(cfg, dense) atol=3e-12
            lap = copy(spectral)
            @test SHTnsKit.dist_apply_laplacian!(cfg, lap) === lap
            @test spectral_pencil_to_matrix(cfg, lap) ≈
                  divergence_from_spheroidal(cfg, dense) atol=3e-12

            mx = zeros(Float64, 2cfg.nlm)
            mul_ct_matrix(CPU(), cfg, mx)
            neighbour = similar(spectral)
            @test SH_mul_mx(CPU(), cfg, mx, spectral, neighbour) === neighbour
            @test all(isfinite, parent(neighbour))
            MPI.Barrier(subgroup)

            # A representative staged composition (analysis → diagonal →
            # synthesis) must likewise remain inside the subgroup.
            prototype = create_spatial_array(cfg; comm=subgroup)
            fill!(parent(prototype), 0.25 + 0.05color)
            composed = dist_scalar_laplacian(
                cfg, prototype; prototype_θφ=prototype,
            )
            # The public composition returns local storage, so the successful
            # result plus the subgroup barrier is the communicator assertion:
            # a WORLD validation would mix the deliberately different cfgs.
            @test composed isa Matrix{Float64}
            @test all(isfinite, parent(composed))
            MPI.Barrier(subgroup)

            sentinel = ComplexF64(73, -19)
            output = similar(spectral)
            fill!(parent(output), sentinel)
            wrong = if MPI.Comm_rank(subgroup) == 0
                candidate = PencilArray{ComplexF64}(
                    undef,
                    Pencil(
                        (cfg.lmax + 1, cfg.mmax + 1), (2,), MPI.COMM_SELF,
                    ),
                )
                fill!(parent(candidate), sentinel)
                candidate
            else
                output
            end
            @test _all_ranks_catch(
                subgroup; message_contains="communicator mismatch",
            ) do
                divergence_from_spheroidal!(cfg, wrong, spectral)
            end
            @test all(==(sentinel), parent(wrong))
            MPI.Barrier(subgroup)
            Base.get_extension(SHTnsKit, :SHTnsKitParallelExt)._safe_comm_free(
                subgroup,
            )
        end
    else
        @test_skip MPI.Comm_size(MPI.COMM_WORLD) == 4
    end
end

if isempty(ARGS) || "shtns37_fixtures" in ARGS
    test_shtns37_mpi_fixtures(MPI.COMM_WORLD)
end

MPI.Barrier(MPI.COMM_WORLD)
