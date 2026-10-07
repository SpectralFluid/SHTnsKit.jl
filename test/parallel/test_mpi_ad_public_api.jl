#!/usr/bin/env julia

# Reverse-mode AD through the public transforms and gradient helpers on
# PencilArrays, checked against the serial gradients of the same loss.
# Run with: mpiexec -n 2 julia --project test/parallel/test_mpi_ad_public_api.jl

using MPI
MPI.Init()

using ChainRulesCore
using ForwardDiff
using PencilArrays
using PencilFFTs
using Random
using SHTnsKit
using Test
using Zygote

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

function distribute(pen::Pencil, values::AbstractMatrix)
    result = PencilArray{eltype(values)}(undef, pen)
    owned = PencilArrays.global_view(result)
    for index in CartesianIndices(owned)
        owned[index] = values[index]
    end
    return result
end

function collect_global(value::PencilArray)
    full = zeros(eltype(value), size_global(value))
    owned = PencilArrays.global_view(value)
    for index in CartesianIndices(owned)
        full[index] = owned[index]
    end
    return MPI.Allreduce(full, +, comm)
end

@testset "distributed AD through the public API ($nprocs ranks)" begin
    cfg = create_gauss_config(6, 9; nlon=13)
    rng = MersenneTwister(3)
    F = randn(rng, cfg.nlat, cfg.nlon)
    G = randn(rng, cfg.nlat, cfg.nlon)
    A = zeros(ComplexF64, cfg.lmax + 1, cfg.mmax + 1)
    for m in 0:cfg.mmax, l in m:cfg.lmax
        A[l + 1, m + 1] = m == 0 ? randn(rng) : complex(randn(rng), randn(rng))
    end
    B = 0.5 .* A
    spatial = Pencil((cfg.nlat, cfg.nlon), (1,), comm)
    spectral = Pencil((cfg.lmax + 1, cfg.mmax + 1), (2,), comm)
    f, g = distribute(spatial, F), distribute(spatial, G)
    a, b = distribute(spectral, A), distribute(spectral, B)
    tol = 1e-12

    @testset "energy gradient helpers keep the input's layout" begin
        gradient = zgrad_scalar_energy(cfg, f)
        @test gradient isa PencilArray
        @test PencilArrays.pencil(gradient) === PencilArrays.pencil(f)
        @test collect_global(gradient) ≈ zgrad_scalar_energy(cfg, F) rtol=tol
        gt, gp = zgrad_vector_energy(cfg, f, g)
        st, sp = zgrad_vector_energy(cfg, F, G)
        @test collect_global(gt) ≈ st rtol=tol
        @test collect_global(gp) ≈ sp rtol=tol
        @test collect_global(zgrad_enstrophy_Tlm(cfg, a)) ≈ zgrad_enstrophy_Tlm(cfg, A) rtol=tol

        # ForwardDiff cannot run the distributed kernels; the field is gathered.
        gradient = fdgrad_scalar_energy(cfg, f)
        @test gradient isa PencilArray
        @test collect_global(gradient) ≈ fdgrad_scalar_energy(cfg, F) rtol=tol
        gt, gp = fdgrad_vector_energy(cfg, f, g)
        st, sp = fdgrad_vector_energy(cfg, F, G)
        @test collect_global(gt) ≈ st rtol=tol
        @test collect_global(gp) ≈ sp rtol=tol
    end

    # Each rank's loss uses its own block of a distributed output; the total
    # loss is their sum, which is the serial loss.
    @testset "scalar transforms" begin
        gd = Zygote.gradient(x -> sum(abs2, parent(analysis(cfg, x))), f)[1]
        gs = Zygote.gradient(x -> sum(abs2, analysis(cfg, x)), F)[1]
        @test gd isa PencilArray
        @test collect_global(gd) ≈ gs rtol=tol

        # A replicated output takes the same loss on every rank.
        gd = Zygote.gradient(x -> sum(abs2, analysis(cfg, x; return_pencil=false)), f)[1]
        @test collect_global(gd) ≈ gs rtol=tol

        gd = Zygote.gradient(c -> sum(abs2, parent(synthesis(cfg, c; prototype_θφ=f))), a)[1]
        gs = Zygote.gradient(c -> sum(abs2, synthesis(cfg, c)), A)[1]
        @test gd isa PencilArray
        @test collect_global(gd) ≈ gs rtol=tol
    end

    @testset "vector and QST transforms" begin
        gd = Zygote.gradient(f, g) do x, y
            S, T = analysis_sphtor(cfg, x, y)
            sum(abs2, parent(S)) + 2sum(abs2, parent(T))
        end
        gs = Zygote.gradient(F, G) do x, y
            S, T = analysis_sphtor(cfg, x, y)
            sum(abs2, S) + 2sum(abs2, T)
        end
        @test all(k -> collect_global(gd[k]) ≈ gs[k], 1:2)

        gd = Zygote.gradient(a, b) do s, t
            X, Y = synthesis_sphtor(cfg, s, t; prototype_θφ=f)
            sum(abs2, parent(X)) + 2sum(abs2, parent(Y))
        end
        gs = Zygote.gradient(A, B) do s, t
            X, Y = synthesis_sphtor(cfg, s, t)
            sum(abs2, X) + 2sum(abs2, Y)
        end
        @test all(k -> collect_global(gd[k]) ≈ gs[k], 1:2)

        gd = Zygote.gradient(f, g, f) do x, y, z
            Q, S, T = analysis_qst(cfg, x, y, z)
            sum(abs2, parent(Q)) + sum(abs2, parent(S)) + 2sum(abs2, parent(T))
        end
        gs = Zygote.gradient(F, G, F) do x, y, z
            Q, S, T = analysis_qst(cfg, x, y, z)
            sum(abs2, Q) + sum(abs2, S) + 2sum(abs2, T)
        end
        @test all(k -> collect_global(gd[k]) ≈ gs[k], 1:3)

        gd = Zygote.gradient(a, b, a) do q, s, t
            X, Y, Z = synthesis_qst(cfg, q, s, t; prototype_θφ=f)
            sum(abs2, parent(X)) + sum(abs2, parent(Y)) + 2sum(abs2, parent(Z))
        end
        gs = Zygote.gradient(A, B, A) do q, s, t
            X, Y, Z = synthesis_qst(cfg, q, s, t)
            sum(abs2, X) + sum(abs2, Y) + 2sum(abs2, Z)
        end
        @test all(k -> collect_global(gd[k]) ≈ gs[k], 1:3)
    end

    @testset "rules accept every primal keyword" begin
        ltr = cfg.lmax - 2
        gd = Zygote.gradient(c -> sum(abs2, SHTnsKit.dist_synthesis(
            cfg, c; prototype_θφ=f, ltr, comm)), a)[1]
        truncated = copy(A)
        truncated[(ltr + 2):end, :] .= 0
        reference = Zygote.gradient(c -> sum(abs2, synthesis(cfg, c)), truncated)[1]
        reference[(ltr + 2):end, :] .= 0
        # The distributed output is per rank, so the summed loss is the serial one.
        @test collect_global(gd) ≈ reference rtol=tol
        gd = Zygote.gradient(x -> sum(abs2, SHTnsKit.dist_analysis(cfg, x; comm)), f)[1]
        @test collect_global(gd) ≈ Zygote.gradient(x -> sum(abs2, analysis(cfg, x)), F)[1] rtol=tol
    end

    @testset "replicated NaN cotangents are not rank-varying" begin
        _, pullback = ChainRulesCore.rrule(SHTnsKit.dist_analysis, cfg, f)
        cotangent = fill(complex(NaN, 0.0), cfg.lmax + 1, cfg.mmax + 1)
        @test all(isnan, parent(pullback(cotangent)[3]))
    end

    @testset "replicated energy cotangents must agree across ranks" begin
        # Blocks scaled by different cotangents would not form one gradient.
        for (energy, args) in ((energy_scalar, (a,)), (energy_vector, (a, b)),
                               (enstrophy, (a,)))
            _, pullback = ChainRulesCore.rrule(energy, cfg, args...)
            nprocs > 1 && @test_throws ArgumentError pullback(1.0 + rank)
            tangents = pullback(2.0)
            @test length(tangents) == 2 + length(args)
            @test all(tangent -> tangent isa PencilArray, tangents[3:end])
            @test pullback(ZeroTangent())[3] isa ChainRulesCore.AbstractZero
        end
    end

    @testset "PencilArray transforms without a distributed adjoint fail clearly" begin
        batch_pen = Pencil((cfg.nlat, cfg.nlon, 2), (1,), comm)
        batch = PencilArray{Float64}(undef, batch_pen)
        fill!(parent(batch), 1.0)
        @test_throws ArgumentError Zygote.gradient(x -> sum(abs2, analysis_batch(cfg, x)), batch)
    end
end
