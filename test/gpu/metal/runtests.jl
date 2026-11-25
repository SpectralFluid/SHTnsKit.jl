using Test, Random, SHTnsKit, Metal

Metal.functional() || error("The Metal tests require a functional Apple GPU")
Metal.allowscalar(false)
println("Metal ", pkgversion(Metal), " on ", Metal.device().name)

function check_device(actual, expected)
    @test actual isa MtlArray
    @test eltype(actual) === eltype(expected)
    @test Array(actual) ≈ expected rtol=4f-5 atol=3f-6
end
check_components(actual, expected) = foreach(check_device, actual, expected)

@testset "Metal device routing" begin
    @test Base.get_extension(SHTnsKit, :SHTnsKitMetalExt) !== nothing
    @test get_device() isa GPU
    field = ones(Float32, 5, 8)
    cfg = create_gauss_config(3, 5; nlon=8)
    d = to_device(GPU(), field)
    @test on_device(d) isa GPU
    @test to_device(GPU(), d) === d
    @test to_device(CPU(), d) == field
    check_device(analysis(GPU(), cfg, field; prototype=d), analysis(cfg, field))
    s, t = analysis_sphtor(cfg, field, field)
    check_components(analysis_sphtor(cfg, field, d), (s,t))
    check_components(synthesis_sphtor(cfg, MtlArray(s), t), synthesis_sphtor(cfg, s, t))
    check_components(synthesis_sphtor_cplx(cfg, s, MtlArray(t)), synthesis_sphtor_cplx(cfg, s, t))
    @test_throws ArgumentError analysis(GPU(), cfg, Float64.(field); prototype=d)
    @test_throws ArgumentError analysis(cfg, MtlArray(Float16.(field)))
    @test_throws ArgumentError analysis(CPU(), cfg, d)
    @test_throws DimensionMismatch analysis(cfg, MtlArray(ones(Float32, 2, 2)))
    @test_throws ArgumentError analysis(cfg, ComplexF32.(d); use_rfft=true)
    @test_throws ArgumentError synthesis(cfg, analysis(cfg, d); real_output=false, use_rfft=true)
    @test_throws ArgumentError analysis(cfg, d; fft_scratch=zeros(ComplexF32, size(d)))
    dest = similar(d)
    SHTnsKit.@sht_loop dest[I] = 3f0 * d[I] over I ∈ CartesianIndices(dest)
    check_device(dest, 3f0 .* field)
    @test isnothing(gpu_clear_cache!(GPU(); prototype=d))
end

@testset "Band-limited roundtrip" begin
    cfg = create_gauss_config(8, 10)
    a = zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
    a[3, 2] = 0.5f0 - 0.25f0im
    field = synthesis(cfg, MtlArray(a))
    recovered = analysis(cfg, field)
    @test field isa MtlMatrix{Float32}
    @test recovered isa MtlMatrix{ComplexF32}
    @test maximum(abs, Array(recovered) - a) < 2f-6
end

@testset "Packed scalar layouts and degree truncation" begin
    rng = MersenneTwister(62)
    for mres in (1, 2)
        cfg = create_gauss_config(7, 10; mmax=4, mres, nlon=15, norm=:schmidt, cs_phase=false)
        f = randn(rng, Float32, cfg.nlat, cfg.nlon)
        z = complex.(f, randn(rng, Float32, size(f)))
        packed = analysis_packed(cfg, vec(f))
        fd = MtlArray(vec(f))
        pd = analysis_packed(cfg, fd)
        check_device(pd, packed)
        check_device(synthesis_packed(GPU(), cfg, pd), synthesis_packed(cfg, packed))
        dense = SHTnsKit.unpack_lm(cfg, pd)
        check_device(dense, SHTnsKit.unpack_lm(cfg, packed))
        check_device(SHTnsKit.pack_lm(cfg, dense), packed)
        padded = fill(ComplexF32(21), size(dense))
        padded_d = MtlArray(padded)
        SHTnsKit.unpack_lm!(padded, cfg, packed)
        @test SHTnsKit.unpack_lm!(padded_d, cfg, pd) === padded_d
        check_device(padded_d, padded)
        @test SHTnsKit.pack_lm!(pd, cfg, dense) === pd
        check_device(pd, packed)
        zd = MtlArray(z)
        if mres == 1
            alm = analysis_packed_cplx(cfg, z)
            check_device(analysis_packed_cplx(GPU(), cfg, zd), alm)
            check_device(synthesis_packed_cplx(cfg, MtlArray(alm)), synthesis_packed_cplx(cfg, alm))
        else
            @test_throws ArgumentError analysis_packed_cplx(cfg, zd)
        end
        for cap in (0, 3, cfg.lmax)
            check_device(analysis_packed_l(cfg, fd, cap), analysis_packed_l(cfg, vec(f), cap))
            check_device(synthesis_packed_l(cfg, pd, cap), synthesis_packed_l(cfg, packed, cap))
            if mres == 1
                alm = analysis_packed_cplx(cfg, z)
                check_device(analysis_packed_cplx_l(cfg, zd, cap), analysis_packed_cplx_l(cfg, z, cap))
                check_device(synthesis_packed_cplx_l(cfg, MtlArray(alm), cap),
                             synthesis_packed_cplx_l(cfg, alm, cap))
            end
        end
        @test_throws DimensionMismatch synthesis_packed(cfg, pd[2:end])
        @test_throws ArgumentError analysis_packed_l(cfg, fd, cfg.lmax+1)
    end
end

@testset "Concurrent use of cached plan buffers ($nlon longitudes)" for nlon in (11, 16)
    cfg = create_gauss_config(5, 7; nlon)
    rng = MersenneTwister(87)
    fields = [randn(rng, Float32, cfg.nlat, cfg.nlon) for _ in 1:4]
    inputs = MtlArray.(fields)
    plan = SHTPlan(cfg; use_rfft=true)
    jobs = map(inputs) do f
        Threads.@spawn begin
            a = similar(f, ComplexF32, (cfg.lmax+1, cfg.mmax+1))
            y = similar(f)
            for _ in 1:3
                analysis!(plan, a, f)
                synthesis!(plan, y, a)
            end
            a, y
        end
    end
    for (job, f) in zip(jobs, fields)
        a, y = fetch(job)
        expected = analysis(cfg, f)
        check_device(a, expected)
        check_device(y, synthesis(cfg, expected))
    end
end

@testset "Near-pole latitude precision and cached configuration changes" begin
    cfg = create_gauss_fly_config(64, 66; mmax=1, nlon=5)
    coeff = zeros(ComplexF32, cfg.lmax+1, cfg.mmax+1)
    coeff[end, 2] = 1
    d = MtlArray(coeff)
    for x in (1.0 - 1e-8, -1.0 + 3e-9)
        cfg.x[1] = x
        expected = synthesis_cplx(cfg, coeff)
        @test abs(expected[1, 1]) > 0.001f0
        actual = synthesis_cplx(cfg, d)
        check_device(actual, expected)
        @test Array(actual)[1, 1] ≈ expected[1, 1] rtol=3f-6
    end
    field = randn(MersenneTwister(8), Float32, cfg.nlat, cfg.nlon)
    fd = MtlArray(field)
    for mode in (:quad, :dft)
        cfg.phi_scale = mode
        cfg.w[2] *= 0.75
        check_device(analysis(cfg, fd), analysis(cfg, field))
        check_device(synthesis(cfg, d), synthesis(cfg, coeff))
    end
end

@testset "Spectral vector operators" begin
    cfg = create_gauss_config(6, 8; mres=2)
    # Include forbidden orders, l<m and l=0 to check the mask explicitly.
    a = randn(MersenneTwister(31), ComplexF32, cfg.lmax+1, cfg.mmax+1)
    for (op, inplace) in ((divergence_from_spheroidal, divergence_from_spheroidal!),
                          (vorticity_from_toroidal, vorticity_from_toroidal!),
                          (spheroidal_from_divergence, spheroidal_from_divergence!),
                          (toroidal_from_vorticity, toroidal_from_vorticity!))
        d = MtlArray(a)
        expected = op(cfg, a)
        check_device(op(GPU(), cfg, d), expected)
        @test inplace(GPU(), cfg, d, d) === d
        check_device(d, expected)
    end
end

@testset "Full-grid scalar, vector and QST transforms" begin
    rng = MersenneTwister(401)
    grids = (
        create_gauss_config(5, 9; nlon=16),
        create_gauss_config(5, 9; nlon=13, mres=2, norm=:schmidt, real_norm=true,
                            cs_phase=false, robert_form=true),
        create_regular_config(5, 12; nlon=15, norm=:fourpi, include_poles=true),
        create_gauss_config(0, 1; nlon=1),
    )
    set_south_pole_first!(grids[3])
    for cfg in grids
        dims = (cfg.nlat, cfg.nlon)
        fields = ntuple(_ -> randn(rng, Float32, dims), 3)
        device = map(MtlArray, fields)
        a = analysis(cfg, fields[1])
        s, t = analysis_sphtor(cfg, fields[2], fields[3])
        coeff = map(MtlArray, (a, s, t))
        for half in (false, true)
            check_device(analysis(cfg, device[1]; use_rfft=half), a)
            check_device(synthesis(cfg, coeff[1]; use_rfft=half), synthesis(cfg, a))
            check_components(analysis_qst(cfg, device...; use_rfft=half), (a, s, t))
            check_components(synthesis_qst(cfg, coeff...; use_rfft=half), synthesis_qst(cfg, a, s, t))
            plan = SHTPlan(cfg; use_rfft=half)
            out = similar(coeff[1])
            f = similar(device[1])
            @test analysis!(GPU(), plan, out, device[1]) === out
            check_device(out, a)
            @test synthesis!(plan, f, out) === f
            check_device(f, synthesis(cfg, a))
            so, to = similar(out), similar(out)
            vo, wo = similar(f), similar(f)
            @test analysis_sphtor!(plan, so, to, device[2], device[3]) === (so, to)
            check_components((so, to), (s, t))
            @test synthesis_sphtor!(plan, vo, wo, so, to) === (vo, wo)
            check_components((vo, wo), synthesis_sphtor(cfg, s, t))
            if half && isodd(cfg.nlon) && cfg.nlon > 1
                analysis_only = SHTPlan(cfg; use_rfft=true)
                @test (Metal.@allocated analysis!(analysis_only, out, device[1])) <
                      length(f) * sizeof(ComplexF32)
                # A warmed inverse must not allocate another complex Fourier grid.
                @test (Metal.@allocated synthesis!(plan, f, out)) < sizeof(f)
                @test (Metal.@allocated synthesis_sphtor!(plan, vo, wo, so, to)) < sizeof(f)
                # The next forward transform uses the prefix of the larger workspace.
                analysis_sphtor!(plan, so, to, device[3], device[2])
                check_components((so, to), analysis_sphtor(cfg, fields[3], fields[2]))
                @test_throws ArgumentError synthesis!(plan, f, out; real_output=false)
                @test_throws ArgumentError synthesis_sphtor!(plan, vo, wo, so, to; real_output=false)
            end
        end
        z = complex.(fields[1], fields[2])
        check_device(analysis(cfg, MtlArray(z)), analysis(cfg, z))
        check_device(synthesis_cplx(cfg, coeff[1]), synthesis_cplx(cfg, a))
        check_components(synthesis_qst_cplx(cfg, coeff...), synthesis_qst_cplx(cfg, a, s, t))
        check_components(analysis_qst_cplx(cfg, MtlArray(z), MtlArray(z), MtlArray(z)),
                         analysis_qst_cplx(cfg, z, z, z))
        check_components(synthesis_sph_cplx(cfg, coeff[2]), synthesis_sph_cplx(cfg, s))
        check_components(synthesis_tor(cfg, coeff[3]), synthesis_tor(cfg, t))
        d = MtlArray(a)
        @test gpu_apply_laplacian!(cfg, d) === d
        check_device(d, a .* Float32.(-((0:cfg.lmax) .* (1:cfg.lmax+1))))
        @test isnothing(gpu_clear_cache!(GPU(); prototype=d))
    end
end

@testset "Offset and strided device views" begin
    rng = MersenneTwister(12)
    cfg = create_gauss_config(4, 8; nlon=12)
    storage = randn(rng, Float32, cfg.nlat, 2cfg.nlon)
    dev = MtlArray(storage)
    for columns in (cfg.nlon+1:2cfg.nlon, 2:2:2cfg.nlon)
        f = view(dev, :, columns)
        expected = analysis(cfg, storage[:, columns])
        @test on_device(f) isa GPU
        for half in (false, true)
            check_device(analysis(cfg, f; use_rfft=half), expected)
            out = similar(dev)
            fill!(out, -17f0)
            v = view(out, :, columns)
            @test synthesis!(cfg, v, MtlArray(expected); use_rfft=half) === v
            @test Array(v) ≈ synthesis(cfg, expected) rtol=4f-5 atol=3f-6
            @test all(==(-17f0), Array(out)[:, setdiff(axes(out, 2), columns)])
        end
    end
    @test Array(dev) == storage
    flat = vec(dev)
    for indices in (1:cfg.nspat, cfg.nspat+1:2cfg.nspat, 2:2:2cfg.nspat)
        f = view(flat, indices)
        packed = analysis_packed(cfg, f)
        expected = analysis_packed(cfg, vec(storage)[indices])
        check_device(packed, expected)
        check_device(synthesis_packed(cfg, packed), synthesis_packed(cfg, expected))
    end
end

@testset "Batched transforms and caller-owned scratch" begin
    rng = MersenneTwister(102)
    for nlon in (15, 16)
        cfg = create_gauss_config(5, 9; nlon, mres=2, norm=:fourpi)
        host = ntuple(_ -> randn(rng, Float32, cfg.nlat, nlon, 3), 3)
        device = map(MtlArray, host)
        expected = analysis_qst_batch(cfg, host...)
        q, s, t = analysis_qst_batch(GPU(), cfg, device...)
        check_components((q, s, t), expected)
        check_components(synthesis_qst_batch(cfg, q, s, t), synthesis_qst_batch(cfg, expected...))
        if iseven(nlon)
            full_fft_bytes = 3 * length(device[1]) * sizeof(ComplexF32)
            @test (Metal.@allocated analysis_qst_batch(cfg, device...)) <
                  sum(sizeof, (q, s, t)) + full_fft_bytes
            @test (Metal.@allocated synthesis_qst_batch(cfg, q, s, t)) <
                  sum(sizeof, device) + full_fft_bytes
        end
        check_components(synthesis_qst_batch_cplx(cfg, q, s, t), synthesis_qst_batch_cplx(cfg, expected...))
        check_components(analysis_sphtor_batch(cfg, device[2], device[3]), expected[2:3])
        # A dense radial field and a strided tangential field may take different FFT paths.
        reversed = view(device[2], :, nlon:-1:1, :)
        check_components(analysis_qst_batch(cfg, device[1], reversed, device[3]),
                         analysis_qst_batch(cfg, host[1], host[2][:, end:-1:1, :], host[3]))
        check_components(synthesis_sphtor_batch_cplx(cfg, s, t), synthesis_sphtor_batch_cplx(cfg, expected[2:3]...))
        @test_throws DimensionMismatch analysis_qst_batch(cfg, device[1], device[2], device[3][:, :, 1:1])
        for half in (false, true)
            a = analysis_batch(cfg, device[1]; use_rfft=half)
            check_device(a, expected[1])
            f = synthesis_batch(cfg, a; use_rfft=half)
            field = synthesis_batch(cfg, expected[1])
            check_device(f, field)
            scratch = MtlArray{ComplexF32}(undef, cfg.nlat, half ? nlon÷2+1 : nlon, 3)
            @test analysis_batch!(cfg, a, device[1]; use_rfft=half, fft_batch=scratch) === a
            check_device(a, expected[1])
            fill!(scratch, ComplexF32(NaN))
            @test synthesis_batch!(GPU(), cfg, f, a; use_rfft=half, fft_batch=scratch) === f
            check_device(f, field)
        end
    end
end
