using Test, Random, SHTnsKit, Metal

Metal.functional() || error("The Metal tests require a functional Apple GPU")
# Shared storage, the default from Metal 1.12, ignores allowscalar(false), so a
# silent host fallback would still pass. Project.toml selects private storage.
MtlArray(zeros(Float32, 1)) isa MtlArray{Float32,1,Metal.PrivateStorage} ||
    error("set Metal's default_storage preference to \"private\" for these tests")
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

@testset "Point and latitude evaluation are unavailable" begin
    cfg = create_gauss_config(4, 6)
    dense = MtlArray(zeros(ComplexF32, cfg.lmax + 1, cfg.mmax + 1))
    packed = MtlArray(zeros(ComplexF32, cfg.nlm))
    for device in ((), (GPU(),))
        @test_throws BackendUnavailableError synthesis_point(device..., cfg, dense, 0.3, 0.2)
        @test_throws BackendUnavailableError synthesis_point_cplx(device..., cfg, packed, 0.3, 0.2)
        @test_throws BackendUnavailableError SH_to_lat(device..., cfg, packed, 0.3)
        @test_throws BackendUnavailableError SH_to_lat_cplx(device..., cfg, packed, 0.3)
        @test_throws BackendUnavailableError SHqst_to_point(device..., cfg, packed, packed, packed, 0.3, 0.2)
        @test_throws BackendUnavailableError SHqst_to_lat(device..., cfg, packed, packed, packed, 0.3)
        @test_throws BackendUnavailableError SH_to_grad_point(device..., cfg, packed, packed, 0.3, 0.2)
    end
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
        # CPU() is strict about storage, as for the transforms.
        @test_throws ArgumentError op(CPU(), cfg, d)
        @test_throws ArgumentError inplace(CPU(), cfg, similar(d), d)
        @test_throws ArgumentError inplace(CPU(), cfg, similar(a), d)
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
                # A warmed inverse must not allocate another complex Fourier
                # grid, even after a collection.
                GC.gc()
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

@testset "Degree-truncated, single-order and axisymmetric transforms" begin
    rng = MersenneTwister(77)
    grids = (
        create_gauss_config(7, 10; nlon=16, mres=2, norm=:schmidt, cs_phase=false),
        create_gauss_config(6, 9; nlon=13, robert_form=true, real_norm=true),
    )
    for cfg in grids
        fields = ntuple(_ -> randn(rng, Float32, cfg.nlat, cfg.nlon), 3)
        device = map(MtlArray, fields)
        q, s, t = analysis_qst(cfg, fields...)
        coeff = map(MtlArray, (q, s, t))
        for ltr in (1, 4, cfg.lmax), placement in ((), (GPU(),))
            check_components(analysis_sphtor_l(placement..., cfg, device[2], device[3], ltr),
                             analysis_sphtor_l(cfg, fields[2], fields[3], ltr))
            check_components(synthesis_sphtor_l(placement..., cfg, coeff[2], coeff[3], ltr),
                             synthesis_sphtor_l(cfg, s, t, ltr))
            check_components(synthesis_sphtor_l_cplx(placement..., cfg, coeff[2], coeff[3], ltr),
                             synthesis_sphtor_l_cplx(cfg, s, t, ltr))
            check_components(synthesis_sph_l(placement..., cfg, coeff[2], ltr), synthesis_sph_l(cfg, s, ltr))
            check_components(synthesis_tor_l_cplx(placement..., cfg, coeff[3], ltr),
                             synthesis_tor_l_cplx(cfg, t, ltr))
            check_components(analysis_qst_l(placement..., cfg, device..., ltr),
                             analysis_qst_l(cfg, fields..., ltr))
            check_components(synthesis_qst_l(placement..., cfg, coeff..., ltr),
                             synthesis_qst_l(cfg, q, s, t, ltr))
            check_device(analysis_axisym_l(placement..., cfg, MtlArray(fields[1][:, 1]), ltr),
                         analysis_axisym_l(cfg, fields[1][:, 1], ltr))
            check_device(synthesis_axisym_l(placement..., cfg, MtlArray(q[:, 1]), ltr),
                         synthesis_axisym_l(cfg, q[:, 1], ltr))
        end
        check_device(analysis_axisym(cfg, MtlArray(fields[1][:, 1])), analysis_axisym(cfg, fields[1][:, 1]))
        check_device(synthesis_axisym(GPU(), cfg, MtlArray(q[:, 1])), synthesis_axisym(cfg, q[:, 1]))
        for im in (0, 1, cfg.mmax ÷ cfg.mres)
            m = im * cfg.mres
            ltr = max(m, 5)
            modes = ntuple(_ -> randn(rng, ComplexF32, cfg.nlat), 3)
            coefficients = ntuple(_ -> randn(rng, ComplexF32, ltr - m + 1), 3)
            mode_d, coefficients_d = map(MtlArray, modes), map(MtlArray, coefficients)
            for placement in ((), (GPU(),))
                check_device(analysis_packed_ml(placement..., cfg, im, mode_d[1], ltr),
                             analysis_packed_ml(cfg, im, modes[1], ltr))
                check_device(synthesis_packed_ml(placement..., cfg, im, coefficients_d[1], ltr),
                             synthesis_packed_ml(cfg, im, coefficients[1], ltr))
                check_components(analysis_sphtor_ml(placement..., cfg, im, mode_d[2], mode_d[3], ltr),
                                 analysis_sphtor_ml(cfg, im, modes[2], modes[3], ltr))
                check_components(synthesis_sphtor_ml(placement..., cfg, im, coefficients_d[2], coefficients_d[3], ltr),
                                 synthesis_sphtor_ml(cfg, im, coefficients[2], coefficients[3], ltr))
                check_components(synthesis_tor_ml(placement..., cfg, im, coefficients_d[3], ltr),
                                 synthesis_tor_ml(cfg, im, coefficients[3], ltr))
                check_components(analysis_qst_ml(placement..., cfg, im, mode_d..., ltr),
                                 analysis_qst_ml(cfg, im, modes..., ltr))
                check_components(synthesis_qst_ml(placement..., cfg, im, coefficients_d..., ltr),
                                 synthesis_qst_ml(cfg, im, coefficients..., ltr))
            end
        end
        @test_throws ArgumentError analysis_sphtor_l(cfg, device[2], device[3], cfg.lmax + 1)
        @test_throws ArgumentError synthesis_axisym_l(cfg, MtlArray(q[1:2, 1]), 2)
        @test_throws DimensionMismatch analysis_packed_ml(cfg, 0, MtlArray(zeros(ComplexF32, 2)), 3)
    end
end

@testset "Mixed CPU and Metal operands run on the GPU" begin
    cfg = create_gauss_config(5, 8; nlon=12)
    rng = MersenneTwister(5)
    host = ntuple(_ -> randn(rng, Float32, cfg.nlat, cfg.nlon, 2), 3)
    dev = map(MtlArray, host)
    expected = analysis_qst_batch(cfg, host...)
    check_components(analysis_sphtor_batch(cfg, host[2], dev[3]), expected[2:3])
    check_components(analysis_qst_batch(cfg, dev[1], host[2], dev[3]), expected)
    check_components(synthesis_qst_batch(cfg, expected[1], MtlArray(expected[2]), expected[3]),
                     synthesis_qst_batch(cfg, expected...))
    check_components(synthesis_sphtor_batch_cplx(cfg, MtlArray(expected[2]), expected[3]),
                     synthesis_sphtor_batch_cplx(cfg, expected[2:3]...))
    v = ntuple(_ -> randn(rng, Float32, cfg.nlat, cfg.nlon), 3)
    check_components(analysis_sphtor_l(cfg, v[1], MtlArray(v[2]), 3), analysis_sphtor_l(cfg, v[1], v[2], 3))
    check_components(analysis_qst_l(cfg, MtlArray(v[1]), v[2], v[3], 3), analysis_qst_l(cfg, v..., 3))
    modes = ntuple(_ -> randn(rng, ComplexF32, cfg.nlat), 2)
    check_components(analysis_sphtor_ml(cfg, 1, MtlArray(modes[1]), modes[2], 4),
                     analysis_sphtor_ml(cfg, 1, modes..., 4))
end

@testset "QST components share the promoted precision" begin
    # Metal stores half precision, so a Float16 component is the one mix it can
    # see. As on the CPU, it is widened, on the device, to the others' Float32.
    cfg = create_gauss_config(5, 8; nlon=12)
    rng = MersenneTwister(13)
    v = ntuple(_ -> randn(rng, Float32, cfg.nlat, cfg.nlon), 3)
    a = analysis_qst(cfg, v...)
    for narrow in 1:3
        narrowed(x, T) = ntuple(i -> i == narrow ? T.(x[i]) : x[i], 3)
        fields = narrowed(v, Float16)
        check_components(analysis_qst(cfg, map(MtlArray, fields)...), analysis_qst(cfg, fields...))
        check_components(analysis_qst_l(cfg, map(MtlArray, fields)..., 3),
                         analysis_qst_l(cfg, fields..., 3))
        spectra = narrowed(a, ComplexF16)
        check_components(synthesis_qst(cfg, map(MtlArray, spectra)...), synthesis_qst(cfg, spectra...))
        check_components(synthesis_qst_l(cfg, map(MtlArray, spectra)..., 3),
                         synthesis_qst_l(cfg, spectra..., 3))
        modes = narrowed(map(x -> x[2:end, 2], a), ComplexF16)
        profiles = synthesis_qst_ml(cfg, 1, modes..., cfg.lmax)
        check_components(synthesis_qst_ml(cfg, 1, map(MtlArray, modes)..., cfg.lmax), profiles)
        profiles = narrowed(profiles, ComplexF16)
        check_components(analysis_qst_ml(cfg, 1, map(MtlArray, profiles)..., cfg.lmax),
                         analysis_qst_ml(cfg, 1, profiles..., cfg.lmax))
    end
    # A host component, and a nested wrapper of device storage, widen there too.
    r = Float16.(v[1])
    nested = view(PermutedDimsArray(MtlArray(permutedims(r)), (2, 1)), :, :)
    expected = analysis_qst(cfg, r, v[2], v[3])
    for x in (r, nested)
        check_components(analysis_qst(cfg, x, MtlArray(v[2]), MtlArray(v[3])), expected)
    end
    # Batches, and the vector transforms' two components, widen the same way.
    batch = ntuple(_ -> randn(rng, Float32, cfg.nlat, cfg.nlon, 2), 3)
    spectra = analysis_qst_batch(cfg, batch...)
    for narrow in 1:3
        fields = ntuple(i -> i == narrow ? Float16.(batch[i]) : batch[i], 3)
        check_components(analysis_qst_batch(cfg, map(MtlArray, fields)...),
                         analysis_qst_batch(cfg, fields...))
        coefficients = ntuple(i -> i == narrow ? ComplexF16.(spectra[i]) : spectra[i], 3)
        check_components(synthesis_qst_batch(cfg, map(MtlArray, coefficients)...),
                         synthesis_qst_batch(cfg, coefficients...))
        check_components(synthesis_qst_batch_cplx(cfg, map(MtlArray, coefficients)...),
                         synthesis_qst_batch_cplx(cfg, coefficients...))
    end
    h, hb = Float16.(v[2]), Float16.(batch[2])
    check_components(analysis_sphtor(cfg, MtlArray(h), MtlArray(v[3])), analysis_sphtor(cfg, h, v[3]))
    check_components(analysis_sphtor_batch(cfg, MtlArray(hb), MtlArray(batch[3])),
                     analysis_sphtor_batch(cfg, hb, batch[3]))
    s16 = ComplexF16.(a[2])
    check_components(synthesis_sphtor(cfg, MtlArray(s16), MtlArray(a[3])), synthesis_sphtor(cfg, s16, a[3]))
end

@testset "Nested device wrappers stay on the GPU" begin
    cfg = create_gauss_config(4, 6; nlon=9)
    host = randn(MersenneTwister(3), Float32, cfg.nlon, cfg.nlat)
    A = MtlArray(host)
    field = permutedims(host)
    a = analysis(cfg, field)
    s, t = analysis_sphtor(cfg, field, field)
    for x in (view(PermutedDimsArray(A, (2, 1)), :, :), view(A', :, :))
        @test on_device(x) isa GPU
        for half in (false, true)
            check_device(analysis(cfg, x; use_rfft=half), a)
        end
        check_components(analysis_sphtor(cfg, x, x), (s, t))
        check_components(analysis_qst(cfg, x, x, x), (a, s, t))
        # The other allocating families copy the wrapper into a dense device array.
        check_components(analysis_sphtor_l(cfg, x, x, 2), analysis_sphtor_l(cfg, field, field, 2))
        @test to_device(CPU(), x) == field
        # In-place transforms write outputs where they are, so they refuse it.
        out = MtlArray{ComplexF32}(undef, size(a))
        @test_throws BackendUnavailableError analysis!(cfg, out, x)
        @test_throws ArgumentError analysis!(SHTPlan(cfg), out, x)
    end
    B = MtlArray(permutedims(a))
    y = view(PermutedDimsArray(B, (2, 1)), :, :)
    check_device(synthesis(cfg, y), synthesis(cfg, a))
    check_device(synthesis_cplx(cfg, y), synthesis_cplx(cfg, a))
    @test_throws BackendUnavailableError synthesis!(cfg, MtlArray(field), y)

    # Batches, packed and single-order transforms, axisymmetric profiles.
    nested(h, perm) = view(PermutedDimsArray(MtlArray(permutedims(h, perm)), perm),
                           ntuple(_ -> Colon(), ndims(h))...)
    column(h) = view(PermutedDimsArray(MtlArray(reshape(h, 1, :)), (2, 1)), :, 1)
    fields = cat(field, 2f0 .* field; dims=3)
    spectra = analysis_batch(cfg, fields)
    check_device(analysis_batch(cfg, nested(fields, (2, 1, 3))), spectra)
    check_device(synthesis_batch(cfg, nested(spectra, (2, 1, 3))), synthesis_batch(cfg, spectra))
    check_device(synthesis_batch_cplx(cfg, nested(spectra, (2, 1, 3))), synthesis_batch_cplx(cfg, spectra))
    profile = field[:, 2]
    check_device(analysis_axisym(cfg, column(profile)), analysis_axisym(cfg, profile))
    check_device(analysis_axisym_l(cfg, column(profile), 2), analysis_axisym_l(cfg, profile, 2))
    q = analysis_axisym(cfg, profile)
    check_device(synthesis_axisym(cfg, column(q)), synthesis_axisym(cfg, q))
    mode = complex.(field[:, 3], field[:, 4])
    check_device(analysis_packed_ml(cfg, 1, column(mode), 3), analysis_packed_ml(cfg, 1, mode, 3))
    flat = vec(field)
    packed = analysis_packed_l(cfg, flat, 3)
    check_device(analysis_packed_l(cfg, column(flat), 3), packed)
    check_device(synthesis_packed_l(cfg, column(packed), 3), synthesis_packed_l(cfg, packed, 3))
    check_device(synthesis_packed(cfg, column(packed)), synthesis_packed(cfg, packed))
end

@testset "Disjoint parts of one buffer are not aliases" begin
    cfg = create_gauss_config(5, 8; nlon=12)
    rng = MersenneTwister(21)
    vt, vp = randn(rng, Float32, cfg.nlat, cfg.nlon), randn(rng, Float32, cfg.nlat, cfg.nlon)
    s, t = analysis_sphtor(cfg, vt, vp)
    plan = SHTPlan(cfg)
    spectra = MtlArray{ComplexF32}(undef, cfg.lmax + 1, cfg.mmax + 1, 2)
    so, to = view(spectra, :, :, 1), view(spectra, :, :, 2)
    analysis_sphtor!(plan, so, to, MtlArray(vt), MtlArray(vp))
    @test Array(so) ≈ s rtol=4f-5 atol=3f-6
    @test Array(to) ≈ t rtol=4f-5 atol=3f-6
    fields = MtlArray{Float32}(undef, cfg.nlat, cfg.nlon, 2)
    vo, wo = view(fields, :, :, 1), view(fields, :, :, 2)
    synthesis_sphtor!(plan, vo, wo, MtlArray(s), MtlArray(t))
    expected = synthesis_sphtor(cfg, s, t)
    @test Array(vo) ≈ expected[1] rtol=4f-5 atol=3f-6
    @test Array(wo) ≈ expected[2] rtol=4f-5 atol=3f-6
    # Interleaved columns and separate row blocks, which span the same bytes.
    n, L = cfg.mmax + 1, cfg.lmax + 1
    interleaved = MtlArray{ComplexF32}(undef, L, 2n)
    rows = MtlArray{ComplexF32}(undef, 2L, n)
    for (so, to) in ((view(interleaved, :, 1:2:2n), view(interleaved, :, 2:2:2n)),
                     (view(rows, 1:L, :), view(rows, (L + 1):2L, :)))
        analysis_sphtor!(plan, so, to, MtlArray(vt), MtlArray(vp))
        @test Array(so) ≈ s rtol=4f-5 atol=3f-6
        @test Array(to) ≈ t rtol=4f-5 atol=3f-6
    end
    grids = MtlArray{Float32}(undef, cfg.nlat, 2cfg.nlon)
    vo, wo = view(grids, :, 1:2:2cfg.nlon), view(grids, :, 2:2:2cfg.nlon)
    synthesis_sphtor!(plan, vo, wo, MtlArray(s), MtlArray(t))
    @test Array(vo) ≈ expected[1] rtol=4f-5 atol=3f-6
    @test Array(wo) ≈ expected[2] rtol=4f-5 atol=3f-6
    # Views that share entries are still aliases.
    @test_throws ArgumentError analysis_sphtor!(plan, view(rows, 1:L, :), view(rows, 2:(L + 1), :),
                                                MtlArray(vt), MtlArray(vp))
    # FFT scratch and output carved from one allocation.
    nbins, nspec = cfg.nlat * (cfg.nlon ÷ 2 + 1), (cfg.lmax + 1) * (cfg.mmax + 1)
    buffer = MtlArray{ComplexF32}(undef, nbins + nspec)
    scratch = reshape(view(buffer, 1:nbins), cfg.nlat, cfg.nlon ÷ 2 + 1)
    out = reshape(view(buffer, nbins+1:nbins+nspec), cfg.lmax + 1, cfg.mmax + 1)
    @test analysis!(cfg, out, MtlArray(vt); fft_scratch=scratch, use_rfft=true) === out
    @test Array(out) ≈ analysis(cfg, vt) rtol=4f-5 atol=3f-6
    overlapping = reshape(view(buffer, 1:nspec), cfg.lmax + 1, cfg.mmax + 1)
    @test_throws ArgumentError analysis!(cfg, overlapping, MtlArray(vt); fft_scratch=scratch, use_rfft=true)
    # A complex field may be its own FFT scratch, as on the CPU.
    z = complex.(vt, vp)
    Z = MtlArray(z)
    zout = MtlArray{ComplexF32}(undef, cfg.lmax + 1, cfg.mmax + 1)
    analysis!(cfg, zout, Z; fft_scratch=Z)
    @test Array(zout) ≈ analysis(cfg, z) rtol=4f-5 atol=3f-6
end

@testset "Plans follow a raised degree" begin
    cfg = create_gauss_config(4, 8; mmax=4)
    rng = MersenneTwister(9)
    for half in (false, true)
        plan = SHTPlan(cfg; use_rfft=half)
        cfg.lmax = 6
        f, g = randn(rng, Float32, cfg.nlat, cfg.nlon), randn(rng, Float32, cfg.nlat, cfg.nlon)
        fd = MtlArray(f)
        out = MtlArray{ComplexF32}(undef, cfg.lmax + 1, cfg.mmax + 1)
        analysis!(plan, out, fd)
        a = analysis(cfg, f)
        check_device(out, a)
        y = similar(fd)
        synthesis!(plan, y, out)
        check_device(y, synthesis(cfg, a))
        so, to = similar(out), similar(out)
        analysis_sphtor!(plan, so, to, fd, MtlArray(g))
        check_components((so, to), analysis_sphtor(cfg, f, g))
        vo, wo = similar(fd), similar(fd)
        synthesis_sphtor!(plan, vo, wo, so, to)
        check_components((vo, wo), synthesis_sphtor(cfg, analysis_sphtor(cfg, f, g)...))
        cfg.lmax = 4
    end
end
