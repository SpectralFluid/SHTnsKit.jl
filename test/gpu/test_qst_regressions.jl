# Run the same public calls against the host-backed wrappers and each vendor's
# device arrays. Only copy results back after checking their storage and shape.
function run_gpu_qst_regressions(api, place, array_type)
    # The analytic monopole and gradient values below use DFT normalization.
    withenv("SHTNSKIT_PHI_SCALE" => "dft") do
        test_gpu_qst_batches(api, place, array_type)
        test_gpu_local_shapes_and_poles(api, place, array_type)
        test_gpu_mixed_local_spectra(api, place, array_type)
        test_local_coordinate_precision(api, place)
    end
end

function test_gpu_mixed_local_spectra(api, place, array_type)
    @testset "Promotion of packed local spectra" begin
        cfg = create_gauss_config(6, 8; norm=:schmidt, cs_phase=false, real_norm=true,
                                  robert_form=true)
        a = complex.(sin.(1:cfg.nlm), cos.(1:cfg.nlm))
        for wide in 1:3
            bare = ntuple(i -> (i == wide ? ComplexF64 : ComplexF32).((i + 0.2im) .* a), 3)
            q, s, t = map(place, bare)
            x, phi = 0.37, 0.6
            for (actual, expected) in (
                (api.SHqst_to_point(SHTnsKit.GPU(), cfg, q, s, t, x, phi),
                 SHTnsKit.SHqst_to_point(cfg, bare..., x, phi)),
                (api.SHqst_to_lat(cfg, q, s, t, x; nphi=4, ltr=5, mtr=3),
                 SHTnsKit.SHqst_to_lat(cfg, bare..., x; nphi=4, ltr=5, mtr=3)),
                (api.SH_to_grad_point(cfg, q, s, x, phi),
                 SHTnsKit.SH_to_grad_point(cfg, bare[1], bare[2], x, phi)),
            )
                for (out, ref) in zip(actual, expected)
                    @test out isa array_type && size(out) == size(ref)
                    @test eltype(out) === eltype(ref)
                    host = ref isa Number ? only(Array(out)) : Array(out)
                    tol = eltype(ref) === Float32 ? 3e-5 : 3e-13
                    @test host ≈ ref atol=tol rtol=tol
                end
            end
        end
        q = place(a)
        invalid = place(Complex{Float16}.(a))
        @test_throws ArgumentError api.SHqst_to_point(cfg, q, invalid, q, 0.3, 0.0)
    end
end

function test_local_coordinate_precision(api, place)
    @testset "Wide coordinates with ComplexF32 local spectra" begin
        l = 64
        cfg = create_gauss_config(l, l + 2; mmax=1, nlon=4)
        dense = zeros(ComplexF32, l + 1, 2)
        dense[end, 2] = 1
        q = SHTnsKit.pack_lm(cfg, dense)
        s = zero(q)
        s[LM_index(l, 1, 1, 0) + 1] = 1
        z = zeros(ComplexF32, nlm_cplx_calc(l, 1, 1))
        z[LM_cplx_index(l, 1, l, 1) + 1] = 1
        inputs = (dense, q, s, z)
        # The first two latitudes round to the same Float32 pole. A large
        # longitude also exposes rounding the angle before evaluating its phase.
        for (backend, device, (a, qp, sp, zp)) in (
            (SHTnsKit, SHTnsKit.CPU(), inputs), (api, SHTnsKit.GPU(), map(place, inputs)),
        ), (x, phi) in ((1 - 1e-8, 0.0), (1 - 2e-8, 2.0^25 + 0.25), (-1 + 1e-8, 0.375), (1.0, 0.0))
            y = setprecision(256) do
                # Y_l^1(x, 0) from the ordinary Legendre derivative recurrence.
                X = BigFloat(x)
                prev, d = zero(X), one(X)
                for n in 2:l
                    prev, d = d, ((2n - 1)*X*d - n*prev) / (n - 1)
                end
                Float64(-sqrt((2l + 1) / (4big(pi)*l*(l + 1))) * sqrt(1 - X^2) * d)
            end
            wave = y * cis(phi)
            dtheta = -sqrt(3 / (4pi)) * sqrt((1-x)*(1+x))
            point = backend.synthesis_point(device, cfg, a, x, phi)
            cpoint = backend.synthesis_point_cplx(device, cfg, zp, x, phi)
            qst = backend.SHqst_to_point(device, cfg, qp, sp, sp, x, phi)
            grad = backend.SH_to_grad_point(device, cfg, qp, sp, x, phi)
            lat = backend.SH_to_lat(device, cfg, qp, x; nphi=1)
            clat = backend.SH_to_lat_cplx(device, cfg, zp, x; nphi=1)
            qlat = backend.SHqst_to_lat(device, cfg, qp, sp, sp, x; nphi=1)
            actual = (point, cpoint, qst..., grad..., lat, clat, qlat...)
            expected = (2real(wave), wave, 2real(wave), dtheta, dtheta, 2real(wave),
                        dtheta, 0.0, 2y, complex(y), 2y, dtheta, dtheta)
            for (result, ref) in zip(actual, expected)
                value = result isa Number ? result : only(Array(result))
                @test typeof(value) === (ref isa Complex ? ComplexF32 : Float32)
                @test value ≈ ref rtol=8eps(Float32)
            end
        end

        # The harmonic or the sine of the longitude can underflow in Float32
        # even when its product with a stored coefficient is representable.
        m = 14
        cfg = create_gauss_config(m, m + 2)
        a = zeros(ComplexF32, m + 1, m + 1)
        a[end, end] = 1f30
        x = 1 - 1e-8
        sinth = sqrt((1 - x) * (1 + x))
        y = prod(-sqrt((2k + 1) / (2k)) * sinth for k in 1:m) / sqrt(4pi)
        expected = 2 * Float64(real(a[end, end])) * y
        @test iszero(Float32(y)) && !iszero(Float32(expected))
        @test only(Array(api.synthesis_point(cfg, place(a), x, 0.0))) ≈ expected rtol=4eps(Float32)
        @test only(SHTnsKit.SH_to_lat(cfg, SHTnsKit.pack_lm(cfg, a), x; nphi=1)) ≈ expected rtol=4eps(Float32)

        fill!(a, 0)
        a[2, 2] = 1f30im
        phi = 1e-50
        expected = 2sqrt(3 / (8pi)) * Float64(imag(a[2, 2])) * sin(phi)
        @test only(Array(api.synthesis_point(cfg, place(a), 0.0, phi))) ≈ expected rtol=4eps(Float32)
        @test SHTnsKit.synthesis_point(cfg, a, 0.0, phi) ≈ expected rtol=4eps(Float32)
    end
end

function test_gpu_qst_batches(api, place, array_type)
    @testset "QST batch precision and shapes" begin
        cfg = create_gauss_config(1, 4; nlon=4)
        radial = zeros(Float32, 4, 4, 2)
        radial[:, :, 1] .= reshape(Float32[2^26, -2^26, 1, 0], 1, 4)
        radial[:, :, 2] .= 1f38
        tangent = reshape(Float32.(1:32), 4, 4, 2) ./ 7f0
        fields = (radial, tangent, -2tangent)
        q = zeros(ComplexF32, 2, 2, 2)
        q[2, 2, 1] = 0.25f0 - 0.5f0im
        s, t = 2q, -q
        q[1, 1, 2] = floatmax(Float32)

        for wide in 1:3
            inputs = ntuple(i -> (i == wide ? Float64 : Float32).(fields[i]), 3)
            # Exercise explicit and storage-inferred dispatch.
            device = wide == 1 ? () : (SHTnsKit.GPU(),)
            actual = api.analysis_qst_batch(device..., cfg, map(place, inputs)...)
            expected = SHTnsKit.analysis_qst_batch(cfg, inputs...)
            @test all(a -> a isa array_type && eltype(a) === ComplexF64, actual)
            qlm, slm, tlm = map(Array, actual)
            # Both examples fail if the radial FFT runs in Float32 before promotion.
            @test qlm[1, 1, 1] ≈ sqrt(4pi) / 4 rtol=1e-12
            @test qlm[1, 1, 2] ≈ sqrt(4pi) * Float64(1f38) rtol=1e-12
            @test slm ≈ expected[2] rtol=1e-12
            @test tlm ≈ expected[3] rtol=1e-12

            spectra = ntuple(i -> (i == wide ? ComplexF64 : ComplexF32).((q, s, t)[i]), 3)
            resident = map(place, spectra)
            for real_output in (true, false)
                output = real_output ? api.synthesis_qst_batch(device..., cfg, resident...) :
                                       api.synthesis_qst_batch_cplx(device..., cfg, resident...)
                reference = SHTnsKit.synthesis_qst_batch(cfg, spectra...; real_output)
                @test all(a -> a isa array_type && eltype(a) === (real_output ? Float64 : ComplexF64), output)
                for (values, ref) in zip(map(Array, output), reference), k in 1:2
                    @test values[:, :, k] ≈ ref[:, :, k] rtol=1e-12 atol=1e-13
                end
            end
        end

        for (call, inputs) in ((api.analysis_qst_batch, fields), (api.synthesis_qst_batch, (q, s, t)))
            a, b, c = map(place, inputs)
            @test_throws DimensionMismatch call(cfg, view(a, :, :, 1:1), b, c)
            @test_throws DimensionMismatch call(SHTnsKit.GPU(), cfg, a, b, view(c, :, 1:1, :))
        end
    end
end

function test_gpu_local_shapes_and_poles(api, place, array_type)
    @testset "One-longitude latitudes, points, and zonal gradients" begin
        cfg = create_gauss_config(10, 12; mmax=0, nlon=4)
        for T in (Float32, Float64)
            dense = zeros(Complex{T}, 11, 1)
            dense[end, 1] = 1
            packed = SHTnsKit.pack_lm(cfg, dense)
            z = complex.(real.(packed), real.(packed))
            a, p, c, dr = map(place, (dense, packed, z, zero(packed)))
            x, phi = T(0.3), zero(T)
            points = (api.synthesis_point(cfg, a, x, phi),
                      api.synthesis_point_cplx(cfg, c, x, phi),
                      api.SHqst_to_point(cfg, p, p, p, x, phi)...)
            latitudes = (api.SH_to_lat(cfg, p, x; nphi=1),
                         api.SH_to_lat_cplx(cfg, c, x; nphi=1),
                         api.SHqst_to_lat(cfg, p, p, p, x; nphi=1)...)
            for (point, latitude) in zip(points, latitudes)
                @test point isa array_type && size(point) == ()
                @test latitude isa array_type && size(latitude) == (1,)
                @test only(Array(latitude)) ≈ only(Array(point)) rtol=8eps(T)
            end
            for x in (prevfloat(one(T)), -prevfloat(one(T)), one(T), -one(T))
                grad = api.SH_to_grad_point(SHTnsKit.GPU(), cfg, dr, p, x, phi)
                @test all(v -> v isa array_type && size(v) == (), grad)
                # P′₁₀(±1) = ±55. At the nearest interior float its relative
                # correction is below 27eps(T); at the poles the gradient is zero.
                sinth = sqrt((1.0 - Float64(x)) * (1.0 + Float64(x)))
                limit = -sign(x) * 55sqrt(21 / (4pi)) * sinth
                @test only(Array(grad[2])) ≈ limit rtol=64eps(T)
            end
        end

        cfg = create_gauss_config(2, 4; mmax=0, nlon=4, robert_form=true)
        s = ComplexF64[0, 0, 1]
        dr, mode = map(place, (zero(s), s))
        for x in (1 - 1e-8, -1 + 1e-8)
            # In Robert form the extra sinθ gives -3√(5/4π) x (1-x²) for Y₂⁰.
            expected = -3sqrt(5 / (4pi)) * x * (1 - x) * (1 + x)
            grad = api.SH_to_grad_point(cfg, dr, mode, x, 0.0)
            @test only(Array(grad[2])) ≈ expected rtol=8eps()
            @test SHTnsKit.SH_to_grad_point(cfg, zero(s), s, x, 0.0)[2] ≈ expected rtol=8eps()
        end
    end
end
