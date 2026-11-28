function test_transform_storage(vendor, device_array)
    @testset "GPU packed views and in-place Fourier storage" begin
        cfg = create_gauss_config(2, 4; nlon=7, norm=:fourpi)
        field = reshape(Float64.(1:cfg.nspat), cfg.nlat, cfg.nlon)
        nested(x) = view(PermutedDimsArray(device_array(reshape(x, 1, :)), (2, 1)), :, 1)
        packed = analysis_packed(cfg, vec(field))
        for (f, x) in ((analysis_packed, vec(field)), (synthesis_packed, packed))
            out = f(cfg, nested(x))
            @test on_device(out) isa SHTnsKit.GPU
            @test Array(out) ≈ f(cfg, x)
        end

        field .= 1.5e38
        input = device_array(field)
        batch = device_array(cat(field, -field; dims=3))
        out = similar(input, ComplexF32, cfg.lmax + 1, cfg.mmax + 1)
        spectra = similar(out, size(out)..., 2)
        spatial = similar(batch, Float32)
        for use_rfft in (false, true)
            nbins = use_rfft ? cfg.nlon ÷ 2 + 1 : cfg.nlon
            padded = device_array(fill(17.0 + 0im, cfg.nlat, 2nbins, 2))
            scratch = view(padded, :, 1:2:2nbins, :)
            analysis!(cfg, out, input; use_rfft)
            analysis_batch!(cfg, spectra, batch; use_rfft, fft_batch=scratch)
            @test Array(out)[1, 1] ≈ 1.5f38
            @test Array(spectra)[1, 1, :] ≈ [1.5f38, -1.5f38]
            @test all(==(17), Array(padded)[:, 2:2:end, :])

            # Use moderate coefficients in single precision for the inverse FFT.
            spectra .= spectra ./ 1f38
            inverse_padded = device_array(fill(17f0 + 0f0im, cfg.nlat, 2nbins, 2))
            inverse_scratch = view(inverse_padded, :, 1:2:2nbins, :)
            synthesis_batch!(cfg, spatial, spectra; use_rfft, fft_batch=inverse_scratch)
            @test Array(spatial) ≈ Float32.(Array(batch) ./ 1e38) rtol=2f-5
            @test all(==(17), Array(inverse_padded)[:, 2:2:end, :])
        end
    end

    @testset "Transforms follow the allocation's physical GPU" begin
        devices = collect(vendor.devices())
        if length(devices) < 2
            @test_skip length(devices) >= 2
        else
            caller, owner = devices[1:2]
            cfg = create_gauss_config(3, 6; nlon=9)
            host = reshape(sin.(1:cfg.nspat), cfg.nlat, cfg.nlon)
            field = vendor.device!(owner) do
                device_array(host)
            end
            vendor.device!(caller) do
                other = device_array(host)
                placed = to_device(SHTnsKit.GPU(), host, field)
                @test vendor.device(placed) == owner
                coefficients = analysis(cfg, view(field, :, :))
                @test vendor.device(coefficients) == owner
                @test vendor.device(synthesis(cfg, coefficients)) == owner
                qst = analysis_qst(SHTnsKit.GPU(), cfg, host, field, host)
                @test all(a -> vendor.device(a) == owner, qst)
                @test vendor.device() == caller
                @test_throws ArgumentError analysis_qst(cfg, field, other, field)
                @test_throws ArgumentError analysis!(cfg, coefficients, other)
                @test vendor.device() == caller
                vendor.device!(owner) do
                    @test Array(coefficients) ≈ analysis(cfg, host)
                end
            end
        end
    end
end
