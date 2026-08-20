using Test
using SHTnsKit

@testset "Cleanup contract" begin
    @test CPU() isa ComputeDevice
    @test GPU() isa ComputeDevice
    @test all(isdefined(SHTnsKit, name) for name in
              (:get_device, :set_device!, :to_device, :on_device))

    removed = (
        :SHTBackend,
        :SHTDevice,
        :CPU_DEVICE,
        :CUDA_DEVICE,
        :set_backend!,
        :current_backend,
        :use_gpu,
        :with_backend,
        :reset_backend!,
        :select_compute_device,
        :device_transfer_arrays,
        :dispatch_to_backend,
        Symbol("@dispatch_backend"),
        :device_info,
        :ensure_backend_initialized,
        :create_gauss_config_gpu,
        :set_config_device!,
        :get_config_device,
        :is_gpu_config,
        :MultiGPUConfig,
        :create_multi_gpu_config,
        :multi_gpu_analysis,
        :multi_gpu_synthesis,
        :multi_gpu_analysis_streaming,
        :multi_gpu_synthesis_streaming,
        :estimate_streaming_chunks,
        :shtns_init,
        :shtns_create,
        :shtns_set_grid,
        :shtns_malloc,
        :SHT_GAUSS,
        :SHT_ALLOW_GPU,
    )
    @test all(name -> !isdefined(SHTnsKit, name), removed)

    cfg = create_gauss_config(3, 4; nlon=7)
    @test !hasproperty(cfg, :compute_device)
    @test !hasproperty(cfg, :device_preference)

    @testset "CUDA plan wrapper accepts concrete plan variants" begin
        package_root = dirname(dirname(pathof(SHTnsKit)))
        extension_source = read(joinpath(package_root, "ext", "SHTnsKitGPUExt.jl"), String)
        @test occursin(r"struct CuFFTPlan\{", extension_source)
        @test !occursin("inverse_plan::CUFFT.CuFFTPlan", extension_source)
        @test count("SHTnsKit._externalize_coefficients!", extension_source) >= 3
        @test count("SHTnsKit._internal_coefficients", extension_source) >= 3
    end

    @testset "Breaking release uses a major version" begin
        package_root = dirname(dirname(pathof(SHTnsKit)))
        project = read(joinpath(package_root, "Project.toml"), String)
        changelog = read(joinpath(package_root, "CHANGELOG.md"), String)
        @test occursin(r"(?m)^version = \"2\.0\.0\"$", project)
        @test occursin("## Unreleased (v2.0.0)", changelog)
    end
end
