using Test

@testset "SHTnsKit GPU Tests" begin
    include("test_cuda_contracts.jl")
end
