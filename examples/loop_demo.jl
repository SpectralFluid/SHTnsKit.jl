#=
Demo of the @sht_loop macro for unified CPU/GPU execution.

This example shows how the @sht_loop macro provides a BioFlow.jl-style
unified loop abstraction that works seamlessly on both CPU and GPU arrays.
=#

using SHTnsKit

# Example 1: Simple element-wise operation on CPU arrays
println("Example 1: CPU array operation")
A = rand(10, 10)
B = zeros(10, 10)
scale = 2.0

@sht_loop B[I] = A[I] * scale over I ∈ CartesianIndices(A)
println("B[1,1] = $(B[1,1]), expected: $(A[1,1] * scale)")

# Example 2: Using spectral_range for coefficient iteration
println("\nExample 2: Spectral coefficient iteration")
lmax, mmax = 4, 4
coeffs = zeros(ComplexF64, lmax+1, mmax+1)

@sht_loop coeffs[idx] = complex(idx[1], idx[2]) over idx ∈ spectral_range(lmax, mmax)
println("coeffs[3,2] = $(coeffs[3,2]), expected: 3.0 + 2.0im")

# Example 3: Using spatial_range for grid operations
println("\nExample 3: Spatial grid iteration")
nlat, nlon = 32, 64
field = zeros(nlat, nlon)

@sht_loop field[I] = sin(I[1]/nlat * π) * cos(I[2]/nlon * 2π) over I ∈ spatial_range(nlat, nlon)
println("field[16, 32] = $(field[16, 32])")

# Example 4: Using δ for stencil operations
println("\nExample 4: Stencil operation with δ")
data = rand(10, 10)
laplacian = zeros(size(data))  # interior indices run to 9; an 8×8 array overflowed

# Compute discrete Laplacian on interior points
@sht_loop laplacian[I] = (
    data[I + δ(1, I)] + data[I - δ(1, I)] +
    data[I + δ(2, I)] + data[I - δ(2, I)] - 4 * data[I]
) over I ∈ inside(data)

println("Computed discrete Laplacian on interior points")
println("laplacian[4,4] = $(laplacian[4,4])")

# Example 5: Backend switching
println("\nExample 5: Backend control")
println("Current backend: $(loop_backend())")
set_loop_backend("SIMD")
println("After setting to SIMD: $(loop_backend())")
set_loop_backend("auto")
println("Reset to auto: $(loop_backend())")

println("\nAll examples completed successfully!")

#=
GPU Usage (when CUDA is available):

using CUDA, SHTnsKit

# Transfer arrays to GPU
A_gpu = CuArray(A)
B_gpu = similar(A_gpu)

# Same macro works on GPU arrays!
@sht_loop B_gpu[I] = A_gpu[I] * scale over I ∈ CartesianIndices(A_gpu)

# Backend is automatically detected based on array type
# For GPU arrays, KernelAbstractions kernels are launched
# For CPU arrays, SIMD loops are used
=#

#=
PencilArray Usage (for MPI-distributed computing):

using MPI, PencilArrays, SHTnsKit

MPI.Init()
comm = MPI.COMM_WORLD

# Create distributed array
pencil = Pencil((nlat, nlon), comm)
dist_field = PencilArray{Float64}(undef, pencil)

# Use local_range to iterate over local portion
# @sht_loop automatically detects PencilArray and uses SIMD on local data
local_data = parent(dist_field)
@sht_loop local_data[I] = sin(I[1]) over I ∈ local_range(dist_field)

# Or equivalently, iterate over CartesianIndices of local data directly
@sht_loop local_data[I] = cos(I[2]) over I ∈ CartesianIndices(local_data)

# For operations needing MPI communication, use the dist_* API:
# coeffs = dist_analysis(cfg, dist_field)

# Key points:
# - @sht_loop on PencilArrays runs SIMD on local data
# - MPI communication (Allreduce, etc.) must be done separately
# - Use local_range(arr) for convenient iteration over local portion
# - Use local_size(arr) to get dimensions of local data
=#
