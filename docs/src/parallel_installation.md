# Parallel Computing Installation Guide

This comprehensive guide covers installing and configuring SHTnsKit.jl for high-performance parallel computing with MPI, PencilArrays, PencilFFTs, and SIMD optimizations.

## Overview

SHTnsKit.jl supports multiple levels of performance optimization:

1. **Serial**: Basic Julia threading and FFTW optimization
2. **SIMD**: Enhanced vectorization with LoopVectorization.jl  
3. **MPI Parallel**: Distributed computing with domain decomposition
4. **Full Stack**: Combined MPI + SIMD + threading for maximum performance

## System Requirements

### Package Version Compatibility

SHTnsKit.jl's distributed extension supports the versions declared in
`Project.toml`:

| Package | Supported Version | Notes |
|---------|----------------|-------|
| **MPI.jl** | v0.20 | Uses `Allgatherv!` with `VBuffer` API |
| **PencilArrays.jl** | v0.19 | Uses `range_local`, `size_local`, `get_comm` API |
| **PencilFFTs.jl** | v0.15 | Compatible distributed FFT support |
| **Julia** | 1.10 or later 1.x | Matches package compatibility bounds |

**Important**: PencilArrays releases outside the declared v0.19 compatibility
range are not supported.

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows with WSL
- **Julia**: Version 1.10 or later 1.x
- **Memory**: 8GB RAM (32GB+ for large parallel problems)
- **Network**: Fast interconnect recommended for multi-node MPI

### Recommended Hardware
- **CPU**: Modern multi-core processor with AVX2/AVX512 support
- **Network**: InfiniBand or 10+ Gbps Ethernet for multi-node scaling
- **Storage**: NFS or parallel filesystem for multi-node jobs

## Installation Steps

### Step 1: Basic SHTnsKit Installation

```julia
using Pkg
Pkg.add("SHTnsKit")
```

### Step 2: MPI Setup

**Linux (Ubuntu/Debian):**
```bash
# Install MPI library
sudo apt-get update
sudo apt-get install libopenmpi-dev openmpi-bin

# Optional: Install development tools
sudo apt-get install build-essential gfortran
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install openmpi-devel
# or for newer systems:
sudo dnf install openmpi-devel

# Load MPI module
module load mpi/openmpi-x86_64
```

**macOS:**
```bash
# Install via Homebrew
brew install open-mpi

# Optional: Install via MacPorts
# sudo port install openmpi
```

### Step 3: Julia MPI Configuration

```julia
using Pkg

# Install MPI.jl
Pkg.add("MPI")

# Build MPI with system library
Pkg.build("MPI")

# Verify installation
using MPI
MPI.Init()
println("MPI initialized successfully")
MPI.Finalize()
```

### Step 4: Parallel Computing Packages

```julia
using Pkg

# Install complete parallel stack
Pkg.add([
    "MPI",           # Message Passing Interface
    "PencilArrays",  # Domain decomposition
    "PencilFFTs",    # Distributed FFTs
    "LoopVectorization"  # SIMD enhancements
])

# Optional performance packages
Pkg.add([
    "BenchmarkTools",
    "Profile",
    "ProfileView"
])
```

### Step 5: Verification

**Test MPI functionality:**
```julia
# Save as test_mpi.jl
using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = Comm_rank(comm)
size = Comm_size(comm)

println("Hello from process $rank of $size")

MPI.Finalize()
```

```bash
# Run with multiple processes
mpiexec -n 4 julia test_mpi.jl
```

**Test SHTnsKit parallel functionality:**

Save as `test_parallel.jl`:
```julia
using MPI
MPI.Init()

using SHTnsKit, PencilArrays, PencilFFTs

rank = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)

if rank == 0
    println("Testing with $nprocs MPI processes")
end

# Create configuration
lmax = 16
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Create distributed array
pen = Pencil((nlat, nlon), (1,), MPI.COMM_WORLD)
fθφ = PencilArray(pen, zeros(Float64, PencilArrays.size_local(pen)...))

# Fill with test data (Y_2^0)
ranges = PencilArrays.range_local(pen)
for (i_local, i_global) in enumerate(ranges[1])
    x = cfg.x[i_global]
    for j in 1:length(ranges[2])
        fθφ[i_local, j] = (3*x^2 - 1)/2
    end
end

# Test distributed transforms
Alm = SHTnsKit.dist_analysis(cfg, fθφ)
fθφ_out = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true)

# Verify accuracy
max_err = maximum(abs.(parent(fθφ_out) .- parent(fθφ)))
global_max_err = MPI.Allreduce(max_err, MPI.MAX, MPI.COMM_WORLD)

if rank == 0
    println("Roundtrip error: $global_max_err")
    println(global_max_err < 1e-10 ? "SUCCESS!" : "FAILED")
end

destroy_config(cfg)
MPI.Finalize()
```

```bash
# Test parallel mode
mpiexec -n 2 julia --project test_parallel.jl
```

**Run the built-in parallel testset (includes PencilArrays/PencilFFTs):**
```bash
SHTNSKIT_RUN_MPI_TESTS=1 JULIA_NUM_THREADS=1 \
    mpiexec -n 2 julia --project -e 'using Pkg; Pkg.test()'
```
This exercises distributed analysis/synthesis, vector/QST transforms, rotations, and diagnostics across ranks using PencilArrays and PencilFFTs.

## Advanced Configuration

### Environment Variables

**MPI tuning:**
```bash
# Reduce MPI warnings
export OMPI_MCA_mpi_warn_on_fork=0

# Network interface selection
export OMPI_MCA_btl_tcp_if_include=eth0

# Memory pinning
export OMPI_MCA_mpi_leave_pinned=1

# Collective algorithm selection
export OMPI_MCA_coll_hcoll_enable=1
```

**Julia optimization:**
```bash
# Threading
export JULIA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export FFTW_NUM_THREADS=4

# Memory
export JULIA_GC_ALLOC_POOL_GROW_THRESHOLD=0.1
```

### Performance Tuning

**Process binding (recommended):**
```bash
# Bind to cores
mpiexec --bind-to core -n 8 julia script.jl

# NUMA-aware binding
mpiexec --map-by socket --bind-to core -n 16 julia script.jl
```

**Large problem optimization:**
```bash
# Increase memory limits
ulimit -s unlimited
ulimit -v unlimited

# Run with large heap
mpiexec -n 8 julia --heap-size-hint=32G script.jl
```

## Container Deployment

### Docker

**Basic parallel container:**
```dockerfile
FROM julia:1.11

# Install MPI
RUN apt-get update && \
    apt-get install -y libopenmpi-dev openmpi-bin && \
    rm -rf /var/lib/apt/lists/*

# Install Julia packages
RUN julia -e 'using Pkg; \
              Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs", "LoopVectorization"]); \
              using MPI; \
              MPI.install_mpiexecjl()'

# Precompile
RUN julia -e 'using SHTnsKit, MPI, PencilArrays, PencilFFTs'

WORKDIR /app
COPY . .

# Run with: docker run --rm -it image mpiexecjl -n 4 julia script.jl
```

### Singularity/Apptainer

**HPC-ready container:**
```singularity
Bootstrap: docker
From: julia:1.11

%post
    apt-get update
    apt-get install -y libopenmpi-dev openmpi-bin
    
    julia -e 'using Pkg; 
              Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs", "LoopVectorization"]); 
              using MPI; MPI.install_mpiexecjl()'
    
    julia -e 'using SHTnsKit, MPI, PencilArrays, PencilFFTs'

%runscript
    exec julia "$@"
```

```bash
# Build and run
singularity build shtns.sif shtns.def
mpirun -n 8 singularity exec shtns.sif julia script.jl
```

## HPC Cluster Setup

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=shtns_parallel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00
#SBATCH --partition=compute

# Load modules
module load julia/1.11
module load openmpi/4.1.0

# Set environment
export JULIA_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1

# Run parallel job
mpirun julia --project=. parallel_example.jl --benchmark
```

### PBS/Torque Script

```bash
#!/bin/bash
#PBS -N shtns_job
#PBS -l nodes=4:ppn=8
#PBS -l walltime=02:00:00
#PBS -q normal

cd $PBS_O_WORKDIR

# Load modules
module load julia/1.11
module load openmpi/3.1.4

# Run job
mpirun -np 32 julia --project=. examples/parallel_example.jl
```

## Troubleshooting

### Common Issues

**1. MPI library mismatch:**
```
ERROR: MPI library not found
```

**Solution:**
```julia
# Force MPI.jl to use system MPI
ENV["JULIA_MPI_BINARY"] = "system"
using Pkg; Pkg.build("MPI")
```

**2. PencilArrays compilation errors:**
```
ERROR: LoadError: FFTW not found
```

**Solution:**
```julia
# Install FFTW explicitly
using Pkg
Pkg.add("FFTW")
Pkg.build("FFTW")
Pkg.build("PencilFFTs")
```

**3. Process binding warnings:**
```
WARNING: A process refused to die!
```

**Solution:**
```bash
# Use proper MPI cleanup
export OMPI_MCA_orte_tmpdir_base=/tmp
mpiexec --mca orte_base_help_aggregate 0 -n 4 julia script.jl
```

**4. PencilArrays API errors (version mismatch):**
```
ERROR: MethodError: no method matching communicator(::Pencil{...})
ERROR: MethodError: no method matching globalindices(::PencilArray{...})
```

**Cause:** The environment resolved a PencilArrays version outside the package's
declared v0.19 compatibility range.

**Solution:**
```julia
# Resolve the versions declared by SHTnsKit
using Pkg
Pkg.resolve()

# Verify version
Pkg.status("PencilArrays")  # Should show v0.19.x
```

The supported PencilArrays v0.19 API uses:
- `get_comm(pen)` instead of `communicator(pen)`
- `range_local(pen)` instead of `globalindices(arr, dim)`
- `size_local(pen)` instead of other size functions

**5. Precompilation cache conflicts with MPI:**
```
ERROR: Permission denied @ mkdir_pid_file
ERROR: InexactError: ... (random memory errors)
```

**Cause:** Multiple MPI processes trying to write to the same precompilation cache simultaneously.

**Solution:**
```bash
# Use a fresh depot for MPI runs
JULIA_DEPOT_PATH=/tmp/fresh_depot:$HOME/.julia mpiexec -n 4 julia script.jl

# Or precompile in serial first
julia --project -e 'using SHTnsKit, MPI, PencilArrays, PencilFFTs'
mpiexec -n 4 julia --project script.jl
```

### Performance Issues

**Slow initialization:**
- Precompile packages: `julia -e 'using SHTnsKit, MPI, PencilArrays, PencilFFTs'`
- Use system image: `julia --sysimage=shtns_parallel.so script.jl`

**Poor scaling:**
- Check network bandwidth: `iperf3` between nodes
- Verify process binding: `numactl --show`
- Monitor MPI communication: `mpiP` profiling

**Memory errors:**
- Increase system limits: `ulimit -v unlimited`
- Use memory-efficient transforms: `memory_efficient_parallel_transform!()`
- Process data in chunks for very large problems

## Validation and Testing

### Comprehensive Test Script

Save as `test_complete_setup.jl`:
```julia
using Test
using SHTnsKit
using LinearAlgebra

@testset "Complete Setup Verification" begin
    # Test basic functionality
    lmax = 16
    cfg = create_gauss_config(lmax, lmax+2; nlon=2*lmax+1)
    @test cfg.nlm > 0

    # Test parallel packages availability
    @testset "Package Loading" begin
        @test_nowarn using MPI
        @test_nowarn using PencilArrays
        @test_nowarn using PencilFFTs
    end

    # Test basic transforms
    @testset "Transform Accuracy" begin
        spatial = zeros(cfg.nlat, cfg.nlon)
        for i in 1:cfg.nlat
            x = cfg.x[i]
            spatial[i, :] .= (3*x^2 - 1)/2  # Y_2^0
        end

        Alm = analysis(cfg, spatial)
        recovered = synthesis(cfg, Alm)
        @test norm(spatial - recovered) < 1e-12
    end

    destroy_config(cfg)
end

println("All tests passed!")
```

```bash
# Run validation
julia --project test_complete_setup.jl
```

### Performance Benchmarking

```julia
# benchmark_setup.jl
using SHTnsKit, BenchmarkTools

function run_benchmarks()
    println("SHTnsKit.jl Performance Benchmark")
    println("=" ^ 50)

    # Test different problem sizes
    for lmax in [16, 32, 64]
        nlat = lmax + 2
        nlon = 2*lmax + 1
        cfg = create_gauss_config(lmax, nlat; nlon=nlon)

        # Create test data
        spatial = zeros(cfg.nlat, cfg.nlon)
        for i in 1:cfg.nlat
            x = cfg.x[i]
            spatial[i, :] .= (3*x^2 - 1)/2
        end

        println("\nlmax = $lmax ($(cfg.nlm) coefficients, $(nlat)×$(nlon) grid)")

        # Serial transform benchmarks
        t_analysis = @belapsed analysis($cfg, $spatial)
        Alm = analysis(cfg, spatial)
        t_synthesis = @belapsed synthesis($cfg, $Alm)

        println("  Analysis: $(t_analysis*1000) ms")
        println("  Synthesis: $(t_synthesis*1000) ms")

        destroy_config(cfg)
    end
end

run_benchmarks()
```

Your parallel SHTnsKit.jl installation is now complete and optimized for high-performance computing!
