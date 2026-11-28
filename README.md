# SHTnsKit.jl

[![Build Status](https://github.com/SpectralFluid/SHTnsKit.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/SpectralFluid/SHTnsKit.jl/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/SpectralFluid/SHTnsKit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SpectralFluid/SHTnsKit.jl)
[![MPI Examples](https://github.com/SpectralFluid/SHTnsKit.jl/actions/workflows/mpi-examples.yml/badge.svg)](https://github.com/SpectralFluid/SHTnsKit.jl/actions/workflows/mpi-examples.yml)
[![JET](https://img.shields.io/badge/tested%20with-JET.jl-9cf)](https://github.com/aviatesk/JET.jl)
[![Aqua](https://img.shields.io/badge/tested%20with-Aqua.jl-aqua)](https://github.com/JuliaTesting/Aqua.jl)

<!-- Badges -->
 <p align="left">
    <a href="https://spectralfluid.github.io/SHTnsKit.jl/stable/">
        <img alt="stable docs" src="https://img.shields.io/badge/documentation-stable%20-blue">
    </a>
      <a href="https://spectralfluid.github.io/SHTnsKit.jl/dev/">
        <img alt="latest docs" src="https://img.shields.io/badge/documentation-dev%20-orange">
    </a>
</p>

**High-Performance Spherical Harmonic Transforms in Julia**

SHTnsKit.jl provides a pure-Julia implementation of spherical harmonic
transforms on CPUs, NVIDIA CUDA, AMD ROCm, and Apple Metal GPUs, with distributed
execution through MPI/PencilArrays.


## Key Features

### **High-Performance Computing**
- **Pure Julia transforms** with FFTW for CPU longitude transforms
- **Multi-threading**: Optimized with Julia threads and FFTW parallelization
- **MPI Parallel**: Distributed computing with MPI + PencilArrays + PencilFFTs
- **GPU Native**: CUDA, AMDGPU, and Metal device-array transforms through package extensions
- **SIMD Optimized**: Vectorization with LoopVectorization.jl support
- **Extensible**: Modular architecture for CPU/GPU/distributed computing

### **Complete Scientific Functionality**  
- **Transform Types**: Scalar, vector, and complex field transforms
- **Grid Support**: Gauss-Legendre, regular Fejér, pole-inclusive, and Driscoll-Healy grids
- **Conventions**: Orthonormal, four-pi, and Schmidt normalization with configurable Condon-Shortley phase
- **Vector Analysis**: Spheroidal-toroidal decomposition for flow fields
- **Differential Operators**: Laplacian, gradient, divergence, vorticity
- **Spectral Analysis**: Power spectra, correlation functions, filtering

### **Advanced Capabilities**
- **Automatic Differentiation**: Native ForwardDiff.jl and Zygote.jl support  
- **Field Rotations**: Wigner D-matrix rotations and coordinate transforms
- **Matrix Operators**: Efficient spectral differential operators
- **Performance Tuning**: Comprehensive benchmarking and optimization tools


## Installation

### Basic Installation (Serial Computing)

```julia
using Pkg
Pkg.add("SHTnsKit")
```

### Full Installation (Parallel Computing)

For high-performance parallel computing on clusters:

```julia
using Pkg
Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs", "LoopVectorization"])
```

### GPU Extensions

```julia
using Pkg
Pkg.add(["SHTnsKit", "CUDA", "GPUArrays", "GPUArraysCore", "KernelAbstractions"])   # NVIDIA
Pkg.add(["SHTnsKit", "AMDGPU", "GPUArrays", "GPUArraysCore", "KernelAbstractions"]) # AMD

Pkg.add(["SHTnsKit", "Metal"])                                                    # Apple Silicon
```

Generic `analysis(cfg, device_array)` and `synthesis(cfg, coefficients)` calls
preserve the vendor device. Strict `GPU()` dispatch never silently falls back
to CPU execution.

On Apple Silicon, load `SHTnsKit` and `Metal` and use `Float32` fields or
`ComplexF32` coefficients. See the [Metal example and supported transforms](docs/src/gpu.md#apple-metal).

### System Requirements

**MPI Setup** (for parallel computing):
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev

# macOS  
brew install open-mpi

# To use the system MPI from your Julia project
julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences"); using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Citation

If you use SHTnsKit.jl in your research, please cite:
```bibtex
@article{schaeffer2013efficient,
  title={Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations},
  author={Schaeffer, Nathana{\"e}l},
  journal={Geochemistry, Geophysics, Geosystems},
  volume={14},
  number={3},
  pages={751--758},
  year={2013},
  publisher={Wiley Online Library}
}
```

##  License

SHTnsKit.jl is released under the GNU General Public License v3.0 (GPL-3.0), ensuring compatibility with the underlying SHTns library.

## References

- **[SHTns Documentation](https://nschaeff.bitbucket.io/shtns/)**: Original C library
- **[Spherical Harmonics Theory](https://en.wikipedia.org/wiki/Spherical_harmonics)**: Mathematical background  
