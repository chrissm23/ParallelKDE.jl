# Installation

To install `ParallelKDE.jl`, use Julia's package manager:

```
pkg> add ParallelKDE
```

Or, with the `Pkg` API:

```julia
using Pkg
Pkg.add("ParallelKDE")
```

!!! compat "Compatible Julia versions"
    This package supports Julia v1.11+ and is expected to work on Julia 1.x (SemVer: ^1.11).

This will automatically install the package and its dependencies including [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). In order to use CUDA acceleration, please ensure that you have a compatible NVIDIA GPU.

A Python wrapper is also available, allowing users to call estimation routines from Python. For detailed instructions, please refer to the [ParallelKDEpy](https://github.com/chrissm23/ParallelKDEpy) package.
