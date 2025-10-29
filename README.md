# ParallelKDE: Rapid Parallel Kernel Density Estimation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chrissm23.github.io/ParallelKDE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chrissm23.github.io/ParallelKDE.jl/dev/)
[![Build Status](https://github.com/chrissm23/ParallelKDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chrissm23/ParallelKDE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

High performance implementation in Julia of a parallel kernel density estimation algorithm described in [Sustay Martinez *et al.* (2025)]. The algorithm is specially suited for high-dimensional data, with CPU/CUDA support.

**Quick links:**

- 📑 Docs: [stable](https://chrissm23.github.io/ParallelKDE.jl/stable/) | [dev](https://chrissm23.github.io/ParallelKDE.jl/dev/)
- 📇 [Citing](#citing)
- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) Python wrapper: [ParallelKDEpy](https://github.com/chrissm23/ParallelKDEpy)

### Why this project?
- ParallelKDE provides a framework to perform kernel density estimation (KDE) using parallel algorithms.
- It is friendly to high-dimensional data and big datasets.
- ParallelKDE is easily extensible, allowing users to implement their own parallel algorithms.

## Installation

```julia
using Pkg
Pkg.add("ParallelKDE")
```

or, similarly,

```
pkg> add ParallelKDE
```

## Quick Start

```julia
using ParallelKDE

# Assume 'data' is a 2D array of points for which we want to estimate the density
data = randn(1, 10000)

density_estimation = initialize_estimation(
  data,
  grid=true,
  device=:cpu, # or :cuda for GPU support
)
estimate_density!(density_estimation, :parallelEstimator)

density_estimated = get_density(density_estimation)
```

See the [documentation] for more details on how to use the package.

## Features

Currently, there are two estimators available:

- `:parallelEstimator`: As described in [Sustay Martinez *et al.* (2025)], this estimator is designed for high-dimensional data and can be run on both CPU and GPU.
- `:rotEstimator`: Implements the rules of thumb (Silverman and Scott) for bandwidth selection. It makes use of some of the routines from `:parallelEstimator` to evaluate the density on a grid.

## Python Wrapper

For integration with Python-based workflows, a Python wrapper is available via the [ParallelKDEpy](https://github.com/chrissm23/ParallelKDEpy) package. This wrapper allows you to use the ParallelKDE functionality directly in Python.

## Citing

Please cite the following papers when using ParallelKDE in your work:
