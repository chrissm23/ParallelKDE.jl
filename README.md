# ParallelKDE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chrissm23.github.io/ParallelKDE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chrissm23.github.io/ParallelKDE.jl/dev/)
[![Build Status](https://github.com/chrissm23/ParallelKDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chrissm23/ParallelKDE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Implementation of a parallel kernel density estimation algorithm in Julia described ....

⚠️ **Note:** Use of [`ParallelKDE`]("https://github.com/chrissm23/ParallelKDE.jl") via the python wrapper [`ParallelKDEpy`](https://github.com/chrissm23/ParallelKDEpy) does not require additional installation of this package since the wrapper will automatically install it as a dependency. However, if changes to `ParallelKDE` are necessary (e.g., modify default parameters), it is recommended to install `ParallelKDE` directly in Julia as described below and follow the instructions in `ParallelKDEpy` to use the locally installed version.

## Installation
- *Julia installation:*
It is recommended to use `juliaup` to install Julia. See [their repository](https://github.com/JuliaLang/juliaup) for installation instructions.

- *ParallelKDE installation:*
Since ParallelKDE is in a private repository, it needs to be installed from github with valid access rights using the following command:

```julia
using Pkg
Pkg.add(url="https://github.com/chrissm23/ParallelKDE.jl", rev="main") # or use the latest dev version with rev="dev"
```

During the installation, you will be prompted to enter your GitHub username and password (or personal access token) to authenticate the installation.

## Usage
`ParallelKDE` can be used in Julia as follows:
```julia
using ParallelKDE

# Assume `data` is a 2D array of points for which you want to estimate the density
# However, it can also be an AbstractVector of AbstractVectors
data
grid_ranges = fill(range(x_min, x_max, length=n_gridpoints_x), n_dims) # Define the grid ranges for each dimension

"
ParallelKDE currently supports two devices using the `device` keyword argument: `:cpu` and `:cuda`.
"
density_estimation = initialize_estimation(
  data,
  grid=true,
  grid_ranges=grid_ranges,
)

"
Parameters of `estimate_density!` for parallel estimation include:
- `time_step`: Time step of propagation.
- `n_steps`: Number of time steps for propagation. Defaults to 1000. Only one of `time_step` or `n_steps` should be specified.
- `n_bootstraps`: Number of bootstraps to perform. Defaults to 100.
- `eps1`: Threshold of first derivative as stopping criterion for the propagation. Defaults to 1.5.
- `eps2`: Threshold of second derivative as stopping criterion for the propagation. Defaults to 0.1 (CPU) or 1.0 (CUDA).
- `smoothness_duration`: fraction of the total time to require smoothness before entering the next propagation regime. Defaults to 0.005 (0.5%).
- `stable_duration`: fraction of the total time to require stability before entering the stopping propagation. Defaults to 0.01 (1%).
- `method`: Method to use for the specified device. For `device=:cpu`, `method=:serial` and `method=:threaded` are available, for `device=:cuda` only `method=:cuda` is available.
"
estimate_density!(
  density_estimation,
  :parallelEstimation, # So far this is the only implemented estimation method
)

"Returns the estimated density as an array of the same shape as the grid defined by `grid_ranges`."
density_estimated = get_density(density_estimation)
```

## Contributing
For contributions of specific features or bug fixes, please open an issue in the [ParallelKDE repository](https://github.com/chrissm23/ParallelKDE.jl/issues) or create a pull request to the [`dev`](https://github.com/chrissm23/ParallelKDE.jl/tree/dev) branch.
