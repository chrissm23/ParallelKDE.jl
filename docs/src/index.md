```@meta
CurrentModule = ParallelKDE
```

# ParallelKDE

[ParallelKDE](https://github.com/chrissm23/ParallelKDE.jl) is a package for flexible and efficient kernel density estimation (KDE), with a strong focus on parallel implementations. Its core estimator, the [Parallel Estimator](@ref "ParallelEstimator") described here, supports CPU and GPU acceleration (threaded and CUDA), and its designed to scale with modern hardware. While the package is centered around grid-based KDEs, it also provides extensible infrastructure to support and implement other types of estimators.

The user interface is built around a modular design that separates concerns between grids, devices, density objects and estimation routines. This allows users to easily switch estimators, control execution targets (e.g., CPU or GPU), and prototype new estimation strategies without rewriting boilerplate code.

Typical usage involves:
- Instantiating an estimation in a specific device (e.g., CPU, CUDA),
- This also involves defining a grid for the estimation or using a default grid,
- Estimating the density with a chosen estimator,
- Accessing the resulting density.

For example, to estimate a density on a CPU with a default grid using the [Parallel Estimator](@ref "ParallelEstimator"), you can use:

```@example 1
using ParallelKDE

data = randn(1, 10000) # 1-dimensional sample of 10000 points

density_estimation = initialize_estimation(
  data,
  grid=true,
  device=:cpu,
)
estimate_density!(
  density_estimation,
  :parallelEstimator,
)

density_estimated = get_density(density_estimation)
nothing # hide
```

We can evaluate the standard normal distribution for comparison:

```@example 1
using Distributions

grid_coordinates = get_coordinates(get_grid(density_estimation))[1, :]
density_true = pdf.(Normal(), grid_coordinates)
nothing # hide
```

which would yield a plot like this:

```@example 1
using Plots # hide
p = plot(grid_coordinates, density_true, label="True Density", color=:cornflowerblue, lw=2)
plot!(p, grid_coordinates, density_estimated, label="Estimated Density", color=:firebrick, lw=2)
plot!(p, xlabel="Random Variable", ylabel="Density")
savefig("basic_usage.svg"); nothing # hide
```

![](basic_usage.svg)

As it is exemplified above, it is possible to initialize an estimation using

```@docs; canonical=false
initialize_estimation
```

Then, the density can be estimated with a chosen estimator and its settings using

```@docs; canonical=false
estimate_density!
```

Finally, the estimated density can be accessed using

```@docs; canonical=false
get_density
```

More details regarding the currently implemented estimators as well as further information about the package can be found throughout the documentation.

```@contents
```
