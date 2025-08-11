```@meta
CurrentModule = ParallelKDE
```

# [ParallelKDE Interface](@id APIParallelKDE)

This is the main module users interact with. It re-exports the essential tools to:
- Define and configure estimators
- Run KDEs on different devices (CPU, CUDA)
- Access results and manipulate densities

Currently, the main objects for the estimation are:

```@docs
AbstractDensityEstimation
DensityEstimation
```

It is possible to initialize the estimation object with:

```@docs
initialize_estimation
```

Estimation objects are designed to support a grid for grid-based density estimation. However, this is not mandatory. To test if a grid is present and obtain a grid object, you can use:

```@docs
has_grid
get_grid
```

Executing the estimation and obtaining the density is done with:

```@docs
estimate_density!
get_density(density_estimation::DensityEstimation)
```
