```@meta
CurrentModule = ParallelKDE
```

# [Grids Interface](@id APIGrids)

Grids define the space over which the KDE is computed. The package supports:

- Regular, Cartesian grids
- User-defined bounds and resolution
- Grid selection based on data distribution

There are currently two concrete grid types, one for CPU and one for CUDA devices. However, the interface is otherwise the same, so that it is possible to use the same code for both devices. It is also possible to create custom grids that conform to the interface.

```@docs
AbstractGrid
Grid
CuGrid
```

There is a set of features that can be extracted from grid objects. This is done by the following methods:

```@docs
size(::AbstractGrid)
ndims(::AbstractGrid)
get_coordinates
get_device(::AbstractGrid)
spacings
bounds
low_bounds
high_bounds
initial_bandwidth
```

It is possible to create a grid with the desired ranges with

```@docs
initialize_grid
```

as well as to create a grid appropriate for the data using

```@docs
find_grid
```

For convenience, the package includes a method to create a grid for Fourier space from a grid in direct space:

```@docs
fftgrid
```
