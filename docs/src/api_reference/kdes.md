```@meta
CurrentModule = ParallelKDE
```

# [KDE Interface](@id APIKDEs)

KDE objects store the final result of the estimation. This usually consists of an array of densities mapped to the selected grid. They also store the samples used for the estimation.

There are currently two concrete KDE types, one for CPU and one for CUDA devices. However, the interface is otherwise the same, so that it is possible to use the same code for both devices. It is also possible to create custom KDE objects that conform to the interface.

```@docs
AbstractKDE
KDE
CuKDE
```

The following is a set of functions that extract information from KDE objects.

```@docs
get_device
get_data
get_density(kde::AbstractKDE)
get_nsamples
```

It is possible to create a KDE object from the samples to be used for the estimation with

```@docs
initialize_kde
```

Setting the density currently stored in the KDE object to a new set of values is done with

```@docs
set_density!
```

whereas resetting the density to `NaN` can be done with

```@docs
set_nan_density!
```

Creating sets of sample indices of bootstrap samples from the KDE object is also possible with

```@docs
bootstrap_indices
```
