```@meta
CurrentModule = ParallelKDE
```

# [Estimators Interface](@id APIEstimators)

This module contains the estimator definitions and dispatch infrastructure. Each estimator must implement the required interface to integrate with the current framework.

If you are implementing you own estimator, use this API as your reference point for compliance.

The supertype for all estimators is `AbstractEstimator`, as described below.

```@docs
DensityEstimators.AbstractEstimator
```

Estimators need to be registered so that users can access them using a symbol as a name. This is done with:

```@docs
DensityEstimators.add_estimator!
```

The function that will be called by the [User API](@ref "APIParallelKDE") to create an estimator is:

```@docs
DensityEstimators.estimate!(estimator_name::Symbol, kde::AbstractKDE; kwargs...)
```

which requires the following methods to be implemented for the estimator:

```@docs
DensityEstimators.initialize_estimator
DensityEstimators.estimate!(estimator_type::AbstractEstimator, kde::AbstractKDE; kwargs...)
```
