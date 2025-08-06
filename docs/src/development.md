# [Development](@id Development)

This section is for contributors and advanced users interested in extending or modifying the package. To request features or report issues, please [open an issue on GitHub](https://github.com/chrissm23/ParallelKDE.jl/issues).

## Adding new estimators
To add a new estimator:

1. Create a new subtype of the `AbstractEstimator` [interface](@ref "APIEstimators").
2. Register your new estimator using the `add_estimator!` function.
3. Implement the `initialize_estimator` method for your type, making use of any available [grid](@ref "APIGrids"), [KDE](@ref "APIKDEs"), and [device](@ref "APIDevices") abstractions. This method initializes the state of your estimator.
4. Implement the `estimate!` method for your type. This method should modify the `density` array in the instance of the `AbstractKDE` that it takes as argument, and optionally, also modify the estimator's state.

See the [Estimators API](@ref "APIEstimators") for more details on the required methods and how to implement them.

## Current tools

The package already provides a number of reusable utilities for building new estimators, including:

- Framework for device and method management (CPU, CUDA)
- Grid generation and manipulation functions
- Density object construction and manipulation
- Initialization of approximated empirical distributions
- Kernel convolution routines in Fourier space
