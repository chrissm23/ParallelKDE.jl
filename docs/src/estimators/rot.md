# [Rules of Thumb Estimator](@id RotEstimator)

The Rules of Thumb Estimator (ROT) is the estimator for calculating the bandwidth of a kernel density estimation based on rules of thumb. It is particularly useful when you want to quickly estimate the bandwidth if your underlying data is close to a Gaussian distribution.

There are currently two implemented rules of thumb:

- Silverman's rule [^1]
- Scott's rule [^2]

## Usage

To use this estimator, call the `estimate_density!` method and pass `:rot` to the `estimation_method` argument.

```julia
estimate_density!(density_estimation, :rot; kwargs...)
```

### Available Keyword Arguments

- `method`: The method to use for the estimation. Options are `:serial`, `:threaded`, or `:cuda`. Default is `:serial` for CPU and `:cuda` must be selected for CUDA devices.
- `rule_of_thumb`: The rule of thumb to use for the bandwidth estimation. Current options are `:silverman` or `:scott`. Default is `:scott`.

!!! warning "High dimensionality with CUDA"
    Implementation with CUDA was not written to support higher dimensionality than 3D estimations.

[^1]: [Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis. *Chapman and Hall/CRC.*](https://doi.org/10.1201/9781315140919)

[^2]: [Scott, D. W. (1992). Multivariate Density Estimation: Theory, Practice, and Visualization. *Wiley & Sons.*](https://doi.org/10.1002/9780470316849)
