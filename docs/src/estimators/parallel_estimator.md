# [Parallel Estimator](@id ParallelEstimator)

The Parallel Estimator is the primary estimator provided by the package. It is designed to operate on dense regular grids and is optimized for parallel performance across different hardware configurations:
- Serial (single-threaded)
- Threaded (multi-core CPU)
- CUDA (GPU acceleration)

The underlying ideas for this estimator can be found in [1]

## Usage

To use this estimator, call the `estimate_density!` method and pass `:parallelEstimator` to the `estimation_method` argument.

```julia
estimate_density!(density_estimation, :parallelEstimator; kwargs...)
```

### Available Keyword Arguments
- `method`: The method to use for the estimation. Options are `:serial`, `:threaded`, or `:cuda`. Default is `:serial` for CPU and `:cuda` must be selected for CUDA devices.
- `n_bootstraps`: Number of bootstraps to use for the estimation. Default is 100.
- `time_final`: Maximum bandwidth to iterate to. Default is given by Silverman's rule.
- `time_step`: Size of the time step between iterations. Default is chosen for 1000 iterations.
- `n_steps`: Number of iterations to perform. Default is 1000. This has precedence over `time_step`.
- `threshold_crossing_percentage`: Percentage of the total time steps to require beyond the thresholds before entering the corresponding propagation regime. Default is 0.01 (1%).
- `eps_high`: Threshold for convergence of high density regions. Default is -2.5.
- `eps_low_id`: Threshold for convergence of low density regions. Default is 2.5.
- `eps_low`: Threshold for convergence of low density regions. Default is 10.0.
- `alpha`: Weight given to the first derivative as indicator for stopping propagation over the second derivative. Default is 0.75.

## References

[1] [Novel Rapid Approach for Gaussian Kernel Density Estimation]
