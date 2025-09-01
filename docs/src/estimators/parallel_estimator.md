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
- `time_step`: Size of the time step between iterations. Default is chosen for 250 iterations.
- `n_steps`: Number of iterations to perform. Default is 250. This has precedence over `time_step`.
- `fraction_buffer`: Percentage of the total time steps of persistence beyond the thresholds before registering the crossing event. Default is 0.03 (3%).
- `fraction_stopping`: Maximum percentage of the total time steps allowed before the next estimation update before halting the estimated density updates. Default is 0.2 (20%).
- `eps_high`: Threshold for convergence of high density regions. Default is 0.0 for CPU and -1.0 for GPU.
- `eps_low_id`: Threshold for convergence of low density regions. Default is 2.0 for CPU and 6.0 for GPU.

## References

[1] [Novel Rapid Approach for Gaussian Kernel Density Estimation]
