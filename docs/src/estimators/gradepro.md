# [GradePro Estimator](@id GradeproEstimator)

The GradePro Estimator is the primary estimator provided by the package. It is designed to operate on dense regular grids and is optimized for parallel performance across different hardware configurations:

- Serial (single-threaded)
- Threaded (multi-core CPU)
- CUDA (GPU acceleration)

The underlying ideas for this estimator can be found in Sustay Martinez et al[^1].

## Usage

To use this estimator, call the `estimate_density!` method and pass `:gradepro` to the `estimation_method` argument.

```julia
estimate_density!(density_estimation, :gradepro; kwargs...)
```

### Available Keyword Arguments

- `method`: The method to use for the estimation. Options are `:serial`, `:threaded`, or `:cuda`. Default is `:serial` for CPU and `:cuda` must be selected for CUDA devices.
- `n_bootstraps`: Number of bootstraps to use for the estimation. Default is 100.
- `bw_final`: Maximum bandwidth to iterate to. Default is given by Silverman's rule.
- `bw_step`: Size of the bandwidth step between iterations. Default is chosen for 250 iterations.
- `n_steps`: Number of iterations to perform. Default is 250. This has precedence over `bw_step`.
- `alpha_s`: Percentage of the total bandwidth steps of persistence beyond the thresholds before registering the crossing event. Default is 0.0 (0%) for 1D, and 0.06 (6%) for 2D.
- `alpha_os`: Maximum percentage of the total bandwidth steps allowed before the next estimation update for halting the estimated density updates. Default is 0.1 (10%).
- `eps`: Threshold for convergence of low density regions. Default is 2.0 for 1D, and -0.5 for 2D.

!!! warning "Parameters for higher dimensions"
    Currently there are no tested parameters for estimations of dimensionality higher than 2D.
    The default in those cases is to use the 2D parameters.

!!! warning "High dimensionality with CUDA"
    Implementation with CUDA was not written to support higher dimensionality than 3D estimations.

## References

[^1]: [Sustay Martinez, C., Quoika P. K., Zacharias, M. (2025). Novel Rapid Approach for Adaptive Gaussian Kernel Density Estimation: Gridpoint-wise Propagation of Anisotropic Diffusion Equation]
