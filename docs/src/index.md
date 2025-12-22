```@meta
CurrentModule = ParallelKDE
```

# ParallelKDE

[ParallelKDE](https://github.com/chrissm23/ParallelKDE.jl) is a package for flexible and efficient kernel density estimation (KDE), with a strong focus on parallel implementations. Its core estimator, the [GradePro Estimator](@ref "GradeproEstimator") described here, supports CPU and GPU acceleration (threaded and CUDA), and its designed to scale with modern hardware. While the package is centered around grid-based KDEs, it also provides extensible infrastructure to support and implement other types of estimators.

The user interface is built around a modular design that separates concerns between grids, devices, density objects and estimation routines. This allows users to easily switch estimators, control execution targets (e.g., CPU or GPU), and prototype new estimation strategies without rewriting boilerplate code.

Typical usage involves:

- Instantiating an estimation in a specific device (e.g., CPU, CUDA),
- This also involves defining a grid for the estimation or using a default grid,
- Estimating the density with a chosen estimator,
- Accessing the resulting density.

## Basic Example

For example, to estimate a density on a CPU with a default grid using the [GradePro Estimator](@ref "GradeproEstimator"), you can use:

```@example 1
using ParallelKDE

data = randn(1, 10000) # 1-dimensional sample of 10000 points

density_estimation = initialize_estimation(
  data,
  grid=true, # default grid
  device=:cpu,
)
estimate_density!(
  density_estimation,
  :gradepro,
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
ENV["GKSwstype"]="nul" # hide
using Plots

p = plot(grid_coordinates, density_true, label="True Density", color=:cornflowerblue, lw=2)
plot!(p, grid_coordinates, density_estimated, label="Estimated Density", color=:firebrick, lw=2)
plot!(p, xlabel="Random Variable", ylabel="Density")
savefig("basic_usage.svg"); nothing # hide
```

![](basic_usage.svg)

## Density Estimation for Conformational Samples

Here, we exemplify the use of `ParallelKDE` to estimate the conformational density of alanine dipeptide via dihedral angles.

We start by downloading and reading the dataset of dihedral angles obtained from molecular dynamics trajectories [^1] [^2].

```@example 2
using Downloads
using NPZ

url = "http://ftp.imp.fu-berlin.de/pub/cmb-data/alanine-dipeptide-3x250ns-backbone-dihedrals.npz"
dest = joinpath(pwd(), "ala_dipeptide_dihderals.npz")
Downloads.download(url, dest)

ala_npz = npzread(dest)
ala = vcat(values(ala_npz)...)
# afterwards one may need to subsample the dataset to obtain uncorrelated samples.
nothing # hide
```

This time around we will define the grid that we want to use for the estimation instead of using the default grid:

```@example 2
using ParallelKDE

phi_range, psi_range = fill(range(-π, π, length=250), 2)
dihedral_grid = initialize_grid([phi_range, psi_range], device=:cpu) # or device=:cuda

density_estimation = initialize_estimation(
  ala', grid=dihedral_grid, device=:cpu # or device=:cuda
)
estimate_density!(density_estimation, :gradepro)
estimated_density = get_density(density_estimation)
nothing # hide
```

Finally, we can create a contour plot of the estimated density:

```@example 2
ENV["GKSwstype"]="nul" # hide
using Plots
using LaTeXStrings

p = contourf(phi_range, psi_range, estimated_density')
plot!(p, xlabel=L"$\phi$", ylabel=L"$\psi$", colorbar_title="Estimated Density")
savefig("ala_dihedrals.svg"); nothing # hide
```

![](ala_dihedrals.svg)

## GradePro vs Rules of Thumb

`ParallelKDE` also provides a [Rules of Thumb Estimator](@ref "RotEstimator") for bandwidth selection based on widely used rule-of-thumb heuristics. As with the [GradePro Estimator](@ref "GradeproEstimator"), this estimator is available in both serial and CUDA variants. In practice, the Rules of Thumb approach can be extremely fast and yields competitive accuracy when the sample distribution is close to Gaussian; however, its performance can degrade rapidly as the underlying distribution departs from Gaussianity. Now, we illustrate this trade-off by comparing results produced by both estimators on samples drawn from a bimodal distribution.

We first initialize the distribution and obtain samples from it:

```@example 3
using Distributions

distro = MixtureModel(
  Normal[
    Normal(-6.0, 0.2),
    Normal(0.0, 2),
  ]
)
samples = rand(distro, 1, 10000) # 1-dimensional sample of 10000 points
nothing # hide
```

We can again initialize a grid and the estimations for both methods with:

```@example 3
using ParallelKDE

xs = range(-10, 10, length=250)

grid_bimodal = initialize_grid([xs], device=:cpu) # or device=:cuda

estimation_gradepro = initialize_estimation(
  samples, grid=grid_bimodal, device=:cpu # or device=:cuda
)
estimation_rot = initialize_estimation(
  samples, grid=grid_bimodal, device=:cpu # or device=:cuda
)
nothing # hide
```

Now we can execute both estimations and calculate the true density with:

```@example 3
estimate_density!(estimation_gradepro, :gradepro)
density_gradepro = get_density(estimation_gradepro)

estimate_density!(estimation_rot, :rot)
density_rot = get_density(estimation_rot)

density_true = pdf.(distro, xs)
nothing # hide
```

Finally, we can visualize the results with:

```@example 3
ENV["GKSwstype"]="nul" # hide
using Plots

p = plot(xs, density_true, lw=3, ls=:dash, label="True Distribution")
plot!(p, xs, density_gradepro, lw=2, label="GradePro Estimation")
plot!(p, xs, density_rot, lw=2, label="Rule of Thumb Estimation")
plot!(p, xlabel="Random Variable", ylabel="Density")
savefig("rot_vs_gradepro.svg"); nothing # hide
```

![](rot_vs_gradepro.svg)

## Usage Summary

As is exemplified above, it is possible to initialize an estimation using

```@docs; canonical=false
initialize_estimation
```

Then, the density can be estimated with a chosen estimator and its settings using

```@docs; canonical=false
estimate_density!
```

Finally, the estimated density can be accessed using

```@docs; canonical=false
get_density(density_estimation::DensityEstimation; normalize, dx)
```

More details regarding the currently implemented estimators as well as further information about the package can be found throughout the documentation.

```@contents
```

[^1]: [Nüske, F. et al (2017). Markov state models from short non-equilibrium simulations—Analysis and correction of estimation bias. *J. Chem. Phys.*](https://doi.org/10.1063/1.49765)
[^2]: [Wehmeyer, C. and Noé, F. (2018). Time-lagged autoencoders: deep learning of slow collective variables for molecular kinetics. *J. Chem. Phys.*](https://doi.org/10.1063/1.5011399)
