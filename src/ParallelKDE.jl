module ParallelKDE

include("Devices.jl")
include("Grids/Grids.jl")
include("KDEs/KDEs.jl")
include("StatisticsPropagation/DirectSpace.jl")
include("StatisticsPropagation/FourierSpace.jl")
include("DensityEstimators/DensityEstimators.jl")

using .Devices
using .Grids
using .KDEs
using .DirectSpace
using .FourierSpace
using .DensityEstimators

using StaticArrays,
  StatsBase,
  FFTW,
  CUDA

using Statistics,
  LinearAlgebra

# General API exports
export AbstractDensityEstimation,
  DensityEstimation,
  has_grid,
  initialize_estimation,
  estimate_density!,
  get_grid

# Grids exports
export initialize_grid,
  find_grid,
  spacings,
  bounds,
  low_bounds,
  high_bounds,
  get_coordinates,
  fftgrid,
  initial_bandwidth

# KDE exports
export initialize_kde,
  get_nsamples,
  get_data,
  bootstrap_indices,
  get_density,
  set_density!,
  set_nan_density!

# Dirac sequences
export initialize_dirac_sequence

"""
    AbstractDensityEstimation

Supertype for all density estimation objects.

This is the base for all objects that intended to store the estimated density, and, optionally,
the grid on which the density is estimated.

See also [`DensityEstimation`](@ref) for the concrete implementation.
"""
abstract type AbstractDensityEstimation end

"""
    DensityEstimation{K<:AbstractKDE,G<:Union{Nothing,AbstractGrid}}

Concrete type for density estimation objects.

This type holds a kernel density estimation (KDE) object `kde` and an optional grid `grid`.
"""
struct DensityEstimation{K<:AbstractKDE,G<:Union{Nothing,AbstractGrid}} <: AbstractDensityEstimation
  kde::K
  grid::G
end

"""
    initialize_estimation(data; kwargs...)

Initialize a density estimation object based on the provided data.

# Arguments
- `data::Union{AbstractMatrix,AbstractVector{<:AbstractVector}}`: The data to be used for density estimation.
- `grid::Union{Bool,G<:AbstractGrid}=false`: Whether to create a grid for the density estimation.
If `true`, a grid will be created based on the data ranges. A grid can also be provided directly.
- `grid_ranges=nothing`: The ranges for the grid coordinates if `grid` is `true`.
This has priority over other grid parameters.
- `dims=nothing`: The dimensions of the grid if `grid` is `true`.
- `grid_bounds=nothing`: The bounds for the grid if `grid` is `true`.
- `grid_padding=nothing`: Padding for the grid if `grid` is `true`.
- `device=:cpu`: The device to use for the density estimation. It should be compatible with the estimator to be used.

# Examples
```julia
data = randn(1, 1000);
density_estimation = initialize_estimation(data; grid=true, grid_ranges=-5.0:0.1:5.0, device=:cpu);
```
"""
function initialize_estimation(
  data::Union{AbstractMatrix,AbstractVector{<:AbstractVector}};
  grid::Union{Bool,G}=false,
  grid_ranges=nothing,
  dims=nothing,
  grid_bounds=nothing,
  grid_padding=nothing,
  device=:cpu,
) where {G<:AbstractGrid}
  if isa(data, AbstractMatrix)
    n_dims = size(data, 1)
  elseif isa(data, AbstractVector)
    n_dims = length(data[1])
    data = reduce(hcat, data)
  else
    throw(ArgumentError("Data must be a matrix or vector."))
  end

  fallback_option = :cpu
  if !haskey(Devices.AVAILABLE_DEVICES, device)
    throw(ArgumentError("Invalid device: $device"))
  elseif device == :cuda
    if !CUDA.functional()
      @warn "No functional CUDA detected. Falling back to ':cpu'"
      device = fallback_option
    end
  end

  # Create grid object if required
  if grid isa Bool
    if grid
      if grid_ranges !== nothing
        grid = initialize_grid(grid_ranges; device)
      else
        grid = find_grid(data; grid_bounds, grid_dims=dims, grid_padding, device)
      end

    else
      grid = nothing
    end

  elseif get_device(grid) !== get_device(device)
    throw(ArgumentError("Grid must be of the same device as the KDE."))
  end

  # Create KDE object
  if grid === nothing && dims === nothing
    throw(ArgumentError("Grid must be required if dims is not specified."))
  elseif grid !== nothing
    dims = size(grid)
  end

  kde = initialize_kde(data, dims; device)

  return DensityEstimation(kde, grid)

end

"""
    has_grid(density_estimation::DensityEstimation)

Return `true` if the `DensityEstimation` object has a grid associated with it, `false` otherwise.

# Examples
```julia
data = randn(1, 1000);
density_estimation = initialize_estimation(data; grid=true, grid_ranges=-5.0:0.1:5.0, device=:cpu);
has_grid(density_estimation)
```
"""
has_grid(density_estimation::DensityEstimation) = density_estimation.grid !== nothing

"""
    get_grid(density_estimation::DensityEstimation)

Extract the grid from a `DensityEstimation` object.
"""
get_grid(density_estimation::DensityEstimation) = density_estimation.grid

"""
    get_density(density_estimation::DensityEstimation; normalize=false, dx=nothing)

Obtain the estimated density from a `DensityEstimation` object.

If the `normalize` argument is set to `true`, the density will be normalized. If density_estimation
has a grid, its spacing will be used for normalization. Otherwise, `dx` must be provided to normalize the density.
"""
function KDEs.get_density(density_estimation::DensityEstimation; normalize=false, dx=nothing)
  density = get_density(density_estimation.kde)
  if normalize && has_grid(density_estimation)
    grid = density_estimation.grid
    density ./= sum(density) * prod(spacings(grid))
  elseif normalize && dx !== nothing
    density ./= sum(density) * dx
  elseif normalize
    throw(ArgumentError("Normalization requires a grid or dx to be specified."))
  end

  return density
end

"""
    estimate_density!(density_estimation::DensityEstimation, estimation_method::Symbol; kwargs...)

Estimate the density using the specified method and update the `DensityEstimation` object.

For a list of available estimation methods and their keywords, see the documentation for the specific estimator.
"""
function estimate_density!(
  density_estimation::DensityEstimation,
  estimation_method::Symbol;
  kwargs...
)
  set_nan_density!(density_estimation.kde)

  if has_grid(density_estimation)
    grid = density_estimation.grid
    estimate!(estimation_method, density_estimation.kde; grid, kwargs...)
  else
    estimate!(estimation_method, density_estimation.kde; kwargs...)
  end

  return nothing
end

end
