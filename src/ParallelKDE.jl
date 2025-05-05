module ParallelKDE

include("Devices.jl")
include("Grids/Grids.jl")
include("KDEs/KDEs.jl")
include("StatisticsPropagation/FourierSpace.jl")
include("StatisticsPropagation/DirectSpace.jl")
include("DensityEstimations/DensityEstimations.jl")

using .Devices
using .Grids
using .KDEs
using .FourierSpace
using .DirectSpace
using .DensityEstimations

using StaticArrays,
  StatsBase,
  FFTW,
  CUDA

using Statistics,
  LinearAlgebra

# TODO: Check which other functions or types from the other modules may be useful to export
export AbstractDensityEstimator,
  initialize_estimator,
  DensityEstimator,
  estimate_density!

abstract type AbstractDensityEstimator end

struct DensityEstimator{K<:AbstractKDE,G<:Union{Nothing,AbstractGrid}} <: AbstractDensityEstimator
  kde::K
  grid::G
end

function initialize_estimator(
  data;
  device=:cpu,
  grid::Union{Bool,G}=false,
  grid_ranges=nothing,
  dims=nothing,
  grid_bounds=nothing,
  grid_padding=nothing,
)::DensityEstimator where {G<:AbstractGrid}
  if isa(data, AbstractMatrix)
    n_dims = size(data, 1)
  elseif isa(data, AbstractVector)
    n_dims = length(data[1])
    data = reduce(hcat, data)
  else
    throw(ArgumentError("Data must be a matrix or vector."))
  end

  if !haskey(Devices.AVAILABLE_DEVICES, device)
    throw(ArgumentError("Invalid device: $device"))
  end
  device = Devices.AVAILABLE_DEVICES[device]

  # Create grid object if required
  if grid isa Bool
    if grid
      if grid_ranges !== nothing
        grid = initialize_grid(device, grid_ranges)
      else
        grid = find_grid(device, data; grid_bounds, dims, nothing, grid_padding)
      end

    else
      grid = nothing
    end

  elseif Device(grid) !== device
    throw(ArgumentError("Grid must be of the same device as the KDE."))
  end

  # Create KDE object
  if grid === nothing && dims === nothing
    throw(ArgumentError("Grid must be required if dims is not specified."))
  elseif grid !== nothing
    dims = size(grid)
  end

  kde = initialize_kde(device, data, dims)

  return DensityEstimator(kde, grid)

end

has_grid(density_estimator::DensityEstimator)::Bool =
  density_estimator.grid !== nothing

function estimate_density!(
  density_estimator::DensityEstimator,
  estimation::Symbol;
  kwargs...
)::Nothing
  set_nan_density!(density_estimator.kde)

  if density_estimator.grid !== nothing
    grid = density_estimator.grid
    estimate!(estimation, density_estimator.kde; grid, kwargs...)
  else
    estimate!(estimation, density_estimator.kde; kwargs...)
  end

  return nothing
end

end
