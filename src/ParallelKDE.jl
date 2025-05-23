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
  initialize_estimation,
  estimate_density!

# Grids exports
export initialize_grid,
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

abstract type AbstractDensityEstimation end

struct DensityEstimation{K<:AbstractKDE,G<:Union{Nothing,AbstractGrid}} <: AbstractDensityEstimation
  kde::K
  grid::G
end

function initialize_estimation(
  data;
  grid::Union{Bool,G}=false,
  grid_ranges=nothing,
  dims=nothing,
  grid_bounds=nothing,
  grid_padding=nothing,
  device=:cpu,
)::DensityEstimation where {G<:AbstractGrid}
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

  elseif get_device(grid) !== device
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

has_grid(density_estimation::DensityEstimation)::Bool =
  density_estimation.grid !== nothing

function estimate_density!(
  density_estimation::DensityEstimation,
  estimation::Symbol;
  kwargs...
)::Nothing
  set_nan_density!(density_estimation.kde)

  if has_grid(density_estimation)
    grid = density_estimation.grid
    estimate!(estimation, density_estimation.kde; grid, kwargs...)
  else
    estimate!(estimation, density_estimation.kde; kwargs...)
  end

  return nothing
end

end
