module Grids

using CUDA: DeviceIterator
using ..Devices

using FFTW
using StaticArrays
using CUDA

export AbstractGrid,
  Grid,
  CuGrid,
  initialize_grid,
  spacings,
  bounds,
  low_bounds,
  high_bounds,
  initial_bandwidth,
  get_coordinates,
  fftgrid,
  find_grid

abstract type AbstractGrid{N,T<:Real,M} end

function initialize_grid(
  ranges::Union{Tuple{Vararg{AbstractVector{T}}},AbstractVector{<:AbstractVector{T}}};
  kwargs...
)::AbstractGrid where {T<:Real}
  kwargs_dict = Dict(kwargs)
  device = pop!(kwargs_dict, :device, :cpu)
  device_type = obtain_device(device)

  return initialize_grid(device_type, ranges; kwargs_dict...)
end

initialize_grid(::IsCPU, ranges) = Grid(ranges)
initialize_grid(::IsCUDA, ranges) = CuGrid(ranges)

struct Grid{N,T<:Real,M,V} <: AbstractGrid{N,T,M}
  coordinates::SVector{N,V}
  spacings::SVector{N,T}
  bounds::SMatrix{2,N,T}

  function Grid(ranges::NTuple{N,<:AbstractVector{T}}) where {N,T<:Real}
    if !hasmethod(step, Tuple{typeof(first(ranges))})
      throw(ArgumentError("Elements of ranges must implement 'step' method."))
    end

    ranges_static = SVector{N}(ranges)
    spacings = SVector{N}(step.(ranges_static))

    extreme_points = extrema.(ranges_static)
    bounds = SMatrix{2,N,T}(
      reinterpret(reshape, T, extreme_points)
    )

    return Grid{N,T,N + 1}(ranges_static, spacings, bounds)
  end
end
function Grid(ranges)
  ranges_tuple = ntuple(i -> ranges[i], Val(length(ranges)))

  return Grid(ranges_tuple)
end

struct CuGrid{N,T<:Real,M} <: AbstractGrid{N,T,M}
  coordinates::CuArray{T,M}
  spacings::CuVector{T}
  bounds::CuMatrix{T}

  function CuGrid(ranges::NTuple{N,<:AbstractVector{T}}; b32=true) where {N,T<:Real}
    if !hasmethod(step, Tuple{typeof(first(ranges))})
      throw(ArgumentError("Elements of ranges must implement 'step' method."))
    end

    ranges_vec = collect(ranges)

    complete_array = reinterpret(
      reshape,
      T,
      collect(Iterators.product(ranges_vec...))
    )
    if N == 1
      complete_array = reshape(complete_array, 1, :)
    end

    if b32
      coordinates = CuArray{Float32}(complete_array)
      spacings = CuArray{Float32}(step.(ranges_vec))
      bounds = CuArray{Float32}(
        reinterpret(reshape, T, extrema.(ranges))
      )

      return CuGrid{N,Float32,N + 1}(coordinates, spacings, bounds)
    else
      coordinates = CuArray{T}(complete_array)
      spacings = CuArray{T}(step.(ranges_vec))
      bounds = CuArray{T}(
        reinterpret(reshape, T, extrema.(ranges_vec))
      )

      return CuGrid{N,T,N + 1}(coordinates, spacings, bounds)
    end
  end
end
function CuGrid(ranges; b32=true)
  ranges_tuple = ntuple(i -> ranges[i], Val(length(ranges)))

  return CuGrid(ranges_tuple; b32)
end

Base.size(grid::Grid{N,T,M}) where {N,T<:Real,M} = ntuple(i -> length(grid.coordinates[i]), Val(N))
Base.size(grid::CuGrid{N,T,M}) where {N,T<:Real,M} = size(grid.coordinates)[2:end]

Base.ndims(::Grid{N,T,M}) where {N,T<:Real,M} = N
Base.ndims(::CuGrid{N,T,M}) where {N,T<:Real,M} = N

function get_coordinates(grid::Grid{N,T,M}) where {N,T<:Real,M}
  complete_array = reinterpret(
    reshape,
    T,
    collect(Iterators.product(grid.coordinates...))
  )

  if N == 1
    complete_array = reshape(complete_array, 1, :)
  end

  return complete_array
end
function get_coordinates(grid::CuGrid)
  return grid.coordinates
end

Base.broadcastable(grid::AbstractGrid) = get_coordinates(grid)

Devices.Device(::Grid) = IsCPU()
Devices.Device(::CuGrid) = IsCUDA()

spacings(grid::AbstractGrid) = grid.spacings
bounds(grid::AbstractGrid) = grid.bounds
low_bounds(grid::AbstractGrid) = grid.bounds[1, :]
high_bounds(grid::AbstractGrid) = grid.bounds[2, :]
initial_bandwidth(grid::AbstractGrid) = spacings(grid) ./ 2
fftgrid(grid::AbstractGrid) = fftgrid(Device(grid), grid)

function fftgrid(::IsCPU, grid::AbstractGrid{N,T}) where {N,T<:Real}
  n_points = size(grid)
  spacing = spacings(grid)
  fourier_coordinates = @. 2π * fftfreq(n_points, 1 / spacing)

  if N == 1
    fourier_coordinates_tuple = (fourier_coordinates,)
  else
    fourier_coordinates_tuple = ntuple(i -> fourier_coordinates, N)
  end

  return Grid(fourier_coordinates)
end
function fftgrid(::IsCUDA, grid::AbstractGrid{N,T}) where {N,T<:Real}
  n_points = size(grid)
  spacing = Array{T}(spacings(grid))
  fourier_coordinates = @. T(2π) * fftfreq(n_points, 1 / spacing)

  if N == 1
    fourier_coordinates_tuple = (fourier_coordinates,)
  else
    fourier_coordinates_tuple = ntuple(i -> fourier_coordinates, N)
  end

  return CuGrid(fourier_coordinates)
end

function find_grid(
  data;
  grid_bounds=nothing,
  grid_dims=nothing,
  grid_steps=nothing,
  grid_padding=nothing,
  device=:cpu,
)
  # Turn data into array
  if isa(data, AbstractMatrix)
    n_dims = size(data, 1)
  elseif isa(data, AbstractVector)
    n_dims = length(data[1])
    data_matrix = reduce(hcat, data)
  else
    throw(ArgumentError("Data must be a matrix or vector."))
  end

  # Find grid bounds
  if grid_bounds === nothing
    grid_bounds = [extrema(row) for row in eachrow(data_matrix)]
  end
  if grid_padding === nothing
    grid_padding = [0.1 * abs(bounds[2] - bounds[1]) for bounds in grid_bounds]
  end
  grid_bounds = [
    bounds .+ (-padding, padding)
    for (bounds, padding) in zip(grid_bounds, grid_padding)
  ]

  # Find grid ranges
  default_size = 300
  if (grid_steps === nothing) && (grid_dims === nothing)
    grid_dims = ntuple(i -> default_size, Val(n_dims))
  elseif grid_dims !== nothing
    grid_ranges = ntuple(
      i -> range(grid_bounds[i][1], stop=grid_bounds[i][2], length=grid_dims[i]),
      Val(n_dims)
    )
  elseif grid_steps !== nothing
    grid_ranges = ntuple(
      i -> range(grid_bounds[i][1], stop=grid_bounds[i][2], step=grid_steps[i]),
      Val(n_dims)
    )
  end

  return initialize_grid(grid_ranges; device)
end

end
