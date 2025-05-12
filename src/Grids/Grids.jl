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

function initialize_grid(
  ::IsCPU,
  ranges::Union{Tuple{Vararg{AbstractVector{T}}},AbstractVector{<:AbstractVector{T}}};
)::Grid where {T<:Real}
  return Grid(ranges)
end
function initialize_grid(
  ::IsCUDA,
  ranges::Union{Tuple{Vararg{AbstractVector{T}}},AbstractVector{<:AbstractVector{T}}};
  b32=true
)::CuGrid where {T<:Real}
  return CuGrid(ranges; b32)
end

struct Grid{N,T<:Real,M} <: AbstractGrid{N,T,M}
  coordinates::SVector{N,Union{StepRangeLen,Frequencies}}
  spacings::SVector{N,T}
  bounds::SMatrix{2,N,T}
end
function Grid(
  ranges::Union{Tuple{Vararg{AbstractVector{T}}},AbstractVector{<:AbstractVector{T}}},
) where {T<:Real}
  N = length(ranges)

  ranges = SVector{N}(ranges)
  spacings = SVector{N}(step.(ranges))

  if ranges isa Tuple
    extreme_points = collect(extrema.(ranges))
  else
    extreme_points = extrema.(ranges)
  end
  bounds = SMatrix{2,N,T}(
    reinterpret(reshape, T, extreme_points)
  )

  return Grid{N,T,N + 1}(ranges, spacings, bounds)
end

function Base.size(grid::Grid{N,T,M}) where {N,T<:Real,M}
  return ntuple(i -> length(grid.coordinates[i]), N)
end

function Base.ndims(::Grid{N,T,M}) where {N,T<:Real,M}
  return N
end

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

function Base.broadcastable(grid::Grid{N,T,M}) where {N,T<:Real,M}
  return get_coordinates(grid)
end

struct CuGrid{N,T<:Real,M} <: AbstractGrid{N,T,M}
  coordinates::CuArray{T,M}
  spacings::CuVector{T}
  bounds::CuMatrix{T}
end
function CuGrid(
  ranges::Union{Tuple{Vararg{AbstractVector{T}}},AbstractVector{<:AbstractVector{T}}};
  b32::Bool=true
) where {T<:Real}
  N = length(ranges)

  if ranges isa Tuple
    ranges_vec = collect(ranges)
  else
    ranges_vec = ranges
  end

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
      reinterpret(reshape, T, extrema.(ranges))
    )

    return CuGrid{N,T,N + 1}(coordinates, spacings, bounds)
  end

end

function Base.size(grid::CuGrid{N,T,M}) where {N,T<:Real,M}
  return size(grid.coordinates)[2:end]
end

function Base.ndims(::CuGrid{N,T,M}) where {N,T<:Real,M}
  return N
end

function get_coordinates(grid::CuGrid{N,T,M}) where {N,T<:Real,M}
  return grid.coordinates
end

function Base.broadcastable(grid::CuGrid{N,T,M}) where {N,T<:Real,M}
  return get_coordinates(grid)
end

Device(::Grid) = IsCPU()
Device(::CuGrid) = IsCUDA()

function spacings(grid::AbstractGrid)::AbstractVector
  return grid.spacings
end

function bounds(grid::AbstractGrid)::AbstractMatrix
  return grid.bounds
end

function low_bounds(grid::AbstractGrid)::AbstractVector
  return grid.bounds[1, :]
end

function high_bounds(grid::AbstractGrid)::AbstractVector
  return grid.bounds[2, :]
end

function initial_bandwidth(grid::AbstractGrid)::AbstractVector
  return spacings(grid) ./ 2
end

function fftgrid(grid::AbstractGrid)::AbstractGrid
  return fftgrid(Device(grid), grid)
end
function fftgrid(::IsCPU, grid::AbstractGrid)::Grid
  n_points = size(grid)
  spacing = spacings(grid)
  fourier_coordinates = @. 2π * fftfreq(n_points, 1 / spacing)

  if fourier_coordinates isa Frequencies
    fourier_coordinates = [fourier_coordinates,]
  end

  return Grid(fourier_coordinates)
end
function fftgrid(::IsCUDA, grid::AbstractGrid)::CuGrid
  n_points = size(grid)
  spacing = Array{Float32}(spacings(grid))
  fourier_coordinates = @. Float32(2π) * fftfreq(n_points, 1 / spacing)

  if fourier_coordinates isa Frequencies
    fourier_coordinates = [fourier_coordinates,]
  end

  return CuGrid(fourier_coordinates)
end

function find_grid(
  device::AbstractDevice,
  data;
  grid_bounds=nothing,
  grid_dims=nothing,
  grid_steps=nothing,
  grid_padding=nothing,
)
  # Turn data into array
  if isa(data, AbstractMatrix)
    n_dims = size(data, 1)
  elseif isa(data, AbstractVector)
    n_dims = length(data[1])
    data = reduce(hcat, data)
  else
    throw(ArgumentError("Data must be a matrix or vector."))
  end

  # Find grid bounds
  if grid_bounds === nothing
    grid_bounds = [extrema(row) for row in eachrow(data)]
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
    grid_dims = fill(default_size, n_dims)
  elseif grid_dims !== nothing
    grid_ranges = [
      range(bounds[1], stop=bounds[2], length=size)
      for (bounds, size) in zip(grid_bounds, grid_dims)
    ]
  elseif grid_steps !== nothing
    grid_ranges = [
      range(bounds[1], stop=bounds[2], step=step)
      for (bounds, step) in zip(grid_bounds, grid_steps)
    ]
  end

  if device isa IsCPU
    return Grid(grid_ranges)
  elseif device isa IsCUDA
    return CuGrid(grid_ranges)
  end
end

# NOTE: This function is not used in the codebase, but it may be useful for testing purposes.
function Base.isapprox(
  grid1::T,
  grid2::S;
  atol::Real=0.0,
  rtol::Real=atol > 0 ? 0 : √eps(),
)::Bool where {T<:AbstractGrid,S<:AbstractGrid}
  is_T_CuGrid = T <: CuGrid
  is_S_CuGrid = S <: CuGrid

  if is_T_CuGrid ⊻ is_S_CuGrid
    if is_S_CuGrid
      grid1_coordinates = get_coordinates(grid1)
      grid1_spacings = spacings(grid1)
      grid1_bounds = bounds(grid1)
      grid2_coordinates = Array{Float32}(get_coordinates(grid2))
      grid2_spacings = Array{Float32}(spacings(grid2))
      grid2_bounds = Array{Float32}(bounds(grid2))
    elseif is_T_CuGrid
      grid1_coordinates = Array{Float32}(get_coordinates(grid1))
      grid1_spacings = Array{Float32}(spacings(grid1))
      grid1_bounds = Array{Float32}(bounds(grid1))
      grid2_coordinates = get_coordinates(grid2)
      grid2_spacings = spacings(grid2)
      grid2_bounds = bounds(grid2)
    end
  else
    grid1_coordinates = get_coordinates(grid1)
    grid1_spacings = spacings(grid1)
    grid1_bounds = bounds(grid1)
    grid2_coordinates = get_coordinates(grid2)
    grid2_spacings = spacings(grid2)
    grid2_bounds = bounds(grid2)
  end

  coordinates_check = isapprox(grid1_coordinates, grid2_coordinates, atol=atol, rtol=rtol)
  spacings_check = isapprox(grid1_spacings, grid2_spacings, atol=atol, rtol=rtol)
  bounds_check = isapprox(grid1_bounds, grid2_bounds, atol=atol, rtol=rtol)

  if coordinates_check && spacings_check && bounds_check
    return true
  else
    return false
  end
end

end
