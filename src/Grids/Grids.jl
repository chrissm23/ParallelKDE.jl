module Grids

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

"""
    initialize_grid(ranges; device=:cpu, b32=false)
    initialize_grid(ranges...; device=:cpu, b32=false)

Create a grid object based on the provided ranges of coordinates for each dimension.

`device` specifies the device type (default is `:cpu` but `:cuda` is also implemented),
and, if a GPU is used, `b32` determines whether to use `Float32` or `Float64`
for the grid coordinates. For CPU grids, `b32` is ignored and the data type matches the
data type of `ranges`.

# Examples
```julia
initialize_grid(0.0:0.1:1.0, 0.0:0.1:1.0, device=:cuda, b32=true)
```
"""
initialize_grid(ranges::AbstractVector{<:AbstractVector}; kwargs...) = initialize_grid(Tuple(ranges); kwargs...)
initialize_grid(ranges::Vararg{AbstractVector}; kwargs...) = initialize_grid(ranges; kwargs...)
function initialize_grid(ranges::NTuple{N,<:AbstractVector{T}}; device=:cpu, kwargs...) where {N,T<:Real}
  if (get_device(device) isa IsCUDA) && !CUDA.functional()
    @warn "No functional CUDA detected. Falling back to ':cpu'."
    device = :cpu
  end
  device_type = get_device(device)

  return initialize_grid(device_type, ranges; kwargs...)
end
initialize_grid(::IsCPU, ranges; b32=false) = Grid(ranges)
initialize_grid(::IsCUDA, ranges; b32=true) = CuGrid(ranges; b32)

"""
    AbstractGrid{N,T<:Real,M}

Supertype for all grid types, where `N` is the number of dimensions, `T` is the type of the
coordinates (usually `Float64` or `Float32`), and `M` is the number of dimensions in the
underlying array (usually `N + 1`).
"""
abstract type AbstractGrid{N,T<:Real,M} end

"""
    Grid{N,T<:Real,M}

CPU object for a grid with `N` dimensions, `T` type for coordinates, and `M=N+1` dimensions
for the underlying array.
"""
struct Grid{N,T<:Real,M} <: AbstractGrid{N,T,M}
  coordinates::SVector{N,Union{StepRangeLen,Frequencies}}
  spacings::SVector{N,T}
  bounds::SMatrix{2,N,T}

  function Grid(ranges::NTuple{N,<:AbstractVector{T}}; b32=false) where {N,T<:Real}
    if !hasmethod(step, Tuple{typeof(first(ranges))})
      throw(ArgumentError("Elements of ranges must implement 'step' method."))
    end

    ranges_static = SVector{N}(ranges)
    spacings = SVector{N}(step.(ranges_static))

    extreme_points = extrema.(ranges_static)
    bounds = SMatrix{2,N,T}(
      reinterpret(reshape, T, extreme_points)
    )

    return new{N,T,N + 1}(ranges_static, spacings, bounds)
  end
end

"""
    CuGrid{N,T<:Real,M}

CUDA object for a grid with `N` dimensions, `T` type for coordinates, and `M=N+1` dimensions
for the underlying array.
"""
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
        reinterpret(reshape, T, extrema.(ranges_vec))
      )

      return new{N,Float32,N + 1}(coordinates, spacings, bounds)
    else
      coordinates = CuArray{T}(complete_array)
      spacings = CuArray{T}(step.(ranges_vec))
      bounds = CuArray{T}(
        reinterpret(reshape, T, extrema.(ranges_vec))
      )

      return new{N,T,N + 1}(coordinates, spacings, bounds)
    end
  end
end

"""
    size(grid::AbstractGrid)

Return the size of the grid, which is a tuple containing the number of points in each dimension.
"""
Base.size(grid::Grid{N,T,M}) where {N,T<:Real,M} = ntuple(i -> length(grid.coordinates[i]), Val(N))
Base.size(grid::CuGrid) = size(grid.coordinates)[2:end]

"""
    ndims(grid::AbstractGrid)

Return the number of dimensions of the grid.
"""
Base.ndims(::Grid{N,T,M}) where {N,T<:Real,M} = N
Base.ndims(::CuGrid{N,T,M}) where {N,T<:Real,M} = N

"""
    get_coordinates(grid::AbstractGrid)

Return a view of the coordinates of the grid as an array.
"""
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

"""
    get_device(grid::AbstractGrid)

Identify the device type of the grid, returning `IsCPU` for CPU grids and `IsCUDA` for CUDA grids.
"""
Devices.get_device(::Grid) = IsCPU()
Devices.get_device(::CuGrid) = IsCUDA()

"""
    spacings(grid::AbstractGrid)

Return the spacings of the grid, which is a vector containing the spacing between points in each dimension.
"""
spacings(grid::AbstractGrid) = grid.spacings

"""
    bounds(grid::AbstractGrid)

Return the bounds of the grid, which is a 2xN matrix containing the minimum and maximum values for each dimension.
"""
bounds(grid::AbstractGrid) = grid.bounds

"""
    low_bounds(grid::AbstractGrid)

Return the lower bounds of the grid, which is a vector containing the minimum values for each dimension.
"""
low_bounds(grid::AbstractGrid) = grid.bounds[1, :]

"""
    high_bounds(grid::AbstractGrid)

Return the upper bounds of the grid, which is a vector containing the maximum values for each dimension.
"""
high_bounds(grid::AbstractGrid) = grid.bounds[2, :]

"""
    initial_bandwidth(grid::AbstractGrid)

Return the initial bandwidth for the grid, which is half of the spacings in each dimension.
"""
initial_bandwidth(grid::AbstractGrid) = spacings(grid) ./ 2

"""
    fftgrid(grid::AbstractGrid)

Calculate the Fourier grid based on the spacings of the input grid.
The Fourier grid contains frequencies corresponding to the grid points in each dimension.
"""
function fftgrid(grid::Grid{N,T,M}) where {N,T<:Real,M}
  n_points = size(grid)
  spacing = spacings(grid)
  fourier_coordinates = @. 2π * fftfreq(n_points, 1 / spacing)

  return Grid(Tuple(fourier_coordinates))
end
function fftgrid(grid::CuGrid{N,T,M}) where {N,T<:Real,M}
  n_points = size(grid)
  spacing = Array{T}(spacings(grid))
  fourier_coordinates = @. T(2π) * fftfreq(n_points, 1 / spacing)

  return CuGrid(Tuple(fourier_coordinates))
end

"""
    find_grid(data; kwargs...)

Find a grid based on the provided data, which can be a matrix or a vector of vectors.

# Arguments
- `data`: The input data, which can be an `AbstractMatrix` or an `AbstractVector` of vectors.
- `grid_bounds`: Optional bounds for the grid. If not provided, bounds are calculated from the data.
- `grid_dims`: Optional dimensions for the grid. If not provided, defaults to 300 points in each dimension.
- `grid_steps`: Optional spacing steps for the grid to be used instead of dimensions. `grid_dims` takes precedence.
- `grid_padding`: Optional padding for the grid bounds. If not provided, defaults to 10% of the range.
"""
function find_grid(
  data::Union{AbstractVector{<:AbstractVector{<:Real}},AbstractMatrix{<:Real}};
  grid_bounds=nothing,
  grid_dims=nothing,
  grid_steps=nothing,
  grid_padding=nothing,
  device=:cpu,
)
  # Turn data into array
  if isa(data, AbstractMatrix)
    n_dims = size(data, 1)
    data_matrix = data
  elseif isa(data, AbstractVector)
    n_dims = length(data[1])
    data_matrix = reduce(hcat, data)
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
  default_size = 150
  if (grid_steps === nothing) && (grid_dims === nothing)
    grid_dims = ntuple(i -> default_size, Val(n_dims))
  end

  if grid_dims !== nothing
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
