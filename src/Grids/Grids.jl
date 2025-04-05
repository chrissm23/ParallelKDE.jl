module Grids

import ..Devices: Device, IsCPU, IsCUDA

using FFTW
using StaticArrays
using CUDA

export AbstractGrid,
  Grid,
  CuGrid,
  spacings,
  bounds,
  low_bounds,
  high_bounds,
  initial_bandwidth,
  get_coordinates,
  fftgrid

abstract type AbstractGrid{N,T<:Real,M} end

struct Grid{N,T<:Real,M} <: AbstractGrid{N,T,M}
  coordinates::SVector{N,StepRangeLen}
  spacings::SVector{N,T}
  bounds::SMatrix{2,N,T}
end
function Grid(
  ranges::AbstractVector{<:Union{AbstractRange{T},Frequencies{T}}},
)::Grid where {T<:Real}
  N = length(ranges)
  ranges = SVector{N}(ranges)
  spacings = SVector{N}(step.(ranges))
  bounds = SMatrix{2,N,T}(
    reinterpret(reshape, T, extrema.(ranges))
  )

  return Grid{N,T,N + 1}(ranges, spacings, bounds)
end
function Grid(
  ranges::NTuple{N,Union{AbstractRange{T},Frequencies{T}}}
)::Grid{N,T,N + 1} where {N,T<:Real}
  ranges = SVector{N}(ranges)
  spacings = SVector{N}(step.(ranges))
  bounds = SMatrix{2,N,T}(
    reinterpret(reshape, T, collect(extrema.(ranges)))
  )

  return Grid{N,T,N + 1}(ranges, spacings, bounds)
end

function Base.size(grid::Grid{N,T}) where {N,T<:Real}
  return ntuple(i -> length(grid.coordinates[i]), N)
end

function get_coordinates(grid::Grid{N,T}) where {N,T<:Real}
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

function Base.broadcastable(grid::Grid{N,T}) where {N,T<:Real}
  return get_coordinates(grid)
end

struct CuGrid{N,T<:Real,M} <: AbstractGrid{N,T,M}
  coordinates::CuArray{T,M}
  spacings::CuArray{T,1}
  bounds::CuArray{T,2}
end

function CuGrid(
  ranges::Union{
    AbstractVector{<:Union{AbstractRange{T},Frequencies{T}}},
    NTuple{N,Union{AbstractRange{T},Frequencies{T}}}
  };
  b32::Bool=true
)::CuGrid where {N,T<:Real}
  if isa(ranges, NTuple)
    ranges = collect(ranges)
    n = N
  else
    n = length(ranges)
  end

  complete_array = reinterpret(
    reshape,
    T,
    collect(Iterators.product(ranges...))
  )
  if n == 1
    complete_array = reshape(complete_array, 1, :)
  end

  if b32
    coordinates = CuArray{Float32}(complete_array)
    spacings = CuArray{Float32}(step.(ranges))
    bounds = CuArray{Float32}(
      reinterpret(reshape, T, extrema.(ranges))
    )

    return CuGrid{n,Float32,n + 1}(coordinates, spacings, bounds)
  else
    coordinates = CuArray{T}(complete_array)
    spacings = CuArray{T}(step.(ranges))
    bounds = CuArray{T}(
      reinterpret(reshape, T, extrema.(ranges))
    )

    return CuGrid{n,T,n + 1}(coordinates, spacings, bounds)
  end
end

function Base.size(grid::CuGrid{N,T}) where {N,T<:Real}
  return size(grid.coordinates)[2:end]
end

function get_coordinates(grid::CuGrid{N,T}) where {N,T<:Real}
  return grid.coordinates
end

function Base.broadcastable(grid::CuGrid{N,T}) where {N,T<:Real}
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
