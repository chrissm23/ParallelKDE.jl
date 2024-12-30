module Grids

using FFTW
using StaticArrays
using CUDA

export AbstractGrid,
  Grid,
  CuGrid,
  spacings,
  bounds,
  low_bounds,
  high_bounds

abstract type AbstractGrid{N,T<:Real,M} end

struct Grid{N,T<:Real,M} <: AbstractGrid{N,T,M}
  coordinates::SVector{N,StepRangeLen}
  spacings::SVector{N,T}
  bounds::SMatrix{2,N,T}
end
function Grid(
  ranges::AbstractVector{<:AbstractRange{T}},
)::Grid where {T<:Real}
  N = length(ranges)
  ranges = SVector{N}(ranges)
  spacings = SVector{N}(step.(ranges))
  bounds = SMatrix{2,N,T}(
    reinterpret(reshape, T, extrema.(ranges))
  )

  return Grid{N,T,N + 1}(ranges, spacings, bounds)
end
function Grid(ranges::NTuple{N,AbstractRange{T}})::Grid{N,T,N + 1} where {N,T<:Real}
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
function Base.broadcastable(grid::Grid{N,T}) where {N,T<:Real}
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

struct CuGrid{N,T<:Real,M} <: AbstractGrid{N,T,M}
  coordinates::CuArray{T,M}
  spacings::CuArray{T,1}
  bounds::CuArray{T,2}
end

function CuGrid(
  ranges::Union{AbstractVector{<:AbstractRange{T}},NTuple{N,AbstractRange{T}}};
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

function Base.broadcastable(grid::CuGrid{N,T}) where {N,T<:Real}
  return grid.coordinates
end

abstract type DeviceGrid end
struct IsCPUGrid <: DeviceGrid end
struct IsGPUGrid <: DeviceGrid end
DeviceGrid(grid::Grid) = IsCPUGrid()
DeviceGrid(grid::CuGrid) = IsGPUGrid()

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

end
