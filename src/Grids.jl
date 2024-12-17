module Grids

using FFTW
using StaticArrays
using CUDA

export AbstractGrid,
  Grid,
  CuGrid

abstract type AbstractGrid{N,T<:Real,M} end

struct Grid{N,T<:Real,M} <: AbstractGrid{N,T,M}
  coordinates::SVector{N,StepRangeLen}
  spacings::SVector{N,T}
  bounds::SMatrix{N,2,T}
end
function Grid(
  ranges::AbstractVector{<:AbstractRange{T}},
)::Grid where {T<:Real}
  N = length(ranges)
  ranges = SVector{N}(ranges)
  spacings = SVector{N}(step.(ranges))
  bounds = SMatrix{N,2,T}(reinterpret(reshape, T, extrema.(ranges)))

  return Grid{N,T,N + 1}(ranges, spacings, bounds)
end
function Grid(ranges::NTuple{N,AbstractRange{T}})::Grid{N,T,N + 1} where {N,T<:Real}
  ranges = SVector{N}(ranges)
  spacings = SVector{N}(step.(ranges))
  bounds = SMatrix{N,2,T}(
    reinterpret(reshape, T, collect(extrema.(ranges)))
  )

  return Grid{N,T,N + 1}(ranges, spacings, bounds)
end

function Base.size(grid::Grid{N,T}) where {N,T<:Real}
  return ntuple(i -> length(grid.coordinates), N)
end
function Base.broadcastable(grid::Grid{N,T}) where {N,T<:Real}
  return reinterpret(
    reshape,
    T,
    collect(Iterators.product(grid.coordinates...))
  )
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

end
