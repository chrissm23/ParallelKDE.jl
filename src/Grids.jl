module Grids

using FFTW
using StaticArrays
using CUDA

abstract type AbstractGrid{T<:Real,N} end

struct Grid{T<:Real,N} <: AbstractGrid{T,N}
  coordinates::SVector{N,StepRangeLen}
  spacings::SVector{N,T}
  bounds::SMatrix{N,2,T,2N}
end
function Grid(
  coordinates::AbstractVector{<:AbstractRange{T}},
) where {T<:Real}
  N = length(coordinates)
  coordinates = SVector{N}(coordinates)
  spacings = SVector{N}(step.(coordinates))
  bounds = SMatrix{N,2,T,2N}(
    reinterpret(reshape, T, extrema.(coordinates))
  )

  return Grid(coordinates, spacings, bounds)
end
function Grid(coordinates::NTuple{N,AbstractRange{T}}) where {T<:Real,N}
  coordinates = SVector{N}(coordinates)
  spacings = SVector{N}(step.(coordinates))
  bounds = SMatrix{N,2,T,2N}(
    reinterpret(reshape, T, collect(extrema.(coordinates)))
  )

  return Grid(coordinates, spacings, bounds)
end

end
