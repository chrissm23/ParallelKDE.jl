module KDEs

using ..Grids

using StaticArrays,
  CUDA

export AbstractKDE,
  KDE,
  CuKDE,
  DeviceKDE,
  IsCPUKDE,
  IsGPUKDE,
  DeviceKDE,
  converged_kde,
  set_density!,
  set_nan_density!,
  get_nsamples

abstract type AbstractKDE{N} end

struct KDE{N,T<:Real,S<:Real,M} <: AbstractKDE{N,T,S,M}
  data::Vector{SVector{N,S}}
  grid::Grid{N,S,M}
  t::SVector{N,Float64}
  density::Array{T,N}
end

struct CuKDE{N,T<:Real,S<:Real,M} <: AbstractKDE{N,T,S,M}
  data::CuArray{S,2}
  grid::CuGrid{N,S,M}
  t::CuArray{Float32,1}
  density::CuArray{T,N}
end

abstract type DeviceKDE end
struct IsCPUKDE <: DeviceKDE end
struct IsGPUKDE <: DeviceKDE end
DeviceKDE(::KDE) = IsCPUKDE()
DeviceKDE(::CuKDE) = IsGPUKDE()

function converged_kde(kde::AbstractKDE{N,T,S,M}) where {N,T,S,M}
  return converged_kde(DeviceKDE(kde), kde)
end
function converged_kde(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T,S,M}
  return all(isfinite, kde.density)
end
function converged_kde(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T,S,M}
  return all(isfinite, kde.density)
end

function set_density!(kde::AbstractKDE{N,T,S,M}, density::Array{T,N}) where {N,T,S,M}
  set_density!(DeviceKDE(kde), kde, density)
end
function set_density!(::IsCPUKDE, kde::KDE{N,T,S,M}, density::Array{T,N}) where {N,T,S,M}
  kde.density .= density
end
function set_density!(::IsGPUKDE, kde::CuKDE{N,T,S,M}, density::CuArray{T,N}) where {N,T,S,M}
  kde.density .= density
end

function set_nan_density!(kde::AbstractKDE{N,T,S,M}) where {N,T,S,M}
  set_nan_density!(DeviceKDE(kde), kde)
end
function set_nan_density!(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T,S,M}
  fill!(kde.density, NaN)
end
function set_nan_density!(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T,S,M}
  CUDA.fill!(kde.density, NaN32)
end

function get_nsamples(kde::AbstractKDE{N,T,S,M}) where {N,T,S,M}
  return get_nsamples(DeviceKDE(kde), kde)
end
function get_nsamples(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T,S,M}
  return length(kde.data)
end
function get_nsamples(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T,S,M}
  return size(kde.data, 2)
end

end
