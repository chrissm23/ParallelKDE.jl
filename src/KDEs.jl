module KDEs

using ..Grids

using StaticArrays
using CUDA

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
  get_nsamples,
  bootstrap_indices

abstract type AbstractKDE{N,T,S,M} end

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

function converged_kde(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return converged_kde(DeviceKDE(kde), kde)
end
function converged_kde(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return all(isfinite, kde.density)
end
function converged_kde(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return all(isfinite, kde.density)
end

function set_density!(kde::AbstractKDE{N,T,S,M}, density::Array{T,N}) where {N,T<:Real,S<:Real,M}
  set_density!(DeviceKDE(kde), kde, density)
  return nothing
end
function set_density!(::IsCPUKDE, kde::KDE{N,T,S,M}, density::Array{T,N}) where {N,T<:Real,S<:Real,M}
  kde.density .= density
  return nothing
end
function set_density!(::IsGPUKDE, kde::CuKDE{N,T,S,M}, density::CuArray{T,N}) where {N,T<:Real,S<:Real,M}
  kde.density .= density
  return nothing
end

function set_nan_density!(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  set_nan_density!(DeviceKDE(kde), kde)
  return nothing
end
function set_nan_density!(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  fill!(kde.density, NaN)
  return nothing
end
function set_nan_density!(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  CUDA.fill!(kde.density, NaN32)
  return nothing
end

function get_nsamples(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return get_nsamples(DeviceKDE(kde), kde)
end
function get_nsamples(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return length(kde.data)
end
function get_nsamples(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return size(kde.data, 2)
end

function bootstrap_indices(kde::AbstractKDE{N,T,S,M}, n_bootstraps::Int) where {N,T<:Real,S<:Real,M}
  return bootstrap_indices(DeviceKDE(kde), kde, n_bootstraps)
end
function bootstrap_indices(::IsCPUKDE, kde::KDE{N,T,S,M}, n_bootstraps::Int) where {N,T<:Real,S<:Real,M}
  n_samples = get_nsamples(kde)
  return rand(1:n_samples, n_samples, n_bootstraps)
end
function bootstrap_indices(::IsGPUKDE, kde::CuKDE{N,T,S,M}, n_bootstraps::Int) where {N,T<:Real,S<:Real,M}
  n_samples = get_nsamples(kde)
  return CuArray{Int32}(rand(1:n_samples, n_samples, n_bootstraps))
end

end
