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
  get_data,
  get_grid,
  get_time,
  get_density,
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

function get_data(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return get_data(DeviceKDE(kde), kde)
end
function get_data(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  if N == 1
    return reshape(reinterpret(reshape, S, kde.data), 1, :)
  else
    return reinterpret(reshape, S, kde.data)
  end
end
function get_data(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.data
end

function get_grid(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return get_grid(DeviceKDE(kde), kde)
end
function get_grid(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.grid
end
function get_grid(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.grid
end

function get_time(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return get_time(DeviceKDE(kde), kde)
end
function get_time(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.t
end
function get_time(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.t
end

function get_density(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return get_density(DeviceKDE(kde), kde)
end
function get_density(::IsCPUKDE, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.density
end
function get_density(::IsGPUKDE, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.density
end

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

function bootstrap_indices(kde::AbstractKDE{N,T,S,M}, n_bootstraps::Integer) where {N,T<:Real,S<:Real,M}
  return bootstrap_indices(DeviceKDE(kde), kde, n_bootstraps)
end
function bootstrap_indices(::IsCPUKDE, kde::KDE{N,T,S,M}, n_bootstraps::Integer) where {N,T<:Real,S<:Real,M}
  n_samples = get_nsamples(kde)
  if n_bootstraps == 1
    return reshape(1:n_samples, n_samples, 1)
  else
    return rand(1:n_samples, n_samples, n_bootstraps)
  end
end
function bootstrap_indices(::IsGPUKDE, kde::CuKDE{N,T,S,M}, n_bootstraps::Integer) where {N,T<:Real,S<:Real,M}
  n_samples = get_nsamples(kde)
  if n_bootstraps == 1
    return reshape(CuArray{Int32}(1:n_samples), n_samples, 1)
  else
    return CuArray{Int32}(rand(1:n_samples, n_samples, n_bootstraps))
  end
end

end
