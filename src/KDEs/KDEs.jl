module KDEs

using ..Devices

using StaticArrays
using CUDA

export AbstractKDE,
  KDE,
  CuKDE,
  initialize_kde,
  get_data,
  get_density,
  set_density!,
  set_nan_density!,
  get_nsamples,
  bootstrap_indices

abstract type AbstractKDE{N,T,S} end

struct KDE{N,T<:Real,S<:Real} <: AbstractKDE{N,T,S}
  data::Vector{SVector{N,S}}
  density::Array{T,N}
end

struct CuKDE{N,T<:Real,S<:Real} <: AbstractKDE{N,T,S}
  data::CuMatrix{S}
  density::CuArray{T,N}
end

Devices.get_device(::KDE) = IsCPU()
Devices.get_device(::CuKDE) = IsCUDA()

initialize_kde(data, dims::Vararg{Integer}; device=:cpu) = initialize_kde(data, dims; device)
function initialize_kde(
  data::Union{AbstractVector{<:AbstractVector{T}},AbstractMatrix{T}},
  dims::NTuple{N,Z};
  device::Symbol=:cpu,
) where {T<:Real,N,Z<:Integer}
  device_type = get_device(device)

  return initialize_kde(device_type, data, dims)
end

function initialize_kde(
  ::IsCPU,
  data::Union{AbstractVector{<:AbstractVector{<:Real}},AbstractMatrix{<:Real}},
  dims::NTuple{N};
  T=Float64,
) where {N}
  if isa(data, AbstractMatrix)
    if size(data, 1) != N
      throw(ArgumentError("Data must have $N dimensions"))
    end
    data_reinterpreted = Vector{SVector{N,T}}(eachcol(data))
  else
    data_reinterpreted = Vector{SVector{N,T}}(data)
  end

  density = fill(T(NaN), dims)
  return KDE(data_reinterpreted, density)
end
function initialize_kde(
  ::IsCUDA,
  data::Union{AbstractVector{<:AbstractVector{<:Real}},AbstractMatrix{<:Real}},
  dims::NTuple{N};
  T=Float32,
) where {N}
  if isa(data, AbstractMatrix)
    if size(data, 1) != N
      throw(ArgumentError("Data must have $N dimensions"))
    end
    data_reinterpreted = CuMatrix{T}(data)
  else
    data_reinterpreted = CuMatrix{T}(reduce(hcat, data))
  end

  density = CUDA.fill(T(NaN), dims)
  return CuKDE(data_reinterpreted, density)
end

function get_data(kde::KDE{N,T,S}) where {N,T<:Real,S<:Real}
  if N == 1
    return reshape(reinterpret(reshape, S, kde.data), 1, :)
  else
    return reinterpret(reshape, S, kde.data)
  end
end
get_data(kde::CuKDE) = kde.data

get_density(kde::KDE) = kde.density
get_density(kde::CuKDE) = kde.density

get_nsamples(kde::KDE) = length(kde.data)
get_nsamples(kde::CuKDE) = size(kde.data, 2)

function set_density!(kde::KDE{N,T,<:Real}, density::AbstractArray{T,N}) where {N,T<:Real}
  kde.density .= density
  return nothing
end
function set_density!(kde::CuKDE{N,T,<:Real}, density::AbstractArray{T,N}) where {N,T<:Real}
  kde.density .= density
  return nothing
end

function set_nan_density!(kde::KDE{N,T,<:Real}) where {N,T<:Real}
  fill!(kde.density, T(NaN))
  return nothing
end
function set_nan_density!(kde::CuKDE{N,T,<:Real}) where {N,T<:Real}
  CUDA.fill!(kde.density, T(NaN))
  return nothing
end

function bootstrap_indices(kde::KDE, n_bootstraps::Integer)
  n_samples = get_nsamples(kde)
  if n_bootstraps == 0
    return reshape(1:n_samples, n_samples, 1)
  elseif n_bootstraps < 0
    throw(ArgumentError("Number of bootstraps must be non-negative"))
  else
    return rand(1:n_samples, n_samples, n_bootstraps)
  end
end
function bootstrap_indices(kde::CuKDE, n_bootstraps::Integer)
  n_samples = get_nsamples(kde)
  if n_bootstraps == 0
    return reshape(CuArray{Int32}(1:n_samples), n_samples, 1)
  elseif n_bootstraps < 0
    throw(ArgumentError("Number of bootstraps must be non-negative"))
  else
    return CuArray{Int32}(rand(1:n_samples, n_samples, n_bootstraps))
  end
end

end
