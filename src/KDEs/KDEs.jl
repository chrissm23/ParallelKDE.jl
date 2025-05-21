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

Devices.Device(::KDE) = IsCPU()
Devices.Device(::CuKDE) = IsCUDA()

function initialize_kde(
  data::Union{AbstractVector{<:AbstractVector{T}},AbstractMatrix{T}},
  dims::Vararg{Z,N};
  device::Symbol=:cpu,
)::AbstractKDE where {T<:Real,N,Z<:Integer}
  device_type = obtain_device(device)

  return initialize_kde(device_type, data, dims)
end
initialize_kde(data, dims; device=:cpu) = initialize_kde(data, dims...; device)

function initialize_kde(
  ::IsCPU,
  data::Union{AbstractVector{<:AbstractVector{<:Real}},AbstractMatrix{<:Real}},
  dims::NTuple{N}
)::KDE where {N}
  if isa(data, AbstractMatrix)
    if size(data, 1) != N
      throw(ArgumentError("Data must have $N dimensions"))
    end
    data_reinterpreted = Vector{SVector{N,Float64}}(eachcol(data))
  else
    data_reinterpreted = Vector{SVector{N,Float64}}(data)
  end

  density = fill(NaN, dims)
  return KDE(data_reinterpreted, density)
end
function initialize_kde(
  ::IsCUDA,
  data::Union{AbstractVector{<:AbstractVector{<:Real}},AbstractMatrix{<:Real}},
  dims::NTuple{N}
)::CuKDE where {N}
  if isa(data, AbstractMatrix)
    if size(data, 1) != N
      throw(ArgumentError("Data must have $N dimensions"))
    end
    data_reinterpreted = CuMatrix{Float32}(data)
  else
    data_reinterpreted = CuMatrix{Float32}(reduce(hcat, data))
  end

  density = CUDA.fill(NaN32, dims)
  return CuKDE(data_reinterpreted, density)
end

function get_data(kde::AbstractKDE{N,T,S}) where {N,T<:Real,S<:Real}
  return get_data(Device(kde), kde)
end
function get_data(::IsCPU, kde::KDE{N,T,S}) where {N,T<:Real,S<:Real}
  if N == 1
    return reshape(reinterpret(reshape, S, kde.data), 1, :)
  else
    return reinterpret(reshape, S, kde.data)
  end
end
function get_data(::IsCUDA, kde::CuKDE{N,T,S}) where {N,T<:Real,S<:Real}
  return kde.data
end

function get_density(kde::AbstractKDE{N,T,S}) where {N,T<:Real,S<:Real}
  return get_density(Device(kde), kde)
end
function get_density(::IsCPU, kde::KDE{N,T,S}) where {N,T<:Real,S<:Real}
  return kde.density
end
function get_density(::IsCUDA, kde::CuKDE{N,T,S}) where {N,T<:Real,S<:Real}
  return kde.density
end

function set_density!(kde::AbstractKDE{N,T,S}, density::AbstractArray{T,N}) where {N,T<:Real,S<:Real}
  set_density!(Device(kde), kde, density)
  return nothing
end
function set_density!(::IsCPU, kde::KDE{N,T,S}, density::AbstractArray{T,N}) where {N,T<:Real,S<:Real}
  kde.density .= density
  return nothing
end
function set_density!(::IsCUDA, kde::CuKDE{N,T,S}, density::AbstractArray{T,N}) where {N,T<:Real,S<:Real}
  kde.density .= density
  return nothing
end

function set_nan_density!(kde::AbstractKDE{N,T,S}) where {N,T<:Real,S<:Real}
  set_nan_density!(Device(kde), kde)
  return nothing
end
function set_nan_density!(::IsCPU, kde::KDE{N,T,S}) where {N,T<:Real,S<:Real}
  fill!(kde.density, NaN)
  return nothing
end
function set_nan_density!(::IsCUDA, kde::CuKDE{N,T,S}) where {N,T<:Real,S<:Real}
  CUDA.fill!(kde.density, NaN32)
  return nothing
end

function get_nsamples(kde::AbstractKDE{N,T,S}) where {N,T<:Real,S<:Real}
  return get_nsamples(Device(kde), kde)
end
function get_nsamples(::IsCPU, kde::KDE{N,T,S}) where {N,T<:Real,S<:Real}
  return length(kde.data)
end
function get_nsamples(::IsCUDA, kde::CuKDE{N,T,S}) where {N,T<:Real,S<:Real}
  return size(kde.data, 2)
end

function bootstrap_indices(kde::AbstractKDE{N,T,S}, n_bootstraps::Integer) where {N,T<:Real,S<:Real}
  return bootstrap_indices(Device(kde), kde, n_bootstraps)
end
function bootstrap_indices(::IsCPU, kde::KDE{N,T,S}, n_bootstraps::Integer) where {N,T<:Real,S<:Real}
  n_samples = get_nsamples(kde)
  if n_bootstraps == 0
    return reshape(1:n_samples, n_samples, 1)
  elseif n_bootstraps < 0
    throw(ArgumentError("Number of bootstraps must be non-negative"))
  else
    return rand(1:n_samples, n_samples, n_bootstraps)
  end
end
function bootstrap_indices(::IsCUDA, kde::CuKDE{N,T,S}, n_bootstraps::Integer) where {N,T<:Real,S<:Real}
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
