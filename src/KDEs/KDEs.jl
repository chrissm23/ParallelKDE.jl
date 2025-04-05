module KDEs

import ..Devices: Device, IsCPU, IsCUDA

using StaticArrays
using CUDA

# TODO: Remove t, get_time, and converged_kde
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

abstract type AbstractKDE{N,T,S,M} end

struct KDE{N,T<:Real,S<:Real,M} <: AbstractKDE{N,T,S,M}
  data::Vector{SVector{N,S}}
  density::Array{T,N}
end

struct CuKDE{N,T<:Real,S<:Real,M} <: AbstractKDE{N,T,S,M}
  data::CuMatrix{S}
  density::CuArray{T,N}
end

Device(::KDE) = IsCPU()
Device(::CuKDE) = IsCUDA()

function initialize_kde(
  data::Union{AbstractVector{<:AbstractVector{<:Real}},AbstractMatrix{<:Real}},
  dims::NTuple{N};
  device::Symbol=:cpu
)::AbstractKDE where {N}
  if device == :cpu
    initialize_kde(IsCPU(), data, dims)
  elseif device == :gpu
    initialize_kde(IsCUDA(), data, dims)
  else
    throw(ArgumentError("Invalid device: $device"))
  end
end

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

  density = CUDA.zeros(Float64, dims)
  return CuKDE(data_reinterpreted, density)
end
function initialize_kde(
  ::IsCUDA,
  data::Union{AbstractVector{<:AbstractVector{<:Real}},AbstractMatrix{<:Real}},
  dims::NTuple{N}
)::KDE where {N}
  if isa(data, AbstractMatrix)
    if size(data, 1) != N
      throw(ArgumentError("Data must have $N dimensions"))
    end
    data_reinterpreted = CuMatrix{Float32}(data)
  else
    data_reinterpreted = CuMatrix{Float32}(reduce(hcat, data))
  end

  density = zeros(Float64, dims)
  return KDE(data_reinterpreted, density)
end

function get_data(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return get_data(Device(kde), kde)
end
function get_data(::IsCPU, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  if N == 1
    return reshape(reinterpret(reshape, S, kde.data), 1, :)
  else
    return reinterpret(reshape, S, kde.data)
  end
end
function get_data(::IsCUDA, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.data
end

function get_density(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return get_density(Device(kde), kde)
end
function get_density(::IsCPU, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.density
end
function get_density(::IsCUDA, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return kde.density
end

function set_density!(kde::AbstractKDE{N,T,S,M}, density::Array{T,N}) where {N,T<:Real,S<:Real,M}
  set_density!(Device(kde), kde, density)
  return nothing
end
function set_density!(::IsCPU, kde::KDE{N,T,S,M}, density::Array{T,N}) where {N,T<:Real,S<:Real,M}
  kde.density .= density
  return nothing
end
function set_density!(::IsCUDA, kde::CuKDE{N,T,S,M}, density::AnyCuArray{T,N}) where {N,T<:Real,S<:Real,M}
  kde.density .= density
  return nothing
end

function set_nan_density!(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  set_nan_density!(Device(kde), kde)
  return nothing
end
function set_nan_density!(::IsCPU, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  fill!(kde.density, NaN)
  return nothing
end
function set_nan_density!(::IsCUDA, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  CUDA.fill!(kde.density, NaN32)
  return nothing
end

function get_nsamples(kde::AbstractKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return get_nsamples(Device(kde), kde)
end
function get_nsamples(::IsCPU, kde::KDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return length(kde.data)
end
function get_nsamples(::IsCUDA, kde::CuKDE{N,T,S,M}) where {N,T<:Real,S<:Real,M}
  return size(kde.data, 2)
end

function bootstrap_indices(kde::AbstractKDE{N,T,S,M}, n_bootstraps::Integer) where {N,T<:Real,S<:Real,M}
  return bootstrap_indices(Device(kde), kde, n_bootstraps)
end
function bootstrap_indices(::IsCPU, kde::KDE{N,T,S,M}, n_bootstraps::Integer) where {N,T<:Real,S<:Real,M}
  n_samples = get_nsamples(kde)
  if n_bootstraps == 0
    return reshape(1:n_samples, n_samples, 1)
  elseif n_bootstraps < 0
    throw(ArgumentError("Number of bootstraps must be non-negative"))
  else
    return rand(1:n_samples, n_samples, n_bootstraps)
  end
end
function bootstrap_indices(::IsCUDA, kde::CuKDE{N,T,S,M}, n_bootstraps::Integer) where {N,T<:Real,S<:Real,M}
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
