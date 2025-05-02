module DensityEstimations

import ..Devices: Device, IsCPU, IsCUDA
using ..KDEs
using ..Grids
using ..FourierSpace
using ..DirectSpace

using LinearAlgebra,
  StaticArrays,
  FFTW,
  CUDA

export AbstractEstimation,
  estimate!

abstract type AbstractEstimation end

const CPU_SERIAL = :serial
const CPU_THREADED = :threaded
const GPU_CUDA = :cuda

const DEVICE_IMPLEMENTATIONS = Dict{Device,Symbol}(
  IsCPU() => Set([CPU_SERIAL, CPU_THREADED]),
  IsCUDA() => Set([GPU_CUDA]),
)

is_valid_implementation(device::Device, implementation::Symbol)::Bool =
  implementation in get(DEVICE_IMPLEMENTATIONS, typeof(device), Set())

function ensure_valid_implementation(device::Device, implementation::Symbol)::Bool
  if !is_valid_implementation(device, implementation)
    throw(ArgumentError("Invalid implementation $implementation for device $device"))
  end

  return true
end

estimation_lookup = Dict()
function add_estimation!(key::Symbol, value::Type{T})::Nothing where {T<:AbstractEstimation}
  if haskey(estimation_lookup, key)
    throw(ArgumentError("Key $key already exists"))
  end

  estimation_lookup[key] = value

  return nothing
end

# NOTE: This is the User API function for KDE estimations
function estimate!(estimation_name::Symbol, kde::AbstractKDE; kwargs...)::Nothing
  estimation_type = estimation_lookup[estimation_name]
  estimate!(estimation_type, kde; kwargs...)
end
function estimate!(estimation_type::Type{T}, kde::AbstractKDE; kwargs...)::Nothing where {T<:AbstractEstimation}
  device = Device(kde)
  method = get(kwargs, :method, CPU_SERIAL)
  ensure_valid_implementation(device, method)

  estimation = initialize_estimation(estimation_type, kde; kwargs...)
  estimate!(estimation, kde; kwargs...)
  return nothing
end

# NOTE: These function have to be implemented in the estimation modules
function initialize_estimation(
  ::Type{T},
  kde::AbstractKDE
)::AbstractEstimation where {T<:AbstractEstimation}
  throw(ArgumentError("Estimation not implemented"))

end

function estimate!(
  estimation::AbstractEstimation,
  kde::AbstractKDE;
  kwargs...
)::Nothing
  throw(ArgumentError("Estimation not implemented"))

end

include("ParallelEstimation.jl")

end
