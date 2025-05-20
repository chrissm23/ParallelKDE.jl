module DensityEstimators

using ..Devices
using ..KDEs
using ..Grids
using ..FourierSpace
using ..DirectSpace

using LinearAlgebra,
  StaticArrays,
  FFTW,
  CUDA

export AbstractEstimator,
  estimate!

abstract type AbstractEstimator end

const CPU_SERIAL = :serial
const CPU_THREADED = :threaded
const GPU_CUDA = :cuda

const DEVICE_IMPLEMENTATIONS = Dict{AbstractDevice,Set{Symbol}}(
  IsCPU() => Set([CPU_SERIAL, CPU_THREADED]),
  IsCUDA() => Set([GPU_CUDA]),
  DeviceNotSpecified() => Set()
)

is_valid_implementation(device::AbstractDevice, implementation::Symbol)::Bool =
  implementation in get(DEVICE_IMPLEMENTATIONS, typeof(device), Set())

function ensure_valid_implementation(device::AbstractDevice, implementation::Symbol)::Bool
  if !is_valid_implementation(device, implementation)
    throw(ArgumentError("Invalid implementation $implementation for device $device"))
  end

  return true
end

estimator_lookup = Dict()
function add_estimator!(key::Symbol, value::Type{T})::Nothing where {T<:AbstractEstimator}
  if haskey(estimator_lookup, key)
    throw(ArgumentError("Key $key already exists"))
  end

  estimator_lookup[key] = value

  return nothing
end

# NOTE: This is the User API function for KDE estimators
function estimate!(estimator_name::Symbol, kde::AbstractKDE; kwargs...)::Nothing
  estimator_type = estimator_lookup[estimator_name]
  estimate!(estimator_type, kde; kwargs...)
end
function estimate!(estimator_type::Type{T}, kde::AbstractKDE; kwargs...)::Nothing where {T<:AbstractEstimator}
  device = Device(kde)
  method = get(kwargs, :method, CPU_SERIAL)
  ensure_valid_implementation(device, method)

  estimator = initialize_estimator(estimator_type, kde; kwargs...)
  estimate!(estimator, kde; kwargs...)
  return nothing
end

# NOTE: These functions have to be implemented in the estimator modules
function initialize_estimator(
  ::Type{T},
  kde::AbstractKDE
)::AbstractEstimator where {T<:AbstractEstimator}
  throw(ArgumentError("Estimator not implemented"))

end

function estimate!(
  estimator::AbstractEstimator,
  kde::AbstractKDE;
  kwargs...
)::Nothing
  throw(ArgumentError("Estimator not implemented"))

end

include("ParallelEstimator.jl")

end
