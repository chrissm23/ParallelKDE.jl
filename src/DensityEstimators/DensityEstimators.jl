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

estimator_lookup = Dict()
function add_estimator!(key::Symbol, value::Type{<:AbstractEstimator})
  if haskey(estimator_lookup, key)
    throw(ArgumentError("Key $key already exists"))
  end

  estimator_lookup[key] = value

  return nothing
end

# NOTE: This is the User API function for KDE estimators
function estimate!(estimator_name::Symbol, kde::AbstractKDE; kwargs...)
  estimator_type = estimator_lookup[estimator_name]
  estimate!(estimator_type, kde; kwargs...)

  return nothing
end
function estimate!(estimator_type::Type{<:AbstractEstimator}, kde::AbstractKDE; kwargs...)
  device = get_device(kde)
  method = get(kwargs, :method, CPU_SERIAL)
  ensure_valid_implementation(device, method)

  estimator = initialize_estimator(estimator_type, kde; kwargs...)
  estimate!(estimator, kde; kwargs...)

  return nothing
end

# NOTE: These functions have to be implemented in the estimator modules
function initialize_estimator(
  ::Type{<:AbstractEstimator},
  kde::AbstractKDE;
  kwargs...
)
  throw(ArgumentError("Estimator not implemented"))

end

function estimate!(
  estimator::AbstractEstimator,
  kde::AbstractKDE;
  kwargs...
)
  throw(ArgumentError("Estimator not implemented"))

end

include("ParallelEstimator.jl")

end
