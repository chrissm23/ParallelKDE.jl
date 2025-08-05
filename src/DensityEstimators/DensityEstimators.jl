module DensityEstimators

using ..Devices
using ..KDEs
using ..Grids
using ..FourierSpace
using ..DirectSpace

using Statistics,
  LinearAlgebra,
  StaticArrays,
  FFTW,
  CUDA

export AbstractEstimator,
  estimate!

"""
    AbstractEstimator

Supertype for all density estimation estimators.

This is the base for all object that are intended to provide a method for estimating the
density. They store all the necessary parameters and data for the estimation process.

See also [`AbstractDensityEstimation`](@ref) for the base type where the estimated density
is stored.
"""
abstract type AbstractEstimator end

estimator_lookup = Dict()
"""
    add_estimator!(key::Symbol, value::Type{<:AbstractEstimator})

Include a new estimator type into the lookup table.
"""
function add_estimator!(key::Symbol, value::Type{<:AbstractEstimator})
  if haskey(estimator_lookup, key)
    throw(ArgumentError("Key $key already exists"))
  end

  estimator_lookup[key] = value

  return nothing
end

# NOTE: This is the User API function for KDE estimators
"""
    estimate!(estimator_name::Symbol, kde::AbstractKDE; kwargs...)

Estimate the density of an `AbstractKDE` object using the specified estimator type.

The `estimator_name` should be a symbol that corresponds to the key in the `estimator_lookup` dictionary.
Therefore, this is the method that the `ParallelKDE.jl` API uses to estimate the density.

See [`add_estimator!`](@ref) for how to add a new estimator type to the lookup table.
"""
function estimate!(estimator_name::Symbol, kde::AbstractKDE; kwargs...)
  estimator_type = estimator_lookup[estimator_name]
  estimate!(estimator_type, kde; kwargs...)

  return nothing
end
function estimate!(
  estimator_type::Type{<:AbstractEstimator},
  kde::AbstractKDE;
  method::Union{Nothing,Symbol}=nothing,
  kwargs...
)
  device = get_device(kde)
  if method === nothing
    method = Devices.DEFAULT_IMPLEMENTATIONS[device]
  end
  ensure_valid_implementation(device, method)

  if device isa IsCUDA
    (_, kwargs) = convert32(; kwargs...)
  end

  estimator = initialize_estimator(estimator_type, kde; method, kwargs...)
  estimate!(estimator, kde; method, kwargs...)

  return nothing
end

# NOTE: These functions have to be implemented in the estimator modules
"""
    initialize_estimator(::Type{<:AbstractEstimator}, kde::AbstractKDE; kwargs...)

Initialize an instance of the given estimator type with the provided KDE object and keyword arguments.

Each estimator type should implement this method to set up its internal state based on the KDE data.
"""
function initialize_estimator(
  ::Type{<:AbstractEstimator},
  kde::AbstractKDE;
  kwargs...
)
  throw(ArgumentError("Estimator not implemented"))

end

"""
    estimate!(estimator_type::AbstractEstimator, kde::AbstractKDE; kwargs...)

Defined this way is how each estimator type should implement the estimation process.
"""
function estimate!(
  estimator::AbstractEstimator,
  kde::AbstractKDE;
  kwargs...
)
  throw(ArgumentError("Estimator not implemented"))

end

include("ParallelEstimator.jl")
include("RoTEstimator.jl")

end
