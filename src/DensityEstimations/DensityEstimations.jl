module DensityEstimations

import ..Devices: Device, IsCPU, IsCUDA
using ..KDEs
using ..Grids
using ..FourierSpace
using ..DirectSpace

using StaticArrays,
  FFTW,
  CUDA

export AbstractEstimation,
  estimate!

abstract type AbstractEstimation end

method_lookup = Dict()
function add_method!(key::Symbol, value::Type{T})::Nothing where {T<:AbstractEstimation}
  if haskey(method_lookup, key)
    throw(ArgumentError("Key $key already exists"))
  end

  method_lookup[key] = value

  return nothing
end

function initialize_estimation(
  ::Type{T},
  kde::AbstractKDE
)::AbstractEstimation where {T<:AbstractEstimation}
  throw(ArgumentError("Estimation not implemented"))
end

function estimate!(::Type{<:AbstractEstimation}, kde::AbstractKDE; kwargs...)::Nothing
  estimation = initialize_estimation(AbstractEstimation, kde; kwargs...)
  estimate!(kde, estimation)
  return nothing
end
function estimate!(kde::AbstractKDE, method::Symbol; kwargs...)::Nothing
  method_type = method_lookup[method]
  estimate!(method_type, kde; kwargs...)
end

include("ParallelEstimation.jl")

end
