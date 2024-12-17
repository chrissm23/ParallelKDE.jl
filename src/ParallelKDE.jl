module ParallelKDE

include("Grids.jl")
include("KDEs.jl")

using .Grids
using .KDEs

using StaticArrays,
  FFTW,
  CUDA

function initialize_kde(
  data::AbstractVector{<:AbstractVector{<:Real}},
  grid::AbstractVector{<:AbstractRange{<:Real}},
  device::Symbol
)
  if device == :cpu
    return initialize_kde(data, grid, IsCPUKDE())
  elseif device == :gpu
    @assert CUDA.functional() "CUDA.jl is not functional. Use a different method."
    return initialize_kde(data, grid, IsGPUKDE())
  else
    throw(ArgumentError("Invalid device: $device"))
  end
end

function initialize_kde(
  data::AbstractVector{<:AbstractVector{T}},
  grid_ranges::AbstractVector{<:AbstractRange{S}},
  ::IsCPUKDE
)::KDE where {T<:Real,S<:Real}
  N = length(data[1])

  grid = Grid(grid_ranges)
  density = fill(NaN, size(grid))

  # TODO:
  # Calculate the actual initial bandwidth given by the Dirac series
  t = @SVector zeros(N)

  kde = KDE{N,Float64,T,N + 1}(Vector{SVector{N,T}}(data), grid, t, density)

  return kde
end

function initialize_kde(
  data::AbstractVector{<:AbstractVector{T<:Real}},
  grid_ranges::AbstractVector{<:AbstractRange{S<:Real}},
  ::IsGPUKDE
)::CuKDE where {T<:Real,S<:Real}
  N = length(data[1])

  grid = CuGrid(grid_ranges, b32=true)
  density = CUDA.fill(NaN32, size(grid))

  # TODO:
  # Calculcate the actual initial bandwidth given by the Dirac series
  t = CUDA.zeros(Float32, N)

  rearanged_data = reduce(hcat, data)

  kde = CuKDE{N,Float32,Float32,N + 1}(CuArray{Float32}(rearanged_data), grid, t, density)

  return kde
end

function fit_kde!(
  kde::AbstractKDE{N,T,S,M};
  dt::Union{Real,Vector{<:Real},Nothing}=nothing,
  n_steps::Union{Int,Nothing}=nothing,
  t_final::Union{Real,Vector{<:Real},Nothing}=nothing,
  n_bootstraps::Union{Int,Nothing}=nothing,
  smoothness::Real=1.0,
  performance::Symbol=:low,
  ignore_t0::Bool=true
) where {N,T,S,M}
  n_bootstraps = get_nbootstraps(n_bootstraps)
  n_samples = get_nsamples(kde)
  threshold = get_smoothness(smoothness, n_samples, N)
  t0 = kde.t
  time_range = get_times(dt, n_steps, t_final, t0, ignore_t0)

  if ignore_t0
    set_nan_density!(kde)
  end

  density = find_density(DeviceKDE(kde), kde, time_range, n_bootstraps, threshold, performance)

  set_density!(kde, density)
end

function get_times(
  dt::Union{Real,Vector{<:Real},Nothing},
  n_steps::Union{Int,Nothing},
  t_final::Union{Real,Vector{<:Real},Nothing},
  t0::AbstractVector{<:Real},
  ignore_t0::Bool
)
  N = length(t0)

  if ignore_t0
    t0 = @SVector zeros(N)
  elseif isa(t0, CuArray)
    t0 = Array(t0)
  end

  if dt !== nothing && n_steps !== nothing && t_final !== nothing
    throw(ArgumentError("Only two of dt, n_steps, or tf can be provided."))
  elseif dt !== nothing && n_steps !== nothing
    time_range = range.(t0, step=dt, length=n_steps)
  elseif dt !== nothing && t_final !== nothing
    time_range = range.(t0, t_final, step=dt)
  elseif t_final !== nothing && n_steps !== nothing
    time_range = range.(t0, t_final, length=n_steps)
  else
    throw(ArgumentError("Two of dt, n_steps, or tf must be provided."))
  end
end

function get_nbootstraps(n_bootstraps::Union{Int,Nothing})
  if n_bootstraps === nothing
    n_bootstraps = 100
  end

  return n_bootstraps
end

function get_smoothness(smoothness::Real, n_samples::Int, n_dims::Int)
  if smoothness === nothing
    smoothness = 1.0
  elseif smoothness < 0.0
    throw(ArgumentError("Smoothness must be positive."))
  end
  threshold = smoothness * optimal_variance(n_samples, n_dims)

  return threshold
end

end
