module ParallelKDE

include("Grids.jl")
include("KDEs.jl")
include("DirectSpace.jl")
include("FourierSpace.jl")

using .Grids
using .KDEs
using .DirectSpace
using .FourierSpace

using StaticArrays,
  Statistics,
  FFTW,
  CUDA

export initialize_kde,
  fit_kde!

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
  data::AbstractVector{<:AbstractVector{T}},
  grid_ranges::AbstractVector{<:AbstractRange{S}},
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
  ignore_t0::Bool=true,
  method::Symbol=:serial
) where {N,T<:Real,S<:Real,M}
  n_bootstraps = get_nbootstraps(n_bootstraps)
  n_samples = get_nsamples(kde)
  threshold = get_smoothness(smoothness, n_samples, N)
  t0 = kde.t
  time_range = get_times(dt, n_steps, t_final, t0, ignore_t0)

  if ignore_t0
    set_nan_density!(kde)
  end

  if method == :serial || method == :threaded
    density = find_density(DeviceKDE(kde), kde, time_range, n_bootstraps, threshold, method=method)
  elseif method == :cuda
    density = find_density(DeviceKDE(kde), kde, time_range, n_bootstraps, threshold)
  else
    throw(ArgumentError("Invalid method: $method"))
  end

  set_density!(kde, density)
end

function find_density(
  ::IsCPUKDE,
  kde::KDE{N,T,S,M},
  time_range::Vector{<:AbstractRange{<:Real}},
  n_bootstraps::Int,
  threshold::Real;
  method::Symbol=:serial
)::Array{T,N} where {N,T<:Real,S<:Real,M}
  density = fill(NaN, size(kde.grid))

  means_0, variances_0 = initialize_statistics(kde, n_bootstraps, method)
  complete_distribution = initialize_distribution(kde.data, kde.grid, method)
  fourier_grid = fft_grid(kde.grid)

  times = reinterpret(reshape, T, collect(zip(time_range...)))

  for time in eachcol(times)
    means_t, variances_t = propagate_bandwidth(means_0, variances_0, fourier_grid, time, method)
    means, variances = calculate_statistics(means_t, variances_t, method)
    assign_density!(density, complete_distribution, means, variances, threshold, method)

    if all(isfinite, density)
      kde.t .= time
      break
    end

    if time === times[end]
      @warn "Not all points converged with the specified time ranges."
    end
  end

  return density
end
function find_denisty(
  ::IsGPUKDE,
  kde::CuKDE{N,T,S,M},
  time_range::Vector{<:AbstractRange{<:Real}},
  n_bootstraps::Int,
  threshold::Real;
)::CuArray{T,N} where {N,T<:Real,S<:Real,M}
  density = CUDA.fill(NaN32, size(kde.grid))

  means_0, variances_0 = initialize_statistics(kde, n_bootstraps)
  complete_distribution = initialize_distribution(kde.data, kde.grid, method)
  fourier_grid = fft_grid(kde.grid)

  times = CuArray{T,N}(
    reinterpret(reshape, T, collect(zip(time_range...)))
  )

  for col in 1:size(times, 2)
    time = view(times, :, col)

    means_t, variances_t = propagate_bandwidth(means_0, variances_0, fourier_grid, time, method)
    means, variances = calculate_statistics(means_t, variances_t, method)

    assign_density!(density, complete_distribution, means, variances, threshold, method)

    if all(isfinite, density)
      kde.t .= time
      break
    end

    if col == size(times, 2)
      @warn "Not all points converged with the specified time ranges."
    end
  end

  return density
end

function initialize_statistics(
  kde::KDE{N,T,S,M},
  n_bootstraps::Int,
  method::Symbol
)::NTuple{2,Array{Complex{T},N + 1}} where {N,T<:Real,S<:Real,M}
  n_samples = length(kde.data)
  dirac_series = initialize_dirac_series(Val(method), kde, n_bootstraps)

  s_0 = fftshift(fft(dirac_series, 2:N+1), 2:N+1)
  means_0 = s_0 ./ n_samples
  variances_0 = (abs2.(s_0) ./ n_samples^2) .- (means_0 .^ 2)

  return means_0, variances_0
end

function initialize_statistics(
  kde::CuKDE{N,T,S,M},
  n_bootstraps::Int,
)::NTuple{2,CuArray{Complex{T},N + 1}} where {N,T<:Real,S<:Real,M}
  n_samples = size(kde.data, 2)
  dirac_series = initialize_dirac_series(Val(:cuda), kde, n_bootstraps)

  s_0 = fftshift(fft(dirac_series, 2:N+1), 2:N+1)
  means_0 = s_0 ./ n_samples
  variances_0 = (abs2.(s_0) ./ n_samples^2) .- means_0 .^ 2

  return means_0, variances_0
end

function get_times(
  dt::Union{Real,Vector{<:Real},Nothing},
  n_steps::Union{Int,Nothing},
  t_final::Union{Real,Vector{<:Real},Nothing},
  t0::AbstractVector{<:Real},
  ignore_t0::Bool
)::Vector{<:AbstractRange{<:Real}}
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

function get_available_memory(::IsCPUKDE)
  return Sys.free_memory() / 1024^2
end
function get_available_memory(::IsGPUKDE)
  if !CUDA.functional()
    throw(ArgumentError("CUDA.jl is not functional. Use a different method."))
  end
  return CUDA.memory_status().free / 1024^2
end

end
