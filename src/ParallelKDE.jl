module ParallelKDE

include("Grids.jl")
include("KDEs.jl")
include("FourierSpace.jl")
include("DirectSpace.jl")

using .Grids
using .KDEs
using .FourierSpace
using .DirectSpace

using StaticArrays,
  FFTW,
  CUDA

using Statistics

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

  t = initial_bandwidth(grid)

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

  t = initial_bandwidth(grid)

  rearanged_data = reduce(hcat, data)

  kde = CuKDE{N,Float32,Float32,N + 1}(CuArray{Float32}(rearanged_data), grid, t, density)

  return kde
end

# TODO: Change the threshold_factor into smoothness once the unknown factor is known
function fit_kde!(
  kde::AbstractKDE{N,T,S,M};
  dt::Union{Real,Vector{<:Real},Nothing}=nothing,
  n_steps::Union{Int,Nothing}=nothing,
  t_final::Union{Real,Vector{<:Real},Nothing}=nothing,
  n_bootstraps::Union{Int,Nothing}=nothing,
  smoothness::Real=1 / 150,
  ignore_t0::Bool=true,
  method::Symbol=:serial
) where {N,T<:Real,S<:Real,M}
  n_bootstraps = get_nbootstraps(n_bootstraps)
  n_samples = get_nsamples(kde)

  threshold = get_smoothness(smoothness, n_samples, N)

  t0 = kde.t
  time_range = get_times(dt, n_steps, t_final, t0, ignore_t0, kde.grid)

  if ignore_t0
    set_nan_density!(kde)
  end

  if method == :serial || method == :threaded
    find_density!(DeviceKDE(kde), kde, time_range, n_bootstraps, threshold, method=method)
  elseif method == :cuda
    find_density!(DeviceKDE(kde), kde, time_range, n_bootstraps, threshold)
  else
    throw(ArgumentError("Invalid method: $method"))
  end

end

function find_density!(
  ::IsCPUKDE,
  kde::KDE{N,T,S,M},
  time_range::Vector{<:AbstractRange{<:Real}},
  n_bootstraps::Integer,
  threshold::Real;
  method::Symbol=:serial
)::Array{T,N} where {N,T<:Real,S<:Real,M}
  means_0, variances_0 = initialize_statistics(kde, n_bootstraps, method)
  ifft_plan_multi = plan_ifft!(means_0, dims=2:N+1)

  density_0, var_0 = initialize_distribution(kde, method)
  ifft_plan_single = plan_ifft!(density_0, 1:N)

  fourier_grid = fftgrid(kde.grid)
  fourier_grid_array = get_coordinates(fourier_grid)

  times = reinterpret(reshape, T, collect(zip(time_range...)))
  time_initial = initial_bandwidth(kde.grid)

  means_dst = Array{Complex{T},N + 1}(undef, size(means_0))
  variances_dst = Array{Complex{T},N + 1}(undef, size(variances_0))

  for time in eachcol(times)
    means_bootstraps, variances_bootstraps = propagate_bandwidth!(
      means_0,
      variances_0,
      fourier_grid_array,
      time,
      time_initial,
      method,
      dst_mean=means_dst,
      dst_var=variances_dst,
    )

    vmr_variance = calculate_statistics!(
      means_bootstraps,
      variances_bootstraps,
      ifft_plan_multi,
      method,
      dst_vmr=selectdim(variances_dst, 1, 1)
    )

    means_complete, variances_complete = propagate_bandwidth!(
      reshape(density_0, 1, size(density_0)...),
      reshape(var_0, 1, size(var_0)...),
      fourier_grid_array,
      time,
      time_initial,
      method,
      dst_mean=reshape(selectdim(means_dst, 1, 1), 1, size(means_dst)[2:end]...),
      dst_var=reshape(selectdim(means_dst, 1, 2), 1, size(means_dst)[2:end]...),
    )
    means_complete = dropdims(means_complete, dims=1)
    variances_complete = dropdims(variances_complete, dims=1)

    assign_density!(
      kde,
      means_complete,
      variances_complete,
      vmr_variance,
      time,
      threshold,
      ifft_plan_single,
      method
    )

    if all(isfinite, kde.density)
      kde.t .= time
      break
    end

    if time === times[end]
      @warn "Not all points converged with the specified time ranges."
    end
  end

  return nothing
end
function find_denisty!(
  ::IsGPUKDE,
  kde::CuKDE{N,T,S,M},
  time_range::Vector{<:AbstractRange{<:Real}},
  n_bootstraps::Integer,
  threshold::Real;
)::CuArray{T,N} where {N,T<:Real,S<:Real,M}
  means_0, variances_0 = initialize_statistics(kde, n_bootstraps)
  ifft_plan_multi = plan_ifft!(means_0, 2:N+1)

  density_0, var_0 = initialize_distribution(kde)
  ifft_plan_single = plan_ifft!(density_0, 1:N)

  fourier_grid = fftgrid(kde.grid)
  fourier_grid_array = get_coordinates(fourier_grid)

  times = CuArray{T,N}(
    reinterpret(reshape, T, collect(zip(time_range...)))
  )
  time_initial = initial_bandwidth(kde.grid)

  means_dst = CuArray{Complex{T},N + 1}(undef, size(means_0))
  variances_dst = CuArray{Complex{T},N + 1}(undef, size(variances_0))

  for col in 1:size(times, 2)
    time = view(times, :, col)

    means_bootstraps, variances_bootstraps = propagate_bandwidth!(
      means_0,
      variances_0,
      fourier_grid_array,
      time,
      time_initial,
      dst_mean=means_dst,
      dst_var=variances_dst,
    )

    vmr_variance = calculate_statistics!(
      means_bootstraps,
      variances_bootstraps,
      ifft_plan_multi,
      dst_vmr=selectdim(variances_dst, 1, 1)
    )

    means_complete, variances_complete = propagate_bandwidth!(
      reshape(density_0, 1, size(density_0)...),
      reshape(var_0, 1, size(var_0)...),
      fourier_grid_array,
      time,
      time_initial,
      dst_mean=reshape(selectdim(means_dst, 1, 1), 1, size(means_dst)[2:end]...),
      dst_var=reshape(selectdim(means_dst, 1, 2), 1, size(means_dst)[2:end]...),
    )
    means_complete = dropdims(means_complete, dims=1)
    variances_complete = dropdims(variances_complete, dims=1)

    assign_density!(
      kde,
      means_complete,
      variances_complete,
      vmr_variance,
      time,
      threshold,
      ifft_plan_single
    )

    if all(isfinite, kde.density)
      kde.t .= time
      break
    end

    if col == size(times, 2)
      @warn "Not all points converged with the specified time ranges."
    end
  end

  return nothing
end

function initialize_statistics(
  kde::KDE{N,T,S,M},
  n_bootstraps::Integer,
  method::Symbol
)::NTuple{2,Array{Complex{T},N + 1}} where {N,T<:Real,S<:Real,M}
  dirac_series, dirac_series_squared = initialize_dirac_series(
    Val(method),
    kde,
    n_bootstraps=n_bootstraps,
    calculate_squared=true
  )
  sk_0, s2k_0 = initialize_fourier_statistics(dirac_series, dirac_series_squared)

  return sk_0, s2k_0
end
function initialize_statistics(
  kde::CuKDE{N,T,S,M},
  n_bootstraps::Integer,
)::NTuple{2,CuArray{Complex{T},N + 1}} where {N,T<:Real,S<:Real,M}
  dirac_series, dirac_series_squared = initialize_dirac_series(
    Val(:cuda),
    kde,
    n_bootstraps=n_bootstraps,
    calculate_squared=true
  )
  s_0, s2_0 = initialize_fourier_statistics(dirac_series, dirac_series_squared)

  return s_0, s2_0
end

function initialize_distribution(
  kde::KDE{N,T,S,M}, method::Symbol
)::Array{Complex{T},N} where {N,T<:Real,S<:Real,M}
  dirac_series, dirac_series_squared = initialize_dirac_series(
    Val(method),
    kde,
    calculate_squared=true
  )
  s_0, s2_0 = initialize_fourier_statistics(dirac_series, dirac_series_squared)

  return dropdims(s_0, dims=1), dropdims(s2_0, dims=1)
end
function initialize_distribution(
  kde::CuKDE{N,T,S,M}
)::CuArray{Complex{T},N} where {N,T<:Real,S<:Real,M}
  dirac_series, dirac_series_squared = initialize_dirac_series(
    Val(:cuda),
    kde,
    calculate_squared=true
  )

  s_0, s2_0 = initialize_fourier_statistics(dirac_series, dirac_series_squared)

  return dropdims(s_0, dims=1), dropdims(s2_0, dims=1)
end

function propagate_bandwidth!(
  means_0::Array{Complex{T},M},
  variances_0::Array{Complex{T},M},
  grid_array::AbstractArray{S,M},
  time::Vector{<:Real},
  time_initial::Vector{<:Real},
  method::Symbol;
  dst_mean::AbstractArray{Complex{T},M}=Array{Complex{T},M}(undef, size(means_0)),
  dst_var::AbstractArray{Complex{T},M}=Array{Complex{T},M}(undef, size(variances_0)),
)::Nothing where {T<:Real,S<:Real,M}
  propagate_statistics!(
    Val(method),
    dst_mean,
    dst_mean,
    means_0,
    variances_0,
    grid_array,
    SVector{M - 1,Float64}(time),
    SVector{M - 1,Float64}(time_initial)
  )

  return dst_mean, dst_var
end
function propagate_bandwidth!(
  means_0::CuArray{Complex{T},M},
  variances_0::CuArray{Complex{T},M},
  grid_array::CuArray{S,M},
  time::CuVector{<:Real},
  time_initial::CuVector{<:Real};
  dst_mean::CuArray{Complex{T},M}=CuArray{Complex{T},M}(undef, size(means_0)),
  dst_var::CuArray{Complex{T},M}=CuArray{Complex{T},M}(undef, size(variances_0)),
)::Nothing where {T<:Real,S<:Real,M}
  propagate_statistics!(
    Val(:cuda),
    dst_mean,
    dst_var,
    means_0,
    variances_0,
    grid_array,
    time,
    time_initial
  )

  return dst_mean, dst_var
end

function calculate_statistics!(
  means_bootstraps::AbstractArray{Complex{T},M},
  variances_bootstraps::AbstractArray{Complex{T},M},
  ifft_plan::AbstractFFTs.ScaledPlan,
  method::Symbol;
  dst_vmr::AbstractArray{Complex{T},N}=Array{Complex{T},M - 1}(undef, size(means_bootstraps))
)::AbstractArray{T,N} where {N,T<:Real,M}
  vmr_variance = mean_var_vmr!(
    Val(method),
    means_bootstraps,
    variances_bootstraps,
    ifft_plan,
    dst_vmr
  )

  return vmr_variance
end
function calculate_statistics!(
  means_bootstraps::CuArray{Complex{T},M},
  variances_bootstraps::CuArray{Complex{T},M},
  ifft_plan::AbstractFFTs.ScaledPlan;
  dst_vmr::CuArray{Complex{T},N}=CuArray{Complex{T},M - 1}(undef, size(means_bootstraps))
)::CuArray{T,N} where {N,T<:Real,M}
  vmr_variance = mean_var_vmr!(
    Val(:cuda),
    means_bootstraps,
    variances_bootstraps,
    ifft_plan,
    dst_vmr
  )

  return vmr_variance
end

function get_times(
  dt::Union{Real,Vector{<:Real},Nothing},
  n_steps::Union{Int,Nothing},
  t_final::Union{Real,Vector{<:Real},Nothing},
  t0::AbstractVector{<:Real},
  ignore_t0::Bool,
  grid::AbstractGrid
)::Vector{<:AbstractRange{<:Real}}

  if ignore_t0
    bandwidth_0 = initial_bandwidth(grid)
    t0 = bandwidth_0
  end
  if t0 isa CuArray
    t0 = Array{Float64}(t0)
  end

  if any(t_final .< t0)
    throw(ArgumentError("The final time must be greater than the initial bandwidth."))
  end

  if ignore_t0
    t0 = @SVector zeros(Float64, length(t0))
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

function get_smoothness(smoothness::Real, n_samples::Integer, n_dims::Int)
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
