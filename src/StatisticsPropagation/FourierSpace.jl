module FourierSpace

using LinearAlgebra

using StaticArrays,
  FFTW,
  CUDA

using CUDA: i32

export fourier_statistics!,
  ifourier_statistics!,
  propagate_statistics!

function fourier_statistics!(
  ::Val{:serial},
  plan::FFTW.FFTWPlan,
  stat::AbstractArray{Complex{T},M},
) where {T<:Real,M}
  n_bootstraps = size(stat)[end]

  for i in 1:n_bootstraps
    plan * selectdim(stat, ndims(stat), i)
  end
end
function fourier_statistics!(
  ::Val{:threaded},
  plan::FFTW.FFTWPlan,
  stat::AbstractArray{Complex{T},M},
) where {T<:Real,M}
  n_bootstraps = size(stat)[end]

  Threads.@threads for i in 1:n_bootstraps
    plan * selectdim(stat, ndims(stat), i)
  end
end
function fourier_statistics!(
  ::Val{:cuda},
  plan::AbstractFFTs.ScaledPlan{Complex{T}},
  stat::AnyCuArray{Complex{T},M},
) where {T<:Real,M}
  plan * stat
end

function propagate_statistics!(
  ::Val{:serial},
  means_t::AbstractArray{Complex{T},M},
  variances_t::AbstractArray{Complex{T},M},
  means_0::AbstractArray{Complex{T},M},
  variances_0::AbstractArray{Complex{T},M},
  time::SVector{N,P},
  time_intial::SVector{N,P},
  grid_array::AbstractArray{S,M},
) where {N,T<:Real,S<:Real,M,P<:Real}
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  n_bootstraps = size(means_0)[end]

  for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, M, i),
      selectdim(variances_t, M, i),
      selectdim(means_0, M, i),
      selectdim(variances_0, M, i),
      time,
      time_intial,
      grid_array,
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:serial},
  means_t::AbstractArray{Complex{T},M},
  means_0::AbstractArray{Complex{T},M},
  time::SVector{N,P},
  grid_array::AbstractArray{S,M},
) where {N,T<:Real,S<:Real,M,P<:Real}
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  n_bootstraps = size(means_0)[end]

  for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, M, i),
      selectdim(means_0, M, i),
      time,
      grid_array,
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:threaded},
  means_t::AbstractArray{Complex{T},M},
  variances_t::AbstractArray{Complex{T},M},
  means_0::AbstractArray{Complex{T},M},
  variances_0::AbstractArray{Complex{T},M},
  time::SVector{N,P},
  time_initial::SVector{N,P},
  grid_array::AbstractArray{S,M},
) where {N,T<:Real,S<:Real,M,P<:Real}
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  n_bootstraps = size(means_0)[end]

  Threads.@threads for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, M, i),
      selectdim(variances_t, M, i),
      selectdim(means_0, M, i),
      selectdim(variances_0, M, i),
      time,
      time_initial,
      grid_array,
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:threaded},
  means_t::AbstractArray{Complex{T},M},
  means_0::AbstractArray{Complex{T},M},
  time::SVector{N,P},
  grid_array::AbstractArray{S,M},
) where {N,T<:Real,S<:Real,M,P<:Real}
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  n_bootstraps = size(means_0)[end]

  Threads.@threads for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, M, i),
      selectdim(means_0, M, i),
      time,
      grid_array,
    )
  end

  return nothing
end

function propagate_time_cpu!(
  means_t::AbstractArray{Complex{T},N},
  variances_t::AbstractArray{Complex{T},N},
  means_0::AbstractArray{Complex{T},N},
  variances_0::AbstractArray{Complex{T},N},
  time::SVector{N,P},
  time_initial::SVector{N,P},
  grid_array::AbstractArray{S,M},
) where {T<:Real,N,S<:Real,M,P<:Real}
  det_t0 = prod(time_initial .^ 2)
  det_t = prod(time .^ 2)
  sqrt_t0 = sqrt(det_t0)
  sqrt_t0t = sqrt(det_t0 + det_t)

  @inbounds @simd for idx in eachindex(means_t)
    propagator = 0.0
    @inbounds @simd for i in 1:N
      propagator += (grid_array[i, idx] * time[i])^2
    end
    propagator = exp(-0.5 * propagator)

    @inbounds means_t[idx] = means_0[idx] * propagator
    @inbounds variances_t[idx] = variances_0[idx] * propagator * sqrt_t0 / sqrt_t0t
  end

  return nothing
end
function propagate_time_cpu!(
  means_t::AbstractArray{Complex{T},N},
  means_0::AbstractArray{Complex{T},N},
  time::SVector{N,P},
  grid_array::AbstractArray{S,M},
) where {T<:Real,N,S<:Real,M,P<:Real}
  @inbounds @simd for idx in eachindex(means_t)
    propagator = 0.0
    @inbounds @simd for i in 1:N
      propagator += (grid_array[i, idx] * time[i])^2
    end

    propagator = exp(-0.5 * propagator)

    @inbounds means_t[idx] = means_0[idx] * propagator
  end

  return nothing
end

function propagate_statistics!(
  ::Val{:cuda},
  means_t::AnyCuArray{Complex{T},M},
  variances_t::AnyCuArray{Complex{T},M},
  means_0::AnyCuArray{Complex{T},M},
  variances_0::AnyCuArray{Complex{T},M},
  time::CuVector{P},
  time_initial::CuVector{P},
  grid_array::AnyCuArray{S,M},
) where {T<:Real,S<:Real,M,P<:Real}
  n_points = prod(size(means_t))
  time_squared = time .^ 2
  time_initial_squared = time_initial .^ 2

  kernel = @cuda maxregs = 32 launch = false propagate_time_cuda!(
    means_t,
    variances_t,
    means_0,
    variances_0,
    time_squared,
    time_initial_squared,
    grid_array,
  )

  config = launch_configuration(kernel.fun)
  threads = min(n_points, config.threads)
  blocks = cld(n_points, threads)

  CUDA.@sync blocking = true begin
    kernel(
      means_t,
      variances_t,
      means_0,
      variances_0,
      time_squared,
      time_initial_squared,
      grid_array;
      threads,
      blocks
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:cuda},
  means_t::AnyCuArray{Complex{T},M},
  means_0::AnyCuArray{Complex{T},M},
  time::CuVector{P},
  grid_array::AnyCuArray{S,M},
) where {T<:Real,S<:Real,M,P<:Real}
  n_points = prod(size(means_t))
  time_squared = time .^ 2

  kernel = @cuda maxregs = 32 launch = false propagate_time_cuda!(
    means_t,
    means_0,
    grid_array,
    time_squared,
  )

  config = launch_configuration(kernel.fun)
  threads = min(n_points, config.threads)
  blocks = cld(n_points, threads)

  CUDA.@sync blocking = true begin
    kernel(
      means_t,
      means_0,
      grid_array,
      time_squared;
      threads,
      blocks
    )
  end

  return nothing
end

function propagate_time_cuda!(
  means_t::CuDeviceArray{Complex{T},M},
  variances_t::CuDeviceArray{Complex{T},M},
  means_0::CuDeviceArray{Complex{T},M},
  variances_0::CuDeviceArray{Complex{T},M},
  time_squared::CuDeviceVector{P},
  time_initial_squared::CuDeviceVector{P},
  grid_array::CuDeviceArray{S,M},
) where {T<:Real,S<:Real,M,P<:Real}
  N = M - 1i32
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  n_bootstraps = size(means_t, 1i32)
  grid_size = size(grid_array)[2i32:end]
  n_gridpoints = prod(grid_size)
  n_points = n_bootstraps * n_gridpoints

  bootstrap_idx_tmp, point_idx_tmp = divrem(idx - 1i32, n_gridpoints)
  bootstrap_idx = bootstrap_idx_tmp + 1i32
  point_idx = point_idx_tmp + 1i32

  if idx > n_points
    return
  end

  cartesian_indices = CartesianIndices(
    grid_size
  )
  cartesian_idx = Tuple(cartesian_indices[point_idx])

  propagator_mean = 1.0f0
  i = 1i32
  while i <= N
    @inbounds propagator_mean *= exp(
      -0.5f0 * grid_array[i, cartesian_idx...]^2i32 * time_squared[i]
    )

    i += 1i32
  end

  det_t = prod(time_squared)
  det_t0 = prod(time_initial_squared)
  propagator_variance = sqrt(det_t0) * sqrt(propagator_mean) / sqrt(det_t + det_t0)

  @inbounds means_t[cartesian_idx..., bootstrap_idx] = (
    propagator_mean * means_0[cartesian_idx..., bootstrap_idx]
  )
  @inbounds variances_t[cartesian_idx..., bootstrap_idx] = (
    propagator_variance * variances_0[cartesian_idx..., bootstrap_idx]
  )

  return
end
function propagate_time_cuda!(
  means_t::CuDeviceArray{Complex{T},M},
  means_0::CuDeviceArray{Complex{T},M},
  time_squared::CuDeviceVector{P},
  grid_array::CuDeviceArray{S,M},
) where {T<:Real,S<:Real,M,P<:Real}
  N = M - 1i32
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  n_bootstraps = size(means_t, 1i32)
  grid_size = size(grid_array)[2i32:end]
  n_gridpoints = prod(grid_size)
  n_points = n_bootstraps * n_gridpoints

  bootstrap_idx_temp, point_idx_temp = divrem(idx - 1i32, n_gridpoints)
  bootstrap_idx = bootstrap_idx_temp + 1i32
  point_idx = point_idx_temp + 1i32

  if idx > n_points
    return
  end

  cartesian_indices = CartesianIndices(grid_size)
  cartesian_idx = Tuple(cartesian_indices[point_idx])

  propagator_exponential = 1.0f0
  i = 1i32
  while i <= N
    @inbounds propagator_exponential *= exp(
      -0.5f0 * grid_array[i, cartesian_idx...]^2i32 * time_squared[i]
    )

    i += 1i32
  end

  @inbounds means_t[cartesian_idx..., bootstrap_idx] = (
    propagator_exponential * means_0[cartesian_idx..., bootstrap_idx]
  )

  return
end

function ifourier_statistics!(
  ::Val{:serial},
  sk::AbstractArray{Complex{T},M},
  s2k::AbstractArray{Complex{T},M},
  n_samples::P,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M,P<:Integer}
  n_bootstraps = size(sk)[end]

  for i in 1:n_bootstraps
    ifft_plan * selectdim(sk, ndims(sk), i)
    ifft_plan * selectdim(s2k, ndims(s2k), i)
  end

  return nothing
end
function ifourier_statistics!(
  ::Val(:serial),
  sk::AbstractArray{Complex{T},M},
  n_samples::P,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M,P<:Integer}
  n_bootstraps = size(sk)[end]

  for i in 1:n_bootstraps
    ifft_plan * selectdim(sk, ndims(sk), i)
  end

  return nothing
end
function ifourier_statistics!(
  ::Val{:threaded},
  sk::AbstractArray{Complex{T},M},
  s2k::AbstractArray{Complex{T},M},
  n_samples::P,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M,P<:Integer}
  n_bootstraps = size(sk, M)

  Threads.@threads for i in 1:n_bootstraps
    ifft_plan * selectdim(sk, M, i)
    ifft_plan * selectdim(s2k, M, i)
  end

  # Threads.@threads for i in 1:n_bootstraps
  #   sk_i = selectdim(sk, M, i)
  #   s2k_i = selectdim(s2k, M, i)
  #
  #   @inbounds @simd for idx in eachindex(selectdim(sk, M, i))
  #     sk_j = abs(sk_i[idx]) / n_samples
  #     sk_i[idx] = sk_j
  #     s2k_i[idx] = (abs(s2k_i[idx]) / n_samples) - sk_j^2
  #   end
  # end

  return nothing
end
function ifourier_statistics!(
  ::Val(:threaded),
  sk::AbstractArray{Complex{T},M},
  n_samples::P,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M,P<:Integer}
  n_bootstraps = size(sk)[end]

  Threads.@threads for i in 1:n_bootstraps
    ifft_plan * selectdim(sk, ndims(sk), i)
  end

  # Threads.@threads for idx in eachindex(sk)
  #   sk[idx] = abs(sk[idx]) / n_samples
  # end

  return nothing
end
function ifourier_statistics!(
  ::Val{:cuda},
  sk::AnyCuArray{Complex{T},M},
  s2k::AnyCuArray{Complex{T},M},
  n_samples::P,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M,P<:Integer}
  ifft_plan * sk
  ifft_plan * s2k

  # @. sk = abs(sk) / n_samples
  # @. s2k = (abs(s2k) / n_samples) - sk^2

  return nothing
end
function ifourier_statistics!(
  ::Val{:cuda},
  sk::AnyCuArray{Complex{T},M},
  n_samples::P,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M,P<:Integer}
  ifft_plan * sk

  # @. sk = abs(sk) / n_samples

  return nothing
end

end
