module FourierSpace

using StaticArrays,
  FFTW,
  CUDA

using CUDA: i32

export initialize_fourier_statistics,
  ifft_statistics

function initialize_fourier_statistics(
  dirac_series::Array{T,M},
  dirac_series_squared::Array{T,M}
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  sk_0 = fftshift(fft(dirac_series, 2:N+1), 2:N+1)
  s2k_0 = fftshift(fft(dirac_series_squared, 2:N+1), 2:N+1)

  return sk_0, s2k_0
end
function initialize_fourier_statistics(
  dirac_series::Array{T,M},
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  sk_0 = fftshift(fft(dirac_series, 2:N+1), 2:N+1)

  return sk_0
end
function initialize_fourier_statistics(
  dirac_series::CuArray{T,M},
  dirac_series_squared::CuArray{T,M}
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  sk_0 = fftshift(fft(dirac_series, 2:N+1), 2:N+1)
  s2k_0 = fftshift(fft(dirac_series_squared, 2:N+1), 2:N+1)

  return sk_0, s2k_0
end
function initialize_fourier_statistics(
  dirac_series::CuArray{T,M},
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  sk_0 = fftshift(fft(dirac_series, 2:N+1), 2:N+1)

  return sk_0
end

function propagate_statistics!(
  ::Val{:cpu},
  means_t::Array{Complex{T},M},
  variances_t::Array{Complex{T},M},
  means_0::Array{Complex{T},M},
  variances_0::Array{Complex{T},M},
  grid_array::Array{S,M},
  time::SVector{N,Float64},
) where {N,T<:Real,S<:Real,M}
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  n_bootstraps = size(means_0, 1)

  for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, 1, i),
      selectdim(variances_t, 1, i),
      selectdim(means_0, 1, i),
      selectdim(variances_0, 1, i),
      grid_array,
      time
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:cpu},
  means_t::Array{Complex{T},M},
  means_0::Array{Complex{T},M},
  grid_array::Array{S,M},
  time::SVector{N,Float64},
) where {N,T<:Real,S<:Real,M}
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  n_bootstraps = size(means_0, 1)

  for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, 1, i),
      selectdim(means_0, 1, i),
      grid_array,
      time
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:threaded},
  means_t::Array{Complex{T},M},
  variances_t::Array{Complex{T},M},
  means_0::Array{Complex{T},M},
  variances_0::Array{Complex{T},M},
  grid_array::Array{S,M},
  time::SVector{N,Float64},
) where {N,T<:Real,S<:Real,M}
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  n_bootstraps = size(means_0, 1)

  Threads.@threads for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, 1, i),
      selectdim(variances_t, 1, i),
      selectdim(means_0, 1, i),
      selectdim(variances_0, 1, i),
      grid_array,
      time
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:threaded},
  means_t::Array{Complex{T},M},
  means_0::Array{Complex{T},M},
  grid_array::Array{S,M},
  time::SVector{N,Float64},
) where {N,T<:Real,S<:Real,M}
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  n_bootstraps = size(means_0, 1)

  Threads.@threads for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, 1, i),
      selectdim(means_0, 1, i),
      grid_array,
      time
    )
  end

  return nothing
end

function propagate_time_cpu!(
  means_t::Array{Complex{T},N},
  variances_t::Array{Complex{T},N},
  means_0::Array{Complex{T},N},
  variance_0::Array{Complex{T},N},
  grid_array::Array{S,M},
  time::SVector{<:Real},
) where {T<:Real,N,S<:Real,M}
  time_squared = reshape(time .^ 2, :, ones(Int, N)...)
  propagator_exponent = Array{T,N}(undef, size(means_t))

  propagator_exponent .= dropdims(-1.0 .* sum(grid_array .^ 2 .* time_squared, dims=1), dims=1)
  @. means_t = exp(propagator_exponent / 2.0) * means_0
  @. variances_t = exp(propagator_exponent / 4.0) * variance_0

  return nothing
end
function propagate_time_cpu!(
  means_t::Array{Complex{T},N},
  means_0::Array{Complex{T},N},
  grid_array::Array{S,M},
  time::SVector{<:Real},
) where {T<:Real,N,S<:Real,M}
  time_squared = reshape(time .^ 2, :, ones(Int, N)...)
  propagator_exponent = Array{T,N}(undef, size(means_t))

  propagator_exponent .= dropdims(-1.0 .* sum(grid_array .^ 2 .* time_squared, dims=1), dims=1)
  @. means_t = exp(propagator_exponent / 2.0) * means_0

  return nothing
end

function propagate_statistics!(
  ::Val{:cuda},
  means_t::CuArray{Complex{T},M},
  variances_t::CuArray{Complex{T},M},
  means_0::CuArray{Complex{T},M},
  variances_0::CuArray{Complex{T},M},
  grid_array::CuArray{S,M},
  time::CuVector{<:Real},
) where {T<:Real,S<:Real,M}
  n_points = prod(size(means_t)[2:end])
  time_squared = time .^ 2

  kernel = @cuda launch = false propagate_time_gpu!(
    means_t,
    variances_t,
    means_0,
    variances_0,
    grid_array,
    time_squared,
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
      grid_array,
      time_squared;
      threads,
      blocks
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:cuda},
  means_t::CuArray{Complex{T},M},
  means_0::CuArray{Complex{T},M},
  grid_array::CuArray{S,M},
  time::CuVector{<:Real},
) where {T<:Real,S<:Real,M}
  n_points = prod(size(means_t)[2:end])
  time_squared = time .^ 2

  kernel = @cuda launch = false propagate_time_gpu!(
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

function propagate_time_gpu!(
  means_t::CuDeviceArray{Complex{T},M},
  variances_t::CuDeviceArray{Complex{T},M},
  means_0::CuDeviceArray{Complex{T},M},
  variances_0::CuDeviceArray{Complex{T},M},
  grid_array::CuDeviceArray{S,M},
  time_squared::CuDeviceVector{<:Real},
) where {T<:Real,S<:Real,M}
  N = M - 1
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  n_bootstraps = size(means_t, 1i32)
  grid_size = size(grid_array)[2:end]
  n_gridpoints = prod(grid_size)
  n_points = n_bootstraps * n_gridpoints

  bootstrap_idx_temp, point_idx_temp = divrem(idx - 1i32, n_bootstraps)
  bootstrap_idx = bootstrap_idx_temp + 1i32
  point_idx = point_idx_temp + 1i32

  if idx > n_points
    return
  end

  cartesian_indices = CartesianIndices(grid_size)
  cartesian_idx = Tuple(cartesian_indices[point_idx])

  propagator_exponent = 0.0f0
  i = 1i32
  while i <= N
    grid_index = (i, cartesian_idx...)
    @inbounds propagator_exponent -= grid_array[grid_index...]^2 * time_squared[i]
    i += 1
  end

  array_index = (bootstrap_idx, cartesian_idx...)
  @inbounds means_t[array_index...] = exp(propagator_exponent / 2.0f0) * means_0[array_index...]
  @inbounds variances_t[array_index...] = exp(propagator_exponent / 4.0f0) * variances_0[array_index...]

  return
end
function propagate_time_gpu!(
  means_t::CuDeviceArray{Complex{T},M},
  means_0::CuDeviceArray{Complex{T},M},
  grid_array::CuDeviceArray{S,M},
  time_squared::CuDeviceVector{<:Real},
) where {T<:Real,S<:Real,M}
  N = M - 1
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  n_bootstraps = size(means_t, 1i32)
  grid_size = size(grid_array)[2:end]
  n_gridpoints = prod(grid_size)
  n_points = n_bootstraps * n_gridpoints

  bootstrap_idx_temp, point_idx_temp = divrem(idx - 1i32, n_bootstraps)
  bootstrap_idx = bootstrap_idx_temp + 1i32
  point_idx = point_idx_temp + 1i32

  if idx > n_points
    return
  end

  cartesian_indices = CartesianIndices(grid_size)
  cartesian_idx = Tuple(cartesian_indices[point_idx])

  propagator_exponent = 0.0f0
  i = 1i32
  while i <= N
    grid_index = (i, cartesian_idx...)
    @inbounds propagator_exponent -= grid_array[grid_index...]^2 * time_squared[i]
    i += 1
  end

  array_index = (bootstrap_idx, cartesian_idx...)
  @inbounds means_t[array_index...] = exp(propagator_exponent / 2.0f0) * means_0[array_index...]

  return
end

function ifft_statistics(
  sk::Array{Complex{T},M},
  s2k::Array{Complex{T},M},
  n_samples::Integer;
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  s = abs.(ifft(ifftshift(sk, 2:N+1), 2:N+1))
  s2 = abs.(ifft(ifftshift(s2k, 2:N+1), 2:N+1))

  means = s ./ n_samples
  variances = (s2 ./ n_samples) .- means .^ 2

  return means, variances
end
function ifft_statistics(
  sk::Array{Complex{T},M},
  s2k::Array{Complex{T},M},
  n_samples::Integer,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  s = abs.(ifft_plan * ifftshift(sk, 2:N+1))
  s2 = abs.(ifft_plan * ifftshift(s2k, 2:N+1))

  means = s ./ n_samples
  variances = (s2 ./ n_samples) .- means .^ 2

  return means, variances
end
function ifft_statistics(
  sk::CuArray{Complex{T},M},
  s2k::CuArray{Complex{T},M},
  n_samples::Integer;
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  s = abs.(ifft(ifftshift(sk, 2:N+1), 2:N+1))
  s2 = abs.(ifft(ifftshift(s2k, 2:N+1), 2:N+1))

  means = s ./ n_samples
  variances = (s2 ./ n_samples) .- means .^ 2

  return means, variances
end
function ifft_statistics(
  sk::CuArray{Complex{T},M},
  s2k::CuArray{Complex{T},M},
  n_samples::Integer,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  s = abs.(ifft_plan * ifftshift(sk, 2:N+1))
  s2 = abs.(ifft_plan * ifftshift(s2k, 2:N+1))

  means = s ./ n_samples
  variances = (s2 ./ n_samples) .- means .^ 2

  return means, variances
end

end
