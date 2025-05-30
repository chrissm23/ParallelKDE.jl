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
  sk::AbstractArray{<:Complex,M},
  s2k::AbstractArray{<:Complex,M},
  plan::FFTW.FFTWPlan,
) where {M}
  n_bootstraps = size(sk)[end]

  for i in 1:n_bootstraps
    plan * selectdim(sk, ndims(sk), i)
    plan * selectdim(s2k, ndims(sk), i)
  end

  return nothing
end
function fourier_statistics!(
  ::Val{:serial},
  sk::AbstractArray{<:Complex,M},
  plan::FFTW.FFTWPlan,
) where {M}
  n_bootstraps = size(sk)[end]

  for i in 1:n_bootstraps
    plan * selectdim(sk, ndims(sk), i)
  end

  return nothing
end
function fourier_statistics!(
  ::Val{:threaded},
  sk::AbstractArray{<:Complex,M},
  s2k::AbstractArray{<:Complex,M},
  plan::FFTW.FFTWPlan,
) where {M}
  n_bootstraps = size(sk)[end]

  Threads.@threads for i in 1:n_bootstraps
    plan * selectdim(sk, ndims(sk), i)
    plan * selectdim(s2k, ndims(s2k), i)
  end

  return nothing
end
function fourier_statistics!(
  ::Val{:threaded},
  sk::AbstractArray{<:Complex,M},
  plan::FFTW.FFTWPlan,
) where {M}
  n_bootstraps = size(sk)[end]

  Threads.@threads for i in 1:n_bootstraps
    plan * selectdim(sk, ndims(sk), i)
  end

  return nothing
end
function fourier_statistics!(
  ::Val{:cuda},
  sk::AnyCuArray{<:Complex,M},
  s2k::AnyCuArray{<:Complex,M},
  plan::CUFFT.CuFFTPlan{<:Complex},
) where {M}
  plan * sk
  plan * s2k

  return nothing
end
function fourier_statistics!(
  ::Val{:cuda},
  sk::AnyCuArray{<:Complex,M},
  plan::CUFFT.CuFFTPlan{<:Complex},
) where {M}
  plan * sk

  return nothing
end

function propagate_statistics!(
  ::Val{:serial},
  means_t::AbstractArray{<:Complex,M},
  variances_t::AbstractArray{<:Complex,M},
  means_0::AbstractArray{<:Complex,M},
  variances_0::AbstractArray{<:Complex,M},
  time_propagated::AbstractVector{<:Real},
  time_intial::AbstractVector{<:Real},
  grid_array::AbstractArray{<:Real,M},
) where {M}
  n_bootstraps = size(means_0)[end]

  for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, M, i),
      selectdim(variances_t, M, i),
      selectdim(means_0, M, i),
      selectdim(variances_0, M, i),
      time_propagated,
      time_intial,
      grid_array,
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:serial},
  means_t::AbstractArray{<:Complex,M},
  means_0::AbstractArray{<:Complex,M},
  time_propagated::AbstractVector{<:Real},
  grid_array::AbstractArray{<:Real,M},
) where {M}
  n_bootstraps = size(means_0)[end]

  for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, M, i),
      selectdim(means_0, M, i),
      time_propagated,
      grid_array,
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:threaded},
  means_t::AbstractArray{<:Complex,M},
  variances_t::AbstractArray{<:Complex,M},
  means_0::AbstractArray{<:Complex,M},
  variances_0::AbstractArray{<:Complex,M},
  time_propagated::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real},
  grid_array::AbstractArray{<:Real,M},
) where {M}
  n_bootstraps = size(means_0)[end]

  Threads.@threads for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, M, i),
      selectdim(variances_t, M, i),
      selectdim(means_0, M, i),
      selectdim(variances_0, M, i),
      time_propagated,
      time_initial,
      grid_array,
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:threaded},
  means_t::AbstractArray{<:Complex,M},
  means_0::AbstractArray{<:Complex,M},
  time_propagated::AbstractVector{<:Real},
  grid_array::AbstractArray{<:Real,M},
) where {M}
  n_bootstraps = size(means_0)[end]

  Threads.@threads for i in 1:n_bootstraps
    propagate_time_cpu!(
      selectdim(means_t, M, i),
      selectdim(means_0, M, i),
      time_propagated,
      grid_array,
    )
  end

  return nothing
end

function propagate_time_cpu!(
  means_t::AbstractArray{<:Complex,N},
  variances_t::AbstractArray{<:Complex,N},
  means_0::AbstractArray{<:Complex,N},
  variances_0::AbstractArray{<:Complex,N},
  time_propagated::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real},
  grid_array::AbstractArray{<:Real,M},
) where {N,M}
  det_t0 = prod(time_initial)
  det_t = sqrt(prod(time_initial .^ 2 .+ time_propagated .^ 2))

  cartesian_idxs = CartesianIndices(means_t)
  @inbounds @simd for idx in eachindex(means_t)
    propagator = 0.0
    grid_point = view(grid_array, :, cartesian_idxs[idx])
    @inbounds @simd for i in 1:N
      propagator += (grid_point[i] * time_propagated[i])^2
    end
    propagator = exp(-0.5 * propagator)

    @inbounds means_t[idx] = means_0[idx] * propagator
    @inbounds variances_t[idx] = variances_0[idx] * sqrt(propagator) * det_t0 / det_t
  end

  return nothing
end
function propagate_time_cpu!(
  means_t::AbstractArray{<:Complex,N},
  means_0::AbstractArray{<:Complex,N},
  time_propagated::AbstractVector{<:Real},
  grid_array::AbstractArray{<:Real,M},
) where {N,M}
  cartesian_idxs = CartesianIndices(means_t)
  @inbounds @simd for idx in eachindex(means_t)
    propagator = 0.0
    grid_point = view(grid_array, :, cartesian_idxs[idx])
    @inbounds @simd for i in 1:N
      propagator += (grid_point[i] * time_propagated[i])^2
    end

    propagator = exp(-0.5 * propagator)

    @inbounds means_t[idx] = means_0[idx] * propagator
  end

  return nothing
end

function propagate_statistics!(
  ::Val{:cuda},
  means_t::AnyCuArray{<:Complex,M},
  variances_t::AnyCuArray{<:Complex,M},
  means_0::AnyCuArray{<:Complex,M},
  variances_0::AnyCuArray{<:Complex,M},
  time_propagated::CuVector{<:Real},
  time_initial::CuVector{<:Real},
  grid_array::AnyCuArray{<:Real,M},
) where {M}
  n_points = prod(size(means_t))
  kernel = @cuda maxregs = 32 launch = false propagate_time_cuda!(
    means_t,
    variances_t,
    means_0,
    variances_0,
    time_propagated,
    time_initial,
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
      time_propagated,
      time_initial,
      grid_array;
      threads,
      blocks
    )
  end

  return nothing
end
function propagate_statistics!(
  ::Val{:cuda},
  means_t::AnyCuArray{<:Complex,M},
  means_0::AnyCuArray{<:Complex,M},
  time_propagated::CuVector{<:Real},
  grid_array::AnyCuArray{<:Real,M},
) where {M}
  n_points = prod(size(means_t))
  kernel = @cuda maxregs = 32 launch = false propagate_time_cuda!(
    means_t,
    means_0,
    time_propagated,
    grid_array,
  )

  config = launch_configuration(kernel.fun)
  threads = min(n_points, config.threads)
  blocks = cld(n_points, threads)

  CUDA.@sync blocking = true begin
    kernel(
      means_t,
      means_0,
      time_propagated,
      grid_array;
      threads,
      blocks
    )
  end

  return nothing
end

function propagate_time_cuda!(
  means_t::CuDeviceArray{<:Complex,M},
  variances_t::CuDeviceArray{<:Complex,M},
  means_0::CuDeviceArray{<:Complex,M},
  variances_0::CuDeviceArray{<:Complex,M},
  time_propagated::CuDeviceVector{<:Real},
  time_initial::CuDeviceVector{<:Real},
  grid_array::CuDeviceArray{<:Real,M},
) where {M}
  N = M - 1i32
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  n_bootstraps = size(means_t, Int32(M))
  grid_size = size(grid_array)[2i32:end]
  n_gridpoints = prod(grid_size)
  n_points = n_bootstraps * n_gridpoints

  bootstrap_idx_tmp, point_idx_tmp = divrem(idx - 1i32, n_gridpoints)
  bootstrap_idx = bootstrap_idx_tmp + 1i32
  point_idx = point_idx_tmp + 1i32

  if idx > n_points
    return
  end

  cartesian_idx = Tuple(
    CartesianIndices(grid_size)[point_idx]
  )

  propagator_mean = 1.0f0
  det_t = 1.0f0
  i = 1i32
  while i <= N
    @inbounds propagator_mean *= exp(
      -0.5f0 * (grid_array[i, cartesian_idx...] * time_propagated[i])^2i32
    )
    @inbounds det_t *= time_initial[i]^2i32 + time_propagated[i]^2i32

    i += 1i32
  end

  det_t0 = prod(time_initial)
  det_t = sqrt(det_t)
  propagator_variance = sqrt(propagator_mean) * det_t0 / det_t

  @inbounds means_t[cartesian_idx..., bootstrap_idx] = (
    propagator_mean * means_0[cartesian_idx..., bootstrap_idx]
  )
  @inbounds variances_t[cartesian_idx..., bootstrap_idx] = (
    propagator_variance * variances_0[cartesian_idx..., bootstrap_idx]
  )

  return
end
function propagate_time_cuda!(
  means_t::CuDeviceArray{<:Complex,M},
  means_0::CuDeviceArray{<:Complex,M},
  time_propagated::CuDeviceVector{<:Real},
  grid_array::CuDeviceArray{<:Real,M},
) where {M}
  N = M - 1i32
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  n_bootstraps = size(means_t, Int32(M))
  grid_size = size(grid_array)[2i32:end]
  n_gridpoints = prod(grid_size)
  n_points = n_bootstraps * n_gridpoints

  bootstrap_idx_temp, point_idx_temp = divrem(idx - 1i32, n_gridpoints)
  bootstrap_idx = bootstrap_idx_temp + 1i32
  point_idx = point_idx_temp + 1i32

  if idx > n_points
    return
  end

  cartesian_idx = Tuple(
    CartesianIndices(grid_size)[point_idx]
  )

  propagator_exponential = 1.0f0
  i = 1i32
  while i <= N
    @inbounds propagator_exponential *= exp(
      -0.5f0 * (grid_array[i, cartesian_idx...] * time_propagated[i])^2i32
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
  sk::AbstractArray{<:Complex,M},
  s2k::AbstractArray{<:Complex,M},
  ifft_plan::AbstractFFTs.ScaledPlan{<:Complex},
) where {M}
  n_bootstraps = size(sk)[end]

  for i in 1:n_bootstraps
    ifft_plan * selectdim(sk, ndims(sk), i)
    ifft_plan * selectdim(s2k, ndims(s2k), i)
  end

  return nothing
end
function ifourier_statistics!(
  ::Val{:serial},
  sk::AbstractArray{<:Complex,M},
  ifft_plan::AbstractFFTs.ScaledPlan{<:Complex},
) where {M}
  n_bootstraps = size(sk)[end]

  for i in 1:n_bootstraps
    ifft_plan * selectdim(sk, ndims(sk), i)
  end

  return nothing
end
function ifourier_statistics!(
  ::Val{:threaded},
  sk::AbstractArray{<:Complex,M},
  s2k::AbstractArray{<:Complex,M},
  ifft_plan::AbstractFFTs.ScaledPlan{<:Complex},
) where {M}
  n_bootstraps = size(sk, M)

  Threads.@threads for i in 1:n_bootstraps
    ifft_plan * selectdim(sk, M, i)
    ifft_plan * selectdim(s2k, M, i)
  end

  return nothing
end
function ifourier_statistics!(
  ::Val{:threaded},
  sk::AbstractArray{<:Complex,M},
  ifft_plan::AbstractFFTs.ScaledPlan{<:Complex},
) where {M}
  n_bootstraps = size(sk)[end]

  Threads.@threads for i in 1:n_bootstraps
    ifft_plan * selectdim(sk, ndims(sk), i)
  end

  return nothing
end
function ifourier_statistics!(
  ::Val{:cuda},
  sk::CuArray{<:Complex,M},
  s2k::CuArray{<:Complex,M},
  ifft_plan::AbstractFFTs.ScaledPlan{<:Complex},
) where {M}
  ifft_plan * sk
  ifft_plan * s2k

  return nothing
end
function ifourier_statistics!(
  ::Val{:cuda},
  sk::CuArray{<:Complex,M},
  ifft_plan::AbstractFFTs.ScaledPlan{<:Complex},
) where {M}
  ifft_plan * sk

  return nothing
end

end
