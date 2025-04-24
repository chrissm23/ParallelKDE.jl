module DirectSpace

using ..Grids
using ..KDEs
using ..FourierSpace

using Statistics,
  LinearAlgebra

using StaticArrays,
  StatsBase,
  CUDA,
  FFTW

using CUDA: i32

export silverman_rule,
  initialize_dirac_sequence,
  calculate_scaled_vmr!,
  identify_convergence!

include("../CuStatistics/CuStatistics.jl")

function initialize_dirac_sequence(
  ::Val{:serial},
  data::AbstractVector{SVector{N,S}},
  grid::AbstractGrid{N,P,M},
  bootstrap_idxs::AbstractMatrix{F};
  include_var::Bool=false,
  T=Float64,
) where {N,S<:Real,M,P<:Real,F<:Integer}
  grid_size = size(grid)
  n_bootstraps = size(bootstrap_idxs, 2)

  dirac_sequence = zeros(Complex{T}, grid_size..., n_bootstraps)
  dirac_sequence_real = selectdim(reinterpret(reshape, T, dirac_sequence), 1, 1)

  spacing = spacings(grid)
  low_bound = low_bounds(grid)

  if include_var
    dirac_sequence_squared = zeros(Complex{T}, grid_size..., n_bootstraps)
    dirac_sequence_squared_real = selectdim(
      reinterpret(reshape, T, dirac_sequence_squared), 1, 1
    )

    for i in 1:n_bootstraps
      generate_dirac_cpu!(
        selectdim(dirac_sequence_real, ndims(dirac_sequence), i),
        selectdim(dirac_sequence_squared_real, ndims(dirac_sequence_squared), i),
        view(data, bootstrap_idxs[:, i]),
        spacing,
        low_bound,
      )
    end

    return dirac_sequence, dirac_sequence_squared
  else
    for i in 1:n_bootstraps
      generate_dirac_cpu!(
        selectdim(dirac_sequence_real, ndims(dirac_sequence), i),
        view(data, bootstrap_idxs[:, i]),
        spacing,
        low_bound,
      )
    end

    return dirac_sequence
  end
end
function initialize_dirac_sequence(
  ::Val{:threaded},
  data::AbstractVector{SVector{N,S}},
  grid::AbstractGrid{N,P,M},
  bootstrap_idxs::AbstractMatrix{F};
  include_var::Bool=false,
  T=Float64,
) where {N,S<:Real,M,P<:Real,F<:Integer}
  grid_size = size(grid)
  n_bootstraps = size(bootstrap_idxs, 2)

  dirac_sequence = zeros(Complex{T}, grid_size..., n_bootstraps)
  dirac_sequence_real = selectdim(reinterpret(reshape, T, dirac_sequence), 1, 1)

  spacing = spacings(grid)
  low_bound = low_bounds(grid)

  if include_var
    dirac_sequence_squared = zeros(Complex{T}, grid_size..., n_bootstraps)
    dirac_sequence_squared_real = selectdim(
      reinterpret(reshape, T, dirac_sequence_squared), 1, 1
    )

    Threads.@threads for i in 1:n_bootstraps
      generate_dirac_cpu!(
        selectdim(dirac_sequence_real, ndims(dirac_sequence), i),
        selectdim(dirac_sequence_squared_real, ndims(dirac_sequence_squared), i),
        view(data, bootstrap_idxs[:, i]),
        spacing,
        low_bound,
      )
    end

    return dirac_sequence, dirac_sequence_squared
  else
    Threads.@threads for i in 1:n_bootstraps
      generate_dirac_cpu!(
        selectdim(dirac_sequence_real, ndims(dirac_sequence), i),
        view(data, bootstrap_idxs[:, i]),
        spacing,
        low_bound,
      )
    end

    return dirac_sequence
  end
end

function generate_dirac_cpu!(
  dirac_series::AbstractArray{T,N},
  dirac_series_squared::AbstractArray{T,N},
  data::AbstractVector{SVector{N,S}},
  spacing::SVector{N,T},
  low_bound::SVector{N,T},
) where {N,T<:Real,S<:Real}
  n_samples = length(data)
  spacing_squared = prod(spacing)^2

  indices_l = @MVector zeros(Int64, N)
  indices_h = @MVector zeros(Int64, N)
  remainder_l = @MVector zeros(T, N)
  remainder_h = @MVector zeros(T, N)

  mask = falses(N)
  grid_indices = @MVector zeros(Int64, N)

  for sample in data
    @. indices_l = floor(Int64, (sample - low_bound) / spacing) + 1
    @. remainder_l = (sample - low_bound) % spacing
    @. indices_h = indices_l + 1
    @. remainder_h = spacing - remainder_l

    @inbounds for i in 0:(2^N-1)
      @inbounds @simd for j in 1:N
        mask[j] = (i >> (N - j)) & 1 == 1
        grid_indices[j] = ifelse(mask[j], indices_l[j], indices_h[j])
      end

      product = prod(remainder_l[mask]) * prod(remainder_h[.!mask])

      dirac_series_term = product / (n_samples * spacing_squared)
      dirac_series[grid_indices...] += dirac_series_term
      dirac_series_squared[grid_indices...] += dirac_series_term^2
    end
  end

  return nothing
end
function generate_dirac_cpu!(
  dirac_series::AbstractArray{T,N},
  data::AbstractVector{SVector{N,S}},
  spacing::SVector{N,T},
  low_bound::SVector{N,T},
) where {N,T<:Real,S<:Real}
  n_samples = length(data)
  spacing_squared = prod(spacing)^2

  indices_l = @MVector zeros(Int64, N)
  indices_h = @MVector zeros(Int64, N)
  remainder_l = @MVector zeros(T, N)
  remainder_h = @MVector zeros(T, N)

  mask = falses(N)
  grid_indices = @MVector zeros(Int64, N)

  for sample in data
    @. indices_l = floor(Int64, (sample - low_bound) / spacing) + 1
    @. remainder_l = (sample - low_bound) % spacing
    @. indices_h = indices_l + 1
    @. remainder_h = spacing - remainder_l

    @inbounds for i in 0:(2^N-1)
      @inbounds @simd for j in 1:N
        mask[j] = (i >> (N - j)) & 1 == 1
        grid_indices[j] = ifelse(mask[j], indices_l[j], indices_h[j])
      end

      product = prod(remainder_l[mask]) * prod(remainder_h[.!mask])

      dirac_series_term = product / (n_samples * spacing_squared)
      dirac_series[grid_indices...] += dirac_series_term
    end
  end

  return nothing
end

function initialize_dirac_sequence(
  ::Val{:cuda},
  data::CuMatrix{S},
  grid::AbstractGrid{N,P,M},
  bootstrap_idxs::CuMatrix{Int32};
  include_var::Bool=false,
  T=Float32,
) where {N,S<:Real,M,P<:Real}
  grid_size = size(grid)
  n_samples, n_bootstraps = size(bootstrap_idxs)

  dirac_sequence = CUDA.zeros(Complex{T}, grid_size..., n_bootstraps)
  dirac_sequence_real = selectdim(reinterpret(reshape, T, dirac_sequence), 1, 1)

  spacing = spacings(grid)
  low_bound = low_bounds(grid)

  n_modified_gridpoints = n_samples * 2^N * n_bootstraps

  if include_var
    dirac_sequence_squared = CUDA.zeros(Complex{T}, grid_size..., n_bootstraps)
    dirac_sequence_squared_real = selectdim(
      reinterpret(reshape, T, dirac_sequence_squared), 1, 1
    )
    kernel = @cuda maxregs = 32, launch = false generate_dirac_cuda!(
      dirac_sequence_real, dirac_sequence_squared_real, data, bootstrap_idxs, spacing, low_bound
    )
  else
    kernel = @cuda maxregs = 32 launch = false generate_dirac_cuda!(
      dirac_sequence_real, data, bootstrap_idxs, spacing, low_bound
    )
  end

  config = launch_configuration(kernel.fun)
  threads = min(n_modified_gridpoints, config.threads)
  blocks = cld(n_modified_gridpoints, threads)

  if include_var
    CUDA.@sync blocking = true begin
      kernel(
        dirac_sequence,
        dirac_sequence_squared,
        data,
        bootstrap_idxs,
        spacing,
        low_bound;
        threads,
        blocks
      )
    end

    return dirac_sequence, dirac_sequence_squared
  else
    CUDA.@sync blocking = true begin
      kernel(
        dirac_sequence,
        data,
        bootstrap_idxs,
        spacing,
        low_bound;
        threads,
        blocks
      )
    end

    return dirac_sequence
  end
end

function generate_dirac_cuda!(
  dirac_series::CuDeviceArray{T,M},
  dirac_series_squared::CuDeviceArray{T,M},
  data::CuDeviceArray{S,2},
  bootstrap_idxs::CuDeviceArray{Int32,2},
  spacing::CuDeviceArray{T,1},
  low_bound::CuDeviceArray{T,1}
) where {M,T<:Real,S<:Real}
  spacing_squared = prod(spacing)^2i32

  n_dims = Int32(M) - 1i32
  n_samples = size(data, 2i32)
  n_bootstraps = size(bootstrap_idxs, 2i32)
  n_modified_gridpoints = (2i32)^n_dims * n_bootstraps * n_samples

  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  if idx > n_modified_gridpoints
    return
  end

  idx_sample_tmp, remainder = divrem(idx - 1i32, (2i32)^n_dims * n_bootstraps)
  idx_bootstrap_tmp, idx_dim = divrem(remainder, (2i32)^n_dims)
  idx_bootstrap = idx_bootstrap_tmp + 1i32
  idx_sample = idx_sample_tmp + 1i32

  mask = ntuple(i -> (idx_dim >> (n_dims - i)) & 1i32 == 1i32, n_dims)

  boot_idx = bootstrap_idxs[idx_sample, idx_bootstrap]
  grid_idxs = ntuple(
    i -> let
      index_l = floor(Int32, (data[i, boot_idx] - low_bound[i]) / spacing[i])
    end,
    n_dims
  )

  remainder_product = 1.0f0
  i = 1i32
  @inbounds while i <= n_dims
    remainder_l = (
      data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]
    ) % spacing[i]
    remainder_h = spacing[i] - remainder_l

    remainder_product *= ifelse(
      mask[end-i+1i32],
      remainder_l,
      remainder_h,
    )

    i += 1i32
  end

  dirac_series_term = remainder_product / (n_samples * spacing_squared)
  dirac_idx = (grid_idxs..., idx_bootstrap)
  @inbounds CUDA.@atomic dirac_series[dirac_idx...] += dirac_series_term
  @inbounds CUDA.@atomic dirac_series_squared[dirac_idx...] += dirac_series_term^2

  return
end
function generate_dirac_cuda!(
  dirac_series::CuDeviceArray{T,M},
  data::CuDeviceArray{S,2},
  bootstrap_idxs::CuDeviceArray{Int32,2},
  spacing::CuDeviceArray{T,1},
  low_bound::CuDeviceArray{T,1}
) where {M,T<:Real,S<:Real}
  spacing_squared = prod(spacing)^2i32

  n_dims = Int32(M) - 1i32
  n_samples = size(data, 2i32)
  n_bootstraps = size(bootstrap_idxs, 2i32)
  n_modified_gridpoints = (2i32)^n_dims * n_bootstraps * n_samples

  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  if idx > n_modified_gridpoints
    return
  end

  idx_sample_tmp, remainder = divrem(idx - 1i32, (2i32)^n_dims * n_bootstraps)
  idx_bootstrap_tmp, idx_dim = divrem(remainder, (2i32)^n_dims)
  idx_bootstrap = idx_bootstrap_tmp + 1i32
  idx_sample = idx_sample_tmp + 1i32

  mask = ntuple(i -> (idx_dim >> (n_dims - i)) & 1i32 == 1i32, n_dims)

  boot_idx = bootstrap_idxs[idx_sample, idx_bootstrap]
  grid_idxs = ntuple(
    i -> let
      index_l = floor(Int32, (data[i, boot_idx] - low_bound[i]) / spacing[i])
      return mask[i] ? index_l : index_l + 1i32
    end,
    n_dims
  )

  remainder_product = 1.0f0
  i = 1i32
  @inbounds while i <= n_dims
    remainder_l = (
      data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]
    ) % spacing[i]
    remainder_h = spacing[i] - remainder_l

    remainder_product *= ifelse(
      mask[end-i+1i32],
      remainder_l,
      remainder_h,
    )

    i += 1i32
  end

  dirac_series_term = remainder_product / (n_samples * spacing_squared)
  dirac_idx = (grid_idxs..., idx_bootstrap)
  @inbounds CUDA.@atomic dirac_series[dirac_idx...] += dirac_series_term

  return
end

function calculate_scaled_vmr!(
  ::Val{:serial},
  sk::AbstractArray{Complex{T},M},
  s2k::AbstractArray{Complex{T},M},
  time::AbstractVector{P},
  time_initial::AbstractVector{P},
  n_samples::F,
) where {M,T<:Real,P<:Real,F<:Integer}
  array_dims = size(sk)
  grid_dims = array_dims[begin:end-1]

  n_elements = prod(array_dims)
  n_bootstraps = array_dims[end]
  n_points = prod(grid_dims)

  sk_raw = vec(reinterpret(T, sk))
  s2k_raw = vec(reinterpret(T, s2k))

  @inbounds @simd for i in 1:n_elements
    sk_real = sk_raw[2i-1]
    sk_imag = sk_raw[2i]
    s2k_real = s2k_raw[2i-1]
    s2k_imag = s2k_raw[2i]

    sk_i = sqrt(sk_real^2 + sk_imag^2) / n_samples
    s2k_i = sqrt(s2k_real^2 + s2k_imag^2) / n_samples - sk_i^2
    s2k_raw[i] = s2k_i / sk_i
  end

  vmrs = reshape(view(s2k_raw, 1:n_elements), array_dims)
  vmrs_transposed = reshape(
    view(s2k_raw, n_elements+1:2n_elements),
    n_bootstraps, grid_dims...
  )

  perm = (M, 1:M-1...)
  permutedims!(vmrs_transposed, vmrs, perm)

  scaling_factor = prod(time .^ 2 .+ time_initial .^ 2)^(3 / 2) * n_samples^4

  vmr_var = reshape(view(s2k_raw, 1:n_points), grid_dims)
  @inbounds for j in 1:n_points
    base_idx = (j - 1) * n_bootstraps

    mean = zero(T)
    m2 = zero(T)
    @inbounds @simd for i in 1:n_bootstraps
      x = vmrs_transposed[base_idx+i]
      delta = x - mean
      mean += delta / i
      m2 += delta * (x - mean)
    end

    vmr_v = scaling_factor * m2 / (n_bootstraps - 1)
    vmr_var[j] = ifelse(isfinite(vmr_v), vmr_v, NaN)
  end

  return vmr_var
end
function calculate_scaled_vmr!(
  ::Val{:threaded},
  sk::AbstractArray{Complex{T},M},
  s2k::AbstractArray{Complex{T},M},
  time::AbstractVector{P},
  time_initial::AbstractVector{P},
  n_samples::F,
) where {M,T<:Real,P<:Real,F<:Integer}
  array_dims = size(sk)
  grid_dims = array_dims[begin:end-1]

  n_elements = prod(array_dims)
  n_bootstraps = array_dims[end]
  n_points = prod(grid_dims)

  sk_raw = vec(reinterpret(T, sk))
  s2k_raw = vec(reinterpret(T, s2k))

  # Calculate means and variances
  Threads.@threads @inbounds for i in 1:n_elements
    sk_real = sk_raw[2i-1]
    sk_imag = sk_raw[2i]
    s2k_real = s2k_raw[2i-1]
    s2k_imag = s2k_raw[2i]

    sk_i = sqrt(sk_real^2 + sk_imag^2) / n_samples
    s2k_i = sqrt(s2k_real^2 + s2k_imag^2) / n_samples - sk_i^2
    s2k_raw[2i-1] = s2k_i / sk_i
  end

  # Re-arrange into contiguous memory
  Threads.@threads @inbounds for i in 2:2:n_elements
    sk_raw[i] = sk_raw[2i-1]
    s2k_raw[i] = s2k_raw[2i-1]
  end
  Threads.@threads @inbounds for i in 3:4:n_elements
    sk_raw[i] = sk_raw[2i-1]
    s2k_raw[i] = s2k_raw[2i-1]
  end
  Threads.@threads @inbounds for i in 1:4:n_elements
    sk_raw[i] = sk_raw[2i-1]
    s2k_raw[i] = s2k_raw[2i-1]
  end

  # Transpose the array
  vmrs = reshape(view(s2k_raw, 1:n_elements), array_dims)
  vmrs_transposed = reshape(
    view(s2k_raw, n_elements+1:2n_elements),
    n_bootstraps, grid_dims...
  )
  perm = (M, 1:M-1...)
  idxs = CartesianIndices(vmrs)
  Threads.@threads for idx in eachindex(idxs)
    I = idxs[idx]
    J = ntuple(i -> I[perm[i]], M)
    vmrs_transposed[J...] = vmrs[I]
  end

  # Calculate the variance of the vmrs
  scaling_factor = prod(time .^ 2 + time_initial .^ 2)(3 / 2) * n_samples^4

  vmr_var = reshape(view(s2k_raw, 1:n_points), grid_dims)

  Threads.@threads for point in 0:n_points-1
    start_idx = point * n_bootstraps + 1
    end_idx = n_bootstraps * (point + 1)

    mean = zero(T)
    m2 = zero(T)
    count = 0

    @simd for i in start_idx:end_idx
      x = vmrs_transposed[i]
      count += 1
      delta = x - mean
      mean += delta / count
      m2 += delta * (x - mean)
    end

    vmr_v = scaling_factor * m2 / (count - 1)
    vmr_var[point+1] = ifelse(isfinite(vmr_v), vmr_v, NaN)
  end

  return vmr_var
end
function calculate_scaled_vmr!(
  ::Val{:cuda},
  means::AnyCuArray{Complex{T},M},
  vars::AnyCuArray{Complex{T},M},
  time::AnyCuVector{P},
  time_initial::AnyCuVector{P},
  n_samples::F,
) where {M,T<:Real,P<:Real,F<:Integer}
  scaling_factor = prod(time .^ 2i32 .+ time_initial .^ 2i32)^(1.5f0) * n_samples^4i32

  @. vars /= means
  vmr = selectdim(vars, M, 1)
  vmr .= dropdims(var(vars, dims=M), dims=M)
  vmr .*= scaling_factor

  @. vmr = ifelse(isfinite(vmr), vmr, NaN32)

  return vmr
end

function identify_convergence!(
  ::Val{:serial},
  vmr_current::AbstractArray{T,N},
  vmr_prev1::AbstractArray{T,N},
  vmr_prev2::AbstractArray{T,N},
  density::AbstractArray{T,N},
  means::AbstractArray{T,N},
  time_step::Real,
  tol1::Real,
  tol2::Real,
  smoothness_duration::Integer,
  smooth_counters::Array{Int8,N},
  is_smooth::Array{Bool,N},
  has_decreased::Array{Bool,N},
  stable_duration::Integer,
  stable_counters::Array{Int8,N},
  is_stable::Array{Bool,N},
) where {N,T<:Real}
  @inbounds @simd for i in eachindex(vmr_current)
    if !is_smooth[i]
      is_smooth[i], smooth_counters[i] = find_smoothness(
        vmr_current[i],
        vmr_prev1[i],
        vmr_prev2[i],
        tol2,
        time_step,
        smooth_counters[i],
        smoothness_duration
      )

      if (vmr_current[i] > vmr_prev1[i]) && (vmr_prev2[i] > vmr_prev1[i])
        density[i] = means[i]
      end

    elseif !has_decreased[i]
      has_decreased[i] = find_decrease(
        vmr_current[i],
        vmr_prev1[i],
      )

    elseif !is_stable[i]
      is_stable[i], stable_counters[i] = find_stability(
        vmr_current[i],
        vmr_prev1[i],
        vmr_prev2[i],
        tol1,
        tol2,
        time_step,
        stable_counters[i],
        stable_duration
      )

      if is_stable[i]
        density[i] = means[i]
      end

    end
  end
end
function identify_convergence!(
  ::Val{:threaded},
  vmr_current::AbstractArray{T,N},
  vmr_prev1::AbstractArray{T,N},
  vmr_prev2::AbstractArray{T,N},
  density::AbstractArray{T,N},
  means::AbstractArray{T,N},
  time_step::Real,
  tol1::Real,
  tol2::Real,
  smoothness_duration::Integer,
  smooth_counters::Array{Int8,N},
  is_smooth::Array{Bool,N},
  has_decreased::Array{Bool,N},
  stable_duration::Integer,
  stable_counters::Array{Int8,N},
  is_stable::Array{Bool,N},
) where {N,T<:Real}
  Threads.@threads @inbounds for i in eachindex(vmr_current)
    if !is_smooth[i]
      is_smooth[i], smooth_counters[i] = find_smoothness(
        vmr_current[i],
        vmr_prev1[i],
        vmr_prev2[i],
        tol2,
        time_step,
        smooth_counters[i],
        smoothness_duration
      )

      if (vmr_current[i] > vmr_prev1[i]) && (vmr_prev2[i] > vmr_prev1[i])
        density[i] = means[i]
      end

    elseif !has_decreased[i]
      has_decreased[i] = find_decrease(
        vmr_current[i],
        vmr_prev1[i],
      )

    elseif is_stable[i]
      is_stable[i], stable_counters[i] = find_stability(
        vmr_current[i],
        vmr_prev1[i],
        vmr_prev2[i],
        tol1,
        tol2,
        time_step,
        stable_counters[i],
        stable_duration
      )

      if is_stable[i]
        density[i] = means[i]
      end

    end
  end
end
function identify_convergence!(
  ::Val{:cuda},
  vmr_current::AnyCuArray{T,N},
  vmr_prev1::AnyCuArray{T,N},
  vmr_prev2::AnyCuArray{T,N},
  density::AnyCuArray{T,N},
  means::AnyCuArray{T,N},
  time_step::Real,
  tol1::Real,
  tol2::Real,
  smoothness_duration::Z,
  smooth_counters::AnyCuArray{Z,N},
  is_smooth::AnyCuArray{Bool,N},
  has_decreased::AnyCuArray{Bool,N},
  stable_duration::Z,
  stable_counters::AnyCuArray{Z,N},
  is_stable::AnyCuArray{Bool,N},
) where {N,T<:Real,Z<:Integer}
  n_points = length(vmr_current)

  # Stability detection
  stability_parms = CuArray{Float32}(
    [tol1, tol2, time_step, stable_duration]
  )
  kernel = @cuda launch = false kernel_stable!(
    vmr_current,
    vmr_prev1,
    vmr_prev2,
    means,
    density,
    is_smooth,
    has_decreased,
    is_stable,
    stable_counters,
    stability_parms,
  )
  config = launch_configuration(kernel.fun)
  threads = min(n_points, config.threads)
  blocks = cld(n_points, threads)

  CUDA.@sync blocking = true begin
    kernel(
      vmr_current,
      vmr_prev1,
      vmr_prev2,
      means,
      density,
      is_smooth,
      has_decreased,
      is_stable,
      stable_counters,
      stability_parms;
      threads,
      blocks
    )
  end

  # Decrease detection
  kernel = @cuda launch = false kernel_decrease!(
    vmr_current, vmr_prev1, is_smooth, has_decreased
  )
  config = launch_configuration(kernel.fun)
  threads = min(n_points, config.threads)
  blocks = cld(n_points, threads)

  CUDA.@sync blocking = true begin
    kernel(
      vmr_current,
      vmr_prev1,
      is_smooth,
      has_decreased;
      threads,
      blocks
    )
  end

  # Smoothness detection
  smooth_parms = CuArray{Float32}([tol2, time_step, smoothness_duration])
  kernel = @cuda launch = false kernel_smooth!(
    vmr_current,
    vmr_prev1,
    vmr_prev2,
    means,
    density,
    is_smooth,
    smooth_counters,
    smooth_parms
  )
  config = launch_configuration(kernel.fun)
  threads = min(n_points, config.threads)
  blocks = cld(n_points, threads)

  CUDA.@sync blocking = true begin
    kernel(
      vmr_current,
      vmr_prev1,
      vmr_prev2,
      means,
      density,
      is_smooth,
      smooth_counters,
      smooth_parms;
      threads,
      blocks
    )
  end

  return nothing
end

@inline function find_smoothness(
  vmr_current::T,
  vmr_prev1::T,
  vmr_prev2::T,
  tol2::Ttol,
  dt::P,
  smooth_counter::Z,
  smoothness_duration::Z,
) where {T<:Real,Ttol<:Real,P<:Real,Z<:Integer}
  second_derivative = second_difference(
    vmr_current, vmr_prev1, vmr_prev2, dt
  )

  is_smooth = false
  if smoothness_check(second_derivative, tol2)
    if smooth_counter >= smoothness_duration
      is_smooth = true
    else
      smooth_counter += Z(1)
    end
  else
    smooth_counter = Z(0)
  end

  return is_smooth, smooth_counter
end

@inline function smoothness_check(
  second_derivative::T1,
  tol::T2,
) where {T1<:Real,T2<:Real}
  factor = 2
  return abs(second_derivative) < factor * tol
end

function kernel_smooth!(
  vmr_current::CuDeviceArray{T,N},
  vmr_prev1::CuDeviceArray{T,N},
  vmr_prev2::CuDeviceArray{T,N},
  means::CuDeviceArray{T,N},
  density::CuDeviceArray{T,N},
  is_smooth::CuDeviceArray{Bool,N},
  smoothness_counter::CuDeviceArray{Z,N},
  parms::CuDeviceArray{Float32,1},
) where {T<:Real,N,Z<:Integer}
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  if idx > length(vmr_current)
    return
  elseif is_smooth[idx] == true
    return
  end

  tol = parms[1]
  dt = parms[2]
  smoothness_duration = Z(parms[3])
  factor = 2i32

  second_derivative = (
    (vmr_current[idx] - 2i32 * vmr_prev1[idx] + vmr_prev2[idx]) / dt^2i32
  )
  counter = smoothness_counter[idx]

  if abs(second_derivative) < factor * tol
    if counter >= smoothness_duration
      is_smooth[idx] = true
    else
      counter += Z(1)
    end
  else
    counter = Z(0)
  end

  smoothness_counter[idx] = counter

  if (vmr_current[idx] > vmr_prev1[idx]) && (vmr_prev2[idx] > vmr_prev1[idx])
    density[idx] = means[idx]
  end

  return
end

@inline function find_decrease(
  vmr_current::T,
  vmr_prev1::T,
) where {T<:Real}
  vmr_diff = vmr_current - vmr_prev1
  if vmr_diff < 0
    has_decreased = true
  else
    has_decreased = false
  end

  return has_decreased
end

function kernel_decrease!(
  vmr_current::CuDeviceArray{T,N},
  vmr_prev1::CuDeviceArray{T,N},
  is_smooth::CuDeviceArray{Bool,N},
  has_decreased::CuDeviceArray{Bool,N},
) where {T<:Real,N}
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  if idx > length(vmr_current)
    return
  elseif is_smooth[idx] == false
    return
  elseif has_decreased[idx] == true
    return
  end

  vmr_diff = vmr_current[idx] - vmr_prev1[idx]
  if vmr_diff < 0
    has_decreased[idx] = true
  end

  return
end

@inline function find_stability(
  vmr_current::T,
  vmr_prev1::T,
  vmr_prev2::T,
  tol1::Ttol,
  tol2::Ttol,
  dt::P,
  stability_counter::Z,
  stability_duration::Z,
) where {T<:Real,Ttol<:Real,P<:Real,Z<:Integer}
  first_derivative = abs(first_difference(vmr_current, vmr_prev1, dt))
  second_derivative = abs(second_difference(vmr_current, vmr_prev1, vmr_prev2, dt))

  is_stable = false
  if (first_derivative < tol1) && (second_derivative < tol2)
    if stability_counter >= stability_duration
      is_stable = true
    else
      stability_counter += Z(1)
    end
  else
    stability_counter = Z(0)
  end

  return is_stable, stability_counter
end

function kernel_stable!(
  vmr_current::CuDeviceArray{T,N},
  vmr_prev1::CuDeviceArray{T,N},
  vmr_prev2::CuDeviceArray{T,N},
  means::CuDeviceArray{T,N},
  density::CuDeviceArray{T,N},
  is_smooth::CuDeviceArray{Bool,N},
  has_decreased::CuDeviceArray{Bool,N},
  is_stable::CuDeviceArray{Bool,N},
  stability_counter::CuDeviceArray{Z,N},
  parms::CuDeviceArray{Float32,1},
) where {T<:Real,N,Z<:Integer}
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  if idx > length(vmr_current)
    return
  elseif is_smooth[idx] == false
    return
  elseif has_decreased[idx] == false
    return
  elseif is_stable[idx] == true
    return
  end

  tol1 = parms[1]
  tol2 = parms[2]
  dt = parms[3]
  stable_duration = parms[4]

  first_derivative = abs(vmr_current[idx] - vmr_prev1[idx]) / (2i32 * dt)
  second_derivative = abs(
    vmr_current[idx] - 2i32 * vmr_prev1[idx] + vmr_prev2[idx]
  ) / dt^2i32
  counter = stability_counter[idx]

  if (first_derivative < tol1) && (second_derivative < tol2)
    if counter >= stable_duration
      is_stable[idx] = true
      density[idx] = means[idx]
    else
      counter += Z(1)
    end
  else
    counter = Z(0)
  end
  stability_counter[idx] = counter

  return
end

@inline function first_difference(
  f::T1, f_prev::T2, dt::P
) where {T1<:Real,T2<:Real,P<:Real}
  return (f - f_prev) / (2dt)
end

@inline function second_difference(
  f::T1, f_prev1::T2, f_prev2::T3, dt::P
) where {T1<:Real,T2<:Real,T3<:Real,P<:Real}
  return (f - 2f_prev1 + f_prev2) / dt^2
end

function silverman_rule(data::AbstractMatrix{T}) where {T<:Real}
  n_dims, n_samples = size(data)

  iqrs = ntuple(i -> iqr(selectdim(data, 1, i)), n_dims)
  stds = ntuple(i -> std(selectdim(data, 1, i)), n_dims)

  return min.(stds, iqrs ./ 1.34) .* (n_samples * (n_dims + 2) / 4)^(-1 / (n_dims + 4))
end

end
