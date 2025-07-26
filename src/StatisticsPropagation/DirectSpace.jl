module DirectSpace

using ..Devices
using ..Grids
using ..KDEs

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
  calculate_full_means!,
  identify_convergence!

include("../CuStatistics/CuStatistics.jl")

function initialize_dirac_sequence(
  data::Union{AbstractMatrix{<:Real},AbstractVector{<:AbstractVector{<:Real}}};
  grid::Union{AbstractGrid{N,<:Real,M},Nothing}=nothing,
  bootstrap_idxs::Union{AbstractMatrix{<:Integer},Nothing}=nothing,
  device=:cpu,
  method=nothing,
  include_var=false,
  T=Float64,
) where {N,M}
  if method === nothing
    method = Devices.DEFAULT_IMPLEMENTATIONS[get_device(device)]
  end
  ensure_valid_implementation(device, method)
  if grid === nothing
    grid = find_grid(data; device)
  end

  if (get_device(device) isa IsCUDA) && !CUDA.functional()
    @warn "No functional CUDA detected. Falling back to ':cpu'."
    device = :cpu
  end

  if get_device(device) isa IsCPU
    if data isa AbstractMatrix
      data = Vector{SVector{N,T}}(eachcol(data))
    else
      data = Vector{SVector{N,T}}(data)
    end
    n_samples = length(data)
  else
    if !(data isa AbstractMatrix)
      data = reduce(hcat, data)
    end
    n_samples = size(data, 2)
  end

  if bootstrap_idxs === nothing
    bootstrap_idxs = zeros(Int, n_samples, 1)
    bootstrap_idxs[:, 1] = 1:n_samples
  end

  if get_device(device) isa IsCUDA
    data = CUDA.CuArray(data)
    bootstrap_idxs = CUDA.CuArray(bootstrap_idxs)
  end

  return initialize_dirac_sequence(
    Val(method),
    data,
    grid,
    bootstrap_idxs;
    include_var=include_var,
    T=T,
  )
end
function initialize_dirac_sequence(
  ::Val{:serial},
  data::AbstractVector{<:SVector{N,<:Real}},
  grid::AbstractGrid{N,<:Real,M},
  bootstrap_idxs::AbstractMatrix{<:Integer};
  include_var=false,
  T=Float64,
) where {N,M}
  grid_size = size(grid)
  n_bootstraps = size(bootstrap_idxs, 2)

  dirac_sequence = zeros(Complex{T}, grid_size..., n_bootstraps)
  dirac_sequence_real = selectdim(reinterpret(reshape, T, dirac_sequence), 1, 1)

  spacing = spacings(grid)
  low_bound = low_bounds(grid)

  check_data(data, grid)

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
  data::AbstractVector{<:SVector{N,<:Real}},
  grid::AbstractGrid{N,<:Real,M},
  bootstrap_idxs::AbstractMatrix{<:Integer};
  include_var=false,
  T=Float64,
) where {N,M}
  grid_size = size(grid)
  n_bootstraps = size(bootstrap_idxs, 2)

  dirac_sequence = zeros(Complex{T}, grid_size..., n_bootstraps)
  dirac_sequence_real = selectdim(reinterpret(reshape, T, dirac_sequence), 1, 1)

  spacing = spacings(grid)
  low_bound = low_bounds(grid)

  check_data(data, grid)

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
  data::AbstractVector{<:AbstractVector{<:Real}},
  spacing::AbstractVector{<:Real},
  low_bound::AbstractVector{<:Real},
) where {N,T<:Real}
  n_samples = length(data)
  spacing_squared = prod(spacing)^2

  indices_l = @MVector zeros(Int, N)
  indices_h = @MVector zeros(Int, N)
  remainder_l = @MVector zeros(T, N)
  remainder_h = @MVector zeros(T, N)

  mask = falses(N)
  grid_indices = @MVector zeros(Int, N)

  for sample in data
    @inbounds @simd for i in 1:N
      sample_i = sample[i]
      low_bound_i = low_bound[i]
      spacing_i = spacing[i]
      delta_i = sample_i - low_bound_i
      index_l = floor(Int, delta_i / spacing_i)
      indices_l[i] = index_l + 1
      remainder_h[i] = delta_i % spacing_i
      indices_h[i] = indices_l[i] + 1
      remainder_l[i] = spacing_i - remainder_h[i]
    end

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
  data::AbstractVector{<:AbstractVector{<:Real}},
  spacing::AbstractVector{<:Real},
  low_bound::AbstractVector{<:Real},
) where {N,T<:Real}
  n_samples = length(data)
  spacing_squared = prod(spacing)^2

  indices_l = @MVector zeros(Int, N)
  indices_h = @MVector zeros(Int, N)
  remainder_l = @MVector zeros(T, N)
  remainder_h = @MVector zeros(T, N)

  mask = falses(N)
  grid_indices = @MVector zeros(Int, N)

  for sample in data
    @inbounds @simd for i in 1:N
      sample_i = sample[i]
      low_bound_i = low_bound[i]
      spacing_i = spacing[i]
      delta_i = sample_i - low_bound_i
      index_l = floor(Int, delta_i / spacing_i)
      indices_l[i] = index_l + 1
      remainder_h[i] = delta_i % spacing_i
      indices_h[i] = indices_l[i] + 1
      remainder_l[i] = spacing_i - remainder_h[i]
    end

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
  data::CuMatrix{<:Real},
  grid::AbstractGrid{N,<:Real,M},
  bootstrap_idxs::CuMatrix{<:Integer};
  include_var=false,
  T=Float32,
) where {N,M}
  grid_size = size(grid)
  n_samples, n_bootstraps = size(bootstrap_idxs)

  dirac_sequence = CUDA.zeros(Complex{T}, grid_size..., n_bootstraps)
  dirac_sequence_real = view(reinterpret(reshape, T, dirac_sequence), 1, fill(Colon(), M)...)

  spacing = spacings(grid)
  low_bound = low_bounds(grid)

  check_data(data, grid)

  n_modified_gridpoints = n_samples * 2^N * n_bootstraps

  if include_var
    dirac_sequence_squared = CUDA.zeros(Complex{T}, grid_size..., n_bootstraps)
    dirac_sequence_squared_real = selectdim(
      reinterpret(reshape, T, dirac_sequence_squared), 1, 1
    )
    kernel = @cuda maxregs = 32 launch = false generate_dirac_cuda!(
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
        dirac_sequence_real,
        dirac_sequence_squared_real,
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
        dirac_sequence_real,
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
  dirac_series::Union{CuDeviceArray{<:Real,M},SubArray{<:Real,M,<:CuDeviceArray}},
  dirac_series_squared::Union{CuDeviceArray{<:Real,M},SubArray{<:Real,M,<:CuDeviceArray}},
  data::CuDeviceMatrix{<:Real},
  bootstrap_idxs::CuDeviceMatrix{<:Integer},
  spacing::CuDeviceVector{<:Real},
  low_bound::CuDeviceVector{<:Real}
) where {M}
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
      index_l = floor(Int32, (data[i, boot_idx] - low_bound[i]) / spacing[i]) + 1i32
      return mask[i] ? index_l : index_l + 1i32
    end,
    n_dims
  )

  remainder_product = 1.0f0
  i = 1i32
  @inbounds while i <= n_dims
    remainder_h = (
      data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]
    ) % spacing[i]
    remainder_l = spacing[i] - remainder_h

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
  dirac_series::Union{CuDeviceArray{<:Real,M},SubArray{<:Real,M,<:CuDeviceArray}},
  data::CuDeviceMatrix{<:Real},
  bootstrap_idxs::CuDeviceMatrix{<:Integer},
  spacing::CuDeviceVector{<:Real},
  low_bound::CuDeviceVector{<:Real}
) where {M}
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
      index_l = floor(Int32, (data[i, boot_idx] - low_bound[i]) / spacing[i]) + 1i32
      return mask[i] ? index_l : index_l + 1i32
    end,
    n_dims
  )

  remainder_product = 1.0f0
  i = 1i32
  @inbounds while i <= n_dims
    remainder_h = (
      data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]
    ) % spacing[i]
    remainder_l = spacing[i] - remainder_h

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

function check_data(
  data::AbstractVector{<:AbstractVector{<:Real}}, grid::Grid{N,<:Real,M}
) where {N,M}
  @assert length(data[1]) == N "Dimensions of data and grid don't match."

  lows = low_bounds(grid)
  highs = high_bounds(grid)

  for point in data
    @inbounds @simd for j in 1:N
      if (point[j] >= highs[j]) | (point[j] <= lows[j])
        throw(
          ArgumentError("Data points lie beyond grid. Remove them or increase grid boundaries.")
        )
      end
    end
  end

  return nothing
end
function check_data(data::CuMatrix{<:Real}, grid::CuGrid{N,<:Real,M}) where {N,M}
  @assert size(data, 1) == N "Dimensions of data and grid don't match."

  lows = low_bounds(grid)
  highs = high_bounds(grid)

  if any((lows .>= data) .| (highs .<= data))
    throw(
      ArgumentError("Data points lie beyond grid. Remove them or increase grid boundaries.")
    )
  end

  return nothing
end

function calculate_scaled_vmr!(
  ::Val{:serial},
  sk::AbstractArray{Complex{T},M},
  s2k::AbstractArray{Complex{T},M},
  time::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real},
  n_samples::Integer,
) where {M,T<:Real}
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
    vmr_var[j] = ifelse(isfinite(vmr_v), log(vmr_v), NaN)
  end

  return nothing
end
function calculate_scaled_vmr!(
  ::Val{:threaded},
  sk::AbstractArray{Complex{T},M},
  s2k::AbstractArray{Complex{T},M},
  time::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real},
  n_samples::Integer,
) where {M,T<:Real}
  array_dims = size(sk)
  grid_dims = array_dims[begin:end-1]

  n_elements = prod(array_dims)
  n_bootstraps = array_dims[end]
  n_points = prod(grid_dims)

  sk_raw = vec(reinterpret(T, sk))
  s2k_raw = vec(reinterpret(T, s2k))

  # Calculate means and variances
  @inbounds Threads.@threads for i in 1:n_elements
    sk_real = sk_raw[2i-1]
    sk_imag = sk_raw[2i]
    s2k_real = s2k_raw[2i-1]
    s2k_imag = s2k_raw[2i]

    sk_i = sqrt(sk_real^2 + sk_imag^2) / n_samples
    s2k_i = sqrt(s2k_real^2 + s2k_imag^2) / n_samples - sk_i^2
    s2k_raw[2i-1] = s2k_i / sk_i
  end

  # Re-arrange into ucontiguous memory
  total_written = 0
  wave = 0
  while total_written < (n_elements - 1)
    step = 2^(wave + 1)
    start = 2^wave + 1

    @inbounds Threads.@threads for i in start:step:n_elements
      sk_raw[i] = sk_raw[2i-1]
      s2k_raw[i] = s2k_raw[2i-1]
    end

    total_written += length(start:step:n_elements)
    wave += 1
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
  scaling_factor = prod(time .^ 2 + time_initial .^ 2)^(3 / 2) * n_samples^4

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
    vmr_var[point+1] = ifelse(isfinite(vmr_v), log(vmr_v), NaN)
  end

  return nothing
end
function calculate_scaled_vmr!(
  ::Val{:cuda},
  sk::AnyCuArray{<:Complex,M},
  s2k::AnyCuArray{<:Complex,M},
  time::AnyCuVector{<:Real},
  time_initial::AnyCuVector{<:Real},
  n_samples::Integer,
) where {M}
  scaling_factor = prod(time .^ 2i32 .+ time_initial .^ 2i32)^(1.5f0) * n_samples^4i32

  @. sk = abs(sk) / n_samples
  @. s2k = abs(s2k) / n_samples - sk^2

  @. s2k /= sk
  vmr = selectdim(s2k, M, 1)
  vmr .= dropdims(var(s2k, dims=M), dims=M)
  vmr .*= scaling_factor

  @. vmr = ifelse(isfinite(vmr), log(vmr), NaN32)

  return nothing
end

function calculate_full_means!(
  ::Val{:serial},
  sk::AbstractArray{Complex{T},N},
  n_samples::Integer,
) where {N,T<:Real}
  sk_raw = vec(reinterpret(T, sk))
  n_elements = length(sk)

  @inbounds @simd for i in 1:n_elements
    sk_real = sk_raw[2i-1]
    sk_imag = sk_raw[2i]

    sk_i = sqrt(sk_real^2 + sk_imag^2)
    sk_raw[i] = sk_i
  end

  return nothing
end
function calculate_full_means!(
  ::Val{:threaded},
  sk::AbstractArray{Complex{T},N},
  n_samples::Integer,
) where {N,T<:Real}
  sk_raw = vec(reinterpret(T, sk))
  n_elements = length(sk)

  # Calculate means
  @inbounds Threads.@threads for i in 1:n_elements
    sk_real = sk_raw[2i-1]
    sk_imag = sk_raw[2i]

    sk_i = sqrt(sk_real^2 + sk_imag^2)
    sk_raw[2i-1] = sk_i
  end

  # Re-arrange into ucontiguous memory
  total_written = 0
  wave = 0
  while total_written < (n_elements - 1)
    step = 2^(wave + 1)
    start = 2^wave + 1

    @inbounds Threads.@threads for i in start:step:n_elements
      sk_raw[i] = sk_raw[2i-1]
    end

    total_written += length(start:step:n_elements)
    wave += 1
  end

  return nothing
end
function calculate_full_means!(::Val{:cuda}, sk::AnyCuArray, n_samples::Integer)
  @. sk = abs(sk)

  return nothing
end

function identify_convergence!(
  ::Val{:serial},
  density::AbstractArray{<:Real,N},
  means::AbstractArray{<:Real,N},
  vmr_current::AbstractArray{<:Real,N},
  vmr_prev1::AbstractArray{<:Real,N},
  vmr_prev2::AbstractArray{<:Real,N},
  dlogt::Real,
  tol::Real,
  average_factor::Real,
  normalization1::AbstractArray{<:Real,N},
  normalization2::AbstractArray{<:Real,N},
  stable_counters::AbstractArray{<:Integer,N},
  stable_duration::Real,
) where {N}
  @inbounds @simd for i in eachindex(vmr_current)
    if !isnan(density[i])
      results = find_stability(
        vmr_current[i],
        vmr_prev1[i],
        vmr_prev2[i],
        normalization1[i],
        normalization2[i],
        stable_counters[i],
        dlogt,
        tol,
        average_factor,
        stable_duration,
      )
      normalization1[i] = results.norm1
      normalization2[i] = results.norm2
      stable_counters[i] = results.counter

      if results.is_stable
        density[i] = means[i]
      end

    end
  end

  return nothing
end
function identify_convergence!(
  ::Val{:threaded},
  density::AbstractArray{<:Real,N},
  means::AbstractArray{<:Real,N},
  vmr_current::AbstractArray{<:Real,N},
  vmr_prev1::AbstractArray{<:Real,N},
  vmr_prev2::AbstractArray{<:Real,N},
  dlogt::Real,
  tol::Real,
  average_factor::Real,
  normalization1::AbstractArray{<:Real,N},
  normalization2::AbstractArray{<:Real,N},
  stable_counters::AbstractArray{<:Integer,N},
  stable_duration::Integer,
) where {N}
  Threads.@threads for i in eachindex(vmr_current)
    if !isnan(density[i])
      results = find_stability(
        vmr_current[i],
        vmr_prev1[i],
        vmr_prev2[i],
        normalization1[i],
        normalization2[i],
        stable_counters[i],
        dlogt,
        tol,
        average_factor,
        stable_duration,
      )

      normalization1[i] = results.norm1
      normalization2[i] = results.norm2
      stable_counters[i] = results.counter

      if results.is_stable
        density[i] = means[i]
      end

    end
  end

  return nothing
end
function identify_convergence!(
  ::Val{:cuda},
  density::AnyCuArray{<:Real,N},
  means::AnyCuArray{<:Real,N},
  vmr_current::AnyCuArray{<:Real,N},
  vmr_prev1::AnyCuArray{<:Real,N},
  vmr_prev2::AnyCuArray{<:Real,N},
  dlogt::Real,
  tol::Real,
  average_factor::Real,
  normalization1::AnyCuArray{<:Real,N},
  normalization2::AnyCuArray{<:Real,N},
  stable_counters::AnyCuArray{<:Integer,N},
  stable_duration::Integer,
) where {N}
  n_points = length(vmr_current)

  # Stability detection
  stability_parms = CuArray{Float32}(
    [tol, average_factor, dlogt, stable_duration]
  )
  kernel = @cuda launch = false kernel_stable!(
    vmr_current,
    vmr_prev1,
    vmr_prev2,
    normalization1,
    normalization2,
    means,
    density,
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
      normalization1,
      normalization2,
      means,
      density,
      stable_counters,
      stability_parms;
      threads,
      blocks
    )
  end

  return nothing
end

@inline function find_stability(
  vmr_current::Real,
  vmr_prev1::Real,
  vmr_prev2::Real,
  normalization1::Real,
  normalization2::Real,
  stability_counter::Integer,
  dlogt::Real,
  tol::Real,
  average_factor::Real,
  stability_duration::Integer,
)
  first_derivative = first_difference(vmr_current, vmr_prev1, dlogt)
  normalization1 = average_factor * normalization1 + (1 - average_factor) * first_derivative
  second_derivative = second_difference(vmr_current, vmr_prev1, vmr_prev2, dlogt)
  normalization2 = average_factor * normalization2 + (1 - average_factor) * second_derivative

  indicator = sqrt((first_derivative / normalization1)^2 + (second_derivative / normalization2)^2)

  is_stable = false
  if indicator < tol
    if stability_counter >= stability_duration
      is_stable = true
    else
      stability_counter += one(stability_counter)
    end
  else
    stability_counter = zero(stability_counter)
  end

  return (is_stable=is_stable, norm1=normalization1, norm2=normalization2, counter=stability_counter)
end

function kernel_stable!(
  vmr_current::Union{CuDeviceArray{T,N},SubArray{T,N,<:CuDeviceArray}},
  vmr_prev1::CuDeviceArray{T,N},
  vmr_prev2::CuDeviceArray{T,N},
  normalization1::CuDeviceArray{T,N},
  normalization2::CuDeviceArray{T,N},
  means::Union{CuDeviceArray{T,N},SubArray{T,N,<:CuDeviceArray}},
  density::CuDeviceArray{T,N},
  stability_counter::CuDeviceArray{Z,N},
  parms::CuDeviceArray{Float32,1},
) where {T<:Real,N,Z<:Integer}
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  if idx > length(vmr_current)
    return
  elseif !isnan(density[idx])
    return
  end

  tol = parms[1]
  average_factor = parms[2]
  dlogt = parms[3]
  stable_duration = Z(parms[4])

  vmr_i = vmr_current[idx]
  vmr_im1 = vmr_prev1[idx]
  vmr_im2 = vmr_prev2[idx]

  first_derivative = log(abs(vmr_i - vmr_im1) / (2i32 * dlogt))
  norm1 = average_factor * normalization1[idx] + (1 - average_factor) * first_derivative
  second_derivative = log(abs(vmr_i - 2i32 * vmr_im1 + vmr_im2) / dlogt^2i32)
  norm2 = average_factor * normalization2[idx] + (1 - average_factor) * second_derivative

  indicator = sqrt((first_derivative / norm1)^2 + (second_derivative / norm2)^2)
  counter = stability_counter[idx]

  if indicator < tol
    if counter >= stable_duration
      density[idx] = means[idx]
    else
      counter += one(Z)
    end
  else
    counter = zero(Z)
  end

  normalization1[idx] = norm1
  normalization2[idx] = norm2
  stability_counter[idx] = counter

  return
end

@inline function first_difference(f::Real, f_prev::Real, dt::Real)
  return log(abs((f - f_prev) / (2dt)))
end

@inline function second_difference(f::Real, f_prev1::Real, f_prev2::Real, dt::Real)
  return log(abs((f - 2f_prev1 + f_prev2) / dt^2))
end

function silverman_rule(data::AbstractMatrix)
  n_dims, n_samples = size(data)

  covariance = cov(data, dims=2)
  stds = sqrt.(diag(covariance))

  prefactor = (4 / ((n_dims + 2) * n_samples))^(1 / (n_dims + 4))

  return prefactor .* stds
end

end
