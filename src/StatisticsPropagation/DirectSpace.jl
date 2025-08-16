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

"""
    initialize_dirac_sequence(data; kwargs...)

Create a Dirac sequence based on the provided data points.

# Arguments
- `data`: A matrix or vector of data points, where each column represents a sample.
- `grid`: (optional) A grid object defining the grid on which the Dirac sequence is initialized.
- `bootstrap_idxs`: (optional) A matrix of indices for bootstrap resampling.
- `device`: (optional) The device type to use for computation, default is `:cpu`.
- `method`: (optional) The method to use for computation, default is determined by the device.
- `include_var`: (optional) If `true`, includes variance in the Dirac sequence, default is `false`.
- `T`: (optional) The type of the elements in the Dirac sequence, default is `Float64` for CPU and `Float32` for CUDA.
"""
function initialize_dirac_sequence(
  data::Union{AbstractMatrix{<:Real},AbstractVector{<:AbstractVector{<:Real}}};
  grid::Union{AbstractGrid{N,<:Real,M},Nothing}=nothing,
  bootstrap_idxs::Union{AbstractMatrix{<:Integer},Nothing}=nothing,
  device=:cpu,
  method=nothing,
  include_var=false,
  T=nothing,
) where {N,M}
  if method === nothing
    method = Devices.DEFAULT_IMPLEMENTATIONS[get_device(device)]
  end
  ensure_valid_implementation(device, method)
  if grid === nothing
    grid = find_grid(data; device)
  end
  if T === nothing
    element_type = ifelse(device == :cpu, Float64, Float32)
  else
    element_type = T
  end

  if (get_device(device) isa IsCUDA) && !CUDA.functional()
    @warn "No functional CUDA detected. Falling back to ':cpu'."
    device = :cpu
  end

  if get_device(device) isa IsCPU
    if data isa AbstractMatrix
      data = Vector{SVector{N,element_type}}(eachcol(data))
    else
      data = Vector{SVector{N,element_type}}(data)
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
    T=element_type,
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

"""
    calculate_scaled_vmr!(
      method::Val{device},
      sk::AbstractArray{Complex{T},M},
      s2k::AbstractArray{Complex{T},M},
      time::AbstractVector{<:Real},
      time_initial::AbstractVector{<:Real},
      n_samples::Integer
    )

Calculate the scaled variance-to-mean ratio (VMR) for an array of kernel means and kernel variances.
"""
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

  scaling_factor = prod(time .^ 2 .+ time_initial .^ 2)^(3 / 2) * n_samples^3

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
  scaling_factor = prod(time .^ 2 + time_initial .^ 2)^(3 / 2) * n_samples^3

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
  scaling_factor = prod(time .^ 2i32 .+ time_initial .^ 2i32)^(1.5f0) * n_samples^3i32

  @. sk = abs(sk) / n_samples
  @. s2k = abs(s2k) / n_samples - sk^2

  @. s2k /= sk
  vmr = selectdim(s2k, M, 1)
  vmr .= dropdims(var(s2k, dims=M), dims=M)
  vmr .*= scaling_factor

  n_dims = M - 1
  @. vmr = ifelse(isfinite(vmr), log(vmr), NaN32)

  return nothing
end

"""
    calculate_full_means!(method::Val{Symbol}, sk::AbstractArray{Complex{T},N}, n_samples::Integer)

Calculate the means of the kernels of the full sample set.
"""
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

"""
    identify_convergence!

Identify the points in the grid that have converged based on the variance-to-mean ratio (VMR)
and update the density accordingly.

# Arguments
- `Val(Symbol)`: The method type, e.g., `:serial`, `:threaded`, or `:cuda`.
- `density`: The density array to be updated.
- `means`: The means array corresponding to the density.
- `vmr_current`: The current VMR values.
- `vmr_prev1`: The previous VMR values.
- `vmr_prev2`: The VMR values from two steps back.
- `dlogt`: The logarithmic time step.
- `tol_high`: The tolerance for convergence of high density regions.
- `tol_low_id`: The tolerance to identify low density regions.
- `tol_low`: The tolerance for convergence of low density regions.
- `alpha`: The weighting factor for the first and second derivatives.
- `threshold_crossing_steps`: The number of steps to consider for threshold crossing.
- `current_minima`: The current minima array to be updated.
- `threshold_counter`: The counter for threshold crossings.
- `low_density_flags`: Flags indicating low density regions.
"""
function identify_convergence!(
  ::Val{:serial},
  density::AbstractArray{<:Real,N},
  means::AbstractArray{<:Real,N},
  vmr_current::AbstractArray{T,N},
  vmr_prev1::AbstractArray{T,N},
  vmr_prev2::AbstractArray{T,N},
  dlogt::Real,
  tol_high::Real,
  tol_low_id::Real,
  tol_low::Real,
  alpha::Real,
  threshold_crossing_steps::Integer,
  current_minima::AbstractArray{T,N},
  threshold_counter::AbstractArray{<:Integer,N},
  low_density_flags::AbstractArray{Bool,N},
) where {T<:Real,N}
  @inbounds @simd for i in eachindex(vmr_current)
    results = find_stability(
      vmr_current[i],
      vmr_prev1[i],
      vmr_prev2[i],
      dlogt,
      tol_high,
      tol_low_id,
      tol_low,
      alpha,
      threshold_crossing_steps,
      current_minima[i],
      threshold_counter[i],
      low_density_flags[i],
    )
    threshold_counter[i] = results.counter
    low_density_flags[i] = results.low_density

    if results.more_stable
      current_minima[i] = results.new_minimum
      density[i] = means[i]
    end

  end

  return nothing
end
function identify_convergence!(
  ::Val{:threaded},
  density::AbstractArray{<:Real,N},
  means::AbstractArray{<:Real,N},
  vmr_current::AbstractArray{T,N},
  vmr_prev1::AbstractArray{T,N},
  vmr_prev2::AbstractArray{T,N},
  dlogt::Real,
  tol_high::Real,
  tol_low_id::Real,
  tol_low::Real,
  alpha::Real,
  threshold_crossing_steps::Integer,
  current_minima::AbstractArray{T,N},
  threshold_counter::AbstractArray{<:Integer,N},
  low_density_flags::AbstractArray{Bool,N},
) where {T<:Real,N}
  Threads.@threads for i in eachindex(vmr_current)
    results = find_stability(
      vmr_current[i],
      vmr_prev1[i],
      vmr_prev2[i],
      dlogt,
      tol_high,
      tol_low_id,
      tol_low,
      alpha,
      threshold_crossing_steps,
      current_minima[i],
      threshold_counter[i],
      low_density_flags[i],
    )
    threshold_counter[i] = results.counter
    low_density_flags[i] = results.low_density

    if results.more_stable
      current_minima[i] = results.new_minimum
      density[i] = means[i]
    end

  end

  return nothing
end
function identify_convergence!(
  ::Val{:cuda},
  density::AnyCuArray{<:Real,N},
  means::AnyCuArray{<:Real,N},
  vmr_current::AnyCuArray{T,N},
  vmr_prev1::AnyCuArray{T,N},
  vmr_prev2::AnyCuArray{T,N},
  dlogt::Real,
  tol_high::Real,
  tol_low_id::Real,
  tol_low::Real,
  alpha::Real,
  threshold_crossing_steps::Integer,
  current_minima::AnyCuArray{T,N},
  threshold_counter::AnyCuArray{<:Integer,N},
  low_density_flags::AnyCuArray{Bool,N}
) where {T<:Real,N}
  n_points = length(vmr_current)

  # Stability detection
  stability_parms = CuArray{Float32}(
    [dlogt, tol_high, tol_low_id, tol_low, alpha, threshold_crossing_steps]
  )
  kernel = @cuda launch = false kernel_stable!(
    vmr_current,
    vmr_prev1,
    vmr_prev2,
    means,
    density,
    current_minima,
    threshold_counter,
    low_density_flags,
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
      current_minima,
      threshold_counter,
      low_density_flags,
      stability_parms;
      threads,
      blocks
    )
  end

  return nothing
end

function find_stability(
  vmr_current::Real,
  vmr_prev1::Real,
  vmr_prev2::Real,
  dlogt::Real,
  tol_high::Real,
  tol_low_id::Real,
  tol_low::Real,
  alpha::Real,
  threshold_crossing_steps::Integer,
  current_minimum::Real,
  threshold_counter::Integer,
  low_denisty_flag::Integer,
)
  first_derivative = first_difference(vmr_current, vmr_prev1, dlogt)
  second_derivative = second_difference(vmr_current, vmr_prev1, vmr_prev2, dlogt)
  indicator = alpha * log(abs(first_derivative)) + (1 - alpha) * log(abs(second_derivative))

  more_stable = false
  new_minimum = NaN

  if !low_denisty_flag
    if indicator > tol_low_id
      # Look for a sustained run above tol_low to switch on
      threshold_counter += one(threshold_counter)
      if threshold_counter > threshold_crossing_steps
        low_denisty_flag = true
        threshold_counter = zero(threshold_counter)
      end
    elseif indicator < tol_high
      # Look for a sustained run below tol_high to save minimum
      threshold_counter += one(threshold_counter)
      if threshold_counter > threshold_crossing_steps
        if (indicator < current_minimum) || isnan(current_minimum)
          more_stable = true
          new_minimum = indicator
        end
      end
    else
      threshold_counter = zero(threshold_counter)
    end
  else
    if vmr_current < tol_low
      threshold_counter += one(threshold_counter)
      if (threshold_counter > threshold_crossing_steps) && isnan(current_minimum)
        more_stable = true
        new_minimum = indicator
      end
    else
      threshold_counter = zero(threshold_counter)
    end
  end

  if isnan(new_minimum)
    new_minimum = current_minimum
  end

  return (
    more_stable=more_stable,
    new_minimum=new_minimum,
    counter=threshold_counter,
    low_density=low_denisty_flag,
  )
end

function kernel_stable!(
  vmr_current::Union{CuDeviceArray{T,N},SubArray{T,N,<:CuDeviceArray}},
  vmr_prev1::CuDeviceArray{T,N},
  vmr_prev2::CuDeviceArray{T,N},
  means::Union{CuDeviceArray{T,N},SubArray{T,N,<:CuDeviceArray}},
  density::CuDeviceArray{T,N},
  current_minima::CuDeviceArray{T,N},
  threshold_counter::CuDeviceArray{Z,N},
  low_density_flags::CuDeviceArray{Bool,N},
  parms::CuDeviceArray{Float32,1},
) where {T<:Real,N,Z<:Integer}
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  if idx > length(vmr_current)
    return
  end

  dlogt = parms[1]
  tol_high = parms[2]
  tol_low_id = parms[3]
  tol_low = parms[4]
  alpha = parms[5]
  threshold_crossing_steps = Z(parms[6])

  vmr_i = vmr_current[idx]
  vmr_im1 = vmr_prev1[idx]
  vmr_im2 = vmr_prev2[idx]

  diff1 = vmr_i - vmr_im1
  diff2 = vmr_im1 - vmr_im2
  first_derivative = log(abs(diff1)) - log(2i32 * dlogt)
  second_derivative = log(abs(diff1 - diff2)) - log(dlogt^2i32)
  indicator = alpha * first_derivative + (1.0f0 - alpha) * second_derivative

  counter = threshold_counter[idx]

  if !low_density_flags[idx]
    if indicator > tol_low_id
      # Look for a sustained run above tol_high to switch on
      counter += one(counter)
      if counter > threshold_crossing_steps
        low_density_flags[idx] = true
        counter = zero(counter)
      end
    elseif indicator < tol_high
      # Look for a sustained run below tol_low to save minimum
      counter += one(counter)
      if counter > threshold_crossing_steps
        if (indicator < current_minima[idx]) || isnan(current_minima[idx])
          density[idx] = means[idx]
          current_minima[idx] = indicator
        end
      end
    else
      counter = zero(counter)
    end
  else
    if vmr_i < tol_low
      counter += one(counter)
      if (counter > threshold_crossing_steps) && isnan(current_minima[idx])
        density[idx] = means[idx]
        current_minima[idx] = indicator
      end
    else
      counter = zero(counter)
    end
  end

  threshold_counter[idx] = counter

  return
end

@inline function first_difference(f::Real, f_prev::Real, dt::Real)
  return (f - f_prev) / (2dt)
end

@inline function second_difference(f::Real, f_prev1::Real, f_prev2::Real, dt::Real)
  return (f - 2f_prev1 + f_prev2) / dt^2
end

function silverman_rule(data::AbstractMatrix)
  n_dims, n_samples = size(data)

  covariance = cov(data, dims=2)
  stds = sqrt.(diag(covariance))

  prefactor = (4 / ((n_dims + 2) * n_samples))^(1 / (n_dims + 4))

  return prefactor .* stds
end

end
