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
  initialize_dirac_sequence

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

    vmr_var[j] = scaling_factor * m2 / (n_bootstraps - 1)
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

  # scaling_factor = prod(time .^ 2 .+ time_initial .^ 2)^(3 / 2) * n_samples^4
  #
  # Threads.@threads for i in 1:size(means, M)
  #   means_i = selectdim(means, M, i)
  #   vars_i = selectdim(vars, M, i)
  #
  #   @inbounds @simd for idx in eachindex(means_i)
  #     vars_i[idx] /= means_i[idx]
  #   end
  # end
  # Threads.@threads for idx in CartesianIndices(selectdim(means, M, 1))
  #   vars[idx] = var(vars[idx, :]) * scaling_factor
  #
  #   vars[idx] = ifelse(isnan(vars[idx]), NaN, vars[idx])
  # end
  #
  # return nothing
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

  @. vmr = ifelse(isnan(vmr), NaN32, vmr)

  return nothing
end

function assign_converged_density!(
  ::Val{:serial},
  density::AbstractArray{T,N},
  means::AbstractArray{T,N},
  vmr_current::AbstractArray{T,N},
  vmr_prev1::AbstractArray{T,N},
  vmr_prev2::AbstractArray{T,N},
  time_step::Real;
  tol1::Real=1e-10,
  tol2::Real=1e-10,
)::Nothing where {N,T<:Real}
  @. density = ifelse(
    (
      ((abs(vmr_current - vmr_prev1) / (2time_step)) < tol1) &
      ((abs(vmr_current - 2vmr_prev1 + vmr_prev2) / time_step^2) < tol2) &
      isnan(density)
    ),
    means,
    density,
  )

  # @. density = ifelse(
  #   isnan(density),
  #   means,
  #   density,
  # )

  return nothing
end
function assign_converged_density!(
  ::Val{:threaded},
  density::AbstractArray{T,N},
  means::AbstractArray{T,N},
  vmr_current::AbstractArray{T,N},
  vmr_prev1::AbstractArray{T,N},
  vmr_prev2::AbstractArray{T,N},
  time_step::Real;
  tol1::Real=1e-10,
  tol2::Real=1e-10,
)::Nothing where {N,T<:Real}
  @warn "Threaded assignment of converged density not implemented. Using serial implementation."
  assign_converged_density!(
    Val{:serial},
    density,
    means,
    vmr_current,
    vmr_prev1,
    vmr_prev1,
    time_step;
    tol1,
    tol2,
  )

  return nothing
end
function assign_converged_density!(
  ::Val{:cuda},
  density::AnyCuArray{T,N},
  means::AnyCuArray{Complex{T},N},
  vmr_current::AnyCuArray{T,N},
  vmr_prev1::AnyCuArray{T,N},
  vmr_prev2::AnyCuArray{T,N},
  time_step::Real;
  tol1::Real=1.0f-10,
  tol2::Real=1.0f-10,
)::Nothing where {N,T<:Real}
  time_step_32 = T(time_step)

  @. density = ifelse(
    (
      ((abs(vmr_current - vmr_prev1) / (2i32 * time_step_32)) < tol1) &
      ((abs(vmr_current - 2i32 * vmr_prev1 + vmr_prev2) / time_step_32^2i32) < tol2)
    ),
    means,
    density,
  )

  return nothing
end

function silverman_rule(data::AbstractMatrix{T}) where {T<:Real}
  n_dims, n_samples = size(data)

  iqrs = ntuple(i -> iqr(selectdim(data, 1, i)), n_dims)
  stds = ntuple(i -> std(selectdim(data, 1, i)), n_dims)

  return min.(stds, iqrs ./ 1.34) .* (n_samples * (n_dims + 2) / 4)^(-1 / (n_dims + 4))
end

end
