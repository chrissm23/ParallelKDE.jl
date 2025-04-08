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
  bootstrap_idxs::AbstractMatrix{Integer};
  include_var::Bool=false,
  T=Float64,
) where {N,S<:Real,M,P<:Real}
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
  bootstrap_idxs::AbstractMatrix{Integer};
  include_var::Bool=false,
  T=Float64,
) where {N,S<:Real,M,P<:Real}
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

function mean_var_vmr!(
  ::Val{:serial},
  means::AbstractArray{T,M},
  variances::AbstractArray{T,M},
  dst_vmr::AbstractArray{T,N},
)::AbstractArray{T,N} where {N,T<:Real,M}
  n_bootstraps = size(means, 1)
  for i in 1:n_bootstraps
    vmr_cpu!(
      selectdim(means, 1, i),
      selectdim(variances, 1, i)
    )
  end

  dst_vmr .= dropdims(var(variances, dims=1), dims=1)

  # Variance with Inf will give NaN
  @. dst_vmr = ifelse(isnan(dst_vmr), NaN, dst_vmr)

  return dst_vmr
end
function mean_var_vmr!(
  ::Val{:threaded},
  means::AbstractArray{T,M},
  variances::AbstractArray{T,M},
  dst_vmr::AbstractArray{T,N},
)::AbstractArray{T,N} where {N,T<:Real,M}
  n_bootstraps = size(means, 1)
  Threads.@threads for i in 1:n_bootstraps
    vmr_cpu!(
      selectdim(means, 1, i),
      selectdim(variances, 1, i)
    )
  end

  dst_vmr .= dropdims(var(variances, dims=1), dims=1)

  # Variance with Inf will give NaN
  @. dst_vmr = ifelse(isnan(dst_vmr), NaN, dst_vmr)

  return dst_vmr
end

function vmr_cpu!(
  means::AbstractArray{T,N},
  variances::AbstractArray{T,N},
)::Nothing where {N,T<:Real}
  @. variances /= means

  return nothing
end

function mean_var_vmr!(
  ::Val{:cuda},
  means::AnyCuArray{T,M},
  variances::AnyCuArray{T,M},
  dst_vmr::AnyCuArray{T,N},
) where {N,T<:Real,M}
  vmr_gpu!(means, variances)

  dst_vmr .= dropdims(var(variances, dims=1), dims=1)

  # Variance with Inf will give NaN
  @. dst_vmr = ifelse(isnan(dst_vmr), NaN32, dst_vmr)

  return dst_vmr
end

function vmr_gpu!(
  means::AnyCuArray{T,N},
  variances::AnyCuArray{T,N},
) where {N,T<:Real}
  @. variances /= means

  return
end

function calculate_variance_products!(
  ::Val{:serial},
  vmr_variance::AbstractArray{T,N},
  time::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real},
  n_samples::Integer;
  dst_var_products::AbstractArray{T,N}=Array{T,N}(undef, size(vmr_variance)),
) where {N,T<:Real}
  det_t = prod(time .^ 2 .+ time_initial .^ 2)

  dst_var_products .= vmr_variance .* (det_t^(3 / 2) * n_samples^4)

  return dst_var_products
end
function calculate_variance_products!(
  ::Val{:threaded},
  vmr_variance::AbstractArray{T,N},
  time::AbstractVector{<:Real},
  n_samples::Integer;
  dst_var_products::AbstractArray{T,N}=Array{T,N}(undef, size(vmr_variance)),
) where {N,T<:Real}
  @warn "Threaded calculation of variance products not implemented. Using serial implementation."

  return calculate_variance_products!(
    Val{:serial},
    vmr_variance,
    time,
    n_samples,
    dst_var_products=dst_var_products
  )
end
function calculate_variance_products!(
  ::Val{:cuda},
  vmr_variance::AnyCuArray{T,N},
  time::CuVector{<:Real},
  time_initial::CuVector{<:Real},
  n_samples::Integer;
  dst_var_products::AnyCuArray{T,N}=CuArray{T,N}(undef, size(vmr_variance)),
) where {N,T<:Real}
  det_t = prod(time .^ 2 .+ time_initial .^ 2)

  dst_var_products .= vmr_variance .* (det_t^(3 / 2) * n_samples^4)

  return dst_var_products
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
