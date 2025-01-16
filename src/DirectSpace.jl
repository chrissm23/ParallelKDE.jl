module DirectSpace

using ..Grids
using ..KDEs
using ..FourierSpace

using Statistics

using StaticArrays,
  CUDA,
  FFTW

using CUDA: i32

export initialize_dirac_series,
  mean_var_vmr!,
  calculate_variance_products!,
  assign_converged_density!

function initialize_dirac_series(
  ::Val{:serial},
  kde::KDE{N,T,S,M};
  n_bootstraps::Integer=1,
  calculate_squared::Bool=false,
)::Union{Array{T,N + 1},NTuple{2,Array{T,N + 1}}} where {N,T<:Real,S<:Real,M}
  grid_size = size(kde.grid)

  dirac_series = zeros(T, n_bootstraps, grid_size...)
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  spacing = spacings(kde.grid)
  low_bound = low_bounds(kde.grid)

  if calculate_squared
    dirac_series_squared = zeros(T, n_bootstraps, grid_size...)

    for i in 1:n_bootstraps
      generate_dirac_cpu!(
        selectdim(dirac_series, 1, i),
        selectdim(dirac_series_squared, 1, i),
        view(kde.data, bootstrap_idxs[:, i]),
        spacing,
        low_bound,
      )
    end

    return dirac_series, dirac_series_squared
  else
    for i in 1:n_bootstraps
      generate_dirac_cpu!(
        selectdim(dirac_series, 1, i),
        view(kde.data, bootstrap_idxs[:, i]),
        spacing,
        low_bound,
      )
    end

    return dirac_series
  end

end
function initialize_dirac_series(
  ::Val{:threaded},
  kde::KDE{N,T,S,M};
  n_bootstraps::Integer=1,
  calculate_squared::Bool=false,
)::Union{Array{T,N + 1},NTuple{2,Array{T,N + 1}}} where {N,T<:Real,S<:Real,M}
  grid_size = size(kde.grid)

  dirac_series = zeros(T, n_bootstraps, grid_size...)
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  spacing = spacings(kde.grid)
  low_bound = low_bounds(kde.grid)

  if calculate_squared
    dirac_series_squared = zeros(T, n_bootstraps, grid_size...)

    Threads.@threads for i in 1:n_bootstraps
      generate_dirac_cpu!(
        selectdim(dirac_series, 1, i),
        selectdim(dirac_series_squared, 1, i),
        view(kde.data, bootstrap_idxs[:, i]),
        spacing,
        low_bound,
      )
    end

    return dirac_series, dirac_series_squared
  else
    Threads.@threads for i in 1:n_bootstraps
      generate_dirac_cpu!(
        selectdim(dirac_series, 1, i),
        view(kde.data, bootstrap_idxs[:, i]),
        spacing,
        low_bound,
      )
    end

    return dirac_series
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

  products = @MArray zeros(T, fill(2, N)...)
  grid_points = Array{CartesianIndex{N}}(undef, fill(2, N)...)

  for sample in data
    @. indices_l = floor(Int64, (sample - low_bound) / spacing) + 1
    @. remainder_l = (sample - low_bound) % spacing
    @. indices_h = indices_l + 1
    @. remainder_h = spacing - remainder_l

    products .= map(prod, Iterators.product(zip(remainder_l, remainder_h)...))
    grid_points .= CartesianIndex{N}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

    dirac_series_term = products / (n_samples * spacing_squared)
    @inbounds dirac_series[grid_points] .+= dirac_series_term
    @inbounds dirac_series_squared[grid_points] .+= dirac_series_term .^ 2
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

  products = @MArray zeros(T, fill(2, N)...)
  grid_points = Array{CartesianIndex{N}}(undef, fill(2, N)...)

  for sample in data
    @. indices_l = floor(Int64, (sample - low_bound) / spacing) + 1
    @. remainder_l = (sample - low_bound) % spacing
    @. indices_h = indices_l + 1
    @. remainder_h = spacing - remainder_l

    products .= map(prod, Iterators.product(zip(remainder_l, remainder_h)...))
    grid_points .= CartesianIndex{N}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

    dirac_series_term = products / (n_samples * spacing_squared)
    @inbounds dirac_series[grid_points] .+= dirac_series_term
  end

  return nothing
end

function initialize_dirac_series(
  ::Val{:cuda},
  kde::CuKDE{N,T,S,M};
  n_bootstraps::Integer=1,
  calculate_squared::Bool=false,
)::Union{CuArray{T,N + 1},NTuple{2,CuArray{T,N + 1}}} where {N,T<:Real,S<:Real,M}
  grid_size = size(kde.grid)

  dirac_series = CUDA.zeros(T, n_bootstraps, grid_size...)
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  spacing = spacings(kde.grid)
  low_bound = low_bounds(kde.grid)

  n_samples = get_nsamples(kde)
  n_modified_gridpoints = n_samples * 2^N * n_bootstraps

  if calculate_squared
    dirac_series_squared = CUDA.zeros(T, n_bootstraps, grid_size...)
    kernel = @cuda launch = false generate_dirac_gpu!(
      dirac_series, dirac_series_squared, kde.data, bootstrap_idxs, spacing, low_bound
    )
  else
    kernel = @cuda launch = false generate_dirac_gpu!(
      dirac_series, kde.data, bootstrap_idxs, spacing, low_bound
    )
  end

  config = launch_configuration(kernel.fun)
  threads = min(n_modified_gridpoints, config.threads)
  blocks = cld(n_modified_gridpoints, threads)

  if calculate_squared
    CUDA.@sync blocking = true begin
      kernel(
        dirac_series,
        dirac_series_squared,
        kde.data,
        bootstrap_idxs,
        spacing,
        low_bound;
        threads,
        blocks
      )
    end

    return dirac_series, dirac_series_squared
  else
    CUDA.@sync blocking = true begin
      kernel(
        dirac_series,
        kde.data,
        bootstrap_idxs,
        spacing,
        low_bound;
        threads,
        blocks
      )
    end

    return dirac_series
  end

end

function generate_dirac_gpu!(
  dirac_series::CuDeviceArray{T,M},
  data::CuDeviceArray{S,2},
  bootstrap_idxs::CuDeviceArray{Int32,2},
  spacing::CuDeviceArray{T,1},
  low_bound::CuDeviceArray{T,1}
) where {M,T<:Real,S<:Real}
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  n_dims = Int32(M) - 1i32
  n_samples = size(data, 2i32)
  n_bootstraps = size(bootstrap_idxs, 2i32)
  n_modified_gridpoints = (2i32)^n_dims * n_bootstraps * n_samples

  if idx > n_modified_gridpoints
    return
  end

  idx_sample_tmp, remainder = divrem(idx - 1i32, (2i32)^n_dims * n_bootstraps)
  idx_bootstrap_tmp, idx_dim = divrem(remainder, (2i32)^n_dims)

  # idx_dim should be 0-indexed but idx_bootstrap and idx_sample should be 1-indexed
  idx_bootstrap = idx_bootstrap_tmp + 1i32
  idx_sample = idx_sample_tmp + 1i32

  bit_dim_mask = ntuple(i -> (idx_dim >> (n_dims - i)) & 1i32 == 1i32, n_dims)
  tuple_dim_mask = ntuple(
    i -> ifelse(
      bit_dim_mask[end-i+1i32],
      floor(
        Int32,
        (data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]) / spacing[i]
      ) + 1i32,
      ceil(
        Int32,
        (data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]) / spacing[i]
      ) + 1i32
    ),
    n_dims
  )

  remainder_product = 1.0f0
  i = 1i32
  @inbounds while i <= n_dims
    remainder_product *= ifelse(
      bit_dim_mask[end-i+1i32],
      (data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]) % spacing[i],
      spacing[i] - (data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]) % spacing[i]
    )

    i += 1i32
  end

  spacing_squared = prod(spacing)^2i32
  dirac_series_term = remainder_product / (n_samples * spacing_squared)

  dirac_idx = (idx_bootstrap, tuple_dim_mask...)
  @inbounds CUDA.@atomic dirac_series[dirac_idx...] += dirac_series_term

  return
end
function generate_dirac_gpu!(
  dirac_series::CuDeviceArray{T,M},
  dirac_series_squared::CuDeviceArray{T,M},
  data::CuDeviceArray{S,2},
  bootstrap_idxs::CuDeviceArray{Int32,2},
  spacing::CuDeviceArray{T,1},
  low_bound::CuDeviceArray{T,1}
) where {M,T<:Real,S<:Real}
  idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

  n_dims = Int32(M) - 1i32
  n_samples = size(data, 2i32)
  n_bootstraps = size(bootstrap_idxs, 2i32)
  n_modified_gridpoints = (2i32)^n_dims * n_bootstraps * n_samples

  if idx > n_modified_gridpoints
    return
  end

  idx_sample_tmp, remainder = divrem(idx - 1i32, (2i32)^n_dims * n_bootstraps)
  idx_bootstrap_tmp, idx_dim = divrem(remainder, (2i32)^n_dims)

  # idx_dim should be 0-indexed but idx_bootstrap and idx_sample should be 1-indexed
  idx_bootstrap = idx_bootstrap_tmp + 1i32
  idx_sample = idx_sample_tmp + 1i32

  bit_dim_mask = ntuple(i -> (idx_dim >> (n_dims - i)) & 1i32 == 1i32, n_dims)
  tuple_dim_mask = ntuple(
    i -> ifelse(
      bit_dim_mask[end-i+1i32],
      floor(
        Int32,
        (data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]) / spacing[i]
      ) + 1i32,
      ceil(
        Int32,
        (data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]) / spacing[i]
      ) + 1i32
    ),
    n_dims
  )

  remainder_product = 1.0f0
  i = 1i32
  @inbounds while i <= n_dims
    remainder_product *= ifelse(
      bit_dim_mask[end-i+1i32],
      (data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]) % spacing[i],
      spacing[i] - (data[i, bootstrap_idxs[idx_sample, idx_bootstrap]] - low_bound[i]) % spacing[i]
    )

    i += 1i32
  end

  spacing_squared = prod(spacing)^2i32
  dirac_series_term = remainder_product / (n_samples * spacing_squared)

  dirac_idx = (idx_bootstrap, tuple_dim_mask...)
  @inbounds CUDA.@atomic dirac_series[dirac_idx...] += dirac_series_term
  @inbounds CUDA.@atomic dirac_series_squared[dirac_idx...] += dirac_series_term^2i32

  return
end

function mean_var_vmr!(
  ::Val{:serial},
  means::AbstractArray{Complex{T},M},
  variances::AbstractArray{Complex{T},M},
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
  dst_vmr::AbstractArray{Complex{T},N},
)::AbstractArray{T,N} where {N,T<:Real,M}
  ifft_plan * means
  ifft_plan * variances

  n_bootstraps = size(means, 1)
  for i in 1:n_bootstraps
    vmr_cpu!(
      selectdim(means, 1, i),
      selectdim(variances, 1, i)
    )
  end

  variances_real = selectdim(reinterpret(reshape, T, variances), 1, 1)
  dst_vmr_real = selectdim(reinterpret(reshape, T, dst_vmr), 1, 1)
  dst_vmr_real .= dropdims(var(variances_real, dims=1), dims=1)

  # Variance with Inf will give NaN
  @. dst_vmr_real = ifelse(isnan(dst_vmr_real), Inf, dst_vmr_real)

  return dst_vmr_real
end
function mean_var_vmr!(
  ::Val{:threaded},
  means::AbstractArray{Complex{T},M},
  variances::AbstractArray{Complex{T},M},
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
  dst_vmr::AbstractArray{Complex{T},N},
)::AbstractArray{T,N} where {N,T<:Real,M}
  ifft_plan * means
  ifft_plan * variances

  n_bootstraps = size(means, 1)
  Threads.@threads for i in 1:n_bootstraps
    vmr_cpu!(
      selectdim(means, 1, i),
      selectdim(variances, 1, i)
    )
  end

  variances_real = selectdim(reinterpret(reshape, T, variances), 1, 1)
  dst_vmr_real = selectdim(reinterpret(reshape, T, dst_vmr), 1, 1)
  dst_vmr_real .= dropdims(var(variances_real, dims=1), dims=1)

  # Variance with Inf will give NaN
  @. dst_vmr_real = ifelse(isnan(dst_vmr_real), Inf, dst_vmr_real)

  return dst_vmr_real
end

function vmr_cpu!(
  means::AbstractArray{Complex{T},N},
  variances::AbstractArray{Complex{T},N},
)::Nothing where {N,T<:Real}
  @. means = abs(means)
  @. variances = abs(variances)

  means_real = selectdim(reinterpret(reshape, T, means), 1, 1)
  variances_real = selectdim(reinterpret(reshape, T, variances), 1, 1)
  @. variances_real /= means_real

  return nothing
end

function mean_var_vmr!(
  ::Val{:cuda},
  means::CuArray{Complex{T},M},
  variances::CuArray{Complex{T},M},
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
  dst_vmr::CuArray{Complex{T},N},
) where {N,T<:Real,M}
  ifft_plan * means
  ifft_plan * variances

  vmr_gpu!(means, variances)

  variances_real = selectdim(reinterpret(reshape, T, variances), 1, 1)
  dst_vmr_real = selectdim(reinterpret(reshape, T, dst_vmr), 1, 1)
  dst_vmr_real .= dropdims(var(variances_real, dims=1), dims=1)

  # Variance with Inf will give NaN
  @. dst_vmr_real = ifelse(isnan(dst_vmr_real), Inf, dst_vmr_real)

  return dst_vmr_real
end

function vmr_gpu!(
  means::CuArray{Complex{T},N},
  variances::CuArray{Complex{T},N},
) where {N,T<:Real}
  @. means = abs(means)
  @. variances = abs(variances)

  means_real = selectdim(reinterpret(reshape, T, means), 1, 1)
  variances_real = selectdim(reinterpret(reshape, T, variances), 1, 1)
  @. variances_real /= means_real

  return
end

# TODO: Change the determinant to include also the initial bandwidth
function calculate_variance_products!(
  ::Val{:serial},
  vmr_variance::AbstractArray{T,N},
  variance_complete::AbstractArray{Complex{T},N},
  time::AbstractVector{<:Real},
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
  dst_var_products::AbstractArray{T,N},
) where {N,T<:Real}
  ifft_plan * variance_complete
  @. variance_complete = abs(variance_complete)
  var_real = selectdim(reinterpret(reshape, T, variance_complete), 1, 1)

  det_t = prod(time .^ 2)

  dst_var_products .= vmr_variance .* var_real .* det_t^2

  return dst_var_products
end
function calculate_variance_products!(
  ::Val{:threaded},
  vmr_variance::AbstractArray{T,N},
  variance_complete::AbstractArray{Complex{T},N},
  time::AbstractVector{<:Real},
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
  dst_var_products::AbstractArray{T,N},
) where {N,T<:Real}
  @warn "Threaded calculation of variance products not implemented. Using serial implementation."

  return calculate_variance_products!(
    Val{:serial},
    vmr_variance,
    variance_complete,
    time,
    ifft_plan,
    dst_var_products
  )
end
function calculate_variance_products!(
  ::Val{:cuda},
  vmr_variance::CuArray{T,N},
  variance_complete::CuArray{Complex{T},N},
  time::CuVector{<:Real},
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
  dst_var_products::CuArray{T,N},
) where {N,T<:Real}
  ifft_plan * variance_complete
  @. variance_complete = abs(variance_complete)
  var_real = selectdim(reinterpret(reshape, T, variance_complete), 1, 1)

  det_t = prod(time .^ 2)

  dst_var_products .= vmr_variance .* var_real .* det_t^2

  return dst_var_products
end

function assign_converged_density!(
  ::Val{:serial},
  dst::AbstractArray{T,N},
  src::AbstractArray{Complex{T},N},
  indicator::AbstractArray{T,N},
  threshold::T,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
)::Nothing where {N,T<:Real}
  ifft_plan * src
  @. src = abs(src)
  src_real = selectdim(reinterpret(reshape, T, src), 1, 1)

  @. dst = ifelse(
    ((indicator < threshold) || isinf(indicator)) && !(isfinite(dst)),
    src_real,
    dst
  )

  return nothing
end
function assign_converged_density!(
  ::Val{:threaded},
  dst::AbstractArray{T,N},
  src::AbstractArray{Complex{T},N},
  indicator::AbstractArray{T,N},
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
  threshold::T,
)::Nothing where {N,T<:Real}
  @warn "Threaded assignment of converged density not implemented. Using serial implementation."
  assign_converged_density!(
    Val{:serial},
    dst,
    src,
    indicator,
    threshold,
    ifft_plan
  )

  return nothing
end
function assign_converged_density!(
  ::Val{:cuda},
  dst::CuArray{T,N},
  src::CuArray{Complex{T},N},
  indicator::CuArray{T,N},
  threshold::T,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
)::Nothing where {N,T<:Real}
  ifft_plan * src
  @. src = abs(src)
  src_real = selectdim(reinterpret(reshape, T, src), 1, 1)

  @. dst = ifelse(
    ((indicator < threshold) || isinf(indicator)) && !(isfinite(dst)),
    src_real,
    dst
  )

  return nothing
end

end
