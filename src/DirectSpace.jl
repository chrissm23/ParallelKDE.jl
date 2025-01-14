module DirectSpace

using ..Grids
using ..KDEs
using ..FourierSpace

using Statistics

using StaticArrays,
  CUDA,
  FFTW

using CUDA: i32

export initialize_dirac_series, mean_var_vmr!

function initialize_dirac_series(
  ::Val{:serial},
  kde::KDE{N,T,S,M};
  n_bootstraps::Integer=1,
  calculate_squared::Bool=false,
)::Array{T,N + 1} where {N,T<:Real,S<:Real,M}
  grid_size = size(kde.grid)

  dirac_series = zeros(T, n_bootstraps, grid_size...)
  dirac_series_squared = zeros(T, n_bootstraps, grid_size...)
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  spacing = spacings(kde.grid)
  low_bound = low_bounds(kde.grid)

  for i in 1:n_bootstraps
    generate_dirac_cpu!(
      selectdim(dirac_series, 1, i),
      selectdim(dirac_series_squared, 1, i),
      view(kde.data, bootstrap_idxs[:, i]),
      spacing,
      low_bound,
      calculate_squared
    )
  end

  return dirac_series, dirac_series_squared
end
function initialize_dirac_series(
  ::Val{:threaded},
  kde::KDE{N,T,S,M};
  n_bootstraps::Integer=1,
  calculate_squared::Bool=false,
)::Array{T,N + 1} where {N,T<:Real,S<:Real,M}
  grid_size = size(kde.grid)

  dirac_series = zeros(T, n_bootstraps, grid_size...)
  dirac_series_squared = zeros(T, n_bootstraps, grid_size...)
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  spacing = spacings(kde.grid)
  low_bound = low_bounds(kde.grid)

  Threads.@threads for i in 1:n_bootstraps
    generate_dirac_cpu!(
      selectdim(dirac_series, 1, i),
      selectdim(dirac_series_squared, 1, i),
      view(kde.data, bootstrap_idxs[:, i]),
      spacing,
      low_bound,
      calculate_squared
    )
  end

  return dirac_series, dirac_series_squared
end

function generate_dirac_cpu!(
  dirac_series::Array{T,N},
  dirac_series_squared::Array{T,N},
  data::AbstractVector{SVector{N,S}},
  spacing::SVector{N,T},
  low_bound::SVector{N,T},
  calculate_squared::Bool
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
    if calculate_squared
      @inbounds dirac_series_squared[grid_points] .+= dirac_series_term .^ 2
    end
  end

  return nothing
end

function initialize_dirac_series(
  ::Val{:cuda},
  kde::CuKDE{N,T,S,M};
  n_bootstraps::Integer=1,
  calculate_squared::Bool=false,
)::Array{T,N + 1} where {N,T<:Real,S<:Real,M}
  grid_size = size(kde.grid)

  dirac_series = CUDA.zeros(T, n_bootstraps, grid_size...)
  dirac_series_squared = CUDA.zeros(T, n_bootstraps, grid_size...)
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  spacing = spacings(kde.grid)
  low_bound = low_bounds(kde.grid)

  n_samples = get_nsamples(kde)
  n_modified_gridpoints = n_samples * 2^N * n_bootstraps

  if calculate_squared
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

  dst_vmr .= var(variances, dims=1)

  return selectdim(reinterpret(reshape, T, dst_vmr), 1, 1)
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

  dst_vmr .= var(variances, dims=1)

  return selectdim(reinterpret(reshape, T, dst_vmr), 1, 1)
end

function vmr_cpu!(
  means::AbstractArray{Complex{T},N},
  variances::AbstractArray{Complex{T},N},
)::Nothing where {N,T<:Real}
  @. means = abs(means)
  @. variances = abs(variances)

  @. variances = variances / means
  return nothing
end

end
