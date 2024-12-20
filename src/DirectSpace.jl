module DirectSpace

using ..Grids
using ..KDEs

using StaticArrays,
  CUDA

export initialize_dirac_series

function initialize_dirac_series(
  ::Val{:cpu},
  kde::KDE{N,T,S,M},
  n_bootstraps::Int
)::Array{T,N + 1} where {N,T<:Real,S<:Real,M}
  grid_size = size(kde.grid)

  dirac_series = zeros(T, n_bootstraps, grid_size...)
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  spacing = spacings(kde.grid)
  low_bound = low_bounds(kde.grid)

  for i in 1:n_bootstraps
    @views generate_dirac_cpu!(
      dirac_series[fill(Colon(), i, ndims(dirac_series) - 1)...],
      kde.data[bootstrap_idxs[:, i]],
      spacing,
      low_bound
    )
  end

  return dirac_series
end
function initialize_dirac_series(
  ::Val{:threaded},
  kde::KDE{N,T,S,M},
  n_bootstraps::Int
)::Array{T,N + 1} where {N,T<:Real,S<:Real,M}
  grid_size = size(kde.grid)
  spacing_squared = prod(spacings(kde.grid))^2

  dirac_series = zeros(T, n_bootstraps, grid_size...)
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  spacing = spacings(kde.grid)
  low_bound = low_bounds(kde.grid)

  Threads.@threads for i in 1:n_bootstraps
    @views generate_dirac_cpu!(
      dirac_series[fill(Colon(), i, ndims(dirac_series) - 1)...],
      kde.data[bootstrap_idxs[:, i]],
      spacing,
      low_bound
    )
  end

  return dirac_series ./ spacing_squared
end

function generate_dirac_cpu!(
  dirac_series::AbstractArray{T,N},
  data::AbstractVector{SVector{N,S}},
  spacing::SVector{N,T},
  low_bound::SVector{N,T}
) where {N,T<:Real,S<:Real}
  indices_l = @MVector zeros(Int64, N)
  indices_h = @MVector zeros(Int64, N)
  remainder_l = @MVector zeros(T, N)
  remainder_h = @MVector zeros(T, N)

  products = @MArray zeros(T, fill(2, N)...)
  grid_points = Array{CartesianIndex{N}}(undef, fill(2, N)...)

  @simd for sample in data
    @. indices_l = floor(Int64, (sample - low_bound) / spacing)
    @. remainder_l = (sample - low_bound) % spacing
    @. indices_h = indices_l + 1
    @. remainder_h = spacing - remainder_l

    products .= map(prod, Iterators.product(zip(remainder_l, remainder_h)...))
    grid_points .= CartesianIndex{N}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

    @inbounds dirac_series[grid_points] .+= products
  end
end

function initialize_dirac_series(
  ::Val{:gpu},
  kde::CuKDE{N,T,S,M},
  n_bootstraps::Int
)::Array{T,N + 1} where {N,T<:Real,S<:Real,M}
  grid_size = size(kde.grid)

  dirac_series = CUDA.zeros(T, n_bootstraps, grid_size...)
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  spacing = spacings(kde.grid)
  low_bound = low_bounds(kde.grid)

  n_samples = get_nsamples(kde)
  n_modified_gridpoints = n_samples * 2^N * n_bootstraps

  kernel = @cuda launch = false generate_dirac_gpu!(
    dirac_series, kde.data, bootstrap_idxs, spacing, low_bound
  )

  config = launch_configuration(kernel.fun)

  threads = min(n_modified_gridpoints, config.threads)
  blocks = cld(n_modified_gridpoints, threads)

  CUDA.@sync blocking = true begin
    kernel(dirac_series, kde.data, bootstrap_idxs, spacing, low_bound; threads, blocks)
  end

  return dirac_series
end

function generate_dirac_gpu!(
  dirac_series::CuDeviceArray{T,N},
  data::CuDeviceArray{S,2},
  bootstrap_idxs::CuDeviceArray{Int,2},
  spacing::CuDeviceArray{T,1},
  low_bound::CuDeviceArray{T,1}
) where {N,T<:Real,S<:Real}
  n_samples = size(data, 2)
  n_bootstraps = size(bootstrap_idxs, 2)
  n_modified_gridpoints = 2^N * n_bootstraps * n_samples
  idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  if idx > n_modified_gridpoints
    return
  end

  idx_dim = ((idx - 1) % 2^N) + 1
  quot, rem = divrem(idx - 1, 2^N)
  idx_bootstrap = (rem % n_bootstraps) + 1
  idx_sample = (quot ÷ n_bootstraps) + 1

end

end
