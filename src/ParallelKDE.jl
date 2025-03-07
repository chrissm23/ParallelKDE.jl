module ParallelKDE

include("Grids.jl")
include("KDEs.jl")
include("FourierSpace.jl")
include("DirectSpace.jl")

using .Grids
using .KDEs
using .FourierSpace
using .DirectSpace

using StaticArrays,
  FFTW,
  CUDA

using Statistics

export initialize_kde,
  fit_kde!

function initialize_kde(
  data::AbstractVector{<:AbstractVector{<:Real}},
  grid::AbstractVector{<:AbstractRange{<:Real}},
  device::Symbol
)
  if device == :cpu
    return initialize_kde(data, grid, IsCPUKDE())
  elseif device == :gpu
    @assert CUDA.functional() "CUDA.jl is not functional. Use a different method."
    return initialize_kde(data, grid, IsGPUKDE())
  else
    throw(ArgumentError("Invalid device: $device"))
  end
end

function initialize_kde(
  data::AbstractVector{<:AbstractVector{T}},
  grid_ranges::AbstractVector{<:AbstractRange{S}},
  ::IsCPUKDE
)::KDE where {T<:Real,S<:Real}
  N = length(data[1])

  grid = Grid(grid_ranges)
  density = fill(NaN, size(grid))

  t = initial_bandwidth(grid)

  kde = KDE{N,Float64,T,N + 1}(Vector{SVector{N,T}}(data), grid, t, density)

  return kde
end

function initialize_kde(
  data::AbstractVector{<:AbstractVector{T}},
  grid_ranges::AbstractVector{<:AbstractRange{S}},
  ::IsGPUKDE
)::CuKDE where {T<:Real,S<:Real}
  N = length(data[1])

  grid = CuGrid(grid_ranges, b32=true)
  density = CUDA.fill(NaN32, size(grid))

  t = initial_bandwidth(grid)

  rearanged_data = reduce(hcat, data)

  kde = CuKDE{N,Float32,Float32,N + 1}(CuArray{Float32}(rearanged_data), grid, t, density)

  return kde
end

# TODO: Normalize the smoothness to 1 once the threshold factor is known
function fit_kde!(
  kde::AbstractKDE{N,T,S,M};
  dt::Union{Real,Vector{<:Real},Nothing}=nothing,
  n_steps::Union{Int,Nothing}=nothing,
  t_final::Union{Real,Vector{<:Real},Nothing}=nothing,
  n_bootstraps::Union{Int,Nothing}=nothing,
  smoothness::Real=1 / 150,
  ignore_t0::Bool=true,
  method::Symbol=:serial
) where {N,T<:Real,S<:Real,M}
  n_bootstraps = get_nbootstraps(n_bootstraps)
  n_samples = get_nsamples(kde)

  threshold = get_smoothness(smoothness, n_samples, N)

  time_range = get_times(dt, n_steps, t_final, kde.t, ignore_t0, kde.grid)

  if ignore_t0
    set_nan_density!(kde)
  end

  if method == :serial || method == :threaded
    find_density!(DeviceKDE(kde), kde, time_range, n_bootstraps, threshold, method=method)
  elseif method == :cuda
    find_density!(DeviceKDE(kde), kde, time_range, n_bootstraps, threshold)
  else
    throw(ArgumentError("Invalid method: $method"))
  end

end

function find_density!(
  ::IsCPUKDE,
  kde::KDE{N,T,S,M},
  time_range::AbstractVector{<:AbstractRange{<:Real}},
  n_bootstraps::Integer,
  threshold::Real;
  method::Symbol=:serial
)::Nothing where {N,T<:Real,S<:Real,M}
  print("\n")
  println("Threshold: $threshold")
  n_samples = get_nsamples(kde)

  convergence_tmp = fill(NaN, 2, size(kde.grid)...)
  fourier_tmp = Array{Complex{T},M}(undef, n_bootstraps, size(kde.grid)...)

  means_0, variances_0 = initialize_statistics(kde, n_bootstraps, method, tmp=fourier_tmp)
  density_0, var_0 = initialize_distribution(kde, method, tmp=selectdim(fourier_tmp, 1, 1))

  ifft_plan_multi = plan_ifft(means_0, 2:N+1)
  ifft_plan_single = plan_ifft(selectdim(means_0, 1, 1))

  fourier_grid = fftgrid(kde.grid)
  fourier_grid_array = get_coordinates(fourier_grid)

  times = reinterpret(reshape, T, collect(zip(time_range...)))
  if length(time_range) == 1
    times = reshape(times, 1, :)
  end
  time_initial = initial_bandwidth(kde.grid)

  means_dst = Array{Complex{T},N + 1}(undef, size(means_0))
  variances_dst = Array{Complex{T},N + 1}(undef, size(variances_0))

  for time in eachcol(times)
    means_fourier, variances_fourier = propagate_bandwidth!(
      means_0,
      variances_0,
      fourier_grid_array,
      time,
      time_initial,
      method,
      dst_mean=means_dst,
      dst_var=variances_dst,
    )
    means_direct, variances_direct = ifft_statistics!(
      means_fourier,
      variances_fourier,
      n_samples,
      ifft_plan_multi,
      tmp=fourier_tmp,
      bootstraps_dim=true
    )

    vmr_variance = calculate_statistics!(
      means_direct,
      variances_direct,
      method,
      dst_vmr=selectdim(
        selectdim(reinterpret(reshape, T, variances_dst), 1, 1),
        1,
        1
      ),
    )

    mean_fourier, variance_fourier = propagate_bandwidth!(
      reshape(density_0, 1, size(density_0)...),
      reshape(var_0, 1, size(var_0)...),
      fourier_grid_array,
      time,
      time_initial,
      method,
      dst_mean=reshape(selectdim(means_dst, 1, 1), 1, size(means_dst)[2:end]...),
      dst_var=reshape(selectdim(means_dst, 1, 2), 1, size(means_dst)[2:end]...),
    )
    mean_fourier = dropdims(mean_fourier, dims=1)
    variance_fourier = dropdims(variance_fourier, dims=1)

    mean_direct, variance_direct = ifft_statistics!(
      mean_fourier,
      variance_fourier,
      n_samples,
      ifft_plan_single,
      tmp=selectdim(fourier_tmp, 1, 1),
      bootstraps_dim=false
    )
    mean_direct .*= n_samples

    assign_density!(
      kde,
      mean_direct,
      variance_direct,
      vmr_variance,
      time,
      time_initial,
      threshold,
      method,
      dst_var_products=vmr_variance,
      distances_tmp=convergence_tmp
    )

    if time === times[end]
      @warn "Not all points converged with the specified time ranges."
    end
  end

  return nothing
end
function find_denisty!(
  ::IsGPUKDE,
  kde::CuKDE{N,T,S,M},
  time_range::AbstractVector{<:AbstractRange{<:Real}},
  n_bootstraps::Integer,
  threshold::Real;
)::Nothing where {N,T<:Real,S<:Real,M}
  n_samples = get_nsamples(kde)

  convergence_tmp = CUDA.fill(NaN32, 2, size(kde.grid)...)
  fourier_tmp = CuArray{Complex{T},M}(undef, n_bootstraps, size(kde.grid)...)

  means_0, variances_0 = initialize_statistics(kde, n_bootstraps, tmp=fourier_tmp)
  density_0, var_0 = initialize_distribution(kde, tmp=selectdim(fourier_tmp, 1, 1))

  ifft_plan_multi = plan_ifft(means_0, 2:N+1)
  ifft_plan_single = plan_ifft(selectdim(means_0, 1, 1))

  fourier_grid = fftgrid(kde.grid)
  fourier_grid_array = get_coordinates(fourier_grid)

  times = CuArray{T,N}(
    reinterpret(reshape, T, collect(zip(time_range...)))
  )
  if length(time_range) == 1
    times = reshape(times, 1, :)
  end
  time_initial = initial_bandwidth(kde.grid)

  means_dst = CuArray{Complex{T},N + 1}(undef, size(means_0))
  variances_dst = CuArray{Complex{T},N + 1}(undef, size(variances_0))

  for col in 1:size(times, 2)
    time = view(times, :, col)

    means_fourier, variances_fourier = propagate_bandwidth!(
      means_0,
      variances_0,
      fourier_grid_array,
      time,
      time_initial,
      dst_mean=means_dst,
      dst_var=variances_dst,
    )
    means_direct, variances_direct = ifft_statistics!(
      means_fourier,
      variances_fourier,
      n_samples,
      ifft_plan_multi,
      tmp=fourier_tmp,
      bootstraps_dim=true
    )

    vmr_variance = calculate_statistics!(
      means_direct,
      variances_direct,
      dst_vmr=selectdim(
        selectdim(reinterpret(reshape, T, variances_dst), 1, 1),
        1,
        1
      ),
    )

    mean_fourier, variance_fourier = propagate_bandwidth!(
      reshape(density_0, 1, size(density_0)...),
      reshape(var_0, 1, size(var_0)...),
      fourier_grid_array,
      time,
      time_initial,
      dst_mean=reshape(selectdim(means_dst, 1, 1), 1, size(means_dst)[2:end]...),
      dst_var=reshape(selectdim(means_dst, 1, 2), 1, size(means_dst)[2:end]...),
    )
    mean_fourier = dropdims(mean_fourier, dims=1)
    variance_fourier = dropdims(variance_fourier, dims=1)

    mean_direct, variance_direct = ifft_statistics!(
      mean_fourier,
      variance_fourier,
      n_samples,
      ifft_plan_single,
      tmp=selectdim(fourier_tmp, 1, 1),
      bootstraps_dim=false
    )
    means_direct .*= n_samples

    assign_density!(
      kde,
      mean_direct,
      variance_direct,
      vmr_variance,
      time,
      time_initial,
      threshold,
      dst_var_products=vmr_variance,
      distances_tmp=convergence_tmp
    )

    if col == size(times, 2)
      @warn "Not all points converged with the specified time ranges."
    end
  end

  return nothing
end

function initialize_statistics(
  kde::KDE{N,T,S,M},
  n_bootstraps::Integer,
  method::Symbol;
  tmp::AbstractArray{Complex{T},M}=Array{Complex{T},M}(undef, n_bootstraps, size(kde.grid)...)
)::NTuple{2,Array{Complex{T},N + 1}} where {N,T<:Real,S<:Real,M}
  dirac_series, dirac_series_squared = initialize_dirac_series(
    Val(method),
    kde,
    n_bootstraps=n_bootstraps,
    calculate_squared=true
  )
  sk_0, s2k_0 = initialize_fourier_statistics(dirac_series, dirac_series_squared, tmp)

  return sk_0, s2k_0
end
function initialize_statistics(
  kde::CuKDE{N,T,S,M},
  n_bootstraps::Integer;
  tmp::AnyCuArray{Complex{T},M}=CuArray{Complex{T},M}(undef, n_bootstraps, size(kde.grid)...)
)::NTuple{2,AnyCuArray{Complex{T},N + 1}} where {N,T<:Real,S<:Real,M}
  dirac_series, dirac_series_squared = initialize_dirac_series(
    Val(:cuda),
    kde,
    n_bootstraps=n_bootstraps,
    calculate_squared=true
  )
  s_0, s2_0 = initialize_fourier_statistics(dirac_series, dirac_series_squared, tmp)

  return s_0, s2_0
end

function initialize_distribution(
  kde::KDE{N,T,S,M},
  method::Symbol;
  tmp::AbstractArray{Complex{T},N}=Array{Complex{T},N}(undef, size(kde.grid))
)::NTuple{2,Array{Complex{T},N}} where {N,T<:Real,S<:Real,M}
  dirac_series, dirac_series_squared = initialize_dirac_series(
    Val(method),
    kde,
    calculate_squared=true
  )
  s_0, s2_0 = initialize_fourier_statistics(
    dirac_series, dirac_series_squared, reshape(tmp, 1, size(tmp)...)
  )

  return dropdims(s_0, dims=1), dropdims(s2_0, dims=1)
end
function initialize_distribution(
  kde::CuKDE{N,T,S,M};
  tmp::AnyCuArray{Complex{T},N}=CuArray{Complex{T},N}(undef, size(kde.grid))
)::NTuple{2,AnyCuArray{Complex{T},N}} where {N,T<:Real,S<:Real,M}
  dirac_series, dirac_series_squared = initialize_dirac_series(
    Val(:cuda),
    kde,
    calculate_squared=true
  )

  s_0, s2_0 = initialize_fourier_statistics(dirac_series, dirac_series_squared, reshape(tmp, 1, size(tmp)...))

  return dropdims(s_0, dims=1), dropdims(s2_0, dims=1)
end

function propagate_bandwidth!(
  means_0::Array{Complex{T},M},
  variances_0::Array{Complex{T},M},
  grid_array::AbstractArray{S,M},
  time::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real},
  method::Symbol;
  dst_mean::AbstractArray{Complex{T},M}=Array{Complex{T},M}(undef, size(means_0)),
  dst_var::AbstractArray{Complex{T},M}=Array{Complex{T},M}(undef, size(variances_0)),
) where {T<:Real,S<:Real,M}
  propagate_statistics!(
    Val(method),
    dst_mean,
    dst_var,
    means_0,
    variances_0,
    grid_array,
    SVector{M - 1,Float64}(time),
    SVector{M - 1,Float64}(time_initial)
  )

  return dst_mean, dst_var
end
function propagate_bandwidth!(
  means_0::AnyCuArray{Complex{T},M},
  variances_0::AnyCuArray{Complex{T},M},
  grid_array::AnyCuArray{S,M},
  time::CuVector{<:Real},
  time_initial::CuVector{<:Real};
  dst_mean::AnyCuArray{Complex{T},M}=CuArray{Complex{T},M}(undef, size(means_0)),
  dst_var::AnyCuArray{Complex{T},M}=CuArray{Complex{T},M}(undef, size(variances_0)),
) where {T<:Real,S<:Real,M}
  propagate_statistics!(
    Val(:cuda),
    dst_mean,
    dst_var,
    means_0,
    variances_0,
    grid_array,
    time,
    time_initial
  )

  return dst_mean, dst_var
end

function calculate_statistics!(
  means_bootstraps::AbstractArray{T,M},
  variances_bootstraps::AbstractArray{T,M},
  method::Symbol;
  dst_vmr::AbstractArray{T,N}=Array{T,M - 1}(undef, size(means_bootstraps)[2:end]),
)::AbstractArray{T,N} where {N,T<:Real,M}
  vmr_variance = mean_var_vmr!(
    Val(method),
    means_bootstraps,
    variances_bootstraps,
    dst_vmr,
  )

  return vmr_variance
end
function calculate_statistics!(
  means_bootstraps::AnyCuArray{T,M},
  variances_bootstraps::AnyCuArray{T,M};
  dst_vmr::AnyCuArray{T,N}=CuArray{T,M - 1}(undef, size(means_bootstraps)),
)::AnyCuArray{T,N} where {N,T<:Real,M}
  vmr_variance = mean_var_vmr!(
    Val(:cuda),
    means_bootstraps,
    variances_bootstraps,
    dst_vmr,
  )

  return vmr_variance
end

function assign_density!(
  kde::KDE{N,T,S,M},
  mean_complete::AbstractArray{Complex{T},N},
  variance_complete::AbstractArray{Complex{T},N},
  vmr_variance::AbstractArray{T,N},
  time::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real},
  threshold::Real,
  method::Symbol;
  dst_var_products::AbstractArray{T,N}=similar(vmr_variance),
  distances_tmp::AbstractArray{T,M}=Array{T,N}(undef, 2, size(variance_complete)...)
)::Nothing where {N,T<:Real,S<:Real,M}
  variance_products = calculate_variance_products!(
    Val(method),
    vmr_variance,
    variance_complete,
    time,
    time_initial,
    dst_var_products=dst_var_products
  )

  assign_converged_density!(
    Val(method), kde.density, mean_complete, variance_products, threshold, distances_tmp
  )

  kde.t .= time

  return nothing
end
function assign_density!(
  kde::CuKDE{N,T,S,M},
  mean_complete::AnyCuArray{Complex{T},N},
  variance_complete::AnyCuArray{Complex{T},N},
  vmr_variance::AnyCuArray{T,N},
  time::CuVector{<:Real},
  time_initial::CuVector{<:Real},
  threshold::Real;
  dst_var_products::AnyCuArray{T,N}=similar(vmr_variance),
  distances_tmp::AnyCuArray{T,M}=CuArray{T,N}(undef, 2, size(variance_complete)...)
)::Nothing where {N,T<:Real,S<:Real,M}
  variance_products = calculate_variance_products!(
    Val(:cuda),
    vmr_variance,
    variance_complete,
    time,
    time_initial,
    dst_var_products=dst_var_products
  )

  assign_converged_density!(
    Val(:cuda), kde.density, mean_complete, variance_products, threshold, distances_tmp
  )

  kde.t .= time

  return nothing
end

function get_times(
  dt::Union{Real,Vector{<:Real},Nothing},
  n_steps::Union{Int,Nothing},
  t_final::Union{Real,Vector{<:Real},Nothing},
  t0::AbstractVector{<:Real},
  ignore_t0::Bool,
  grid::AbstractGrid
)::AbstractVector{<:AbstractRange{<:Real}}

  if ignore_t0
    bandwidth_0 = initial_bandwidth(grid)
    t0 = bandwidth_0
  end
  if t0 isa AnyCuArray
    t0 = Array{Float64}(t0)
  end

  if t_final !== nothing
    if any(t_final .< t0)
      throw(ArgumentError("The final time must be greater than the initial bandwidth."))
    end
  end

  if ignore_t0
    t0 = @SVector zeros(Float64, length(t0))
  end

  if dt !== nothing && n_steps !== nothing && t_final !== nothing
    throw(ArgumentError("Only two of dt, n_steps, or tf can be provided."))
  elseif dt !== nothing && n_steps !== nothing
    time_range = range.(t0, step=dt, length=n_steps)
  elseif dt !== nothing && t_final !== nothing
    time_range = range.(t0, t_final, step=dt)
  elseif t_final !== nothing && n_steps !== nothing
    time_range = range.(t0, t_final, length=n_steps)
  else
    # TODO: Implement some automatic time step determination
    throw(ArgumentError("Two of dt, n_steps, or tf must be provided."))
  end
end

function get_nbootstraps(n_bootstraps::Union{Int,Nothing})
  if n_bootstraps === nothing
    n_bootstraps = 100
  end

  return n_bootstraps
end

function get_smoothness(smoothness::Real, n_samples::Integer, n_dims::Int)
  if smoothness === nothing
    smoothness = 1 / 150
  elseif smoothness < 0.0
    throw(ArgumentError("Smoothness must be positive."))
  end
  threshold = smoothness * optimal_threshold(n_samples, n_dims)

  return threshold
end

function optimal_threshold(n_samples::Integer, n_dims::Int)
  return 1 / ((2^n_dims - 3^(n_dims / 2)) * 2^(3n_dims - 1) * π^(2n_dims) * Float64(n_samples)^5)
end

function get_available_memory(::IsCPUKDE)
  return Sys.free_memory() / 1024^2
end
function get_available_memory(::IsGPUKDE)
  if !CUDA.functional()
    throw(ArgumentError("CUDA.jl is not functional. Use a different method."))
  end
  return CUDA.memory_status().free / 1024^2
end

end
