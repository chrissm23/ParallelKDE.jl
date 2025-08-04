abstract type AbstractRoT end

function propagation_time(::AbstractRoT)
  throw(ArgumentError("Propagation time not defined for this rule of thumb."))
end

struct SilvermanRoT{N,T<:Real} <: AbstractRoT
  n_dims::Int
  n_samples::Int
end
function initialize_rot(
  ::Val{:silverman},
  n_dims::Int,
  n_samples::Int,
)
  return SilvermanRoT(n_dims, n_samples)
end

function propagation_time(rot::SilvermanRoT, data::AbstractMatrix{T}) where {T<:Real}
  factor = (rot.n_samples * (rot.n_dims + 2) / 4)^(-1 / (rot.n_dims + 4))
  cov_matrix = cov(data, dims=2)

  return factor .* sqrt.(diag(cov_matrix))
end

struct ScottRoT{N,T<:Real} <: AbstractRoT
  n_dims::Int
  n_samples::Int
end
function initialize_rot(
  ::Val{:scott},
  n_dims::Int,
  n_samples::Int,
)
  return ScottRoT(n_dims, n_samples)
end

function propagation_time(rot::ScottRoT, data::AbstractMatrix{T}) where {T<:Real}
  factor = rot.n_samples^(1 / (rot.n_dims + 4))
  cov_matrix = cov(data, dims=2)

  return factor .* sqrt.(diag(cov_matrix))
end

abstract type AbstractRoTEstimator <: AbstractEstimator end

struct RoTEstimator{N,R<:AbstractRoT,T<:Real,M,P<:Real,S<:Real} <: AbstractRoTEstimator
  rule_of_thumb::R
  fourier_density::AbstractArray{T,N}
  grid_direct::Grid{N,P,M}
  grid_fourier::Grid{N,P,M}
  time_propagated::SVector{N,S}
end
function RoTEstimator(
  rule_of_thumb::AbstractRoT,
  data::AbstractMatrix{N,S},
  fourier_density::AbstractArray{Complex{T},N},
  grid_direct::Grid{N,P,M},
) where {N,T<:Real,M,P<:Real,S<:Real}
  grid_fourier = fftgrid(grid_direct)
  time_final = propagation_time(rule_of_thumb, data)
  time_initial = initial_bandwidth(grid_direct)
  time_propagated = SVector{N}(time_final .- time_initial)

  return RoTEstimator(
    rule_of_thumb,
    fourier_density,
    grid_direct,
    grid_fourier,
    time_propagated
  )
end

struct CuRoTEstimator{N,R<:AbstractRoT,T<:Real,M,P<:Real,S<:Real} <: AbstractRoTEstimator
  rule_of_thumb::R
  fourier_density::CuArray{Complex{T},N}
  grid_direct::CuGrid{N,P,M}
  grid_fourier::CuGrid{N,P,M}
  time_propagated::CuVector{N,S}
end
function CuRoTEstimator(
  rule_of_thumb::AbstractRoT,
  data::CuMatrix{N,S},
  fourier_density::CuArray{T,N},
  grid_direct::CuGrid{N,P,M},
) where {N,T<:Real,M,P<:Real,S<:Real}
  grid_fourier = fftgrid(grid_direct)
  time_final = propagation_time(rule_of_thumb, data)
  time_initial = initial_bandwidth(grid_direct)
  time_propagated = time_final .- time_initial

  return CuRoTEstimator(
    rule_of_thumb,
    fourier_density,
    grid_direct,
    grid_fourier,
    time_propagated,
  )
end

add_estimator!(:rotEstimator, AbstractRoTEstimator)

function initialize_estimator(
  ::Type{<:AbstractRoTEstimator},
  kde::AbstractKDE{N,T,S};
  method::Symbol,
  grid::Union{Nothing,AbstractGrid}=nothing,
  rule_of_thumb::Union{Nothing,Symbol}=:scott,
  kwargs...,
) where {N,T<:Real,S<:Real}
  device = get_device(kde)

  if get_device(grid) != device
    throw(ArgumentError("KDE device $device does not match Grid device $(get_device(grid))."))
  end

  n_samples = get_nsamples(kde)
  n_dims = ndims(grid)
  rot = initialize_rot(rule_of_thumb, n_dims, n_samples, kwargs...)

  initial_density = initialize_density(device, kde, grid; method)

  return RoTEstimator(rot, get_data(kde), initial_density, grid)
end

function estimate!(
  estimator::RoTEstimator,
  kde::AbstractKDE;
  method::Symbol,
  kwargs...
)
  propagated_kernels = similar(estimator.fourier_density)
  propagate_kernels!(propagated_kernels, estimator; method)
  ifft_density!(propagated_kernels, kde; method)

  return nothing
end

function initialize_density(
  ::IsCPU,
  kde::AbstractKDE{N,T,<:Real},
  grid::AbstractGrid{N,<:Real,M};
  method=Devices.CPU_SERIAL,
) where {N,T<:Real,M}
  initial_density = initialize_dirac_sequence(Val(method), kde.data, grid, nothing, T=T)

  plan = plan_fft!(selectdim(initial_density, ndims(initial_density), 1))
  fourier_statistics!(Val(method), initial_density, plan)

  return initial_density
end
function initialize_density(
  ::IsCUDA,
  kde::AbstractKDE{N,T,<:Real},
  grid::CuGrid{N,<:Real,M};
  method=Devices.GPU_CUDA,
) where {N,T<:Real,M}
  initial_density = initialize_dirac_sequence(Val(method), kde.data, grid, nothing, T=T)

  plan = plan_fft!(initial_density, ntuple(i -> i, N))
  fourier_statistics!(Val(method), initial_density, plan)

  return initial_density
end

function propagate_kernels!(
  propagated_kernels::AbstractArray{Complex{T},N},
  estimator::AbstractRoTEstimator;
  method=Devices.CPU_SERIAL,
) where {N,T<:Real}
  initial_density = estimator.fourier_density
  time_propagated = estimator.time_propagated
  grid = estimator.grid_fourier

  grid_array = get_coordinates(grid)

  propagate_statistics!(
    Val(method),
    propagated_kernels,
    initial_density,
    time_propagated,
    grid_array
  )
end

function ifft_density!(
  propagated_kernels::AbstractArray{Complex{T},N},
  kde::AbstractKDE;
  method=Devices.CPU_SERIAL,
) where {N,T<:Real}
  ifft_plan = plan_ifft!(selectdim(propagated_kernels, ndims(propagated_kernels), 1))
  ifourier_statistics!(Val(method), propagated_kernels, ifft_plan)

  copy_density!(Val(method), kde.density, propagated_kernels)

  return nothing
end
function ifft_density!(
  propagated_kernels::CuArray{Complex{T},N},
  kde::CuKDE;
  method=Devices.GPU_CUDA,
) where {N,T<:Real}
  ifft_plan = plan_ifft!(propagated_kernels, ntuple(i -> i, N))
  ifourier_statistics!(Val(method), propagated_kernels, ifft_plan)

  copy_density!(Val(method), kde.density, propagated_kernels)

  return nothing
end
