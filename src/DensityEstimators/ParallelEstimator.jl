abstract type AbstractKernelStatistics{N,T,M} end
abstract type AbstractKernelMeans{N,T,M} <: AbstractKernelStatistics{N,T,M} end
using Core: Argument
abstract type AbstractKernelVars{N,T,M} <: AbstractKernelStatistics{N,T,M} end

struct KernelMeans{N,T<:Real,M} <: AbstractKernelMeans{N,T,M}
  statistic::Array{Complex{T},M}
  bootstrapped::Bool

  function KernelMeans(statistic::Array{Complex{T},M}, bootstrapped::Bool) where {T<:Real,M}
    if bootstrapped
      return new{M - 1,T,M}(statistic, bootstrapped)
    else
      return new{M,T,M}(statistic, bootstrapped)
    end
  end
end
struct CuKernelMeans{N,T<:Real,M} <: AbstractKernelMeans{N,T,M}
  statistic::CuArray{Complex{T},M}
  bootstrapped::Bool

  function CuKernelMeans(statistic::CuArray{Complex{T},M}, bootstrapped::Bool) where {T<:Real,M}
    if bootstrapped
      return new{M - 1,T,M}(statistic, bootstrapped)
    else
      return new{M,T,M}(statistic, bootstrapped)
    end
  end
end
struct KernelVars{N,T<:Real,M} <: AbstractKernelVars{N,T,M}
  statistic::Array{Complex{T},M}
  bootstrapped::Bool

  function KernelVars(statistic::Array{Complex{T},M}, bootstrapped::Bool) where {T<:Real,M}
    if bootstrapped
      return new{M - 1,T,M}(statistic, bootstrapped)
    else
      return new{M,T,M}(statistic, bootstrapped)
    end
  end
end
struct CuKernelVars{N,T<:Real,M} <: AbstractKernelVars{N,T,M}
  statistic::CuArray{Complex{T},M}
  bootstrapped::Bool

  function CuKernelVars(statistic::CuArray{Complex{T},M}, bootstrapped::Bool) where {T<:Real,M}
    if bootstrapped
      return new{M - 1,T,M}(statistic, bootstrapped)
    else
      return new{M,T,M}(statistic, bootstrapped)
    end
  end
end

Devices.get_device(::KernelMeans) = IsCPU()
Devices.get_device(::CuKernelMeans) = IsCUDA()

function is_bootstrapped(
  ::AbstractKernelStatistics{N,T,M}
) where {N,T<:Real,M}
  return ifelse(N == M, false, true)
end

function get_ndims(::AbstractKernelStatistics{N,T,M}) where {N,T<:Real,M}
  return N
end

function initialize_kernels(
  ::IsCPU,
  kde::AbstractKDE{N,T,<:Real},
  grid::AbstractGrid{N,<:Real,M};
  n_bootstraps=0,
  include_var=false,
  method=Devices.CPU_SERIAL,
) where {N,T<:Real,M}
  if n_bootstraps < 0
    throw(ArgumentError("Number of boostraps must be positive."))

  end
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  if include_var == false
    means = initialize_dirac_sequence(
      Val(method), kde.data, grid, bootstrap_idxs; T=T
    )
  else
    means, vars = initialize_dirac_sequence(
      Val(method), kde.data, grid, bootstrap_idxs; include_var=true, T=T
    )
  end

  plan = plan_fft!(selectdim(means, ndims(means), 1))
  if include_var
    fourier_statistics!(Val(method), means, vars, plan)
  else
    fourier_statistics!(Val(method), means, plan)
  end

  if n_bootstraps == 0
    kernel_means = KernelMeans(
      dropdims(means; dims=ndims(means)), false
    )
    if include_var
      kernel_vars = KernelVars(
        dropdims(vars; dims=ndims(vars)), false
      )

      return kernel_means, kernel_vars
    end

    return kernel_means
  else
    kernel_means = KernelMeans(means, true)
    if include_var
      kernel_vars = KernelVars(vars, true)

      return kernel_means, kernel_vars
    end

    return kernel_means
  end

end
function initialize_kernels(
  ::IsCUDA,
  kde::AbstractKDE{N,T,<:Real},
  grid::AbstractGrid{N,<:Real,M};
  n_bootstraps=0,
  include_var=false,
  method=Devices.GPU_CUDA
) where {N,T<:Real,M}
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  if include_var == false
    means = initialize_dirac_sequence(
      Val(method), kde.data, grid, bootstrap_idxs; T=T
    )
  else
    means, vars = initialize_dirac_sequence(
      Val(method), kde.data, grid, bootstrap_idxs; include_var=true, T=T
    )
  end

  plan = plan_fft!(means, ntuple(i -> i, N))
  if include_var
    fourier_statistics!(Val(:cuda), means, vars, plan)
  else
    fourier_statistics!(Val(:cuda), means, plan)
  end

  if n_bootstraps == 0
    kernel_means = CuKernelMeans(
      dropdims(means; dims=ndims(means)), false
    )
    if include_var
      kernel_vars = CuKernelVars(
        dropdims(vars; dims=ndims(vars)), false
      )

      return kernel_means, kernel_vars
    end

    return kernel_means
  else
    kernel_means = CuKernelMeans(means, true)
    if include_var
      kernel_vars = CuKernelVars(vars, true)

      return kernel_means, kernel_vars
    end

    return kernel_means
  end
end

function propagate!(
  means::AbstractArray{Complex{T},M},
  vars::AbstractArray{Complex{T},M},
  kernel_means::AbstractKernelMeans{N,T,M},
  kernel_vars::AbstractKernelVars{N,T,M},
  grid::AbstractGrid{N,<:Real,L},
  time_propagated::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real};
  method=Devices.CPU_SERIAL,
) where {N,T<:Real,M,L}
  if is_bootstrapped(kernel_means)
    kernel_means_reshaped = kernel_means.statistic
    kernel_vars_reshaped = kernel_vars.statistic
    means_reshaped = means
    vars_reshaped = vars
  else
    kernel_means_reshaped = reshape(kernel_means.statistic, size(kernel_means.statistic)..., 1)
    kernel_vars_reshaped = reshape(kernel_vars.statistic, size(kernel_vars.statistic)..., 1)
    means_reshaped = reshape(means, size(means)..., 1)
    vars_reshaped = reshape(vars, size(vars)..., 1)
  end
  grid_array = get_coordinates(grid)

  propagate_statistics!(
    Val(method),
    means_reshaped,
    vars_reshaped,
    kernel_means_reshaped,
    kernel_vars_reshaped,
    time_propagated,
    time_initial,
    grid_array,
  )
end
function propagate!(
  means::AbstractArray{Complex{T},M},
  kernel_means::AbstractKernelMeans{N,T,M},
  grid::AbstractGrid{N,<:Real,L},
  time_propagated::AbstractVector{<:Real};
  method=Devices.CPU_SERIAL,
) where {N,T<:Real,M,L}
  if is_bootstrapped(kernel_means)
    kernel_means_reshaped = kernel_means.statistic
    means_reshaped = means
  else
    kernel_means_reshaped = reshape(kernel_means.statistic, size(kernel_means.statistic)..., 1)
    means_reshaped = reshape(means, size(means)..., 1)
  end
  grid_array = get_coordinates(grid)

  propagate_statistics!(
    Val(method),
    means_reshaped,
    kernel_means_reshaped,
    time_propagated,
    grid_array,
  )

end

abstract type AbstractKernelPropagation{N,T<:Real,M} end

mutable struct KernelPropagation{N,T<:Real,M} <: AbstractKernelPropagation{N,T,M}
  kernel_means::Array{Complex{T},M}
  kernel_vars::Array{Complex{T},M}
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}}
  calculated_vmr::Bool
  calculated_means::Bool

  function KernelPropagation(
    kernel_means::KernelMeans{N,T,M},
    kernel_vars::KernelVars{N,T,M},
  ) where {N,T<:Real,M}
    if !(is_bootstrapped(kernel_means) && is_bootstrapped(kernel_vars))
      throw(ArgumentError("Both kernel means and kernel vars must be bootstrapped"))
    end

    plan = plan_ifft!(selectdim(kernel_means.statistic, get_ndims(kernel_means) + 1, 1))

    new{M - 1,T,M}(
      similar(kernel_means.statistic),
      similar(kernel_vars.statistic),
      plan,
      false,
      false
    )
  end
end
mutable struct CuKernelPropagation{N,T<:Real,M} <: AbstractKernelPropagation{N,T,M}
  kernel_means::CuArray{Complex{T},M}
  kernel_vars::CuArray{Complex{T},M}
  ifft_plan_bootstraps::AbstractFFTs.ScaledPlan{Complex{T}}
  ifft_plan_means::AbstractFFTs.ScaledPlan{Complex{T}}
  calculated_vmr::Bool
  calculated_means::Bool

  function CuKernelPropagation(
    kernel_means::CuKernelMeans{N,T,M},
    kernel_vars::CuKernelVars{N,T,M},
  ) where {N,T<:Real,M}
    if !(is_bootstrapped(kernel_means) && is_bootstrapped(kernel_vars))
      throw(ArgumentError("Both kernel means and kernel vars must be bootstrapped"))
    end

    plan_bootstraps = plan_ifft!(kernel_means.statistic, ntuple(i -> i, get_ndims(kernel_means)))
    plan_means = plan_ifft!(view(kernel_means.statistic, fill(Colon(), N)..., 1))

    new{M - 1,T,M}(
      similar(kernel_means.statistic),
      similar(kernel_vars.statistic),
      plan_bootstraps,
      plan_means,
      false,
      false
    )
  end

end

Devices.get_device(::KernelPropagation) = IsCPU()
Devices.get_device(::CuKernelPropagation) = IsCUDA()

function is_vmr_calculated(kernel_propagation::AbstractKernelPropagation)
  return kernel_propagation.calculated_vmr
end

function is_means_calculated(kernel_propagation::AbstractKernelPropagation)
  return kernel_propagation.calculated_means
end

function get_vmr(kernel_propagation::KernelPropagation{N,T,M}) where {N,T<:Real,M}
  if !is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR not calculated"))
  end
  grid_size = size(kernel_propagation.kernel_vars)[begin:end-1]
  grid_length = prod(grid_size)

  reinterpreted_results = reinterpret(reshape, T, kernel_propagation.kernel_vars)
  vmr_var = reshape(
    reinterpreted_results[begin:grid_length],
    grid_size
  )

  return vmr_var
end
function get_vmr(kernel_propagation::CuKernelPropagation{N,T,M}) where {N,T<:Real,M}
  if !is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR not calculated"))
  end

  vmr_slice = view(kernel_propagation.kernel_vars, fill(Colon(), N)..., 1)
  vmr_reinterpreted = reinterpret(reshape, T, vmr_slice)
  vmr_var = selectdim(vmr_reinterpreted, 1, 1)

  return vmr_var
end

function get_means(kernel_propagation::KernelPropagation{N,T,M}) where {N,T<:Real,M}
  if !is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR not calculated"))
  elseif !is_means_calculated(kernel_propagation)
    throw(ArgumentError("Means not calculated"))
  end

  grid_dims = size(kernel_propagation.kernel_means)[begin:end-1]
  n_elements = prod(grid_dims)

  reinterpreted_means = vec(
    reinterpret(T, kernel_propagation.kernel_means)
  )[begin:n_elements]

  return reshape(reinterpreted_means, grid_dims)
end
function get_means(kernel_propagation::CuKernelPropagation{N,T,M}) where {N,T<:Real,M}
  if !is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR not calculated"))
  elseif !is_means_calculated(kernel_propagation)
    throw(ArgumentError("Means not calculated"))
  end

  means_complex = view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1)
  reinterpreted_means = reinterpret(reshape, T, means_complex)
  means = selectdim(reinterpreted_means, 1, 1)

  return means
end

function propagate_bootstraps!(
  kernel_propagation::AbstractKernelPropagation{N,T,M},
  kernel_means::AbstractKernelMeans{N,T,M},
  kernel_vars::AbstractKernelVars{N,T,M},
  grid::AbstractGrid{N,<:Real,M},
  time_propagated::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real};
  method=Devices.CPU_SERIAL,
) where {N,T<:Real,M}
  propagate!(
    kernel_propagation.kernel_means,
    kernel_propagation.kernel_vars,
    kernel_means,
    kernel_vars,
    grid,
    time_propagated,
    time_initial;
    method,
  )

  kernel_propagation.calculated_vmr = false
  kernel_propagation.calculated_means = false

  return nothing
end

function propagate_means!(
  kernel_propagation::AbstractKernelPropagation{N,T,M},
  means::AbstractKernelMeans{N,T,N},
  grid::AbstractGrid{N,<:Real,M},
  time_propagated::AbstractVector{<:Real};
  method=Devices.CPU_SERIAL,
) where {N,T<:Real,M}
  if !is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR not calculated. Destination array may be in use."))
  end

  means_dst = view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1)
  propagate!(
    means_dst,
    means,
    grid,
    time_propagated;
    method,
  )

  return nothing
end

function ifft_bootstraps!(
  kernel_propagation::KernelPropagation;
  method=Devices.CPU_SERIAL,
)
  ifourier_statistics!(
    Val(method),
    kernel_propagation.kernel_means,
    kernel_propagation.kernel_vars,
    kernel_propagation.ifft_plan,
  )

  return nothing
end
function ifft_bootstraps!(
  kernel_propagation::CuKernelPropagation;
  method=Devices.GPU_CUDA,
)
  ifourier_statistics!(
    Val(method),
    kernel_propagation.kernel_means,
    kernel_propagation.kernel_vars,
    kernel_propagation.ifft_plan_bootstraps,
  )

  return nothing
end

function ifft_means!(
  kernel_propagation::KernelPropagation{N,T,M};
  method=Devices.CPU_SERIAL,
) where {N,T<:Real,M}
  means = reshape(
    view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1),
    size(kernel_propagation.kernel_means)[begin:end-1]..., 1
  )
  ifourier_statistics!(
    Val(method),
    means,
    kernel_propagation.ifft_plan,
  )

  return nothing
end
function ifft_means!(
  kernel_propagation::CuKernelPropagation{N,T,M};
  method::Symbol=Devices.GPU_CUDA,
) where {N,T<:Real,M}
  means = view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1)
  ifourier_statistics!(
    Val(method),
    means,
    kernel_propagation.ifft_plan_means,
  )

  return nothing
end

function calculate_vmr!(
  kernel_propagation::AbstractKernelPropagation{N,<:Real,M},
  time_propagated::AbstractVector{<:Real},
  grid::AbstractGrid{N,<:Real,M},
  n_samples::Integer;
  method=Devices.CPU_SERIAL,
) where {N,M}
  if is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR already calculated. Propagate bootstraps first."))
  end

  t0 = initial_bandwidth(grid)
  calculate_scaled_vmr!(
    Val(method),
    kernel_propagation.kernel_means,
    kernel_propagation.kernel_vars,
    time_propagated,
    t0,
    n_samples,
  )

  kernel_propagation.calculated_vmr = true

  return nothing
end

function calculate_means!(
  kernel_propagation::AbstractKernelPropagation{N,T,M},
  n_samples::Integer;
  method=Devices.CPU_SERIAL,
) where {N,T<:Real,M}
  if is_means_calculated(kernel_propagation)
    throw(ArgumentError("Means already calculated. Propagate bootstraps first."))
  end
  means = view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1)

  calculate_full_means!(
    Val(method),
    means,
    n_samples,
  )

  kernel_propagation.calculated_means = true
end

abstract type AbstractDensityState{N,T} end

@kwdef mutable struct DensityState{N,T} <: AbstractDensityState{N,T}
  # Parameters
  eps_low_id::T

  steps_low::Int
  steps_over::Int

  # State
  indicator_minima::Array{T,N}
  counters_low::Array{UInt16,N}
  counters_over::Array{UInt16,N}
  low_density_flags::Array{Bool,N}

  # Buffers
  f_prev1::Array{T,N}
  f_prev2::Array{T,N}
end
function DensityState(
  dims::NTuple{N,<:Integer};
  T::Type{<:Real}=Float64,
  eps_low_id::Union{Real,Nothing}=nothing,
  steps_low::Integer,
  steps_over::Integer,
) where {N}
  # TODO: Find scaling behavior of Var(vmr) with N and
  # change scaling function to be independent of N.
  if eps_low_id === nothing
    if N == 1
      eps_low_id = 2.0
    elseif N == 2
      eps_low_id = -0.5
    else
      @warn "Parameters for 3D and higher have not been tested yet. Using parameters for 2D."
      eps_low_id = -0.5
    end
  end

  DensityState{N,T}(;
    f_prev1=fill(T(NaN), dims),
    f_prev2=fill(T(NaN), dims),
    eps_low_id=T(eps_low_id),
    steps_low=Int(steps_low),
    steps_over=Int(steps_over),
    indicator_minima=fill(T(NaN), dims),
    counters_low=zeros(UInt16, dims),
    counters_over=zeros(UInt16, dims),
    low_density_flags=fill(false, dims),
  )
end

@kwdef mutable struct CuDensityState{N,T} <: AbstractDensityState{N,T}
  # Parameters
  eps_low_id::T

  steps_low::Int32
  steps_over::Int32

  # State
  indicator_minima::CuArray{T,N}
  counters_low::CuArray{UInt16,N}
  counters_over::CuArray{UInt16,N}
  low_density_flags::CuArray{Bool,N}

  # Buffers
  f_prev1::CuArray{T,N}
  f_prev2::CuArray{T,N}
end
function CuDensityState(
  dims::NTuple{N,<:Integer};
  T::Type{<:Real}=Float32,
  eps_low_id::Union{Real,Nothing}=nothing,
  steps_low::Integer,
  steps_over::Integer,
) where {N}
  # TODO: Find scaling behavior of Var(vmr) with N and
  # change scaling function to be independent of N.
  if eps_low_id === nothing
    if N == 1
      eps_low_id = 2.0f0
    elseif N == 2
      eps_low_id = -0.5f0
    elseif N == 3
      @warn "Parameters for 3D and higher have not been tested yet. Using parameters for 2D."
      eps_low_id = -0.5f0
    else
      throw(ArgumentError("CUDA implementation for dimensions higher than 3 is not supported."))
    end
  end

  CuDensityState{N,T}(;
    f_prev1=CUDA.fill(T(NaN), dims),
    f_prev2=CUDA.fill(T(NaN), dims),
    eps_low_id=T(eps_low_id),
    steps_low=Int32(steps_low),
    steps_over=Int32(steps_over),
    indicator_minima=CUDA.fill(T(NaN), dims),
    counters_low=CUDA.zeros(UInt16, dims),
    counters_over=CUDA.zeros(UInt16, dims),
    low_density_flags=CUDA.fill(false, dims),
  )
end

function update_density_state_vmr!(
  density_state::DensityState{N,<:Real},
  vmr_var::AbstractArray{<:Real,N},
) where {N}
  density_state.f_prev2 .= density_state.f_prev1
  density_state.f_prev1 .= vmr_var

  return nothing
end
function update_density_state_vmr!(
  density_state::CuDensityState{N,<:Real},
  vmr_var::AnyCuArray{<:Real,N},
) where {N}
  density_state.f_prev2 .= density_state.f_prev1
  density_state.f_prev1 .= vmr_var

  CUDA.synchronize()

  return nothing
end

function update_state!(
  density_state::AbstractDensityState{N,<:Real},
  dlogt::Real,
  kde::AbstractKDE{N,<:Real,<:Real},
  kernel_propagation::AbstractKernelPropagation{N,<:Real,M};
  method=Devices.CPU_SERIAL,
) where {N,M}
  vmr_var = get_vmr(kernel_propagation)
  means = get_means(kernel_propagation)

  identify_convergence!(
    Val(method),
    kde.density,
    means,
    vmr_var,
    density_state.f_prev1,
    density_state.f_prev2,
    dlogt,
    density_state.eps_low_id,
    density_state.steps_low,
    density_state.steps_over,
    density_state.indicator_minima,
    density_state.counters_low,
    density_state.counters_over,
    density_state.low_density_flags,
  )

  update_density_state_vmr!(density_state, vmr_var)

  return nothing

end

abstract type AbstractParallelEstimator{N,T,M} <: AbstractEstimator end

struct ParallelEstimator{N,T<:Real,M,P<:Real,S<:Real} <: AbstractParallelEstimator{N,T,M}
  means_bootstraps::KernelMeans{N,T,M}
  vars_bootstraps::KernelVars{N,T,M}
  means::KernelMeans{N,T,N}
  kernel_propagation::KernelPropagation{N,T,M}
  grid_direct::Grid{N,P,M}
  grid_fourier::Grid{N,P,M}
  times::Vector{<:SVector{N,S}}
  density_state::DensityState{N,T}
end

struct CuParallelEstimator{N,T<:Real,M,P<:Real,S<:Real} <: AbstractParallelEstimator{N,T,M}
  means_bootstraps::CuKernelMeans{N,T,M}
  vars_bootstraps::CuKernelVars{N,T,M}
  means::CuKernelMeans{N,T,N}
  kernel_propagation::CuKernelPropagation{N,T,M}
  grid_direct::CuGrid{N,P,M}
  grid_fourier::CuGrid{N,P,M}
  times::CuMatrix{S}
  density_state::CuDensityState
end

add_estimator!(:parallelEstimator, AbstractParallelEstimator)

Devices.get_device(::ParallelEstimator) = IsCPU()
Devices.get_device(::CuParallelEstimator) = IsCUDA()

function initialize_estimator(
  ::Type{<:AbstractParallelEstimator},
  kde::AbstractKDE;
  method::Symbol,
  grid::Union{AbstractGrid,Nothing}=nothing,
  n_bootstraps::Integer=100,
  time_step::Union{Nothing,Real}=nothing,
  n_steps::Union{Nothing,Integer}=nothing,
  kwargs...,
)
  device = get_device(kde)

  if get_device(grid) != device
    throw(ArgumentError("KDE device $device does not match Grid device $(get_device(grid))."))
  end

  means_bootstraps, vars_bootstraps = initialize_kernels(
    device, kde, grid; n_bootstraps=n_bootstraps, include_var=true, method=method
  )
  means = initialize_kernels(
    device, kde, grid; n_bootstraps=0, include_var=false, method=method
  )

  return initialize_estimator_propagation(
    device,
    kde,
    means_bootstraps,
    vars_bootstraps,
    means,
    grid;
    time_step,
    n_steps,
    kwargs...,
  )

end

function initialize_estimator_propagation(device::AbstractDevice, args...; kwargs...)
  throw(ArgumentError("Propagation not implemented for device: $(typeof(device))"))
end
function initialize_estimator_propagation(
  ::IsCPU,
  kde::KDE{N,T,<:Real},
  means_bootstraps::KernelMeans{N,T,M},
  vars_bootstraps::KernelVars{N,T,M},
  means::KernelMeans{N,T,N},
  grid::Grid{N,<:Real,M};
  time_step::Union{Nothing,Real}=nothing,
  time_final::Union{Nothing,Real}=nothing,
  n_steps::Union{Nothing,Integer}=nothing,
  fraction_low::Union{Real,Nothing}=nothing,
  fraction_over::Union{Real,Nothing}=nothing,
  kwargs...
) where {N,T<:Real,M}
  # TODO: Find scaling behavior of Var(vmr) with N and
  # change scaling function to be independent of N.
  if fraction_low === nothing
    if N == 1
      fraction_low = 0.0
    elseif N == 2
      fraction_low = 0.06
    else
      @warn "Parameters for 3D and higher have not been tested yet. Using parameters for 2D."
      fraction_low = 0.06
    end
  end
  if fraction_over === nothing
    if N == 1
      fraction_over = 0.2
    elseif N == 2
      fraction_over = 0.1
    else
      @warn "Parameters for 3D and higher have not been tested yet. Using parameters for 2D."
      fraction_over = 0.1
    end
  end

  grid_fourier = fftgrid(grid)

  kernel_propagation = KernelPropagation(means_bootstraps, vars_bootstraps)
  if time_final === nothing
    time_final = silverman_rule(get_data(kde))
  elseif time_final isa Real
    time_final = fill(time_final, N)
  end
  times, dt = get_time(IsCPU(), time_final; time_step, n_steps)

  steps_low, steps_over = calculate_duration_steps(
    times[end], dt; fraction_low, fraction_over
  )

  density_state = DensityState(
    size(grid); T=T, steps_low, steps_over, kwargs...
  )

  return ParallelEstimator(
    means_bootstraps,
    vars_bootstraps,
    means,
    kernel_propagation,
    grid,
    grid_fourier,
    times,
    density_state,
  )
end
function initialize_estimator_propagation(
  ::IsCUDA,
  kde::CuKDE{N,T,<:Real},
  means_bootstraps::CuKernelMeans{N,T,M},
  vars_bootstraps::CuKernelVars{N,T,M},
  means::CuKernelMeans{N,T,N},
  grid::CuGrid{N,<:Real,M};
  time_step=nothing,
  time_final=nothing,
  n_steps=nothing,
  fraction_low::Union{Real,Nothing}=nothing,
  fraction_over::Union{Real,Nothing}=nothing,
  kwargs...
) where {N,T<:Real,M}
  # TODO: Find scaling behavior of Var(vmr) with N and
  # change scaling function to be independent of N.
  if fraction_low === nothing
    if N == 1
      fraction_low = 0.0f0
    elseif N == 2
      fraction_low = 0.06f0
    elseif N == 3
      @warn "Parameters for 3D and higher have not been tested yet. Using parameters for 2D."
      fraction_low = 0.06f0
    else
      throw(ArgumentError("CUDA implementation for dimensions higher than 3 is not supported."))
    end
  end
  if fraction_over === nothing
    if N == 1
      fraction_over = 0.2f0
    elseif N == 2
      fraction_over = 0.1f0
    elseif N == 3
      @warn "Parameters for 3D and higher have not been tested yet. Using parameters for 2D."
      fraction_over = 0.1f0
    else
      throw(ArgumentError("CUDA implementation for dimensions higher than 3 is not supported."))
    end
  end

  grid_fourier = fftgrid(grid)

  kernel_propagation = CuKernelPropagation(means_bootstraps, vars_bootstraps)
  if time_final === nothing
    time_final = silverman_rule(Array(get_data(kde)))
  elseif time_final isa Real
    time_final = CUDA.fill(Float32(time_final), N)
  end
  times, dt = get_time(IsCUDA(), time_final; time_step, n_steps)

  steps_low, steps_over = calculate_duration_steps(
    times[:, end], dt; fraction_low, fraction_over
  )

  density_state = CuDensityState(
    size(grid); T=typeof(kde).parameters[2], steps_low, steps_over, kwargs...
  )

  return CuParallelEstimator(
    means_bootstraps,
    vars_bootstraps,
    means,
    kernel_propagation,
    grid,
    grid_fourier,
    times,
    density_state,
  )
end

function get_time(
  ::IsCPU,
  time_final::Union{Real,AbstractVector{<:Real}};
  n_dims=nothing,
  n_steps=nothing,
  time_step=nothing,
)
  if time_final isa Real
    if n_dims !== nothing
      time_final = SVector{n_dims,Float64}(time_final)
    elseif time_step isa AbstractVector
      n_dims = length(time_step)
      time_final = SVector{n_dims,Float64}(time_final)
    else
      throw(
        ArgumentError("For real time_final, n_dims must be provided or time_step must be a vector")
      )
    end
  else
    n_dims = length(time_final)
  end

  if n_steps !== nothing
    dt = SVector{n_dims,Float64}(time_final ./ n_steps)
    times = [dt .* i for i in 0:n_steps]

    return times, dt
  elseif time_step !== nothing
    if time_step isa Real
      time_step = @SVector fill(time_step, n_dims)
    end
    if any(time_step .> time_final) || any(time_step .<= 0)
      throw(ArgumentError("Invalid time step"))
    end

    n_steps = minimum(floor.(time_final ./ time_step))
    dt = time_step
    times = [time_step .* i for i in 0:n_steps]

    return times, dt
  else

    return get_time(IsCPU(), time_final, n_steps=250)
  end

end
function get_time(
  ::IsCUDA,
  time_final::Union{Real,AbstractVector{<:Real}};
  n_dims=nothing,
  n_steps=nothing,
  time_step=nothing,
)
  if time_final isa Real
    if n_dims !== nothing
      time_final = CUDA.fill(Float32(time_final), n_dims)
    elseif time_step isa AbstractVector{<:Real}
      n_dims = length(time_step)
      time_final = CUDA.fill(Float32(time_final), n_dims)
    else
      throw(
        ArgumentError("For real time_final, n_dims must be provided or time_step must be a vector")
      )
    end
  else
    n_dims = length(time_final)
  end

  time_final = CuArray{Float32}(time_final)
  if n_steps !== nothing
    dt = time_final ./ n_steps
    times = mapreduce(i -> dt .* i, hcat, 0:n_steps)

    return times, dt
  elseif time_step !== nothing
    if time_step isa Real
      time_step = CUDA.fill(Float32(time_step), n_dims)
    end
    if any(time_step .> time_final) || any(time_step .<= 0)
      throw(ArgumentError("Invalid time step"))
    end

    n_steps = minimum(floor.(time_final ./ time_step))
    dt = CuArray{Float32}(time_step)
    times = mapreduce(i -> dt .* i, hcat, 0:n_steps)

    return times, dt
  else

    return get_time(IsCUDA(), time_final, n_steps=250)
  end

end

function calculate_duration_steps(time_max, dt; fraction_low=0.01, fraction_over=0.3)
  if fraction_low < 0 || fraction_low > 1
    throw(ArgumentError("Fraction must be in the range [0, 1]"))
  end
  if fraction_over < 0 || fraction_over > 1
    throw(ArgumentError("Fraction must be in the range [0, 1]"))
  end
  n_steps = time_max ./ dt
  steps_buffer = fraction_low .* n_steps
  steps_stopping = fraction_over .* n_steps

  return round.(Int, (mean(steps_buffer), mean(steps_stopping)))
end

function estimate!(
  estimator::AbstractParallelEstimator,
  kde::AbstractKDE;
  method::Symbol,
  kwargs...,
)
  check_memory(estimator)

  n_samples = get_nsamples(kde)

  time_initial = initial_bandwidth(estimator.grid_direct)
  time_initial_squared = time_initial .^ 2
  time_initial_squared_dlogt = Vector(time_initial_squared)

  if estimator.times isa AbstractVector{<:AbstractVector{<:Real}}
    times = estimator.times
    times_dlogt = times
  elseif estimator.times isa AbstractMatrix{<:Real}
    times = eachcol(estimator.times)
    times_dlogt = eachcol(Matrix(estimator.times))
  else
    throw(ArgumentError("Unsupported type for times: $(typeof(estimator.times))"))
  end

  for (time_idx, time_propagated) in enumerate(times)
    propagate_bootstraps!(
      estimator.kernel_propagation,
      estimator.means_bootstraps,
      estimator.vars_bootstraps,
      estimator.grid_fourier,
      time_propagated,
      time_initial;
      method,
    )

    ifft_bootstraps!(estimator.kernel_propagation; method)

    calculate_vmr!(
      estimator.kernel_propagation,
      time_propagated,
      estimator.grid_direct,
      n_samples;
      method
    )

    propagate_means!(
      estimator.kernel_propagation,
      estimator.means,
      estimator.grid_fourier,
      time_propagated;
      method
    )

    ifft_means!(estimator.kernel_propagation; method)

    calculate_means!(
      estimator.kernel_propagation,
      n_samples;
      method
    )

    if time_idx == 1
      dlogt = calculate_dlogt(time_initial_squared_dlogt, times_dlogt[time_idx])
    else
      dlogt = calculate_dlogt(
        time_initial_squared_dlogt, times_dlogt[time_idx], previous_time=times_dlogt[time_idx-1]
      )
    end
    update_state!(
      estimator.density_state,
      dlogt,
      kde,
      estimator.kernel_propagation;
      method
    )
  end
  replace_nans!(kde, estimator)

  if get_device(estimator) isa IsCUDA
    CUDA.device_synchronize()
  end

  return nothing
end

function replace_nans!(
  kde::KDE,
  estimator::ParallelEstimator,
)
  remaining_nans = findall(isnan, kde.density)
  kde.density[remaining_nans] .= get_means(estimator.kernel_propagation)[remaining_nans]

  return nothing
end
function replace_nans!(
  kde::CuKDE,
  estimator::CuParallelEstimator,
)
  remaining_nans = findall(isnan, kde.density)
  kde.density[remaining_nans] .= get_means(estimator.kernel_propagation)[remaining_nans]

  CUDA.synchronize()

  return nothing
end

function calculate_dlogt(
  time_initial_squared::AbstractVector{<:Real},
  time_propagated::AbstractVector{<:Real};
  previous_time::Union{AbstractVector{<:Real},Nothing}=nothing,
)
  if previous_time === nothing
    det_prev = prod(time_initial_squared)
  else
    det_prev = prod(time_initial_squared .+ previous_time .^ 2)
  end
  det_curr = prod(time_initial_squared .+ time_propagated .^ 2)

  dlogt = log(det_curr / det_prev)

  return dlogt
end

function get_necessary_memory(estimator::AbstractParallelEstimator)
  bootstraps_memory = sizeof(estimator.means_bootstraps.statistic) * 4
  means_memory = sizeof(estimator.means.statistic) * 3

  return 1.25(bootstraps_memory + means_memory) / 1024^2
end

function check_memory(estimator::AbstractParallelEstimator)
  if get_device(estimator) isa IsCUDA
    return nothing
  end
  available_memory = get_available_memory(get_device(estimator))
  required_memory = get_necessary_memory(estimator)

  if available_memory < required_memory
    throw(
      ArgumentError("Not enough memory available. Required: $required_memory MiB, Available: $available_memory MiB")
    )
  end

  return nothing
end
