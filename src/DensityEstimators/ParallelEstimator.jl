abstract type AbstractKernelStatistics{N,T,M} end
abstract type AbstractKernelMeans{N,T,M} <: AbstractKernelStatistics{N,T,M} end
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

Device(::KernelMeans) = IsCPU()
Device(::CuKernelMeans) = IsCUDA()

function is_bootstrapped(
  ::AbstractKernelStatistics{N,T,M}
)::Bool where {N,T<:Real,M}
  return ifelse(N == M, false, true)
end

function get_ndims(::AbstractKernelStatistics{N,T,M})::Int where {N,T<:Real,M}
  return N
end

function initialize_kernels(
  ::IsCPU,
  kde::AbstractKDE{N,T,S},
  grid::AbstractGrid{N,P,M};
  n_bootstraps::Integer=0,
  include_var::Bool=false,
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,S<:Real,M,P<:Real}
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
  kde::AbstractKDE{N,T,S},
  grid::AbstractGrid{N,P,M};
  n_bootstraps::Integer=0,
  include_var::Bool=false,
) where {N,T<:Real,S<:Real,M,P<:Real}
  bootstrap_idxs = bootstrap_indices(kde, n_bootstraps)

  if include_var == false
    means = initialize_dirac_sequence(
      Val(:cuda), kde.data, grid, bootstrap_idxs; T=T
    )
  else
    means, vars = initialize_dirac_sequence(
      Val(:cuda), kde.data, grid, bootstrap_idxs; include_var=true, T=T
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
  grid::AbstractGrid{N,P,M},
  time::AbstractVector{<:Real},
  time_initial::AbstractVector{<:Real};
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,M,P<:Real}
  if is_bootstrapped(kernel_means)
    means_bootstrap = kernel_means.statistic
  else
    means_bootstrap = reshape(kernel_means.statistic, size(kernel_means.statistic)..., 1)
  end
  if is_bootstrapped(kernel_vars)
    vars_bootstrap = kernel_vars.statistic
  else
    vars_bootstrap = reshape(kernel_vars.statistic, size(kernel_vars.statistic)..., 1)
  end

  grid_array = get_coordinates(grid)

  propagate_statistics!(
    Val(method),
    means,
    vars,
    means_bootstrap,
    vars_bootstrap,
    time,
    time_initial,
    grid_array,
  )
end
function propagate!(
  means::AbstractArray{Complex{T},M},
  kernel_means::AbstractKernelMeans{N,T,M},
  grid::AbstractGrid{N,P,M},
  time::AbstractVector{<:Real};
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,M,P<:Real}
  if is_bootstrapped(kernel_means)
    means_initial = kernel_means.statistic
  else
    means_initial = reshape(kernel_means.statistic, size(kernel_means.statistic)..., 1)
  end
  grid_array = get_coordinates(grid)

  propagate_statistics!(
    Val(method),
    means,
    means_initial,
    time,
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
  calcualted_means::Bool

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

Device(::KernelPropagation) = IsCPU()
Device(::CuKernelPropagation) = IsCUDA()

function is_vmr_calculated(kernel_propagation::AbstractKernelPropagation{N,T,M})::Bool where {N,T,M}
  return kernel_propagation.calculated_vmr
end

function is_means_calculated(kernel_propagation::AbstractKernelPropagation{N,T,M})::Bool where {N,T,M}
  return kernel_propagation.calculated_means
end

function get_vmr(kernel_propagation::KernelPropagation{N,T,M})::AbstractArray{T,M} where {N,T,M}
  if !is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR not calculated"))
  end

  reinterpreted_results = reinterpret(reshape, T, kernel_propagation.kernel_vars)
  vmr_var = reshape(
    reinterpreted_results[begin:length(kernel_propagation.kernel_vars)],
    size(kernel_propagation.kernel_vars)
  )

  return vmr_var
end
function get_vmr(kernel_propagation::CuKernelPropagation{N,T,M})::AbstractArray{T,M} where {N,T,M}
  if !is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR not calculated"))
  end

  vmr_slice = selectdim(kernel_propagation.kernel_vars, M, 1)
  vmr_reinterpreted = reinterpret(reshape, T, vmr_slice)
  vmr_var = selectdim(vmr_reinterpreted, 1, 1)

  return vmr_var
end

function get_means(kernel_propagation::KernelPropagation{N,T,M})::AbstractArray{T,M} where {N,T,M}
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
function get_means(kernel_propagation::CuKernelPropagation{N,T,M})::AbstractArray{T,M} where {N,T,M}
  if !is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR not calculated"))
  elseif !is_means_calculated(kernel_propagation)
    throw(ArgumentError("Means not calculated"))
  end

  means = view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1)
  reinterpreted_means = reinterpret(reshape, T, means)

  return selectdim(reinterpreted_means, 1, 1)
end

function propagate_bootstraps!(
  kernel_propagation::AbstractKernelPropagation{N,T,M},
  kernel_means::AbstractKernelMeans{N,T,M},
  kernel_vars::AbstractKernelVars{N,T,M},
  grid::AbstractGrid{N,P,M};
  time::AbstractVector{S},
  time_initial::AbstractVector{S},
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,M,P<:Real,S<:Real}
  propagate!(
    kernel_propagation.kernel_means,
    kernel_propagation.kernel_vars,
    kernel_means,
    kernel_vars,
    grid,
    time,
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
  grid::AbstractGrid{N,P,M},
  time::AbstractVector{S};
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,M,P<:Real,S<:Real}
  if !is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR not calculated. Destination array may be in use."))
  end

  means_dst = view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1)
  propagate!(
    means_dst,
    means,
    grid,
    time;
    method,
  )

  return nothing
end

function ifft_bootstraps!(
  kernel_propagation::KernelPropagation{N,T,M};
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,M}
  ifourier_statistics!(
    Val(method),
    kernel_propagation.kernel_means,
    kernel_propagation.kernel_vars,
    kernel_propagation.ifft_plan,
  )

  return nothing
end
function ifft_bootstraps!(
  kernel_propagation::CuKernelPropagation{N,T,M};
  method::Symbol=GPU_CUDA,
) where {N,T<:Real,M}
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
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,M}
  ifourier_statistics!(
    Val(method),
    reshape(
      view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1),
      size(kernel_propagation.kernel_means)[begin:end-1]..., 1
    ),
    kernel_propagation.ifft_plan,
  )

  return nothing
end
function ifft_means!(
  kernel_propagation::CuKernelPropagation{N,T,M};
  method::Symbol=GPU_CUDA,
) where {N,T<:Real,M}
  ifourier_statistics!(
    Val(method),
    view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1),
    kernel_propagation.ifft_plan_means,
  )

  return nothing
end

function calculate_vmr!(
  kernel_propagation::AbstractKernelPropagation{N,T,M},
  time::AbstractVector{S},
  grid::AbstractGrid{N,P,M},
  n_samples::F;
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,M,P<:Real,S<:Real,F<:Integer}
  if is_vmr_calculated(kernel_propagation)
    throw(ArgumentError("VMR already calculated. Propagate bootstraps first."))
  end

  t0 = initial_bandwidth(grid)
  vmr_var = calculate_scaled_vmr!(
    Val(method),
    kernel_propagation.kernel_means,
    kernel_propagation.kernel_vars,
    time,
    t0,
    n_samples,
  )

  kernel_propagation.calculated_vmr = true

  return vmr_var
end

function calculate_means!(
  kernel_propagation::AbstractKernelPropagation{N,T,M},
  n_samples::F;
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,M,F<:Real}
  means = view(kernel_propagation.kernel_means, fill(Colon(), N)..., 1)

  calculate_full_means!(
    Val(method),
    means,
    n_samples,
  )

  kernel_propagation.calculated_means = true
end

abstract type AbstractDensityState{T,N} end

@kwdef mutable struct DensityState{T,N} <: AbstractDensityState{T,N}
  # Parameters
  dt::Float64
  eps1::Float64 = 1.0
  eps2::Float64 = 10.0
  smoothness_duration::Int8 = Int8(3)
  stable_duration::Int8 = Int8(3)

  # State
  smooth_counters::Array{Int8,N}
  stable_counters::Array{Int8,N}
  is_smooth::Array{Bool,N}
  has_decreased::Array{Bool,N}
  is_stable::Array{Bool,N}

  # Buffers
  f_prev1::Array{T,N}
  f_prev2::Array{T,N}
end
function DensityState(
  dims::NTuple{N,Int},
  T::Type{<:Real},
  dt::Float64,
) where {N}
  DensityState{T,N}(;
    dt=dt,
    f_prev=fill(T(NaN), dims),
    f_prev2=fill(T(NaN), dims),
    smooth_counters=fill(Int8(0), dims),
    stable_counters=fill(Int8(0), dims),
    is_smooth=fill(false, dims),
    has_decreased=fill(false, dims),
    is_stable=fill(false, dims),
  )
end
function DensityState(
  dims::NTuple{N,Int},
  T::Type{<:Real};
  kwargs...
) where {N}
  DensityState{T,N}(;
    f_prev=fill(T(NaN), dims),
    f_prev2=fill(T(NaN), dims),
    smooth_counters=fill(Int8(0), dims),
    stable_counters=fill(Int8(0), dims),
    is_smooth=fill(false, dims),
    has_decreased=fill(false, dims),
    kwargs...
  )
end

@kwdef mutable struct CuDensityState{T,N} <: AbstractDensityState{T,N}
  # Parameters
  dt::Float32
  eps1::Float32 = 1.0f0
  eps2::Float32 = 10.0f0
  smoothness_duration::Int8 = Int8(3)
  stable_duration::Int8 = Int8(3)

  # State
  smooth_counters::CuArray{Int8,N}
  stable_counters::CuArray{Int8,N}
  is_smooth::CuArray{Bool,N}
  has_decrease::CuArray{Bool,N}
  is_stable::CuArray{Bool,N}

  # Buffers
  f_prev1::CuArray{T,N}
  f_prev2::CuArray{T,N}
end
function CuDensityState(
  dims::NTuple{N,Int},
  T::Type{<:Real},
  dt::Float32,
) where {N}
  CuDensityState{T,N}(;
    dt=dt,
    f_prev1=CUDA.fill(T(NaN), dims),
    f_prev2=CUDA.fill(T(NaN), dims),
    smooth_counters=CUDA.fill(Int8(0), dims),
    stable_counters=CUDA.fill(Int8(0), dims),
    is_smooth=CUDA.fill(false, dims),
    has_decreased=CUDA.fill(false, dims),
    is_stable=CUDA.fill(false, dims),
  )
end
function CuDensityState(
  dims::NTuple{N,Int},
  T::Type{<:Real};
  kwargs...
) where {N}
  CuDensityState{T,N}(;
    f_prev1=CUDA.fill(T(NaN), dims),
    f_prev2=CUDA.fill(T(NaN), dims),
    smooth_counters=CUDA.fill(Int8(0), dims),
    stable_counters=CUDA.fill(Int8(0), dims),
    is_smooth=CUDA.fill(false, dims),
    has_decreased=CUDA.fill(false, dims),
    kwargs...
  )
end

function update_state!(
  density_state::AbstractDensityState{T,N},
  kde::AbstractKDE{N,T,S},
  kernel_propagation::AbstractKernelPropagation{N,T,M};
  method::Symbol=CPU_SERIAL,
) where {N,T<:Real,S<:Real,M}
  n_samples = get_nsamples(kde)

  vmr_var = get_vmr(kernel_propagation)
  means = get_means(kernel_propagation)

  identify_convergence!(
    Val(method),
    vmr_var,
    density_state.f_prev1,
    density_state.f_prev2,
    kde.density,
    means,
    density_state.dt,
    density_state.eps1,
    density_state.eps2,
    density_state.smoothness_duration,
    density_state.smooth_counters,
    density_state.is_smooth,
    density_state.has_decreased,
    density_state.stable_duration,
    density_state.stable_counters,
    density_state.is_stable,
  )

  return nothing

end

abstract type AbstractParallelEstimator{N,T,M} <: AbstractEstimator end

struct ParallelEstimator{N,T<:Real,M} <: AbstractParallelEstimator{N,T,M}
  means_bootstraps::KernelMeans{N,T,M}
  vars_bootstraps::KernelVars{N,T,M}
  means::KernelMeans{N,T,N}
  kernel_propagation::KernelPropagation{N,T,M}
  grid_direct::Grid{N,T,M}
  grid_fourier::Grid{N,T,M}
  times::Vector{SVector{N,<:Real}}
  dt::SVector{N,<:Real}
  density_state::DensityState
end

struct CuParallelEstimator{N,T<:Real,M} <: AbstractParallelEstimator{N,T,M}
  means_bootstraps::CuKernelMeans{N,T,M}
  vars_bootstraps::CuKernelVars{N,T,M}
  means::CuKernelMeans{N,T,N}
  kernel_propagation::CuKernelPropagation{N,T,M}
  grid_direct::CuGrid{N,T,M}
  grid_fourier::CuGrid{N,T,M}
  times::CuMatrix{<:Real}
  dt::CuVector{<:Real}
  density_state::CuDensityState
end

add_estimator!(:parallelEstimator, ParallelEstimator)

Device(::ParallelEstimator) = IsCPU()
Device(::CuParallelEstimator) = IsCUDA()

function initialize_estimator(
  ::Type{<:AbstractParallelEstimator},
  kde::AbstractKDE;
  kwargs...
)::AbstractParallelEstimator
  device = Device(kde)

  if !haskey(kwargs, :grid)
    throw(ArgumentError("Missing required keyword argument: 'grid'"))
  end

  grid = kwargs[:grid]
  if Device(grid) != device
    throw(ArgumentError("KDE device $device does not match Grid device $(Device(grid))"))
  end

  n_bootstraps = get(kwargs, :n_bootstraps, 100)
  time_step = get(kwargs, :dt, nothing)
  n_steps = get(kwargs, :n_steps, nothing)

  method = get(kwargs, :method, CPU_SERIAL)

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
    n_steps
  )

end

function initialize_estimator_propagation(device::AbstractDevice, args...)::AbstractKernelPropagation
  throw(ArgumentError("Propagation not implemented for device: $(typeof(device))"))
end
function initialize_estimator_propagation(
  ::IsCPU,
  kde::KDE{N,T,S},
  means_bootstraps::KernelMeans{N,T,M},
  vars_bootstraps::KernelVars{N,T,M},
  means::KernelMeans{N,T,N},
  grid::Grid{N,P,M};
  time_step::Union{Nothing,Real}=nothing,
  n_steps::Union{Nothing,Integer}=nothing,
) where {N,T<:Real,S<:Real,P<:Real,M}
  grid_fourier = fftgrid(grid)

  kernel_propagation = KernelPropagation(means_bootstraps, vars_bootstraps)
  time_final = silverman_rule(get_data(kde))
  times, dt = get_time(IsCPU(), time_final; time_step, n_steps)

  density_state = DensityState(size(grid), T; dt=norm(dt))

  return ParallelEstimator(
    means_bootstraps,
    vars_bootstraps,
    means,
    kernel_propagation,
    grid,
    grid_fourier,
    times,
    dt,
    density_state,
  )
end
function initialize_estimator_propagation(
  ::IsCUDA,
  kde::CuKDE{N,T,S},
  means_bootstraps::CuKernelMeans{N,T,M},
  vars_bootstraps::CuKernelVars{N,T,M},
  means::CuKernelMeans{N,T,M},
  grid::CuGrid{N,P,M};
  time_step::Union{Nothing,Real}=nothing,
  n_steps::Union{Nothing,Integer}=nothing,
) where {N,T<:Real,S<:Real,P<:Real,M}
  grid_fourier = fftgrid(grid)

  kernel_propagation = CuKernelPropagation(means_bootstraps, vars_bootstraps)
  time_final = silverman_rule(Array(get_data(kde)))
  times, dt = get_time(IsCUDA(), time_final; time_step, n_steps)

  density_state = CuDensityState(size(grid), typeof(kde).parameters[2]; dt=norm(dt))

  return CuParallelEstimator(
    means_bootstraps,
    vars_bootstraps,
    means,
    kernel_propagation,
    grid,
    grid_fourier,
    times,
    dt,
    density_state,
  )
end

function get_time(
  ::IsCPU,
  time_final::Union{Real,AbstractVector{<:Real}};
  n_steps::Union{Nothing,Integer}=nothing,
  time_step::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
)
  if time_final isa Real
    time_final = SVector{1,Float64}(time_final)
  end

  if n_steps !== nothing
    dt = time_final ./ n_steps
    times = [dt .* i for i in 0:n_steps]

    return times, dt
  elseif time_step !== nothing
    if time_step isa Real
      time_step = SVector{1,Float64}(time_step)
    end
    if any(time_step .< time_final) || any(time_step .<= 0)
      throw(ArgumentError("Invalid time step"))
    end

    n_steps = minimum(floor.(time_final ./ time_step))
    dt = time_step
    times = [time_step .* i for i in 0:n_steps]

    return times, dt
  else

    return get_time(IsCPU(), time_final, n_steps=50)
  end

end
function get_time(
  ::IsCUDA,
  time_final::Union{Real,AbstractVector{<:Real}};
  n_steps::Union{Nothing,Integer}=nothing,
  time_step::Union{Nothing,Real,AbstractVector{<:Real}}=nothing,
)
  if time_final isa Real
    time_final = SVector{1,Float64}(time_final)
  end

  if n_steps !== nothing
    dt = CuArray{Float32}(time_final ./ n_steps)
    times = mapreduce(i -> dt .* i, hcat, 0:n_steps)

    return times, dt
  elseif time_step !== nothing
    if time_step isa Real
      time_step = SVector{1,Float64}(time_step)
    end
    if any(time_step .< time_final) || any(time_step .<= 0)
      throw(ArgumentError("Invalid time step"))
    end

    n_steps = minimum(floor.(time_final ./ time_step))
    dt = CuArray{Float32}(time_step)
    times = mapreduce(i -> dt .* i, hcat, 0:n_steps)

    return times, dt
  else

    return get_time(IsCUDA(), time_final, n_steps=50)
  end

end

function estimate!(
  estimator::AbstractParallelEstimator,
  kde::AbstractKDE{N,T,S};
  kwargs...
)::Nothing where {N,T,S}
  check_memory(estimator)

  device = Device(kde)
  method = get(kwargs, :method, CPU_SERIAL)
  n_samples = get_nsamples(kde)

  time_initial = initial_bandwidth(estimator.grid_direct)

  for time in estimator.times
    propagate_bootstraps!(
      estimator.kernel_propagation,
      estimator.means_bootstraps,
      estimator.vars_bootstraps,
      estimator.grid_fourier;
      time,
      time_initial,
      method
    )

    ifft_bootstraps!(estimator.kernel_propagation; method)

    calculate_vmr!(
      estimator.kernel_propagation,
      time,
      estimator.grid_direct,
      n_samples;
      method
    )

    propagate_means!(
      estimator.kernel_propagation,
      estimator.means,
      time,
      estimator.grid_fourier;
      method
    )

    ifft_means!(estimator.kernel_propagation; method)

    calculate_means!(
      estimator.kernel_propagation,
      n_samples;
      method
    )

    # Use DensityState to identify stopping time
    update_state!(
      estimator.density_state,
      kde,
      estimator.kernel_propagation;
      method
    )
  end

  return nothing
end

function get_necessary_memory(estimator::AbstractParallelEstimator)::Float64
  bootstraps_memory = sizeof(estimator.means_boostraps.statistic) * 4
  means_memory = sizeof(estimator.means.statistic) * 3

  return 1.25(bootstraps_memory + means_memory) / 1024^2
end

function check_memory(estimator::AbstractParallelEstimator)::Nothing
  available_memory = get_available_memory(Device(estimator))
  required_memory = get_necessary_memory(estimator)

  if available_memory < required_memory
    throw(
      ArgumentError("Not enough memory available. Required: $required_memory MiB, Available: $available_memory MiB")
    )
  end

  return nothing
end
