abstract type AbstractKernelStatistics{N,T,M} end
abstract type AbstractKernelMeans{N,T,M} <: AbstractKernelStatistics{N,T,M} end
abstract type AbstractKernelVars{N,T,M} <: AbstractKernelStatistics{N,T,M} end

struct KernelMeans{N,T<:Real,M} <: AbstractKernelMeans{N,T,M}
  statistic::Array{Complex{T},M}
  bootstrapped::Bool

  function KernelMeans(statistic::Array{Complex{T},M}, bootstrapped::Bool)
    if bootstrapped
      return new{M - 1,T,M}(statistic, bootstrapped)
    else
      return new{M,T,M}(statistic, bootstrapped)
    end
  end
end
struct CuKernelMeans{T<:Real,M} <: AbstractKernelMeans{T,M}
  statistic::CuArray{Complex{T},M}
  bootstrapped::Bool

  function CuKernelMeans(statistic::CuArray{Complex{T},M}, bootstrapped::Bool)
    if bootstrapped
      return new{M - 1,T,M}(statistic, bootstrapped)
    else
      return new{M,T,M}(statistic, bootstrapped)
    end
  end
end
struct KernelVars{T<:Real,M} <: AbstractKernelVars{T,M}
  statistic::Array{Complex{T},M}
  bootstrapped::Bool

  function KernelVars(statistic::Array{Complex{T},M}, bootstrapped::Bool)
    if bootstrapped
      return new{M - 1,T,M}(statistic, bootstrapped)
    else
      return new{M,T,M}(statistic, bootstrapped)
    end
  end
end
struct CuKernelVars{T<:Real,M} <: AbstractKernelVars{T,M}
  statistic::CuArray{Complex{T},M}
  bootstrapped::Bool

  function CuKernelVars(statistic::CuArray{Complex{T},M}, bootstrapped::Bool)
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
  kde::AbstractKDE{N,T,S,M},
  grid::AbstractGrid{N,P,M};
  n_bootstraps::Integer=0,
  include_var::Bool=false,
  method::Symbol=:serial,
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
  fourier_statistics!(Val(method), plan, means)
  if include_var
    fourier_statistics!(Val(method), plan, vars)
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
  kde::AbstractKDE{N,T,S,M},
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
  fourier_statistics!(Val(:cuda), plan, means)
  if include_var
    fourier_statistics!(Val(:cuda), plan, vars)
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
  time::AbstractVector{<:Real},
  grid::AbstractGrid{N,P,M};
  method::Symbol=:serial,
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

  time_initial = initial_bandwidth(grid)
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

abstract type AbstractKernelPropagation{N,T<:Real,M} end

struct KernelPropagation{N,T<:Real,M} <: AbstractKernelPropagation{N,T,M}
  kernel_means::Array{Complex{T},M}
  kernel_vars::Array{Complex{T},M}
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}}
  calculated_vmr::Bool

  function KernelPropagation(
    kernel_means::KernelMeans{N,T,M},
    kernel_vars::KernelVars{N,T,M},
  )
    if !(is_bootstrapped(kernel_means) && is_bootstrapped(kernel_vars))
      throw(ArgumentError("Both kernel means and kernel vars must be bootstrapped"))
    end

    plan = plan_ifft!(selectdim(kernel_means, get_ndims(kernel_means) + 1, 1))

    new{N,T,M}(similar(kernel_means.statistic), similar(kernel_vars.statistic), plan, false)
  end
end
struct CuKernelPropagation{N,T<:Real,M} <: AbstractKernelPropagation{N,T,M}
  kernel_means::CuArray{Complex{T},M}
  kernel_vars::CuArray{Complex{T},M}
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}}
  calculated_vmr::Bool

  function CuKernelPropagation(
    kernel_means::CuKernelMeans{T,M},
    kernel_vars::CuKernelVars{T,M},
  )
    if !(is_bootstrapped(kernel_means) && is_bootstrapped(kernel_vars))
      throw(ArgumentError("Both kernel means and kernel vars must be bootstrapped"))
    end

    plan = plan_ifft!(kernel_means, ntuple(i -> i, get_ndims(kernel_means)))

    new{N,T,M}(similar(kernel_means.statistic), similar(kernel_vars.statistic), plan, false)
  end

end

Device(::KernelPropagation) = IsCPU()
Device(::CuKernelPropagation) = IsCUDA()

function is_vmr_calculated(kernel_propagation::AbstractKernelPropagation{T,M})::Bool where {T,M}
  return kernel_propagation.calculated_vmr
end

function propagate_bootstraps!(
  kernel_propagation::AbstractKernelPropagation{N,T,M},
  kernel_means::AbstractKernelMeans{N,T,M},
  kernel_vars::AbstractKernelVars{N,T,M},
  time::AbstractVector{<:Real},
  grid::AbstractGrid{N,P,M};
  method::Symbol=:serial,
) where {N,T<:Real,M,P<:Real}
  propagate!(
    kernel_propagation.kernel_means,
    kernel_propagation.kernel_vars,
    kernel_means,
    kernel_vars,
    time,
    grid;
    method,
  )

  kernel_propagation.calculated_vmr = false

  return nothing
end

function calculate_vmr!(
  kernel_propagation::AbstractKernelPropagation{N,T,M},
  n_samples::Integer;
  method::Symbol=:serial,
) where {N,T<:Real,M}
  ifourier_statistics!(
    Val(method),
    kernel_propagation.kernel_means,
    kernel_propagation.kernel_vars,
    n_samples,
    kernel_propagation.ifft_plan,
  )

  # TODO: Calculate VMR from only real part (how to do this fast without new allocation?)

  kernel_propagation.calculated_vmr = true
end

abstract type AbstractDensityState end

# TODO: Define DensityState
struct DensityState <: AbstractDensityState
end

struct CuDensityState <: AbstractDensityState
end

abstract type AbstractParallelEstimation{N,T,M} <: AbstractEstimation end

struct ParallelEstimation{N,T,M} <: AbstractParallelEstimation
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

struct CuParallelEstimation{N,T,M} <: AbstractParallelEstimation
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

add_method!(:parallelEstimation, ParallelEstimation)

function initialize_estimation(
  ::Type{<:AbstractParallelEstimation},
  kde::AbstractKDE;
  kwargs...
)::AbstractParallelEstimation
  device = Device(kde)

  if !haskey(kwargs, :grid)
    throw(ArgumentError("Missing required keyword argument: 'grid'"))
  end

  grid = kwargs[:grid]
  if Device(grid) != device
    throw(ArgumentError("Grid device $device does not match KDE device $(Device(grid))"))
  end
  grid_fourier = fftgrid(grid)

  n_bootstraps = get(kwargs, :n_bootstraps, 100)
  method = get(kwargs, :method, :serial)
  time_step = get(kwargs, :dt, nothing)
  n_steps = get(kwargs, :n_steps, nothing)

  means_bootstraps, vars_bootstraps = initialize_kernels(
    device, kde, grid; n_bootstraps=n_bootstraps, include_var=true, method=method
  )
  means = initialize_kernels(
    device, kde, grid; n_bootstraps=0, include_var=false, method=method
  )

  # TODO: Initialize density state

  if device == IsCPU()
    kernel_propagation = KernelPropagation(means_bootstraps, vars_bootstraps)
    time_final = silverman_rule(get_data(kde))
    times, dt = get_time(IsCPU(), time_final; time_step, n_steps)

    return ParallelEstimation(
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
  elseif device == IsCUDA()
    time_final = silverman_rule(Array(get_data(kde)))
    kernel_propagation = CuKernelPropagation(means_bootstraps, vars_bootstraps)
    times, dt = get_time(IsCUDA(), time_final; time_step, n_steps)

    return CuParallelEstimation(
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
  else
    throw(ArgumentError("Parallel estimation not implemented for device: $device"))
  end

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

    return get_time(::IsCPU, time_final, n_steps=50)
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

    return get_time(::IsCUDA, time_final, n_steps=50)
  end

end

function estimate!(
  kde::AbstractKDE{N,T,S,M},
  estimation::AbstractParallelEstimation;
  kwargs...
)::Nothing where {N,T,S,M}
  # TODO: Implement specific function of parallel estimation

  # Calculate in KernelPropagation arrays the kernel means and kernel vars according to time
  # Inverse Fourier transform in place
  # Calculate VMR which reduces the memory necessary to only one element in bootstrap dimension
  # Propagate means of all samples according to time in one element of bootstrap dimension
  # Use DensityState to identify stopping time

  return nothing
end
