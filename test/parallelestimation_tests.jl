@testset "CPU parallel estimation interfaces tests. $(n_dims)D" for n_dims in 1:3
  n_samples = 1000
  n_bootstraps = 30

  grid_ranges = fill(-5.0:0.5:5.0, n_dims)
  grid = initialize_grid(grid_ranges)

  data = generate_samples(n_samples, n_dims)
  kde = initialize_kde(data, size(grid), device=:cpu)
  kde_device = ParallelKDE.get_device(kde)

  @testset "Implementation: $implementation tests" for implementation in [:serial, :threaded]
    @testset "Kernel statistics tests" begin
      means_bootstraps, vars_bootstraps = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=n_bootstraps, include_var=true, method=implementation
      )

      @test all(isfinite.(means_bootstraps.statistic)) & !all(iszero.(means_bootstraps.statistic))
      @test all(isfinite.(vars_bootstraps.statistic)) & !all(iszero.(vars_bootstraps.statistic))
      @test means_bootstraps.bootstrapped == true
      @test vars_bootstraps.bootstrapped == true

      means = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=0, method=implementation
      )

      @test all(isfinite.(means.statistic)) & !all(iszero.(means.statistic))
      @test means.bootstrapped == false

      means_bootstraps_array = Array{ComplexF64}(undef, size(grid)..., n_bootstraps)
      fill!(means_bootstraps_array, NaN)
      vars_bootstraps_array = Array{ComplexF64}(undef, size(grid)..., n_bootstraps)
      fill!(vars_bootstraps_array, NaN)
      means_array = Array{ComplexF64}(undef, size(grid))
      fill!(means_array, NaN)

      ParallelKDE.DensityEstimators.propagate!(
        means_bootstraps_array,
        vars_bootstraps_array,
        means_bootstraps,
        vars_bootstraps,
        grid,
        SVector{n_dims,Float64}(fill(0.2, n_dims)),
        initial_bandwidth(grid),
        method=implementation,
      )
      ParallelKDE.DensityEstimators.propagate!(
        means_array,
        means,
        grid,
        SVector{n_dims,Float64}(fill(0.2, n_dims)),
        method=implementation,
      )

      @test all(means_bootstraps_array .!= NaN) & any(means_bootstraps_array .!= means_bootstraps.statistic)
      @test all(vars_bootstraps_array .!= NaN) & any(vars_bootstraps_array .!= vars_bootstraps.statistic)
      @test all(means_array .!= NaN) & any(means_array .!= means.statistic)
    end

    @testset "Kernel propagation tests" begin
      means_bootstraps, vars_bootstraps = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=n_bootstraps, include_var=true, method=implementation
      )
      means = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=0, method=implementation
      )

      kernel_propagation = ParallelKDE.DensityEstimators.KernelPropagation(
        means_bootstraps, vars_bootstraps
      )

      @test (kernel_propagation.kernel_means isa Array{ComplexF64,n_dims + 1}) & (kernel_propagation.kernel_vars isa Array{ComplexF64,n_dims + 1})
      @test !ParallelKDE.DensityEstimators.is_vmr_calculated(kernel_propagation)
      @test !ParallelKDE.DensityEstimators.is_means_calculated(kernel_propagation)

      ParallelKDE.DensityEstimators.propagate_bootstraps!(
        kernel_propagation,
        means_bootstraps,
        vars_bootstraps,
        grid,
        SVector{n_dims,Float64}(fill(0.2, n_dims)),
        initial_bandwidth(grid),
        method=implementation,
      )
      ParallelKDE.DensityEstimators.ifft_bootstraps!(
        kernel_propagation, method=implementation
      )
      ParallelKDE.DensityEstimators.calculate_vmr!(
        kernel_propagation,
        SVector{n_dims,Float64}(fill(0.2, n_dims)),
        grid,
        n_samples,
        method=implementation
      )

      @test ParallelKDE.DensityEstimators.is_vmr_calculated(kernel_propagation)

      vmr = ParallelKDE.DensityEstimators.get_vmr(kernel_propagation)

      @test vmr isa AbstractArray{Float64,n_dims}
      @test size(vmr) == size(grid)
      @test all(isfinite.(vmr))

      ParallelKDE.DensityEstimators.propagate_means!(
        kernel_propagation,
        means,
        grid,
        SVector{n_dims,Float64}(fill(0.2, n_dims)),
        method=implementation,
      )
      ParallelKDE.DensityEstimators.ifft_means!(
        kernel_propagation, method=implementation
      )
      ParallelKDE.DensityEstimators.calculate_means!(
        kernel_propagation, n_samples, method=implementation
      )

      @test ParallelKDE.DensityEstimators.is_means_calculated(kernel_propagation) == true

      means_complete = ParallelKDE.DensityEstimators.get_means(kernel_propagation)

      @test means_complete isa AbstractArray{Float64,n_dims}
      @test size(means_complete) == size(grid)
      @test all(isfinite.(means_complete))
    end

    @testset "Density state tests" begin
      density_state = ParallelKDE.DensityEstimators.DensityState(size(grid), dt=0.2)

      # Parameter tests
      @test density_state.dt == 0.2
      @test isfinite(density_state.eps1)
      @test isfinite(density_state.eps2)
      @test isfinite(density_state.smoothness_duration)
      @test isfinite(density_state.stable_duration)
      # State arrays test
      @test density_state.smooth_counters isa Array{Int8,n_dims}
      @test density_state.stable_counters isa Array{Int8,n_dims}
      @test density_state.is_smooth isa Array{Bool,n_dims}
      @test density_state.has_decreased isa Array{Bool,n_dims}
      @test density_state.is_stable isa Array{Bool,n_dims}
      # Buffers tests
      @test all(isnan.(density_state.f_prev1))
      @test all(isnan.(density_state.f_prev2))

      means_bootstraps, vars_bootstraps = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=n_bootstraps, include_var=true, method=implementation
      )
      kernel_propagation = ParallelKDE.DensityEstimators.KernelPropagation(
        means_bootstraps, vars_bootstraps
      )
      kernel_propagation.kernel_means .= 1.0
      kernel_propagation.kernel_vars .= 5.0
      kernel_propagation.calculated_vmr = true
      kernel_propagation.calculated_means = true

      density_state.f_prev1 .= 1.0
      density_state.f_prev2 .= 2.0

      ParallelKDE.DensityEstimators.update_state!(density_state, kde, kernel_propagation, method=implementation)

      @test all(density_state.f_prev1 .== ParallelKDE.DensityEstimators.get_vmr(kernel_propagation))
      @test all(density_state.f_prev2 .== 1.0)
    end

    @testset "Time settings tests" begin
      times, dt = ParallelKDE.DensityEstimators.get_time(
        kde_device,
        fill(2.0, n_dims),
        time_step=0.2,
      )

      @test times isa AbstractVector{<:SVector{n_dims,<:Real}}
      @test all(times[begin] .== fill(0.0, n_dims))
      @test all(times[end] .== fill(2.0, n_dims))
      @test all(dt .== fill(0.2, n_dims))
    end

    @testset "Parallel estimation tests" begin
      means_bootstraps, vars_bootstraps = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=n_bootstraps, include_var=true, method=implementation
      )
      means = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=0, method=implementation
      )
      kernel_propagation = ParallelKDE.DensityEstimators.KernelPropagation(
        means_bootstraps, vars_bootstraps
      )

      grid_fourier = fftgrid(grid)

      density_state = ParallelKDE.DensityEstimators.DensityState(size(grid), dt=0.2)

      dt = @SVector fill(0.2, n_dims)

      parallel_estimator = ParallelKDE.DensityEstimators.ParallelEstimator(
        means_bootstraps,
        vars_bootstraps,
        means,
        kernel_propagation,
        grid,
        grid_fourier,
        [dt .* i for i in 0:10],
        dt,
        density_state,
      )

      @test parallel_estimator.means_bootstraps isa ParallelKDE.DensityEstimators.KernelMeans{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.vars_bootstraps isa ParallelKDE.DensityEstimators.KernelVars{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.means isa ParallelKDE.DensityEstimators.KernelMeans{n_dims,Float64,n_dims}
      @test parallel_estimator.kernel_propagation isa ParallelKDE.DensityEstimators.KernelPropagation{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.grid_direct isa ParallelKDE.Grids.Grid{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.grid_fourier isa ParallelKDE.Grids.Grid{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.density_state isa ParallelKDE.DensityEstimators.DensityState

      parallel_estimator = nothing

      parallel_estimator = ParallelKDE.DensityEstimators.initialize_estimator(
        ParallelKDE.DensityEstimators.AbstractParallelEstimator,
        kde,
        grid=grid,
        n_bootstraps=n_bootstraps,
        time_step=0.2,
        n_steps=10,
        method=implementation,
      )

      @test parallel_estimator.means_bootstraps isa ParallelKDE.DensityEstimators.KernelMeans{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.vars_bootstraps isa ParallelKDE.DensityEstimators.KernelVars{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.means isa ParallelKDE.DensityEstimators.KernelMeans{n_dims,Float64,n_dims}
      @test parallel_estimator.kernel_propagation isa ParallelKDE.DensityEstimators.KernelPropagation{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.grid_direct isa ParallelKDE.Grids.Grid{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.grid_fourier isa ParallelKDE.Grids.Grid{n_dims,Float64,n_dims + 1}
      @test parallel_estimator.density_state isa ParallelKDE.DensityEstimators.DensityState
    end

  end
end
if CUDA.functional()
  @testset "GPU parallel estimation interfaces tests. $(n_dims)D" for n_dims in 1:3
    n_samples = 1000
    n_bootstraps = 30

    grid_ranges = fill(-5.0:0.5:5.0, n_dims)
    grid = initialize_grid(grid_ranges, device=:cuda)

    data_cpu = generate_samples(n_samples, n_dims)
    kde = initialize_kde(data_cpu, size(grid), device=:cuda)
    kde_device = ParallelKDE.get_device(kde)
    data = get_data(kde)

    @testset "Kernel statistics tests" begin
      means_bootstraps, vars_bootstraps = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=n_bootstraps, include_var=true
      )

      @test all(isfinite.(means_bootstraps.statistic)) & !all(iszero.(means_bootstraps.statistic))
      @test all(isfinite.(vars_bootstraps.statistic)) & !all(iszero.(vars_bootstraps.statistic))
      @test means_bootstraps.bootstrapped == true
      @test vars_bootstraps.bootstrapped == true

      means = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=0
      )

      @test all(isfinite.(means.statistic)) & !all(iszero.(means.statistic))
      @test means.bootstrapped == false

      means_bootstraps_array = CuArray{ComplexF32}(
        undef, size(grid)..., n_bootstraps
      )
      fill!(means_bootstraps_array, NaN32)
      vars_bootstraps_array = CuArray{ComplexF32}(
        undef, size(grid)..., n_bootstraps
      )
      fill!(vars_bootstraps_array, NaN32)
      means_array = CuArray{ComplexF32}(undef, size(grid))
      fill!(means_array, NaN32)

      ParallelKDE.DensityEstimators.propagate!(
        means_bootstraps_array,
        vars_bootstraps_array,
        means_bootstraps,
        vars_bootstraps,
        grid,
        CUDA.fill(0.2f0, n_dims),
        initial_bandwidth(grid),
        method=:cuda,
      )
      ParallelKDE.DensityEstimators.propagate!(
        means_array,
        means,
        grid,
        CUDA.fill(0.2f0, n_dims),
        method=:cuda,
      )

      @test all(means_bootstraps_array .!= NaN32) & any(means_bootstraps_array .!= means_bootstraps.statistic)
      @test all(vars_bootstraps_array .!= NaN32) & any(vars_bootstraps_array .!= vars_bootstraps.statistic)
      @test all(means_array .!= NaN32) & any(means_array .!= means.statistic)
    end

    @testset "Kernel propagation tests" begin
      means_bootstraps, vars_bootstraps = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=n_bootstraps, include_var=true
      )
      means = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=0
      )

      kernel_propagation = ParallelKDE.DensityEstimators.CuKernelPropagation(
        means_bootstraps, vars_bootstraps
      )

      @test (kernel_propagation.kernel_means isa CuArray{ComplexF32,n_dims + 1}) & (kernel_propagation.kernel_vars isa CuArray{ComplexF32,n_dims + 1})
      @test !ParallelKDE.DensityEstimators.is_vmr_calculated(kernel_propagation)
      @test !ParallelKDE.DensityEstimators.is_means_calculated(kernel_propagation)

      ParallelKDE.DensityEstimators.propagate_bootstraps!(
        kernel_propagation,
        means_bootstraps,
        vars_bootstraps,
        grid,
        CUDA.fill(0.2f0, n_dims),
        initial_bandwidth(grid),
        method=:cuda,
      )
      ParallelKDE.DensityEstimators.ifft_bootstraps!(
        kernel_propagation, method=:cuda
      )
      ParallelKDE.DensityEstimators.calculate_vmr!(
        kernel_propagation,
        CUDA.fill(0.2f0, n_dims),
        grid,
        n_samples,
        method=:cuda
      )

      @test ParallelKDE.DensityEstimators.is_vmr_calculated(kernel_propagation)

      vmr = ParallelKDE.DensityEstimators.get_vmr(kernel_propagation)

      @test vmr isa AbstractArray{Float32,n_dims}
      @test size(vmr) == size(grid)
      @test all(isfinite.(vmr))

      ParallelKDE.DensityEstimators.propagate_means!(
        kernel_propagation,
        means,
        grid,
        CUDA.fill(0.2f0, n_dims),
        method=:cuda,
      )
      ParallelKDE.DensityEstimators.ifft_means!(
        kernel_propagation, method=:cuda
      )
      ParallelKDE.DensityEstimators.calculate_means!(
        kernel_propagation, n_samples, method=:cuda
      )

      @test ParallelKDE.DensityEstimators.is_means_calculated(kernel_propagation) == true

      means_complete = ParallelKDE.DensityEstimators.get_means(kernel_propagation)

      @test means_complete isa AbstractArray{Float32,n_dims}
      @test size(means_complete) == size(grid)
      @test all(isfinite.(means_complete))
    end

    @testset "Density state tests" begin
      density_state = ParallelKDE.DensityEstimators.CuDensityState(size(grid), dt=0.2f0)

      # Parameter tests
      @test density_state.dt == 0.2f0
      @test isfinite(density_state.eps1)
      @test isfinite(density_state.eps2)
      @test isfinite(density_state.smoothness_duration)
      @test isfinite(density_state.stable_duration)
      # State arrays test
      @test density_state.smooth_counters isa CuArray{Int8,n_dims}
      @test density_state.stable_counters isa CuArray{Int8,n_dims}
      @test density_state.is_smooth isa CuArray{Bool,n_dims}
      @test density_state.has_decreased isa CuArray{Bool,n_dims}
      @test density_state.is_stable isa CuArray{Bool,n_dims}
      # Buffers tests
      @test all(isnan.(density_state.f_prev1))
      @test all(isnan.(density_state.f_prev2))

      means_bootstraps, vars_bootstraps = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=n_bootstraps, include_var=true, method=:cuda
      )
      kernel_propagation = ParallelKDE.DensityEstimators.CuKernelPropagation(
        means_bootstraps, vars_bootstraps
      )
      kernel_propagation.kernel_means .= 1.0
      kernel_propagation.kernel_vars .= 5.0
      kernel_propagation.calculated_vmr = true
      kernel_propagation.calculated_means = true

      density_state.f_prev1 .= 1.0
      density_state.f_prev2 .= 2.0

      ParallelKDE.DensityEstimators.update_state!(density_state, kde, kernel_propagation, method=:cuda)

      @test all(density_state.f_prev1 .== ParallelKDE.DensityEstimators.get_vmr(kernel_propagation))
      @test all(density_state.f_prev2 .== 1.0)
    end

    @testset "Time settings tests" begin
      times, dt = ParallelKDE.DensityEstimators.get_time(
        kde_device,
        CUDA.fill(2.0f0, n_dims),
        time_step=0.2f0,
      )

      @test times isa CuMatrix{Float32}
      @test all(times[:, begin] .== CUDA.fill(0.0f0, n_dims))
      @test all(times[:, end] .== CUDA.fill(2.0f0, n_dims))
      @test all(dt .== CUDA.fill(0.2f0, n_dims))
    end

    @testset "Parallel estimation tests" begin
      means_bootstraps, vars_bootstraps = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=n_bootstraps, include_var=true
      )
      means = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=0
      )
      kernel_propagation = ParallelKDE.DensityEstimators.CuKernelPropagation(
        means_bootstraps, vars_bootstraps
      )

      grid_fourier = fftgrid(grid)

      density_state = ParallelKDE.DensityEstimators.CuDensityState(size(grid), dt=0.2f0)

      dt = CUDA.fill(0.2f0, n_dims)

      parallel_estimator = ParallelKDE.DensityEstimators.CuParallelEstimator(
        means_bootstraps,
        vars_bootstraps,
        means,
        kernel_propagation,
        grid,
        grid_fourier,
        mapreduce(i -> dt .* i, hcat, 0:10),
        dt,
        density_state,
      )

      @test parallel_estimator.means_bootstraps isa ParallelKDE.DensityEstimators.CuKernelMeans{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.vars_bootstraps isa ParallelKDE.DensityEstimators.CuKernelVars{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.means isa ParallelKDE.DensityEstimators.CuKernelMeans{n_dims,Float32,n_dims}
      @test parallel_estimator.kernel_propagation isa ParallelKDE.DensityEstimators.CuKernelPropagation{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.grid_direct isa ParallelKDE.Grids.CuGrid{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.grid_fourier isa ParallelKDE.Grids.CuGrid{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.density_state isa ParallelKDE.DensityEstimators.CuDensityState

      parallel_estimator = nothing

      parallel_estimator = ParallelKDE.DensityEstimators.initialize_estimator(
        ParallelKDE.DensityEstimators.CuParallelEstimator,
        kde,
        grid=grid,
        n_bootstraps=n_bootstraps,
        time_step=0.2f0,
        n_steps=10,
        method=:cuda,
      )

      @test parallel_estimator.means_bootstraps isa ParallelKDE.DensityEstimators.CuKernelMeans{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.vars_bootstraps isa ParallelKDE.DensityEstimators.CuKernelVars{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.means isa ParallelKDE.DensityEstimators.CuKernelMeans{n_dims,Float32,n_dims}
      @test parallel_estimator.kernel_propagation isa ParallelKDE.DensityEstimators.CuKernelPropagation{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.grid_direct isa ParallelKDE.Grids.CuGrid{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.grid_fourier isa ParallelKDE.Grids.CuGrid{n_dims,Float32,n_dims + 1}
      @test parallel_estimator.density_state isa ParallelKDE.DensityEstimators.CuDensityState
    end

  end
end

# TODO: Extend tests to cover more dimensions once tested
@testset "Parallel estimation tests (CPU). $(n_dims)D" for n_dims in 1:1
  @testset "Implementation: $implementation tests" for implementation in [:serial, :threaded]
    data = generate_samples(1000, n_dims)

    grid = initialize_grid(fill(range(-6.0, 6.0, length=300), n_dims), device=:cpu)
    kde = inittailize_kde(data, size(grid), device=:cpu)

    estimator = ParallelKDE.DensityEstimators.inititalize_estimator(
      ParallelKDE.DensityEstimators.AbstractParallelEstimator,
      kde,
      grid=grid,
      method=implementation
    )
    estimate!(estimator, kde; method=implementation)
    density_estimated = get_density(kde)

    density_mean = @SVector zeros(n_dims)
    grid_vectors = eachslice(grid, dims=Tuple(2:n_dims+1))

    density_true = normal_distribution.(grid_vectors, Ref(density_mean), Ref(Diagonal(ones(n_dims))))

    dx = prod(spacings.(grid))
    mise = calculate_mise(density_estimated, density_true, dx)

    # TODO: Replace with appropriate MISE threshold
    @test mise < 0.01
  end
end

# TODO: Extend tests to cover more dimensions once tested
if CUDA.functional()
  @testset "Parallel estimation tests (GPU). $(n_dims)D" for n_dims in 1:1
    data = generate_samples(1000, n_dims)

    grid = initialize_grid(fill(range(-6.0, 6.0, length=300), n_dims), device=:cuda)
    kde = inittailize_kde(data, size(grid), device=:cuda)

    estimator = ParallelKDE.DensityEstimators.inititalize_estimator(
      ParallelKDE.DensityEstimators.AbstractParallelEstimator,
      kde,
      grid=grid,
      method=:cuda
    )
    estimate!(estimator, kde; method=:cuda)
    density_estimated_d = get_density(kde)
    density_estimated = Array{Float64}(density_estimated_d)

    density_mean = @SVector zeros(n_dims)
    grid_vectors = eachslice(grid, dims=Tuple(2:n_dims+1))

    density_true = normal_distribution.(grid_vectors, Ref(density_mean), Ref(Diagonal(ones(n_dims))))

    dx = prod(spacings.(grid))
    mise = calculate_mise(density_estimated, density_true, dx)

    # TODO: Replace with appropriate MISE threshold
    @test mise < 0.01
  end
end
