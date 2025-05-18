@testset "CPU parallel estimation tests. $(n_dims)D" for n_dims in 1:3
  n_samples = 1000
  n_bootstraps = 30

  grid_ranges = fill(-5.0:0.1:5.0, n_dims)
  grid = initialize_grid(grid_ranges)

  data = generate_samples(n_samples, n_dims)
  kde = initialize_kde(data, size(grid), device=:cpu)
  kde_device = ParallelKDE.Device(kde)

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
        kde_device, kde, grid, n_bootstraps=0, method=:serial
      )

      @test all(isfinite.(means.statistic)) & !all(iszero.(means.statistic))
      @test means.bootstrapped == false

      means_bootstraps_array = Array{ComplexF64}(undef, size(grid)..., n_bootstraps)
      fill!(means_bootstraps_array, NaN)
      vars_bootstraps_array = Array{ComplexF64}(undef, size(grid)..., n_bootstraps)
      fill!(vars_bootstraps_array, NaN)
      means_array = Array{ComplexF64}(undef, size(grid)..., 1)
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
        kde_device, kde, grid, n_bootstraps=0, method=:serial
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
      # TODO:
      # - Test creation of DensityState
      # - Test updating state after tests for identify convergence (todo)
    end

    @testset "Parallel estimation tests" begin
      # TODO:
      # - Test initialization of estimator and kernel propagation
      # - Test creation of ParallelEstimation
    end

    @testset "Time settings tests" begin
      # TODO:
      # - Test correct creation of time vectors
    end

    @testset "Complete parallel estimation tests" begin
      # TODO:
      # - Test the whole parallel estimation method (todo)
    end
  end
end
