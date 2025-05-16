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

      @test all(isfinite.(means_bootstraps.statistic) .& .!iszero.(means_bootstraps.statistic))
      @test all(isfinite.(vars_bootstraps.statistic) .& .!iszero.(vars_bootstraps.statistic))
      @test means_bootstraps.bootstrapped == true
      @test vars_bootstraps.bootstrapped == true

      means = ParallelKDE.DensityEstimators.initialize_kernels(
        kde_device, kde, grid, n_bootstraps=0, method=:serial
      )

      @test all(isfinite.(means.statistic) .& .!iszero.(means.statistic))
      @test means.bootstrapped == false
    end
  end
end
