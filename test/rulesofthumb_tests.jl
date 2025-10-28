@testset "CPU rules of thumb interfaces tests. $(n_dims)D" for n_dims in 1:3
  n_samples = 1000

  grid_ranges = fill(-5.0:0.5:5.0, n_dims)
  grid = initialize_grid(grid_ranges)

  data = generate_samples(n_samples, n_dims)
  kde = initialize_kde(data, size(grid), device=:cpu)

  data = get_data(kde)
  kde_device = ParallelKDE.get_device(kde)

  @testset "Rule of thumb: $rule_of_thumb tests" for rule_of_thumb in [:silverman, :scott]
    rot = ParallelKDE.DensityEstimators.initialize_rot(Val(rule_of_thumb), n_dims, n_samples)

    @test ParallelKDE.DensityEstimators.propagation_time(rot, data) isa AbstractVector{Float64}
  end

  @testset "Implementation: $implementation tests" for implementation in [:serial, :threaded]
    initial_density = ParallelKDE.DensityEstimators.initialize_density(
      kde_device,
      kde,
      grid;
      method=implementation
    )
    @test initial_density isa Array{ComplexF64,n_dims + 1}

    estimator = ParallelKDE.DensityEstimators.initialize_estimator(
      ParallelKDE.DensityEstimators.initialize_rot(Val(:silverman), n_dims, n_samples),
      data,
      initial_density,
      grid;
      device=kde_device,
    )
    @test estimator isa ParallelKDE.DensityEstimators.AbstractRoTEstimator

    propagated_density = similar(initial_density)
    ParallelKDE.DensityEstimators.propagate_kernels!(
      propagated_density,
      estimator;
      method=implementation
    )
    @test !all(propagated_density .== estimator.fourier_density)

    ParallelKDE.DensityEstimators.ifft_density!(propagated_density, kde; method=implementation)
    @test !any(isnan, kde.density)
  end
end
if CUDA.functional()
  @testset "GPU rules of thumb interfaces tests. $(n_dims)D" for n_dims in 1:3
    n_samples = 1000

    grid_ranges = fill(-5.0:0.5:5.0, n_dims)
    grid = initialize_grid(grid_ranges, device=:cuda)

    data = generate_samples(n_samples, n_dims)
    kde = initialize_kde(data, size(grid), device=:cuda)

    data = get_data(kde)
    kde_device = ParallelKDE.get_device(kde)

    initial_density = ParallelKDE.DensityEstimators.initialize_density(
      kde_device,
      kde,
      grid;
      method=:cuda,
    )
    @test initial_density isa CuArray{ComplexF32,n_dims + 1}

    estimator = ParallelKDE.DensityEstimators.initialize_estimator(
      ParallelKDE.DensityEstimators.initialize_rot(Val(:silverman), n_dims, n_samples),
      data,
      initial_density,
      grid;
      device=kde_device,
    )
    @test estimator isa ParallelKDE.DensityEstimators.CuRoTEstimator

    propagated_density = similar(initial_density)
    ParallelKDE.DensityEstimators.propagate_kernels!(
      propagated_density,
      estimator;
      method=:cuda,
    )
    @test !all(propagated_density .== estimator.fourier_density)

    ParallelKDE.DensityEstimators.ifft_density!(propagated_density, kde; method=:cuda)
    @test !any(isnan, kde.density)
  end
end

@testset "Rules of Thumb estimation tests (CPU). $(n_dims)D" for n_dims in 1:2
  @testset "Implementation: $implementation tests" for implementation in [:serial, :threaded]
    data = generate_samples(1000, n_dims)

    grid = initialize_grid(fill(range(-6.0, 6.0, length=300), n_dims), device=:cpu)

    @testset "Rule of thumb: $rule_of_thumb tests" for rule_of_thumb in [:silverman, :scott]
      kde = initialize_kde(data, size(grid), device=:cpu)

      estimator = ParallelKDE.DensityEstimators.initialize_estimator(
        ParallelKDE.DensityEstimators.AbstractRoTEstimator,
        kde,
        grid=grid,
        method=implementation
      )
      ParallelKDE.DensityEstimators.estimate!(
        estimator, kde; method=implementation, rule_of_thumb
      )
      density_estimated = get_density(kde)

      density_mean = @SVector zeros(n_dims)
      grid_vectors = eachslice(get_coordinates(grid), dims=Tuple(2:n_dims+1))

      density_true = normal_distribution.(grid_vectors, Ref(density_mean), Ref(Diagonal(ones(n_dims))))

      dx = prod(spacings(grid))
      mise = calculate_mise(density_estimated, density_true, dx)

      @test mise < 1e-3
    end
  end
end
if CUDA.functional()
  @testset "Parallel estimation tests (GPU). $(n_dims)D" for n_dims in 1:2
    data = generate_samples(1000, n_dims)

    grid = initialize_grid(fill(range(-6.0, 6.0, length=300), n_dims), device=:cuda)

    @testset "Rule of thumb: $rule_of_thumb tests" for rule_of_thumb in [:silverman, :scott]
      kde = initialize_kde(data, size(grid), device=:cuda)

      estimator = ParallelKDE.DensityEstimators.initialize_estimator(
        ParallelKDE.DensityEstimators.AbstractRoTEstimator,
        kde,
        grid=grid,
        method=:cuda
      )
      ParallelKDE.DensityEstimators.estimate!(estimator, kde; method=:cuda, rule_of_thumb)
      density_estimated_d = get_density(kde)
      density_estimated = Array{Float64}(density_estimated_d)

      density_mean = @SVector zeros(n_dims)
      grid_vectors = eachslice(Array(get_coordinates(grid)), dims=Tuple(2:n_dims+1))

      density_true = normal_distribution.(grid_vectors, Ref(density_mean), Ref(Diagonal(ones(n_dims))))

      dx = prod(spacings(grid))
      mise = calculate_mise(density_estimated, density_true, dx)

      @test mise < 1e-3
    end
  end
end
