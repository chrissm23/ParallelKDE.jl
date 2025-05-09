@testset "CPU KDEs tests. $(n_dims)D" for n_dims in 1:3
  n_samples = 100
  data = generate_samples(n_samples, n_dims)
  grid_ranges = fill(-5.0:0.1:5.0, n_dims)
  grid_size = Tuple(length.(grid_ranges))

  kde = initialize_kde(data, grid_size, device=:cpu)

  @test get_nsamples(kde) == n_samples
  @test all(isnan.(get_density(kde)))

  if n_dims == 1
    @test get_data(kde) ≈ reshape(reinterpret(reshape, Float64, data), 1, :)
  else
    @test get_data(kde) ≈ reinterpret(reshape, Float64, data)
  end

  bootstraps = bootstrap_indices(kde, 2)
  @test size(bootstraps) == (n_samples, 2)
  @test all(isa.(bootstraps, Integer))
  @test all(in.(bootstraps, Ref(1:n_samples)))

  set_density!(kde, zeros(Float64, grid_size))
  @test all(get_density(kde) .== 0)
end

if CUDA.functional()
  @testset "GPU KDEs tests. $(n_dims)D" for n_dims in 1:3
    n_samples = 100
    data = generate_samples(n_samples, n_dims)
    grid_ranges = fill(-5.0:0.1:5.0, n_dims)
    grid_size = Tuple(length.(grid_ranges))

    kde = initialize_kde(data, grid_size, device=:cuda)

    @test get_nsamples(kde) == n_samples
    @test all(isnan.(get_density(kde)))

    if n_dims == 1
      @test Array(get_data(kde)) ≈ reshape(reinterpret(reshape, Float64, data), 1, :)
    else
      @test Array(get_data(kde)) ≈ reinterpret(reshape, Float64, data)
    end

    bootstraps = bootstrap_indices(kde, 2)
    @test size(bootstraps) == (n_samples, 2)
    @test all(isa.(bootstraps, Integer))
    @test all(in.(bootstraps, Ref(1:n_samples)))

    set_density!(kde, CUDA.zeros(Float32, grid_size))
    @test all(get_density(kde) .== 0)
  end
end
