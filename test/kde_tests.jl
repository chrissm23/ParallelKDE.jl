@testset "CPU KDEs tests" for n_dims in 1:3
  n_samples = 100
  @testset "dimensions : $n_dims" begin
    data = generate_samples(n_samples, n_dims)
    grid_ranges = fill(-5.0:0.1:5.0, n_dims)
    kde = initialize_kde(data, grid_ranges, :cpu)

    if n_dims == 1
      @test get_data(kde) ≈ reshape(reinterpret(reshape, Float64, data), 1, :)
    else
      @test get_data(kde) ≈ reinterpret(reshape, Float64, data)
    end
    @test get_grid(kde) ≈ Grid(grid_ranges)
    @test get_time(kde) ≈ @SVector zeros(Float64, n_dims)
    @test all(isnan.(get_density(kde)))
  end
end

if CUDA.functional()
  @testset "GPU KDEs tests" for n_dims in 1:3
    n_samples = 100
    @testset "dimensions : $n_dims" begin
      data = generate_samples(n_samples, n_dims)
      grid_ranges = fill(-5.0:0.1:5.0, n_dims)
      kde = initialize_kde(data, grid_ranges, :gpu)
      if n_dims == 1
        @test get_data(kde) ≈ CuArray{Float32}(reshape(reinterpret(reshape, Float64, data), 1, :))
      else
        @test get_data(kde) ≈ CuArray{Float32}(reinterpret(reshape, Float64, data))
      end
      @test get_grid(kde) ≈ CuGrid(grid_ranges)
      @test get_time(kde) ≈ CUDA.zeros(Float32, n_dims)
      @test all(isnan.(get_density(kde)))
    end
  end
end
