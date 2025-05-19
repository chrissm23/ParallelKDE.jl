@testset "CPU DensityEstimation tests. $(n_dims)D" for n_dims in 1:3
  n_samples = 1000
  data = generate_samples(n_samples, n_dims)
  dims = ntuple(i -> 101, n_dims)

  estimation = initialize_estimation(data, device=:cpu, dims=dims, grid=true)

  @test estimation.kde isa ParallelKDE.KDEs.KDE
  @test estimation.grid isa ParallelKDE.Grids.Grid

  estimation = initialize_estimation(data, device=:cpu, dims=dims, grid=false)

  @test !ParallelKDE.has_grid(estimation)
end
if CUDA.functional()
  @testset "GPU DensityEstimation tests. $(n_dims)D" for n_dims in 1:3
    n_samples = 1000
    data = generate_samples(n_samples, n_dims)
    dims = ntuple(i -> 101, n_dims)

    estimation = initialize_estimation(data, device=:cuda, dims=dims, grid=true)

    @test estimation.kde isa ParallelKDE.KDEs.CuKDE
    @test estimation.grid isa ParallelKDE.Grids.CuGrid

    estimation = initialize_estimation(data, device=:cuda, dims=dims, grid=false)

    @test !ParallelKDE.has_grid(estimation)
  end
end
