@testset "CPU grid tests" for n_dims in 1:3
  @testset "dimensions: $n_dims" begin
    grid_min = -1.0
    grid_max = 1.0
    grid_step = 0.1

    grid_range = grid_min:grid_step:grid_max
    n_steps = length(grid_range)

    fourier_range = 2π * fftfreq(n_steps, 1 / grid_step)

    grid_ranges = fill(grid_range, n_dims)
    fourier_grid_ranges = fill(fourier_range, n_dims)

    spacings_vector = @SVector fill(grid_step, n_dims)
    bounds_matrix = SMatrix{2,n_dims,Float64}(
      [grid_min; grid_max] .* ones(1, n_dims)
    )

    grid = ParallelKDE.Grids.Grid(grid_ranges)

    grid_fourier = ParallelKDE.Grids.fftgrid(grid)
    fourier_grid = ParallelKDE.Grids.Grid(grid_fourier_ranges)

    @test size(grid) == ntuple(i -> n_steps, n_dims)
    @test ndims(grid) == n_dims
    @test ParallelKDE.Grids.spacings(grid) == spacings_vector
    @test ParallelKDE.Grids.bounds(grid) == bounds_matrix
    @test ParallelKDE.Grids.low_bounds(grid) == bounds_matrix[1, :]
    @test ParallelKDE.Grids.high_bounds(grid) == bounds_matrix[2, :]
    @test Base.broadcastable(grid) isa AbstractArray{Float64,n_dims + 1}
    @test get_coordinates(grid_foureir) ≈ get_coordinates(fourier_grid)
  end
end
if CUDA.functional()
  @testset "GPU grid tests" for n_dims in 1:3
    @testset "dimensions : $n_dims" begin
      grid_min = -1.0
      grid_max = 1.0
      grid_step = 0.1

      grid_range = grid_min:grid_step:grid_max
      n_steps = length(grid_range)

      fourier_range = 2π * fftfreq(n_steps, 1 / grid_step)

      grid_ranges = fill(grid_range, n_dims)
      fourier_grid_ranges = fill(fourier_range, n_dims)

      spacings_vector = CUDA.fill(Float32(grid_step), n_dims)
      bounds_matrix = CuArray{Float32,2}(
        [grid_min; grid_max] .* ones(1, n_dims)
      )

      grid = ParallelKDE.Grids.CuGrid(grid_ranges)

      grid_fourier = ParallelKDE.Grids.fftgrid(grid)
      fourier_grid = ParallelKDE.Grids.CuGrid(fourier_grid_ranges)

      grid = CuGrid(grid_ranges, b32=true)

      @test size(grid) == ntuple(i -> n_steps, n_dims)
      @test ndims(grid) == n_dims
      @test ParallelKDE.Grids.spacings(grid) == spacings_vector
      @test ParallelKDE.Grids.bounds(grid) == bounds_matrix
      @test ParallelKDE.Grids.low_bounds(grid) == bounds_matrix[1, :]
      @test ParallelKDE.Grids.high_bounds(grid) == bounds_matrix[2, :]
      @test Base.broadcastable(grid) isa CuArray{Float32,n_dims + 1}
      @test get_coordinates(grid_fourier) ≈ get_coordinates(fourier_grid)
    end
  end
end
