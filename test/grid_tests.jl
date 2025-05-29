@testset "CPU grid tests. $(n_dims)D" for n_dims in 1:3
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

  grid = initialize_grid(grid_ranges, device=:cpu)

  grid_fourier = fftgrid(grid)
  fourier_grid = initialize_grid(fourier_grid_ranges, device=:cpu)

  @test size(grid) == ntuple(i -> n_steps, n_dims)
  @test ndims(grid) == n_dims
  @test spacings(grid) == spacings_vector
  @test bounds(grid) == bounds_matrix
  @test low_bounds(grid) == bounds_matrix[1, :]
  @test high_bounds(grid) == bounds_matrix[2, :]
  @test get_coordinates(grid_fourier) ≈ get_coordinates(fourier_grid)

  data = [SVector{Float64}(fill(-1.0, n_dims)), SVector{Float64}(fill(1.0, n_dims))]
  grid = find_grid(data)

  @test bounds(grid) == SMatrix{2,n_dims,Float64}([-1.2; 1.2] .* ones(1, n_dims))
end

if CUDA.functional()
  @testset "GPU grid tests. $(n_dims)D" for n_dims in 1:3
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

    grid = initialize_grid(grid_ranges, device=:cuda)

    grid_fourier = fftgrid(grid)
    fourier_grid = initialize_grid(fourier_grid_ranges, device=:cuda)

    @test size(grid) == ntuple(i -> n_steps, n_dims)
    @test ndims(grid) == n_dims
    @test spacings(grid) == spacings_vector
    @test bounds(grid) == bounds_matrix
    @test low_bounds(grid) == bounds_matrix[1, :]
    @test high_bounds(grid) == bounds_matrix[2, :]
    @test Base.broadcastable(grid) isa CuArray{Float32,n_dims + 1}
    @test get_coordinates(grid_fourier) ≈ get_coordinates(fourier_grid)
    @test initial_bandwidth(grid) == CUDA.fill(0.05f0, n_dims)

    data = CuMatrix{Float32}(ones(n_dims, 1) * [-1.0 1.0])
    grid = find_grid(data, device=:cuda)

    @test bounds(grid) == CuMatrix{Float32}([-1.2; 1.2] .* ones(1, n_dims))
  end
end
