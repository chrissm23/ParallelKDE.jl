function check_sample_frequencies(grid::Grid)
  fourier_range = fftshift(2π * fftfreq(21, 1 / 0.1))
  result_grid = Grid(fill(fourier_range, length(size(grid))))

  fourier_grid = fft_grid(grid)

  @test fourier_grid ≈ result_grid

  return nothing
end
function check_sample_frequencies(grid::CuGrid)
  fourier_range = fftshift(2π * fftfreq(21, 1 / 0.1))
  result_grid = Grid(fill(fourier_range, length(size(grid))))

  fourier_grid = fft_grid(grid)

  @test fourier_grid ≈ result_grid atol = 1.0f-1

  return nothing
end

@testset "CPU grid tests" for n_dims in 1:3
  @testset "dimensions : $n_dims" begin
    grid_ranges = fill(-1.0:0.1:1.0, n_dims)
    spacings_vector = @SVector fill(0.1, n_dims)
    bounds_matrix = SMatrix{2,n_dims,Float64}(
      [-1.0; 1.0] .* ones(1, n_dims)
    )

    grid = Grid(grid_ranges)
    @test size(grid) == ntuple(i -> length(grid_ranges[i]), n_dims)
    @test spacings(grid) == spacings_vector
    @test bounds(grid) == bounds_matrix
    @test low_bounds(grid) == bounds_matrix[1, :]
    @test high_bounds(grid) == bounds_matrix[2, :]

    @test isa(Base.broadcastable(grid), AbstractArray{Float64,n_dims + 1})

    @testset "Fourier sample frequencies" check_sample_frequencies(grid)
  end
end

if CUDA.functional()
  @testset "GPU grid tests" for n_dims in 1:3

    @testset "dimensions : $n_dims" begin
      grid_ranges = fill(-1.0:0.1:1.0, n_dims)
      spacings_vector = CUDA.fill(1.0f-1, n_dims)
      bounds_matrix = CuArray{Float32,2}(
        [-1.0; 1.0] .* ones(1, n_dims)
      )
      grid_array = reinterpret(
        reshape,
        Float32,
        collect(Iterators.product(grid_ranges...))
      )

      grid = CuGrid(grid_ranges, b32=true)
      @test size(grid) == size(grid_array)[2:end]
      @test spacings(grid) == spacings_vector
      @test bounds(grid) == bounds_matrix
      @test low_bounds(grid) == bounds_matrix[1, :]
      @test high_bounds(grid) == bounds_matrix[2, :]

      @test isa(Base.broadcastable(grid), AbstractArray{Float32,n_dims + 1})

      @testset "Fourier sample frequencies" check_sample_frequencies(grid)
    end
  end
end
