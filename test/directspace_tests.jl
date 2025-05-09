function calculate_test_sequences(n_dims)
  grid_ranges = fill(-5.0:0.1:5.0, n_dims)
  density = zeros(ComplexF64, length.(grid_ranges)..., 2)
  density_squared = zeros(ComplexF64, size(density))

  # Sample 1
  indices_l = fill(51, n_dims)
  indices_h = fill(52, n_dims)
  remainder_l = fill(0.05, n_dims)
  remainder_h = fill(0.05, n_dims)

  products = map(prod, Iterators.product(zip(remainder_l, remainder_h)...))

  grid_points = CartesianIndex{n_dims}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

  selectdim(density, n_dims + 1, 1)[grid_points] .+= products
  selectdim(density, n_dims + 1, 2)[grid_points] .+= products

  # Sample 1
  indices_l = fill(52, n_dims)
  indices_h = fill(53, n_dims)
  remainder_l = fill(0.05, n_dims)
  remainder_h = fill(0.05, n_dims)

  products = map(prod, Iterators.product(zip(remainder_l, remainder_h)...))

  grid_points = CartesianIndex{n_dims}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

  selectdim(density, n_dims + 1, 1)[grid_points] .+= products
  selectdim(density, n_dims + 1, 2)[grid_points] .+= products

  @. density_squared = density^2

  return density, density_squared
end

function calculate_test_result(n_dims::Integer, n_samples::Int)
  indices_l = @MVector fill(43, n_dims)
  indices_h = @MVector fill(44, n_dims)
  remainder_l = @MVector fill(0.06, n_dims)
  remainder_h = @MVector fill(0.01, n_dims)

  products = map(prod, Iterators.product(zip(remainder_l, remainder_h)...))

  grid_points = CartesianIndex{n_dims}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

  result_array = zeros(Float64, ntuple(i -> 86, n_dims))
  result_array_squared = zeros(Float64, ntuple(i -> 86, n_dims))

  @inbounds result_array[grid_points] .+= products
  @inbounds result_array_squared[grid_points] .+= products .^ 2

  if n_samples == 2
    indices_l .= fill(44, n_dims)
    indices_h .= fill(45, n_dims)
    remainder_l .= fill(0.04, n_dims)
    remainder_h .= fill(0.03, n_dims)

    products .= map(prod, Iterators.product(zip(remainder_l, remainder_h)...))
    grid_points .= CartesianIndex{n_dims}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

    @inbounds result_array[grid_points] .+= products
    @inbounds result_array_squared[grid_points] .+= products .^ 2
  end

  result_array ./= n_samples * prod(0.07 * ones(n_dims))^2
  result_array_squared ./= (n_samples * prod(0.07 * ones(n_dims))^2)^2

  return result_array, result_array_squared
end

@testset "CPU direct space operations tests" begin
  @testset "Implementation: $implementation tests" for implementation in [:serial, :threaded]
    @testset "Dirac sequences tests. $(n_dims)D" for n_dims in 1:3
      data_matrix = [fill(0.05, n_dims) fill(0.15, n_dims)]
      data = Vector{SVector{n_dims,Float64}}(eachcol(data_matrix))
      bootstrap_idxs = [1 2; 2 1]

      grid_ranges = fill(-5.0:0.1:5.0, n_dims)
      grid = initialize_grid(grid_ranges)

      dirac_sequences, dirac_squared = ParallelKDE.initialize_dirac_sequence(
        Val(:serial), data, grid, bootstrap_idxs, include_var=true
      )
      resulting_sequences, resulting_squared = calculate_test_sequences(n_dims)

      @test dirac_sequences ≈ resulting_sequences
      @test dirac_squared ≈ resulting_squared
    end
    # @testset "VMR calculation tests" begin end
    # @testset "All samples means calculation tests" begin end
    # @testset "Indentify convergence tests" begin end
  end
end

if CUDA.functional()
  @testset "GPU direct space operations tests" begin
    @testset "Dirac sequences tests. $(n_dims)D" for n_dims in 1:3
      data_matrix = [fill(0.05, n_dims) fill(0.15, n_dims)]
      data = Vector{SVector{n_dims,Float64}}(eachcol(data_matrix))
      bootstrap_idxs = [1 2; 2 1]

      grid_ranges = fill(-5.0:0.1:5.0, n_dims)
      grid = initialize_grid(grid_ranges, device=:cuda)

      dirac_sequences, dirac_squared = ParallelKDE.initialize_dirac_sequence(
        Val(:cuda), CuArray{Float32}(data_matrix), grid, CuArray{Int32}(bootstrap_idxs), include_var=true
      )
      resulting_sequences, resulting_squared = calculate_test_sequences(n_dims)

      @test Array(dirac_sequences) ≈ resulting_sequences
      @test Array(dirac_squared) ≈ resulting_squared
    end
    # @testset "VMR calculation tests" begin end
    # @testset "All samples means calculation tests" begin end
    # @testset "Indentify convergence tests" begin end
  end
end
