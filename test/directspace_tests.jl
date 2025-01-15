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

    products .= map(prod, Iterators.product(zip(remainder_l, remainder_h)...)
    )
    grid_points .= CartesianIndex{n_dims}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

    @inbounds result_array[grid_points] .+= products
    @inbounds result_array_squared[grid_points] .+= products .^ 2
  end

  result_array ./= n_samples * prod(0.07 * ones(n_dims))^2
  result_array_squared ./= (n_samples * prod(0.07 * ones(n_dims))^2)^2

  return result_array, result_array_squared
end

function test_diracseries_cpu(n_dims::Integer, n_samples::Int)
  if n_samples == 1
    data = [@SVector fill(0.0, n_dims)]
  elseif n_samples == 2
    data = [SVector{n_dims,Float64}(fill(0.0, n_dims)), SVector{n_dims,Float64}(fill(0.05, n_dims))]
  else
    throw(ArgumentError("n_samples must be 1 or 2."))
  end
  grid_ranges = fill(-3.0:0.07:3.0, n_dims)

  spacing = @SVector fill(0.07, n_dims)
  low_bound = @SVector fill(-3.0, n_dims)
  dirac_series = zeros(Float64, length.(grid_ranges)...)
  dirac_series_squared = zeros(Float64, length.(grid_ranges)...)

  DirectSpace.generate_dirac_cpu!(dirac_series, dirac_series_squared, data, spacing, low_bound)

  result, result_squared = calculate_test_result(n_dims, n_samples)

  @test dirac_series ≈ result atol = 1.0e-8 rtol = 1.0e-5
  @test dirac_series_squared ≈ result_squared atol = 1.0e-8 rtol = 1.0e-5

  return nothing
end

function test_diracseries_gpu(n_dims::Integer, n_samples::Int)
  if n_samples == 1
    data = CUDA.zeros(Float32, n_dims, 1)
  elseif n_samples == 2
    data = CuArray{Float32,2}([fill(0.0, n_dims) fill(0.05, n_dims)])
  else
    throw(ArgumentError("n_samples must be 1 or 2."))
  end
  grid_ranges = fill(-3.0:0.07:3.0, n_dims)

  spacing = CUDA.fill(0.07f0, n_dims)
  low_bound = CUDA.fill(-3.0f0, n_dims)
  bootstrap_idxs = CuArray{Int32,2}([1 1; 2 2])
  dirac_series = CUDA.zeros(Float32, 2, length.(grid_ranges)...)
  dirac_series_squared = CUDA.zeros(Float32, 2, length.(grid_ranges)...)

  kernel = @cuda launch = false DirectSpace.generate_dirac_gpu!(
    dirac_series, dirac_series_squared, data, bootstrap_idxs, spacing, low_bound
  )
  config = launch_configuration(kernel.fun)

  n_modified_gridpoints = 2^n_dims * n_samples * 2
  threads = min(n_modified_gridpoints, config.threads)
  blocks = cld(n_modified_gridpoints, threads)

  CUDA.@sync blocking = true begin
    kernel(dirac_series, dirac_series_squared, data, bootstrap_idxs, spacing, low_bound; threads, blocks)
  end

  result, result_squared = calculate_test_result(n_dims, n_samples)
  result = permutedims(cat(result, result, dims=n_dims + 1), (n_dims + 1, ntuple(i -> i, n_dims)...))
  result_squared = permutedims(
    cat(result_squared, result_squared, dims=n_dims + 1),
    (n_dims + 1, ntuple(i -> i, n_dims)...)
  )
  result = CuArray{Float32,n_dims + 1}(result)
  result_squared = CuArray{Float32,n_dims + 1}(result_squared)

  @test dirac_series ≈ result atol = 1.0f-8 rtol = 1.0f-5
  @test dirac_series_squared ≈ result_squared atol = 1.0f-8 rtol = 1.0f-5

  return nothing
end

@testset "CPU direct space operations tests" begin
  @testset "Dirac sequences tests" for n_dims in 1:1
    @testset "dimensions : $n_dims" begin
      @testset "Single data point" test_diracseries_cpu(n_dims, 1)
      @testset "Multiple data points" test_diracseries_cpu(n_dims, 2)
    end
  end
end

if CUDA.functional()
  @testset "GPU direct space operations tests" begin
    @testset "Dirac sequences tests" for n_dims in 1:1
      @testset "dimensions : $n_dims" begin
        @testset "Single data point" test_diracseries_gpu(n_dims, 1)
        @testset "Multiple data points" test_diracseries_gpu(n_dims, 2)
      end
    end
  end
end

