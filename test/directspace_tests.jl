function calculate_test_sequences(n_dims)
  grid_ranges = fill(-5.0:0.1:5.0, n_dims)
  spacing_squared = prod(step.(grid_ranges))^2

  density = zeros(ComplexF64, length.(grid_ranges)..., 2)
  density_squared = zeros(ComplexF64, size(density))

  # Sample 1
  indices_l = fill(51, n_dims)
  indices_h = fill(52, n_dims)
  remainder_l = fill(0.05, n_dims)
  remainder_h = fill(0.05, n_dims)

  products = map(prod, Iterators.product(zip(remainder_l, remainder_h)...))

  grid_points = CartesianIndex{n_dims}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

  sequence_terms = products / (2 * spacing_squared)
  selectdim(density, n_dims + 1, 1)[grid_points] .+= sequence_terms
  selectdim(density, n_dims + 1, 2)[grid_points] .+= sequence_terms
  selectdim(density_squared, n_dims + 1, 1)[grid_points] .+= sequence_terms .^ 2
  selectdim(density_squared, n_dims + 1, 2)[grid_points] .+= sequence_terms .^ 2

  # Sample 2
  indices_l = fill(52, n_dims)
  indices_h = fill(53, n_dims)
  remainder_l = fill(0.05, n_dims)
  remainder_h = fill(0.05, n_dims)

  products = map(prod, Iterators.product(zip(remainder_l, remainder_h)...))

  grid_points = CartesianIndex{n_dims}.(collect(Iterators.product(zip(indices_l, indices_h)...)))

  sequence_terms = products / (2 * spacing_squared)
  selectdim(density, n_dims + 1, 1)[grid_points] .+= sequence_terms
  selectdim(density, n_dims + 1, 2)[grid_points] .+= sequence_terms
  selectdim(density_squared, n_dims + 1, 1)[grid_points] .+= sequence_terms .^ 2
  selectdim(density_squared, n_dims + 1, 2)[grid_points] .+= sequence_terms .^ 2

  return density, density_squared
end

function create_test_array(n_dims)
  return rand(ComplexF64, ntuple(i -> 50, n_dims)..., 5)
end

function calculate_test_vmr(test_mean, test_var; time=1.0, time_initial=0.0, n_samples=2)
  n_dims = ndims(test_var) - 1

  means = @. abs(test_mean) / n_samples
  vars = @. abs(test_var) / n_samples - means^2
  vmr = vars ./ means

  times = fill(time, n_dims)
  times_initial = fill(time_initial, n_dims)

  scaling_factor = prod(times .^ 2 .+ times_initial .^ 2)^(3 / 2) * n_samples^4
  vmr_v = scaling_factor .* dropdims(var(vmr, dims=n_dims + 1); dims=n_dims + 1)

  return @. ifelse(isfinite(vmr_v), vmr_v, NaN)
end

function calculate_test_means(test_mean; n_samples=2)
  means = @. abs(test_mean) / n_samples

  return means
end

@testset "CPU direct space operations tests" begin
  @testset "Implementation: $implementation tests" for implementation in [:serial, :threaded]
    @testset "Dirac sequences tests. $(n_dims)D" for n_dims in 1:3
      data_matrix = [fill(0.05, n_dims) fill(0.15, n_dims)]
      data = Vector{SVector{n_dims,Float64}}(eachcol(data_matrix))
      bootstrap_idxs = [1 2; 2 1]

      grid_ranges = fill(-5.0:0.1:5.0, n_dims)
      grid = initialize_grid(grid_ranges)

      dirac_sequences, dirac_squared = initialize_dirac_sequence(
        Val(implementation), data, grid, bootstrap_idxs, include_var=true
      )
      resulting_sequences, resulting_squared = calculate_test_sequences(n_dims)

      @test dirac_sequences ≈ resulting_sequences
      @test dirac_squared ≈ resulting_squared
    end

    @testset "VMR calculation tests. $(n_dims)D" for n_dims in 1:3
      test_var = create_test_array(n_dims)
      test_mean = create_test_array(n_dims)

      time = 1.0
      time_initial = 0.0
      n_samples = 2

      test_vmr = calculate_test_vmr(test_mean, test_var; time, time_initial, n_samples)

      grid_size = size(test_var)[begin:end-1]
      grid_length = prod(grid_size)

      ParallelKDE.calculate_scaled_vmr!(
        Val(implementation), test_mean, test_var, fill(time, n_dims), fill(time_initial, n_dims), 2
      )
      calculated_vmr = vec(reinterpret(Float64, test_var))[begin:grid_length]
      calculated_vmr = reshape(calculated_vmr, grid_size)

      @test test_vmr ≈ calculated_vmr
    end

    @testset "All samples means calculation tests. $(n_dims)D" for n_dims in 1:3
      test_mean = create_test_array(n_dims)
      n_samples = 2
      test_result = calculate_test_means(test_mean; n_samples)

      array_size = size(test_mean)
      array_length = length(test_mean)

      ParallelKDE.calculate_full_means!(Val(implementation), test_mean, n_samples)
      calculated_mean = vec(reinterpret(Float64, test_mean))[begin:array_length]
      calculated_mean = reshape(calculated_mean, array_size)

      @test test_result ≈ calculated_mean
    end

    # TODO: Design test to check that the detection of the stopping point works correctly.
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

      dirac_sequences, dirac_squared = initialize_dirac_sequence(
        Val(:cuda), CuArray{Float32}(data_matrix), grid, CuArray{Int32}(bootstrap_idxs), include_var=true
      )
      resulting_sequences, resulting_squared = calculate_test_sequences(n_dims)

      @test Array(dirac_sequences) ≈ resulting_sequences
      @test Array(dirac_squared) ≈ resulting_squared
    end

    @testset "VMR calculation tests. $(n_dims)D" for n_dims in 1:3
      test_var = create_test_array(n_dims)
      test_mean = create_test_array(n_dims)

      time = 1.0
      time_initial = 0.0
      n_samples = 2

      test_vmr = calculate_test_vmr(test_mean, test_var; time, time_initial, n_samples)

      grid_size = size(test_var)[begin:end-1]
      grid_length = prod(grid_size)

      test_mean_d = CuArray{ComplexF32}(test_mean)
      test_var_d = CuArray{ComplexF32}(test_var)

      ParallelKDE.calculate_scaled_vmr!(
        Val(:cuda), test_mean_d, test_var_d, CUDA.fill(time, n_dims), CUDA.fill(time_initial, n_dims), 2
      )
      calculated_vmr_d = vec(reinterpret(Float32, test_var_d))[begin:2:2*grid_length]
      calculated_vmr_d = reshape(calculated_vmr_d, grid_size)
      calculated_vmr = Array(calculated_vmr_d)

      @test test_vmr ≈ calculated_vmr
    end

    @testset "All samples means calculation tests. $(n_dims)D" for n_dims in 1:3
      test_mean = create_test_array(n_dims)
      n_samples = 2
      test_result = calculate_test_means(test_mean; n_samples)

      array_size = size(test_mean)
      array_length = length(test_mean)

      test_mean_d = CuArray{ComplexF32}(test_mean)
      ParallelKDE.calculate_full_means!(Val(:cuda), test_mean_d, n_samples)
      calculated_mean_d = vec(reinterpret(Float32, test_mean_d))[begin:2:2*array_length]
      calculated_mean_d = reshape(calculated_mean_d, array_size)

      @test test_result ≈ Array(calculated_mean_d)
    end

    # TODO: Design test to check that the detection of the stopping point works correctly.
    # @testset "Indentify convergence tests" begin end
  end
end
