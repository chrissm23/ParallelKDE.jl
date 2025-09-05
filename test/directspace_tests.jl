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

function calculate_test_vmr(test_mean, test_var; time_final=1.0, time_initial=0.0, n_samples=2)
  n_dims = ndims(test_var) - 1

  means = @. abs(test_mean) / n_samples
  vars = @. abs(test_var) / n_samples - means^2
  vmr = vars ./ means

  times = fill(time_final, n_dims)
  times_initial = fill(time_initial, n_dims)

  scaling_factor = prod(times .^ 2 .+ times_initial .^ 2)^(3 / 2) * n_samples^3
  vmr_v = scaling_factor .* dropdims(var(vmr, dims=n_dims + 1); dims=n_dims + 1)

  return @. ifelse(isfinite(vmr_v), log(vmr_v), NaN)
end

function calculate_test_means(test_mean)
  means = @. abs(test_mean)

  return means
end

function make_snapshot(::Val{:cpu}, n_dims, stage, steps_low, steps_over)
  grid_size = ntuple(i -> 101, n_dims)
  if stage == :low_density_recon
    vmr_current = fill(1.0e6, grid_size)
    vmr_prev1 = fill(0.8, grid_size)
    vmr_prev2 = fill(1.0e6, grid_size)

    current_minima = zeros(Float64, grid_size)
    counters_low = fill(UInt16(steps_low), grid_size)
    counters_over = zeros(UInt16, grid_size)
    low_density_flags = fill(false, grid_size)

  elseif stage == :low_density_stop
    vmr_current = fill(1.0, grid_size)
    vmr_prev1 = fill(0.99, grid_size)
    vmr_prev2 = fill(1.0, grid_size)

    current_minima = fill(NaN, grid_size)
    counters_low = fill(UInt16(steps_low + 1), grid_size)
    counters_over = zeros(UInt16, grid_size)
    low_density_flags = fill(true, grid_size)

  elseif stage == :high_density_update
    vmr_current = fill(1.00005, grid_size)
    vmr_prev1 = fill(1.0, grid_size)
    vmr_prev2 = fill(1.00005, grid_size)

    current_minima = zeros(Float64, grid_size)
    counters_low = zeros(UInt16, grid_size)
    counters_over = zeros(UInt16, grid_size)
    low_density_flags = fill(false, grid_size)

  elseif stage == :high_density_noupdate
    vmr_current = fill(1.01, grid_size)
    vmr_prev1 = fill(1.0, grid_size)
    vmr_prev2 = fill(1.01, grid_size)

    current_minima = fill(-5.0, grid_size)
    counters_low = zeros(UInt16, grid_size)
    counters_over = zeros(UInt16, grid_size)
    low_density_flags = fill(false, grid_size)

  else
    throw(
      ArgumentError(
        "stage can only be :low_density_recon, :low_density_stop, :high_density_update, :high_density_noupdate"
      )
    )
  end

  return (
    vmr_current=vmr_current,
    vmr_prev1=vmr_prev1,
    vmr_prev2=vmr_prev2,
    current_minima=current_minima,
    counters_low=counters_low,
    counters_over=counters_over,
    low_density_flags=low_density_flags,
  )

end
function make_snapshot(::Val{:cuda}, n_dims, stage, steps_low, steps_over)
  grid_size = ntuple(i -> 101, n_dims)
  if stage == :low_density_recon
    vmr_current = CUDA.fill(1.0f6, grid_size)
    vmr_prev1 = CUDA.fill(0.8f0, grid_size)
    vmr_prev2 = CUDA.fill(1.0f6, grid_size)

    current_minima = CUDA.zeros(Float32, grid_size)
    counters_low = CUDA.fill(UInt16(steps_low), grid_size)
    counters_over = CUDA.zeros(UInt16, grid_size)
    low_density_flags = CUDA.fill(false, grid_size)

  elseif stage == :low_density_stop
    vmr_current = CUDA.fill(1.0f0, grid_size)
    vmr_prev1 = CUDA.fill(0.99f0, grid_size)
    vmr_prev2 = CUDA.fill(1.0f0, grid_size)

    current_minima = CUDA.fill(NaN32, grid_size)
    counters_low = CUDA.fill(UInt16(steps_low + 1), grid_size)
    counters_over = CUDA.zeros(UInt16, grid_size)
    low_density_flags = CUDA.fill(true, grid_size)

  elseif stage == :high_density_update
    vmr_current = CUDA.fill(1.00005f0, grid_size)
    vmr_prev1 = CUDA.fill(1.0f0, grid_size)
    vmr_prev2 = CUDA.fill(1.00005f0, grid_size)

    current_minima = CUDA.zeros(Float32, grid_size)
    counters_low = CUDA.zeros(UInt16, grid_size)
    counters_over = CUDA.zeros(UInt16, grid_size)
    low_density_flags = CUDA.fill(false, grid_size)

  elseif stage == :high_density_noupdate
    vmr_current = CUDA.fill(1.01f0, grid_size)
    vmr_prev1 = CUDA.fill(1.0f0, grid_size)
    vmr_prev2 = CUDA.fill(1.01f0, grid_size)

    current_minima = CUDA.fill(-5.0f0, grid_size)
    counters_low = CUDA.zeros(UInt16, grid_size)
    counters_over = CUDA.zeros(UInt16, grid_size)
    low_density_flags = CUDA.fill(false, grid_size)

  else
    throw(
      ArgumentError(
        "stage can only be :low_density_recon, :low_density_stop, :high_density_update, :high_density_noupdate"
      )
    )
  end

  return (
    vmr_current=vmr_current,
    vmr_prev1=vmr_prev1,
    vmr_prev2=vmr_prev2,
    current_minima=current_minima,
    counters_low=counters_low,
    counters_over=counters_over,
    low_density_flags=low_density_flags,
  )
end

@testset "CPU direct space operations tests" begin
  @testset "Implementation: $implementation tests" for implementation in [:serial]
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

      time_final = 1.0
      time_initial = 0.0
      n_samples = 2

      test_vmr = calculate_test_vmr(test_mean, test_var; time_final, time_initial, n_samples)

      grid_size = size(test_var)[begin:end-1]
      grid_length = prod(grid_size)

      ParallelKDE.calculate_scaled_vmr!(
        Val(implementation), test_mean, test_var, fill(time_final, n_dims), fill(time_initial, n_dims), 2
      )
      calculated_vmr = vec(reinterpret(Float64, test_var))[begin:grid_length]
      calculated_vmr = reshape(calculated_vmr, grid_size)

      @test test_vmr ≈ calculated_vmr
    end

    @testset "All samples means calculation tests. $(n_dims)D" for n_dims in 1:3
      test_mean = create_test_array(n_dims)
      n_samples = 2
      test_result = calculate_test_means(test_mean)

      array_size = size(test_mean)
      array_length = length(test_mean)

      ParallelKDE.calculate_full_means!(Val(implementation), test_mean, n_samples)
      calculated_mean = vec(reinterpret(Float64, test_mean))[begin:array_length]
      calculated_mean = reshape(calculated_mean, array_size)

      @test test_result ≈ calculated_mean
    end

    @testset "Indentify convergence tests. $(n_dims)D" for n_dims in 1:3
      dlogt = 1.0e-1
      tol_low_id = 2.0
      steps_low = 3
      steps_over = 30

      # Test low density recognition
      snap = make_snapshot(Val(:cpu), n_dims, :low_density_recon, steps_low, steps_over)

      density = fill(NaN, size(snap.vmr_current))
      means = ones(Float64, size(snap.vmr_current))

      ParallelKDE.DirectSpace.identify_convergence!(
        Val(implementation),
        density,
        means,
        snap.vmr_current,
        snap.vmr_prev1,
        snap.vmr_prev2,
        dlogt,
        tol_low_id,
        steps_low,
        steps_over,
        snap.current_minima,
        snap.counters_low,
        snap.counters_over,
        snap.low_density_flags,
      )

      @test all(isnan.(density))
      @test all(isnan.(snap.current_minima))
      @test all(snap.counters_low .== steps_low + 1)
      @test all(iszero.(snap.counters_over))
      @test all(snap.low_density_flags)

      # Test low density stop
      snap = make_snapshot(Val(:cpu), n_dims, :low_density_stop, steps_low, steps_over)

      density = fill(NaN, size(snap[1]))
      means = ones(Float64, size(snap[1]))

      ParallelKDE.DirectSpace.identify_convergence!(
        Val(implementation),
        density,
        means,
        snap.vmr_current,
        snap.vmr_prev1,
        snap.vmr_prev2,
        dlogt,
        tol_low_id,
        steps_low,
        steps_over,
        snap.current_minima,
        snap.counters_low,
        snap.counters_over,
        snap.low_density_flags,
      )

      @test all(isone.(density))
      @test !any(iszero.(snap.current_minima))
      @test all(snap.counters_low .== steps_low + 1)
      @test all(iszero.(snap.counters_over))
      @test all(snap.low_density_flags)

      # Test high density update
      snap = make_snapshot(Val(:cpu), n_dims, :high_density_update, steps_low, steps_over)

      density = fill(NaN, size(snap[1]))
      means = ones(Float64, size(snap[1]))

      ParallelKDE.DirectSpace.identify_convergence!(
        Val(implementation),
        density,
        means,
        snap.vmr_current,
        snap.vmr_prev1,
        snap.vmr_prev2,
        dlogt,
        tol_low_id,
        steps_low,
        steps_over,
        snap.current_minima,
        snap.counters_low,
        snap.counters_over,
        snap.low_density_flags,
      )

      @test all(isone.(density))
      @test all(snap.current_minima .<= 0)
      @test all(iszero.(snap.counters_low))
      @test all(iszero.(snap.counters_over))
      @test !any(snap.low_density_flags)

      # Test high density no update
      snap = make_snapshot(Val(:cpu), n_dims, :high_density_noupdate, steps_low, steps_over)

      density = fill(NaN, size(snap[1]))
      means = ones(Float64, size(snap[1]))

      ParallelKDE.DirectSpace.identify_convergence!(
        Val(implementation),
        density,
        means,
        snap.vmr_current,
        snap.vmr_prev1,
        snap.vmr_prev2,
        dlogt,
        tol_low_id,
        steps_low,
        steps_over,
        snap.current_minima,
        snap.counters_low,
        snap.counters_over,
        snap.low_density_flags,
      )

      @test all(isnan.(density))
      @test all(snap.current_minima .== -5.0)
      @test all(iszero.(snap.counters_low))
      @test all(isone.(snap.counters_over))
      @test !any(snap.low_density_flags)
    end
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

      time_final = 1.0
      time_initial = 0.0
      n_samples = 2

      test_vmr = calculate_test_vmr(test_mean, test_var; time_final, time_initial, n_samples)

      grid_size = size(test_var)[begin:end-1]
      grid_length = prod(grid_size)

      test_mean_d = CuArray{ComplexF32}(test_mean)
      test_var_d = CuArray{ComplexF32}(test_var)

      ParallelKDE.calculate_scaled_vmr!(
        Val(:cuda), test_mean_d, test_var_d, CUDA.fill(time_final, n_dims), CUDA.fill(time_initial, n_dims), 2
      )
      calculated_vmr_d = vec(reinterpret(Float32, test_var_d))[begin:2:2*grid_length]
      calculated_vmr_d = reshape(calculated_vmr_d, grid_size)
      calculated_vmr = Array(calculated_vmr_d)

      @test test_vmr ≈ calculated_vmr
    end

    @testset "All samples means calculation tests. $(n_dims)D" for n_dims in 1:3
      test_mean = create_test_array(n_dims)
      n_samples = 2
      test_result = calculate_test_means(test_mean)

      array_size = size(test_mean)
      array_length = length(test_mean)

      test_mean_d = CuArray{ComplexF32}(test_mean)
      ParallelKDE.calculate_full_means!(Val(:cuda), test_mean_d, n_samples)
      calculated_mean_d = vec(reinterpret(Float32, test_mean_d))[begin:2:2*array_length]
      calculated_mean_d = reshape(calculated_mean_d, array_size)

      @test test_result ≈ Array(calculated_mean_d)
    end

    @testset "Indentify convergence tests. $(n_dims)D" for n_dims in 1:3
      dlogt = 1.0f-1
      tol_low_id = 2.0f0
      steps_low = Int32(3)
      steps_over = Int32(30)

      # Test low density recognition
      snap = make_snapshot(Val(:cuda), n_dims, :low_density_recon, steps_low, steps_over)

      density = CUDA.fill(NaN32, size(snap.vmr_current))
      means = CUDA.ones(Float32, size(snap.vmr_current))

      ParallelKDE.DirectSpace.identify_convergence!(
        Val(:cuda),
        density,
        means,
        snap.vmr_current,
        snap.vmr_prev1,
        snap.vmr_prev2,
        dlogt,
        tol_low_id,
        steps_low,
        steps_over,
        snap.current_minima,
        snap.counters_low,
        snap.counters_over,
        snap.low_density_flags,
      )

      @test all(isnan.(density))
      @test all(isnan.(snap.current_minima))
      @test all(snap.counters_low .== steps_low + 1)
      @test all(iszero.(snap.counters_over))
      @test all(snap.low_density_flags)

      # Test low density stop
      snap = make_snapshot(Val(:cuda), n_dims, :low_density_stop, steps_low, steps_over)

      density = CUDA.fill(NaN32, size(snap.vmr_current))
      means = CUDA.ones(Float32, size(snap.vmr_current))

      ParallelKDE.DirectSpace.identify_convergence!(
        Val(:cuda),
        density,
        means,
        snap.vmr_current,
        snap.vmr_prev1,
        snap.vmr_prev2,
        dlogt,
        tol_low_id,
        steps_low,
        steps_over,
        snap.current_minima,
        snap.counters_low,
        snap.counters_over,
        snap.low_density_flags,
      )

      @test all(isone.(density))
      @test !any(iszero.(snap.current_minima))
      @test all(snap.counters_low .== steps_low + 1)
      @test all(iszero.(snap.counters_over))
      @test all(snap.low_density_flags)

      # Test high density update
      snap = make_snapshot(Val(:cuda), n_dims, :high_density_update, steps_low, steps_over)

      density = CUDA.fill(NaN32, size(snap.vmr_current))
      means = CUDA.ones(Float32, size(snap.vmr_current))

      ParallelKDE.DirectSpace.identify_convergence!(
        Val(:cuda),
        density,
        means,
        snap.vmr_current,
        snap.vmr_prev1,
        snap.vmr_prev2,
        dlogt,
        tol_low_id,
        steps_low,
        steps_over,
        snap.current_minima,
        snap.counters_low,
        snap.counters_over,
        snap.low_density_flags,
      )

      @test all(isone.(density))
      @test all(snap.current_minima .<= 0)
      @test all(iszero.(snap.counters_low))
      @test all(iszero.(snap.counters_over))
      @test !any(snap.low_density_flags)

      # Test high density no update
      snap = make_snapshot(Val(:cuda), n_dims, :high_density_noupdate, steps_low, steps_over)

      density = CUDA.fill(NaN32, size(snap.vmr_current))
      means = CUDA.ones(Float32, size(snap.vmr_current))

      ParallelKDE.DirectSpace.identify_convergence!(
        Val(:cuda),
        density,
        means,
        snap.vmr_current,
        snap.vmr_prev1,
        snap.vmr_prev2,
        dlogt,
        tol_low_id,
        steps_low,
        steps_over,
        snap.current_minima,
        snap.counters_low,
        snap.counters_over,
        snap.low_density_flags,
      )

      @test all(isnan.(density))
      @test all(snap.current_minima .== -5.0)
      @test all(iszero.(snap.counters_low))
      @test all(isone.(snap.counters_over))
      @test !any(snap.low_density_flags)
    end
  end
end
