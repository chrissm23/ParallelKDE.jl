function initialize_kernels(
  data::Vector{SVector{N,S}},
  grid::Grid{N,S};
  bandwidth::Union{Nothing,SMatrix{N,N,Float64}}=nothing
) where {N,S<:Real}
  grid_coordinates = SVector{N,S}.(eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, N)))
  bandwidth_init = SMatrix{N,N,Float64}(
    diagm(initial_bandwidth(grid))
  )

  if bandwidth === nothing
    bandwidth_final = bandwidth_init
  else
    bandwidth_final = @. sqrt((bandwidth^2) + (bandwidth_init^2))
  end

  density = normal_distribution.(
    data,
    reshape(grid_coordinates, 1, size(grid_coordinates)...),
    Ref(bandwidth_final)
  )

  return density
end

function initialize_density(
  data::Vector{SVector{N,S}},
  grid::Grid{N,S};
  bandwidth::Union{Nothing,SMatrix{N,N,Float64}}=nothing
) where {N,S<:Real}
  n_samples = length(data)

  density = initialize_kernels(data, grid; bandwidth) ./ n_samples
  density_squared = density .^ 2
  density_sum = sum(density, dims=1)
  density_sum_squared = sum(density_squared, dims=1)

  return density_sum, density_sum_squared
end

function calculate_means_variances(
  data::Vector{SVector{N,S}},
  grid::Grid{N,S};
  bandwidth::Union{Nothing,SMatrix{N,N,Float64}}=nothing
) where {N,S<:Real}
  n_samples = length(data)

  density = initialize_kernels(data, grid; bandwidth) ./ n_samples
  means = mean(density, dims=1)
  variances = var(density, dims=1)

  return means, variances
end

@testset "Fourier initialization (CPU) tests" for n_dims in 1:1
  n_samples = 100
  @testset "dimensions : $n_dims" begin
    data = generate_samples(n_samples, n_dims)
    grid_ranges = fill(-5.0:0.05:5.0, n_dims)
    grid = Grid(grid_ranges)

    density_initialization, density_initialization_squared = initialize_density(data, grid)

    tmp = Array{ComplexF64}(undef, 1, size(grid)...)
    sk0, s2k0 = initialize_fourier_statistics(density_initialization, density_initialization_squared, tmp)
    ifft_statistics!(sk0, s2k0, n_samples, tmp=tmp)
    means = dropdims(sk0, dims=1)
    variances = dropdims(s2k0, dims=1)

    means_test, variances_test = calculate_means_variances(data, grid)
    means_test = dropdims(means_test, dims=1)
    variances_test = dropdims(variances_test, dims=1)

    @test means ≈ means_test atol = 1e-8 rtol = 1e-1
    @test variances ≈ variances_test atol = 1e-8 rtol = 1e-1
  end
end

if CUDA.functional()
  @testset "Fourier initialization (GPU) tests" for n_dims in 1:1
    n_samples = 100
    @testset "dimensions : $n_dims" begin
      data = generate_samples(n_samples, n_dims)
      grid_ranges = fill(-5.0:0.05:5.0, n_dims)
      grid = Grid(grid_ranges)

      density_initialization, density_initialization_squared = initialize_density(data, grid)
      density_initialization_d = CuArray{Float32}(density_initialization)
      density_initialization_squared_d = CuArray{Float32}(density_initialization_squared)

      tmp = CuArray{ComplexF32}(undef, 1, size(grid)...)
      sk0_d, s2k0_d = initialize_fourier_statistics(
        density_initialization_d, density_initialization_squared_d, tmp
      )
      ifft_statistics!(sk0_d, s2k0_d, n_samples, tmp=tmp)
      means = dropdims(Array{Float32}(sk0_d), dims=1)
      variances = dropdims(Array{Float32}(s2k0_d), dims=1)

      means_test, variances_test = calculate_means_variances(data, grid)
      means_test = dropdims(Array{Float32}(means_test), dims=1)
      variances_test = dropdims(Array{Float32}(variances_test), dims=1)

      @test means ≈ means_test atol = 1.0f-8 rtol = 1.0f-1
      @test variances ≈ variances_test atol = 1.0f-8 rtol = 1.0f-1
    end
  end
end

@testset "Fourier propagation (CPU) tests" for n_dims in 1:1
  n_samples = 100
  @testset "dimensions : $n_dims" begin
    data = generate_samples(n_samples, n_dims)
    time = @SVector fill(0.1, n_dims)

    grid_ranges = fill(-5.0:0.01:5.0, n_dims)
    grid = Grid(grid_ranges)
    time_initial = initial_bandwidth(grid)
    fourier_grid = fftgrid(grid)

    density_initialization, density_initialization_squared = initialize_density(data, grid)

    tmp = Array{ComplexF64}(undef, 1, size(grid)...)
    sk0, s2k0 = initialize_fourier_statistics(density_initialization, density_initialization_squared, tmp)
    means_t = similar(sk0)
    variances_t = similar(s2k0)

    propagate_statistics!(
      Val(:serial),
      means_t,
      variances_t,
      sk0,
      s2k0,
      get_coordinates(fourier_grid),
      time,
      time_initial
    )
    ifft_statistics!(means_t, variances_t, n_samples, tmp=tmp)
    means = dropdims(means_t, dims=1)
    variances = dropdims(variances_t, dims=1)

    time_bandwidth = SMatrix{n_dims,n_dims,Float64}(
      diagm(time)
    )
    means_test, variances_test = calculate_means_variances(data, grid, bandwidth=time_bandwidth)
    means_test = dropdims(means_test, dims=1)
    variances_test = dropdims(variances_test, dims=1)

    @test means ≈ means_test atol = 1e-10 rtol = 1e-1
    @test variances ≈ variances_test atol = 1e-10 rtol = 5e-1
  end
end

if CUDA.functional()
  @testset "Fourier propagation (GPU) tests" for n_dims in 1:1
    n_samples = 100
    @testset "dimensions : $n_dims" begin
      data = generate_samples(n_samples, n_dims)
      time = CUDA.fill(0.1, n_dims)

      grid_ranges = fill(-5.0:0.01:5.0, n_dims)
      grid = Grid(grid_ranges)
      time_initial = CuArray{Float32}(initial_bandwidth(grid))
      fourier_grid = fftgrid(grid)

      density_initialization, density_initialization_squared = initialize_density(data, grid)
      density_initialization_d = CuArray{Float32}(density_initialization)
      density_initialization_squared_d = CuArray{Float32}(density_initialization_squared)

      tmp = CuArray{ComplexF32}(undef, 1, size(grid)...)
      sk0_d, s2k0_d = initialize_fourier_statistics(
        density_initialization_d, density_initialization_squared_d, tmp
      )
      means_t_d = similar(sk0_d)
      variances_t_d = similar(s2k0_d)

      propagate_statistics!(
        Val(:cuda),
        means_t_d,
        variances_t_d,
        sk0_d,
        s2k0_d,
        CuArray{Float32,n_dims + 1}(get_coordinates(fourier_grid)),
        time,
        time_initial
      )
      ifft_statistics!(means_t_d, variances_t_d, n_samples, tmp=tmp)
      means = dropdims(Array{Float32}(means_t_d), dims=1)
      variances = dropdims(Array{Float32}(variances_t_d), dims=1)

      time_bandwidth = SMatrix{n_dims,n_dims,Float64}(
        diagm(Array{Float32}(time))
      )
      means_test, variances_test = calculate_means_variances(data, grid, bandwidth=time_bandwidth)
      means_test = dropdims(Array{Float32}(means_test), dims=1)
      variances_test = dropdims(Array{Float32}(variances_test), dims=1)

      @test means ≈ means_test atol = 1.0f-10 rtol = 1.0f-1
      @test variances ≈ variances_test atol = 1.0f-10 rtol = 5.0f-1
    end
  end
end
