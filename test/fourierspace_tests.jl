function normal_distribution(
  x::SVector{N,S},
  μ::SVector{N,S},
  bandwidth::SMatrix{N,N,<:Real}
) where {N,S<:Real}
  normal_distro = MvNormal(μ, bandwidth .^ 2)

  return pdf(normal_distro, x)
end

function initialize_kernels(data::Vector{SVector{N,S}}, grid::Grid{N,S}) where {N,S<:Real}
  grid_coordinates = SVector{N,S}.(eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, N)))
  bandwidth = SMatrix{N,N,Float64}(
    diagm(initial_bandwidth(grid))
  )

  density = normal_distribution.(
    data,
    reshape(grid_coordinates, 1, size(grid_coordinates)...),
    Ref(bandwidth)
  )

  return density
end

function initialize_density(
  data::Vector{SVector{N,S}},
  grid::Grid{N,S}
) where {N,S<:Real}
  n_samples = length(data)

  density = initialize_kernels(data, grid) ./ n_samples
  density_squared = density .^ 2
  density_sum = sum(density, dims=1)
  density_sum_squared = sum(density_squared, dims=1)

  return density_sum, density_sum_squared
end

function calculate_means_variances(
  data::Vector{SVector{N,S}},
  grid::Grid{N,S}
) where {N,S<:Real}
  n_samples = length(data)

  density = initialize_kernels(data, grid) ./ n_samples
  means = mean(density, dims=1)
  variances = var(density, dims=1)

  return means, variances
end

@testset "Fourier initialization (CPU) tests" for n_dims in 1:3
  n_samples = 100
  @testset "dimensions : $n_dims" begin
    data = generate_samples(n_samples, n_dims)
    grid_ranges = fill(-5.0:0.05:5.0, n_dims)
    grid = Grid(grid_ranges)

    density_initialization, density_initialization_squared = initialize_density(data, grid)

    sk0, s2k0 = initialize_fourier_statistics(density_initialization, density_initialization_squared)
    means, variances = ifft_statistics(sk0, s2k0, n_samples)
    means = dropdims(means, dims=1)
    variances = dropdims(variances, dims=1)

    means_test, variances_test = calculate_means_variances(data, grid)
    means_test = dropdims(means_test, dims=1)
    variances_test = dropdims(variances_test, dims=1)

    @test means ≈ means_test atol = 1e-8 rtol = 1e-1
    @test variances ≈ variances_test atol = 1e-8 rtol = 1e-1
  end
end

if CUDA.functional()
  @testset "Fourier initialization (GPU) tests" for n_dims in 1:3
    n_samples = 100
    @testset "dimensions : $n_dims" begin
      data = generate_samples(n_samples, n_dims)
      grid_ranges = fill(-5.0:0.05:5.0, n_dims)
      grid = Grid(grid_ranges)

      density_initialization, density_initialization_squared = initialize_density(data, grid)
      density_initialization_d = CuArray{Float32}(density_initialization)
      density_initialization_squared_d = CuArray{Float32}(density_initialization_squared)

      sk0_d, s2k0_d = initialize_fourier_statistics(density_initialization_d, density_initialization_squared_d)
      means_d, variances_d = ifft_statistics(sk0_d, s2k0_d, n_samples)
      means = dropdims(Array{Float32}(means_d), dims=1)
      variances = dropdims(Array{Float32}(variances_d), dims=1)

      means_test, variances_test = calculate_means_variances(data, grid)
      means_test = dropdims(Array{Float32}(means_test), dims=1)
      variances_test = dropdims(Array{Float32}(variances_test), dims=1)

      @test means ≈ means_test atol = 1.0f-8 rtol = 1.0f-1
      @test variances ≈ variances_test atol = 1.0f-8 rtol = 1.0f-1
    end
  end
end
