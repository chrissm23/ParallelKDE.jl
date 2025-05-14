function initialize_density(
  data::Vector{SVector{N,S}},
  grid_array::AbstractArray{S,M},
  bandwidth::AbstractMatrix{P},
) where {N,S<:Real,M,P<:Real}
  n_samples = length(data)

  dens = zeros(Float64, size(grid_array)[begin+1:end])
  dens2 = zeros(Float64, size(grid_array)[begin+1:end])
  kernel = Array{Float64}(undef, size(grid_array)[begin+1:end])

  for sample in data
    kernel .= normal_distribution.(
      eachslice(grid_array, dims=Tuple(2:M)),
      Ref(sample),
      Ref(bandwidth)
    )
    kernel ./= n_samples

    @. dens += kernel
    @. dens2 += kernel^2
  end

  return dens, dens2
end

function calculate_stats(
  density::Array{T,N},
  density_squared::Array{T,N},
  n_samples::Z
) where {N,T<:Real,Z<:Integer}
  means = @. density / n_samples
  vars = @. density_squared / n_samples - means^2

  return means, vars
end

@testset "CPU Fourier space operations tests. $(n_dims)D" for n_dims in 1:3
  @testset "Implementation: $implementation tests" for implementation in [:serial]
    n_samples = 100
    data = generate_samples(n_samples, n_dims)

    grid_ranges = fill(-5.0:0.1:5.0, n_dims)
    grid = initialize_grid(grid_ranges)
    grid_array = get_coordinates(grid)
    grid_fourier = fftgrid(grid)
    fourier_array = get_coordinates(grid_fourier)

    bw0 = Diagonal(fill(0.2, n_dims))
    density, density_squared = initialize_density(data, grid_array, bw0)

    density_complex0 = Array{ComplexF64}(density)
    density_squared_complex0 = Array{ComplexF64}(density_squared)

    fft_plan = plan_fft!(density_complex0)

    ParallelKDE.fourier_statistics!(
      Val(implementation),
      reshape(density_complex0, size(density_complex0)..., 1),
      reshape(density_squared_complex0, size(density_squared_complex0)..., 1),
      fft_plan,
    )

    density_fourier = fft(density)
    density_squared_fourier = fft(density_squared)

    @test density_fourier ≈ density_complex0
    @test density_squared_fourier ≈ density_squared_complex0

    bw1 = Diagonal(fill(√(2 * 0.2^2), n_dims))
    density, density_squared = initialize_density(data, grid_array, bw1)

    density_fourier = fft(density)
    density_squared_fourier = fft(density_squared)

    density_complex1 = Array{ComplexF64}(density)
    density_squared_complex1 = Array{ComplexF64}(density_squared)

    ParallelKDE.propagate_statistics!(
      Val(implementation),
      reshape(density_complex1, size(density_complex1)..., 1),
      reshape(density_squared_complex1, size(density_squared_complex1)..., 1),
      reshape(density_complex0, size(density_complex0)..., 1),
      reshape(density_squared_complex0, size(density_squared_complex0)..., 1),
      SVector{n_dims,Float64}(fill(0.2, n_dims)),
      SVector{n_dims,Float64}(fill(0.2, n_dims)),
      fourier_array,
    )

    @test density_fourier ≈ density_complex1
    @test density_squared_fourier ≈ density_squared_complex1 rtol = 0.1

    ifft_plan = plan_ifft!(density_complex1)

    ParallelKDE.ifourier_statistics!(
      Val(implementation),
      reshape(density_complex1, size(density_complex1)..., 1),
      reshape(density_squared_complex1, size(density_squared_complex1)..., 1),
      ifft_plan,
    )
    density_propagated = abs.(density_complex1)
    density_squared_propagated = abs.(density_squared_complex1)

    @test density ≈ density_propagated
    @test density_squared ≈ density_squared_propagated rtol = 0.1
  end
end

# if CUDA.functional()
#   @testset "GPU Fourier space operations tests. $(n_dims)D" for n_dims in 1:3
#     n_samples = 100
#     data = generate_samples(n_samples, n_dims)
#
#     grid_ranges = fill(-5.0:0.1:5.0, n_dims)
#     grid = initialize_grid(grid_ranges)
#     grid_array = get_coordinates(grid)
#     grid_fourier = fftgrid(grid)
#     fourier_array = get_coordinates(grid_fourier)
#
#     bw0 = Diagonal(fill(0.2, n_dims))
#     density, density_squared = initialize_density(data, grid_array, bw0)
#
#     density_complex0 = CuArray{ComplexF64}(density)
#     density_complex0 = reshape(density_complex0, size(density_complex0)..., 1)
#     density_squared_complex0 = CuArray{ComplexF64}(density_squared)
#     density_squared_complex0 = reshape(density_squared_complex0, size(density_squared_complex0)..., 1)
#
#     fft_plan = plan_fft!(density_complex0)
#
#     ParallelKDE.fourier_statistics!(
#       Val(:cuda),
#       density_complex0,
#       density_squared_complex0,
#       fft_plan,
#     )
#
#     density_fourier = fft(density)
#     density_squared_fourier = fft(density_squared)
#
#     @test density_fourier ≈ dropdims(Array(density_complex0), dims=n_dims + 1)
#     @test density_squared_fourier ≈ dropdims(Array(density_squared_complex0), dims=n_dims + 1)
#
#     bw1 = Diagonal(fill(√(2 * 0.2^2), n_dims))
#     density, density_squared = initialize_density(data, grid_array, bw1)
#
#     density_fourier = fft(density)
#     density_squared_fourier = fft(density_squared)
#
#     density_complex1 = CuArray{ComplexF64}(density)
#     density_complex1 = reshape(density_complex1, size(density_complex1)..., 1)
#     density_squared_complex1 = CuArray{ComplexF64}(density_squared)
#     density_squared_complex1 = reshape(density_squared_complex1, size(density_squared_complex1)..., 1)
#
#     ParallelKDE.propagate_statistics!(
#       Val(:cuda),
#       density_complex1,
#       density_squared_complex1,
#       density_complex0,
#       density_squared_complex0,
#       CUDA.fill(0.2, n_dims),
#       CUDA.fill(0.2, n_dims),
#       CuArray{Float32}(fourier_array),
#     )
#
#     @test density_fourier ≈ dropdims(Array(density_complex1), dims=n_dims + 1)
#     @test density_squared_fourier ≈ dropdims(Array(density_squared_complex1), dims=n_dims + 1)
#
#     ifft_plan = plan_ifft!(density_complex1)
#
#     ParallelKDE.ifourier_statistics!(
#       Val(:cuda),
#       density_complex1,
#       density_squared_complex1,
#       ifft_plan,
#     )
#     density_propagated = abs.(density_complex1)
#     density_squared_propagated = abs.(density_squared_complex1)
#
#     @test density ≈ dropdims(Array(density_propagated), dims=n_dims + 1)
#     @test density_squared ≈ dropdims(Array(density_squared_propagated), dims=n_dims + 1)
#   end
# end
