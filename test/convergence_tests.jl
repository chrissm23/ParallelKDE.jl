@testset "Convergence criterium test (CPU)" for n_dims in 1:1
  # Set up
  n_samples = 10000
  data = generate_samples(n_samples, n_dims)

  grid_ranges = fill(-5.0:0.05:5.0, n_dims)
  grid = Grid(grid_ranges)

  kde = initialize_kde(data, grid_ranges, :cpu)
  dt = fill(0.1, n_dims)
  t0 = Grids.initial_bandwidth(grid)
  n_bootstraps = 50

  # Time propagation
  means_0, variances_0 = ParallelKDE.initialize_statistics(kde, n_bootstraps, :serial)
  density_0, var_0 = ParallelKDE.initialize_distribution(kde, :serial)

  ifft_plan_multi = plan_ifft!(means_0, 2:n_dims+1)

  fourier_grid = fftgrid(grid)
  fourier_grid_array = get_coordinates(fourier_grid)

  means_dst = Array{ComplexF64,n_dims + 1}(undef, size(means_0))
  variances_dst = Array{ComplexF64,n_dims + 1}(undef, size(variances_0))

  means_bootstraps, variances_bootstraps = ParallelKDE.propagate_bandwidth!(
    means_0,
    variances_0,
    fourier_grid_array,
    dt,
    t0,
    :serial,
    dst_mean=means_dst,
    dst_var=variances_dst,
  )

  vmr_variance = ParallelKDE.calculate_statistics!(
    means_bootstraps,
    variances_bootstraps,
    ifft_plan_multi,
    :serial,
    dst_vmr=selectdim(variances_dst, 1, 1)
  )

  # Calculation of true PDF
  normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
  true_pdf = pdf.(
    Ref(normal_distro),
    eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
  )

  # Calculation of optimal threshold
  det_t_squared = prod(dt .^ 2 .+ t0 .^ 2)
  adjustable_factor = 1 / 150
  threshold_factor = 1 / (
    2^(n_dims - 1) * (2^n_dims - 3^(n_dims / 2)) * (π^n_dims * det_t_squared)^(3 / 2) * Float64(n_samples)^3
  )
  optimal_threshold = threshold_factor * adjustable_factor

  vmr_true = optimal_threshold ./ true_pdf

  @test isapprox(
    view(vmr_variance, fill(50:150, n_dims)...), view(vmr_true, fill(50:150, n_dims)...),
    rtol=1e0
  )
end

# if CUDA.functional()
#   @testset "Convergence criterium test (GPU)" for n_dims in 1:1
#     # Set up
#     n_samples = 10000
#     data = generate_samples(n_samples, n_dims)
#
#     grid_ranges = fill(-5.0:0.05:5.0, n_dims)
#     grid = CuGrid(grid_ranges)
#
#     kde = initialize_kde(data, grid_ranges, :gpu)
#     dt = CUDA.fill(1.0f-1, n_dims)
#     t0 = Grids.initial_bandwidth(grid)
#     n_bootstraps = 50
#
#     # Time propagation
#     means_0, variances_0 = ParallelKDE.initialize_statistics(kde, n_bootstraps)
#     density_0, var_0 = ParallelKDE.initialize_distribution(kde)
#
#     ifft_plan_multi = plan_ifft!(means_0, 2:n_dims+1)
#
#     fourier_grid = fftgrid(grid)
#     fourier_grid_array = get_coordinates(fourier_grid)
#
#     means_dst = CuArray{ComplexF32,n_dims + 1}(undef, size(means_0))
#     variances_dst = CuArray{ComplexF32,n_dims + 1}(undef, size(variances_0))
#
#     means_bootstraps, variances_bootstraps = ParallelKDE.propagate_bandwidth!(
#       means_0,
#       variances_0,
#       fourier_grid_array,
#       dt,
#       t0,
#       dst_mean=means_dst,
#       dst_var=variances_dst,
#     )
#
#     vmr_variance = ParallelKDE.calculate_statistics!(
#       means_bootstraps,
#       variances_bootstraps,
#       ifft_plan_multi,
#       dst_vmr=selectdim(variances_dst, 1, 1)
#     )
#
#     # Calculation of true PDF
#     normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
#     true_pdf = pdf.(
#       Ref(normal_distro),
#       eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
#     )
#     true_pdf_d = CuArray{Float32}(true_pdf)
#
#     # Calculation of optimal threshold
#     det_t_squared = prod(dt .^ 2 .+ t0 .^ 2)
#     adjustable_factor = 1 / 150
#     threshold_factor = 1 / (
#       2^(n_dims - 1) * (2^n_dims - 3^(n_dims / 2)) * (π^n_dims * det_t_squared)^(3 / 2) * Float64(n_samples)^3
#     )
#     optimal_threshold = threshold_factor * adjustable_factor
#
#     vmr_true = optimal_threshold ./ true_pdf
#
#     @test isapprox(
#       view(vmr_variance, fill(50:150, n_dims)...), view(vmr_true, fill(50:150, n_dims)...),
#       rtol=1e0
#     )
#   end
# end

# TODO: Implement the density-variance proportionality test

# @testset "Density-variance proportionality test (CPU)"

@testset "Testing results (CPU)" for n_dims in 1:1
  n_samples = 100
  data = generate_samples(n_samples, n_dims)

  grid_ranges = fill(-5.0:0.05:5.0, n_dims)
  grid = Grid(grid_ranges)

  kde = initialize_kde(data, grid_ranges, :cpu)
  dt = 0.02
  n_steps = 50
  n_bootstraps = 50
  fit_kde!(
    kde,
    dt=dt,
    n_steps=n_steps,
    n_bootstraps=n_bootstraps,
  )

  normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
  true_pdf = pdf.(
    Ref(normal_distro),
    eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
  )

  # if n_dims == 1
  #   p = plot(grid_ranges[1], true_pdf, label="PDF", dpi=300)
  #   plot!(p, grid_ranges[1], kde.density, label="KDE")
  #   savefig(p, "kde_vs_true_pdf_1d.png")
  # end

  dv = prod(step.(grid_ranges))
  mise = dv * sum((kde.density .- true_pdf) .^ 2)

  # TODO: Give a more appropriate tolerance for the integrated squared error
  @test_broken mise < 0.1
end
