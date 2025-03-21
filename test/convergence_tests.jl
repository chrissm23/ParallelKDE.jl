# @testset "Density-variance proportionality test (CPU)" for n_dims in 1:1
#   n_samples = 10000
#   @testset "Dimensions: $n_dims" begin
#     # Set up
#     data = generate_samples(n_samples, n_dims)
#
#     grid_ranges = fill(-5.0:0.05:5.0, n_dims)
#     grid = Grid(grid_ranges)
#
#     kde = initialize_kde(data, grid_ranges, :cpu)
#     dt = fill(0.1, n_dims)
#     t0 = Grids.initial_bandwidth(grid)
#
#     # Time propagation
#     density_0, var_0 = ParallelKDE.initialize_distribution(kde, :serial)
#     density_0_reshaped = reshape(density_0, 1, size(density_0)...)
#     var_0_reshaped = reshape(var_0, 1, size(var_0)...)
#
#     fourier_grid = fftgrid(grid)
#     fourier_grid_array = get_coordinates(fourier_grid)
#
#     means_dst = Array{ComplexF64,n_dims + 1}(undef, size(density_0_reshaped))
#     variances_dst = Array{ComplexF64,n_dims + 1}(undef, size(var_0_reshaped))
#
#     mean_complete, variance_complete = ParallelKDE.propagate_bandwidth!(
#       density_0_reshaped,
#       var_0_reshaped,
#       fourier_grid_array,
#       dt,
#       t0,
#       :serial,
#       dst_mean=means_dst,
#       dst_var=variances_dst,
#     )
#     mean_complete = dropdims(mean_complete, dims=1)
#     variance_complete = dropdims(variance_complete, dims=1)
#
#     mean_complete, variance_complete = ifft_statistics!(
#       mean_complete,
#       variance_complete,
#       n_samples,
#       bootstraps_dim=false
#     )
#
#     # Calculation of true PDF
#     normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
#     true_pdf = pdf.(
#       Ref(normal_distro),
#       eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
#     )
#
#     # Calculation of optimal threshold
#     det_t = prod(dt .^ 2 .+ t0 .^ 2)
#     scaling_factor = 1 / (2^n_dims * Float64(n_samples)^2 * sqrt(π^n_dims * det_t))
#
#     var_true = true_pdf .* scaling_factor
#
#     @test var_true ≈ real.(variance_complete) atol = 1e-1
#   end
# end
# @testset "Density-variance proportionality test (GPU)" for n_dims in 1:1
#   n_samples = 10000
#   @testset "dimensions: $n_dims" begin
#     # Set up
#     data = generate_samples(n_samples, n_dims)
#
#     grid_ranges = fill(-5.0:0.05:5.0, n_dims)
#     grid = Grid(grid_ranges)
#     grid_d = CuGrid(grid_ranges)
#
#     kde_d = initialize_kde(data, grid_ranges, :gpu)
#     dt_d = CUDA.fill(1.0f-1, n_dims)
#     t0_d = Grids.initial_bandwidth(grid_d)
#
#     # Time propagation
#     density_0_d, var_0_d = ParallelKDE.initialize_distribution(kde_d)
#     density_0_reshaped_d = reshape(density_0_d, 1, size(density_0_d)...)
#     var_0_reshaped_d = reshape(var_0_d, 1, size(var_0_d)...)
#
#     fourier_grid_d = fftgrid(grid_d)
#     fourier_grid_array_d = get_coordinates(fourier_grid_d)
#
#     means_dst_d = CuArray{ComplexF32,n_dims + 1}(undef, size(density_0_reshaped_d))
#     variances_dst_d = CuArray{ComplexF32,n_dims + 1}(undef, size(var_0_reshaped_d))
#
#     mean_complete_d, variance_complete_d = ParallelKDE.propagate_bandwidth!(
#       density_0_reshaped_d,
#       var_0_reshaped_d,
#       fourier_grid_array_d,
#       dt_d,
#       t0_d,
#       dst_mean=means_dst_d,
#       dst_var=variances_dst_d,
#     )
#     mean_complete_d = dropdims(mean_complete_d, dims=1)
#     variance_complete_d = dropdims(variance_complete_d, dims=1)
#
#     mean_complete_d, variance_complete_d = ifft_statistics!(
#       mean_complete_d,
#       variance_complete_d,
#       n_samples,
#       bootstraps_dim=false
#     )
#
#     # Calculation of true PDF
#     normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
#     true_pdf = pdf.(
#       Ref(normal_distro),
#       eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
#     )
#
#     # Calculation of optimal threshold
#     det_t = prod(dt_d .^ 2 .+ t0_d .^ 2)
#     scaling_factor = 1 / (2^n_dims * Float64(n_samples)^2 * sqrt(π^n_dims * det_t))
#
#     var_true = true_pdf .* scaling_factor
#
#     @test var_true ≈ Array(real.(variance_complete_d)) atol = 1e-1
#   end
# end
#
# # TODO: Once the adjustable_factor is known for higher dimensions, extend the loop to test them
# @testset "Convergence criterium test (CPU)" for n_dims in 1:1
#   n_samples = 10000
#   n_bootstraps = 50
#   @testset "Dimensions: $n_dims" begin
#     # Set up
#     data = generate_samples(n_samples, n_dims)
#
#     grid_ranges = fill(-5.0:0.05:5.0, n_dims)
#     grid = Grid(grid_ranges)
#
#     kde = initialize_kde(data, grid_ranges, :cpu)
#     dt = fill(0.1, n_dims)
#     t0 = Grids.initial_bandwidth(grid)
#
#     # Time propagation
#     means_0, variances_0 = ParallelKDE.initialize_statistics(kde, n_bootstraps, :serial)
#
#     fourier_grid = fftgrid(grid)
#     fourier_grid_array = get_coordinates(fourier_grid)
#
#     means_dst = Array{ComplexF64,n_dims + 1}(undef, size(means_0))
#     variances_dst = Array{ComplexF64,n_dims + 1}(undef, size(variances_0))
#
#     means_bootstraps, variances_bootstraps = ParallelKDE.propagate_bandwidth!(
#       means_0,
#       variances_0,
#       fourier_grid_array,
#       dt,
#       t0,
#       :serial,
#       dst_mean=means_dst,
#       dst_var=variances_dst,
#     )
#     means_bootstraps, variances_bootstraps = ifft_statistics!(
#       means_bootstraps,
#       variances_bootstraps,
#       n_samples,
#       bootstraps_dim=true
#     )
#
#     vmr_variance = ParallelKDE.calculate_statistics!(
#       means_bootstraps,
#       variances_bootstraps,
#       :serial,
#       dst_vmr=selectdim(variances_dst, 1, 1)
#     )
#
#     # Calculation of true PDF
#     normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
#     true_pdf = pdf.(
#       Ref(normal_distro),
#       eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
#     )
#
#     # Calculation of optimal threshold
#     det_t = prod(dt .^ 2 .+ t0 .^ 2)
#     adjustable_factor = 1 / 150
#     threshold_factor = 1 / (
#       2^(n_dims - 1) * (2^n_dims - 3^(n_dims / 2)) * (π^n_dims * det_t)^(3 / 2) * Float64(n_samples)^3
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
#
# if CUDA.functional()
#   @testset "Convergence criterium test (GPU)" for n_dims in 1:1
#     n_samples = 10000
#     n_bootstraps = 50
#     @testset "dimensions: $n_dims" begin
#       # Set up
#       data = generate_samples(n_samples, n_dims)
#
#       grid_ranges = fill(-5.0:0.05:5.0, n_dims)
#       grid = Grid(grid_ranges)
#       grid_d = CuGrid(grid_ranges)
#
#       kde_d = initialize_kde(data, grid_ranges, :gpu)
#       dt_d = CUDA.fill(1.0f-1, n_dims)
#       t0_d = Grids.initial_bandwidth(grid_d)
#
#       # Time propagation
#       means_0_d, variances_0_d = ParallelKDE.initialize_statistics(kde_d, n_bootstraps)
#
#       fourier_grid_d = fftgrid(grid_d)
#       fourier_grid_array_d = get_coordinates(fourier_grid_d)
#
#       means_dst_d = CuArray{ComplexF32,n_dims + 1}(undef, size(means_0_d))
#       variances_dst_d = CuArray{ComplexF32,n_dims + 1}(undef, size(variances_0_d))
#
#       means_bootstraps_d, variances_bootstraps_d = ParallelKDE.propagate_bandwidth!(
#         means_0_d,
#         variances_0_d,
#         fourier_grid_array_d,
#         dt_d,
#         t0_d,
#         dst_mean=means_dst_d,
#         dst_var=variances_dst_d,
#       )
#       means_bootstraps_d, variances_bootstraps_d = ifft_statistics!(
#         means_bootstraps_d,
#         variances_bootstraps_d,
#         n_samples,
#         bootstraps_dim=true
#       )
#
#       vmr_variance_d = ParallelKDE.calculate_statistics!(
#         means_bootstraps_d,
#         variances_bootstraps_d,
#         dst_vmr=selectdim(variances_dst_d, 1, 1)
#       )
#       vmr_variance = Array(vmr_variance_d)
#
#       # Calculation of true PDF
#       normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
#       true_pdf = pdf.(
#         Ref(normal_distro),
#         eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
#       )
#
#       # Calculation of optimal threshold
#       det_t = prod(dt_d .^ 2 .+ t0_d .^ 2)
#       adjustable_factor = 1 / 150
#       threshold_factor = 1 / (
#         2^(n_dims - 1) * (2^n_dims - 3^(n_dims / 2)) * (π^n_dims * det_t)^(3 / 2) * Float64(n_samples)^3
#       )
#       optimal_threshold = threshold_factor * adjustable_factor
#
#       vmr_true = optimal_threshold ./ true_pdf
#
#       @test isapprox(
#         view(vmr_variance, fill(50:150, n_dims)...), view(vmr_true, fill(50:150, n_dims)...),
#         rtol=1e0
#       )
#     end
#   end
# end

@testset "Testing product of variances (CPU)" for n_dims in 1:1
  n_samples = 10000
  n_bootstraps = 100
  Random.seed!(1234)

  @testset "Dimensions: $n_dims" begin
    data = generate_samples(n_samples, n_dims)

    grid_ranges = fill(-5.0:0.005:5.0, n_dims)
    grid = Grid(grid_ranges)

    normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
    true_pdf = pdf.(
      Ref(normal_distro),
      eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
    )
    # data = SVector{n_dims,Float64}.(eachcol(rand(normal_distro, n_samples)))

    adjustable_factor = 1 / 80
    threshold = ParallelKDE.get_smoothness(adjustable_factor, n_samples, n_dims)
    threshold_line = fill(threshold, length(0.0:0.01:0.3)...)

    kde = initialize_kde(data, grid_ranges, :cpu)
    t0 = Grids.initial_bandwidth(grid)

    means_0, variances_0 = ParallelKDE.initialize_statistics(kde, n_bootstraps, :serial)
    density_0, var_0 = ParallelKDE.initialize_distribution(kde, :serial)

    fourier_grid = fftgrid(grid)
    fourier_grid_array = get_coordinates(fourier_grid)

    means_dst = Array{ComplexF64,n_dims + 1}(undef, size(means_0))
    variances_dst = Array{ComplexF64,n_dims + 1}(undef, size(variances_0))
    convergence_tmp = fill(NaN, 2, size(kde.grid)...)

    dts = collect(0.0:0.01:1.0)
    # dts = collect(0.0:0.16:2.0)
    for dt_scalar in dts
      dt = fill(dt_scalar, n_dims)

      means_fourier, variances_fourier = ParallelKDE.propagate_bandwidth!(
        means_0,
        variances_0,
        fourier_grid_array,
        dt,
        t0,
        :serial,
        dst_mean=means_dst,
        dst_var=variances_dst,
      )
      means_direct, variances_direct = ifft_statistics!(
        means_fourier,
        variances_fourier,
        n_samples,
        bootstraps_dim=true
      )
      vmr_variance = ParallelKDE.calculate_statistics!(
        means_direct,
        variances_direct,
        :serial,
        dst_vmr=selectdim(
          selectdim(reinterpret(reshape, Float64, variances_dst), 1, 1),
          1,
          1
        )
      )
      mean_fourier, variance_fourier = ParallelKDE.propagate_bandwidth!(
        reshape(density_0, 1, size(density_0)...),
        reshape(var_0, 1, size(var_0)...),
        fourier_grid_array,
        dt,
        t0,
        :serial,
        dst_mean=reshape(selectdim(means_dst, 1, 1), 1, size(means_dst)[2:end]...),
        dst_var=reshape(selectdim(means_dst, 1, 2), 1, size(variances_dst)[2:end]...)
      )
      mean_fourier = dropdims(mean_fourier, dims=1)
      variance_fourier = dropdims(variance_fourier, dims=1)
      mean_direct, variance_direct = ifft_statistics!(
        mean_fourier,
        variance_fourier,
        n_samples,
        bootstraps_dim=false
      )

      vmr_var_scaled = calculate_variance_products!(
        Val(:serial),
        vmr_variance,
        # variance_direct,
        dt,
        t0,
        dst_var_products=vmr_variance
      )

      DirectSpace.assign_converged_density!(
        Val(:serial),
        kde.density,
        mean_direct,
        vmr_var_scaled,
        threshold,
        convergence_tmp
      )
      println("# NaN: ", sum(isnan.(kde.density)))

    end

    p = plot(grid_ranges[1], kde.density .* n_samples, label="Estimated PDF", dpi=300)
    plot!(p, grid_ranges[1], true_pdf, label="True PDF")
    savefig(p, "variance_product.png")
  end
end

# @testset "Testing results (CPU)" for n_dims in 1:1
#   n_samples = 100
#   Random.seed!(1234)
#   @testset "dimensions: $n_dims" begin
#     data = generate_samples(n_samples, n_dims)
#
#     grid_ranges = fill(-5.0:0.005:5.0, n_dims)
#     grid = Grid(grid_ranges)
#
#     kde = initialize_kde(data, grid_ranges, :cpu)
#     dt = 0.02
#     n_steps = 101
#     n_bootstraps = 100
#     fit_kde!(
#       kde,
#       dt=dt,
#       n_steps=n_steps,
#       n_bootstraps=n_bootstraps,
#       smoothness=1 / 50
#     )
#
#     normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
#     true_pdf = pdf.(
#       Ref(normal_distro),
#       eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
#     )
#
#     p2 = plot(grid_ranges[1], true_pdf, label="PDF", dpi=300)
#     plot!(p2, grid_ranges[1], kde.density, label="KDE")
#     savefig(p2, "kde.png")
#
#     dv = prod(step.(grid_ranges))
#     mise = dv * sum((kde.density .- true_pdf) .^ 2)
#
#     # TODO: Give a more appropriate tolerance for the integrated squared error
#     @test mise < 0.1
#   end
# end
