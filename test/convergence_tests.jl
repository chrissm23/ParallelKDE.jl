@testset "Testing results (CPU)" for n_dims in 1:1
  data = generate_samples(100, n_dims)
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

  # println(kde.density)

  normal_distro = MvNormal(zeros(n_dims), Diagonal(ones(n_dims)))
  true_pdf = pdf.(
    Ref(normal_distro),
    eachslice(get_coordinates(grid), dims=ntuple(i -> i + 1, n_dims))
  )

  dv = prod(step.(grid_ranges))
  mise = dv * sum((kde.density - true_pdf) .^ 2)

  # TODO: Give a more appropriate tolerance for the integrated squared error
  @test mise < 0.1
end
