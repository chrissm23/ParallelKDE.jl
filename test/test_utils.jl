function generate_samples(n_samples::Integer, n_dims::Int)
  normal_distro = Normal(0.0, 1.0)
  samples = [@SVector rand(normal_distro, n_dims) for _ in 1:n_samples]

  return samples
end
