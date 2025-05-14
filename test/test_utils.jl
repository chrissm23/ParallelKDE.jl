function normal_distribution(
  x::AbstractVector{S},
  μ::SVector{N,S},
  bandwidth::AbstractMatrix{P}
) where {N,S<:Real,P<:Real}
  normal_distro = MvNormal(μ, bandwidth .^ 2)

  return pdf(normal_distro, x)
end

function generate_samples(n_samples::Integer, n_dims::Int; normal_distro=nothing)
  μ = zeros(n_dims)
  cov = Diagonal(ones(n_dims))

  if normal_distro === nothing
    normal_distro = MvNormal(μ, cov)
  end

  samples = SVector{n_dims,Float64}.(eachcol(rand(normal_distro, n_samples)))

  return samples
end
