module FourierSpace

using ..Grids
using ..KDEs

using StaticArrays,
  FFTW,
  CUDA

export initialize_fourier_statistics,
  ifft_statistics

function initialize_fourier_statistics(
  dirac_series::Array{T,M},
  dirac_series_squared::Array{T,M}
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  sk_0 = fftshift(fft(dirac_series, 2:N+1), 2:N+1)
  s2k_0 = fftshift(fft(dirac_series_squared, 2:N+1), 2:N+1)

  return sk_0, s2k_0

end
function initialize_fourier_statistics(
  dirac_series::CuArray{T,M},
  dirac_series_squared::CuArray{T,M}
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  sk_0 = fftshift(fft(dirac_series, 2:N+1), 2:N+1)
  s2k_0 = fftshift(fft(dirac_series_squared, 2:N+1), 2:N+1)

  return sk_0, s2k_0

end

function ifft_statistics(
  sk::Array{Complex{T},M},
  s2k::Array{Complex{T},M},
  n_samples::Integer;
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  s = abs.(ifft(ifftshift(sk, 2:N+1), 2:N+1))
  s2 = abs.(ifft(ifftshift(s2k, 2:N+1), 2:N+1))

  means = s ./ n_samples
  variances = (s2 ./ n_samples) .- means .^ 2

  return means, variances
end
function ifft_statistics(
  sk::Array{Complex{T},M},
  s2k::Array{Complex{T},M},
  n_samples::Integer,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  s = abs.(ifft_plan * ifftshift(sk, 2:N+1))
  s2 = abs.(ifft_plan * ifftshift(s2k, 2:N+1))

  means = s ./ n_samples
  variances = (s2 ./ n_samples) .- means .^ 2

  return means, variances
end
function ifft_statistics(
  sk::CuArray{Complex{T},M},
  s2k::CuArray{Complex{T},M},
  n_samples::Integer;
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  s = abs.(ifft(ifftshift(sk, 2:N+1), 2:N+1))
  s2 = abs.(ifft(ifftshift(s2k, 2:N+1), 2:N+1))

  means = s ./ n_samples
  variances = (s2 ./ n_samples) .- means .^ 2

  return means, variances
end
function ifft_statistics(
  sk::CuArray{Complex{T},M},
  s2k::CuArray{Complex{T},M},
  n_samples::Integer,
  ifft_plan::AbstractFFTs.ScaledPlan{Complex{T}},
) where {T<:Real,M}
  N = M - 1
  if N < 1
    throw(ArgumentError("The dimension of the input array must be at least 1."))
  end

  s = abs.(ifft_plan * ifftshift(sk, 2:N+1))
  s2 = abs.(ifft_plan * ifftshift(s2k, 2:N+1))

  means = s ./ n_samples
  variances = (s2 ./ n_samples) .- means .^ 2

  return means, variances
end

end
