using ParallelKDE
using ParallelKDE.Grids
using ParallelKDE.KDEs
using ParallelKDE.DirectSpace
using ParallelKDE.FourierSpace

using Test
using Aqua

using Statistics,
  LinearAlgebra

using StaticArrays,
  FFTW,
  CUDA,
  Distributions

include("test_utils.jl")

# TODO: Once the package is ready and tested, limit tests to 1D to run on
# GitHub hosted runners
@testset "ParallelKDE.jl" begin
  @testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(
      ParallelKDE,
      unbound_args=false,
    )
  end

  @testset "Grid tests" begin
    include("grid_tests.jl")
  end

  @testset "KDE object tests" begin
    include("kde_tests.jl")
  end

  @testset "Direct space tests" begin
    include("directspace_tests.jl")
  end

  @testset "Fourier space tests" begin
    include("fourierspace_tests.jl")
  end

end
