using ParallelKDE
using ParallelKDE.Grids
using ParallelKDE.KDEs
using ParallelKDE.DirectSpace

using Test
using Aqua

using Statistics

using StaticArrays
using FFTW
using CUDA
using Distributions

include("test_utils.jl")

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

  # @testset "Fourier space tests" begin
  #   include("fourierspace_tests.jl")
  # end

end
