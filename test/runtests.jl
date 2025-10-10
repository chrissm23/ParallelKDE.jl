using ParallelKDE

using Test
using Aqua

using Statistics,
  LinearAlgebra,
  Random

using StaticArrays,
  FFTW,
  CUDA,
  Distributions

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

  @testset "Fourier space tests" begin
    include("fourierspace_tests.jl")
  end

  @testset "Interface tests (Estimations)" begin
    include("densityestimation_tests.jl")
  end

  @testset "Parallel Estimation tests" begin
    include("parallelestimation_tests.jl")
  end

  @testset "Rules of Thumb tests" begin
    include("rulesofthumb_tests.jl")
  end

  @testset "API tests" begin
    include("parallelkde_tests.jl")
  end

end
