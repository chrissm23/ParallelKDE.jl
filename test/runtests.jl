using ParallelKDE
using ParallelKDE.Grids
using ParallelKDE.DirectSpace

using Test
using Aqua

using StaticArrays
using CUDA

@testset "ParallelKDE.jl" begin
  @testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(ParallelKDE)
  end

  @testset "Grid tests" begin
    include("grid_tests.jl")
  end

  # @testset "KDE object tests" begin
  #   include("kde_tests.jl")
  # end

  @testset "Direct space tests" begin
    include("directspace_tests.jl")
  end

end
