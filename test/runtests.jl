using ParallelKDE
using Test
using Aqua

@testset "ParallelKDE.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ParallelKDE)
    end
    # Write your tests here.
end
