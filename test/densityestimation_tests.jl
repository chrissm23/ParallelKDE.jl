@testset "Density estimators API tests" begin
  @testset "Valid implementation tests" for implementation in [:serial, :threaded, :cuda]
    if implementation in [:serial, :threaded]
      @test_throws ArgumentError ParallelKDE.DensityEstimators.ensure_valid_implementation(
        ParallelKDE.Devices.IsCUDA(), implementation
      )
    else
      @test_throws ArgumentError ParallelKDE.DensityEstimators.ensure_valid_implementation(
        ParallelKDE.Devices.IsCPU(), implementation
      )
    end

    @test_throws ArgumentError ParallelKDE.DensityEstimators.ensure_valid_implementation(
      ParallelKDE.Devices.DeviceNotSpecified(), implementation
    )
  end

  estimator_keys = [
    :parallelEstimator
  ]
  estimator_types = [
    ParallelKDE.DensityEstimators.AbstractParallelEstimator
  ]
  @testset "Included estimators tests" for (estimator_key, estimator_type) in zip(estimator_keys, estimator_types)
    @test (estimator_key => estimator_type) ∈ ParallelKDE.DensityEstimators.estimator_lookup
    @test hasmethod(
      ParallelKDE.DensityEstimators.initialize_estimator,
      (Type{<:estimator_type}, ParallelKDE.KDEs.AbstractKDE)
    )
    @test hasmethod(
      ParallelKDE.DensityEstimators.estimate!, (estimator_type, ParallelKDE.KDEs.AbstractKDE)
    )
  end
end
