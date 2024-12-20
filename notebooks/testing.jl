### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 13e93433-5900-457b-b47e-3de1cf3ee411
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
	using Revise, BenchmarkTools, StaticArrays, CUDA, FFTW, Plots
	using ParallelKDE
end

# ╔═╡ c9ed396b-1701-4d9f-8384-4962b0a1be9c
begin
	n_samples = 1000
	n_bootstraps = 4
	data = [SVector{1, Float64}(rand(Float64, 1)) for _ in 1:n_samples]
	grid_ranges = [-1:0.01:2]
	kde = ParallelKDE.initialize_kde(data, grid_ranges, :cpu)
	# kde_d = ParallelKDE.initialize_kde(data, grid_ranges, :gpu)
end

# ╔═╡ 9e5252b3-96fd-41dc-bdbd-67ccadb59e2e
@btime begin
	means_0, variances_0 = ParallelKDE.initialize_statistics(
		kde,
		n_bootstraps,
		:cpu
	)
end;

# ╔═╡ 2008a9d4-faf1-4184-83ce-2c55d005cbf2
@btime begin
	means_0, variances_0 = ParallelKDE.initialize_statistics(
		kde,
		n_bootstraps,
		:threaded
	)
end;

# ╔═╡ Cell order:
# ╠═13e93433-5900-457b-b47e-3de1cf3ee411
# ╠═c9ed396b-1701-4d9f-8384-4962b0a1be9c
# ╠═9e5252b3-96fd-41dc-bdbd-67ccadb59e2e
# ╠═2008a9d4-faf1-4184-83ce-2c55d005cbf2
