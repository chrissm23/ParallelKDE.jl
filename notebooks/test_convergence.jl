### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 3add1086-3d31-11f0-0a9e-cd0909baedac
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()
	
	using TestEnv
	
	TestEnv.activate()
	
	using Revise
	using ParallelKDE

	using Statistics,
		LinearAlgebra,
		Random

	using StaticArrays,
		FFTW,
		CUDA,
		Distributions,
		StatsBase,
		Plots,
		PlutoUI

	pythonplot()
end

# ╔═╡ c823e934-2430-4c95-ae36-96879b565e47
md"### Distribution definitions and sampling"

# ╔═╡ dd1af6a8-61a4-45dd-8b99-04a133175f04
md"**Note:** Distribution is available in `distro` and samples are available in `data`"

# ╔═╡ 534f3f9e-bce8-47d4-a9a7-303e21cfb0ef
define_distro(distro_name) = define_distro(Val(distro_name))

# ╔═╡ 24267c16-0b02-445d-8d96-4aecd17bb52e
sample_distro(distro, n_samples) = reshape(rand(distro, n_samples), 1, n_samples)

# ╔═╡ ab810aa5-916c-4491-a346-ba098b052e15
function define_distro(::Val{:normal})
	μ = 10
	σ = 1

	return Normal(μ, σ)
end;

# ╔═╡ c22aeb1d-85b8-43d7-b938-d6ee5cf9d3e3
function define_distro(::Val{:bimodal})
	μ1 = 6
	σ1 = 1
	w1 = 0.4

	μ2 = 10
	σ2 = 2
	w2 = 1 - w1

	return MixtureModel(Normal, [(μ1, σ1), (μ2, σ2)], [w1, w2])
end;

# ╔═╡ 7f588fa0-525d-471a-9b3d-af73f4e083f5
function define_distro(::Val{:noncentral_χ2})
	ν = 2
	λ = 1.7

	return NoncentralChisq(ν, λ)
end;

# ╔═╡ 7e79580a-4b79-48b6-984e-1b9cb351611a
md"Distribution (`distro_name`)"

# ╔═╡ ba9f88bc-a309-4ed7-a08b-822ed48f19c1
@bind distro_name Select([:normal, :bimodal, :noncentral_χ2])

# ╔═╡ fe3cd654-ed53-415d-9429-beea88bedace
md"Number of samples (`n_samples`)"

# ╔═╡ 45f698a7-e91f-48b8-af2d-d77fe5264367
@bind n_samples Select([100, 1000, 10000, 100000, 1000000], default=1000)

# ╔═╡ 2473dd08-bff6-4d61-9448-fb9336404afd
md"Number of grid points (`n_gridpoints`)"

# ╔═╡ c9911762-16c7-4490-aa0e-0e5a5ff62524
@bind n_gridpoints Slider(100:100:2000, default=1000)

# ╔═╡ 5bf23fbc-556a-4014-a84b-47c6dedf0b9f
begin
	x_min = 0
	x_max = 20
	distro = define_distro(distro_name)
	data = sample_distro(distro, n_samples)
	distro_support = range(x_min, x_max, length=n_gridpoints)
	distro_pdf = pdf(distro, distro_support)
end;

# ╔═╡ b2645b3b-2251-4d6a-9a41-e69eb101e12b
let
	line_length = 0.02
	p_info = plot(distro_support, distro_pdf, label="PDF", lc=:cornflowerblue, lw=2)
	
	m = Matrix{Float64}(undef, 3, n_samples)
	m[1, :] .= vec(data)
	m[2, :] .= vec(data)
	m[3, :] .= NaN
	n = similar(m)
	n[1, :] .= 0
	n[2, :] .= line_length
	n[3, :] .= NaN

	xx = vec(m)
	yy = vec(n)
	
	plot!(p_info, xx, yy, seriestype=:path, lc=:firebrick, label=false, alpha=0.1)

	p_info
end

# ╔═╡ ca25a9af-0783-492c-8362-2c163c4ec5f1
md"### Full estimation"

# ╔═╡ 87b3bd79-f44a-41a2-9aac-201d6bb3f00a
@bind calculate_seed Button("New random seed")

# ╔═╡ 3e7d87af-6523-4218-86ee-e2af901bf491
begin
	calculate_seed

	random_seed = rand(Int)
end;

# ╔═╡ bbbd2e78-3a28-4783-9601-c883cf99d185
let
	Random.seed!(random_seed)
	density_estimation = initialize_estimation(
		data,
		grid=true,
		grid_ranges=distro_support
	)
	estimate_density!(
		density_estimation,
		:parallelEstimator,
		# smoothness_duration=5,
		# stable_duration=5,
		# eps1=0.1,
		# eps2=0.1,
	)
	
	density_estimated = get_density(density_estimation)

	p_estimate = plot(
		distro_support, distro_pdf, label="PDF", lw=2, lc=:cornflowerblue
	)
	plot!(
		p_estimate,
		distro_support,
		density_estimated,
		label="Estimation",
		lw=2,
		lc=:forestgreen
	)
end

# ╔═╡ 75e020f4-5682-46f3-8dff-7abeba257818
md"### VMR vs time"

# ╔═╡ c7ae5c52-72ca-4449-bfa6-18fb36003ace
begin
	Random.seed!(random_seed)

	grid_support = initialize_grid(distro_support)
	time_initial = initial_bandwidth(grid_support)

	times_range = range(0.0, 2.0, step=0.005)
	times = [[t] for t in times_range]
	
	kde = initialize_kde(data, size(grid_support))
	
	parallel_estimator = ParallelKDE.DensityEstimators.initialize_estimator(
		ParallelKDE.DensityEstimators.AbstractParallelEstimator,
		kde,
		method=:serial,
		grid=grid_support
	)

	vmr_time = fill(NaN, length(distro_support), length(times))
end;

# ╔═╡ fb725626-e5eb-4dfc-93e7-4946af668652
for (idx,time_propagated) in enumerate(times)
	# Propagate bootstrap samples
	ParallelKDE.DensityEstimators.propagate_bootstraps!(
		parallel_estimator.kernel_propagation,
		parallel_estimator.means_bootstraps,
		parallel_estimator.vars_bootstraps,
		parallel_estimator.grid_fourier,
		time_propagated,
		time_initial;
		method=:serial
	)

	# Fourier transform back
	ParallelKDE.DensityEstimators.ifft_bootstraps!(
		parallel_estimator.kernel_propagation; method=:serial
	)

	# Calculate VMR
	ParallelKDE.DensityEstimators.calculate_vmr!(
		parallel_estimator.kernel_propagation,
		time_propagated,
		parallel_estimator.grid_direct,
		n_samples;
		method=:serial
	)

	vmr_time[:, idx] .= ParallelKDE.DensityEstimators.get_vmr(parallel_estimator.kernel_propagation)
end

# ╔═╡ 62ea2943-0007-4278-8e70-e93e243ef1eb
plot(
	times_range,
	eachrow(vmr_time)[begin:200:end],
	label=false,
	ylims=(1e-5, 1e8),
	yaxis=:log,
	palette=palette(:vik, length(eachrow(vmr_time)[begin:200:end]))
)

# ╔═╡ Cell order:
# ╠═3add1086-3d31-11f0-0a9e-cd0909baedac
# ╟─c823e934-2430-4c95-ae36-96879b565e47
# ╟─dd1af6a8-61a4-45dd-8b99-04a133175f04
# ╟─534f3f9e-bce8-47d4-a9a7-303e21cfb0ef
# ╟─24267c16-0b02-445d-8d96-4aecd17bb52e
# ╟─ab810aa5-916c-4491-a346-ba098b052e15
# ╟─c22aeb1d-85b8-43d7-b938-d6ee5cf9d3e3
# ╟─7f588fa0-525d-471a-9b3d-af73f4e083f5
# ╟─7e79580a-4b79-48b6-984e-1b9cb351611a
# ╟─ba9f88bc-a309-4ed7-a08b-822ed48f19c1
# ╟─fe3cd654-ed53-415d-9429-beea88bedace
# ╟─45f698a7-e91f-48b8-af2d-d77fe5264367
# ╟─2473dd08-bff6-4d61-9448-fb9336404afd
# ╟─c9911762-16c7-4490-aa0e-0e5a5ff62524
# ╠═5bf23fbc-556a-4014-a84b-47c6dedf0b9f
# ╟─b2645b3b-2251-4d6a-9a41-e69eb101e12b
# ╟─ca25a9af-0783-492c-8362-2c163c4ec5f1
# ╟─87b3bd79-f44a-41a2-9aac-201d6bb3f00a
# ╟─3e7d87af-6523-4218-86ee-e2af901bf491
# ╠═bbbd2e78-3a28-4783-9601-c883cf99d185
# ╟─75e020f4-5682-46f3-8dff-7abeba257818
# ╠═c7ae5c52-72ca-4449-bfa6-18fb36003ace
# ╠═fb725626-e5eb-4dfc-93e7-4946af668652
# ╠═62ea2943-0007-4278-8e70-e93e243ef1eb
