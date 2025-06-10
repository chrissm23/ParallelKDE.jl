### A Pluto.jl notebook ###
# v0.20.10

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d43e18a8-45e8-11f0-0c94-b3c8669c9ba7
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

# ╔═╡ 179a072f-7053-4b71-9213-d80b9244fe0c
md"### Distribution definition and sampling"

# ╔═╡ a2b3952e-8f2c-4d1f-b210-5de366082603
md"**Note:** Distribution is available in `distro` and samples are available in `data`"

# ╔═╡ 25668615-7996-4fbb-8090-36ce1595a3f8
define_distro(distro_name, n_dims) = define_distro(Val(distro_name), n_dims)

# ╔═╡ 2e39871e-0a75-4ced-989f-5aff3286bbba
sample_distro(distro, n_samples) = reshape(rand(distro, n_samples), :, n_samples)

# ╔═╡ 11806eb9-e028-42ab-b481-22ae6212979d
function define_distro(::Val{:normal}, n_dims)
	μ = fill(0.0, n_dims)
	Σ = Diagonal(fill(0.9, n_dims))

	return MvNormal(μ, Σ)
end;

# ╔═╡ b27b7776-9427-476f-bb7e-8527e562f339
function define_distro(::Val{:bimodal}, n_dims)
	μ1 = [-1.5, 0]
	Σ1 = Diagonal(fill(0.5, n_dims))
	w1 = 0.2
	
	μ2 = [0.5, 0]
	Σ2 = Diagonal(fill(1.5, n_dims))
	w2 = 1 - w1

	return MixtureModel([MvNormal(μ1, Σ1), MvNormal(μ2, Σ2)], [w1, w2])
end;

# ╔═╡ 4b5bd652-f628-4c62-95d3-cbe37c0eb4e5
md"Number of dimensions (`n_dims`)"

# ╔═╡ b4b3920c-bded-484b-8ecf-eaf299aa7f3c
@bind n_dims Select([2, 3], default=2)

# ╔═╡ b7ac3829-9cd4-4a79-a608-0391c9a81092
md"Distribution (`distro_name`)"

# ╔═╡ 6a91687c-ea59-40ce-a2f5-f1471e040d3c
@bind distro_name Select([:normal, :bimodal])

# ╔═╡ 9d9ff675-de05-4325-8596-76b90da71c5d
md"Number of samples (`n_samples`)"

# ╔═╡ 3a542276-188f-4ba7-b96b-532c89e53ae8
@bind n_samples Select([100, 1000, 10000, 100000, 1000000], default=1000)

# ╔═╡ e68287c5-a85b-450d-b1a0-8eacbcd59b9d
md"Number of grid points (`n_gridpoints`)"

# ╔═╡ 83d9fe03-281f-4370-ac5f-c91d75eba896
@bind n_gridpoints Slider(100:100:1000, default=300)

# ╔═╡ 1d736909-cbca-4ed6-8854-68ab46740d62
@bind resample Button("Resample")

# ╔═╡ 43dc3677-c228-4abf-ba6b-b34c106e1dc2
begin
	resample

	grid_min = -5.0
	grid_max = 5.0
	distro = define_distro(distro_name, n_dims)
	data = sample_distro(distro, n_samples)
	distro_support = fill(
		range(grid_min, grid_max, length=n_gridpoints), n_dims
	)
	coordinates_support = reinterpret(
		reshape, Float64, collect(Iterators.product(distro_support...))
	)
	distro_pdf = pdf.(
		Ref(distro),
		eachslice(coordinates_support, dims=Tuple(2:n_dims+1))
	)
	distro_pdf_t = transpose(distro_pdf)
end;

# ╔═╡ 73401400-07a5-4fd6-80f2-9a0774c15aaa
md"#### Test points"

# ╔═╡ b22fbfc4-900f-4b21-9eb1-825f09fbf207
begin
	test_points = [0 -1.0 1.0; 0 0 0]
	n_testpoints = size(test_points, 2)
	test_palette = palette(:bam, n_testpoints)
end;

# ╔═╡ 2e79f15a-4406-4fb2-bfe3-90f1f959240a
let
	p_info = contourf(
		distro_support..., 
		distro_pdf_t, 
		c=:blues, 
		levels=100,
		colorbar=false
	)
	
	scatter!(
		p_info, 
		data[1, :], 
		data[2, :], 
		ms=1, 
		mc=:firebrick, 
		malpha=0.2, 
		label="sample"
	)
end

# ╔═╡ f289e44f-a605-455f-84a5-363590407bcc
@bind calculate_seed Button("New random seed")

# ╔═╡ 90e3e5b2-9d6a-4534-bfaa-2fa1cc789f27
begin
	calculate_seed

	random_seed = rand(Int)
end;

# ╔═╡ ddba9257-1165-4360-a35c-470c6c7a060d
let
	support_minima = vec(minimum(coordinates_support, dims=Tuple(2:n_dims+1)))
	support_maxima = vec(maximum(coordinates_support, dims=Tuple(2:n_dims+1)))
	filter_mask = support_minima .< data .< support_maxima
	data_filtered = data[:, dropdims(reduce(&, filter_mask, dims=1), dims=1)]

	Random.seed!(random_seed)

	density_estimation = initialize_estimation(
		data_filtered,
		grid=true,
		grid_ranges=distro_support,
	)
	estimate_density!(
		density_estimation,
		:parallelEstimator,
		# time_step=0.00001,
		# smoothness_duration=0.005,
		# stable_duration=0.01,
		# eps1=1.5,
		# eps2=0.1,
		# n_bootstraps=1000,
	)

	density_estimated = get_density(density_estimation)
	density_estimated_t = transpose(density_estimated)

	p_true = contourf(
		distro_support...,
		distro_pdf_t,
		c=:blues, 
		levels=100,
		colorbar=false
	)
	p_approx = contourf(
		distro_support...,
		density_estimated_t,
		c=:greens,
		levels=100,
		colorbar=false
	)

	plot(p_true, p_approx, layout=(1,2), size=(1000, 400))
end

# ╔═╡ Cell order:
# ╠═d43e18a8-45e8-11f0-0c94-b3c8669c9ba7
# ╟─179a072f-7053-4b71-9213-d80b9244fe0c
# ╟─a2b3952e-8f2c-4d1f-b210-5de366082603
# ╟─25668615-7996-4fbb-8090-36ce1595a3f8
# ╟─2e39871e-0a75-4ced-989f-5aff3286bbba
# ╟─11806eb9-e028-42ab-b481-22ae6212979d
# ╟─b27b7776-9427-476f-bb7e-8527e562f339
# ╟─4b5bd652-f628-4c62-95d3-cbe37c0eb4e5
# ╟─b4b3920c-bded-484b-8ecf-eaf299aa7f3c
# ╟─b7ac3829-9cd4-4a79-a608-0391c9a81092
# ╟─6a91687c-ea59-40ce-a2f5-f1471e040d3c
# ╟─9d9ff675-de05-4325-8596-76b90da71c5d
# ╟─3a542276-188f-4ba7-b96b-532c89e53ae8
# ╟─e68287c5-a85b-450d-b1a0-8eacbcd59b9d
# ╟─83d9fe03-281f-4370-ac5f-c91d75eba896
# ╟─1d736909-cbca-4ed6-8854-68ab46740d62
# ╟─43dc3677-c228-4abf-ba6b-b34c106e1dc2
# ╟─73401400-07a5-4fd6-80f2-9a0774c15aaa
# ╠═b22fbfc4-900f-4b21-9eb1-825f09fbf207
# ╠═2e79f15a-4406-4fb2-bfe3-90f1f959240a
# ╟─f289e44f-a605-455f-84a5-363590407bcc
# ╠═90e3e5b2-9d6a-4534-bfaa-2fa1cc789f27
# ╠═ddba9257-1165-4360-a35c-470c6c7a060d
