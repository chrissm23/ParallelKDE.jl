### A Pluto.jl notebook ###
# v0.20.13

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

# ╔═╡ c1ec7102-c40a-4e7c-a490-7f2c189a8f07
md"Number of x testpoints (`n_xtestpoints`)"

# ╔═╡ f40919ff-d7f9-40fa-b5eb-c65c2d658323
@bind n_xtestpoints Slider(1:5, default=2)

# ╔═╡ e7fd4798-f05b-422c-a533-3e0fe8724726
md"Number of y testpoints (`n_ytestpoints`)"

# ╔═╡ 3ae37896-fcb4-4d78-9f9a-ba1b2d8f02ff
@bind n_ytestpoints Slider(1:5, default=2)

# ╔═╡ 8bd4c63f-bf8a-4076-95f2-915acd3f0ca0
md"Minimum x testpoint index (`min_xtestpoint`)"

# ╔═╡ fc15524a-1154-4d3d-a916-6ecbb274750d
@bind min_xtestpoint Slider(range(1,length(distro_support[1])), default=1)

# ╔═╡ e3afd164-f7be-46b3-9833-17604932650b
md"Maximum x testpoint index (`max_xtestpoint`)"

# ╔═╡ efe6bf4a-e306-49a2-92ce-0bb7635b631c
@bind max_xtestpoint Slider(
	range(1,length(distro_support[1])), default=length(distro_support[1])
)

# ╔═╡ 8353983d-ae6b-4fe1-9666-35f1425d9f4a
md"Minimum y testpoint index (`min_ytestpoint`)"

# ╔═╡ 0362ca1c-b62a-44b1-a3a7-b68f7a86cf3e
@bind min_ytestpoint Slider(
	range(1,length(distro_support[2])), default=min_xtestpoint
)

# ╔═╡ 9e0a0646-541e-4349-b170-ec418a16aaa3
md"Maximum y testpoint index (`max_ytestpoint`)"

# ╔═╡ a1d1d980-70c6-4439-bed3-d34faba4855b
@bind max_ytestpoint Slider(
	range(1,length(distro_support[2])), default=max_xtestpoint
)

# ╔═╡ b22fbfc4-900f-4b21-9eb1-825f09fbf207
begin
	test_indices_x = floor.(
		Int, range(min_xtestpoint, max_xtestpoint, length=n_xtestpoints)
	)
	test_indices_y = floor.(
		Int, range(min_ytestpoint, max_ytestpoint, length=n_ytestpoints)
	)
	test_indices = vec(collect(Iterators.product(test_indices_x, test_indices_y)))
	test_points = [
		(distro_support[1][idx[1]], distro_support[2][idx[2]]) 
		for idx in test_indices
	]
	test_palette = palette(:bam, length(test_indices))
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
		mc=:peru,
		msc=:peru,
		malpha=0.7, 
		label="sample"
	)

	for (i, tp) in enumerate(test_points)
		scatter!(
			p_info,
			[tp[1]],
			[tp[2]],
			ms=4,
			mc=test_palette[i],
			msc=test_palette[i],
			label=false,
		)
	end

	p_info
end

# ╔═╡ f289e44f-a605-455f-84a5-363590407bcc
@bind calculate_seed Button("New random seed")

# ╔═╡ 90e3e5b2-9d6a-4534-bfaa-2fa1cc789f27
begin
	calculate_seed

	random_seed = rand(Int)
end;

# ╔═╡ 0b4e1a19-e7d2-4acd-9687-ef8b3463f568
md"## Full result"

# ╔═╡ 28bef9db-c4a0-427a-b06c-695c1f51a0ad
md"Coordinate"

# ╔═╡ c4808fbf-2ff1-4503-9f97-a71a84dce25b
@bind slice_index Slider(1:length(distro_support[1]), default=0)

# ╔═╡ f593e0a1-5be6-48d3-8ab0-1e6d16a618cb
begin
	plot(distro_support[1], density_estimated[:, slice_index], c=:green, lw=2)
	plot!(distro_support[1], distro_pdf[:, slice_index], c=:blue, lw=2)
end

# ╔═╡ 515333c8-b1e4-411e-bd3b-d80bbbdaa878
begin
	p_true2 = contourf(
		distro_support...,
		distro_pdf_t,
		c=:blues, 
		levels=20,
		# colorbar=false
	)
	hline!(p_true2, [distro_support[1][slice_index, :]], lc=:red, label=false)
	
	p_approx2 = contourf(
		distro_support...,
		density_estimated_t,
		c=:greens,
		levels=20,
		# colorbar=false
	)
	hline!(p_approx2, [distro_support[1][slice_index, :]], lc=:red, label=false)

	plot(p_true2, p_approx2, layout=(2,1), size=(600, 1000))
end

# ╔═╡ 894fdf5a-5a49-40d5-84bc-ea6098bd9cb4
md"## Propagation behaviour"

# ╔═╡ e3ae3fb5-ab84-419d-95e0-8ad8f0404cbd
#=╠═╡
for (idx,time_propagated) in enumerate(eachcol(times))
	# Propagate bootstrap samples
	ParallelKDE.DensityEstimators.propagate_bootstraps!(
		gradepro_estimator.kernel_propagation,
		gradepro_estimator.means_bootstraps,
		gradepro_estimator.vars_bootstraps,
		gradepro_estimator.grid_fourier,
		time_propagated,
		time_initial;
		method
	)
	# Fourier transform back
	ParallelKDE.DensityEstimators.ifft_bootstraps!(
		gradepro_estimator.kernel_propagation; method=:serial
	)

	# Calculate VMR
	ParallelKDE.DensityEstimators.calculate_vmr!(
		gradepro_estimator.kernel_propagation,
		time_propagated,
		gradepro_estimator.grid_direct,
		n_samples;
		method
	)
	vmr_time[fill(Colon(), n_dims)..., idx] .= ParallelKDE.DensityEstimators.get_vmr(
		gradepro_estimator.kernel_propagation
	)

	# Propagate means of full samples
	ParallelKDE.DensityEstimators.propagate_means!(
		gradepro_estimator.kernel_propagation,
		gradepro_estimator.means,
		gradepro_estimator.grid_fourier,
		time_propagated;
		method
	)
	# Fourier transform back
	ParallelKDE.DensityEstimators.ifft_means!(
		gradepro_estimator.kernel_propagation;
		method
	)
	ParallelKDE.DensityEstimators.calculate_means!(
		gradepro_estimator.kernel_propagation,
		n_samples;
		method
	)
	means_time[fill(Colon(), n_dims)..., idx] .= ParallelKDE.DensityEstimators.get_means(
		gradepro_estimator.kernel_propagation
	)

	# Update convergence state
	ParallelKDE.DensityEstimators.update_state!(
		gradepro_estimator.density_state,
		kde,
		gradepro_estimator.kernel_propagation;
		method
	)
	is_smooth_time[fill(Colon(), n_dims)..., idx] .= (
		gradepro_estimator.density_state.is_smooth
	)
	has_decreased_time[fill(Colon(), n_dims)..., idx] .= (
		gradepro_estimator.density_state.has_decreased
	)
	is_stable_time[fill(Colon(), n_dims)..., idx] .= (
		gradepro_estimator.density_state.is_stable
	)

	@. density_assigned_time[fill(Colon(), n_dims)..., idx] = ifelse(
		(
			(kde.density ≈ previous_density) | 
			(isnan(kde.density) & isnan(previous_density))
		),
		false,
		true
	)
	previous_density .= kde.density
end
  ╠═╡ =#

# ╔═╡ f9cd5bb5-aafd-4c11-bbc4-1d5fed9dc649
md"Propagation time (`propagation_time`)"

# ╔═╡ 7e65eea3-5911-47ad-8a08-9ea809ba8425
#=╠═╡
@bind propagation_time_idx Slider(range( 1,n_times), default=1)
  ╠═╡ =#

# ╔═╡ 6480774f-be27-429a-bbb3-24454b1280d8
#=╠═╡
begin
	propagation_time = times[propagation_time_idx]
	propagation_t = norm(propagation_time)
	ts = norm.(times)
	dt = norm(time_step)
end;
  ╠═╡ =#

# ╔═╡ 27c22b1e-463b-4b01-8fac-0f388094e416
#=╠═╡
p_propagation = contourf(
	distro_support...,
	permutedims(means_time[fill(Colon(), n_dims)..., propagation_time_idx], (2,1)),
	c=:greens,
	levels=100,
	colorbar=false
)
  ╠═╡ =#

# ╔═╡ 3f12b0f1-f7b9-48f0-b364-318a7b32e428
#=╠═╡
begin
	derivatives1[fill(Colon(), n_dims)..., begin+1:end] .= abs.(
		diff(vmr_time, dims=n_dims+1) ./ (2*dt)
	)
	derivatives2[fill(Colon(), n_dims)..., begin+2:end] .= log10.(
		abs.(diff(diff(vmr_time, dims=n_dims+1), dims=n_dims+1) ./ dt^2)
	)
	optimal_times = ts[
		argmin.(
			eachslice(
				abs.(means_time .- reshape(distro_pdf, size(distro_pdf)..., 1)),
				dims=Tuple(1:n_dims)
			)
		)
	]
end;
  ╠═╡ =#

# ╔═╡ 0536cd2b-d2c1-4f7d-aa46-81287dd34623
#=╠═╡
begin
	test_means_time = [
		means_time[i,j,k] for k in 1:size(means_time,3), (i,j) in test_indices
	]
	test_vmr_time = [
		vmr_time[i,j,k] for k in 1:size(vmr_time,3), (i,j) in test_indices
	]
	test_distro_pdf = [
		distro_pdf[i,j] for (i,j) in test_indices
	]
	test_derivatives1 = [
		derivatives1[i,j,k] for k in 1:size(derivatives1,3), (i,j) in test_indices
	]
	test_derivatives2 = [
		derivatives2[i,j,k] for k in 1:size(derivatives2,3), (i,j) in test_indices
	]
	test_optimal = [
		optimal_times[i,j] for (i,j) in test_indices
	]
	test_is_smooth = [
		is_smooth_time[i,j,k] 
		for k in 1:size(is_smooth_time,3), (i,j) in test_indices
	]
	test_has_decreased = [
		has_decreased_time[i,j,k] 
		for k in 1:size(has_decreased_time,3), (i,j) in test_indices
	]
	test_is_stable = [
		is_stable_time[i,j,k] 
		for k in 1:size(is_stable_time,3), (i,j) in test_indices
	]
	test_density_assigned = [
		density_assigned_time[i,j,k] 
		for k in 1:size(density_assigned_time,3), (i,j) in test_indices
	]
end;
  ╠═╡ =#

# ╔═╡ c2eaa783-11ca-4a37-843a-63891fafaf89
#=╠═╡
begin
	p_optimal = plot(
		ts,
		eachcol(test_means_time),
		label=false,
		palette=test_palette,
		lw=2,
		ylims=1.1 .* extrema(test_means_time),
		xlims=extrema(ts),
	)

	for (i, val) in enumerate(test_distro_pdf)
		hline!(
			p_optimal,
			[val],
			label=false,
			lc=test_palette[i],
			lw=2,
			ls=:dash,
		)

		vline!(
			p_optimal,
			[test_optimal[i]],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dashdot,
		)
	end

	p_optimal

	# Manual label
	plot!(
		p_optimal,
		[-10; -20], lw=2, lc=:black, ls=:solid, label="Propagated means"
	)
	plot!(
		p_optimal,
		[-10; -20], lw=2, lc=:black, ls=:dash, label="PDF"
	)
	plot!(
		p_optimal,
		[-10; -20], lw=2, lc=:black, ls=:dashdot, label="Optimal time"
	)
end
  ╠═╡ =#

# ╔═╡ 80839339-a09d-4424-9025-af36aba59948
vmr_bounds = (-2, 8);

# ╔═╡ 9049f192-c0b9-46d6-aa11-86563871926d
md"### Finding smooth regime"

# ╔═╡ 7f3b3272-0d39-4b08-b705-358824d64ebb
#=╠═╡
begin
	p_smooth = plot(
		ts,
		eachcol(test_vmr_time),
		label=false,
		palette=test_palette,
		lw=2,
		ylims=vmr_bounds,
		xlims=extrema(ts),
		# yaxis=:log
	)

	for (i, test_idx) in enumerate(test_indices)
		vline!(
			p_smooth,
			[
				let
					try
						ts[findfirst(test_is_smooth[:, i])]
					catch
						NaN
					end
				end
			],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dash,
		)
	end

	# Manual label
	plot!(
		p_smooth, [-10; -20], lw=2, lc=:black, ls=:solid, label="Scaled VMR"
	)
	plot!(
		p_smooth, [-10; -20], lw=2, lc=:black, ls=:dash, label="Smoothness found"
	)
end
  ╠═╡ =#

# ╔═╡ 074148c8-4fbc-4a45-b7e3-58cea7d03aca
md"### Finding decrease point"

# ╔═╡ 1fa3871b-9f02-4e4c-9f26-c5f8cbf115e3
#=╠═╡
begin
	p_decrease = plot(
		ts,
		eachcol(test_vmr_time),
		label=false,
		palette=test_palette,
		lw=2,
		ylims=vmr_bounds,
		xlims=extrema(ts),
		# yaxis=:log
	)

	for (i, test_idx) in enumerate(test_indices)
		vline!(
			p_decrease,
			[
				let
					try
						ts[findfirst(test_has_decreased[:, i])]
					catch
						NaN
					end
				end
			],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dash,
		)
	end

	# Manual label
	plot!(
		p_decrease, [-10; -20], lw=2, lc=:black, ls=:solid, label="Scaled VMR"
	)
	plot!(
		p_decrease, [-10; -20], lw=2, lc=:black, ls=:dash, label="Decrease found"
	)
end
  ╠═╡ =#

# ╔═╡ 66bc5a7d-707c-4920-8c4e-db238ca87dd9
md"### Find stability point"

# ╔═╡ 64c2750e-bb4b-41bb-96e8-bf2955943fbe
#=╠═╡
begin
	p_stability = plot(
		ts,
		eachcol(test_vmr_time),
		label=false,
		palette=test_palette,
		lw=2,
		ylims=vmr_bounds,
		xlims=extrema(ts),
		# yaxis=:log
	)

	for (i, test_idx) in enumerate(test_indices)
		vline!(
			p_stability,
			[
				let
					try
						ts[findfirst(test_is_stable[:, i])]
					catch
						NaN
					end
				end
			],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dash,
		)
	end

	# Manual label
	plot!(
		p_stability, [-10; -20], lw=2, lc=:black, ls=:solid, label="Scaled VMR"
	)
	plot!(
		p_stability, [-10; -20], lw=2, lc=:black, ls=:dash, label="Stability found"
	)
end
  ╠═╡ =#

# ╔═╡ daac6add-0657-4a3a-bcba-0d4377b62b44
md"### First derivative"

# ╔═╡ bfbae558-2cda-4224-ac88-45122cf751a5
begin
	threshold1_range = range(0, 2, length=100)
	@bind threshold_1 Slider(
		threshold1_range,
		default=threshold1_range[argmin(abs.(threshold1_range .- 1.5))]
	)
end

# ╔═╡ 3114aa7d-f0c8-45cd-a7fa-b1cee9b9cd3b
#=╠═╡
begin
	p_dev1 = plot(
		ts,
		eachcol(test_derivatives1),
		label=false,
		palette=test_palette,
		lw=2,
		ylims=(0,2),
		xlims=extrema(ts),
		# yaxis=:log,
	)

	hline!(p_dev1, [threshold_1], c=:red, label="Threshold: $threshold_1")

	for i in 1:length(test_indices)
		vline!(
			p_dev1,
			[
				let
					try
						ts[findfirst(test_is_stable[:, i])]
					catch
						NaN
					end
				end
			],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dash,
		)
	end

	# Manual label
	plot!(
		p_dev1, [-10; -20], lw=2, lc=:black, ls=:solid, label="Scaled VMR"
	)
	plot!(
		p_dev1, [-10; -20], lw=2, lc=:black, ls=:dash, label="Stability found"
	)
end
  ╠═╡ =#

# ╔═╡ 3e5b37f5-ebc8-4785-a8c2-7d1c258e38c6
md"### Second derivative"

# ╔═╡ 5810171a-0703-4589-9534-12f4fde3bdcb
begin
	threshold2_range = range(0, 4, length=100)
	@bind threshold_2 Slider(
		threshold2_range,
		default=threshold2_range[argmin(abs.(threshold2_range .- 0.1))]
	)
end

# ╔═╡ 26e50e62-8e3b-48e4-8983-ded0c5b871ac
#=╠═╡
begin
	p_dev2 = plot(
		ts,
		eachcol(test_derivatives2),
		label=false,
		palette=test_palette,
		lw=2,
		ylims=(0,8),
		xlims=extrema(ts),
		# yaxis=:log,
		dpi=500,
	)

	hline!(p_dev2, [threshold_2], c=:red, label="Threshold: $threshold_2")

	for i in 1:length(test_indices)
		vline!(
			p_dev2,
			[
				let
					try
						ts[findfirst(test_is_stable[:, i])]
					catch
						NaN
					end
				end
			],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dash,
		)
	end

	# Manual label
	plot!(
		p_dev2, [-10; -20], lw=2, lc=:black, ls=:solid, label="Second derivative"
	)
	# plot!(
	# 	p_dev2, [-10; -20], lw=2, lc=:black, ls=:dash, label="Stability found"
	# )

	# savefig(p_dev2, "dev2_cpu.png")

	p_dev2
end
  ╠═╡ =#

# ╔═╡ 4f1bd110-b5ad-4066-a8d7-9973c989bb20
md"### Final stopping point"

# ╔═╡ 472e1f7c-d8c5-495f-954d-4ccf3adfc8bf
#=╠═╡
begin
	p_stopping = plot(
		ts,
		eachcol(test_vmr_time),
		label=false,
		palette=test_palette,
		lw=2,
		ylims=vmr_bounds,
		xlims=extrema(ts),
		# yaxis=:log,
		dpi=500,
	)

	for i in 1:length(test_points)
		vline!(
			p_stopping,
			[ts[findlast(test_density_assigned[:, i])]],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dash,
		)
		
		vline!(
			p_stopping,
			[test_optimal[i]],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dashdot,
		)
	end

	# Manual label
	plot!(
		p_stopping, [-10; -20], lw=2, lc=:black, ls=:solid, label="Scaled VMR"
	)
	plot!(
		p_stopping, [-10; -20], lw=2, lc=:black, ls=:dash, label="Stopping point"
	)
	plot!(
		p_stopping,
		[-10; -20], lw=1, lc=:black, ls=:dashdot, label="Optimal time"
	)

	# savefig(p_optimal, "vmr_cpu.png")

	p_stopping
end
  ╠═╡ =#

# ╔═╡ 781b579d-3fa5-45d8-b404-6c4bf55d3321
#=╠═╡
begin
	p_error = plot(
		ts,
		eachcol(test_means_time),
		label=false,
		palette=test_palette,
		lw=2,
		ylims=1.1 .* extrema(test_means_time),
		xlims=extrema(ts),
	)

	for (i, val) in enumerate(test_distro_pdf)
		hline!(
			p_error,
			[val],
			label=false,
			lc=test_palette[i],
			lw=2,
			ls=:dash,
		)

		vline!(
			p_error,
			[ts[findlast(test_density_assigned[:, i])]],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dash,
		)

		vline!(
			p_error,
			[test_optimal[i]],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dashdot,
		)
	end

	# Manual label
	plot!(
		p_error,
		[-10; -20], lw=2, lc=:black, ls=:solid, label="Propagated means"
	)
	plot!(
		p_error,
		[-10; -20], lw=2, lc=:black, ls=:dashdotdot, label="PDF"
	)
	plot!(
		p_error,
		[-10; -20], lw=2, lc=:black, ls=:dashdot, label="Optimal time"
	)
	plot!(
		p_error,
		[-10; -20], lw=2, lc=:black, ls=:dash, label="Stopping time"
	)
end
  ╠═╡ =#

# ╔═╡ ddba9257-1165-4360-a35c-470c6c7a060d
#=╠═╡
begin
	support_minima = vec(minimum(coordinates_support, dims=Tuple(2:n_dims+1)))
	support_maxima = vec(maximum(coordinates_support, dims=Tuple(2:n_dims+1)))
	filter_mask = support_minima .< data .< support_maxima
	data_filtered = data[:, dropdims(reduce(&, filter_mask, dims=1), dims=1)]

	Random.seed!(random_seed)

	density_estimation = initialize_estimation(
		data_filtered,
		grid=true,
		grid_ranges=distro_support,
		device=:cuda
	)
	estimate_density!(
		density_estimation,
		:gradepro,
		# time_step=0.00001,
		# stable_duration=0.01,
		# eps=-5.0,
		# n_bootstraps=1000,
	)

	density_estimated = Array(get_density(density_estimation))
	density_estimated_t = transpose(density_estimated)

	p_true1 = contourf(
		distro_support...,
		distro_pdf_t,
		c=:blues, 
		levels=100,
		colorbar=false
	)
	p_approx1 = contourf(
		distro_support...,
		density_estimated_t,
		c=:greens,
		levels=100,
		colorbar=false
	)

	plot(p_true1, p_approx1, layout=(1,2), size=(1000, 400))
end
  ╠═╡ =#

# ╔═╡ 6f8f9399-c8ab-4164-a549-c0e388076a07
# ╠═╡ disabled = true
#=╠═╡
begin
	Random.seed!(random_seed)

	grid_support = initialize_grid(distro_support, device=:cuda)
	time_initial = initial_bandwidth(grid_support)

	method=:cuda

	support_minima = vec(minimum(coordinates_support, dims=Tuple(2:n_dims+1)))
	support_maxima = vec(maximum(coordinates_support, dims=Tuple(2:n_dims+1)))
	filter_mask = support_minima .< data .< support_maxima
	data_filtered = data[:, dropdims(reduce(&, filter_mask, dims=1), dims=1)]

	kde = initialize_kde(data_filtered, size(grid_support), device=:cuda)
	
	gradepro_estimator = ParallelKDE.DensityEstimators.initialize_estimator(
		ParallelKDE.DensityEstimators.AbstractGradeProEstimator,
		kde;
		method,
		grid=grid_support,
		# time_step=0.00001,
		# eps1=1.5,
		# eps2=0.1,
		# smoothness_duration=0.005,
		# stable_duration=0.01,
	)

	times = gradepro_estimator.times
	n_times = length(times)
	times_reinterpreted = reinterpret(reshape, Float64, times)
	times_range = range(
		minimum(times_reinterpreted), maximum(times_reinterpreted), length=n_times
	)

	grid_dims = size(kde.density)
	previous_density = fill(NaN, grid_dims)

	derivatives1 = fill(NaN, grid_dims..., n_times)
	derivatives2 = fill(NaN, grid_dims..., n_times)

	vmr_time = fill(NaN, grid_dims..., n_times)
	means_time = fill(NaN, grid_dims..., n_times)
	density_assigned_time = falses(grid_dims..., n_times)
end;
  ╠═╡ =#

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
# ╠═43dc3677-c228-4abf-ba6b-b34c106e1dc2
# ╟─73401400-07a5-4fd6-80f2-9a0774c15aaa
# ╟─c1ec7102-c40a-4e7c-a490-7f2c189a8f07
# ╟─f40919ff-d7f9-40fa-b5eb-c65c2d658323
# ╟─e7fd4798-f05b-422c-a533-3e0fe8724726
# ╟─3ae37896-fcb4-4d78-9f9a-ba1b2d8f02ff
# ╟─8bd4c63f-bf8a-4076-95f2-915acd3f0ca0
# ╟─fc15524a-1154-4d3d-a916-6ecbb274750d
# ╟─e3afd164-f7be-46b3-9833-17604932650b
# ╟─efe6bf4a-e306-49a2-92ce-0bb7635b631c
# ╟─8353983d-ae6b-4fe1-9666-35f1425d9f4a
# ╟─0362ca1c-b62a-44b1-a3a7-b68f7a86cf3e
# ╟─9e0a0646-541e-4349-b170-ec418a16aaa3
# ╟─a1d1d980-70c6-4439-bed3-d34faba4855b
# ╟─b22fbfc4-900f-4b21-9eb1-825f09fbf207
# ╟─2e79f15a-4406-4fb2-bfe3-90f1f959240a
# ╟─f289e44f-a605-455f-84a5-363590407bcc
# ╟─90e3e5b2-9d6a-4534-bfaa-2fa1cc789f27
# ╟─0b4e1a19-e7d2-4acd-9687-ef8b3463f568
# ╠═ddba9257-1165-4360-a35c-470c6c7a060d
# ╟─28bef9db-c4a0-427a-b06c-695c1f51a0ad
# ╟─c4808fbf-2ff1-4503-9f97-a71a84dce25b
# ╠═f593e0a1-5be6-48d3-8ab0-1e6d16a618cb
# ╠═515333c8-b1e4-411e-bd3b-d80bbbdaa878
# ╟─894fdf5a-5a49-40d5-84bc-ea6098bd9cb4
# ╠═6f8f9399-c8ab-4164-a549-c0e388076a07
# ╠═e3ae3fb5-ab84-419d-95e0-8ad8f0404cbd
# ╟─f9cd5bb5-aafd-4c11-bbc4-1d5fed9dc649
# ╟─7e65eea3-5911-47ad-8a08-9ea809ba8425
# ╟─6480774f-be27-429a-bbb3-24454b1280d8
# ╟─27c22b1e-463b-4b01-8fac-0f388094e416
# ╟─3f12b0f1-f7b9-48f0-b364-318a7b32e428
# ╟─0536cd2b-d2c1-4f7d-aa46-81287dd34623
# ╟─c2eaa783-11ca-4a37-843a-63891fafaf89
# ╠═80839339-a09d-4424-9025-af36aba59948
# ╟─9049f192-c0b9-46d6-aa11-86563871926d
# ╟─7f3b3272-0d39-4b08-b705-358824d64ebb
# ╟─074148c8-4fbc-4a45-b7e3-58cea7d03aca
# ╟─1fa3871b-9f02-4e4c-9f26-c5f8cbf115e3
# ╟─66bc5a7d-707c-4920-8c4e-db238ca87dd9
# ╟─64c2750e-bb4b-41bb-96e8-bf2955943fbe
# ╟─daac6add-0657-4a3a-bcba-0d4377b62b44
# ╟─bfbae558-2cda-4224-ac88-45122cf751a5
# ╟─3114aa7d-f0c8-45cd-a7fa-b1cee9b9cd3b
# ╟─3e5b37f5-ebc8-4785-a8c2-7d1c258e38c6
# ╟─5810171a-0703-4589-9534-12f4fde3bdcb
# ╟─26e50e62-8e3b-48e4-8983-ded0c5b871ac
# ╟─4f1bd110-b5ad-4066-a8d7-9973c989bb20
# ╟─472e1f7c-d8c5-495f-954d-4ccf3adfc8bf
# ╟─781b579d-3fa5-45d8-b404-6c4bf55d3321
