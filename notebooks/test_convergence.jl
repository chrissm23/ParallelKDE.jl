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

# ╔═╡ 3524b6e7-c3f7-4a0c-85d9-25340ac8d5d7
function define_distro(::Val{:bimodal1})
	μ1 = 6
	σ1 = 1
	w1 = 0.4

	μ2 = 11
	σ2 = 2
	w2 = 1 - w1

	return MixtureModel(Normal, [(μ1, σ1), (μ2, σ2)], [w1, w2])
end;

# ╔═╡ c22aeb1d-85b8-43d7-b938-d6ee5cf9d3e3
function define_distro(::Val{:bimodal2})
	μ1 = 4
	σ1 = 0.3
	w1 = 0.3

	μ2 = 12
	σ2 = 1.5
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
@bind distro_name Select(
	[:normal, :bimodal1, :bimodal2, :noncentral_χ2], default=:bimodal2
)

# ╔═╡ fe3cd654-ed53-415d-9429-beea88bedace
md"Number of samples (`n_samples`)"

# ╔═╡ 45f698a7-e91f-48b8-af2d-d77fe5264367
@bind n_samples Select([100, 1000, 10000, 100000, 1000000], default=1000)

# ╔═╡ 2473dd08-bff6-4d61-9448-fb9336404afd
md"Number of grid points (`n_gridpoints`)"

# ╔═╡ c9911762-16c7-4490-aa0e-0e5a5ff62524
@bind n_gridpoints Slider(100:100:2000, default=1000)

# ╔═╡ 394ce4a6-f4da-464f-8fbb-e0c21af54a27
@bind resample Button("Resample")

# ╔═╡ 5bf23fbc-556a-4014-a84b-47c6dedf0b9f
begin
	resample
	
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

# ╔═╡ d904caba-b00e-449b-b0b9-c6132c3a0e6c
md"Number of test points (`n_testpoints`)."

# ╔═╡ d0084330-0998-455c-8f35-7c9818e46c86
@bind n_testpoints Slider(1:n_gridpoints÷50, default=7)

# ╔═╡ bf77fdfb-2c8a-4e25-8423-46d7ab1df8aa
md"Minimum value for test points (`testpoints_min`)."

# ╔═╡ 501df99f-299c-47cf-bc53-0230368663c0
@bind testpoints_min Slider(distro_support, default=minimum(distro_support))

# ╔═╡ e6f5d29a-c5d9-445c-926e-615fea53d495
md"maximum value for test points (`testpoints_max`)."

# ╔═╡ dcf117aa-6d79-4a46-963b-e2a9b5157707
@bind testpoints_max Slider(distro_support, default=maximum(distro_support))

# ╔═╡ 566ac44c-d485-4d9d-b78e-29a6f64794f7
md"
6.506506506506507\
13.633633633633634
"

# ╔═╡ 67a1126e-ca9a-4551-9255-ea75004ec2ff
begin
	println(testpoints_min)
	println(testpoints_max)
end

# ╔═╡ d048004b-ecbf-4455-944b-4a7ea105a50e
md"
Test points in `test_points`.\
Indices in `test_indices`.\
Color palette for those test points in `test_palette`.
"

# ╔═╡ 23b14634-08bf-4a96-8c5c-361392fa4ad2
begin
	testpoints_min_idx = findfirst(x -> (x - testpoints_min) >= 0, distro_support)
	testpoints_max_idx = findfirst(x -> (x - testpoints_max) >= 0, distro_support)
	test_indices = floor.(
		Int, range(testpoints_min_idx, testpoints_max_idx, length=n_testpoints)
	)
	test_points = distro_support[test_indices]
	test_palette = palette(:vik, n_testpoints)
end;

# ╔═╡ 1200e59e-388b-4c62-b2a8-084098311922
function calculate_mise(
	f1::AbstractArray{<:Real,N},
	f2::AbstractArray{<:Real,N},
	dx::Real
) where {N}
	n_points = length(f1)

	return sum((f1 .- f2) .^ 2) * dx / n_points
end

# ╔═╡ cc7df97b-0138-4289-843d-8b5046dea0d9
let
	data_filtered = data[
		(data .>= minimum(distro_support)) .& (data .<= maximum(distro_support))
	]
	data_filtered = reshape(data_filtered, 1, length(data_filtered))
	Random.seed!(random_seed)
	density_estimation = initialize_estimation(
		data_filtered,
		grid=true,
		grid_ranges=distro_support,
		# device=:cuda,
	)
	estimate_density!(
		density_estimation,
		:rotEstimator,
		rule_of_thumb=:silverman,
		# time_step=0.0005,
		# threshold_crossing_percentage=0.01,
		# eps=2.0,
		# alpha=0.75,
		# time_final=0.5,
		# n_bootstraps=1000,
	)

	density_estimated = get_density(density_estimation)
	# density_estimated_d = get_density(density_estimation)
	# density_estimated = Array(density_estimated_d)

	dx = prod(spacings(get_grid(density_estimation)))
	norm = sum(density_estimated) * dx
	density_estimated .= density_estimated / norm

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

	for (i, point) in enumerate(test_points)
		vline!(
			p_estimate,
			[point],
			label=false,
			c=test_palette[i],
		)
	end
	
	println("NaNs: ", findall(isnan, density_estimated))

	mise = calculate_mise(density_estimated, distro_pdf, dx)
	println("MISE: ", mise)

	p_estimate
end

# ╔═╡ bbbd2e78-3a28-4783-9601-c883cf99d185
let
	data_filtered = data[
		(data .>= minimum(distro_support)) .& (data .<= maximum(distro_support))
	]
	data_filtered = reshape(data_filtered, 1, length(data_filtered))
	Random.seed!(random_seed)
	density_estimation = initialize_estimation(
		data_filtered,
		grid=true,
		grid_ranges=distro_support,
		# device=:cuda,
	)
	estimate_density!(
		density_estimation,
		:parallelEstimator,
		# time_step=0.0005,
		# threshold_crossing_percentage=0.01,
		# eps=2.0,
		# alpha=0.75,
		# time_final=0.5,
		# n_bootstraps=1000,
	)

	density_estimated = get_density(density_estimation)
	# density_estimated_d = get_density(density_estimation)
	# density_estimated = Array(density_estimated_d)

	dx = prod(spacings(get_grid(density_estimation)))
	norm = sum(density_estimated) * dx
	density_estimated .= density_estimated / norm

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

	for (i, point) in enumerate(test_points)
		vline!(
			p_estimate,
			[point],
			label=false,
			c=test_palette[i],
		)
	end
	
	println("NaNs: ", findall(isnan, density_estimated))

	mise = calculate_mise(density_estimated, distro_pdf, dx)
	println("MISE: ", mise)

	p_estimate
end

# ╔═╡ 75e020f4-5682-46f3-8dff-7abeba257818
md"### Propagation behavior"

# ╔═╡ c7ae5c52-72ca-4449-bfa6-18fb36003ace
begin
	Random.seed!(random_seed)

	grid_support = initialize_grid(distro_support)
	time_initial = initial_bandwidth(grid_support)
	time_initial_squared = time_initial .^ 2

	method=:serial

	kde = initialize_kde(data, size(grid_support))
	
	parallel_estimator = ParallelKDE.DensityEstimators.initialize_estimator(
		ParallelKDE.DensityEstimators.AbstractParallelEstimator,
		kde;
		method,
		grid=grid_support,
		# time_step=0.00005,
		eps=2.0,
		# alpha=0.5,
		# stable_duration=0.001,
		# time_final=2.0
	)

	times = parallel_estimator.times
	n_times = length(times)
	times_reinterpreted = reinterpret(Float64, times)
	times_range = range(
		minimum(times_reinterpreted), maximum(times_reinterpreted), length=n_times
	)

	previous_density = fill(NaN, size(kde.density))

	derivatives1 = fill(NaN, n_gridpoints, n_times)
	derivatives2 = fill(NaN, n_gridpoints, n_times)

	vmr_time = fill(NaN, n_gridpoints, n_times)
	means_time = fill(NaN, n_gridpoints, n_times)
	density_assigned_time = falses(n_gridpoints, n_times)

	dlogts = fill(NaN, n_times)
	alpha_parm = parallel_estimator.density_state.alpha
	eps_parm = parallel_estimator.density_state.eps
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
		method
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
		method
	)
	vmr_time[:, idx] .= ParallelKDE.DensityEstimators.get_vmr(
		parallel_estimator.kernel_propagation
	)

	# Propagate means of full samples
	ParallelKDE.DensityEstimators.propagate_means!(
		parallel_estimator.kernel_propagation,
		parallel_estimator.means,
		parallel_estimator.grid_fourier,
		time_propagated;
		method
	)
	# Fourier transform back
	ParallelKDE.DensityEstimators.ifft_means!(
		parallel_estimator.kernel_propagation;
		method
	)
	ParallelKDE.DensityEstimators.calculate_means!(
		parallel_estimator.kernel_propagation,
		n_samples;
		method
	)
	means_time[:, idx] .= ParallelKDE.DensityEstimators.get_means(
		parallel_estimator.kernel_propagation
	)

	# Update convergence state
	if idx == 1
		det_prev = prod(time_initial_squared)
	else
		det_prev = prod(time_initial_squared .+ times[idx-1] .^ 2)
	end
	det_curr = prod(time_initial_squared .+ time_propagated .^ 2)
	dlogt = log(det_curr/det_prev)
	dlogts[idx] = dlogt
	ParallelKDE.DensityEstimators.update_state!(
		parallel_estimator.density_state,
		dlogt,
		kde,
		parallel_estimator.kernel_propagation;
		method
	)

	@. density_assigned_time[:, idx] = ifelse(
		(
			(kde.density ≈ previous_density) | 
			(isnan(kde.density) & isnan(previous_density))
		),
		false,
		true
	)
	previous_density .= kde.density
end

# ╔═╡ fec6f503-cd04-4a43-9b69-fdd1136ef981
sum(count(density_assigned_time, dims=1))

# ╔═╡ 5cca44db-4f10-4f9b-9c0a-6e6e1a983720
begin
	derivatives1[:, begin+1:end] .= alpha_parm .* log.(
		abs.(
			diff(vmr_time, dims=2) ./ reshape(2 .* dlogts[begin+1:end],1,:)
		)
	)
	derivatives2[:, begin+2:end] .= (1-alpha_parm) .* log.(
		abs.(
			diff(diff(vmr_time, dims=2), dims=2) ./ reshape(dlogts[begin+2:end] .^2 ,1,:)
		)
	)

	indicator = derivatives1 .+ derivatives2
	
	optimal_times = times_range[
		argmin.(eachrow(abs.(means_time .- reshape(distro_pdf, n_gridpoints, 1))))
	]
end;

# ╔═╡ 95746d99-132a-4f7c-8075-5da357841e1f
md"Propagation time (`propagation_time`)"

# ╔═╡ 33a0e17b-cf1e-4fb0-b9c4-1e2498f285a4
@bind propagation_time Slider(times_range, default=0)

# ╔═╡ badcc2b6-c438-47f0-b8b6-afc4e0cc741d
md"#### Propagation of mean Dirac sequences"

# ╔═╡ f5278add-a94e-4d43-a1be-4093f6bcefae
begin
	p_propagation = plot(
		distro_support, distro_pdf, label="PDF", lw=2, lc=:cornflowerblue, dpi=500
	)
	plot!(
		p_propagation,
		distro_support,
		means_time[:, findfirst(t -> t == propagation_time, times_range)],
		label="Propagated sequence\nt=$propagation_time",
		lw=2,
		lc=:forestgreen,
	)

	for (i, point) in enumerate(test_points)
		vline!(
			p_propagation,
			[point],
			label=false,
			c=test_palette[i],
		)
	end

	# savefig(p_propagation, "dirac_cpu.png")
	
	p_propagation
end

# ╔═╡ 9d33bbd6-2f00-4633-8500-ef1fd18dead9
md"#### Known optimal time propagation"

# ╔═╡ b62dc21c-c3a5-482e-9ad4-97f0e161d3dc
begin
	p_means = plot(
		times_range,
		eachrow(means_time)[test_indices],
		label=false,
		palette=test_palette,
		lw=2,
		ylims=1.1 .* extrema(means_time[test_indices, :]),
		xlims=extrema(times_range),
	)

	vline!(p_means, [propagation_time], c=:forestgreen, label="Propagation time")

	for (i, val) in enumerate(distro_pdf[test_indices])
		hline!(
			p_means,
			[val],
			label=false,
			lc=test_palette[i],
			lw=2,
			ls=:dash,
		)

		vline!(
			p_means,
			[optimal_times[test_indices[i]]],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dashdot,
		)
	end

	# Manual label
	plot!(
		p_means,
		[-10; -20], lw=2, lc=:black, ls=:solid, label="Propagated means"
	)
	plot!(
		p_means,
		[-10; -20], lw=2, lc=:black, ls=:dash, label="PDF"
	)
	plot!(
		p_means,
		[-10; -20], lw=2, lc=:black, ls=:dashdot, label="Optimal time"
	)
end

# ╔═╡ d132f438-3764-4844-9634-7416f2e062b7
vmr_bounds = (0, 45);

# ╔═╡ 7f9cf2ba-2085-4b80-a98a-9feb7fe0c4de
md"#### VMR"

# ╔═╡ 097c40b2-2b34-442d-8cb9-4c63b60abfc4
begin
	p_decrease = plot(
		times_range,
		eachrow(vmr_time)[test_indices],
		label=false,
		palette=test_palette,
		lw=2,
		ylims=vmr_bounds,
		xlims=extrema(times_range),
		# yaxis=:log
	)

	vline!(p_decrease, [propagation_time], c=:forestgreen, label="Propagation time")

	for (i, test_idx) in enumerate(test_indices)
		vline!(
			p_decrease,
			[
				let
					try
						times_range[findfirst(has_decreased_time[test_idx, :])]
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

# ╔═╡ c20c3541-d935-423c-a07f-77d8e0679d8d
begin
	p_stability = plot(
		times_range[begin+1:end],
		eachrow(vmr_time[:, begin+1:end])[test_indices],
		label=false,
		palette=test_palette,
		lw=2,
		ylims=vmr_bounds,
		xlims=extrema(times_range[begin+1:end]),
		# yaxis=:log,
		xaxis=:log
	)

	vline!(p_stability, [propagation_time], c=:forestgreen, label="Propagation time")

	for (i, test_idx) in enumerate(test_indices)
		vline!(
			p_stability,
			[
				let
					try
						times_range[findfirst(is_stable_time[test_idx, :])]
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

# ╔═╡ c6e708d8-4ec9-4a83-8e36-ea25ea9dca8a
md"#### First derivative"

# ╔═╡ 36ff4668-62aa-4cb1-b732-2bc278d723e2
begin
	p_dev1 = plot(
		times_range[begin+1:end],
		eachrow(derivatives1[:,begin+1:end])[test_indices],
		label=false,
		palette=test_palette,
		lw=2,
		ylims=(-10,5),
		xlims=extrema(times_range[begin+1:end]),
		# xlims=(0,0.25),
		# yaxis=:log,
		# xaxis=:log
	)

	vline!(p_dev1, [propagation_time], c=:forestgreen, label="Propagation time")

	for (i, val) in enumerate(distro_pdf[test_indices])
		vline!(
			p_dev1,
			[
				let
					try
						times_range[
							findlast(density_assigned_time[test_indices[i], :])
						]
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

# ╔═╡ 239d91b7-1a08-48aa-9e86-93e4c006bc06
md"#### Second derivative"

# ╔═╡ c2d6f057-8bd6-4b0c-9e24-83b9c8d0ee52
begin
	p_dev2 = plot(
		times_range[begin+1:end],
		eachrow(derivatives2[:, begin+1:end])[test_indices],
		label=false,
		palette=test_palette,
		lw=2,
		ylims=(-10,5),
		xlims=extrema(times_range[begin+1:end]),
		# xlims=(0,0.25),
		# yaxis=:log,
		# xaxis=:log,
		dpi=500,
	)

	vline!(p_dev2, [propagation_time], c=:forestgreen, label="Propagation time")

	for (i, val) in enumerate(distro_pdf[test_indices])
		vline!(
			p_dev2,
			[
				let
					try
						times_range[
							findlast(density_assigned_time[test_indices[i], :])
						]
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

# ╔═╡ fcb3dd19-8c34-4870-9cdb-05aa140da88a
md"#### Combined derivative indicator"

# ╔═╡ adcc7ec7-f8cb-4c62-a6c8-10b2396c3fb3
begin
	threshold_range = range(-5, 5, length=100)
	@bind threshold Slider(
		threshold_range,
		default=threshold_range[argmin(abs.(threshold_range .- eps_parm))]
	)
end

# ╔═╡ 5359ee63-8b88-4215-b216-58dd5af0c2b0
begin
	p_dev12 = plot(
		times_range[begin+1:end],
		eachrow(indicator[:, begin+1:end])[test_indices],
		label=false,
		palette=test_palette,
		lw=2,
		ylims=(-10,5),
		xlims=extrema(times_range[begin+1:end]),
		# xlims=(0,0.25),
		# yaxis=:log,
		# xaxis=:log,
		dpi=500,
	)

	hline!(p_dev12, [threshold], c=:red, label="Threshold: $threshold")
	vline!(p_dev12, [propagation_time], c=:forestgreen, label="Propagation time")

	
	for (i, val) in enumerate(distro_pdf[test_indices])
		vline!(
			p_dev12,
			[
				let
					try
						times_range[
							findlast(density_assigned_time[test_indices[i], :])
						]
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

	p_dev12
end

# ╔═╡ 8bd6926d-dbec-4858-9c72-5e0904bc71d7
md"#### Final stopping point"

# ╔═╡ 123376cd-d638-41e9-a6bf-8ff9fcbe4b6b
begin
	p_optimal = plot(
		times_range[begin+1:end],
		eachrow(vmr_time[:, begin+1:end])[test_indices],
		label=false,
		palette=test_palette,
		lw=2,
		ylims=vmr_bounds,
		xlims=extrema(times_range[begin+1:end]),
		# yaxis=:log,
		xaxis=:log
	)

	vline!(p_optimal, [propagation_time], c=:forestgreen, label="Propagation time")

	for (i, val) in enumerate(distro_pdf[test_indices])
		vline!(
			p_optimal,
			[
				times_range[
					findlast(density_assigned_time[test_indices[i], :])
				]
			],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dash,
		)
		
		vline!(
			p_optimal,
			[optimal_times[test_indices[i]]],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dashdot,
		)
	end

	# Manual label
	plot!(
		p_optimal, [-10; -20], lw=2, lc=:black, ls=:solid, label="Scaled VMR"
	)
	plot!(
		p_optimal, [-10; -20], lw=2, lc=:black, ls=:dash, label="Stopping point"
	)
	plot!(
		p_optimal,
		[-10; -20], lw=1, lc=:black, ls=:dashdot, label="Optimal time"
	)

	# savefig(p_optimal, "vmr_cpu.png")

	p_optimal
end

# ╔═╡ 9f5e3a30-0104-45f8-b3f5-eaa4b73dce32
md"#### Error comparison"

# ╔═╡ c85f42eb-6667-4bcf-8888-db105378cb55
begin
	p_means2 = plot(
		times_range,
		eachrow(means_time)[test_indices],
		label=false,
		palette=test_palette,
		lw=2,
		ylims=1.1 .* extrema(means_time[test_indices, :]),
		xlims=extrema(times_range),
	)

	vline!(p_means2, [propagation_time], c=:forestgreen, label="Propagation time")

	for (i, val) in enumerate(distro_pdf[test_indices])
		hline!(
			p_means2,
			[val],
			label=false,
			lc=test_palette[i],
			lw=2,
			ls=:dashdotdot,
		)

		vline!(
			p_means2,
			[
				times_range[
					findlast(density_assigned_time[test_indices[i], :])
				]
			],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dash,
		)

		vline!(
			p_means2,
			[optimal_times[test_indices[i]]],
			label=false,
			lw=2,
			lc=test_palette[i],
			ls=:dashdot,
		)
	end

	# Manual label
	plot!(
		p_means2,
		[-10; -20], lw=2, lc=:black, ls=:solid, label="Propagated means"
	)
	plot!(
		p_means2,
		[-10; -20], lw=2, lc=:black, ls=:dashdotdot, label="PDF"
	)
	plot!(
		p_means2,
		[-10; -20], lw=2, lc=:black, ls=:dashdot, label="Optimal time"
	)
	plot!(
		p_means2,
		[-10; -20], lw=2, lc=:black, ls=:dash, label="Stopping time"
	)
end

# ╔═╡ Cell order:
# ╠═3add1086-3d31-11f0-0a9e-cd0909baedac
# ╟─c823e934-2430-4c95-ae36-96879b565e47
# ╟─dd1af6a8-61a4-45dd-8b99-04a133175f04
# ╟─534f3f9e-bce8-47d4-a9a7-303e21cfb0ef
# ╟─24267c16-0b02-445d-8d96-4aecd17bb52e
# ╟─ab810aa5-916c-4491-a346-ba098b052e15
# ╟─3524b6e7-c3f7-4a0c-85d9-25340ac8d5d7
# ╟─c22aeb1d-85b8-43d7-b938-d6ee5cf9d3e3
# ╟─7f588fa0-525d-471a-9b3d-af73f4e083f5
# ╟─7e79580a-4b79-48b6-984e-1b9cb351611a
# ╟─ba9f88bc-a309-4ed7-a08b-822ed48f19c1
# ╟─fe3cd654-ed53-415d-9429-beea88bedace
# ╟─45f698a7-e91f-48b8-af2d-d77fe5264367
# ╟─2473dd08-bff6-4d61-9448-fb9336404afd
# ╟─c9911762-16c7-4490-aa0e-0e5a5ff62524
# ╟─394ce4a6-f4da-464f-8fbb-e0c21af54a27
# ╟─5bf23fbc-556a-4014-a84b-47c6dedf0b9f
# ╟─b2645b3b-2251-4d6a-9a41-e69eb101e12b
# ╟─ca25a9af-0783-492c-8362-2c163c4ec5f1
# ╟─87b3bd79-f44a-41a2-9aac-201d6bb3f00a
# ╟─3e7d87af-6523-4218-86ee-e2af901bf491
# ╟─d904caba-b00e-449b-b0b9-c6132c3a0e6c
# ╟─d0084330-0998-455c-8f35-7c9818e46c86
# ╟─bf77fdfb-2c8a-4e25-8423-46d7ab1df8aa
# ╟─501df99f-299c-47cf-bc53-0230368663c0
# ╟─e6f5d29a-c5d9-445c-926e-615fea53d495
# ╟─dcf117aa-6d79-4a46-963b-e2a9b5157707
# ╟─566ac44c-d485-4d9d-b78e-29a6f64794f7
# ╠═67a1126e-ca9a-4551-9255-ea75004ec2ff
# ╟─d048004b-ecbf-4455-944b-4a7ea105a50e
# ╟─23b14634-08bf-4a96-8c5c-361392fa4ad2
# ╟─1200e59e-388b-4c62-b2a8-084098311922
# ╠═cc7df97b-0138-4289-843d-8b5046dea0d9
# ╠═bbbd2e78-3a28-4783-9601-c883cf99d185
# ╟─75e020f4-5682-46f3-8dff-7abeba257818
# ╠═c7ae5c52-72ca-4449-bfa6-18fb36003ace
# ╠═fb725626-e5eb-4dfc-93e7-4946af668652
# ╠═fec6f503-cd04-4a43-9b69-fdd1136ef981
# ╠═5cca44db-4f10-4f9b-9c0a-6e6e1a983720
# ╟─95746d99-132a-4f7c-8075-5da357841e1f
# ╟─33a0e17b-cf1e-4fb0-b9c4-1e2498f285a4
# ╟─badcc2b6-c438-47f0-b8b6-afc4e0cc741d
# ╟─f5278add-a94e-4d43-a1be-4093f6bcefae
# ╟─9d33bbd6-2f00-4633-8500-ef1fd18dead9
# ╟─b62dc21c-c3a5-482e-9ad4-97f0e161d3dc
# ╠═d132f438-3764-4844-9634-7416f2e062b7
# ╟─7f9cf2ba-2085-4b80-a98a-9feb7fe0c4de
# ╟─097c40b2-2b34-442d-8cb9-4c63b60abfc4
# ╟─c20c3541-d935-423c-a07f-77d8e0679d8d
# ╟─c6e708d8-4ec9-4a83-8e36-ea25ea9dca8a
# ╟─36ff4668-62aa-4cb1-b732-2bc278d723e2
# ╟─239d91b7-1a08-48aa-9e86-93e4c006bc06
# ╟─c2d6f057-8bd6-4b0c-9e24-83b9c8d0ee52
# ╟─fcb3dd19-8c34-4870-9cdb-05aa140da88a
# ╟─adcc7ec7-f8cb-4c62-a6c8-10b2396c3fb3
# ╟─5359ee63-8b88-4215-b216-58dd5af0c2b0
# ╟─8bd6926d-dbec-4858-9c72-5e0904bc71d7
# ╟─123376cd-d638-41e9-a6bf-8ff9fcbe4b6b
# ╟─9f5e3a30-0104-45f8-b3f5-eaa4b73dce32
# ╟─c85f42eb-6667-4bcf-8888-db105378cb55
