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

# ╔═╡ a78042b1-37d7-46be-bb9b-3417516f97bf
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

	include("../test/test_utils.jl")
	
	pythonplot()
end;

# ╔═╡ eb56f071-20fc-4414-b6a7-46de9fc92d3a
md"""
## Distribution definitions
"""

# ╔═╡ 15cd0c5d-0862-4469-93fd-afee4fce91aa
begin
	unimodal = Normal(10, 1)
	bimodal = MixtureModel([Normal(7, 1.5), Normal(13, 1.5)], [0.6, 0.4])
	chisq = Chisq(1)
end;

# ╔═╡ e52bec6f-0222-41dd-b502-328b78cd93c7
md"""
## PDF and grid features
"""

# ╔═╡ 6bb28d06-d659-4ea0-8711-c49dea4e973a
begin
	grid_range = -1.0:0.01:21.0
	
	test_grid = ParallelKDE.Grids.Grid([grid_range])
	test_grid_array = ParallelKDE.Grids.get_coordinates(test_grid)

	t0 = ParallelKDE.Grids.initial_bandwidth(test_grid)
	
	fourier_grid = ParallelKDE.Grids.fftgrid(test_grid)
	fourier_grid_array = ParallelKDE.Grids.get_coordinates(fourier_grid)
end;

# ╔═╡ 5afc8bd1-248d-49b4-9384-58ece9ddd367
md"""
Distribution
"""

# ╔═╡ 873a82cd-b2e0-46dd-96d9-eff3d0d8d4e7
@bind distro_str Select(["unimodal", "bimodal", "chisq"])

# ╔═╡ ccc2061c-a357-43b3-be48-bf0f1b7b214e
distro_dict = Dict(
	"unimodal" => unimodal,
	"bimodal" => bimodal,
	"chisq" => chisq,
);

# ╔═╡ 453b2cf5-e612-48c1-8200-5c9d6ef8fe27
begin
	distro = distro_dict[distro_str]
	pdf_og = pdf.(Ref(distro), grid_range)
end;

# ╔═╡ 8b799664-b2af-471a-8fc2-1c9424c5a538
begin
	plot(
		grid_range,
		pdf_og,
		label="PDF",
		line_z=range(0, stop=1, length= length(grid_range)),
		c=:vik,
		colorbar=false,
	)
end

# ╔═╡ f3c53e7d-43a7-4127-9558-75042c73b115
md"""
# Test ParallelKDE workflow
"""

# ╔═╡ e951cdf9-dfa7-49ef-aa54-b301cd4fe266
md"""
### Test settings
"""

# ╔═╡ 72b7d720-0635-49bc-a030-f8ba39b18cbc
md"""
Number of samples:
"""

# ╔═╡ 7509f985-af79-48ce-b825-8e4d5ef6241c
@bind n_samples Select([10, 100, 1000, 10000], default=1000)

# ╔═╡ 120c2cb1-419d-42b5-a32a-8c67feff07d9
md"""
Number of bootstraps:
"""

# ╔═╡ 398c15de-3d96-40e1-93a7-08a0b15c1d1e
@bind n_bootstraps Select([10, 50, 100, 500, 1000, 2000], default=100)

# ╔═╡ d6533d34-beac-4b73-bccc-dc1dca0efadf
md"""
Random seed:
"""

# ╔═╡ 37c3f751-063d-45c1-bbb9-6a1f76ffcf92
@bind seed NumberField(1:9999, default=5000)

# ╔═╡ 8612d607-e3c3-4ee4-b52b-db5ca23d48a5
md"""
## Initializing data and pre-allocating arrays
"""

# ╔═╡ b242b254-373b-4126-a36d-e7481bed6418
data = SVector{1, Float64}.(rand(distro, n_samples));

# ╔═╡ 873771d4-88f9-4c1b-83b6-01a5dbd62c2b
begin
	kde = initialize_kde(data, [grid_range], :cpu)
	
	means_0, variances_0 = ParallelKDE.initialize_statistics(
		kde, n_bootstraps, :serial
	)
	density_0, var_0 = ParallelKDE.initialize_distribution(kde, :serial)

	means_dst = Array{ComplexF64, 2}(undef, size(means_0))
	variances_dst = Array{ComplexF64, 2}(undef, size(variances_0))

	ts = 0:0.01:2
	# ts = 10:0.5:1000.0
end;

# ╔═╡ 7f26750d-7a4f-4ffa-ad09-7efc3e653d82
md"""
Propagation time:
"""

# ╔═╡ 26c2242a-da97-45ff-b550-d1e69ca8b7e2
@bind t Slider(ts, default=0.2)

# ╔═╡ b6b0054b-37d8-458f-a05e-aed1589afa70
# ╠═╡ disabled = true
#=╠═╡
begin
	t_vector = fill(t, 1)
	
	means_fourier, variances_fourier = ParallelKDE.propagate_bandwidth!(
		means_0,
		variances_0,
		fourier_grid_array,
		t_vector,
		t0,
		:serial,
		dst_mean=means_dst,
		dst_var=variances_dst
	)

	means_direct, variances_direct = ParallelKDE.FourierSpace.ifft_statistics!(
		means_fourier,
		variances_fourier,
		n_samples,
		bootstraps_dim=true
	)

	vmr_variance = ParallelKDE.calculate_statistics!(
		means_direct,
		variances_direct,
		:serial,
		dst_vmr=selectdim(
			selectdim(reinterpret(reshape, Float64, variances_dst), 1, 1),
			1,
			1
		)
	)

	mean_fourier, variance_fourier = ParallelKDE.propagate_bandwidth!(
		reshape(density_0, 1, size(density_0)...),
		reshape(var_0, 1, size(var_0)...),
		fourier_grid_array,
		t_vector,
		t0,
		:serial,
		dst_mean=reshape(selectdim(means_dst, 1, 1), 1, size(means_dst)[2:end]...),
		dst_var=reshape(selectdim(means_dst, 1, 2), 1, size(variances_dst)[2:end]...)
	)

	mean_fourier = dropdims(mean_fourier, dims=1)
	variance_fourier = dropdims(variance_fourier, dims=1)

	mean_direct, variance_direct = ParallelKDE.FourierSpace.ifft_statistics!(
		mean_fourier,
		variance_fourier,
		n_samples,
		bootstraps_dim=false
	)

	vmr_var_scaled = ParallelKDE.DirectSpace.calculate_variance_products!(
		Val(:serial),
		vmr_variance,
		t_vector,
		t0,
		dst_var_products=vmr_variance
	)
end;
  ╠═╡ =#

# ╔═╡ 6bceec02-f9b0-444e-99e8-29048bef9bef
# ╠═╡ disabled = true
#=╠═╡
let
	density_calc = mean_direct .* n_samples
	p = plot(
		grid_range,
		density_calc,
		# line_z=range(0, stop=1, length=length(grid_range)),
		c=:blue,
		ylimits=(0,0.5),
		label="Density from propagated mean",
		colorbar=false,
	)
	plot!(p, grid_range, pdf_og, label="PDF", c="firebrick")
end
  ╠═╡ =#

# ╔═╡ 09541b38-3fc6-41eb-8a1e-bddc76934bed
# ╠═╡ disabled = true
#=╠═╡
let
	density_calc = (2√π * Float64(n_samples)^2 * t) .* variance_direct
	p = plot(
		grid_range,
		density_calc,
		# line_z=range(0, stop=1, length=length(grid_range)),
		c=:blue,
		ylimits=(0,0.5),
		label="Density from propagated variance",
		colorbar=false
	)
	plot!(p, grid_range, pdf_og, label="PDF", c="firebrick")
end
  ╠═╡ =#

# ╔═╡ e56442c7-96cd-46b5-835d-947cff2c778f
md"""
Correction factor:
"""

# ╔═╡ 602639dd-b690-4491-a10d-cf81934795ce
@bind correction_factor Slider(0:0.01:10, default=7.18)

# ╔═╡ 394a1c25-da60-41bd-a69f-f037f67f0304
# ╠═╡ disabled = true
#=╠═╡
begin
	threshold = correction_factor/n_samples^4
end;
  ╠═╡ =#

# ╔═╡ 8e1c25b6-f611-4a67-90f9-15d4a146aa48
# ╠═╡ disabled = true
#=╠═╡
let
	p = plot(
		grid_range,
		vmr_var_scaled,
		# line_z=range(0, stop=1, length=length(grid_range)),
		c=:blue,
		yaxis=:log10,
		ylimits=(1e-15, 1e-5),
		label="Var[VMR]|T|^(3/2)",
		colorbar=false,
	)

	hline!(p, [threshold], c=:red, lw=2)
end
  ╠═╡ =#

# ╔═╡ c4272f8b-bb05-4394-ab24-f21f2e57726c
begin 
	vmr_vars_scaled = fill(NaN, length(ts), length(grid_range))
	propagated_densities = fill(NaN, length(ts), length(grid_range))

	means_dst2 = Array{ComplexF64, 2}(undef, size(means_0))
	variances_dst2 = Array{ComplexF64, 2}(undef, size(variances_0))

	for tf in ts
		tf_vector = fill(tf, 1)
	
		means_fourier2, variances_fourier2 = ParallelKDE.propagate_bandwidth!(
			means_0,
			variances_0,
			fourier_grid_array,
			tf_vector,
			t0,
			:serial,
			dst_mean=means_dst2,
			dst_var=variances_dst2
		)
	
		means_direct2, variances_direct2 = ParallelKDE.FourierSpace.ifft_statistics!(
			means_fourier2,
			variances_fourier2,
			n_samples,
			bootstraps_dim=true
		)
	
		vmr_variance2 = ParallelKDE.calculate_statistics!(
			means_direct2,
			variances_direct2,
			:serial,
			dst_vmr=selectdim(
				selectdim(reinterpret(reshape, Float64, variances_dst2), 1, 1),
				1,
				1
			)
		)
	
		mean_fourier2, variance_fourier2 = ParallelKDE.propagate_bandwidth!(
			reshape(density_0, 1, size(density_0)...),
			reshape(var_0, 1, size(var_0)...),
			fourier_grid_array,
			tf_vector,
			t0,
			:serial,
			dst_mean=reshape(selectdim(means_dst2, 1, 1), 1, size(means_dst2)[2:end]...),
			dst_var=reshape(selectdim(means_dst2, 1, 2), 1, size(variances_dst2)[2:end]...)
		)
	
		mean_fourier2 = dropdims(mean_fourier2, dims=1)
		variance_fourier2 = dropdims(variance_fourier2, dims=1)
	
		mean_direct2, variance_direct2 = ParallelKDE.FourierSpace.ifft_statistics!(
			mean_fourier2,
			variance_fourier2,
			n_samples,
			bootstraps_dim=false
		)
	
		vmr_var_scaled2 = ParallelKDE.DirectSpace.calculate_variance_products!(
			Val(:serial),
			vmr_variance2,
			tf_vector,
			t0,
			dst_var_products=vmr_variance2
		)

		vmr_vars_scaled[round(Int64, (tf-ts[begin])/step(ts))+1, :] .= vmr_var_scaled2
		propagated_densities[round(Int64, (tf-ts[begin])/step(ts))+1, :] .= mean_direct2
	end
end

# ╔═╡ 90836f97-6246-42ab-893f-cecc910ffc67
function silverman(data::Vector{SVector{1,Float64}})::Float64
	interquartile = iqr(first.(data))
	stdeviation = std(first.(data))

	return 0.9 * min(stdeviation, interquartile/1.34) * length(data)^(-1/5)
end

# ╔═╡ 44fbd697-1541-4e80-b55f-28cf49ad192a
# ╠═╡ disabled = true
#=╠═╡
let
	x_idx_range = 1:20:2200
	p = plot(
		ts[begin+1:end],
		vmr_vars_scaled[begin+1:end, x_idx_range],
		legend=false,
		ylimits=(1e-15,1e-5),
		yaxis=:log10,
		palette=palette(:vik, length(x_idx_range)),
		xaxis=:log10,
		xlimits=(step(ts), ts[end]),
		# xlimits=(0, 1.5)
	)
	
	plot!(
		p,
		ts[begin+1:end],
		(0.006/n_samples^3) .* ts[begin+1:end],
		c=:black,
		lw=2,
	)
	
	vline!(p, [t], c=:forestgreen, lw=2)
	hline!(p, [threshold], c=:red, lw=2)

	vline!(p, [silverman(data)], c=:green, s=:dash, lw=2)
	
	# vline!(
	# 	p,
	# 	maxs[x_idx_range],
	# 	c=:indigo,
	# 	s=:dash,
	# 	alpha=0.5
	# )

end
  ╠═╡ =#

# ╔═╡ 2307a92a-720e-409c-a207-75a284567eda
# ╠═╡ disabled = true
#=╠═╡
let
	x_idx_range = 1:20:2200
	subrange1 = 50:2:60
	p = plot(
		ts[begin+1:end],
		vmr_vars_scaled[begin+1:end, x_idx_range[subrange1]],
		legend=false,
		ylimits=(5e-15,5e-14),
		yaxis=:log10,
		c=palette(:vik, 3)[2],
		# xlimits=(step(ts), ts[end]),
		# xaxis=:log10,
		xlimits=(0, 2)
	)

	vline!(p, [t], c=:green, lw=2)
	hline!(p, [threshold], c=:red, lw=2)

	vline!(p, [silverman(data)], c=:green, s=:dash, lw=2)

end
  ╠═╡ =#

# ╔═╡ 64cdb3ba-9284-402f-8396-34b4d609b966
begin
	matching_times = argmin(
		abs.((propagated_densities .* n_samples) .- reshape(pdf_og, 1, size(pdf_og)...)),
		dims=1
	)
	matching_times = getindex.(reshape(matching_times, length(grid_range)), 1)
end;

# ╔═╡ 043a56f8-9562-41cc-ac65-febb97db29af
function find_stability(
	signal::Array{Float64,2},
	time_step::Real;
	threshold1::Real,
	threshold2::Real,
)
	n_times, n_samples = size(signal)
	
	single_diff = diff(signal, dims=1)
	double_diff = diff(single_diff, dims=1)

	first_derivative = abs.(single_diff ./ (2 * time_step))
	second_derivative = abs.(double_diff ./ (time_step^2))

	stopping_condition = @. (
		(first_derivative[2:end, :] < threshold1) & (second_derivative < threshold2)
	)

	idxs = findfirst.(eachcol(stopping_condition))

	return ifelse.(
		idxs .=== nothing,
		-1,
		idxs
	) .+ 2
end

# ╔═╡ 16987b86-9c64-47c3-8e70-fc9d884f15d4
stopping_times = find_stability(
	log10.(vmr_vars_scaled .* n_samples^4),
	step(ts),
	threshold1=1,
	threshold2=10,
);

# ╔═╡ 92da7406-dbd6-48b8-858e-bee7e5c001b8
points = 750:100:1150

# ╔═╡ ef850305-9e1a-446e-b725-e8adf8ca996d
let
	p = plot(legend=true)

	optimal_density = similar(matching_times, eltype(propagated_densities))
	@inbounds for i in eachindex(matching_times)
		optimal_density[i] = propagated_densities[matching_times[i], i] * n_samples
	end

	stopped_density = similar(stopping_times, eltype(propagated_densities))
	@inbounds for i in eachindex(matching_times)
		stopped_density[i] = propagated_densities[stopping_times[i], i] * n_samples
	end

	plot!(
		p,
		grid_range,
		pdf_og,
		s=:solid,
		lw=2,
		alpha=0.5,
		label="true density"
	)
	
	plot!(
		p,
		grid_range,
		optimal_density,
		s=:solid,
		lw=2,
		alpha=0.5,
		label="closest propagated density"
	)

	plot!(
		p,
		grid_range,
		stopped_density,
		s=:solid,
		lw=2,
		label="stopped propagation"
	)

	for (idx, point) in enumerate(points)
		vline!(
			p,
			[grid_range[point]],
			c=palette(:vik, length(points))[idx],
			s=:dash,
			label=false
		)
	end

	plot(p)
end

# ╔═╡ c92a645a-9378-4fbc-8c7e-3c15495c88ae
let
	p = plot(
		ts,
		log10.(vmr_vars_scaled[:, points] .* n_samples^4),
		legend=false,
		ylimits=(-1, 5),
		# yaxis=:log10,
		palette=palette(:vik, length(points)),
		# xlimits=(step(ts), ts[end]),
		# xaxis=:log10,
		xlimits=(0, 2)
	)

	for (idx, point) in enumerate(points)
		vline!(
			p,
			[ts[matching_times[point]]],
			c=palette(:vik, length(points))[idx],
			ls=:dot,
			lw=2
		)

		vline!(
			p,
			[ts[stopping_times[point]]],
			c=palette(:vik, length(points))[idx],
			ls=:dash,
			lw=1.2,
		)
	end

	# vline!(p, [silverman(data)], c=:green, s=:dash, lw=2, alpha=0.2)
	plot(p)

end

# ╔═╡ 5f90e737-c69f-4791-a7a0-55b64953d536
size(transpose(repeat(pdf_og[points], 1, 201)))

# ╔═╡ 5af82e9a-f62f-4284-9926-32909d2300ee
begin
	optimal_density = similar(matching_times, eltype(propagated_densities))
	@inbounds for i in eachindex(matching_times)
		optimal_density[i] = propagated_densities[matching_times[i], i] * n_samples
	end
end

# ╔═╡ a5f7ad01-a7cd-42d7-a74b-5b1d6682e471
let
	p = plot(legend=false, xlimits=(0, 2), ylimits=(0,4))
	
	plot!(
		p,
		ts,
		propagated_densities[:, points] .* n_samples,
		palette=palette(:vik, length(points)),
		# alpha=0.2,
		# palette=palette(:blues, length(x_idx_range))
	)
	
	for (idx, point) in enumerate(points)
		hline!(
			p,
			[pdf_og[point]],
			c=palette(:vik, length(points))[idx],
			ls=:dash
		)

		vline!(
			p,
			[ts[matching_times[point]]],
			c=palette(:vik, length(points))[idx],
			ls=:dot,
			lw=2
		)

		vline!(
			p,
			[ts[stopping_times[point]]],
			c=palette(:vik, length(points))[idx],
			ls=:dash,
			lw=1.2,
		)
	end
	
	# vline!(p, [t], c=:green, lw=2)
	plot(p)
end

# ╔═╡ Cell order:
# ╠═a78042b1-37d7-46be-bb9b-3417516f97bf
# ╟─eb56f071-20fc-4414-b6a7-46de9fc92d3a
# ╠═15cd0c5d-0862-4469-93fd-afee4fce91aa
# ╟─e52bec6f-0222-41dd-b502-328b78cd93c7
# ╠═6bb28d06-d659-4ea0-8711-c49dea4e973a
# ╟─5afc8bd1-248d-49b4-9384-58ece9ddd367
# ╟─873a82cd-b2e0-46dd-96d9-eff3d0d8d4e7
# ╠═ccc2061c-a357-43b3-be48-bf0f1b7b214e
# ╠═453b2cf5-e612-48c1-8200-5c9d6ef8fe27
# ╟─8b799664-b2af-471a-8fc2-1c9424c5a538
# ╟─f3c53e7d-43a7-4127-9558-75042c73b115
# ╟─e951cdf9-dfa7-49ef-aa54-b301cd4fe266
# ╟─72b7d720-0635-49bc-a030-f8ba39b18cbc
# ╟─7509f985-af79-48ce-b825-8e4d5ef6241c
# ╟─120c2cb1-419d-42b5-a32a-8c67feff07d9
# ╟─398c15de-3d96-40e1-93a7-08a0b15c1d1e
# ╟─d6533d34-beac-4b73-bccc-dc1dca0efadf
# ╟─37c3f751-063d-45c1-bbb9-6a1f76ffcf92
# ╟─8612d607-e3c3-4ee4-b52b-db5ca23d48a5
# ╠═b242b254-373b-4126-a36d-e7481bed6418
# ╠═873771d4-88f9-4c1b-83b6-01a5dbd62c2b
# ╟─7f26750d-7a4f-4ffa-ad09-7efc3e653d82
# ╠═26c2242a-da97-45ff-b550-d1e69ca8b7e2
# ╟─b6b0054b-37d8-458f-a05e-aed1589afa70
# ╟─6bceec02-f9b0-444e-99e8-29048bef9bef
# ╟─09541b38-3fc6-41eb-8a1e-bddc76934bed
# ╟─e56442c7-96cd-46b5-835d-947cff2c778f
# ╟─602639dd-b690-4491-a10d-cf81934795ce
# ╟─394a1c25-da60-41bd-a69f-f037f67f0304
# ╟─8e1c25b6-f611-4a67-90f9-15d4a146aa48
# ╟─c4272f8b-bb05-4394-ab24-f21f2e57726c
# ╠═90836f97-6246-42ab-893f-cecc910ffc67
# ╟─44fbd697-1541-4e80-b55f-28cf49ad192a
# ╟─2307a92a-720e-409c-a207-75a284567eda
# ╠═64cdb3ba-9284-402f-8396-34b4d609b966
# ╠═043a56f8-9562-41cc-ac65-febb97db29af
# ╠═16987b86-9c64-47c3-8e70-fc9d884f15d4
# ╠═ef850305-9e1a-446e-b725-e8adf8ca996d
# ╠═92da7406-dbd6-48b8-858e-bee7e5c001b8
# ╠═c92a645a-9378-4fbc-8c7e-3c15495c88ae
# ╠═5f90e737-c69f-4791-a7a0-55b64953d536
# ╟─5af82e9a-f62f-4284-9926-32909d2300ee
# ╠═a5f7ad01-a7cd-42d7-a74b-5b1d6682e471
