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
@bind n_bootstraps Select([10, 50, 100, 500], default=100)

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

	ts = 0:0.01:20
	# ts = 10:0.5:1000.0
end;

# ╔═╡ 7f26750d-7a4f-4ffa-ad09-7efc3e653d82
md"""
Propagation time:
"""

# ╔═╡ 26c2242a-da97-45ff-b550-d1e69ca8b7e2
@bind t Slider(ts, default=0.2)

# ╔═╡ b6b0054b-37d8-458f-a05e-aed1589afa70
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

# ╔═╡ 6bceec02-f9b0-444e-99e8-29048bef9bef
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

# ╔═╡ 09541b38-3fc6-41eb-8a1e-bddc76934bed
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

# ╔═╡ e56442c7-96cd-46b5-835d-947cff2c778f
md"""
Correction factor:
"""

# ╔═╡ 602639dd-b690-4491-a10d-cf81934795ce
@bind correction_factor Slider(0:0.01:10, default=7.18)

# ╔═╡ 394a1c25-da60-41bd-a69f-f037f67f0304
begin
	threshold = correction_factor/n_samples^4
end;

# ╔═╡ 8e1c25b6-f611-4a67-90f9-15d4a146aa48
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

# ╔═╡ 00ee5a97-4e37-4a3d-8f31-9053a942f913
function last_local_maxima(fs::AbstractMatrix{Float64})
	maxima_indices = map(eachcol(fs)) do col
		inds = findall(i -> col[i] > col[i-1] && col[i] > col[i+1], 3:length(col)-1)
		isempty(inds) ? size(fs)[1] : last(inds)
	end

	return ts[maxima_indices]
end

# ╔═╡ 61d7a7cd-0602-4bb2-b18a-cb9627a75e25
maxs = last_local_maxima(vmr_vars_scaled);

# ╔═╡ 44fbd697-1541-4e80-b55f-28cf49ad192a
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

	# vline!(
	# 	p,
	# 	maxs[x_idx_range],
	# 	c=:indigo,
	# 	s=:dash,
	# 	alpha=0.5
	# )

end

# ╔═╡ 2307a92a-720e-409c-a207-75a284567eda
let
	x_idx_range = 1:20:2200
	subrange = 45:1:65
	p = plot(
		ts[begin+1:end],
		vmr_vars_scaled[begin+1:end, x_idx_range[subrange]],
		legend=false,
		ylimits=(1e-15,1e-5),
		yaxis=:log10,
		palette=palette(:vik, length(x_idx_range))[subrange],
		# xlimits=(step(ts), ts[end]),
		# xaxis=:log10,
		xlimits=(0, 0.5)
	)
	
	plot!(
		p,
		ts[begin+1:end],
		(0.0045/n_samples^3) .* ts[begin+1:end],
		c=:black,
		lw=2,
	)

	vline!(p, [t], c=:green, lw=2)
	hline!(p, [threshold], c=:red, lw=2)

	vline!(
		p,
		maxs[x_idx_range[subrange]],
		c=:indigo,
		s=:dash,
		alpha=0.5
	)

end

# ╔═╡ a5f7ad01-a7cd-42d7-a74b-5b1d6682e471
let
	x_idx_range = 1:100:2200
	cs = palette(:coolwarm, 4)
	
	p = plot(legend=false, xlimits=(0, 2))
	plot!(
		p,
		ts[begin+1:end],
		propagated_densities[begin+1:end, x_idx_range] .* n_samples,
		palette=palette(:vik, length(x_idx_range)),
		# alpha=0.2,
		# palette=palette(:blues, length(x_idx_range))
	)
	
	plot!(
		p,
		ts[begin+1:end],
		transpose(repeat(pdf_og[x_idx_range], 1, length(ts[begin+1:end]))),
		palette=palette(:vik, length(x_idx_range)),
		ls=:dash
		)
	
	vline!(p, [t], c=:green, lw=2)

	vline!(
		p,
		maxs[x_idx_range],
		c=:indigo,
		s=:dash,
		alpha=0.5
	)
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
# ╟─26c2242a-da97-45ff-b550-d1e69ca8b7e2
# ╟─b6b0054b-37d8-458f-a05e-aed1589afa70
# ╟─6bceec02-f9b0-444e-99e8-29048bef9bef
# ╟─09541b38-3fc6-41eb-8a1e-bddc76934bed
# ╟─e56442c7-96cd-46b5-835d-947cff2c778f
# ╟─602639dd-b690-4491-a10d-cf81934795ce
# ╟─394a1c25-da60-41bd-a69f-f037f67f0304
# ╟─8e1c25b6-f611-4a67-90f9-15d4a146aa48
# ╟─c4272f8b-bb05-4394-ab24-f21f2e57726c
# ╠═00ee5a97-4e37-4a3d-8f31-9053a942f913
# ╠═61d7a7cd-0602-4bb2-b18a-cb9627a75e25
# ╟─44fbd697-1541-4e80-b55f-28cf49ad192a
# ╠═2307a92a-720e-409c-a207-75a284567eda
# ╟─a5f7ad01-a7cd-42d7-a74b-5b1d6682e471
