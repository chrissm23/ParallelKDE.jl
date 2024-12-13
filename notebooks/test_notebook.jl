### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 6ca5c154-bf2b-4097-9b3a-2e1df93d043a
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using FFTW,
		StaticArrays,
		CUDA
end

# ╔═╡ 0852562f-d8df-42f7-8189-c6119eae7a86
abstract type AbstractGrid{N,T<:Real,M} end

# ╔═╡ 21612857-e4e5-46b7-915d-5f41218f84be
begin
	struct Grid{N,T<:Real,M} <: AbstractGrid{N,T,M}
		coordinates::SVector{N,StepRangeLen}
		spacings::SVector{N,T}
		bounds::SMatrix{N,2,T}
	end
	function Grid(
		coordinates::AbstractVector{<:AbstractRange{T}},
	) where {T<:Real}
		N = length(coordinates)
		coordinates = SVector{N}(coordinates)
		spacings = SVector{N}(step.(coordinates))
		bounds = SMatrix{N,2,T}(reinterpret(reshape, T, extrema.(coordinates)))
		
		return Grid{N,T,N+1}(coordinates, spacings, bounds)
	end
	function Grid(coordinates::NTuple{N,AbstractRange{T}}) where {N,T<:Real}
		coordinates = SVector{N}(coordinates)
		spacings = SVector{N}(step.(coordinates))
		bounds = SMatrix{N,2,T}(
			reinterpret(reshape, T, collect(extrema.(coordinates)))
		)

		return Grid{N,T,N+1}(coordinates, spacings, bounds)
	end

	Base.size(grid::Grid{N,T}) where {N,T<:Real} = ntuple(
		i -> length(grid.coordinates), N
	)
	function Base.broadcastable(grid::Grid{N,T}) where {N,T<:Real}
		return reinterpret(
			reshape,
			T,
			collect(Iterators.product(grid.coordinates...))
		)
	end
end

# ╔═╡ e65784bb-2484-473d-ba03-649df17285e8
begin
	struct CuGrid{N,T<:Real,M} <: AbstractGrid{N,T,M}
		coordinates::CuArray{T,M}
		spacings::CuArray{T,1}
		bounds::CuArray{T,2}
	end

	function CuGrid(
	  ranges::Union{AbstractVector{<:AbstractRange{T}},NTuple{N,AbstractRange{T}}},
	)::CuGrid where {N,T<:Real}
		if isa(ranges, NTuple)
			ranges = collect(ranges)
			n = N
		else
			n = length(ranges)
		end
		complete_array = reinterpret(
			reshape,
			T,
			collect(Iterators.product(ranges...))
		)

		coordinates = CuArray{T}(complete_array)
		spacings = CuArray{T}(step.(ranges))
		bounds = CuArray{T}(
			reinterpret(reshape, T, extrema.(ranges))
		)

		return CuGrid{n,T,n+1}(coordinates, spacings, bounds)
	end

	function Base.size(grid::CuGrid{N,T}) where {N,T<:Real}
		return size(grid.coordinates)[2:end]
	end

	function Base.broadcastable(grid::CuGrid{N,T}) where {N,T<:Real}
		return grid.coordinates
	end

end

# ╔═╡ 71c71f33-5bbf-4979-9451-fc13b66f0c30
begin
	grid = CuGrid([0:0.1:1, 1:0.1:2])
	grid .+ 1.5
end

# ╔═╡ fec5d7ab-61ee-450b-a952-0e4fd3a102cf
# ╠═╡ disabled = true
#=╠═╡
begin
	grid = Grid([0:0.1:1, 1:0.1:2])
	grid .+ 1.5
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═6ca5c154-bf2b-4097-9b3a-2e1df93d043a
# ╠═0852562f-d8df-42f7-8189-c6119eae7a86
# ╠═21612857-e4e5-46b7-915d-5f41218f84be
# ╠═fec5d7ab-61ee-450b-a952-0e4fd3a102cf
# ╠═e65784bb-2484-473d-ba03-649df17285e8
# ╠═71c71f33-5bbf-4979-9451-fc13b66f0c30
