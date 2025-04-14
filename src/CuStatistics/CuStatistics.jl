Statistics._var(A::AnyCuArray, corrected::Bool, mean, dims) =
  sum((A .- something(mean, Statistics.mean(A, dims=dims))) .^ 2, dims=dims) / (prod(size(A)[[dims...]])::Int - corrected)

Statistics._var(A::AnyCuArray, corrected::Bool, mean, ::Colon) =
  sum((A .- something(mean, Statistics.mean(A))) .^ 2) / (length(A) - corrected)

Statistics._std(A::AnyCuArray, corrected::Bool, mean, dims) =
  Base.sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=dims))

Statistics._std(A::AnyCuArray, corrected::Bool, mean, ::Colon) =
  Base.sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=:))

Statistics._mean(A::AnyCuArray, ::Colon) = sum(A) / length(A)
Statistics._mean(f, A::AnyCuArray, ::Colon) = sum(f, A) / length(A)
Statistics._mean(A::AnyCuArray, dims) = mean!(Base.reducedim_init(t -> t / 2, +, A, dims), A)
Statistics._mean(f, A::AnyCuArray, dims) = sum(f, A, dims=dims) / mapreduce(i -> size(A, i), *, unique(dims); init=1)

# HACK: I think 'mean' should never be called this way, and 'dims' should be an integer 
# or a tuple of integers. However, this removes the ambiguity found by Aqua.jl
Statistics._mean(A::AnyCuArray, dims::AbstractArray) = mean!(Base.reducedim_init(t -> t / 2, +, A, dims), A)
