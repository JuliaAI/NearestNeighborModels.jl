
abstract type KNNKernel end

sort_idxs(::KNNKernel) = false

# Function that lists all defined `KNNKernels`
list_kernels() = subtypes(KNNKernel) 

####
# User Defined Kernel (UDK)
####

_nothing(x) = nothing

"""
    UDK(;func::Function = x->nothing, sort::Bool=false)

Wrap a user defined nearest neighbors weighting function `func` as a `KNNKernel`.
 
# Keywords

- `func` : user-defined nearest neighbors weighting function. The function 
   should have the signature `func(dists_matrix)::Union{Nothing, <:AbstractMatrix}`, 
   where `dists_matrix` is the k-nearest neighbors distances matrix and outputs a matrix 
   of the same shape as `dists_matrix`.   
- `sort` : if true sorts the `dists_matrix` so that in each row the gives 
   k-nearest neighbors in acesending order.
  
"""
struct UDK{T<:Function} <: KNNKernel
   func::T
   sort::Bool
   UDK(func::F, sort::Bool) where {F<:Function} = new{F}(func, sort)
end

UDK(;func= _nothing, sort=false) = UDK(func, sort)
sort_idxs(kernel::UDK) = kernel.sort

function get_weights(kernel::UDK, dists_matrix)
    func = kernel.func
    nsamples, K = size(dists_matrix)
    weights = func(dists_matrix)
    weights === nothing && return nothing
    check_onebased_indexing("weights", weights)
    (nsamples, K) == size(weights) || throw(
        DimensionMismatch(
            "Expected a $((nsamples, K)) knn weights, got a $(size(weights)) knn "*
            "weights matrix"
        )
    )
    eltype(weights) <: AbstractFloat || throw(
        error(
            "The element type of knn weights matrix must be a subtype of `AbstractFloat`"
        )
    )
    return weights
end

####
# Uniform Kernel (Uniform)
####

"""
    Uniform()

assigns uniform weights to all k-nearest neighbors.

see also:
[`Inverse`](@ref), [`ISquared`](@ref), [`Zavreal`](@ref)     
"""
struct Uniform <: KNNKernel end

get_weights(::Uniform, dists_matrix) = nothing


####
# Dudani
####

"""
    Dudani()

Assigns the closest neighbor a weight of `1`, the furthest neighbor weight `0` and the 
others are scaled between by a linear mapping.

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

see also:
[`Macleod`](@ref), [`DualU`](@ref), [`DualD`](@ref)
"""
struct Dudani <: KNNKernel end

sort_idxs(::Dudani) = true

function _dudani_weights(dists_matrix, inds)
    i, j = inds.I
    dij = dists_matrix[i, j]
    dik = dists_matrix[i, end]
    di1 = dists_matrix[i, 1]
    w = (dik - dij)/ (dik - di1)
    return ifelse(isfinite(w), w, oneunit(w))
end

function get_weights(::Dudani, dists_matrix)
    return _dudani_weights.(Ref(dists_matrix), eachindex(dists_matrix))
end

####
# Inverse
####

"""
    Inverse()

Assigns each neighbor a weight equal to the inverse of the corresponsting distance of the 
neighbor.

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

see also:
[`ISquared`](@ref)
"""
struct Inverse <: KNNKernel end

function _inverse_weights(dist)
    w = 1/dist
    return w
end

function replace_infinite_weights!(weights)
    inf_mask = (!isfinite).(weights)
    inf_row = @view(any(inf_mask, dims=2)[:])
    @inbounds(weights[inf_row, :] .= @view(inf_mask[inf_row, :]))
    return weights
end

function get_weights(::Inverse, dists_matrix)
   weights =  _inverse_weights.(dists_matrix)
   # Check and correct for non-finite rows
   replace_infinite_weights!(weights)
   return weights
end

####
# Inverse Squared (ISquared)
####

## Takes SKLearn's approah of removing non finite values, 
## instead of adding a constant for numerical stability. 
"""
    ISquared()

Assigns each neighbor a weight equal to the inverse of the corresponsting squared-distance 
of the neighbor.

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

"""
struct ISquared <: KNNKernel end

function _inversesquared_weights(dist)
    w = 1/dist^2
    return w
end

function get_weights(::ISquared, dists_matrix)
    weights = _inversesquared_weights.(dists_matrix)
    # Check and correct for non-finite rows
    replace_infinite_weights!(weights)
    return weights
end

####
# Rank
####

"""
    Rank()

Assigns each neighbor a weight as a rank such that the closest neighbor get's a weight of 
`1` and the Kth closest neighbor gets a weight of `K`.

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

see also:
[`ReciprocalRank`](@ref)
"""
struct Rank <: KNNKernel end

sort_idxs(::Rank) = true

function get_weights(kernel::Rank, dists_matrix)
    k = size(dists_matrix, 2)
    weights = similar(dists_matrix, Int)
    for i in axes(weights, 2)
        weights[:, i] .= k - i + 1
    end
    return weights
end

####
# Macleod
####

#implemented with `s` = `k`
"""
    Macleod(;a::Real= 0.0)
    
Assigns the closest neighbor a weight of `1`, the furthest neighbor weight `0` and the 
others are scaled between by a linear mapping.

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

see also:
[`Dudani`](@ref), [`DualU`](@ref), [`DualD`](@ref)

"""
struct Macleod{T <: AbstractFloat} <: KNNKernel
    a::T
    Macleod(a::AbstractFloat) = new{typeof(a)}(a) 
end

Macleod(a::Real) = Macleod(float(a))
Macleod(;a=1.0) = Macleod(a)
sort_idxs(::Macleod) = true

function _macleod_weights(dists_matrix, inds, s, a)
    i, j = inds.I
    dij = dists_matrix[i, j]
    dis = dists_matrix[i, s]
    di1 = dists_matrix[i, 1]
    w = ((dis - dij) + a*(dis - di1)) / (1 + a)*(dis - di1)
    return ifelse(isfinite(w), w, oneunit(w))
end

function get_weights(kernel::Macleod, dists_matrix::Mat{T}) where {T}
    k = size(dists_matrix, 2)
    s = k
    #s = ifelse(iszero(kernel.s), k, kernel.s)
    #s >= 0 || throw(ArgumentError("`s` can't be less than number of neighbors `K`"))
    return _macleod_weights.(Ref(dists_matrix), eachindex(dists_matrix), s, T(kernel.a))
end

####
# Zavreal
####

"""
    Zavreal(;s::Real = 0.0, a::Real=1.0)
    
Assigns each neighbor an exponential weight given by
    ``e^{ - α ⋅ d_i^{\beta}}``
where `α` and `β` are constants and `dᵢ` is the distance of the given neighbor.

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

"""
struct Zavreal{T <: AbstractFloat} <: KNNKernel
    a::T
    b::T
    Zavreal(a::K, b::K) where {K <: AbstractFloat} = new{K}(a, b)  
end

Zavreal(;a=0.0, b= 1.0) = Zavreal(a, b)
Zavreal(a::Real, b::Real) = Zavreal(float(a), float(b))

function get_weights(kernel::Zavreal, dists_matrix::Mat{T}) where {T}
    return exp.(T(kernel.a) .* dists_matrix .^ T(kernel.b))
end

####
# ReciprocalRank
####
"""
    ReciprocalRank(;a::Real= 0.0)
    
Assigns each closest neighbor a weight which is equal to the reciprocal of it's rank. 
i.e the closest neighbor get's a weight of `1` and the Kth closest weight get's a weight 
of `1/K` 

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

see also:
[`Rank`](@ref)

"""
struct ReciprocalRank <: KNNKernel end

sort_idxs(::ReciprocalRank) = true

function get_weights(kernel::ReciprocalRank, dists_matrix)
    rowvector = 1 ./ axes(dists_matrix, 2)
    nrows = size(dists_matrix, 1)
    return RankOneMatrix(rowvector, nrows)
end

####
# DualU
####
"""
    DualU()
    
Assigns the closest neighbor a weight of `1`, the furthest neighbor weight `0` and the 
others are scaled between by a mapping.

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

see also:
[`DualD`](@ref)

"""
struct DualU <: KNNKernel end

sort_idxs(::DualU) = true

function _dualu_weights(dists_matrix, inds)
    i, j = inds.I
    dij = dists_matrix[i, j]
    dik = dists_matrix[i, end]
    di1 = dists_matrix[i, 1]
    w = (dik - dij)/ (dik - di1)
    return ifelse(isfinite(w), w, oneunit(w))
end

function get_weights(::DualU, dists_matrix)
    return _dualu_weights.(Ref(dists_matrix), eachindex(dists_matrix))
end

####
# DualD
####

"""
    DualD()
    
Assigns the closest neighbor a weight of `1`, the furthest neighbor weight `0` and the 
others are scaled between by a mapping.

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

see also:
[`DualU`](@ref)

"""
struct DualD <: KNNKernel end

sort_idxs(::DualD) = true

function _duald_weights(dists_matrix, inds)
    i, j = inds.I
    dij = dists_matrix[i, j]
    dik = dists_matrix[i, end]
    di1 = dists_matrix[i, 1]
    w = ((dik - dij)/ (dik - di1)) * ((dik + di1) / (dik + dij))
    return ifelse(isfinite(w), w, oneunit(w))
end

# eachindex(dists_matrix) gives `CartesianIndices` because 
# `Base.IndexStyle(dists_matrix) == IndexCartesian`
function get_weights(::DualD, dists_matrix)
    return _dualu_weights.(Ref(dists_matrix), eachindex(dists_matrix))
end

####
# Fibonacci
####
"""
    Fibonacci()
    
Assigns neighbors weights corresponding to fibonacci numbers starting from the furthest
neighbor. i.e the furthest neighbor a weight of `1`, the second furthest neighbor a 
weight of `1` and the third furthest neighbor a weight of `2` and so on.  

For more information see the paper by Geler et.al [Comparison of different weighting 
schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/
radovanovic/publications/2016-kais-knn-weighting.pdf).

"""
struct Fibonacci <: KNNKernel end

sort_idxs(::Fibonacci) = true

function get_weights(::Fibonacci, dists_matrix)
    nrows, ncols = size(dists_matrix)
    rowvector = similar(Array{Int}, ncols)
    rowvector[end] = rowvector[end - 1] = 1
    for i in  StepRange(ncols - 2, -1, 1)
       rowvector[i] = rowvector[i+1] + rowvector[i+2]
    end
    return RankOneMatrix(rowvector, nrows)
end


