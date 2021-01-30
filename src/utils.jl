# The following check is done because `MLJBase.clean!(model)` isn't called for 
# `MLJBase.predict` and the user may modify the model's`K` value
# wrapped in a machine.  
function err_if_given_invalid_K(K)
    K > 0 || throw(ArgumentError("Number of neighbors `K` has to be greater than 0"))
    return K
end

# internal method essentially the same as Base.replace!(y, (z .=> r)...)
# but more efficient.
# Similar to the behaviour of `Base.replace!` if `z` contain repetions of values in 
# `y` then only the transformation corresponding to the first occurence is performed
# i.e `_replace!([1,5,3], [1,4], 4:5)` would return `[4,5,3]` rather than `[5,5,3]`
# (which replaces `1=>4` and then `4=>5`)
function _replace!(y::AbstractVector, z::AbstractVector, r::AbstractVector)
    length(r) == length(z) || 
     throw(DimensionMismatch("`z` and `r` has to be of the same length"))
    @inbounds for i in eachindex(y)
        for j in eachindex(z) 
            isequal(z[j], y[i]) && (y[i] = r[j]; break)
        end
    end
    return y
end

### Methods for combining knn_weights and sample weights
combine_weights(idxs_matrix, knn_weights, ::Nothing) = knn_weights 
combine_weights(idxs_matrix, ::Nothing, ::Nothing) = Fill(1, size(idxs_matrix))
combine_weights(idxs_matrix, ::Nothing, sample_weights) = @inbounds(@view(sample_weights[idxs_matrix]))

@inline function combine_weights(idxs_matrix, knn_weights, sample_weights)
    @inbounds(sw = @view(sample_weights[idxs_matrix]))
    weights = knn_weights .* sw
    return weights 
end

### Specialized `sum` methods
_sum(x; dims=:) = sum(x, dims=dims)
_sum(x::Fill{Int, 2}; dims=:) = __sum(x, dims)

@inline function __sum(x::Fill{Int, 2}, dims::Int)
    dims  > 2 && return x
    len = size(x, dims)
    axis = dims == 1 ? (1, size(x, 2)) : (size(x, 1), 1)
    fill_value = @inbounds first(x)
    return Fill(fill_value * len, axis) 
end

@inline function __sum(x::Fill{Int, 2}, dims::Colon)
    fill_value = @inbounds first(x)
    return *(size(x)...) * fill_value 
end

### Normalize the rows or columns of an `AbstractMatrix` by its sum, rowsum or columnsum.
### Specialization introduced for some julia Base `<:Real` types to gain further 
### performance.
function scale(A::Mat{<:Union{Base.IEEEFloat, BigFloat, Rational}}; dims=:)
    axis_sum = sum(A,dims=dims)
    A ./= axis_sum
    return A
end

function scale(A::Mat{<:Real}; dims=:)
    axis_sum = sum(A, dims=dims)
    A_scaled = A ./ axis_sum
    return A_scaled
end


### Wrap knn distances and indices which are `Vector{Vector{<:AbstractFloat}}` as
## `NeighDistsMatrix` and `NeighIndsMatrix` respectively. This among other things 
## makes it easy to write kernels
struct NeighIndsMatrix{T<:Integer} <: AbstractMatrix{T}
    vecofvecs::Vector{Vector{T}}
end

struct NeighDistsMatrix{T} <: AbstractMatrix{T}
    vecofvecs::Vector{Vector{T}}
end

function Base.size(A::Union{NeighIndsMatrix, NeighDistsMatrix})
    return (length(A.vecofvecs), length(@inbounds(A.vecofvecs[1])))
end

Base.IndexStyle(m::Union{NeighIndsMatrix, NeighDistsMatrix}) = Base.IndexCartesian()

@inline function Base.getindex(A::Union{NeighIndsMatrix, NeighDistsMatrix}, i::Int, j::Int)
    @boundscheck Base.checkbounds(A, i, j)
    # Ai is the ith row of matrix `A`
    @inbounds(Ai = A.vecofvecs[i])
    # Aij is the element in the ith row and jth column of matrix `A`
    @inbounds(Aij = Ai[j])  
    return Aij
end

@inline function Base.view(A::Union{NeighIndsMatrix, NeighDistsMatrix}, i::Int, j::Colon)
    #J = map(i->unalias(A,i), to_indices(A, i))
    J = Base.to_indices(A, (i, j))
    @boundscheck Base.checkbounds(A, J...)
    # Ai is the ith row of matrix `A`
    @inbounds(Ai = A.vecofvecs[i])
    return Ai
end

function Base.unaliascopy(A::Union{NeighIndsMatrix, NeighDistsMatrix})::typeof(A)
    v = A.vecofvecs
    return typeof(A)([copy(@inbounds(v[i])) for i in eachindex(v)])
end

@inline function Base.setindex!(
    A::Union{NeighIndsMatrix, NeighDistsMatrix},
    val,
    i::Int,
    j::Int
) 
    @boundscheck Base.checkbounds(A, i, j)
    # Ai is the ith row of matrix `A`
    @inbounds(Ai = A.vecofvecs[i])
    @inbounds Base.setindex!(Ai, val, j)
    return val
end


### Efficient way to store and index into Matrices with 
### the same rows i.e Rank One Matrices. This is perfect for our usecase 
### since we don't modify weight matrix.
struct RankOneMatrix{T} <: AbstractMatrix{T}
    rowvector::Vector{T}
    nrows::Int
end

Base.IndexStyle(::RankOneMatrix) = Base.IndexCartesian()
Base.size(A::RankOneMatrix) = (A.nrows, length(A.rowvector))

@inline function Base.getindex(A::RankOneMatrix, i::Int, j::Int)
    @boundscheck Base.checkbounds(A, i, j) 
    Aij = @inbounds(A.rowvector[j])
    return Aij
end

function Base.unaliascopy(A::RankOneMatrix)::typeof(A)
    row_vector = A.rowvector
    nrows = A.nrows
    return typeof(A)(copy(row_vector), nrows)
end

### Check to make sure a given `AbstractArray` isa one based indexed array.
@inline function check_onebased_indexing(argname::String, matorvec; istable=false)
    Base.has_offset_axes(matorvec) && begin
        if matorvec isa AbstractVector
            throw(ArgumentError(
                "$argname vector must be a 1-based indexed `AbstractVector`"
                )
            )
        elseif matorvec isa AbstractMatrix     
            throw(
                ArgumentError(
                    "$(argname) matrix $(
                        ifelse(istable, "generated from $(argname) table", "")
                    )  must be a 1-based indexed `AbstractMatrix`"
                )
            )
        end
    end
    return nothing
end
