module MLJNearestNeighborsInterface

import MLJModelInterface
import MLJModelInterface: MODEL_TRAITS

using ScientificTypes
using MLJBase
import MLJBase: @load
import MLJBase: Table, Continuous, Count, Finite, OrderedFactor, Multiclass

using Requires, Pkg, Pkg.TOML, OrderedCollections, Parameters
using Tables, CategoricalArrays, StatsBase, Statistics
import Distributions




const srcdir = dirname(@__FILE__) # the directory containing this file

# TODO remove when the functionality has been merged in ScientificTypes.jl
# and use ScientificTypes.nonmissing then.
if VERSION < v"1.3"
    nonmissingtype(::Type{T}) where T =
        T isa Union ? ifelse(T.a == Missing, T.b, T.a) : T
end


# # lazily load in strap-on model interfaces for external packages:
# function __init__()

    @require(NearestNeighbors="b8a86587-4115-5ab1-83bc-aa920d37bbce",
             include("NearestNeighbors_.jl"))

# end

end # module
