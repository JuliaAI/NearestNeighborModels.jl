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

# for administrators to update Metadata.toml:
export @update, check_registry

# from loading.jl:
export load, @load, info

# from model_search:
export models, localmodels


const srcdir = dirname(@__FILE__) # the directory containing this file

# TODO remove when the functionality has been merged in ScientificTypes.jl
# and use ScientificTypes.nonmissing then.
if VERSION < v"1.3"
    nonmissingtype(::Type{T}) where T =
        T isa Union ? ifelse(T.a == Missing, T.b, T.a) : T
end
nonmissing = nonmissingtype

include("metadata.jl")
include("model_search.jl")
include("loading.jl")
include("registry/src/Registry.jl")
include("registry/src/check_registry.jl")
import .Registry.@update


const INFO_GIVEN_HANDLE = Dict{Handle,Any}()
const PKGS_GIVEN_NAME   = Dict{String,Vector{String}}()
const AMBIGUOUS_NAMES   = String[]
const NAMES             = String[]

metadata_file = joinpath(srcdir, "registry", "Metadata.toml")

merge!(INFO_GIVEN_HANDLE, info_given_handle(metadata_file))
merge!(PKGS_GIVEN_NAME, pkgs_given_name(INFO_GIVEN_HANDLE))
append!(AMBIGUOUS_NAMES, ambiguous_names(INFO_GIVEN_HANDLE))
append!(NAMES, model_names(INFO_GIVEN_HANDLE))
@info "Model metadata loaded from registry. "

# lazily load in strap-on model interfaces for external packages:
function __init__()

    @require(NearestNeighbors="b8a86587-4115-5ab1-83bc-aa920d37bbce",
             include("NearestNeighbors.jl"))

end

end # module
