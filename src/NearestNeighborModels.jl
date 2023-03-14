module NearestNeighborModels

# ==============================================================================================
# IMPORTS
import InteractiveUtils: subtypes
import MLJModelInterface
import MLJModelInterface: @mlj_model, metadata_model, metadata_pkg,
    Table, Continuous, Count, Finite, OrderedFactor, Multiclass
import NearestNeighbors
import StatsBase
import Tables

using Distances
using FillArrays
using LinearAlgebra
using Statistics

# ==============================================================================================
## EXPORTS
export list_kernels, ColumnTable, DictTable

# Export KNN models
export KNNClassifier, KNNRegressor, MultitargetKNNClassifier, MultitargetKNNRegressor

# Re-Export Distance Metrics from `Distances.jl`
export Euclidean, Cityblock, Minkowski, Chebyshev, Hamming, WeightedEuclidean,
    WeightedCityblock, WeightedMinkowski

# Export KNN Kernels
export DualU, DualD, Dudani, Fibonacci, Inverse, ISquared, KNNKernel, Macleod, Rank,
    ReciprocalRank, UDK, Uniform, UserDefinedKernel, Zavreal

# ===============================================================================================
## CONSTANTS
const Vec{T} = AbstractVector{T}
const Mat{T} = AbstractMatrix{T}
const Arr{T, N} = AbstractArray{T, N}
const ColumnTable =  Tables.ColumnTable
const DictTable = Tables.DictColumns
const MultiUnivariateFinite = Union{DictTable, ColumnTable}

# Define constants for easy referencing of packages
const MMI = MLJModelInterface
const NN = NearestNeighbors
const PKG = "NearestNeighborModels"

# ==============================================================================================
# Includes
include("utils.jl")
include("kernels.jl")
include("models.jl")
    
# ===============================================================================================
# List of all models interfaced
const MODELS = (
    KNNClassifier, KNNRegressor, MultitargetKNNRegressor, MultitargetKNNClassifier
)

# ===============================================================================================
# PKG_METADATA
metadata_pkg.(
    MODELS,
    package_name = "NearestNeighborModels",
    package_uuid = "6f286f6a-111f-5878-ab1e-185364afe411",
    package_url = "https://github.com/JuliaAI/NearestNeighborModels.jl",
    package_license = "MIT",
    is_pure_julia = true,
    is_wrapper = false
)

end # module
