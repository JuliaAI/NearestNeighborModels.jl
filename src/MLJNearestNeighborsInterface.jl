module NearestNeighborsModels

# ===================================================================
# IMPORTS
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

# ===================================================================
## EXPORTS
export list_kernels, ColumnTable, DictTable

# Export KNN models
# KNN models are exported automatically by `@mjl_model`

# Re-Export Distance Metrics from `Distances.jl`
export Euclidean, Cityblock, Minkowski, Chebyshev, Hamming, WeightedEuclidean,
    WeightedCityblock, WeightedMinkowski

# Export KNN Kernels
export DualU, DualD, Dudani, Fibonacci, Inverse, ISquared, KNNKernel, Macleod, Rank,
    ReciprocalRank, UDK, Uniform, UserDefinedKernel, Zavreal

# ===================================================================
## CONSTANTS
const Vec{T} = AbstractVector{T}
const Mat{T} = AbstractMatrix{T}
const Arr{T, N} = AbstractArray{T, N}
const ColumnTable =  Tables.ColumnTable
const DictTable = Dict{Symbol, <:AbstractVector}
const MultiUnivariateFinite = Union{DictTable, ColumnTable}

# Define constants for easy referencing of packages
const MMI = MLJModelInterface
const NN = NearestNeighbors
const PKG = "MLJNearestNeighborsInterface"

# Definitions of model descriptions for use in model doc-strings.
const KNNRegressorDescription = """
    K-Nearest Neighbors regressor: predicts the response associated with a new point
    by taking an weighted average of the response of the K-nearest points.
    """

const KNNClassifierDescription = """
    K-Nearest Neighbors classifier: predicts the class associated with a new point
    by taking a vote over the classes of the K-nearest points.
    """

const KNNCoreFields = """
    * `K::Int=5` : number of neighbors
    * `algorithm::Symbol = :kdtree` : one of `(:kdtree, :brutetree, :balltree)`
    * `metric::Metric = Euclidean()` : a `Metric` object for the distance between points
    * `leafsize::Int = 10` : at what number of points to stop splitting the tree
    * `reorder::Bool = true` : if true puts points close in distance close in memory
    * `weights::KNNKernel=Uniform()` : kernel used in assigning weights to the 
        k-nearest neighbors for each observation. An instance of one of the types in 
        `list_kernels()`. User-defined weighting functions can be passed by wrapping the 
        function in a `UDF` kernel. If sample weights `w` are passed during machine 
        construction e.g `machine(model, X, y, w)` then the weight assigned to each 
        neighbor is the product of the `KNNKernel` generated weight and the corresponding 
        neighbor sample weight.
     
    """

const SeeAlso = """
    See also the 
    [package documentation](https://github.com/KristofferC/NearestNeighbors.jl).
    For more information about the kernels see the paper by Geler et.al 
    [Comparison of different weighting schemes for the kNN classifier
    on time-series data]
    (https://perun.pmf.uns.ac.rs/radovanovic/publications/2016-kais-knn-weighting.pdf).
    """

const MultitargetKNNClassifierFields = """
    ## Keywords Parameters
    
    $KNNCoreFields
    * `output_type::Type{<:MultiUnivariateFinite}=DictTable` : One of 
       (`ColumnTable`, `DictTable`). The type of table type to use for predictions.
       Setting to `ColumnTable` might improve performance for narrow tables while setting to 
       `DictTable` improves performance for wide tables.
    
    $SeeAlso
    
    """

const KNNFields = """
    ## Keywords Parameters
    
    $KNNCoreFields
    
    $SeeAlso
    
    """

# ===================================================================
# Includes
include("utils.jl")
include("kernels.jl")
include("models.jl")
    
# ===================================================================
# List of all models interfaced
const MODELS = (
    KNNClassifier, KNNRegressor, MultitargetKNNRegressor, MultitargetKNNClassifier
)

# ====================================================================
# PKG_METADATA
metadata_pkg.(
    MODELS,
    name = "NearestNeighborsModels",
    uuid = "6f286f6a-111f-5878-ab1e-185364afe411",
    url = "https://github.com/alan-turing-institute/NearestNeighborsModels.jl",
    license = "MIT",
    julia = true,
    is_wrapper = false
)

end # module
