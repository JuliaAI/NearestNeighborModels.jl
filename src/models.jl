##########################
### MultitargetKNNClassifier
##########################

mutable struct MultitargetKNNClassifier <: MMI.Probabilistic
    K::Int
    algorithm::Symbol
    metric::Metric
    leafsize::Int
    reorder::Bool
    weights::KNNKernel
    output_type::Type{<:MultiUnivariateFinite}
end

function _predictmode_knnclassifier(weights, y, idxsvec)
    return StatsBase.mode.(_predict_knnclassifier(weights, y, idxsvec))
end

function dict_preds(::Val{:columnaccess}, func, target_table, idxsvec, weights)
    cols = Tables.columns(target_table)
    colnames = Tables.columnnames(cols)
    dict = OrderedDict{Symbol, AbstractVector}(
        nm => func(weights, Tables.getcolumn(cols, nm), idxsvec) for nm in colnames
    )
    dict_table = Tables.DictColumnTable(Tables.Schema(colnames, eltype.(values(dict))), dict)
    return dict_table
end

function dict_preds(::Val{:noncolumnaccess}, func, target_table, idxsvec, weights)
    cols = Tables.dictcolumntable(target_table)
    colnames = Tables.columnnames(cols)
    dict = OrderedDict{Symbol, AbstractVector}(
       nm => func(weights, Tables.getcolumn(cols, nm), idxsvec) for nm in colnames
    )
    dict_table = Tables.DictColumnTable(Tables.Schema(colnames, eltype.(values(dict))), dict)
    return dict_table
end

function dict_preds(func::F, target_table, idxsvec, weights) where {F<:Function}
    preds = if Tables.columnaccess(target_table)
        dict_preds(Val(:columnaccess), func, target_table, idxsvec, weights)
    else
        dict_preds(Val(:noncolumnaccess), func, target_table, idxsvec, weights)
    end
    return preds
end

function ntuple_preds(::Val{:columnaccess}, func, target_table, idxsvec, weights)
    cols = Tables.columns(target_table)
    colnames = Tables.columnnames(cols)
    column_table = NamedTuple{Tuple(colnames)}(
        func(weights, Tables.getcolumn(cols, nm), idxsvec) for nm in colnames
    )
    return column_table
end

function ntuple_preds(::Val{:noncolumnaccess}, func, target_table, idxsvec, weights)
    cols = Tables.columntable(target_table)
    colnames = Tables.columnnames(cols)
    column_table = NamedTuple{colnames}(
        func(weights, Tables.getcolumn(col, nm), idxsvec) for nm in colnames
    )
    return column_table
end

function ntuple_preds(func::F, target_table, idxsvec, weights) where {F <: Function}
    preds = if Tables.columnaccess(target_table)
        ntuple_preds(Val(:columnaccess), func, target_table, idxsvec, weights)
    else
        ntuple_preds(Val(:noncolumnaccess), func, target_table, idxsvec, weights)
    end
    return preds
end

function univariate_table(::Type{T}, weights, target_table, idxsvec) where {T}
    table = if T <: DictColumnTable
        dict_preds(_predict_knnclassifier, target_table, idxsvec, weights)
    else
        ntuple_preds(_predict_knnclassifier, target_table, idxsvec, weights)
    end
    return table
end

function categorical_table(::Type{T}, weights, target_table, idxsvec) where {T}
    table = if T <: DictColumnTable
        dict_preds(_predictmode_knnclassifier, target_table, idxsvec, weights)
    else
        ntuple_preds(_predictmode_knnclassifier, target_table, idxsvec, weights)
    end
    return table
end

function MMI.predict(m::MultitargetKNNClassifier, fitresult, X)
    err_if_given_invalid_K(m.K)
    Xmatrix = transpose(MMI.matrix(X))
    check_onebased_indexing("prediction input", Xmatrix)
    args = setup_predict_args(m, Xmatrix, fitresult)
    table_of_univariate_probs = univariate_table(m.output_type, args...)
    return table_of_univariate_probs
end

function MMI.predict_mode(m::MultitargetKNNClassifier, fitresult, X)
    err_if_given_invalid_K(m.K)
    Xmatrix = transpose(MMI.matrix(X))
    check_onebased_indexing("prediction input", Xmatrix)
    args = setup_predict_args(m, Xmatrix, fitresult)
    table_of_probs = categorical_table(m.output_type, args...)
    return table_of_probs
end

##########################
### KNNClassifier
##########################

mutable struct KNNClassifier <: MMI.Probabilistic
    K::Int
    algorithm::Symbol
    metric::Metric
    leafsize::Int
    reorder::Bool
    weights::KNNKernel
end

function setup_predict_args(
    m::Union{KNNClassifier, MultitargetKNNClassifier}, Xmatrix, fitresult
)
    tree, targetvecortable, sample_weights = (
        fitresult.tree, fitresult.target, fitresult.sample_weights
    )
    kernel = m.weights
    # for each entry, get the K closest training point + their distance
    idxsvec, distsvec = NN.knn(tree, Xmatrix, m.K, sort_idxs(kernel))
    idxs_matrix = NeighIndsMatrix(idxsvec)
    dists_matrix = NeighDistsMatrix(distsvec)
    knn_weights = get_weights(kernel, dists_matrix)
    weights = combine_weights(idxs_matrix, knn_weights, sample_weights)
    return weights, targetvecortable, idxsvec
end

function _predict_knnclassifier(weights, y, idxsvec)
    # `classes` and `classes_seen` has an ordering consistent with the pool of y
    classes = @inbounds MMI.classes(y[1])
    classes_seen = filter(in(y), classes)
    nclasses = length(classes)
    nc = length(classes_seen) # Number of classes seen in `y`.

    # `labels` and `integers_seen` assigns ranks {1, 2, ..., nclasses}, to the
    # `CategoricalValue`'s in `y` according the ordering given by the pool of `y`.
    # In the case of `integers_seen` the ranks happen to sorted with the lower ranks
    # comming first and the higher rank coming last.
    # In both cases some ranks may be absent because some categorical values in `classes`
    # may be absent from `y` (in the case of `labels`) and `classes_seen`
    # (in the case of `integers_seen`).
    labels = MMI.int(y)
    integers_seen = MMI.int(classes_seen)

    # Recode `integers_seen` to be in {1,..., nc}
    # same as nc == nclasses || replace!(labels, (integers_seen .=> 1:nc)...)
    nc == nclasses || _replace!(labels, integers_seen, 1:nc)
    nsamples = length(idxsvec)
    probs = zeros(eltype(weights), nsamples, nc)

    @inbounds for i in eachindex(idxsvec)
        idxs = idxsvec[i]
        idxs_labels = @view(labels[idxs])
        @simd for j in eachindex(idxs_labels)
             @inbounds probs[i, idxs_labels[j]] += weights[i, j]
        end
    end
    return MMI.UnivariateFinite(classes_seen, scale(probs, dims=2))
end

function MMI.predict(m::KNNClassifier, fitresult, X)
    err_if_given_invalid_K(m.K)
    Xmatrix = transpose(MMI.matrix(X))
    check_onebased_indexing("prediction input", Xmatrix)
    predict_args = setup_predict_args(m, Xmatrix, fitresult)
    uni_var_probs = _predict_knnclassifier(predict_args...)
    return uni_var_probs
end

##########################
### KNNRegressor
##########################

mutable struct KNNRegressor <: MMI.Deterministic
    K::Int
    algorithm::Symbol
    metric::Metric
    leafsize::Int
    reorder::Bool
    weights::KNNKernel
end

function setup_predict_args(
    m::KNNRegressor, Xmatrix, fitresult
)
    tree, target, sample_weights = (
        fitresult.tree, fitresult.target, fitresult.sample_weights
    )
    kernel = m.weights
    # for each entry, get the K closest training point + their distance
    idxsvec, distsvec = NN.knn(tree, Xmatrix, m.K, sort_idxs(kernel))
    idxs_matrix = NeighIndsMatrix(idxsvec)
    dists_matrix = NeighDistsMatrix(distsvec)
    knn_weights = get_weights(kernel, dists_matrix)
    weights = combine_weights(idxs_matrix, knn_weights, sample_weights)
    return weights, target, idxs_matrix
end

function _predict_knnreg(weights, y, idxs_matrix)
    @inbounds labels = @view(y[idxs_matrix])
    preds = @inbounds(
        @view(_sum(labels .* weights, dims=2)[:]) ./ @view(_sum(weights, dims=2)[:])
    )
    return preds
end

function MMI.predict(m::KNNRegressor, fitresult, X)
    err_if_given_invalid_K(m.K)
    #Xmatrix = MMI.matrix(X, transpose=true) # NOTE: copies the data
    Xmatrix = transpose(MMI.matrix(X))
    check_onebased_indexing("prediction input", Xmatrix)
    predict_args = setup_predict_args(m, Xmatrix, fitresult)
    preds = _predict_knnreg(predict_args...)
    return preds
end

############################
### MultitargetKNNRegressor
############################

mutable struct MultitargetKNNRegressor <: MMI.Deterministic
    K::Int
    algorithm::Symbol
    metric::Metric
    leafsize::Int
    reorder::Bool
    weights::KNNKernel
end

function setup_predict_args(
    m::MultitargetKNNRegressor, Xmatrix, fitresult
)
    tree, names, target, sample_weights = (
        fitresult.tree,
        fitresult.target.names,
        fitresult.target.target,
        fitresult.sample_weights
    )
    kernel = m.weights
    # for each entry, get the K closest training point + their distance
    idxsvec, distsvec = NN.knn(tree, Xmatrix, m.K, sort_idxs(kernel))
    idxs_matrix = NeighIndsMatrix(idxsvec)
    dists_matrix = NeighDistsMatrix(distsvec)
    knn_weights = get_weights(kernel, dists_matrix)
    weights = combine_weights(idxs_matrix, knn_weights, sample_weights)
    return names, (weights, target, idxs_matrix)
end

function mul_label_by_weight(idxs_matrix, weights, Ymatrix, ind)
    i, j, k = ind.I
    @inbounds(label = Ymatrix[idxs_matrix[i, k], j])
    @inbounds(weight = weights[i, k])
    return label * weight
end

function _predict_multiknnreg(weights, Ymatrix, idxs_matrix)
    cartesian_inds = CartesianIndices(
        (axes(idxs_matrix, 1), axes(Ymatrix, 2), axes(idxs_matrix, 2))
    )
    weighted_labels = _sum(
        mul_label_by_weight.(
            Ref(idxs_matrix),
            Ref(weights),
            Ref(Ymatrix),
            cartesian_inds
        );
        dims = 3
   )
   # Normalizing the weighted labels gives us our final preds
   preds = @inbounds(@view(weighted_labels[:, :, 1])) ./ _sum(weights, dims=2)
   return preds
end

function MMI.predict(m::MultitargetKNNRegressor, fitresult, X)
    err_if_given_invalid_K(m.K)
    #Xmatrix = MMI.matrix(X, transpose=true) # NOTE: copies the data
    Xmatrix = transpose(MMI.matrix(X))
    check_onebased_indexing("prediction input", Xmatrix)
    names, args = setup_predict_args(m, Xmatrix, fitresult)
    preds = _predict_multiknnreg(args...)
    # A matrix table was intentionally outputed.
    # Users can coerce to their desired Table type.
    return MMI.table(preds, names = names)
end

################################
### Model Keyword Constructors
################################
const MODELS = (
    KNNClassifier, KNNRegressor, MultitargetKNNRegressor, MultitargetKNNClassifier
)

const KNN = Union{
    MODELS...
}

function MultitargetKNNClassifier(;
    K::Int=5,
    algorithm::Symbol=:kdtree,
    metric::Metric=Euclidean(),
    leafsize::Int = (algorithm == :brutetree) ? 0 : 10,
    reorder::Bool = algorithm != :brutetree,
    weights::KNNKernel=Uniform(),
    output_type::Type{<:MultiUnivariateFinite} = DictColumnTable
)   
    model = MultitargetKNNClassifier(
        K, algorithm, metric, leafsize, reorder, weights, output_type
    )
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

for knnmodel in (:KNNRegressor, :KNNClassifier, :MultitargetKNNRegressor)
    quote
    function $knnmodel(;
        K::Int=5,
        algorithm::Symbol=:kdtree,
        metric::Metric=Euclidean(),
        leafsize::Int = (algorithm == :brutetree) ? 0 : 10,
        reorder::Bool = algorithm != :brutetree,
        weights::KNNKernel=Uniform()
    )   
        model = $knnmodel(
            K, algorithm, metric, leafsize, reorder, weights
        )
        message = MMI.clean!(model)
        isempty(message) || @warn message
        return model
    end
    end |> eval
end

function MMI.clean!(model::KNN)
    warning = ""
    if model.K < 1
        warning *= "Need `K`>=1.\nResetting `K=5`.\n"
        model.K = 5
    end
    if !(model.algorithm in (:kdtree, :balltree, :brutetree))
        warning *= "`algorithm` must be set to one of `:kdtree`, `:balltree`, `:brutetree`." *
        "\nResetting `algorithm = :kdtree`.\n"
        model.algorithm = :kdtree
    end
    if model.algorithm == :brutetree
        if model.leafsize != 0
            warning *= "Non-zero `leafsize` isn't supported for `algorithm = :brutetree`." *
                "\nResetting `leafsize = 0`.\n"
            model.leafsize = 0
        end
        if model.reorder
            warning *= "Reordering isn't supported for `algorithm = :brutetree`." *
                "\nResetting `reorder = false`.\n"
            model.reorder = false
        end
    else
        if model.leafsize < 1
            warning *= "For `algorithm != :brutetree` Need `leafsize >= 1`."*
                "\nResetting `leafsize = 10`.\n"
            model.leafsize = 10
        end
    end
    if model.algorithm == :kdtree && !isa(model.metric, NN.MinkowskiMetric)
        warning *= "For `algorithm = :kdtree` only metrics which are of type" * 
        "`$(NN.MinkowskiMetric)` are supported.\nResetting `metric = Euclidean()`.\n"
        model.metric = Euclidean()
    end
    return warning
end

##################
### Model Fit
##################

struct MultiKNNRegressorTarget{S, M<:AbstractMatrix}
    names::S
    target::M
end

_getcolnames(::Nothing) = nothing
_getcolnames(sch) = sch.names

function get_target(::MultitargetKNNRegressor, y)
    target = MMI.matrix(y)
    check_onebased_indexing("input_target", target)
    names = _getcolnames(MMI.schema(y))
    return MultiKNNRegressorTarget(names, target)
end

function get_target(::KNNRegressor, y)
   check_onebased_indexing("input_target", y)
   return y
end

# `CategoricalArrays` and `Table(Finite)` are one-based indexed arrays and tables.
get_target(::Union{KNNClassifier, MultitargetKNNClassifier}, y) = y

struct KNNResult{T, U, W}
    tree::T
    target::U
    sample_weights::W
end

function MMI.fit(m::KNN, verbosity::Int, X, y, w::Union{Nothing, Vec{<:Real}}=nothing)
    Xmatrix = transpose(MMI.matrix(X))
    target = get_target(m, y)
    if w !== nothing
        check_onebased_indexing("sample_weights", w)
        length(w) == size(Xmatrix, 2) || throw(
            DimensionMismatch(
                "Differing number of observations in sample_weights and target vector."
            )
        )
    end
    if m.K > size(Xmatrix, 2)
        throw(
            ArgumentError(
                "Number of neighbors, `K` can't exceed the number of "*
                "observations or samples in the feature table `X`"
            )
        )
    end
    if m.algorithm == :kdtree
        tree = NN.KDTree(Xmatrix, m.metric; leafsize=m.leafsize, reorder=m.reorder)
    elseif m.algorithm == :balltree
        tree = NN.BallTree(Xmatrix, m.metric; leafsize=m.leafsize, reorder=m.reorder)
    elseif m.algorithm == :brutetree
        tree = NN.BruteTree(Xmatrix, m.metric; leafsize=m.leafsize, reorder=m.reorder)
    end
    report = NamedTuple{}()
    return KNNResult(tree, target, w), nothing, report
end

MMI.fitted_params(model::KNN, fitresult) = (tree=fitresult.tree,)

########################
### Models Metadata
########################

metadata_model(
    KNNRegressor,
    input = Table(Continuous),
    target = Vec{Continuous},
    human_name = "K-nearest neighbor regressor",
    weights = true,
    path = "$(PKG).KNNRegressor"
)

metadata_model(
    KNNClassifier,
    input = Table(Continuous),
    target = Vec{<:Finite},
    human_name = "K-nearest neighbor classifier",
    weights = true,
    path = "$(PKG).KNNClassifier"
)

metadata_model(
    MultitargetKNNClassifier,
    input = Table(Continuous),
    target = Table(Finite),
    human_name = "multitarget K-nearest neighbor classifier",
    weights = true,
    path = "$(PKG).MultitargetKNNClassifier"
)

metadata_model(
    MultitargetKNNRegressor,
    input = Table(Continuous),
    target = Table(Continuous),
    human_name = "multitarget K-nearest neighbor regressor",
    weights = true,
    path = "$(PKG).MultitargetKNNRegressor"
)

########################
### PKG Metadata
########################
metadata_pkg.(
    MODELS,
    package_name = "NearestNeighborModels",
    package_uuid = "6f286f6a-111f-5878-ab1e-185364afe411",
    package_url = "https://github.com/JuliaAI/NearestNeighborModels.jl",
    package_license = "MIT",
    is_pure_julia = true,
    is_wrapper = false
)

########################
### Models Docstrings
########################
const KNNFITTEDPARAMS = """
The fields of `fitted_params(mach)` are:

- `tree`: An instance of either `KDTree`, `BruteTree` or `BallTree` depending on the 
  value of the `algorithm` hyperparameter (See hyper-parameters section above). 
  These are data structures that stores the training data with the view of making 
  quicker nearest neighbor searches on test data points.
"""
const KNNFIELDS = """
    * `K::Int=5` : number of neighbors
    * `algorithm::Symbol = :kdtree` : one of `(:kdtree, :brutetree, :balltree)`
    * `metric::Metric = Euclidean()` : any `Metric` from 
        [Distances.jl](https://github.com/JuliaStats/Distances.jl) for the 
        distance between points. For `algorithm = :kdtree` only metrics which are 
        instances of `$(NN.MinkowskiMetric)` are supported.
    * `leafsize::Int = algorithm == 10` : determines the number of points 
        at which to stop splitting the tree. This option is ignored and always taken as `0` 
        for `algorithm = :brutetree`, since `brutetree` isn't actually a tree.
    * `reorder::Bool = true` : if `true` then points which are close in 
        distance are placed close in memory. In this case, a copy of the original data 
        will be made so that the original data is left unmodified. Setting this to `true` 
        can significantly improve performance of the specified `algorithm` 
        (except `:brutetree`). This option is ignored and always taken as `false` for 
        `algorithm = :brutetree`.
    * `weights::KNNKernel=Uniform()` : kernel used in assigning weights to the 
        k-nearest neighbors for each observation. An instance of one of the types in 
        `list_kernels()`. User-defined weighting functions can be passed by wrapping the 
        function in a [`UserDefinedKernel`](@ref) kernel (do `?NearestNeighborModels.UserDefinedKernel` for more 
        info). If observation weights `w` are passed during machine construction then the 
        weight assigned to each neighbor vote is the product of the kernel generated 
        weight for that neighbor and the corresponding observation weight.
     
    """

"""
$(MMI.doc_header(KNNClassifier))

KNNClassifier implements [K-Nearest Neighbors classifier](https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm) 
which is non-parametric algorithm that predicts a discrete class distribution associated 
with a new point by taking a vote over the classes of the k-nearest points. Each neighbor 
vote is assigned a weight based on proximity of the neighbor point to the test point 
according to a specified distance metric.

For more information about the weighting kernels, see the paper by Geler et.al 
[Comparison of different weighting schemes for the kNN classifier on time-series data](https://perun.pmf.uns.ac.rs/radovanovic/publications/2016-kais-knn-weighting.pdf). 

# Training data
In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

OR

    mach = machine(model, X, y, w)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `<:Finite` (`<:Multiclass` or `<:OrderedFactor` will do); check the scitype with `scitype(y)`

- `w` is the observation weights which can either be `nothing` (default) or an 
  `AbstractVector` whose element scitype is `Count` or `Continuous`. This is 
  different from `weights` kernel which is a model hyperparameter, see below.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

$KNNFIELDS

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew`, which
  should have same scitype as `X` above. Predictions are probabilistic but uncalibrated.

- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions
  returned above.

# Fitted parameters

$KNNFITTEDPARAMS

# Examples
```
using MLJ
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
X, y = @load_crabs; # a table and a vector from the crabs dataset
# view possible kernels
NearestNeighborModels.list_kernels()
# KNNClassifier instantiation
model = KNNClassifier(weights = NearestNeighborModels.Inverse())
mach = machine(model, X, y) |> fit! # wrap model and required data in an MLJ machine and fit
y_hat = predict(mach, X)
labels = predict_mode(mach, X)

```
See also [`MultitargetKNNClassifier`](@ref)
"""
KNNClassifier

"""
$(MMI.doc_header(MultitargetKNNClassifier))

Multi-target K-Nearest Neighbors Classifier (MultitargetKNNClassifier) is a variation of 
[`KNNClassifier`](@ref) that assumes the target variable is vector-valued with
`Multiclass` or `OrderedFactor` components. (Target data must be presented as a table, however.)

# Training data
In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

OR

    mach = machine(model, X, y, w)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any table of responses whose element scitype is either
  `<:Finite` (`<:Multiclass` or `<:OrderedFactor` will do); check the columns scitypes with `schema(y)`. 
  Each column of `y` is assumed to belong to a common categorical pool.  

- `w` is the observation weights which can either be `nothing`(default) or an 
  `AbstractVector` whose element scitype is `Count` or `Continuous`. This is different 
  from `weights` kernel which is a model hyperparameter, see below.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

$KNNFIELDS

* `output_type::Type{<:MultiUnivariateFinite}=DictColumnTable` : One of 
    (`ColumnTable`, `DictColumnTable`). The type of table type to use for predictions.
    Setting to `ColumnTable` might improve performance for narrow tables while setting to 
    `DictColumnTable` improves performance for wide tables.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew`, which
  should have same scitype as `X` above. Predictions are either a `ColumnTable` or 
  `DictColumnTable` of `UnivariateFiniteVector` columns depending on the value set for the 
  `output_type` parameter discussed above. The probabilistic predictions are uncalibrated. 

- `predict_mode(mach, Xnew)`: Return the modes of each column of the table of probabilistic 
  predictions returned above.

# Fitted parameters

$KNNFITTEDPARAMS

# Examples
```
using MLJ, StableRNGs

# set rng for reproducibility
rng = StableRNG(10)

# Dataset generation
n, p = 10, 3
X = table(randn(rng, n, p)) # feature table
fruit, color = categorical(["apple", "orange"]), categorical(["blue", "green"])
y = [(fruit = rand(rng, fruit), color = rand(rng, color)) for _ in 1:n] # target_table
# Each column in y has a common categorical pool as expected
selectcols(y, :fruit) # categorical array
selectcols(y, :color) # categorical array

# Load MultitargetKNNClassifier
MultitargetKNNClassifier = @load MultitargetKNNClassifier pkg=NearestNeighborModels

# view possible kernels
NearestNeighborModels.list_kernels()

# MultitargetKNNClassifier instantiation
model = MultitargetKNNClassifier(K=3, weights = NearestNeighborModels.Inverse())

# wrap model and required data in an MLJ machine and fit
mach = machine(model, X, y) |> fit!

# predict
y_hat = predict(mach, X)
labels = predict_mode(mach, X)

```
See also [`KNNClassifier`](@ref)
        
"""
MultitargetKNNClassifier

"""
$(MMI.doc_header(KNNRegressor))

KNNRegressor implements [K-Nearest Neighbors regressor](https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm) 
which is non-parametric algorithm that predicts the response associated with a new point 
by taking an weighted average of the response of the K-nearest points.

# Training data
In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

OR

    mach = machine(model, X, y, w)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any table of responses whose element scitype is 
    `Continuous`; check the scitype with `scitype(y)`.

- `w` is the observation weights which can either be `nothing`(default) or an 
  `AbstractVector` whoose element scitype is `Count` or `Continuous`. This is different 
  from `weights` kernel which is an hyperparameter to the model, see below.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

$KNNFIELDS

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew`, which
  should have same scitype as `X` above.

# Fitted parameters

$KNNFITTEDPARAMS

# Examples
```
using MLJ
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels
X, y = @load_boston; # loads the crabs dataset from MLJBase
# view possible kernels
NearestNeighborModels.list_kernels()
model = KNNRegressor(weights = NearestNeighborModels.Inverse()) #KNNRegressor instantiation
mach = machine(model, X, y) |> fit! # wrap model and required data in an MLJ machine and fit
y_hat = predict(mach, X)

```
See also [`MultitargetKNNRegressor`](@ref)
"""
KNNRegressor

"""
$(MMI.doc_header(MultitargetKNNRegressor))

Multi-target K-Nearest Neighbors regressor (MultitargetKNNRegressor) is a variation of 
[`KNNRegressor`](@ref) that assumes the target variable is vector-valued with
`Continuous` components. (Target data must be presented as a table, however.)

# Training data
In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

OR

    mach = machine(model, X, y, w)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any table of responses whose element scitype is 
  `Continuous`; check column scitypes with `schema(y)`.

- `w` is the observation weights which can either be `nothing`(default) or an 
  `AbstractVector` whoose element scitype is `Count` or `Continuous`. This is different 
  from `weights` kernel which is an hyperparameter to the model, see below.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

$KNNFIELDS

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew`, which
  should have same scitype as `X` above.

# Fitted parameters

$KNNFITTEDPARAMS

# Examples
```
using MLJ

# Create Data
X, y = make_regression(10, 5, n_targets=2)

# load MultitargetKNNRegressor
MultitargetKNNRegressor = @load MultitargetKNNRegressor pkg=NearestNeighborModels

# view possible kernels
NearestNeighborModels.list_kernels()

# MutlitargetKNNRegressor instantiation
model = MultitargetKNNRegressor(weights = NearestNeighborModels.Inverse())

# Wrap model and required data in an MLJ machine and fit.
mach = machine(model, X, y) |> fit! 

# Predict
y_hat = predict(mach, X)

```
See also [`KNNRegressor`](@ref)
"""
MultitargetKNNRegressor
