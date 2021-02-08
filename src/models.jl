##########################
### MultitargetKNNClassifier
##########################

"""
MultitargetKNNClassifier(;kwargs...)

$KNNClassifierDescription

$MultitargetKNNClassifierFields
"""
@mlj_model mutable struct MultitargetKNNClassifier <: MMI.Probabilistic
    K::Int = 5::(_ > 0)
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :brutetree, :balltree))
    metric::Metric = Euclidean()
    leafsize::Int = 10::(_ ≥ 0)
    reorder::Bool = true
    weights::KNNKernel = Uniform()
    output_type::Type{<:MultiUnivariateFinite} = DictTable
end

function _predictmode_knnclassifier(weights, y, idxsvec)
    return StatsBase.mode.(_predict_knnclassifier(weights, y, idxsvec))
end

function dict_preds(::Val{:columnaccess}, func, target_table, idxsvec, weights)
    cols = Tables.columns(target_table)
    colnames = Tables.columnnames(cols)
    dict_table = Dict(
        nm => func(weights, Tables.getcolumn(cols, nm), idxsvec) for nm in colnames
    )
    return dict_table
end

function dict_preds(::Val{:noncolumnaccess}, func, target_table, idxsvec, weights)
     cols = Tables.dictcolumntable(target_table)
     colnames = Tables.columnnames(cols)
     dict_table = Dict(
        nm => func(weights, Tables.getcolumn(cols, nm), idxsvec) for nm in colnames
     )
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
    table = if T <: DictTable
        dict_preds(_predict_knnclassifier, target_table, idxsvec, weights)
    else
        ntuple_preds(_predict_knnclassifier, target_table, idxsvec, weights)
    end
    return table
end

function categorical_table(::Type{T}, weights, target_table, idxsvec) where {T}
    table = if T <: DictTable
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

"""
KNNClassifier(;kwargs...)

$KNNClassifierDescription

$KNNFields
"""
@mlj_model mutable struct KNNClassifier <: MMI.Probabilistic
    K::Int = 5::(_ > 0)
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :brutetree, :balltree))
    metric::Metric = Euclidean()
    leafsize::Int = 10::(_ ≥ 0)
    reorder::Bool = true
    weights::KNNKernel = Uniform()
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
    classes_seen = filter(in(unique(y)), classes)
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

"""
KNNRegressoor(;kwargs...)

$KNNRegressorDescription

$KNNFields
"""
@mlj_model mutable struct KNNRegressor <: MMI.Deterministic
    K::Int = 5::(_ > 0)
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :brutetree, :balltree))
    metric::Metric = Euclidean()
    leafsize::Int = 10::(_ ≥ 0)
    reorder::Bool = true
    weights::KNNKernel = Uniform()
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
    preds = @view(_sum(labels .* weights, dims=2)[:]) ./ _sum(weights, dims=2)
    return preds
end
function MMI.predict(m::KNNRegressor, fitresult, X)
    err_if_given_invalid_K(m.K)
    #Xmatrix = MMI.matrix(X, transpose=true) # NOTE: copies the data
    Xmatrix = transpose(MMI.matrix(X))
    check_onebased_indexing("prediction input", Xmatrix)
    predict_args = setup_predict_args(m, Xmatrix, fitresult)
    preds = _predict_knnreg(predict_args...) |> vec
    return preds
end

##########################
### MultitargetKNNRegressor
##########################

"""
MultitargetKNNRegressor(;kwargs...)

$KNNRegressorDescription

$KNNFields
"""
@mlj_model mutable struct MultitargetKNNRegressor <: MMI.Deterministic
    K::Int = 5::(_ > 0)
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :brutetree, :balltree))
    metric::Metric = Euclidean()
    leafsize::Int = 10::(_ ≥ 0)
    reorder::Bool = true
    weights::KNNKernel = Uniform()
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

##################
### FIT MODELS
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

const KNN = Union{
    KNNClassifier, KNNRegressor, MultitargetKNNRegressor, MultitargetKNNClassifier
}

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
    if m.algorithm == :kdtree
        tree = NN.KDTree(Xmatrix; leafsize=m.leafsize, reorder=m.reorder)
    elseif m.algorithm == :balltree
        tree = NN.BallTree(Xmatrix; leafsize=m.leafsize, reorder=m.reorder)
    elseif m.algorithm == :brutetree
        tree = NN.BruteTree(Xmatrix; leafsize=m.leafsize, reorder=m.reorder)
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
    weights = true,
    descr = KNNRegressorDescription,
    path = "$(PKG).KNNRegressor"
)

metadata_model(
    KNNClassifier,
    input = Table(Continuous),
    target = Vec{<:Finite},
    weights = true,
    descr = KNNClassifierDescription,
    path = "$(PKG).KNNClassifier"
)

metadata_model(
    MultitargetKNNClassifier,
    input = Table(Continuous),
    target = Table(Finite),
    weights = true,
    descr = KNNClassifierDescription,
    path = "$(PKG).MultitargetKNNClassifier"
)

metadata_model(
    MultitargetKNNRegressor,
    input = Table(Continuous),
    target = Table(Continuous),
    weights = true,
    descr = KNNRegressorDescription,
    path = "$(PKG).MultitargetKNNRegressor"
)
