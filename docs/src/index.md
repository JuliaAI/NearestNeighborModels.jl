# NearestNeighborModels - Docs

NearestNeighborModels is a julia package providing implemtation of various 
k-nearest-neighbor classifiers and regressors models for use with 
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine learning 
framework. It also provides users with an array of weighting kernels to choose 
from for prediction.

NearestNeighborModels builds on Kristoffer Carlsson's 
[NearestNeighbors](https://github.com/KristofferC/NearestNeighbors.jl) package(for 
performing efficient nearest neighbor searches) and earlier contributions from Thibaut 
Lienart originally residing in 
[MLJModels.jl](https://github.com/alan-turing-institute/MLJModels.jl/blob/98618d7be53f72054de284fa1796c5292d9071bb/src/NearestNeighbors.jl#L1).


# Installation

On a Julia>=1.0 NearestNeighborModels can be added via Pkg
as shown below.

```julia
using Pkg
Pkg.add("NearestNeighborModels") 
```

# Usage

To use any model implemented in this package, the model must first be wrapped in an MLJ 
machine alongside the required data. Users also get additional features from MLJ including 
performance evaluation, hyper-parameter tuning, stacking etc.
The following example shows how to train a `KNNClassifier` on the crabs dataset.

```julia
using NearestNeighborModels, MLJBase
X, y = @load_crabs; # loads the crabs dataset from MLJBase
train_inds, test_inds = partition(1:nrows(X), 0.7, shuffle=false);
knnc = KNNClassifier(weights = Inverse()) # KNNClassifier instantiation
knnc_mach = machine(knnc, X, y) # wrap model and required data in an MLJ machine
fit!(knnc_mach, rows=train_inds) # train machine on a subset of the wrapped data `X`
```
`UnivariateFinite` predictions can be obtained from the trained machine as shown below
```@meta
DocTestSetup = quote
    using NearestNeighborModels, MLJBase
    X, y = @load_crabs;
    train_inds, test_inds = partition(1:nrows(X), 0.7, shuffle=false);
    knnc = KNNClassifier(weights = Inverse())
    knnc_mach = machine(knnc, X, y)
    fit!(knnc_mach, rows=train_inds)
end
```
```jldoctest ex1
julia> predict(knnc_mach, rows=test_inds)
60-element UnivariateFiniteVector{Multiclass{2}, String, UInt32, Float64}:
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>0.315, O=>0.685)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 ⋮
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>0.613, O=>0.387)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>0.83, O=>0.17)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>0.361, O=>0.639)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>0.824, O=>0.176)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
 UnivariateFinite{ScientificTypesBase.Multiclass{2}}(B=>1.0, O=>0.0)
```
Alternatively categorical predictions may be obtained using `predict_mode` as shown below.
```jldoctest ex1
julia> predict_mode(knnc_mach, rows=test_inds)
60-element CategoricalArrays.CategoricalArray{String,1,UInt32}:
 "O"
 "B"
 "B"
 "B"
 "B"
 "B"
 "B"
 "B"
 "B"
 "B"
 ⋮
 "B"
 "B"
 "O"
 "B"
 "B"
 "B"
 "B"
 "B"
 "B"

```
```@meta
DocTestSetup = nothing
```

Users who need to implement their weighting kernel may do so by wrapping the kernel function in a `UserDefinedKernel`. The following code shows how to define a `UserDefinedKernel` that assigns weights in such a way that the closest neighbor in each row of `dists` is assigned a weight of `2` while other neighbors in the same row are assigned equal weights of `1`.
```julia
# First we define the kernel function
function custom_kernel(dists::AbstractMatrix)
    # eltype of weights must be `<:AbstractFloat`
    weights = similar(Array{Float16}, size(dists))
    weights[:, 1] .= 2.0
    weights[:, 2:end] .= 1.0
    return weights
end

# Then we wrap it in a `UserDefinedKernel`
# `sort = true` because our `custom_kernel` function relies on `dists` being sorted in 
# ascending order.
weighting_kernel = UserDefinedKernel(func=custom_kernel, sort=true)
```
We will now train a `MultitargetKNNRegressor` that makes use of our simple custom-defined 
`weighting_kernel` for prediction.
```julia
using NearestNeighborModels, MLJBase
using StableRNGs #for reproducibility of this example

n = 50
p = 5
l = 2
rng = StableRNG(100)
# `table` converts an `AbstractMatrix` into a `Tables.jl` compactible table
X = table(randn(rng, (n, p))) # feature table
Y = table(randn(rng, (n, l))) # target table

train_inds, test_inds = partition(1:nrows(X), 0.8, shuffle=false);
multi_knnr = MultitargetKNNRegressor(weights=weighting_kernel)
multi_knnr_mach = machine(multi_knnr, X, Y) #wrap model and required data in an MLJ machine
fit!(multi_knnr_mach, rows=train_inds) # train machine on a subset of the wrapped data `X`
```
And of course predicting with the test-dataset gives:
```@meta
DocTestSetup = quote
    using NearestNeighborModels, MLJBase, StableRNGs
    function custom_kernel(dists::AbstractMatrix)
        weights = similar(Array{Float16}, size(dists))
        weights[:, 1] .= 2.0
        weights[:, 2:end] .= 1.0
        return weights
    end
    weighting_kernel = UserDefinedKernel(func=custom_kernel, sort=true)
    n = 50
    p = 5
    l = 2
    rng = StableRNG(100)
    X = table(randn(rng, (n, p))) # feature table
    Y = table(randn(rng, (n, l))) # target table
    train_inds, test_inds = partition(1:nrows(X), 0.8, shuffle=false);
    multi_knnr = MultitargetKNNRegressor(weights=weighting_kernel)
    multi_knnr_mach = machine(multi_knnr, X, Y) #wrap model and required data in an MLJ machine
    fit!(multi_knnr_mach, rows=train_inds) # train machine on a subset of the wrapped data `X`
end
```
```jldoctest
julia> table_predictions = predict(multi_knnr_mach, rows=test_inds)
Tables.MatrixTable{Matrix{Float64}} with 10 rows, 2 columns, and schema:
 :x1  Float64
 :x2  Float64

julia> MLJBase.matrix(table_predictions)
10×2 Matrix{Float64}:
  0.0867655   0.286866
  0.388722    0.11164
 -0.0572281   0.752707
 -0.0845422   0.125039
  0.308204   -0.0840635
  0.170179   -0.0136838
  0.0916878   0.168172
  0.909645   -0.788285
 -0.35767     0.158968
  0.0352492  -0.30166
```
```@meta
DocTestSetup = nothing
```
see [MLJ docs](https://alan-turing-institute.github.io/MLJ.jl/dev/) for help on additional 
features such as hyper-parameter tuning, performance evaluation, stacking etc.
