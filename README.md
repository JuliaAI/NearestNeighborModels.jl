# NearestNeighborModels
| [Linux] | Coverage | Documentation |
| :------------ | :------- | :------------ |
| [![Build Status](https://github.com/JuliaAI/NearestNeighborModels.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/NearestNeighborModels.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/NearestNeighborModels.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/NearestNeighborModels.jl?branch=master) | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaai.github.io/NearestNeighborModels.jl/dev/) |


Package providing K-nearest neighbor regressors and classifiers, for use with the [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine learning framework.

Builds on Kristoffer Carlsson's [NearestNeighbors](https://github.com/KristofferC/NearestNeighbors.jl) package, for performing efficient nearest neighbor searches.

Builds on contributions of Thibaut Lienart originally residing in [MLJModels.jl](https://github.com/JuliaAI/MLJModels.jl/blob/98618d7be53f72054de284fa1796c5292d9071bb/src/NearestNeighbors.jl#L1).

Provides the following models: `KNNRegressor`, `KNNClassifier`,
`MultitargetKNNRegressor` and `MultitargetKNNClassifier`.

Provides a library of kernels for weighting nearest neighbors, including
all kernels surveyed in the paper [Geler et.al (2016):
Comparison of different weighting schemes for the kNN classifier on
time-series
data](https://perun.pmf.uns.ac.rs/radovanovic/publications/2016-kais-knn-weighting.pdf)

Do `list_kernels()` for a complete list. 

For instructions on defining a custom kernel, do `?UserDefinedKernel`. 
