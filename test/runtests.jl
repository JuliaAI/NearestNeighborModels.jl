using NearestNeighborModels
using Distances
using StableRNGs
using Test
using MLJBase
using OffsetArrays

import NearestNeighborModels: check_onebased_indexing, combine_weights,
    err_if_given_invalid_K, Fill, get_weights, KNNResult, NeighDistsMatrix,
    NeighIndsMatrix, RankOneMatrix, _replace!, scale, sort_idxs, _sum

const NN = NearestNeighborModels.NearestNeighbors

include("testutils.jl")

@testset "NearestNeighborModels.jl" begin
    @testset "models" begin
        include("models.jl")
    end
    @testset "utils" begin
        include("utils.jl")
    end
    @testset "kernels" begin
        include("kernels.jl")
    end
end
