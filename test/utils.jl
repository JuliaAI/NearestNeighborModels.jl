@testset "err_if_given_invalid_K" begin
    @test err_if_given_invalid_K(5) == 5
    @test_throws ArgumentError err_if_given_invalid_K(0)
    @test_throws ArgumentError err_if_given_invalid_K(-5)
end

@testset "replace!" begin
    y = [1, 2, 2, 2, 3]
    z1 = 1:5
    z2 = 0:3
    r1 = -2:2
    r2 = 2:5    
    @test_throws DimensionMismatch _replace!(y, z1, r2)
    @test_throws DimensionMismatch _replace!(y, z2, r1)
    @test _replace!(deepcopy(y), z1, r1) == Base.replace!(deepcopy(y), (z1 .=> r1)...)
    @test _replace!(deepcopy(y), z2, r2) == Base.replace!(deepcopy(y), (z2 .=> r2)...)
end

@testset "combine_weights" begin
    rng = StableRNG(100)
    knn_weights = rand(rng, 5, 3)
    sample_weights = rand(rng, 10)
    v = [sort!(rand(rng, 1:5, 3)) for _ in 1:5]
    idxs_matrix = NeighIndsMatrix(v)  
    @test combine_weights(idxs_matrix, knn_weights, nothing) === knn_weights
    @test combine_weights(idxs_matrix, nothing, nothing) == Fill(1, size(idxs_matrix))
    @test combine_weights(idxs_matrix, nothing, sample_weights) == sample_weights[idxs_matrix]
    @test combine_weights(
        idxs_matrix,
        knn_weights,
        sample_weights
    ) == (knn_weights .* sample_weights[idxs_matrix])
    
end

@testset "_sum" begin
    rng = StableRNG(15)
    A = rand(rng, 5, 3)
    fill_value = 2
    fill_dims = (5, 3)
    x = Fill(fill_value, fill_dims)
    
    @test _sum(A, dims=:)  == sum(A, dims=:)
    @test _sum(A, dims=1)  == sum(A, dims=1)
    @test _sum(A, dims=2)  == sum(A, dims=2)
    @test _sum(A, dims=3)  == sum(A, dims=3)
    @test  _sum(x, dims=:) == fill_value * (*(fill_dims...))
    @test _sum(x, dims=1) == Fill(fill_value * fill_dims[1], (1, fill_dims[2]))
    @test _sum(x, dims=2) == Fill(fill_value * fill_dims[2], (fill_dims[1], 1))
    @test _sum(x, dims=3) === x
end

@testset "scale" begin
    rng = StableRNG(25)
    A = rand(rng, 5, 3)
    B = rand(rng, Int, 5, 3)
    
    # Create 4 copies due to mutation
    A1 = deepcopy(A)
    A2 = deepcopy(A)
    A3 = deepcopy(A)
    A4 = deepcopy(A)
    
    @test scale(A1, dims=:) === (A1 ./= sum(A1, dims=:))
    @test scale(A2, dims=1) === (A2 ./= sum(A2, dims=1))
    @test scale(A3, dims=2) === (A3 ./= sum(A3, dims=2))
    @test scale(A4, dims=3) === (A4 ./= sum(A4, dims=3))
    
    @test scale(B, dims=:) == (B ./ sum(B, dims=:))
    @test scale(B, dims=1) == (B ./ sum(B, dims=1))
    @test scale(B, dims=2) == (B ./ sum(B, dims=2))
    @test scale(B, dims=3) == (B ./ sum(B, dims=3))
    @test !(scale(B, dims=3) === (B ./ sum(B, dims=3)))
end

@testset "matrices" begin
    # create matrices to test on.
    rng = StableRNG(17)
    n, p = (5, 3)
    vdists = [rand(rng, 3) for _ in 1:5]
    vidxs = [rand(rng, UInt, 3) for _ in 1:5]
    vr1 = rand(rng, p)
    r1matrix = RankOneMatrix(vr1, n)
    idxs_matrix = NeighIndsMatrix(vidxs)
    dists_matrix = NeighDistsMatrix(vdists)
    
    @test idxs_matrix isa NeighIndsMatrix
    @test dists_matrix isa NeighDistsMatrix
    @test r1matrix isa RankOneMatrix
    
    # check size
    @test size(idxs_matrix) == (n, p)
    @test size(dists_matrix) == (n, p)
    @test size(r1matrix) == (n, p)
    
    # check `getindex` and `setindex!`
    i = rand(rng, 1:n)
    j = rand(rng, 1:p)
    
    @test idxs_matrix[i, j] == vidxs[i][j]
    idxs_matrix[i, j] = 0
    @test idxs_matrix[i, j] == 0
    @test view(idxs_matrix, i, :) == vidxs[i]
    @test_throws BoundsError view(idxs_matrix, 6, :)
    @test_throws BoundsError idxs_matrix[i, 4]
    @test_throws BoundsError (idxs_matrix[i, 4] = 0)
    
    @test dists_matrix[i, j] == vdists[i][j]
    dists_matrix[i, j] = 0
    @test dists_matrix[i, j] == 0
    @test view(dists_matrix, i, :) == vdists[i]
    @test_throws BoundsError view(dists_matrix, 6, :)
    @test_throws BoundsError dists_matrix[i, 4]
    @test_throws BoundsError (dists_matrix[i, 4] = 0)
    
    @test r1matrix[i, j] == vr1[j]
    @test_throws BoundsError r1matrix[i, 4]
    @test_throws BoundsError r1matrix[8, j]
    
    # check `Base.unaliascopy`
    unaliascopy_idxs = Base.unaliascopy(idxs_matrix)
    @test unaliascopy_idxs == NeighIndsMatrix(
        [copy(@inbounds(vidxs[i])) for i in eachindex(vidxs)]
    )
    @test !(unaliascopy_idxs === idxs_matrix)
    
    unaliascopy_dists = Base.unaliascopy(dists_matrix)
    @test unaliascopy_dists == NeighDistsMatrix(
        [copy(@inbounds(vdists[i])) for i in eachindex(vdists)]
    )
    @test !(unaliascopy_dists === dists_matrix)
    
    unaliascopy_r1 = Base.unaliascopy(r1matrix)
    @test unaliascopy_r1 == RankOneMatrix(copy(vr1), n)
    @test !(unaliascopy_r1 === r1matrix)
    
    # check `IndexStyle`
    @test Base.IndexStyle(idxs_matrix) === Base.IndexCartesian()
    @test Base.IndexStyle(dists_matrix) === Base.IndexCartesian()
    @test Base.IndexStyle(r1matrix) === Base.IndexCartesian()
end

@testset "check_one_based_indexing" begin
    rng = StableRNG(559)
    x = rand(rng, 5)
    X = rand(rng, 5, 3)
    x1 = OffsetArray(x, 0:4)
    X1 = OffsetArray(X, 0:4, 0:2)
    
    @test check_onebased_indexing("input", x) === nothing
    @test check_onebased_indexing("input", X) === nothing
    @test_throws ArgumentError check_onebased_indexing("input", x1)
    @test_throws ArgumentError check_onebased_indexing("input", X1)
end

