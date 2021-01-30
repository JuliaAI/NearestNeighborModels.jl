@testset "kernels" begin
    rng = StableRNG(25)
    v = [
        [3.0, 4.0, 5.0],
        [2.0, 5.0, 5.0],
        [1.0, 4.0, 5.0],
        [2.0, 2.0, 5.0],
        [1.0, 1.0, 3.0],
    ]

    v1 = [sort!(rand(rng, 1.0:5.0, 3)) for _ in 1:5]
    dists_matrix = NeighDistsMatrix(v)
    
    # User Defined Kernel (UDK)
    kernel  = UDK(func=identity, sort=true)
    @test sort_idxs(kernel) == true
    @test get_weights(kernel, dists_matrix) == dists_matrix
    kernel  = UDK(func=x->rand(rng, 3, 5), sort=true)
    @test_throws DimensionMismatch get_weights(kernel, dists_matrix)
    kernel  = UDK(func=x->rand(rng, "String", 5, 3), sort=true)
    @test_throws Exception  get_weights(kernel, dists_matrix)
    
    # Uniform Kernel (Uniform)
    kernel = Uniform()
    @test sort_idxs(kernel) == false
    @test get_weights(kernel, dists_matrix) === nothing
    
    # Dudani
    kernel = Dudani()
    @test sort_idxs(kernel) == true
    @test get_weights(
        kernel, dists_matrix
    ) == [1.0 0.5 0.0;1.0 0.0 0.0;1.0 0.25 0.0;1.0 1.0 0.0;1.0 1.0 0.0]
    
    # Inverse
    kernel = Inverse()
    @test sort_idxs(kernel) == false
    dists_matrix2 = NeighDistsMatrix(deepcopy(v))
    dists_matrix2[1,1] = 0
    dists_matrix2[3,3] = 0
    @test get_weights(
        kernel, dists_matrix2
    ) == [1.0 0.0 0.0;0.5 0.2 0.2;0.0 0.0 1.0;0.5 0.5 0.2;1.0 1.0 1/3]
    
    # Inverse Squared (ISquared)
    kernel = ISquared()
    @test sort_idxs(kernel) == false
    dists_matrix3 = NeighDistsMatrix(deepcopy(v))
    dists_matrix3[1,1] = 0
    dists_matrix3[3,3] = 0
    @test get_weights(
        kernel, dists_matrix3
    ) == [1.0 0.0 0.0;0.25 0.04 0.04;0.0 0.0 1.0;0.25 0.25 0.04;1.0 1.0 1/9]
    
    # Rank
    kernel = Rank()
    @test sort_idxs(kernel) == true
    @test get_weights(
        kernel, dists_matrix
    ) == repeat(collect(size(dists_matrix, 2):-1.0:1.0)', size(dists_matrix, 1)) 
    
    # Macleod
    kernel = Macleod(;a=1.5)
    @test sort_idxs(kernel) == true
    @test get_weights(
        kernel, dists_matrix
    ) == [4.0 3.2 2.4;9.0   5.4  5.4;16.0 11.2 9.6;9.0 9.0 5.4;4.0 4.0 2.4]
    
    # Zavreal
    kernel = Zavreal(a=0.5, b=1.5)
    @test sort_idxs(kernel) == false
    @test get_weights(
        kernel, dists_matrix
    ) == exp.(kernel.a .* dists_matrix .^ kernel.b)
    
    # ReciprocalRank
    kernel = ReciprocalRank()
    @test sort_idxs(kernel) == true
    @test get_weights(
        kernel, dists_matrix
    ) == repeat(1 ./ collect(1:size(dists_matrix, 2))', size(dists_matrix, 1)) 
    
    # DualU
    kernel = DualU()
    @test sort_idxs(kernel) == true
    @test get_weights(
        kernel, dists_matrix
    ) == [1.0 0.5 0.0;1.0 0.0 0.0;1.0 0.25 0.0;1.0 1.0 0.0;1.0 1.0 0.0]
    # DualD
    kernel = DualD()
    @test sort_idxs(kernel) == true
    @test get_weights(
        kernel, dists_matrix
    ) == [1.0 0.5 0.0;1.0 0.0 0.0;1.0 0.25 0.0;1.0 1.0 0.0;1.0 1.0 0.0]
    # Fibonacci
    kernel = Fibonacci()
    @test sort_idxs(kernel) == true
    @test get_weights(
        kernel, dists_matrix
    ) == repeat([2 1 1], size(dists_matrix, 1)) 
end
