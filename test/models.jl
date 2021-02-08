@testset "KNNClassifier" begin
    # create something rather silly: 3 massive clusters very separated
    rng = StableRNG(10)
    n, p = 50, 3
    x, w = make_x_and_w(n, p, rng=rng)
    w1 = rand(rng, n-5)

    y1 = fill("A", n)
    y2 = fill("B", n)
    y3 = fill("C", n)
    y = categorical(vcat(y1, y2, y3))

    # generate test data
    ntest =  5
    xtest = make_xtest(ntest, p, rng=rng)
    ytest1 = fill("A", ntest)
    ytest2 = fill("B", ntest)
    ytest3 = fill("C", ntest)
    ytest = categorical(vcat(ytest1, ytest2, ytest3))

    # Test model fit
    knnc = KNNClassifier(weights=Inverse())
    f, _, _ = fit(knnc, 1, x, y)
    f2, _ , _ = fit(knnc, 1, x, y, w)
    @test_throws DimensionMismatch fit(knnc, 1, x, y, w1)
    @test f2 isa KNNResult
    @test f2.sample_weights == w
    @test fitted_params(knnc, f) == (tree=f.tree,)

    # Check predictions gotten from the model
    p = predict(knnc, f, xtest)
    p2 = predict(knnc, f2, xtest)
    @test p[1] isa UnivariateFinite
    @test p2[1] isa UnivariateFinite
    p = predict_mode(knnc, f, xtest)
    p2 = predict_mode(knnc, f2, xtest)
    @test accuracy(p, ytest) == 1.0
    @test accuracy(p2, ytest) == 1.0
end

@testset "MultitargetKNNClassifier" begin
    # create something rather silly: 3 massive clusters very separated
    rng = StableRNG(10)
    n, p = 50, 3
    x, w = make_x_and_w(n, p, rng=rng)
    y11 = fill("A", n)
    y21 = fill("B", n)
    y31 = fill("C", n)
    y12 = fill("D", n)
    y22 = fill("E", n)
    y32 = fill("F", n)
    y1 = categorical(vcat(y11, y21, y31))
    y2 = categorical(vcat(y12, y22, y32), ordered=true)
    y = (a=y1, b=y2)

    # generate test data
    ntest =  5
    xtest = make_xtest(ntest, p, rng=rng)
    ytest11 = fill("A", ntest)
    ytest21 = fill("B", ntest)
    ytest31 = fill("C", ntest)
    ytest12 = fill("D", ntest)
    ytest22 = fill("E", ntest)
    ytest32 = fill("F", ntest)
    ytest1 = categorical(vcat(ytest11, ytest21, ytest31))
    ytest2 = categorical(vcat(ytest12, ytest22, ytest32), ordered=true)
    ytest = (a=ytest1, b=ytest2)

    # Create two models, the first has an `output_type` of `DictTable` while the second
    # has an `output_type` of `ColumnTable`
    multi_knnc = MultitargetKNNClassifier(weights=Inverse())
    multi_knnc2 = MultitargetKNNClassifier(
        weights=Inverse(), output_type=ColumnTable
    )

    # Test model fit
    f, _, _ = fit(multi_knnc, 1, x, y)
    f2, _ , _ = fit(multi_knnc, 1, x, y, w)
    f3, _, _ = fit(multi_knnc2, 1, x, y)
    f4, _ , _ = fit(multi_knnc2, 1, x, y, w)
    @test f2 isa KNNResult
    @test f2.sample_weights == w
    @test f4.sample_weights == w
    @test fitted_params(multi_knnc, f) == (tree=f.tree,)

    # Check predictions gotten from the model
    p = predict(multi_knnc, f, xtest)
    p2 = predict(multi_knnc, f2, xtest)
    p3 = predict(multi_knnc2, f3, xtest)
    p4 = predict(multi_knnc2, f4, xtest)
    p5 = predict_mode(multi_knnc, f, xtest)
    p6 = predict_mode(multi_knnc, f2, xtest)
    p7 = predict_mode(multi_knnc2, f3, xtest)
    p8 = predict_mode(multi_knnc2, f4, xtest)
    for col in [:a, :b]
        @test p[col][1] isa UnivariateFinite
        @test p2[col][1] isa UnivariateFinite
        @test p3[col][1] isa UnivariateFinite
        @test p4[col][1] isa UnivariateFinite

        @test accuracy(p5[col], ytest[col]) == 1.0
        @test accuracy(p6[col], ytest[col]) == 1.0
        @test accuracy(p7[col], ytest[col]) == 1.0
        @test accuracy(p8[col], ytest[col]) == 1.0
    end
end

# the following test is a little more rigorous:
@testset "classifier sample weights" begin
    rng = StableRNG(200)
    # assign classes a, b and c randomly to 10N points on the interval:
    N = 80
    X = (x = rand(rng, 10N), );
    y = categorical(rand(rng, "abc", 10N));
    model = KNNClassifier(K=N)

    # define sample weights corresponding to class weights 2:4:1 for
    # a:b:c:
    w = map(y) do η
        if η == 'a'
            return 2
        elseif η == 'b'
            return 4
        else
            return 1
        end
    end

    f, _, _ = MLJBase.fit(model, 1, X, y, w)
    posterior3 = average([predict(model, f, X)...])

    # skewed weights gives similarly skewed posterior:
    @test abs(pdf(posterior3, 'b')/(2*pdf(posterior3, 'a'))  - 1) < 0.1
    @test abs(pdf(posterior3, 'b')/(4*pdf(posterior3, 'c'))  - 1) < 0.1
end


@testset "KNNRegressor" begin
    # create something rather silly: 3 massive clusters very separated
    rng = StableRNG(50)
    n, p = 50, 3
    x, w = make_x_and_w(n, p, rng=rng)
    y1 = fill( 0.0, n)
    y2 = fill( 2.0, n)
    y3 = fill(-2.0, n)
    y = vcat(y1, y2, y3)

    # Test model fit
    knnr = KNNRegressor(weights=Inverse())
    f, _, _ = fit(knnr, 1, x, y)
    f2, _, _ = fit(knnr, 1, x, y, w)
    @test f2 isa KNNResult
    @test fitted_params(knnr, f) == (tree=f.tree,)

    # Create test data
    ntest =  5
    xtest = make_xtest(ntest, p, rng=rng)

    # Check predictions gotten from the model
    p = predict(knnr, f, xtest)
    @test p isa AbstractVector
    p2 = predict(knnr, f2, xtest)
    @test all(p[1:ntest] .≈ 0.0)
    @test all(p[ntest+1:2*ntest] .≈ 2.0)
    @test all(p[2*ntest+1:end] .≈ -2.0)
end

@testset "MultitargetKNNRegressor" begin
    # create something rather silly: 3 massive clusters very separated
    rng = StableRNG(50)
    n, p = 50, 3
    x, w = make_x_and_w(n, p, rng=rng)
    ymat = vcat(fill( 0.0, n, 2), fill(2.0, n, 2), fill(-2.0, n, 2))
    Y = table(ymat; names = [:a, :b])

    # Test model fit
    multi_knnr = MultitargetKNNRegressor(weights=Inverse())
    f, _ , _ = fit(multi_knnr, 1, x, Y)
    f2, _, _ = fit(multi_knnr, 1, x, Y, w)
    @test fitted_params(multi_knnr, f) == (tree=f.tree,)
    @test f2 isa KNNResult
    @test f2.sample_weights == w

    # Create test data
    ntest = 5
    xtest = make_xtest(ntest, p, rng=rng)

    # Check predictions gotten from the model
    pr = predict(multi_knnr, f, xtest)
    for col in [:a, :b]
        @test all(pr[col][1:ntest] .≈ [0.0])
        @test all(pr[col][ntest+1:2*ntest] .≈ [2.0])
        @test all(pr[col][2*ntest+1:end] .≈ [-2.0])
    end
end
