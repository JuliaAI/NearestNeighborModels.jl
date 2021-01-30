function make_x_and_w(n, p; rng)
    x1 = randn(rng, n, p)
    x2 = randn(rng, n, p) .+ 5
    x3 = randn(rng, n, p) .- 5
    x = table(vcat(x1, x2, x3))
    w = abs.(50 * randn(rng, nrows(x)))
    return x, w
end

function make_xtest(ntest, p; rng)
    xtest1 = randn(rng, ntest, p)
    xtest2 = randn(rng, ntest, p) .+ 5
    xtest3 = randn(rng, ntest, p) .- 5
    xtest = table(vcat(xtest1, xtest2, xtest3))
    return xtest
end
