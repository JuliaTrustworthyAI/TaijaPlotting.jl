using Flux
using LaplaceRedux
using LaplaceRedux.Data: toy_data_regression
using Plots

# Data:
x, y = toy_data_regression()
xs = [[x] for x in x]
X = permutedims(x)
data = zip(xs, y)

# Model:
n_hidden = 50
D = size(X, 1)
nn = Chain(Dense(D, n_hidden, tanh), Dense(n_hidden, 1))

# Fit:
la = Laplace(nn; likelihood = :regression)
LaplaceRedux.fit!(la, data)

@testset "LaplaceRedux.jl" begin
    # Very minimal testing for basic functionality:
    plot(la, X, y)
    @test true
end
