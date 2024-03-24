using Flux
using LaplaceRedux
using LaplaceRedux.Data: toy_data_regression
using Plots
using TaijaData

@testset "LaplaceRedux.jl" begin

    @testset "Regression" begin
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

        @testset begin
            # Very minimal testing for basic functionality:
            plot(la, X, y)
            @test true
        end
    end

    @testset "Classification" begin
        # Data:
        x, y = TaijaData.load_linearly_separable()
        x = Float32.(x)
        data = zip(eachcol(x), y)

        # Model:
        n_hidden = 50
        D = size(x, 1)
        nn = Chain(Dense(D, n_hidden, tanh), Dense(n_hidden, 1, σ))

        # Fit:
        la = Laplace(nn; likelihood = :classification)
        LaplaceRedux.fit!(la, data)

        @testset begin
            # Very minimal testing for basic functionality:
            plot(la, x, y)
            @test true
        end
    end

end
