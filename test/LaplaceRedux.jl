using Flux
using LaplaceRedux
using LaplaceRedux.Data: toy_data_regression
using OneHotArrays
using Plots
using TaijaData

@testset "LaplaceRedux.jl" begin

    @testset "Regression" begin
        # Data:
        x, y = toy_data_regression()
        x = Float32.(x)
        y = Float32.(y)
        xs = [[x] for x in x]
        X = permutedims(x)
        data = zip(xs, y)

        # Model:
        n_hidden = 50
        D = size(X, 1)
        nn = Chain(Dense(D, n_hidden, tanh), Dense(n_hidden, 1))

        # Fit:
        la = LaplaceRedux.Laplace(nn; likelihood = :regression)
        LaplaceRedux.fit!(la, data)

        plot(la, X, y)
        @test true
    end

    @testset "Classification" begin

        @testset "Single class" begin
            # Data:
            x, y = TaijaData.load_linearly_separable()
            x = Float32.(x)
            data = zip(eachcol(x), y)

            # Model:
            n_hidden = 50
            D = size(x, 1)
            nn = Chain(Dense(D, n_hidden, tanh), Dense(n_hidden, 1, σ))

            # Fit:
            la = LaplaceRedux.Laplace(nn; likelihood = :classification)
            LaplaceRedux.fit!(la, data)

            # Very minimal testing for basic functionality:
            plot(la, x, y)
            plot(la, x, y; zoom = -0.1f32)
            @test true
        end

        @testset "Multi-class" begin
            # Data:
            nout = 4
            x, y = TaijaData.load_blobs(centers = nout)
            x = Float32.(x)
            y = onehotbatch(y, 1:nout)
            data = zip(eachcol(x), eachcol(y))

            # Model:
            n_hidden = 50
            D = size(x, 1)
            nn = Chain(Dense(D, n_hidden, tanh), Dense(n_hidden, nout, σ))

            # Fit:
            la = LaplaceRedux.Laplace(nn; likelihood = :classification)
            LaplaceRedux.fit!(la, data)

            # Very minimal testing for basic functionality:
            y = onecold(y)
            plot(la, x, y)
            plot(la, x, y; zoom = -0.1f32)
            @test true
        end
    end

end
