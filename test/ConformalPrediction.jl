using ConformalPrediction
using MLJBase
using MLJLinearModels
using Plots

isplot(plt) = typeof(plt) <: Plots.Plot

@testset "ConformalPrediction.jl" begin
    @testset "Classification" begin

        # Data:
        X, y = make_moons(500; noise=0.15)
        X = MLJBase.table(convert.(Float32, MLJBase.matrix(X)))
        train, test = partition(eachindex(y), 0.8; shuffle=true)

        # Model:
        model = MLJLinearModels.LogisticClassifier()
        conf_model = conformal_model(model; coverage=0.9)
        mach = machine(conf_model, X, y)
        fit!(mach; rows=train)

        @test isplot(bar(mach.model, mach.fitresult, X))
        @test isplot(plot(mach.model, mach.fitresult, X, y))
        @test isplot(plot(mach.model, mach.fitresult, X, y; input_var=1))
        @test isplot(plot(mach.model, mach.fitresult, X, y; input_var=:x1))
        @test isplot(contourf(mach.model, mach.fitresult, X, y))
        @test isplot(
            contourf(mach.model, mach.fitresult, X, y; zoom=-1, plot_set_size=true)
        )
        @test isplot(
            contourf(mach.model, mach.fitresult, X, y; zoom=-1, plot_set_loss=true)
        )
        @test isplot(
            contourf(
                mach.model, mach.fitresult, X, y; zoom=-1, plot_classification_loss=true
            ),
        )
        @test isplot(contourf(mach.model, mach.fitresult, X, y; target=1))
    end

    @testset "Regression" begin

        # Data:
        X, y = make_regression(500)
        X = MLJBase.table(convert.(Float32, MLJBase.matrix(X)))
        train, test = partition(eachindex(y), 0.8; shuffle=true)

        # Model:
        model = MLJLinearModels.LinearRegressor()
        conf_model = conformal_model(model; coverage=0.9)
        mach = machine(conf_model, X, y)
        fit!(mach; rows=train)

        # Plotting:
        @test isplot(plot(mach.model, mach.fitresult, X, y))
        @test isplot(
            plot(
                mach.model, mach.fitresult, X, y; input_var=1, xlims=(-1, 1), ylims=(-1, 1)
            ),
        )
        @test isplot(plot(mach.model, mach.fitresult, X, y; input_var=:x1))
        @test isplot(plot(mach.model, mach.fitresult, X, y; target=1, plot_set_size=true))
        @test isplot(plot(mach.model, mach.fitresult, X, y; target=1, plot_set_loss=true))
        @test isplot(
            plot(mach.model, mach.fitresult, X, y; target=1, plot_classification_loss=true)
        )
        @test isplot(bar(mach.model, mach.fitresult, X))
    end
end
