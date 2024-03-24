using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing
using CounterfactualExplanations.Models
using Plots
using TaijaData

@testset "CounterfactualExplanations.jl" begin

    @testset "2D" begin

        # Counteractual data and model:
        counterfactual_data = CounterfactualData(TaijaData.load_linearly_separable()...)
        M = fit_model(counterfactual_data, :Linear)
        target = 2
        factual = 1
        chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
        x = select_factual(counterfactual_data, chosen)

        # Search:
        generator = GenericGenerator()
        ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

        plot(M, counterfactual_data)
        plot(ce)
        plot(ce; plot_proba = true, zoom = -0.1f32)
        TaijaPlotting.animate_path(ce)
    end

    @testset "Multi-dim" begin
        # Counteractual data and model:
        counterfactual_data = CounterfactualData(TaijaData.load_blobs(; k = 5)...)
        M = fit_model(counterfactual_data, :Linear)
        target = 2
        factual = 1
        chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
        x = select_factual(counterfactual_data, chosen)

        # Search:
        generator = GenericGenerator()
        ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

        # Plots:
        plot(M, counterfactual_data)
        plot(ce)
        TaijaPlotting.animate_path(ce)
    end

    @test true
end
