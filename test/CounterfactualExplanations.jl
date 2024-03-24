using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing
using CounterfactualExplanations.Models
using Plots
using TaijaData

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

@testset "CounterfactualExplanations.jl" begin
    # Very minimal testing for basic functionality:
    plot(M, counterfactual_data)
    plot(ce)
    TaijaPlotting.animate_path(ce)
    @test true
end
