module TaijaPlotting

using CategoricalArrays
using CounterfactualExplanations
using Flux
using MLJBase
using MultivariateStats
using Plots

export animate_path

include("ConformalPrediction/ConformalPrediction.jl")
include("CounterfactualExplations/CounterfactualExplanations.jl")
include("LaplaceRedux/LaplaceRedux.jl")

end
