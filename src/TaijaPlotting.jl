module TaijaPlotting

using CategoricalArrays
using CounterfactualExplanations
using MLJBase
using MultivariateStats
using OneHotArrays
using Plots
using RecipesBase

export animate_path

include("ConformalPrediction/ConformalPrediction.jl")
include("CounterfactualExplations/CounterfactualExplanations.jl")
include("LaplaceRedux/LaplaceRedux.jl")
export calibration_plot

end
