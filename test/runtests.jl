using Plots
using TaijaPlotting
using Test

@testset "TaijaPlotting.jl" begin
    include("ConformalPrediction.jl")
    include("CounterfactualExplanations.jl")
    include("LaplaceRedux.jl")
end
