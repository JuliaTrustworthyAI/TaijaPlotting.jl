using Plots
using TaijaPlotting
using Test

@testset "TaijaPlotting.jl" begin
    include("aqua.jl")
    include("ConformalPrediction.jl")
    include("CounterfactualExplanations.jl")
    include("LaplaceRedux.jl")
end
