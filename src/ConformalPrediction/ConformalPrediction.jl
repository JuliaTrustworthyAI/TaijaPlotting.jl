using CategoricalArrays
using ConformalPrediction
using ConformalPrediction: ConformalProbabilisticSet, ConformalInterval
using LinearAlgebra
using NaturalSort
using Plots

"""
    generate_lims(x1, x2, xlims, ylims)

Small helper function then generates the `xlims` and `ylims` for the plot.
"""
function generate_lims(x1, x2, xlims, ylims, zoom)
    if isnothing(xlims)
        xlims = (minimum(x1), maximum(x1)) .+ (zoom, -zoom)
    else
        xlims = xlims .+ (zoom, -zoom)
    end
    if isnothing(ylims)
        ylims = (minimum(x2), maximum(x2)) .+ (zoom, -zoom)
    else
        ylims = ylims .+ (zoom, -zoom)
    end
    return xlims, ylims
end

"""
    get_names(X)
Helper function to get variables names of `X`.
"""
function get_names(X)
    try
        global _names = MLJBase.schema(X).names
    catch
        X = MLJBase.table(X)
        global _names = MLJBase.schema(X).names
    end
    return _names
end

include("regression.jl")
include("bar.jl")
include("classification.jl")