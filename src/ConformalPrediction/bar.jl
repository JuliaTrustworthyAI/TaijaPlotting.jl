"""
    plot(
        conf_model::ConformalModel,
        fitresult,
        X
    )

A `Plots.jl` recipe that can be used to visualize the set size distribution of a conformal predictor. In the regression case, prediction interval widths are stratified into discrete bins. It can be useful to plot the distribution of set sizes in order to visually asses how adaptive a conformal predictor is. For more adaptive predictors the distribution of set sizes is typically spread out more widely, which reflects that “the procedure is effectively distinguishing between easy and hard inputs”. This is desirable: when for a given sample it is difficult to make predictions, this should be reflected in the set size (or interval width in the regression case). Since ‘difficult’ lies on some spectrum that ranges from ‘very easy’ to ‘very difficult’ the set size should vary across the spectrum of ‘empty set’ to ‘all labels included’.
"""
@recipe function plot(conf_model::ConformalModel, fitresult, X)

    # Plot attributes:
    xtickfontsize --> 6

    # Setup:
    ŷ = MLJBase.predict(conf_model, fitresult, X)
    idx = ConformalPrediction.size_indicator(ŷ)
    x = sort(levels(idx); lt=natural)
    y = [sum(idx .== _x) for _x in x]

    # Bar chart
    @series begin
        seriestype := :bar
        label --> ""
        x, y
    end
end
