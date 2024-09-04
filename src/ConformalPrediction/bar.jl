@recipe function f(
    conf_model::ConformalModel,
    fitresult,
    X
)

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