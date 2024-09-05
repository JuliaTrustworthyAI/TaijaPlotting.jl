"""
    plot(
        conf_model::ConformalInterval,
        fitresult,
        X,
        y;
        input_var=nothing,
        zoom=-0.5,
        train_lab=nothing,
        test_lab=nothing,
    )

A `Plots.jl` recipe that can be used to visualize the conformal predictions of a fitted conformal regressor. Data (`X`,`y`) are plotted as dots and overlaid with predictions intervals. `y` is plotted on the vertical axis against a single variable `X` on the horizontal axis. A shaded area indicates the prediction interval. The line in the center of the interval is the midpoint of the interval and can be interpreted as the point estimate of the conformal regressor. In case `X` is multi-dimensional, `input_var` can be used to specify the input variable of interest that will be used for the horizontal axis. If unspecified, the first variable will be plotting by default.
"""
@recipe function plot(
    conf_model::ConformalInterval,
    fitresult,
    X,
    y;
    input_var=nothing,
    zoom=-0.5,
    train_lab=nothing,
    test_lab=nothing,
)

    # Get user-defined arguments:
    train_lab = isnothing(train_lab) ? "Observed" : train_lab
    test_lab = isnothing(test_lab) ? "Predicted" : test_lab
    title = get(plotattributes, :xlims, "")
    xlims = get(plotattributes, :xlims, nothing)
    ylims = get(plotattributes, :ylims, nothing)

    # Plot attributes:
    linewidth --> 1

    # Setup:
    x, y, xlims, ylims, Xraw = setup_ci(X, y, input_var, xlims, ylims, zoom)

    # Plot predictions:
    ŷ = MLJBase.predict(conf_model, fitresult, Xraw)
    lb, ub = eachcol(reduce(vcat, map(y -> permutedims(collect(y)), ŷ)))
    ymid = (lb .+ ub) ./ 2
    yerror = (ub .- lb) ./ 2
    xplot = vec(x)
    _idx = sortperm(xplot)
    @series begin
        seriestype := :path
        ribbon := (yerror, yerror)
        label := test_lab
        xplot[_idx], ymid[_idx]
    end

    # Scatter observed data:
    @series begin
        seriestype := :scatter
        label := train_lab
        vec(x), vec(y)
    end
end

function setup_ci(X, y, input_var, xlims, ylims, zoom)
    Xraw = deepcopy(X)
    _names = get_names(Xraw)
    X = permutedims(MLJBase.matrix(X))

    # Dimensions:
    if size(X, 1) > 1
        if isnothing(input_var)
            @info "Multivariate input for regression with no input variable (`input_var`) specified: defaulting to first variable."
            idx = 1
        else
            if typeof(input_var) == Int
                idx = input_var
            else
                @assert input_var ∈ _names "$(input_var) is not among the variable names of `X`."
                idx = findall(_names .== input_var)[1]
            end
        end
        x = X[idx, :]
    else
        idx = 1
        x = X
    end

    # Plot limits:
    xlims, ylims = generate_lims(x, y, xlims, ylims, zoom)

    return x, y, xlims, ylims, Xraw
end
