@doc raw"""
    plot(
        conf_model::ConformalProbabilisticSet,
        fitresult,
        X,
        y;
        input_var=nothing,
        target=nothing,
        ntest=50,
        zoom=-1,
        plot_set_size=false,
        plot_classification_loss=false,
        plot_set_loss=false,
        temp=0.1,
        κ=0,
        loss_matrix=UniformScaling(1.0),
    )

A `Plots.jl` recipe that can be used to visualize the conformal predictions of a fitted conformal classifier.

## Two Dimensional Inputs

Data (`X`,`y`) are plotted as dots and overlaid with predictions sets. `y` is used to indicate the ground-truth labels of samples by colour. Samples are visualized in a two-dimensional feature space, so it is expected that `X` ``\in \mathcal{R}^2``. By default, a contour is used to visualize the softmax output of the conformal classifier for the target label, where `target` indicates can be used to define the index of the target label. Transparent regions indicate that the prediction set does not include the `target` label. 

### Target

In the binary case, `target` defaults to `2`, indexing the second label: assuming the labels are `[0,1]` then the softmax output for `1` is shown. In the multi-class cases, `target` defaults to the first class: for example, if the labels are `["🐶", "🐱", "🐭"]` (in that order) then the contour indicates the softmax output for `"🐶"`.

### Set Size

If `plot_set_size` is set to `true`, then the contour instead visualises the the set size.

## Univariate and Higher Dimensional Inputs

In the case of univariate inputs or higher dimensional inputs, a stacked area plot is created: in particular, this method plots the softmax output(s) contained the the conformal predictions set on the vertical axis against an input variable `X` on the horizontal axis. In the case of multiple input variables, the `input_var` argument can be used to specify the desired input variable.

"""
@recipe function plot(
    conf_model::ConformalProbabilisticSet,
    fitresult,
    X,
    y;
    input_var=nothing,
    target=nothing,
    ntest=50,
    zoom=-1,
    plot_set_size=false,
    plot_classification_loss=false,
    plot_set_loss=false,
    temp=0.1,
    κ=0,
    loss_matrix=UniformScaling(1.0),
)

    # Get user-defined arguments:
    xlims = get(plotattributes, :xlims, nothing)
    ylims = get(plotattributes, :ylims, nothing)

    if size(permutedims(MLJBase.matrix(X)), 1) > 2

        # AREA PLOT FOR MULTI-D

        # Plot attributes:
        xtickfontsize --> 6

        # Setup:
        Xraw = deepcopy(X)
        _names = get_names(Xraw)
        X = permutedims(MLJBase.matrix(X))

        # Dimensions:
        if size(X, 1) > 1
            if isnothing(input_var)
                @info "Multiple inputs no input variable (`input_var`) specified: defaulting to first variable."
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

        # Predictions:
        ŷ = MLJBase.predict(conf_model, fitresult, Xraw)
        nout = length(levels(y))
        ŷ = (_y -> reduce(hcat, _y))(
            map(_y -> ismissing(_y) ? [0 for i in 1:nout] : pdf.(_y, levels(y)), ŷ)
        )
        ŷ = permutedims(ŷ)
        println(x)
        println(ŷ[sortperm(x), :])

        # Area chart
        args = (x, ŷ)
        data = cumsum(args[end]; dims=2)
        x = length(args) == 1 ? (axes(data, 1)) : args[1]
        seriestype := :line
        for i in axes(data, 2)
            @series begin
                fillrange := i > 1 ? data[:, i - 1] : 0
                x, data[:, i]
            end
        end

    else

        # CONTOUR PLOT FOR 2D

        # Setup:
        x1, x2, x1range, x2range, Z, xlims, ylims, _default_title = setup_contour_cp(
            conf_model,
            fitresult,
            X,
            y,
            xlims,
            ylims,
            zoom,
            ntest,
            target,
            plot_set_size,
            plot_classification_loss,
            plot_set_loss,
            temp,
            κ,
            loss_matrix,
        )

        # Contour:
        _n = length(unique(y))
        clim = (0, _n)
        @series begin
            seriestype := :contourf
            x1range, x2range, Z
        end

        # Scatter plot:
        for (i, x) in enumerate(unique(sort(y)))
            @series begin
                seriestype := :scatter
                markercolor := i
                group_idx = findall(y .== x)
                label --> "$(x)"
                x1[group_idx], x2[group_idx]
            end
        end
    end
end

function setup_contour_cp(
    conf_model,
    fitresult,
    X,
    y,
    xlims,
    ylims,
    zoom,
    ntest,
    target,
    plot_set_size,
    plot_classification_loss,
    plot_set_loss,
    temp,
    κ,
    loss_matrix,
)
    X = permutedims(MLJBase.matrix(X))

    x1 = X[1, :]
    x2 = X[2, :]

    # Plot limits:
    xlims, ylims = generate_lims(x1, x2, xlims, ylims, zoom)

    # Surface range:
    x1range = range(xlims[1]; stop=xlims[2], length=ntest)
    x2range = range(ylims[1]; stop=ylims[2], length=ntest)

    # Target
    if !isnothing(target)
        @assert target in levels(y) "Specified target does not match any of the labels."
    end
    if length(unique(y)) > 1
        if isnothing(target)
            @info "No target label supplied, using first."
        end
        target = isnothing(target) ? levels(y)[1] : target
        if plot_set_size
            _default_title = "Set size"
        elseif plot_set_loss
            _default_title = "Smooth set loss"
        elseif plot_classification_loss
            _default_title = "ℒ(C,$(target))"
        else
            _default_title = "p̂(y=$(target))"
        end
    else
        if plot_set_size
            _default_title = "Set size"
        elseif plot_set_loss
            _default_title = "Smooth set loss"
        elseif plot_classification_loss
            _default_title = "ℒ(C,$(target-1))"
        else
            _default_title = "p̂(y=$(target-1))"
        end
    end

    # Predictions
    Z = []
    for x2 in x2range, x1 in x1range
        p̂ = MLJBase.predict(conf_model, fitresult, table([x1 x2]))[1]
        if plot_set_size
            z = ismissing(p̂) ? 0 : sum(pdf.(p̂, p̂.decoder.classes) .> 0)
        elseif plot_classification_loss
            _target = categorical([target]; levels=levels(y))
            z = ConformalPrediction.ConformalTraining.classification_loss(
                conf_model, fitresult, [x1 x2], _target; temp=temp, loss_matrix=loss_matrix
            )
        elseif plot_set_loss
            z = ConformalPrediction.ConformalTraining.smooth_size_loss(
                conf_model, fitresult, [x1 x2]; κ=κ, temp=temp
            )
        else
            z = ismissing(p̂) ? [missing for i in 1:length(levels(y))] : pdf.(p̂, levels(y))
            z = replace(z, 0 => missing)
        end
        push!(Z, z)
    end
    Z = reduce(hcat, Z)
    Z = Z[findall(levels(y) .== target)[1][1], :]

    return x1, x2, x1range, x2range, Z, xlims, ylims, _default_title
end
