"""
    plot(
        ce::CounterfactualExplanation;
        target=nothing,
        length_out=100,
        zoom=-0.1,
        dim_red=:pca,
        plot_loss=false,
        loss_fun=nothing,
        plot_up_to = nothing,
        n_points = nothing,
    )

Calling `Plots.plot` on a `CounterfactualExplanation` object will plot the training data (scatters), model predictions for the specified `target` (contour) and the counterfactual path (scatter).
"""
@recipe function plot(
    ce::CounterfactualExplanation;
    target=nothing,
    length_out=100,
    zoom=-0.1,
    dim_red=:pca,
    plot_loss=false,
    loss_fun=nothing,
    plot_up_to=nothing,
    n_points=nothing,
)
    if !isnothing(n_points)
        if n_points < size(ce.data.X, 2)
            @info "Undersampling to $(n_points) points."
        else
            @info "Oversampling to $(n_points) points."
        end
        xlims, ylims = extrema(ce.data.X[1, :]), extrema(ce.data.X[2, :])
        ce = deepcopy(ce)
        ce.data = DataPreprocessing.subsample(ce.data, n_points)
    else
        xlims, ylims = nothing, nothing
    end

    # Asserts
    @assert !plot_loss || !isnothing(loss_fun) "Need to provide a loss function to plot the loss, e.g. (`loss_fun=Flux.Losses.logitcrossentropy`)."

    # Get user-defined arguments:
    xlims = get(plotattributes, :xlims, xlims)
    ylims = get(plotattributes, :ylims, ylims)
    ms = get(plotattributes, :markersize, 3)
    mspath = ms * 2
    msfinal = mspath * 2

    # Plot attributes
    linewidth --> 0.1

    contour_series, X, y, xlims, ylims = setup_model_plot(
        ce.M, ce.data, target, length_out, zoom, dim_red, plot_loss, loss_fun, xlims, ylims
    )

    xlims --> xlims
    ylims --> ylims

    # Contour plot:
    @series begin
        seriestype := :contourf
        contour_series[1], contour_series[2], contour_series[3]
    end

    # Scatter plot:
    for (i, x) in enumerate(unique(sort(y)))
        @series begin
            seriestype := :scatter
            markercolor := i
            markersize := ms
            group_idx = findall(y .== x)
            label --> "$(x)"
            X[group_idx, 1], X[group_idx, 2]
        end
    end

    max_iter = total_steps(ce)
    max_iter = if isnothing(plot_up_to)
        total_steps(ce)
    else
        minimum([plot_up_to, max_iter])
    end
    max_iter += 1
    path_x, path_y = setup_ce_plot(ce)

    # Outer loop over number of counterfactuals:
    for (num_counterfactual, X) in enumerate(eachslice(path_x; dims=3))
        # Inner loop over counterfactual search steps:
        steps = zip(eachcol(X), path_y)
        for (i, (x, y)) in enumerate(steps)
            i <= max_iter || break
            _final_iter = i == length(steps) || i == max_iter
            _annotate = i == length(steps) && ce.num_counterfactuals > 1
            @series begin
                seriestype := :scatter
                markercolor := CategoricalArrays.levelcode.(y[num_counterfactual])
                markersize := _final_iter ? msfinal : mspath
                series_annotation :=
                    _annotate ? text("C$(num_counterfactual)", mspath) : nothing
                label := :none
                x[1, :], x[2, :]
            end
        end
    end
end

"""
    animate_path(ce::CounterfactualExplanation, path=tempdir(); plot_proba::Bool=false, kwargs...)

Returns and animation of the counterfactual path.

# Examples

```julia-repl
# Search:
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

animate_path(ce)
```
"""
function animate_path(
    ce::CounterfactualExplanation,
    path=tempdir();
    plot_up_to::Union{Nothing,Int}=nothing,
    legend=:topright,
    kwrgs...,
)
    max_iter = total_steps(ce)
    max_iter = if isnothing(plot_up_to)
        total_steps(ce)
    else
        minimum([plot_up_to, max_iter])
    end
    max_iter += 1

    anim = @animate for t in 1:max_iter
        plot(ce; plot_up_to=t, legend=legend, kwrgs...)
    end
    return anim
end

"""
    setup_ce_plot(ce::CounterfactualExplanation)

A helper method that prepares data for plotting.
"""
function setup_ce_plot(ce::CounterfactualExplanation)
    path_embedded = embed_path(ce)
    path_labels = CounterfactualExplanations.counterfactual_label_path(ce)
    y_levels = ce.data.y_levels
    path_labels = map(x -> CategoricalArrays.categorical(x; levels=y_levels), path_labels)
    return path_embedded, path_labels
end
