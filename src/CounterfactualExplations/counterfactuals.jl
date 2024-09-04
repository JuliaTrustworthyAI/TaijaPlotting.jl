using MLUtils: stack

@recipe function f(
    ce::CounterfactualExplanation;
    target=nothing,
    length_out=100,
    zoom=-0.1,
    dim_red=:pca,
    plot_loss=false,
    loss_fun=nothing,
    plot_up_to::Union{Nothing,Int} = nothing,
    n_points = 1000,
)

    ce = deepcopy(ce)
    ce.data = DataPreprocessing.subsample(ce.data, n_points)

    # Asserts
    @assert !plot_loss || !isnothing(loss_fun) "Need to provide a loss function to plot the loss, e.g. (`loss_fun=Flux.Losses.logitcrossentropy`)."

    # Get user-defined arguments:
    xlims = get(plotattributes, :xlims, nothing)
    ylims = get(plotattributes, :ylims, nothing)
    ms = get(plotattributes, :markersize, 3)
    mspath = ms*2
    msfinal = mspath*2

    # Plot attributes
    linewidth --> 0.1

    contour_series, X, y, xlims, ylims = setup_model_plot(
        ce.M,
        ce.data,
        target,
        length_out,
        zoom,
        dim_red,
        plot_loss,
        loss_fun,
        xlims,
        ylims,
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
    for (num_counterfactual, X) in enumerate(eachslice(path_x, dims=3))
        # Inner loop over counterfactual search steps:
        steps = zip(eachcol(X), path_y)       
        for (i,(x,y)) in enumerate(steps)
            i <= max_iter || break
            _annotate = i == length(steps) && ce.num_counterfactuals > 1
            @series begin
                seriestype := :scatter
                markercolor := CategoricalArrays.levelcode.(y[num_counterfactual])
                markersize := i == length(steps) ? msfinal : mspath
                series_annotation := _annotate ? text("C$(num_counterfactual)", mspath) : nothing
                label := :none
                x[1,:], x[2,:]
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
    path = tempdir();
    alpha_ = 0.5,
    plot_up_to::Union{Nothing,Int} = nothing,
    plot_proba::Bool = false,
    kwargs...,
)

    alpha = get(plotattributes, :alpha, 0.5)

    max_iter = total_steps(ce)
    max_iter = if isnothing(plot_up_to)
        total_steps(ce)
    else
        minimum([plot_up_to, max_iter])
    end
    max_iter += 1
    ingredients = set_up_plots(ce; alpha = alpha, plot_proba = plot_proba, kwargs...)

    anim = @animate for t = 1:max_iter
        final_state = t == max_iter
        plot_state(ce, t, final_state; ingredients...)
        if plot_proba
            plot(ingredients.p1, ingredients.p2; kwargs...)
        else
            plot(ingredients.p1; kwargs...)
        end
    end
    return anim
end

"""
    plot_state(
        ce::CounterfactualExplanation,
        t::Int,
        final_state::Bool;
        kwargs...
    )

Helper function that plots a single step of the counterfactual path.
"""
function plot_state(ce::CounterfactualExplanation, t::Int, final_state::Bool; kwargs...)
    args = PlotIngredients(; kwargs...)
    x1 = args.path_embedded[1, t, :]
    x2 = args.path_embedded[2, t, :]
    y = args.path_labels[t]
    _c = CategoricalArrays.levelcode.(y)
    n_ = ce.num_counterfactuals
    label_ = reshape(["C$i" for i = 1:n_], 1, n_)
    if !final_state
        scatter!(args.p1, x1, x2; group = y, colour = _c, ms = 5, label = "")
    else
        scatter!(args.p1, x1, x2; group = y, colour = _c, ms = 10, label = "")
        if n_ > 1
            label_1 = vec([text(lab, 5) for lab in label_])
            annotate!(x1, x2, label_1)
        end
    end
    if args.plot_proba
        probs_ = reshape(reduce(vcat, args.path_probs[1:t]), t, n_)
        if t == 1 && n_ > 1
            label_2 = label_
        else
            label_2 = ""
        end
        plot!(
            args.p2,
            probs_;
            label = label_2,
            color = reshape(1:n_, 1, n_),
            title = "p(y=$(ce.target))",
        )
    end
end

"A container used for plotting."
Base.@kwdef struct PlotIngredients
    p1::Any
    p2::Any
    path_embedded::Any
    path_labels::Any
    path_probs::Any
    alpha::Any
    plot_proba::Any
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
