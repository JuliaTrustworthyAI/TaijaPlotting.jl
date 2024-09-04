using DataAPI
using Distributions: pdf
using NearestNeighborModels: KNNClassifier

"""
    function plot(
        M::AbstractFittedModel,
        data::CounterfactualData;
        target = nothing,
        length_out = 100,
        zoom = -0.1,
        dim_red = :pca,
        plot_loss = false,
        loss_fun = nothing,
    )

Calling `Plots.plot` on a `AbstractFittedModel` will plot the model's predictions as a contour. The `target` argument can be used to plot the predictions for a specific target variable. The `length_out` argument can be used to control the number of points used to plot the contour. The `zoom` argument can be used to control the zoom of the plot. The `dim_red` argument can be used to control the method used to reduce the dimensionality of the data if it has more than two features. 
"""
@recipe function plot(
    M::AbstractFittedModel,
    data::CounterfactualData;
    target = nothing,
    length_out = 100,
    zoom = -0.1,
    dim_red = :pca,
    plot_loss = false,
    loss_fun = nothing,
)

    # Asserts
    @assert !plot_loss || !isnothing(loss_fun) "Need to provide a loss function to plot the loss, e.g. (`loss_fun=Flux.Losses.logitcrossentropy`)."

    # Get user-defined arguments:
    xlims = get(plotattributes, :xlims, nothing)
    ylims = get(plotattributes, :ylims, nothing)

    # Plot attributes
    linewidth --> 0.1

    contour_series, X, y, xlims, ylims = setup_model_plot(
        M,
        data,
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
            group_idx = findall(y .== x)
            label --> "$(x)"
            X[group_idx, 1], X[group_idx, 2]
        end
    end

end

function setup_model_plot(
    M::AbstractFittedModel,
    data::CounterfactualData,
    target,
    length_out,
    zoom,
    dim_red,
    plot_loss,
    loss_fun,
    xlims,
    ylims,
)
    X, _ = DataPreprocessing.unpack_data(data)
    ŷ = probs(M, X) # true predictions
    if size(ŷ, 1) > 1
        ŷ = vec(OneHotArrays.onecold(ŷ, 1:size(ŷ, 1)))
    else
        ŷ = vec(ŷ)
    end

    # Target:
    if isnothing(target)
        target = data.y_levels[1]
        @info "No target label supplied, using first."
    end
    target_encoded = data.output_encoder(target)

    X, y, multi_dim = prepare_for_plotting(data; dim_red = dim_red)

    # Surface range:
    zoom = zoom * maximum(abs.(X))
    if isnothing(xlims)
        xlims = (minimum(X[:, 1]), maximum(X[:, 1])) .+ (zoom, -zoom)
    else
        xlims = xlims .+ (zoom, -zoom)
    end

    if isnothing(ylims)
        ylims = (minimum(X[:, 2]), maximum(X[:, 2])) .+ (zoom, -zoom)
    else
        ylims = ylims .+ (zoom, -zoom)
    end
    x_range = convert.(eltype(X), range(xlims[1]; stop = xlims[2], length = length_out))
    y_range = convert.(eltype(X), range(ylims[1]; stop = ylims[2], length = length_out))

    plot_loss = plot_loss || !isnothing(loss_fun)

    if plot_loss
        # Loss surface:
        Z = [
            loss_fun(logits(M, [x, y][:, :]), target_encoded) for x in x_range, y in y_range
        ]
    else
        # Prediction surface:
        if multi_dim
            knn1, y_train = voronoi(X, ŷ)
            predict_ =
                (X::AbstractVector) -> vec(
                    pdf(
                        MLJBase.predict(knn1, MLJBase.table(reshape(X, 1, 2))),
                        DataAPI.levels(y_train),
                    ),
                )
            Z = [predict_([x, y]) for x in x_range, y in y_range]
        else
            predict_ = function (X::AbstractVector)
                X = permutedims(permutedims(X))
                z = predict_proba(M, data, X)
                return z
            end
            Z = [predict_([x, y]) for x in x_range, y in y_range]
        end
    end

    # Pre-processes:
    Z = reduce(hcat, Z)
    target_idx = get_target_index(data.y_levels, target)
    z = plot_loss ? Z[1, :] : Z[target_idx, :]

    # Collect:
    contour_series = (x_range, y_range, z)

    return contour_series, X, y, xlims, ylims
end

function voronoi(X::AbstractMatrix, y::AbstractVector)
    knnc = KNNClassifier(; K = 1) # KNNClassifier instantiation
    X = MLJBase.table(X)
    y = CategoricalArrays.categorical(y)
    knnc_mach = MLJBase.machine(knnc, X, y)
    MLJBase.fit!(knnc_mach)
    return knnc_mach, y
end
