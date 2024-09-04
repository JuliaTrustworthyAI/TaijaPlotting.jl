using MLUtils

function embed(data::CounterfactualData, X::AbstractArray = nothing; dim_red::Symbol = :pca)

    # Training compressor:
    if typeof(data.input_encoder) <: MultivariateStats.AbstractDimensionalityReduction
        tfn = data.input_encoder
    else
        X_train, _ = DataPreprocessing.unpack_data(data)
        if size(X_train, 1) < 3
            tfn = nothing
        else
            @info "Training model to compress data."
            if dim_red == :pca
                tfn = MultivariateStats.fit(PCA, X_train; maxoutdim = 2)
            elseif dim_red == :tsne
                tfn = MultivariateStats.fit(TSNE, X_train; maxoutdim = 2)
            end
            data.input_encoder = nothing
            X = isnothing(X) ? X_train : X
        end
    end

    # Transforming:
    X = typeof(X) <: Vector{<:Matrix} ? MLUtils.stack(X, dims = 2) : X
    if !isnothing(tfn) && !isnothing(X)
        X = mapslices(x -> MultivariateStats.predict(tfn, x), X, dims = 1)
    else
        X = isnothing(X) ? X_train : X
    end

    return X
end

"""
    embed_path(ce::CounterfactualExplanation)

Helper function that embeds path into two dimensions for plotting.
"""
function embed_path(ce::CounterfactualExplanation)
    data_ = ce.data
    return embed(data_, path(ce))
end

function prepare_for_plotting(data::CounterfactualData; dim_red::Symbol = :pca)
    X, _ = DataPreprocessing.unpack_data(data)
    y = data.output_encoder.labels
    @assert size(X, 1) != 1 "Don't know how to plot 1-dimensional data."
    multi_dim = size(X, 1) > 2
    if multi_dim
        X = embed(data, X; dim_red = dim_red)
    end
    return X', y, multi_dim
end

"""
    plot(data::CounterfactualData; dim_red = :pca)

Calling `Plots.plot` on a `data::CounterfactualData` object will generate a scatter plot of the data.
"""
@recipe function plot(data::CounterfactualData; dim_red = :pca)

    # Set up:
    X, y, _ = prepare_for_plotting(data; dim_red = dim_red)

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
