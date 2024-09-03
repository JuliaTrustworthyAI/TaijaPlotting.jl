using DataAPI
using Distributions: pdf
using NearestNeighborModels: KNNClassifier



@recipe function f(
    M::AbstractFittedModel,
    data::CounterfactualData; 
    target=nothing, 
    length_out=100,
    zoom=-0.1,
    dim_red=:pca,
    plot_loss=false,
    loss_fun=nothing,
)

    # Get user-defined arguments:
    xlims = get(plotattributes, :xlims, nothing)
    ylims = get(plotattributes, :ylims, nothing)

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

    X, y, multi_dim = prepare_for_plotting(data; dim_red=dim_red)

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
    x_range = convert.(eltype(X), range(xlims[1]; stop=xlims[2], length=length_out))
    y_range = convert.(eltype(X), range(ylims[1]; stop=ylims[2], length=length_out))

    plot_loss = plot_loss || !isnothing(loss_fun)

    if plot_loss
        # Loss surface:
        Z = [loss_fun(logits(M, [x, y][:, :]), target_encoded) for x in x_range, y in y_range]
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

    return x_range, y_range, z
    
end

@userplot struct ModelPlot{T<:Tuple{AbstractModel,CounterfactualData}}
    args::T
end

@recipe function f(mp::ModelPlot)
    model = mp.args[1]
    data = mp.args[2]
    plt = contourf(model, data)
    scatter!(data)
    display(plt)
    return nothing
end


# function Plots.plot(
#     M::AbstractFittedModel,
#     data::DataPreprocessing.CounterfactualData;
#     target::Union{Nothing,RawTargetType} = nothing,
#     colorbar = true,
#     title = "",
#     length_out = 100,
#     zoom = -0.1,
#     xlims = nothing,
#     ylims = nothing,
#     linewidth = 0.1,
#     alpha = 1.0,
#     contour_alpha = 1.0,
#     dim_red::Symbol = :pca,
#     plot_loss::Bool = false,
#     loss_fun::Union{Nothing,Function} = nothing,
#     kwargs...,
# )
#     X, _ = DataPreprocessing.unpack_data(data)
#     ŷ = probs(M, X) # true predictions
#     if size(ŷ, 1) > 1
#         ŷ = vec(Flux.onecold(ŷ, 1:size(ŷ, 1)))
#     else
#         ŷ = vec(ŷ)
#     end

#     # Target:
#     if isnothing(target)
#         target = data.y_levels[1]
#         @info "No target label supplied, using first."
#     end
#     target_encoded = data.output_encoder(target)

#     X, y, multi_dim = prepare_for_plotting(data; dim_red = dim_red)

#     # Surface range:
#     zoom = zoom * maximum(abs.(X))
#     if isnothing(xlims)
#         xlims = (minimum(X[:, 1]), maximum(X[:, 1])) .+ (zoom, -zoom)
#     else
#         xlims = xlims .+ (zoom, -zoom)
#     end
#     if isnothing(ylims)
#         ylims = (minimum(X[:, 2]), maximum(X[:, 2])) .+ (zoom, -zoom)
#     else
#         ylims = ylims .+ (zoom, -zoom)
#     end
#     x_range = convert.(eltype(X), range(xlims[1]; stop = xlims[2], length = length_out))
#     y_range = convert.(eltype(X), range(ylims[1]; stop = ylims[2], length = length_out))

#     plot_loss = plot_loss || !isnothing(loss_fun)

#     if plot_loss 
#         # Loss surface:
#         Z = [loss_fun(logits(M, [x, y][:, :]), target_encoded) for x in x_range, y in y_range]
#     else
#         # Prediction surface:
#         if multi_dim
#             knn1, y_train = voronoi(X, ŷ)
#             predict_ =
#                 (X::AbstractVector) -> vec(
#                     pdf(
#                         MLJBase.predict(knn1, MLJBase.table(reshape(X, 1, 2))),
#                         DataAPI.levels(y_train),
#                     ),
#                 )
#             Z = [predict_([x, y]) for x in x_range, y in y_range]
#         else
#             predict_ = function (X::AbstractVector)
#                 X = permutedims(permutedims(X))
#                 z = predict_proba(M, data, X)
#                 return z
#             end
#             Z = [predict_([x, y]) for x in x_range, y in y_range]
#         end
#     end

#     # Pre-processes:
#     Z = reduce(hcat, Z)
#     target_idx = get_target_index(data.y_levels, target)
#     z = plot_loss ? Z[1, :] : Z[target_idx, :]

#     # Contour:
#     Plots.contourf(
#         x_range,
#         y_range,
#         z;
#         colorbar = colorbar,
#         title = title,
#         linewidth = linewidth,
#         xlims = xlims,
#         ylims = ylims,
#         kwargs...,
#         alpha = contour_alpha,
#     )

#     # Samples:
#     return Plots.scatter!(data; dim_red = dim_red, alpha = alpha, kwargs...)
# end

function voronoi(X::AbstractMatrix, y::AbstractVector)
    knnc = KNNClassifier(; K = 1) # KNNClassifier instantiation
    X = MLJBase.table(X)
    y = CategoricalArrays.categorical(y)
    knnc_mach = MLJBase.machine(knnc, X, y)
    MLJBase.fit!(knnc_mach)
    return knnc_mach, y
end
