using LaplaceRedux
using Trapz

"""
    plot(
        la::Laplace,
        X::AbstractArray,
        y::AbstractArray;
        link_approx=:probit,
        target=nothing,
        length_out=50,
        zoom=-1,
    )

Calling `Plots.plot` on a `Laplace` object will plot the posterior predictive distribution and the training data.
"""
@recipe function plot(
    la::Laplace,
    X::AbstractArray,
    y::AbstractArray;
    link_approx=:probit,
    target=nothing,
    length_out=50,
    zoom=-1,
)

    # Asserts:
    if la.likelihood == :regression
        @assert size(X, 1) == 1 "Cannot plot regression for multiple input variables."
    else
        @assert size(X, 1) == 2 "Cannot plot classification for more than two input variables."
    end

    # Get user-defined arguments:
    xlims = get(plotattributes, :xlims, nothing)
    ylims = get(plotattributes, :ylims, nothing)
    title = get(plotattributes, :title, nothing)

    # Plot attributes
    lw = get(plotattributes, :linewidth, 1)
    lw_yhat = lw*2
    lw_contour = lw*0.1

    if la.likelihood == :regression

        xrange, yrange, xlims, ylims = surface_range(X, y, xlims, ylims, zoom, length_out)
        xlims := xlims
        ylims := ylims

        # Plot predictions:
        _x = collect(xrange)[:, :]'
        fμ, fvar = LaplaceRedux.predict(la, _x)
        fμ = vec(fμ)
        fσ = vec(sqrt.(fvar))
        @series begin
            seriestype := :path
            ribbon := (1.96 * fσ, 1.96 * fσ)
            linewidth := lw_yhat
            label --> "yhat"
            xrange, fμ
        end

        # Scatter training data:
        @series begin
            seriestype := :scatter
            label --> "ytrain"
            vec(X), vec(y)
        end

    end

    if la.likelihood == :classification

        xrange, yrange, xlims, ylims = surface_range(X, xlims, ylims, zoom, length_out)
        xlims := xlims
        ylims := ylims

        Z, target, title = get_contour(la, xrange, yrange, link_approx, target, title)

        # Contour plot:
        @series begin
            seriestype := :contourf
            linewidth := lw_contour
            title --> title
            xrange, yrange, Z[Int(target), :]
        end

        # Scatter plot:
        for (i, x) in enumerate(unique(sort(y)))
            @series begin
                seriestype := :scatter
                markercolor := i
                group_idx = findall(y .== x)
                label --> "$(x)"
                X[1, group_idx], X[2, group_idx]
            end
        end

    end

end

function surface_range(
    X::AbstractArray, y::AbstractArray,
    xlims,ylims,zoom,length_out,
)

    # Surface range:
    if isnothing(xlims)
        xlims = (minimum(X), maximum(X)) .+ (zoom, -zoom)
    else
        xlims = xlims .+ (zoom, -zoom)
    end
    if isnothing(ylims)
        ylims = (minimum(y), maximum(y)) .+ (zoom, -zoom)
    else
        ylims = ylims .+ (zoom, -zoom)
    end
    x_range = range(xlims[1]; stop = xlims[2], length = length_out)
    y_range = range(ylims[1]; stop = ylims[2], length = length_out)
    return x_range, y_range, xlims, ylims

end

function surface_range(X::AbstractArray,xlims,ylims,zoom,length_out)

    if isnothing(xlims)
        xlims = (minimum(X[1, :]), maximum(X[1, :])) .+ (zoom, -zoom)
    else
        xlims = xlims .+ (zoom, -zoom)
    end
    if isnothing(ylims)
        ylims = (minimum(X[2, :]), maximum(X[2, :])) .+ (zoom, -zoom)
    else
        ylims = ylims .+ (zoom, -zoom)
    end
    x_range = range(xlims[1]; stop = xlims[2], length = length_out)
    y_range = range(ylims[1]; stop = ylims[2], length = length_out)

    return x_range, y_range, xlims, ylims
end

function get_contour(la::Laplace, x_range, y_range, link_approx, target, title)

    predict_ = function (la, X::AbstractVector)
        z = LaplaceRedux.predict(la, X; link_approx = link_approx)
        if LaplaceRedux.outdim(la) == 1 # binary
            z = [1.0 - z[1], z[1]]
        end
        return z
    end
    Z = [predict_(la, [x, y]) for x in x_range, y in y_range]
    Z = reduce(hcat, Z)
    if LaplaceRedux.outdim(la) > 1
        if isnothing(target)
            @info "No target label supplied, using first."
        end
        target = isnothing(target) ? 1 : target
        title = isnothing(title) ? "p̂(y=$(target))" : title
    else
        target = isnothing(target) ? 2 : target
        title = isnothing(title) ? "p̂(y=$(target-1))" : title
    end

    return Z, target, title
end

"""
    calibration_plot(y_cal, samp_distr, n_bins)

This plot displays the true frequency of points in each confidence interval relative to the predicted fraction of points in that interval.
The intervals are taken in step of 0.05 quantiles.

## Inputs

- `la::Laplace` -- the laplace model to use.
- `Y_cal` -- a vector of  true values y_t.
- `samp_distr` -- an array of sampled distributions F(x_t) corresponding to the y_t stacked column-wise.
- `n_bins` -- numbers of bins to use.
"""
function calibration_plot(la::Laplace, y_cal, samp_distr; n_bins = 20)
    quantiles = collect(range(0; stop = 1, length = n_bins + 1))
    # Create a new plot object
    p = plot()
    plot!([0, 1], [0, 1], label = "Perfect calibration", linestyle = :dash, color = :black)
    # Compute the empirical frequency
    if la.likelihood == :regression
        emp_freq = empirical_frequency_regression(y_cal, samp_distr; n_bins)
        plot!(p, quantiles, emp_freq, color = :blue, label = "neural network")
        plot!(
            p,
            quantiles,
            emp_freq,
            fillrange = quantiles,
            color = :lightblue,
            label = "miscalibration area",
        )
        # Calculate the area between the curve and the diagonal
        area = trapz((quantiles), vec(abs.(emp_freq - quantiles)))
        annotate!(
            0.75,
            0.05,
            ("Miscalibration area = $(round(area, digits=2))", 8, 11, :bottom),
        )
    elseif la.likelihood == :classification
        num_p_per_interval, emp_freq, bin_centers =
            empirical_frequency_binary_classification(y_cal, samp_distr; n_bins)
        plot!(bin_centers, emp_freq, label = "Observed average", lw = 2)
    end

    # Add labels and title
    title!("Calibration Curve")
    xlabel!("Predicted proportion in interval")
    ylabel!("Observed proportion in interval")
    xlims!(0, 1)
    ylims!(0, 1)

    # Show the plot
    display(p)
end
