using LaplaceRedux
using Trapz

function Plots.plot(
    la::Laplace,
    X::AbstractArray,
    y::AbstractArray;
    link_approx::Symbol = :probit,
    target::Union{Nothing,Real} = nothing,
    colorbar = true,
    title = nothing,
    length_out = 50,
    zoom = -1,
    xlims = nothing,
    ylims = nothing,
    linewidth = 0.1,
    lw = 4,
    kwargs...,
)
    if la.likelihood == :regression
        @assert size(X, 1) == 1 "Cannot plot regression for multiple input variables."
    else
        @assert size(X, 1) == 2 "Cannot plot classification for more than two input variables."
    end

    if la.likelihood == :regression

        # REGRESSION

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

        title = isnothing(title) ? "" : title

        # Plot:
        scatter(
            vec(X),
            vec(y);
            label = "ytrain",
            xlim = xlims,
            ylim = ylims,
            lw = lw,
            title = title,
            kwargs...,
        )
        _x = collect(x_range)[:, :]'
        fμ, fvar = la(_x)
        fμ = vec(fμ)
        fσ = vec(sqrt.(fvar))
        pred_std = sqrt.(fσ .^ 2 .+ la.prior.σ^2)
        plot!(
            x_range,
            fμ;
            color = 2,
            label = "yhat",
            ribbon = (1.96 * pred_std, 1.96 * pred_std),
            lw = lw,
            kwargs...,
        )   # the specific values 1.96 are used here to create a 95% confidence interval
    else

        # CLASSIFICATION

        # Surface range:
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

        # Plot
        predict_ = function (X::AbstractVector)
            z = la(X; link_approx = link_approx)
            if LaplaceRedux.outdim(la) == 1 # binary
                z = [1.0 - z[1], z[1]]
            end
            return z
        end
        Z = [predict_([x, y]) for x in x_range, y in y_range]
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

        # Contour:
        contourf(
            x_range,
            y_range,
            Z[Int(target), :];
            colorbar = colorbar,
            title = title,
            linewidth = linewidth,
            xlims = xlims,
            ylims = ylims,
            kwargs...,
        )
        # Samples:
        scatter!(X[1, :], X[2, :]; group = Int.(y), color = Int.(y), kwargs...)
    end
end




const quantiles= collect(0:0.05:1)

"""
'Calibration_Plot_Regression(y_val, samp_distr)'

This plot displays the true frequency of points in each confidence interval relative to the predicted fraction of points in that interval.
The intervals are taken in step of 0.05 quantiles.

Input: 
-Y_val: a vector of  true values y_t
-samp_distr: an array of sampled distributions F(x_t) corresponding to the y_t stacked column-wise.
"""
function Calibration_Plot_Regression(y_val, samp_distr)
    quantiles= collect(0:0.05:1)
    # Compute the counts
    emp_freq = empirical_freq(y_val, samp_distr)
    # Create a new plot object
    p = plot()

    plot!(p, quantiles, emp_freq, fillrange = quantiles, color = :lightblue)

    plot!(p,[0, 1], [0, 1], color=:orange, linestyle=:dash, label="")

    plot!(p, quantiles , emp_freq, color=:blue, label="")
    # Calculate the area under the curve and subtract the area under the diagonal line


    area= trapz((quantiles),vec(abs.(emp_freq-quantiles)))

    # Add labels and title
    xlabel!("Predicted proportion in interval")
    ylabel!("Observed proportion in interval")
    title!("Average calibration")
    xlims!(0, 1)
    ylims!(0, 1)
    annotate!(0.75,0.05,("Miscalibration area = $(round(area, digits=2))",8,11,:bottom))

    # Show the plot
    display(p)
end
