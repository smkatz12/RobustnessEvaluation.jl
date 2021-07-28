function plot_summary(summary::RobustnessSummary, X; nbins = 8, image_shape = (16, 8), 
    reshape_im = x -> reshape(x, 16, 8)', threshold = 0.9, masked_opacity = 0.2)
    """ Plots parts of the robustness summary
        Args
            - summary: robustness summary
            - X: training examples (each column is an example)
            --------------
            - nbins: number of bins for histograms
            - reshape_im: function to reshape vectorized images to images
            - threshold: highlight pixels above this percentile
            - masked_opacity: opacity for non-highlighted pixels
    """
    default(titlefont = 30, guidefont = 24, tickfont = 20)

    xmax = 1.05 * max(max(maximum(summary.errors_nominal), maximum(summary.errors_adversarial_true)),
    maximum(summary.mae_adversarial_pred))

    # Histograms
    bins = 0:(xmax / nbins):xmax
    h1 = fit(Histogram, summary.errors_nominal, bins)
    h2 = fit(Histogram, summary.errors_adversarial_true, bins)
    h3 = fit(Histogram, summary.errors_adversarial_pred, bins)

    ymax = maximum([maximum(h1.weights), maximum(h2.weights), maximum(h3.weights)])

    p1 = plot(h1, title = "Nominal Absolute Errors", xlims = (0.0, xmax), 
    ylims = (0.0, ymax), legend = false, color = :teal, xlabel = "Absolute Error")
    p2 = plot(h2, title = "Adversarial Absolute Errors", xlims = (0.0, xmax), 
    ylims = (0.0, ymax), legend = false, color = :teal, xlabel = "Absolute Error")
    p3 = plot(h3, title = "Adversarial Perturbations", xlims = (0.0, xmax), 
    ylims = (0.0, ymax), legend = false, color = :teal, xlabel = "Absolute Error")

    # Explanations
    p4 = plot_explanations(summary, X, reshape_im = reshape_im, 
    threshold = threshold, masked_opacity = masked_opacity)

    l = @layout [a b c ; d]
    return plot(p1, p2, p3, p4, layout = l, size = (2200, 1600), margin = 10mm)
end

function plot_nominal_errors(summary::RobustnessSummary; xmax = nothing)
    if xmax === nothing
        xmax = 1.05 * maximum(summary.errors_nominal)
    end

    p = Plots.histogram(summary.errors_nominal, title = "Nominal Absolute Errors",
    xlims = (0.0, xmax), legend = false, xlabel = "Absolute Error", size = (1000, 1000))

    return p
end

function plot_adversarial_errors(summary::RobustnessSummary; xmax = nothing)
    if xmax === nothing
        xmax = 1.05 * maximum(summary.errors_adversarial_true)
    end

    p = Plots.histogram(summary.errors_adversarial_true, 
    title = "Adversarial Absolute Errors", xlims = (0.0, xmax), legend = false,
    xlabel = "Absolute Error", size = (1000, 1000))

    return p
end

function plot_adversarial_perturbations(summary::RobustnessSummary; xmax = nothing)
    if xmax === nothing
        xmax = 1.05 * maximum(summary.errors_adversarial_pred)
    end

    p = Plots.histogram(summary.errors_adversarial_pred, 
    title = "Adversarial Perturbations", xlims = (0.0, xmax), legend = false,
    xlabel = "Output Perturbation", size = (1000, 1000))

    return p
end

function plot_explanations(summary::RobustnessSummary, X; N = 5, 
    reshape_im = x -> reshape(x, 16, 8), threshold = 0.9, masked_opacity = 0.2)

    X_samples = X[:, summary.samples]
    sample_im = reshape_im(X_samples[:, 1])
    m, n = size(sample_im)
    
    total_im = ones(m * 3 + 2, n * N + (N - 1))

    for i = 1:N
        im = reshape_im(X_samples[:, i])
        total_im[1:m, n * (i-1) + 1 + (i - 1):n * i + (i - 1)] = im

        attrs_sal = summary.saliency_maps[:, i]
        masked_attrs_sal = get_mask(abs.(attrs_sal), X_samples[:, i], threshold = threshold,
                            masked_opacity = masked_opacity)
        masked_im_sal = reshape_im(masked_attrs_sal)
        total_im[m + 2:2 * m + 1, n * (i-1) + 1 + (i - 1):n * i + (i - 1)] = masked_im_sal

        attrs_ig = summary.ig_maps[:, i]
        masked_attrs_ig = get_mask(abs.(attrs_ig), X_samples[:, i], threshold = threshold,
                            masked_opacity = masked_opacity)
        masked_im_ig = reshape_im(masked_attrs_ig)
        total_im[2 * m + 3:3 * m + 2, n * (i-1) + 1 + (i - 1):n * i + (i - 1)] = masked_im_ig
    end

    p = plot(Gray.(total_im), axis = [], 
        title = "Explanations (Original, Saliency Map, Integrated Gradients)")

    return p
end