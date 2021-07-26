function saliency_map(network, input; coeffs = [-0.74, -0.44])
    """ Computes the gradient of coeffs' * y with respect to the input x
        Args:
            - network: flux network
            - input: input to get saliency map for
            ------------------
            - coeffs: coefficients to get output (linear combination of network outputs)
    """
    grad_function(x) = coeffs' * network(x)
    _, b = Flux.pullback(() -> grad_function(input), Flux.params(input))
    return b(1)[input]
end

function integrated_gradients(network, input, baseline; coeffs = [-0.74, -0.44], res = 20)
    """
    Args:
        - network: flux network
        - input: input to get integrated gradients for
        - baseline: IG baseline (should be same size as input)
        ----------------
        - coeffs: coefficients to get output (linear combination of network outputs)
        - number of samples used to approximate integral
    """
    grad_function(x) = coeffs' * network(x)
    αs = collect(range(0.0, stop = 1.0, length = res))
    attrs = zeros(length(input))
    for i = 1:res
        interp_inp = αs[i] .* input .+ (1 .- αs[i]) .* baseline
        attrs += saliency_map(network, interp_inp)
    end
    return attrs ./ res
end

function get_mask(attrs, x; threshold = 0.9, masked_opacity = 0.2)
    """ Returns a masked image that highlights the pixels with the highest
        absolute attributions
        Args:
            - attrs: vector of attributions (same size as x)
            - x: input associated with the attrs (vector)
            -------------
            - threshold: highlight pixels above this percentile
            - masked_opacity: opacity for non-highlighted pixels
    """
    perc = quantile(attrs, threshold)
    mask = ones(length(attrs))
    mask[attrs .< perc] .= masked_opacity
    masked_input = mask .* x
    return masked_input
end