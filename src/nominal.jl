function nominal_errors(network, X, Y; coeffs = [-0.74, -0.44])
    """ Computes absolute errors with no adversarial perturbations
        Args:
            - network: neural network model (neural verification format)
            - X: training examples (each column is an example)
            - Y: labels (each column is a label)
            ---------------
            - coeffs: coefficients to get output (linear combination of network outputs)
    """

    N_samples = size(Y, 2)

    truevals = vec(coeffs' * Y)
    predvals = [coeffs' * compute_output(network, X[:, i]) for i = 1:N_samples]

    return abs.(truevals - predvals)
end