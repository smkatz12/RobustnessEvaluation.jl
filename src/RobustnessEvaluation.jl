module RobustnessEvaluation

using Flux
using Statistics
using NeuralVerification
using Plots
using Measures
using Random
using ProgressMeter
using NeuralVerification: compute_output
using NeuralVerification: init_vars, BoundedMixedIntegerLP, encode_network!, _ẑᵢ₊₁, TOL
using GLPK
using JuMP
using LazySets
using StatsBase
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE, JuMP.MOI.INFEASIBLE_OR_UNBOUNDED

struct RobustnessSummary
    samples
    errors_nominal
    mae_nominal
    errors_adversarial_true
    mae_adversarial_true
    errors_adversarial_pred
    mae_adversarial_pred
    saliency_maps
    ig_maps
end

export nominal_errors
include("nominal.jl")

export adversarial_errors
include("adversarial.jl")

export saliency_map, integrated_gradients, get_mask
include("explanations.jl")

export plot_summary, plot_nominal_errors, plot_adversarial_errors, plot_adversarial_perturbations, plot_explanations
include("plotting.jl")

export robustness_summary

function robustness_summary(network, X, Y; N_samples = 100, coeffs = [-0.74, -0.44],
    δ = 0.02, solver = GLPK.Optimizer, use_gurobi = false, PGD = false, 
    step_factor = 0.1, iterations = 600, baseline = zeros(128))
    """ Runs all functions to get robustness summary for a given network and dataset
        Args:
            - network: neural network to analyze (flux model)
            - X: training examples (each column is an example)
            - Y: labels (each column is a label)
            ---------------
            - N_samples: number of samples to calculate on (actual indices will be chosen randomly)
            - coeffs: coefficients to get output (linear combination of network outputs)
            - δ: radius for hyperrectangle to check error within
            - solver: solver for mixed integer programs (ignored if PGD true)
            - use_gurobi: whether we are using gurobi (still need to put in solver)
            - PGD: whether or not to use PGD (if not, solve exactly with MIP Verify)
            - step_factor: step factor for PGD (ignored if PGD false)
            - iterations: number of steps for PGD (ignored if PGD false)
            - baseline: IG baseline (should be same size as input)
    """
    N = size(Y, 2)
    nnv_network = NeuralVerification.network(network)

    # Get samples to work with
    samples = randperm(N)[1:N_samples]
    X_samples = X[:, samples]
    Y_samples = Y[:, samples]

    # Nominal
    println("Calculating nominal errors...")
    errors_nominal = nominal_errors(nnv_network, X_samples, Y_samples, coeffs = coeffs)
    mae_nominal = mean(errors_nominal)

    # Adversarial
    println("Calculating adversarial errors...")
    errors_adversarial_true, errors_adversarial_pred = adversarial_errors(nnv_network,
        X_samples, Y_samples, coeffs = coeffs, δ = δ, solver = solver, 
        use_gurobi = use_gurobi, PGD = PGD, step_factor = step_factor, 
        iterations = iterations)
    mae_adversarial_true = mean(errors_adversarial_true)
    mae_adversarial_pred = mean(errors_adversarial_pred)

    # Explanations
    println("Getting explanations...")
    saliency_maps = zeros(size(X_samples))
    ig_maps = zeros(size(X_samples))
    for i = 1:N_samples
        saliency_maps[:, i] = saliency_map(network, X[:, i], coeffs = coeffs)
        ig_maps[:, i] = integrated_gradients(network, X[:, i], baseline, coeffs = coeffs)
    end

    return RobustnessSummary(samples, errors_nominal, mae_nominal, errors_adversarial_true,
        mae_adversarial_true, errors_adversarial_pred, mae_adversarial_pred,
        saliency_maps, ig_maps)
end

end # module
