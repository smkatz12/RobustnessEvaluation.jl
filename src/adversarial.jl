function adversarial_errors(network, X, Y; coeffs = [-0.74, -0.44],
                            δ = 0.02, solver = GLPK.Optimizer, PGD = false,
                            step_factor = 0.1, iterations = 600)
    """ Computes absolute errors with no adversarial perturbations
        Args:
            - network: neural network (NNV model)
            - X: training examples (each column is an example)
            - Y: labels (each column is a label)
            ---------------
            - coeffs: coefficients to get output (linear combination of network outputs)
            - δ: radius for hyperrectangle to check error within
            - solver: solver for mixed integer programs (ignored if PGD true)
            - PGD: whether or not to use PGD (if not, solve exactly with MIP Verify)
            - step_factor: step factor for PGD (ignored if PGD false)
            - iterations: number of steps for PGD (ignored if PGD false)
    """

    N_samples = size(Y, 2)

    truevals = vec(coeffs' * Y)
    predvals = [coeffs' * compute_output(network, X[:, i]) for i = 1:N_samples]
    
    minvals = zeros(N_samples)
    maxvals = zeros(N_samples)

    if PGD
        @showprogress "Getting adversial results..." for i = 1:N_samples
            lbs = X[:, i] .- Float32.(δ * ones(size(X, 1)))
            ubs = X[:, i] .+ Float32.(δ * ones(size(X, 1)))
            maxvals[i] = pgd(network, X[:, i], lbs, ubs, coeffs = coeffs, 
                        step_size = step_factor, iterations = iterations)
            minvals[i] = -pgd(network, X[:, i], lbs, ubs, coeffs = -coeffs, 
                        step_size = step_factor, iterations = iterations)
        end
    else
        @showprogress "Getting adversial results..." for i = 1:N_samples
            input_set = Hyperrectangle(X[:, i], Float32.(δ * ones(size(X, 1))))
            maxvals[i] = optimize_ouput(network, input_set, coeffs = coeffs, solver = solver)
            minvals[i] = optimize_ouput(network, input_set, coeffs = coeffs, solver = solver, 
                                        maximize = false)
        end
    end

    return max.(abs.(truevals .- maxvals), abs.(truevals .- minvals)),
           max.(abs.(predvals .- maxvals), abs.(predvals .- minvals))
end

function optimize_ouput(network, input_set; 
    coeffs = [-0.74, -0.44], solver = GLPK.Optimizer, maximize=true, 
    obj_threshold=(maximize ? -Inf : Inf))
    """ Function from Chris to get max output
    """
    # Get your bounds
    bounds = NeuralVerification.get_bounds(Ai2z(), network, input_set; before_act=true)

    # Create your model
    model = Model(with_optimizer(solver))
    z = init_vars(model, network, :z, with_input=true)
    δ = init_vars(model, network, :δ, binary=true)
    # get the pre-activation bounds:
    model[:bounds] = bounds
    model[:before_act] = true

    # Add the input constraint 
    NeuralVerification.add_set_constraint!(model, input_set, first(z))

    # Encode the network as an MIP
    encode_network!(model, network, BoundedMixedIntegerLP())
    obj = coeffs'*last(z)
    if maximize 
        @objective(model, Max, obj)
        if obj_threshold != -Inf
            @constraint(model, obj >= obj_threshold)
        end
    else
        @objective(model, Min, obj)
        if obj_threshold != Inf 
            @constraint(model, obj >= obj_threshold)
        end
    end

    # Set lower and upper bounds 
    #for the first layer it's special because it has no ẑ
    set_lower_bound.(z[1], low(bounds[1]))
    set_upper_bound.(z[1], high(bounds[1]))
    for i = 2:length(z)-1
        # Set lower and upper bounds for the intermediate layers
        ẑ_i =  _ẑᵢ₊₁(model, i-1)
        z_i = z[i]
        # @constraint(model, ẑ_i .>= low(bounds[i])) These empirically seem to slow it down?
        # @constraint(model, ẑ_i .<= high(bounds[i]))
        z_low = max.(low(bounds[i]), 0.0)
        z_high = max.(high(bounds[i]), 0.0)
        set_lower_bound.(z_i, z_low)
        set_upper_bound.(model[:z][i], z_high)
    end
    
    # Set lower and upper bounds for the last layer special because 
    # it has no ReLU
    set_lower_bound.(z[end], low(bounds[end]))
    set_upper_bound.(z[end], high(bounds[end]))
    
    optimize!(model)
    if termination_status(model) == OPTIMAL
        return objective_value(model)
    elseif termination_status(model) == INFEASIBLE
        @warn "Infeasible result, did you have an output threshold? If not, then it should never return infeasible"
        return maximize ? -Inf : Inf  
    else
        @assert false "Non optimal result"
    end
end

function fgsm(network, x, lbs, ubs; coeffs = [-0.74, -0.44], step_size = 0.1)
    grad_full = NeuralVerification.get_gradient(network, x)
    grad = grad_full' * coeffs
    return clamp.(x + grad * step_size, lbs, ubs)
end

function pgd(network, x, lbs, ubs; coeffs = [-0.74, -0.44], step_size = 0.01, 
    iterations = 600)
    cur_x = x
    for i = 1:iterations
        cur_x = fgsm(network, cur_x, lbs, ubs; coeffs = coeffs, step_size = step_size)
    end
    return coeffs' * compute_output(network, cur_x)
end