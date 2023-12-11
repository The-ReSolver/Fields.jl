# This file contains the definition of the utility type that will allow easier
# option passing to the optimisation method.

@with_kw struct OptOptions
    # simulation options
    maxiter::Int = 1000
    g_tol::Float64 = 1e-6
    allow_f_increases::Bool = true
    callback::Callback = Callback()
    alg::Optim.FirstOrderOptimizer = LBFGS()
    time_limit::Float64 = NaN

    # writing options
    n_it_write::Int = 1
end
