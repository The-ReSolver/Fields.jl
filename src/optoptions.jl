# This file contains the definition of the utility type that will allow easier
# option passing to the optimisation method.

@with_kw struct OptOptions
    # simulation options
    maxiter::Int = 1000
    g_tol::Float64 = 1e-6
    allow_f_increases::Bool = false
    write_loc::String = "./"; @assert write_loc[end] == '/'
    callback::Callback = Callback()
    alg::Optim.FirstOrderOptimizer = LBFGS()
    time_limit::Float64 = NaN

    # printing options
    verbose::Bool = true
    show_trace::Bool = false
    print_io::IO = stdout
    n_it_print::Int = 1

    # writing options
    sim_dir::String = ""
    n_it_write::Int = 1
end
