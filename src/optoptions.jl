# This file contains the definition of the utility type that will allow easier
# option passing to the optimisation method.

@with_kw struct OptOptions{CB}
    restart::Union{Float64, Int} = Inf; @assert restart === Inf || restart isa Int
    maxiter::Int = 1000
    g_tol::Float64 = 1e-6
    allow_f_increases::Bool = false
    alg::Optim.FirstOrderOptimizer = LBFGS()
    time_limit::Float64 = NaN
    n_it_write::Int = 1
    trace::Fields.Trace = Fields.Trace(Float64[], Float64[], Int[], Float64[], Float64[])
    write::Bool = false
    write_loc::String = "./"; @assert write_loc[end] == '/'
    verbose::Bool = false
    print_io::IO = stdout
    n_it_print::Int = 1
    callback::CB = x -> false; @assert !isempty(methods(callback))
    update_frequency_every::Union{Float64, Int} = Inf; @assert update_frequency_every === Inf || update_frequency_every isa Int
end
