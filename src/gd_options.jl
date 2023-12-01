# This file contains the definition of the utility type that will allow easier
# option passing to the optimisation method.

using Parameters

export GDOptions

@with_kw mutable struct GDOptions
    # simulation options
    α::Float64 = 1e-3
    ϵ::Float64 = 1e-3
    maxiter::Int = 1000
    restart::Int = 0

    # printing options
    verbose::Bool = true
    print_io::IO = stdout
    n_it_print::Int = 1

    # writing options
    sim_dir::String = ""
    res_trace::Vector{Float64} = Float64[]
    tau_trace::Vector{Float64} = Float64[]
    n_it_write::Int = 1
end
