# This file contains the definitions required to solve the variational problem
# using Optim.jl.

# TODO: restart method

# I need the following methods to work:
#   1. starting field with new trace ✓
#   2. starting field with given trace ✓
#   3. from directory structure

# The current method works pretty well, I already have a loading function. So this can be used directly using the location of the directory to get all the stuff needed to start the optimisation

# TODO: different default methods?
# TODO: add option to wipe all data later than starting iteration
# TODO: need a fallback method that can handle stuff properly
function optimise!(path::String, Dy, Dy2, ws; opts::OptOptions=OptOptions(), restart::Int=0)
    # load data from directory
    a, g, modes, base, Re, Ro, free_mean, _, = _load_opt_dir(path, Dy, Dy2, ws, restart)

    # call other optimisation with provided options
    if free_mean
        sol, trace = optimise!(a, g, modes, Re, Ro, opts=opts)
    else
        sol, trace = optimise!(a, g, modes, Re, Ro, mean=base, opts=opts)
    end
end

function optimise!(a::SpectralField{M, Nz, Nt, <:Any, T}, g::Grid{S}, modes::Array{ComplexF64, 4}, Re, Ro; mean::Vector{T}=T[], opts::OptOptions=OptOptions()) where {M, Nz, Nt, T, S}
    # check if mean profile is provided
    if length(mean) == 0
        base = points(g)[1]
        free_mean = true
    else
        base = mean
        free_mean = false
    end

    # create callback function
    cb = Callback(opts)

    # initialise optimisation directory if specified
    opts.write ? _init_opt_dir(opts, g, modes, base, Re, Ro) : nothing

    # initialise cache functor
    dR! = ResGrad(g, modes, base, Re, Ro, free_mean)

    # remove the mean profile if desired
    if !free_mean
        a[:, 1, 1] .= zero(Complex{T})
    end

    # define objective function for optimiser
    function fg!(::Any, G, x)
        G === nothing ? R = dR!(x, false)[2] : (R = dR!(x, true)[2]; G .= dR!.out)

        return R
    end

    # print header for output
    opts.verbose ? _print_header(opts.print_io) : nothing

    # perform optimisation
    sol = optimize(Optim.only_fg!(fg!), a, opts.alg, _gen_optim_opts(opts, cb))

    # update input
    a .= Optim.minimizer(sol)

    return sol, opts.trace
end

# TODO: underlying optimisation method that both top level methods use
function _optimise!(a, g, modes, Re, Ro, base, free_mean, opts::OptOptions) end

_gen_optim_opts(opts, cb) = Optim.Options(; g_tol=opts.g_tol,
                                            allow_f_increases=opts.allow_f_increases,
                                            iterations=opts.maxiter,
                                            show_trace=false,
                                            extended_trace=true,
                                            show_every=1,
                                            callback=cb,
                                            time_limit=opts.time_limit,
                                            store_trace=false)

function _print_header(print_io)
    println(print_io, "-------------------------------------------------------------")
    println(print_io, "|  Iteration  |  Step Size  |  Residual     |  Gradient     |")
    println(print_io, "-------------------------------------------------------------")
    flush(print_io)
    return nothing
end
