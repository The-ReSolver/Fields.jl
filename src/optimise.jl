# This file contains the definitions required to solve the variational problem
# using Optim.jl.

# TODO: test with laminar case + unit tests for output functions

# The callback will handle the first iteration, I just need a way to start the optimisation without a given trace

# ! Don't pass a trace argument to the optimisation options !
function optimise!(options::OptOptions=OptOptions())
    # load the optimisation
    velocityCoefficients, modes, baseProfile, Re, Ro, ifFreeMean, trace = loadOptimisationState(opts.write_loc, options.restart)

    # initialise residual function
    dR! = ResGrad(get_grid(velocityCoefficients), modes, baseProfile, Re, Ro, ifFreeMean)

    # update trace with values from disc
    append!(options.trace.value, trace.value)
    append!(options.trace.g_norm, trace.g_norm)
    append!(options.trace.iter, trace.iter)
    append!(options.trace.time, trace.time)
    append!(options.trace.step_size, trace.step_size)

    # call optimisation
    sol = _optimise!(velocityCeofficients, dR!, ifFreeMean, options)

    return sol, opts.trace
end

optimise!(u::VectorField{3, S}, modes::Array{ComplexF64, 4}, Re, Ro; mean::Vector{T}=T[], opts::OptOptions=OptOptions()) where {Nz, Nt, T, S<:SpectralField{<:Any, Nz, Nt, <:Any, T}} = optimise!(project!(SpectralField(get_grid(u), modes), u, get_ws(u), modes), modes, Re, Ro, mean=mean, opts=opts)
function optimise!(velocityCoefficients::SpectralField{M, Nz, Nt, <:Any, T}, modes::Array{ComplexF64, 4}, Re, Ro; mean::Vector{T}=T[], opts::OptOptions=OptOptions()) where {M, Nz, Nt, T}
    # check if mean profile is provided
    baseProfile, ifFreeMean = _getBaseProfileFromMean(get_grid(velocityCoefficients), mean)

    # initialise directory to write optimisation data
    opts.write ? writeOptimisationParameters(opts.write_loc, get_grid(velocityCoefficients), modes, baseProfile, Re, Ro, ifFreeMean) : nothing

    # generate residual cache
    dR! = ResGrad(get_grid(velocityCoefficients), modes, baseProfile, Re, Ro, ifFreeMean)

    # call fallback optimisation method
    sol, trace = _optimise!(velocityCoefficients, dR!, ifFreeMean, opts)

    return sol, trace
end

# TODO: can I get a to be updated every iteration? (in the callback?)
function _optimise!(a, dR!, ifFreeMean, opts)
    # initialise callback function
    cb = Callback(dR!, opts)

    # remove mean profile if desired
    if !ifFreeMean
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

function _getBaseProfileFromMean(grid, mean)
    if length(mean) == 0
        baseProfile = points(grid)[1]
        ifFreeMean = true
    else
        baseProfile = mean
        ifFreeMean = false
    end
    return baseProfile, ifFreeMean
end

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
    println(print_io, "-----------------------------------------------------------------------------")
    println(print_io, "|  Iteration  |  Step Size  |       ω₀      |  Residual     |  Gradient     |")
    println(print_io, "-----------------------------------------------------------------------------")
    flush(print_io)
    return nothing
end
