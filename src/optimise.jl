# This file contains the definitions required to solve the variational problem
# using Optim.jl.

# ! Don't pass a trace argument to the optimisation options !
function optimise!(options::OptOptions=OptOptions())
    # load the optimisation
    velocityCoefficients, trace, modes, baseProfile, Re, Ro, ifFreeMean = loadOptimisationState(options.write_loc, options.restart)

    # initialise residual function
    dR! = ResGrad(get_grid(velocityCoefficients), modes, baseProfile, Re, Ro, ifFreeMean)

    # update trace with values from disc
    _append_trace!(options.trace, trace)

    # call optimisation
    results = _optimise!(velocityCoefficients, dR!, ifFreeMean, options)

    return results, options.trace
end

optimise!(u::VectorField{3, S}, modes::Array{ComplexF64, 4}, Re, Ro; mean::Vector{T}=T[], opts::OptOptions=OptOptions()) where {Nz, Nt, T, S<:SpectralField{<:Any, Nz, Nt, <:Any, T}} = optimise!(project!(SpectralField(get_grid(u), modes), u, get_ws(u), modes), modes, Re, Ro, mean=mean, opts=opts)
function optimise!(velocityCoefficients::SpectralField{M, Nz, Nt, <:Any, T}, modes::Array{ComplexF64, 4}, Re, Ro; mean::Vector{T}=T[], opts::OptOptions=OptOptions()) where {M, Nz, Nt, T}
    # check if mean profile is provided
    baseProfile, ifFreeMean = _getBaseProfileFromMean(get_grid(velocityCoefficients), mean)

    # initialise directory to write optimisation data
    opts.write ? initialiseOptimisationDirectory(opts.write_loc, velocityCoefficients, modes, baseProfile, Re, Ro, ifFreeMean) : nothing

    # generate residual cache
    dR! = ResGrad(get_grid(velocityCoefficients), modes, baseProfile, Re, Ro, ifFreeMean)

    # call fallback optimisation method
    results, trace = _optimise!(velocityCoefficients, dR!, ifFreeMean, opts)

    return results, trace
end

function _optimise!(velocityCoefficients, dR!, ifFreeMean, opts)
    # initialise callback function
    cb = Callback(dR!, velocityCoefficients, opts)

    # remove mean profile if desired
    if !ifFreeMean
        velocityCoefficients[:, 1, 1] .= zero(Complex{T})
    end

    # print header for output
    opts.verbose ? _print_header(opts.print_io) : nothing

    # perform optimisation
    results = _optimise!(velocityCoefficients, dR!, opts.alg, _gen_optim_opts(opts, cb))

    return results, opts.trace
end

function _optimise!(velocityCoefficients, dR!, algorithm::Optim.AbstractOptimizer, optimOptions)
    function fg!(::Any, G, x)
        G === nothing ? R = dR!(x, false)[2] : (R = dR!(x, true)[2]; G .= dR!.out)

        return R
    end
    optimize(Optim.only_fg!(fg!), velocityCoefficients, algorithm, optimOptions)
end

function _optimise!(velocityCoefficients::SpectralField{Ny, Nz, Nt}, dR!, ::Optim.NelderMead, optimOptions) where {Ny, Nz, Nt}
    velocityCoefficientsTemp = similar(velocityCoefficients)
    function f(x)
        dR!(_vectorToVelocityCoefficients!(velocityCoefficientsTemp, x), false)[2]
    end
    optimize(f, _velocityCoefficientsToVector!(Vector{Float64}(undef, 2*Ny*(Nz >> 1 + 1)*Nt), velocityCoefficients), Optim.NelderMead(), optimOptions)
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
