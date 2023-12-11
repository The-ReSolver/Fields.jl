# This file contains the definitions required to solve the variational problem
# using Optim.jl.

# TODO: restart method

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
    cb = Callback(opts.trace, opts=opts)

    # initialise optimisation directory if specified
    opts.write ? _init_opt_dir(opts, g, modes, base, Re, Ro) : nothing

    # initialise cache functor
    dR! = ResGrad(g, modes, base, Re, Ro, free_mean)

    # remove the mean profile if desired
    if !free_mean
        a[:, 1, 1] .= zero(Complex{T})
    end

    # define objective function for optimiser
    # TODO: can F be replace with ::Any???
    function fg!(F, G, x)
        G === nothing ? R = dR!(x, false)[2] : (R = dR!(x, true)[2]; G .= dR!.out)

        return R
    end

    # print header for output
    opts.verbose ? _print_header(opts.print_io) : nothing

    # perform optimisation
    sol = optimize(Optim.only_fg!(fg!), a, opts.alg, _gen_optim_opts(opts, cb))

    # update input
    a .= Optim.minimizer(sol)

    return sol, cb.trace
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
    println(print_io, "-------------------------------------------------------------")
    println(print_io, "|  Iteration  |  Step Size  |  Residual     |  Gradient     |")
    println(print_io, "-------------------------------------------------------------")
    flush(print_io)
    return nothing
end
