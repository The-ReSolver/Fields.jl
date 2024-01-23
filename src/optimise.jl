# This file contains the definitions required to solve the variational problem
# using Optim.jl.

# Basically, I need two modes:
#   - one interactive for messing around in the REPL and Jupyter notebooks
#   - one non-interactive that can do its work independently and allows me to come and go and inspect the results (as it goes ideally)

# The interactive one works pretty well already, probably would be nice to add an optional wrapper to allow me to pass a velocity field to start.

function optimise_noninteractive!(dir)
    # load optimisation parameters from directory

    # call optimisation
    _optimise!()

    # write the data back to the disk

    # return results (success failures etc.)
end

function optimise!(u::VectorField{3, S}, modes::Array{ComplexF64, 4}, Re, Ro; mean::Vector{T}=T[], opts::OptOptions=OptOptions()) where {Nz, Nt, T, S<:SpectralField{<:Any, Nz, Nt, <:Any, T}}
    # project velocity field onto modes
    a = SpectralField(Grid(Vector{Float64}(undef, size(modes, 2)), Nz, Nt, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), ones(size(modes, 2)), get_ω(u), get_β(u)))
    project!(a, u, get_ws(u), modes)
    
    return optimise!(a, get_grid(u), modes, Re, Ro, mean=mean, opts=opts)
end

# TODO: get rid of grid in arguments since it is now stored in a
function optimise!(a::SpectralField{M, Nz, Nt, <:Any, T}, g::Grid{S}, modes::Array{ComplexF64, 4}, Re, Ro; mean::Vector{T}=T[], opts::OptOptions=OptOptions()) where {M, Nz, Nt, T, S}
    # check if mean profile is provided
    if length(mean) == 0
        base_prof = points(g)[1]
        free_mean = true
    else
        base_prof = mean
        free_mean = false
    end

    # call fallback optimisation method
    sol, trace = _optimise!(a, g, modes, Re, Ro, base_prof, free_mean, opts)

    return sol, trace
end

function _optimise!(a, g, modes, Re, Ro, base_prof, free_mean, opts)
    # initialise callback function
    cb = Callback(opts)

    # initialise directory to write optimisation data
    opts.write ? _write_opt(opts, g, modes, base_prof, Re, Ro, free_mean) : nothing

    # initialise cache function
    dR! = ResGrad(g, modes, base_prof, Re, Ro, free_mean)

    # remove mean profile if desired
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
