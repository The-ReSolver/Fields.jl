# This file contains the methods needed to perform a basic gradient descent
# optimisation of a field.

function gd!(u::VectorField{3, <:SpectralField{Ny, Nz, Nt}}, modes::Array{ComplexF64, 4}, mean::Vector{Float64}, Re::Real, Ro::Real; opts::GDOptions=GDOptions()) where {Ny, Nz, Nt}
    # set the mean of the velocity to zero
    u[1][:, 1, 1] .= 0.0
    u[2][:, 1, 1] .= 0.0
    u[3][:, 1, 1] .= 0.0

    # initialise the gradient function
    dR! = ResGrad(get_grid(u), modes, mean, Re, Ro)

    # obtain the projected velocity coefficients
    a = similar(dR!.out)
    project!(a, u, get_grid(u).ws, modes)

    # initialise simulation directory
    !isempty(opts.sim_dir) ? (f_res = _init_sim_dir(opts.sim_dir,
                                                    get_grid(u).y,
                                                    Ny, Nz, Nt,
                                                    get_β(u), get_ω(u),
                                                    Re, Ro,
                                                    mean,
                                                    opts.α,
                                                    opts.ϵ,
                                                    opts.restart,
                                                    opts.verbose,
                                                    opts.n_it_print,
                                                    opts.n_it_write,
                                                    modes)) : nothing

    # print header
    opts.verbose && _print_header(opts.print_io)

    # loop to step in descent direction
    i = 0
    dRda_norm_prev = 1e6
    while i <= opts.maxiter
        # compute the residual values
        Δa, R = dR!(a)
        dRda_norm = norm(Δa)

        # update traces
        if size(opts.res_trace) == (0,)
            append!(opts.res_trace, R)
            append!(opts.tau_trace, 0.0)
        end
        if i != 0
            append!(opts.res_trace, R)
            append!(opts.tau_trace, opts.tau_trace[end] + opts.α)
        end

        # modify step size if gradient increases
        dRda_norm - dRda_norm_prev > 0 ? opts.α *= 0.5 : nothing

        # print current state
        opts.verbose && (i % opts.n_it_print == 0 ? _print_state(opts.print_io, i, opts.α, opts.tau_trace[end], R, dRda_norm) : nothing)

        # write current state
        !isempty(opts.sim_dir) ? (i % opts.n_it_write == 0 ? _write_data(opts.sim_dir, a, round(opts.tau_trace[end], digits=5), R, dRda_norm) : nothing) : nothing

        # check if converges
        dRda_norm < opts.ϵ ? (println("Converged!"); break) : nothing

        # update the velocity
        a .-= opts.α.*Δa

        # update traces
        !isempty(opts.sim_dir) ? write(f_res, R) : nothing

        # update for next iteration
        dRda_norm_prev = dRda_norm
        i += 1
    end

    # if necessary close io streams
    opts.print_io isa IOStream ? close(opts.print_io) : nothing
    !isempty(opts.sim_dir) ? close(f_res) : nothing

    # take away mean component from cache before returning
    dR!.spec_cache[1][:, 1, 1] .= 0

    # convert final result back to full-space
    u[1] .= dR!.spec_cache[1]
    u[2] .= dR!.spec_cache[2]
    u[3] .= dR!.spec_cache[3]

    return u, dR!, opts
end



function _print_header(print_io)
    println(print_io, "-----------------------------------------------------------------------------")
    println(print_io, "|  Iteration  |  Step Size  |       τ       |  Residual     |  Gradient     |")
    println(print_io, "-----------------------------------------------------------------------------")
    flush(print_io)
    return nothing
end

function _print_state(print_io, i, α, τ, R, dRda_norm)
    str = @sprintf("|%10d   |   %5.2e  |  %5.5e  |  %5.5e  |  %5.5e  |", i, α, τ, R, dRda_norm)
    println(print_io, str)
    flush(print_io)
    return nothing
end



function _init_sim_dir(loc, y, Ny, Nz, Nt, β, ω, Re, Ro, ū, α, ϵ, restart, verbose, n_it_print, n_it_write, modes)
    # strip simulation directory slash at end if needed
    loc[end] != '/' ? loc = loc*'/' : nothing

    # initialise the .ini file
    metadata_ini = Inifile()

    # write simulation data
    set(metadata_ini, "sim_data", "Re", Re)
    set(metadata_ini, "sim_data", "Ro", Ro)
    set(metadata_ini, "sim_data", "step_size", α)
    set(metadata_ini, "sim_data", "eps", ϵ)
    set(metadata_ini, "sim_data", "restart", restart)
    set(metadata_ini, "sim_data", "verbose", verbose)
    set(metadata_ini, "sim_data", "n_it_print", n_it_print)
    set(metadata_ini, "sim_data", "n_it_write", n_it_write)

    # write grid data
    set(metadata_ini, "grid_data", "Ny", Ny)
    set(metadata_ini, "grid_data", "Nz", Nz)
    set(metadata_ini, "grid_data", "Nt", Nt)
    set(metadata_ini, "grid_data", "beta", β)
    set(metadata_ini, "grid_data", "omega", ω)

    # write out the .ini file
    open(loc*"sim_metadata", "w") do f
        write(f, metadata_ini)
    end

    # write wall-normal grid (in case it is non-uniform)
    open(loc*"y", "w") do f
        write(f, y)
    end

    # write mean profile to binary file
    open(loc*"u_mean", "w") do f
        write(f, ū)
    end

    # write modes to binary file
    open(loc*"modes", "w") do f
        write(f, modes)
    end

    # create residual trace
    f_res = open(loc*"Rs", "a+")

    return f_res
end

function _write_data(loc, a, τ, R, dRda_norm)
    # create directory if it doesn't already exist
    isdir(loc*string(τ)) ? nothing : mkdir(loc*string(τ))

    # write velocity coefficients to file
    open(loc*string(τ)*"/"*"a", "w") do f
        write(f, a)
    end

    # create .ini file for state metadata
    state_meta_ini = Inifile()
    set(state_meta_ini, "metadata", "tau", τ)
    set(state_meta_ini, "metadata", "R", R)
    set(state_meta_ini, "metadata", "||dRda||", dRda_norm)

    return nothing
end
