# This file contains the definitions required to create and maintain the
# directory containing all the optimisation/simulation data.

export init_opt, load_opt

# TODO: is custom serialization a good idea for the spectral fields?

function init_opt(a0, opts, grid, modes, base_prof, Re, Ro, free_mean)
    # write grid data to file
    jldopen(opts.write_loc*"optim.jld2", "w") do f
        f["grid"] = grid
        f["base_prof"] = base_prof
        f["modes"] = modes
        f["params/Re"] = Re
        f["params/Ro"] = Ro
        f["params/free_mean"] = free_mean
    end

    # write initial condition
    _write_data(opts.write_loc, 0, a0)

    # # create residual trace
    # open(opts.callback.write_loc*"trace", "w") do f
    #     write(f, opts.callback.trace.value)
    #     write(f, opts.callback.trace.g_norm)
    #     write(f, opts.callback.trace.iter)
    #     write(f, opts.callback.trace.time)
    #     write(f, opts.callback.trace.step_size)
    # end

    return nothing
end

function load_opt(path, i=0)
    # add backslash to path if needed
    path[end] != '/' ? path = path*'/' : nothing

    # read jld2 file
    jldopen(path*"optim.jld2", "r") do f
        global grid = read(f, "grid")
        global base_prof = read(f, "base_prof")
        global modes = read(f, "modes")
        global Re = read(f, "params/Re")
        global Ro = read(f, "params/Ro")
        global free_mean = read(f, "params/free_mean")
    end

    # get grid values
    ω = get_ω(grid)
    β = get_β(grid)
    Ny = length(base_prof)
    Nz = length(points(grid)[2])
    Nt = length(points(grid)[3])

    # get starting field coefficients
    a = SpectralField(Grid(Vector{Float64}(undef, size(modes, 2)), Nz, Nt, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), ones(Ny), ω, β))
    open(path*string(i)*"/a", "r") do f
        return read!(f, parent(a))
    end

    # extract traces
    # trace = open(path*"trace", "r") do f
    #     tr_size = filesize(f)÷(4*sizeof(Float64) + sizeof(Int))
    #     return Trace(read!(f, Vector{Float64}(undef, tr_size)),
    #                 read!(f, Vector{Float64}(undef, tr_size)),
    #                 read!(f, Vector{Int}(undef, tr_size)),
    #                 read!(f, Vector{Float64}(undef, tr_size)),
    #                 read!(f, Vector{Float64}(undef, tr_size)))
    # end

    # return a, grid, modes, base_prof, Re, Ro, free_mean, trace
    return a, grid, modes, base_prof, Re, Ro, free_mean
end

function _write_data(path, iter, a)
    # create directory if it doesn't already exist
    isdir(path*string(iter)) ? nothing : mkdir(path*string(iter))

    # write velocity coefficients to file
    open(path*string(iter)*"/a", "w") do f
        write(f, a)
    end

    return nothing
end
