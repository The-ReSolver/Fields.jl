# This file contains the definitions required to create and maintain the
# directory containing all the optimisation/simulation data.

export write_opt, load_opt

# TODO: check if directory already exists
function write_opt(opts, grid, modes, base_prof, Re, Ro, free_mean)
    # write grid data to file
    jldopen(opts.write_loc*"optim.jld2", "w") do f
        f["grid"] = grid
        f["base_prof"] = base_prof
        f["modes"] = modes
        f["params/Re"] = Re
        f["params/Ro"] = Ro
        f["params/free_mean"] = free_mean
    end

    return nothing
end

function load_opt(path, i)
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
    trace = jldopen(path*string(i)*"/trace.jld2", "r") do f
        return Trace([read(f, "value")], [read(f, "g_norm")], [read(f, "iter")], [read(f, "time")], [read(f, "step_size")])
    end

    return a, grid, modes, base_prof, Re, Ro, free_mean, trace
end

function _write_data(path, trace, a)
    # create directory if it doesn't already exist
    isdir(path*string(trace.iter[end])) ? nothing : mkdir(path*string(trace.iter[end]))

    # write velocity coefficients to file
    open(path*string(trace.iter[end])*"/a", "w") do f
        write(f, a)
    end

    # write trace values
    _write_trace(trace)

    return nothing
end

function _write_trace(trace)
    # open jld file and write states
    jldopen(opts.write_loc*string(trace.iter[end])*"/trace.jld2", "w") do f
        f["value"] = trace.value[end]
        f["g_norm"] = trace.g_norm[end]
        f["iter"] = trace.iter[end]
        f["time"] = trace.time[end]
        f["step_size"] = trace.step_size[end]
    end

    return nothing
end
