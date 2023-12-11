# This file contains the definitions required to create and maintain the
# directory containing all the optimisation/simulation data.

function _init_opt_dir(opts, grid, modes, base_prof, Re, Ro, free_mean)
    # initialise the .ini file
    metadata_ini = Inifile()

    # write optimisation parameters
    pts = points(grid)
    set(metadata_ini, "sim_data", "Re", Re)
    set(metadata_ini, "sim_data", "Ro", Ro)
    set(metadata_ini, "sim_data", "free_mean", free_mean)
    set(metadata_ini, "grid_data", "Ny", length(pts[1]))
    set(metadata_ini, "grid_data", "Nz", length(pts[2]))
    set(metadata_ini, "grid_data", "Nt", length(pts[3]))
    set(metadata_ini, "grid_data", "M", size(modes, 2))
    set(metadata_ini, "grid_data", "beta", get_β(grid))
    set(metadata_ini, "grid_data", "omega", get_ω(grid))

    # write out the .ini file
    open(opts.write_loc*"params", "w") do f
        write(f, metadata_ini)
    end

    # write wall-normal grid (in case it is non-uniform)
    open(opts.write_loc*"y", "w") do f
        write(f, pts[1])
    end

    # write base profile to binary file
    open(opts.write_loc*"base_profile", "w") do f
        write(f, base_prof)
    end

    # write modes to binary file
    open(opts.write_loc*"modes", "w") do f
        write(f, modes)
    end

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

function _load_opt_dir(path, diff_method, diff2_method, quad_method, i=0)
    # add backslash to path if needed
    path[end] != '/' ? path = path*'/' : nothing

    # extract optimisation parameters
    ini = read(Inifile(), path)
    Ny = tryparse(Int, get(ini, "grid_data", "Ny"))
    Nz = tryparse(Int, get(ini, "grid_data", "Nz"))
    Nt = tryparse(Int, get(ini, "grid_data", "Nt"))
    M = tryparse(Int, get(ini, "grid_data", "M"))
    β = tryparse(Float64, get(ini, "grid_data", "beta"))
    ω = tryparse(Float64, get(ini, "grid_data", "omega"))
    free_mean = tryparse(Bool, get(ini, "sim_data", "free_mean"))
    Re = tryparse(Float64, get(ini, "sim_data", "Re"))
    Ro = tryparse(Float64, get(ini, "sim_data", "Ro"))

    # extract wall normal grid
    y = Vector{Float64}(undef, Ny)
    open(path*"y", "r") do f
        read!(f, y)
    end

    # extract base profile
    base_prof = Vector{Float64}(undef, Ny)
    open(path*"base_profile", "r") do f
        read!(f, base_prof)
    end

    # extract modes
    modes = Array{Float64, 4}(undef, 3*Ny, M, (Nz >> 1) + 1, Nt)
    open(path*"modes", "r") do f
        read!(f, modes)
    end

    # create grid object
    grid = Grid(y, Nz, Nt, diff_method(y), diff2_method(y), quad_method(y), ω, β)

    # get starting field coefficients
    a = SpectralField(Grid(Vector{Int}(undef, M), Nz, Nt, Matrix{Int}(undef, Ny, Ny), Matrix{Int}(undef, Ny, Ny), ones(Ny), ω, β))
    open(path*string(i)*"/"*"a", "r") do f
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

    return a, grid, modes, base_prof, Re, Ro, free_mean, trace
end

function _write_data(path, iter, a, if_write::Bool)
    if if_write
        # create directory if it doesn't already exist
        isdir(path*string(iter)) ? nothing : mkdir(path*string(iter))

        # write velocity coefficients to file
        open(path*string(iter)*"/a", "w") do f
            write(f, a)
        end
    end

    return nothing
end
