# This file constains the definitions for the callback function used in the
# optimisation

# TODO: add option to skip logging of trace
# TODO: add option to allow writing to disk
# TODO: add option to skip writing to disk
# TODO: add extension/wrapper interface
# TODO: add my own fancy printing???

struct Callback{WRITE}
    value::Vector{Float64}
    g_norm::Vector{Float64}
    iter::Vector{Int}
    time::Vector{Float64}

    function Callback(value, g_norm, iter, time; write=false)
        length(value) == length(g_norm) == length(iter) == length(time) || throw(ArgumentError("Trace vectors must be the same length!"))

        new{write}(value, g_norm, iter, time)
    end
end
Callback(; write::Bool=false) = Callback(Float64[], Float64[], Int[], Float64[], write=write)

function (f::Callback)(x)
    # write current state to trace
    push!(f.value, x.value)
    push!(f.g_norm, x.g_norm)
    push!(f.iter, x.iteration)
    push!(f.time, x.metadata["time"])

    # write data to disk
    _write_data(f)

    return false
end

_write_data(::Callback{false}) = nothing
function _write_data(::Callback{true})
    # create directory if it doesn't already exist
    isdir(loc*string(τ)) ? nothing : mkdir(loc*string(τ))

    # write velocity coefficients to file
    open(loc*string(τ)*"/"*"a", "w") do f
        write(f, a)
    end

    return nothing
end

# TODO: this function returns all the inputs required to start a new optimisation
function load_opt_dir(loc) end
