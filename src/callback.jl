# This file constains the definitions for the callback function used in the
# optimisation

# TODO: add option to skip logging of trace
# TODO: add option to allow writing to disk
# TODO: add option to skip writing to disk
# TODO: add extension/wrapper interface
# TODO: add my own fancy printing???

struct Trace
    value::Vector{Float64}
    g_norm::Vector{Float64}
    iter::Vector{Int}
    time::Vector{Float64}
    step_size::Vector{Float64}

    function Trace(value, g_norm, iter, time, step_size)
        length(value) == length(g_norm) == length(iter) == length(time) == length(step_size) || throw(ArgumentError("Trace vectors must be the same length!"))

        new(value, g_norm, iter, time, step_size)
    end
end

function _update_trace!(trace::Trace, state)
    push!(trace.value, state.value)
    push!(trace.g_norm, state.g_norm)
    push!(trace.iter, state.iteration)
    push!(trace.time, state.metadata["time"])
    push!(trace.step_size, state.metadata["Current step size"])
end

struct Callback{WRITE}
    trace::Trace

    Callback(trace::Trace; write::Bool=false) = new{write}(trace)
end
Callback(; write::Bool=false) = Callback(Trace(Float64[], Float64[], Int[], Float64[], Float64[]), write=write)

function (f::Callback)(x)
    # write current state to trace
    _update_trace!(f.trace, x)

    # write data to disk
    _write_data(x.iteration, x.metadata["x"], f)

    return false
end

_write_data(::Any, ::Any, ::Callback{false}) = nothing
function _write_data(iter, a, ::Callback{true})
    # create directory if it doesn't already exist
    isdir(loc*string(iter)) ? nothing : mkdir(loc*string(iter))

    # write velocity coefficients to file
    open(loc*string(iter)*"/"*"a", "w") do f
        write(f, a)
    end

    return nothing
end
