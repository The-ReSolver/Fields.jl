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

struct Callback
    trace::Trace
    write::Bool
    write_loc::String

    function Callback(trace; write=false, write_loc="./")
        write_loc[end] != '/' ? write_loc = write_loc*'/' : nothing

        new(trace, write, write_loc)
    end
end
Callback(; write=false, write_loc="./") = Callback(Trace(Float64[], Float64[], Int[], Float64[], Float64[]), write=write, write_loc=write_loc)

function (f::Callback)(x)
    # write current state to trace
    _update_trace!(f.trace, x)

    # write data to disk
    _write_data(f.write_loc, x.iteration, x.metadata["x"], f.write)

    return false
end
