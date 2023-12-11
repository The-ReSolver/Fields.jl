# This file constains the definitions for the callback function used in the
# optimisation

# TODO: add option to skip logging of trace
# TODO: add option to allow writing to disk
# TODO: add option to skip writing to disk
# TODO: add extension/wrapper interface

mutable struct Trace
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

function _update_trace!(trace::Trace, state, start_iter)
    state.iteration != 0 ? push!(trace.value, state.value) : nothing
    state.iteration != 0 ? push!(trace.g_norm, state.g_norm) : nothing
    state.iteration != 0 ? push!(trace.iter, state.iteration + start_iter) : nothing
    state.iteration != 0 ? push!(trace.time, state.metadata["time"]) : nothing
    state.iteration != 0 ? push!(trace.step_size, state.metadata["Current step size"]) : nothing
end

struct Callback
    trace::Trace
    write::Bool
    write_loc::String
    verbose::Bool
    print_io::IO
    n_it_print::Int
    start_iter::Int

    function Callback(trace; write=false, write_loc="./", verbose=false, print_io=stdout, n_it_print=1)
        write_loc[end] != '/' ? write_loc = write_loc*'/' : nothing
        if length(trace.value) == 0
            push!(trace.iter, 0)
        end

        new(trace, write, write_loc, verbose, print_io, n_it_print, trace.iter[end])
    end
end
Callback(; write=false, write_loc="./", verbose=false, print_io=stdout, n_it_print=1) = Callback(Trace(Float64[], Float64[], Int[], Float64[], Float64[]), write=write, write_loc=write_loc, verbose=verbose, print_io=print_io, n_it_print=n_it_print)

function (f::Callback)(x)
    # write current state to trace
    _update_trace!(f.trace, x, f.start_iter)

    # write data to disk
    _write_data(f.write_loc, x.iteration, x.metadata["x"], f.write)

    # print the sate if desired
    f.verbose && x.iteration % f.n_it_print == 0 ? _print_state(f.print_io, x.iteration, x.metadata["Current step size"], x.value, x.g_norm) : nothing

    return false
end

function _print_state(print_io, i, α, R, dRda_norm)
    str = @sprintf("|%10d   |   %5.2e  |  %5.5e  |  %5.5e  |", i, α, R, dRda_norm)
    println(print_io, str)
    flush(print_io)
    return nothing
end
