# Simple defintion to track the trace of the optimisation.

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

Trace() = Trace([], [], [], [], [])

function _update_trace!(trace::Trace, value, g_norm, iteration, time, step_size, start_iter, keep_zero)
    if iteration != 0 || keep_zero
        push!(trace.value, value)
        push!(trace.g_norm, g_norm)
        push!(trace.iter, iteration + start_iter)
        push!(trace.time, time)
        push!(trace.step_size, step_size)
    end
end

function _append_trace!(traceDestination::Trace, traceSource::Trace)
    append!(traceDestination.value, traceSource.value)
    append!(traceDestination.g_norm, traceSource.g_norm)
    append!(traceDestination.iter, traceSource.iter)
    append!(traceDestination.time, traceSource.time)
    append!(traceDestination.step_size, traceSource.step_size)
end
