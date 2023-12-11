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

function _update_trace!(trace::Trace, state, start_iter, keep_zero)
    state.iteration != 0 || keep_zero ? push!(trace.value, state.value) : nothing
    state.iteration != 0 || keep_zero ? push!(trace.g_norm, state.g_norm) : nothing
    state.iteration != 0 || keep_zero ? push!(trace.iter, state.iteration + start_iter) : nothing
    state.iteration != 0 || keep_zero ? push!(trace.time, state.metadata["time"]) : nothing
    state.iteration != 0 || keep_zero ? push!(trace.step_size, state.metadata["Current step size"]) : nothing
end
