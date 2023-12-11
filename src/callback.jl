# This file constains the definitions for the callback function used in the
# optimisation

# TODO: add extension/wrapper interface

struct Callback
    trace::Trace
    opts::OptOptions
    start_iter::Int
    keep_zero::Bool

    function Callback(trace; opts=OptOptions())
        if length(trace.value) == 0
            keep_zero = true
            start_iter = 0
        else
            keep_zero = false
            start_iter = trace.iter[end]
        end

        new(trace, opts, start_iter, keep_zero)
    end
end
Callback(; opts=OptOptions()) = Callback(Trace(Float64[], Float64[], Int[], Float64[], Float64[]), opts=opts)

function (f::Callback)(x)
    # write current state to trace
    x.iteration % f.opts.n_it_trace == 0 ? _update_trace!(f.trace, x, f.start_iter, f.keep_zero) : nothing

    # write data to disk
    _write_data(f.opts.write_loc, x.iteration, x.metadata["x"], f.opts.write)

    # print the sate if desired
    f.opts.verbose && x.iteration % f.opts.n_it_print == 0 ? _print_state(f.opts.print_io, x.iteration, x.metadata["Current step size"], x.value, x.g_norm) : nothing

    return false
end

function _print_state(print_io, iter, step_size, value, g_norm)
    str = @sprintf("|%10d   |   %5.2e  |  %5.5e  |  %5.5e  |", iter, step_size, value, g_norm)
    println(print_io, str)
    flush(print_io)
    return nothing
end
