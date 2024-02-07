# This file constains the definitions for the callback function used in the
# optimisation

# TODO: separate Optim.jl wrapper into separate package
struct Callback{Ny, Nz, Nt, M, FREEMEAN, S, D, T, PLAN, IPLAN, A}
    velocityCoefficients::SpectralField{M, Nz, Nt, Grid{S, T, D}, T, true, A}
    cache::ResGrad{Ny, Nz, Nt, M, FREEMEAN, S, D, T, PLAN, IPLAN}
    opts::OptOptions
    start_iter::Int
    keep_zero::Bool

    function Callback(optimisationCache::ResGrad{Ny, Nz, Nt, M, FREEMEAN, S, D, T, PLAN, IPLAN}, a::SpectralField{M, Nz, Nt, Grid{S, T, D}, T, true, A}, opts::OptOptions=OptOptions()) where {Ny, Nz, Nt, M, FREEMEAN, S, D, T, PLAN, IPLAN, A}
        if length(opts.trace.value) == 0
            keep_zero = true
            start_iter = 0
        else
            keep_zero = false
            start_iter = opts.trace.iter[end]
        end

        new{Ny, Nz, Nt, M, FREEMEAN, S, D, T, PLAN, IPLAN, A}(a, optimisationCache, opts, start_iter, keep_zero)
    end
end

function (f::Callback)(x)
    # run extra callback method
    callbackReturn = f.opts.callback(x)

    # parse the optimisation state
    velocityCoefficients, value, g_norm, iteration, time, step_size = _parseOptimisationState(f, x)

    # write current state to trace
    _update_trace!(f.opts.trace, value, g_norm, iteration, time, step_size, f.start_iter, f.keep_zero)

    # write data to disk
    f.opts.write && x.iteration % f.opts.n_it_write == 0 ? writeIteration(f.opts.write_loc*string(f.opts.trace.iter[end]), velocityCoefficients, f.opts.trace) : nothing

    # print the sate if desired
    f.opts.verbose && x.iteration % f.opts.n_it_print == 0 ? _print_state(f.opts.print_io, iteration, step_size, get_ω(f.cache.spec_cache[1]), value, g_norm) : nothing

    # update frequency
    Int(x.iteration % f.opts.update_frequency_every) == 0 && x.iteration != 0 ? f.cache.spec_cache[1].grid.dom[2] = optimalFrequency(f.cache) : nothing

    return callbackReturn
end

function _parseOptimisationState(callback::Callback, x::Optim.OptimizationState{<:Any, <:Optim.FirstOrderOptimizer})
    callback.velocityCoefficients .= x.metadata["x"]
    return callback.velocityCoefficients, x.value, x.g_norm, x.iteration, x.metadata["time"], x.metadata["Current step size"]
end
function _parseOptimisationState(callback::Callback, x::Optim.OptimizationState{<:Any, <:Optim.NelderMead})
    _vectorToVelocityCoefficients!(callback.velocityCoefficients, x.metadata["centroid"])
    return callback.velocityCoefficients, x.value, x.g_norm, x.iteration, x.metadata["time"], 0.0
end

function _print_state(print_io, iter, step_size, freq, value, g_norm)
    str = @sprintf("|%10d   |   %5.2e  |  %5.5e  |  %5.5e  |  %5.5e  |", iter, step_size, freq, value, g_norm)
    println(print_io, str)
    flush(print_io)
    return nothing
end
