# This file contains the definitions required to create and maintain the
# directory containing all the optimisation/simulation data.

export initialiseOptimisationDirectory, writeIteration, readOptimisationParameters, loadOptimisationState

# TODO: save mean profile not base profile
function initialiseOptimisationDirectory(path, velocityCoefficients, modes, baseProfile, Re, Ro, ifFreeMean)
    _writeOptimisationParameters(path, get_grid(velocityCoefficients), modes, baseProfile, Re, Ro, ifFreeMean)
    writeIteration(path*"0", velocityCoefficients)
end

function _loadOptimisationState(path, iteration)
    grid, baseProfile, modes, Re, Ro, ifFreeMean = readOptimisationParameters(path)
    velocityCoefficients = _readOptimisationVelocityCoefficients(path*string(iteration), SpectralField(grid, modes))
    trace = _tryToReadTrace(path*string(iteration))
    return velocityCoefficients, trace, modes, baseProfile, Re, Ro, ifFreeMean
end
loadOptimisationState(path, ::Float64) = _loadOptimisationState(path, _getFinalIteration(path))
loadOptimisationState(path, iteration::Int) = _loadOptimisationState(path, iteration)


function readOptimisationParameters(path)
    grid, baseProfile, modes, Re, Ro, ifFreeMean = jldopen(path*"parameters.jld2", "r") do f
        return read(f, "grid"), read(f, "baseProfile"), read(f, "modes"), read(f, "params/Re"), read(f, "params/Ro"), read(f, "params/ifFreeMean")
    end
    return grid, baseProfile, modes, Re, Ro, ifFreeMean
end

function _readOptimisationVelocityCoefficients(path, velocityCoefficients)
    open(path*"/velCoeff", "r") do f
        return read!(f, parent(velocityCoefficients))
    end
    return velocityCoefficients
end

function _tryToReadTrace(path)
    trace = nothing
    try
        trace = _readTraceFile(path*"/trace.jld2")
    catch
        trace = Trace()
    end
    return trace
end

function _readTraceFile(path)
    trace = jldopen(path, "a+") do f
        return Trace([read(f, "value")], [read(f, "g_norm")], [read(f, "iter")], [read(f, "time")], [read(f, "step_size")])
    end
    return trace
end

_getFinalIteration(path) = maximum(parse.(Int, filter!(x->tryparse(Int, x) !== nothing, readdir(path))))


function _writeOptimisationParameters(path, grid, modes, baseProfile, Re, Ro, ifFreeMean)
    # write grid data to file
    jldopen(path*"parameters.jld2", "w") do f
        f["grid"] = grid
        f["baseProfile"] = baseProfile
        f["modes"] = modes
        f["params/Re"] = Re
        f["params/Ro"] = Ro
        f["params/ifFreeMean"] = ifFreeMean
    end
end

function writeIteration(path, velocityCoefficients, trace)
    _writeVelocityCoefficients(path, velocityCoefficients)
    _writeTraceTo(path, trace)
end
writeIteration(path, velocityCoefficients) = _writeVelocityCoefficients(path, velocityCoefficients)

function _writeVelocityCoefficients(path, velocityCoefficients)
    _createIterationDirectory(path)
    _writeVelocityCoefficientsTo(path, velocityCoefficients)
end

_createIterationDirectory(path) = isdir(path) ? nothing : mkdir(path)

function _writeVelocityCoefficientsTo(path, velocityCoefficients)
    open(path*"/velCoeff", "w") do f
        write(f, parent(velocityCoefficients))
    end
end

function _writeTraceTo(path, trace)
    # open jld file and write states
    jldopen(path*"/trace.jld2", "w") do f
        f["value"] = trace.value[end]
        f["g_norm"] = trace.g_norm[end]
        f["iter"] = trace.iter[end]
        f["time"] = trace.time[end]
        f["step_size"] = trace.step_size[end]
    end
end
