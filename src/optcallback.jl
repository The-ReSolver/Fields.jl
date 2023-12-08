# This file constains the definitions for the callback function used in the
# optimisation

# TODO: add option to skip logging of trace
# TODO: add option to allow writing to disk
# TODO: add option to skip writing to disk
# TODO: add extension/wrapper interface
# TODO: add my own fancy printing???

struct Callback
    value::Vector{Float64}
    g_norm::Vector{Float64}
    iter::Vector{Int}
    time::Vector{Float64}
end
Callback() = Callback(Float64[], Float64[], Int[], Float64[])

function (f::Callback)(x)
    # write current state to trace
    push!(f.value, x.value)
    push!(f.g_norm, x.g_norm)
    push!(f.iter, x.iteration)
    push!(f.time, x.metadata["time"])

    return false
end
