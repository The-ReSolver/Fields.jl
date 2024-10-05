# Definition to read and write a field from disk.

function writeSpectralField(path, u::SpectralField{<:Any, <:Any, <:Any, <:Any, <:Any, PROJECTED}; name::String="field") where {PROJECTED}
    jldopen(path*name*".jld2", "w") do f
        f["grid"] = get_grid(u)
        f["projected"] = PROJECTED
        f["data"] = parent(u)
    end
    return nothing
end

function readSpectralField(path, ::Type{T}=Float64; name::String="field") where {T}
    u = jldopen(path*name*".jld2", "r") do f
        return SpectralField{f["projected"]}(f["data"], f["grid"])
    end
    return u
end


function writeVectorField(path, u::VectorField{N, S}; name::String="field") where {N, PROJECTED, S<:SpectralField{<:Any, <:Any, <:Any, <:Any, <:Any, PROJECTED}}
    jldopen(path*name*".jld2", "w") do f
        f["grid"] = get_grid(u)
        f["projected"] = PROJECTED
        f["data"] = parent(u)
    end
    return nothing
end

function readVectorField(path; name::String="field")
    u = jldopen(path*name*".jld2", "r") do f
        vectorfield = f["data"]
        return VectorField([SpectralField{f["projected"]}(vectorfield[i], f["grid"]) for i in 1:3]...)
    end
    return u
end
