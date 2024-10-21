# This file contains the definitions to allow the computation of the projections
# of fields onto a set of channel modes. This also works for the projection of
# modes onto other modes.

"""
    Compute the integral of the product of two channel profiles.
"""
channel_int(u::AbstractVector{ComplexF64}, w::Vector{Float64}, v::AbstractVector{ComplexF64}) = @inbounds sum(w[i]*dot(u[i], v[i]) for i in eachindex(u))

"""
    Project a vector field onto a set of modes, returning the projected field
"""
function project!(a::SpectralField{<:Grid{Ny, Nz, Nt}, true}, u::VectorField{N, S}, modes::AbstractArray{ComplexF64, 4}) where {Ny, Nz, Nt, N, S<:SpectralField{<:Grid{Ny, Nz, Nt}, false}}
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), m in axes(a, 1)
        @inbounds a[m, nz, nt] = 0.0
        for i in eachindex(u)
            @inbounds a[m, nz, nt] += channel_int(@view(modes[(Ny*(i - 1) + 1):Ny*i, m, nz, nt]), get_grid(a).ws, @view(u[i][:, nz, nt]))
        end
    end

    return a
end
project(u::VectorField{N, S}, modes::AbstractArray{ComplexF64, 4}) where {N, S<:SpectralField} = project!(SpectralField(get_grid(u), modes), u, modes)

function expand!(u::VectorField{N, S}, a::SpectralField{<:Grid{Ny, Nz, Nt}, true}, modes::AbstractArray{ComplexF64, 4}) where {Ny, Nz, Nt, N, S<:SpectralField{<:Grid{Ny, Nz, Nt}, false}}
    for i in eachindex(u)
        @inbounds u[i] .= 0.0
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), m in axes(a, 1)
            @inbounds @view(u[i][:, nz, nt]) .+= @view(modes[(Ny*(i - 1) + 1):Ny*i, m, nz, nt]).*a[m, nz, nt]
        end
    end

    return u
end
expand(a::SpectralField{G, true}, modes::AbstractArray{ComplexF64, 4}) where {G} = expand!(VectorField(get_grid(a)), a, modes)
