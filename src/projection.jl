# This file contains the definitions to allow the computation of the projections
# of fields onto a set of channel modes. This also works for the projection of
# modes onto other modes.

"""
    Compute the integral of the product of two channel profiles.
"""
channel_int(u::AbstractVector{<:Number}, w::AbstractVector{<:Number}, v::AbstractVector{<:Number}) = sum(w[i]*dot(u[i], v[i]) for i in eachindex(u))

"""
    Project a vector field onto a set of modes, returning the projected field
"""
function project!(a::SpectralField{M, Nz, Nt, G, T, true}, u::VectorField{N, S}, modes::AbstractArray{ComplexF64, 4}) where {M, Ny, Nz, Nt, G, T, N, S<:SpectralField{Ny, Nz, Nt, G, T, false}}
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), m in 1:M
        a[m, nz, nt] = zero(T)
        for i in eachindex(u)
            a[m, nz, nt] += channel_int(@view(modes[(Ny*(i - 1) + 1):Ny*i, m, nz, nt]), get_grid(a).ws, @view(u[i][:, nz, nt]))
        end
    end

    return a
end
project(u::VectorField{N, S}, modes::AbstractArray{ComplexF64, 4}) where {N, S<:SpectralField} = project!(SpectralField(get_grid(u), modes), u, modes)

function expand!(u::VectorField{N, S}, a::SpectralField{M, Nz, Nt, G, T, true}, modes::AbstractArray{ComplexF64, 4}) where {M, Ny, Nz, Nt, G, T, N, S<:SpectralField{Ny, Nz, Nt, G, T, false}}
    for i in eachindex(u), nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
        mul!(@view(u[i][:, nz, nt]), @view(modes[(Ny*(i - 1) + 1):Ny*i, :, nz, nt]), @view(a[:, nz, nt]))
    end

    return u
end
