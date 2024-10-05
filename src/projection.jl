# This file contains the definitions to allow the computation of the projections
# of fields onto a set of channel modes. This also works for the projection of
# modes onto other modes.

"""
    Compute the integral of the product of two channel profiles.
"""
channel_int(u::AbstractVector{<:Number}, w::AbstractVector{<:Number}, v::AbstractVector{<:Number}) = sum(w[i]*dot(u[i], v[i]) for i in eachindex(u))

# TODO: specialiase these methods for spectral fields
"""
    Project a vector field onto a set of modes, returning the projected field
"""
function project!(a::AbstractArray{T, 3}, u::AbstractVector{<:AbstractArray{<:Number, 3}}, w::AbstractVector{<:Number}, modes::AbstractArray{<:Number, 4}) where {T}
    N = Int(size(modes, 1)/length(u))
    _, Nz, Nt = size(a)
    for nt in 1:Nt, nz in 1:Nz, m in axes(modes, 2)
        a[m, nz, nt] = zero(T)
        for i in eachindex(u)
            a[m, nz, nt] += channel_int(@view(modes[(N*(i - 1) + 1):N*i, m, nz, nt]), w, @view(u[i][:, nz, nt]))
        end
    end

    return a
end
project(u::AbstractVector{<:AbstractArray{<:Number, 3}}, w::AbstractVector{<:Number}, modes::AbstractArray{T, 4}) where {T<:Number} = project!(zeros(T, size(modes, 2), size(selectdim(u[1], 1, 1))...), u, w, modes)

function expand!(u::AbstractVector{<:AbstractArray{T, 3}}, a::AbstractArray{<:Number, 3}, modes::AbstractArray{<:Number, 4}) where {T}
    N = Int(size(modes, 1)/length(u))
    _, Nz, Nt = size(a)
    for i in eachindex(u), nt in 1:Nt, nz in 1:Nz
        mul!(@view(u[i][:, nz, nt]), @view(modes[(N*(i - 1) + 1):N*i, :, nz, nt]), @view(a[:, nz, nt]))
    end

    return u
end
