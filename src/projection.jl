# This file contains the definitions to allow the computation of the projections
# of fields onto a set of channel modes. This also works for the projection of
# modes onto other modes.

"""
    Compute the integral of the product of two channel profiles.
"""
channel_int(u::AbstractVector{<:Number}, w::AbstractVector{<:Number}, v::AbstractVector{<:Number}) = sum(w[i]*dot(u[i], v[i]) for i in eachindex(u))

"""
    Project a channel profile onto a set of modes, returning the projected
    profile.
"""
function project!(a::AbstractVector{<:Number}, u::AbstractVector{<:Number}, w::AbstractVector{<:Number}, modes::AbstractMatrix{<:Number})
    for i in axes(modes, 2)
        a[i] = channel_int(u, w, @view(modes[:, i]))
    end

    return a
end
project(u::AbstractVector{<:Number}, w::AbstractVector{<:Number}, modes::AbstractMatrix{T}) where {T<:Number} = project!(zeros(T, size(modes, 2)), u, w, modes)

"""
    Project a spectral field onto a set of modes, returning the projected
    profile
"""
function project!(a::AbstractArray{<:Number, 3}, u::AbstractArray{<:Number, 3}, w::AbstractVector{<:Number}, modes::AbstractArray{<:Number, 4})
    for I in CartesianIndices(eachslice(A, dims=1)[1])
        project!(@view(a[:, I]), @view(u[:, I]), w, @view(modes[:, :, I]))
    end

    return a
end
project(u::AbstractArray{<:Number, 3}, w::AbstractVector{<:Number}, modes::AbstractArray{T, 4}) where {T<:Number} = project!(zeros(T, size(modes, 2), size(selectdim(u, 1, 1))), u, w, modes)
