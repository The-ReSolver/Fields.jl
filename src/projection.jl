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
        a[i] = channel_int(@view(modes[:, i]), w, u)
    end

    return a
end
project(u::AbstractVector{<:Number}, w::AbstractVector{<:Number}, modes::AbstractMatrix{T}) where {T<:Number} = project!(zeros(T, size(modes, 2)), u, w, modes)

"""
    Project a spectral field onto a set of modes, returning the projected field
"""
function project!(a::AbstractArray{<:Number, 3}, u::AbstractArray{<:Number, 3}, w::AbstractVector{<:Number}, modes::AbstractArray{<:Number, 4})
    for I in CartesianIndices(@view(a[1, [Colon() for _ in 1:(ndims(a) - 1)]...]))
        project!(@view(a[:, I]), @view(u[:, I]), w, @view(modes[:, :, I]))
    end

    return a
end
project(u::AbstractArray{<:Number, 3}, w::AbstractVector{<:Number}, modes::AbstractArray{T, 4}) where {T<:Number} = project!(zeros(T, size(modes, 2), size(selectdim(u, 1, 1))...), u, w, modes)

"""
    Project a vector field onto a set of modes, returning the projected field
"""
function project!(a::AbstractArray{T, 3}, u::AbstractVector{<:AbstractArray{<:Number, 3}}, w::AbstractVector{<:Number}, modes::AbstractArray{<:Number, 4}) where {T}
    N = Int(size(modes, 1)/length(u))
    for K in CartesianIndices(@view(a[1, [Colon() for _ in 1:(ndims(a) - 1)]...])), n in axes(modes, 2)
        a[n, K] = zero(T)
        for i in eachindex(u)
            a[n, K] += channel_int(@view(modes[(N*(i - 1) + 1):N*i, n, K]), w, @view(u[i][:, K]))
        end
    end

    return a
end
project(u::AbstractVector{<:AbstractArray{<:Number, 3}}, w::AbstractVector{<:Number}, modes::AbstractArray{T, 4}) where {T<:Number} = project!(zeros(T, size(modes, 2), size(selectdim(u[1], 1, 1))...), u, w, modes)

function reverse_project!(u::AbstractArray{<:Number, 3}, a::AbstractArray{<:Number, 3}, modes::AbstractArray{<:Number, 4})
    for I in CartesianIndices(@view(u[1, [Colon() for _ in 1:(ndims(a) - 1)]...]))
        mul!(@view(u[:, I]), @view(modes[:, :, I]), @view(a[:, I]))
    end

    return u
end

# TODO: much better to have the method use dispatch to choose what dimensions to iterate over with a custom mode type
function expand!(u::AbstractVector{<:AbstractArray{T, 3}}, a::AbstractArray{<:Number, 3}, modes::AbstractArray{<:Number, 4}) where {T}
    N = Int(size(modes, 1)/length(u))
    for i in eachindex(u), K in CartesianIndices(@view(a[1, [Colon() for _ in 1:(ndims(a) - 1)]...]))
        mul!(@view(u[i][:, K]), @view(modes[(N*(i - 1) + 1):N*i, :, K]), @view(a[:, K]))
    end

    return u
end
