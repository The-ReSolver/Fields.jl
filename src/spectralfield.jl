# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

export SpectralField

struct SpectralField{Ny, Nz, Nt, G, T<:Real, A<:AbstractArray{Complex{T}, 3}} <: AbstractArray{Complex{T}, 3}
    data::A
    grid::G

    # construct from grid
    function SpectralField(grid::Grid{S}, ::Type{T}=Float64) where {S, T<:Real}
        data = zeros(Complex{T}, S[1], (S[2] >> 1) + 1, S[3])
        return new{S[1], S[2], S[3], typeof(grid), T, typeof(data)}(data, grid)
    end

    # construct from data
    # THIS CONSTRUCTOR IS NOT INTENDED FOR USE SINCE IT HAS AMBIGUITY IN THE
    # CORRESPONDING SIZE OF THE PHYSICAL SPACE ARRAY.
    function SpectralField(data::A, grid::Grid{S}) where {T<:Real, A<:AbstractArray{Complex{T}, 3}, S}
        shape = size(data)
        Nz = (shape[2] - 1) << 1
        if shape[1] != S[1] || (Nz != S[2] && Nz + 1 != S[2]) || shape[3] != S[3]
            throw(ArgumentError("Grid not a valid shape: $S should equal $shape"))
        end
        Nz = S[2]
        # the inverse bitwise operation always outputs a even number
        return new{shape[1], Nz, shape[3], typeof(grid), T, A}(data, grid)
    end
end

# define interface
Base.size(::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, (Nz >> 1) + 1, Nt)
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()

# get parent array
Base.parent(U::SpectralField) = U.data

# similar
Base.similar(u::SpectralField{Ny, Nz, Nt, G, T}, ::Type{S}=T) where {Ny, Nz, Nt, G, T, S} = SpectralField(u.grid, S)

Base.@propagate_inbounds function Base.getindex(U::SpectralField, I...)
    @boundscheck checkbounds(parent(U), I...)
    @inbounds ret = parent(U)[I...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(U::SpectralField, v, I...)
    @boundscheck checkbounds(parent(U), I...)
    @inbounds parent(U)[I...] = v
    return v
end
