# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

export SpectraField

struct SpectraField{Ny, Nz, Nt, G, T<:Real, A<:AbstractArray{Complex{T}, 3}} <: AbstractArray{Complex{T}, 3}
    data::A

    # constrcut from size and type
    function SpectraField(Ny::Int, Nz::Int, Nt::Int, grid::Grid{S}, ::Type{T}=Float64) where {T<:Real, S}
        data = zeros(Complex{T}, Ny, (Nz >> 1) + 1, Nt)
        if (Ny, Nz, Nt) != S
            throw(ArgumentError("Grid not a valid shape: $S should equal ($Ny, $Nz, $Nt)"))
        end
        new{Ny, Nz, Nt, typeof(grid), T, typeof(data)}(data)
    end

    # construct from grid
    function SpectraField(grid::Grid{S}, ::Type{T}=Float64) where {S, T<:Real}
        data = zeros(Complex{T}, S[1], S[2], S[3])
        new{S[1], S[2], S[3], typeof(grid), T, typeof(data)}(data)
    end

    # construct from data
    # THIS CONSTRUCTOR IS NOT INTENDED FOR USE SINCE IT HAS AMBIGUITY IN THE
    # CORRESPONDING SIZE OF THE PHYSICAL SPACE ARRAY.
    function SpectraField(data::A, grid::Grid{S}) where {T<:Real, A<:AbstractArray{Complex{T}, 3}, S}
        shape = size(data)
        Nz = (shape[2] - 1) << 1
        if shape[1] != S[1] || (Nz != S[2] && Nz + 1 != S[2]) || shape[3] != S[3]
            throw(ArgumentError("Grid not a valid shape: $S should equal $shape"))
        end
        Nz = S[2]
        # the inverse bitwise operation always outputs a even number
        new{shape[1], Nz, shape[3], typeof(grid), T, A}(data)
    end
end

# define interface
Base.size(::SpectraField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)
Base.IndexStyle(::Type{<:SpectraField}) = Base.IndexLinear()

# get parent array
Base.parent(U::SpectraField) = U.data

Base.@propagate_inbounds function Base.getindex(U::SpectraField, I...)
    @boundscheck checkbounds(parent(U), I...)
    @inbounds ret = parent(U)[I...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(U::SpectraField, v, I...)
    @boundscheck checkbounds(parent(U), I...)
    @inbounds parent(U)[I...] = v
    return v
end
