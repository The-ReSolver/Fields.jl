# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

export SpectraField

struct SpectraField{Ny, Nz, Nt, T<:Real, A<:AbstractArray{Complex{T}, 3}} <: AbstractArray{Complex{T}, 3}
    data::A

    # constrcut from size and type
    function SpectraField(Ny::Int, Nz::Int, Nt::Int, ::Type{T}=Float64) where {T<:Real}
        data = zeros(Complex{T}, Ny, (Nz >> 1) + 1, Nt)
        new{Ny, Nz, Nt, T, typeof(data)}(data)
    end

    # construct from data
    # THIS CONSTRUCTOR IS NOT INTENDED FOR USE SINCE IT HAS AMBIGUITY IN THE
    # CORRESPONDING SIZE OF THE PHYSICAL SPACE ARRAY.
    function SpectraField(data::A) where {T<:Real, A<:AbstractArray{Complex{T}, 3}}
        shape = size(data)
        # the inverse bitwise operation always outputs a even number
        new{shape[1], (shape[2] - 1) << 1, shape[3], T, A}(data)
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
