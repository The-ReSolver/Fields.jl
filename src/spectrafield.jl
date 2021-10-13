# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

# TODO: Tests
# TODO: Transform between spectral and physical space
# TODO: Grid object to make the discretisation explicit (not completely necessary unless I want to change the discretisation?)

export SpectraField

struct SpectraField{Ny, Nz, Nt, T<:Real, A<:AbstractArray{Complex{T}, 3}} <: AbstractArray{Complex{T}, 3}
    data::A

    # constrcut from size and type
    function SpectraField(Ny::Int, Nz::Int, Nt::Int, ::Type{T}=Float64) where {T}
        data = zeros(Complex{T}, Ny, Nz, Nt)
        new{Ny, Nz, Nt, T, typeof(data)}(data)
    end

    # construct from data
    function SpectraField(data::A) where {T<:Real, A<:AbstractArray{Complex{T}, 3}}
        shape = size(data)
        new{shape[1], shape[2], shape[3], T, A}(data)
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
