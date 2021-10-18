# This file contains the custom type to define a scalar field in physical space
# for a rotating plane couette flow.

export PhysicalField

struct PhysicalField{Ny, Nz, Nt, G, T<:Real, A<:AbstractArray{T, 3}} <: AbstractArray{T, 3}
    data::A

    # construct from size and type and grid
    function PhysicalField(Ny::Int, Nz::Int, Nt::Int, G::Grid{S}, ::Type{T}=Float64) where {S, T<:Real}
        data = zeros(T, Ny, Nz, Nt)
        if (Ny, Nz, Nt) != S
            throw(ArgumentError("Grid not a valid shape: $S should equal ($Ny, $Nz, $Nt)"))
        end
        new{Ny, Nz, Nt, G, T, typeof(data)}(data)
    end

    # construct from grid
    function PhysicalField(G::Grid{S}, ::Type{T}=Float64) where {S, T<:Real}
        data = zeros(T, S[1], S[2], S[3])
        new{S[1], S[2], S[3], G, T, typeof(data)}(data)
    end

    # construct from data
    function PhysicalField(data::A, G::Grid{S}) where {T<:Real, A<:AbstractArray{T, 3}, S}
        shape = size(data)
        if shape != S
            throw(ArgumentError("Grid not a valid shape: $S should equal $shape"))
        end
        new{shape[1], shape[2], shape[3], G, T, A}(data)
    end
end

# define interface
Base.size(::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()

# get parent array
Base.parent(u::PhysicalField) = u.data

Base.@propagate_inbounds function Base.getindex(u::PhysicalField, I...)
    @boundscheck checkbounds(parent(u), I...)
    @inbounds ret = parent(u)[I...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(u::PhysicalField, v, I...)
    @boundscheck checkbounds(parent(u), I...)
    @inbounds parent(u)[I...] = v
    return v
end
