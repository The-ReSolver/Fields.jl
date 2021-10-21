# This file contains the custom type to define a scalar field in physical space
# for a rotating plane couette flow.

export PhysicalField

struct PhysicalField{Ny, Nz, Nt, G, T<:Real, A<:AbstractArray{T, 3}} <: AbstractArray{T, 3}
    data::A
    grid::G

    # construct from grid
    function PhysicalField(grid::Grid{S}, ::Type{T}=Float64) where {S, T<:Real}
        data = zeros(T, S[1], S[2], S[3])
        return new{S[1], S[2], S[3], typeof(grid), T, typeof(data)}(data, grid)
    end

    # construct from data
    function PhysicalField(data::A, grid::Grid{S}) where {T<:Real, A<:AbstractArray{T, 3}, S}
        shape = size(data)
        if shape != S
            throw(ArgumentError("Grid not a valid shape: $S should equal $shape"))
        end
        return new{shape[1], shape[2], shape[3], typeof(grid), T, A}(data, grid)
    end

    # construct from function
    function PhysicalField(f::Function, grid::Grid{S}) where {T<:Real, S}
        # call the function for all the grid points (how to I expand out all the points??? Broadcasting???)
        new{S[1], S[2], S[3], typeof(grid), T, typeof(data)}(data)
    end
end

# define interface
Base.size(::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()

# get parent array
Base.parent(u::PhysicalField) = u.data

# similar
Base.similar(u::PhysicalField, ::Type{T}=eltype(u)) where {T} = PhysicalField(u.grid, T)

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
