# This file contains the custom type to define a scalar field in physical space
# for a rotating plane couette flow.

struct PhysicalField{Ny, Nz, Nt, G, T, A} <: AbstractArray{T, 3}
    data::A
    grid::G

    # construct from function
    function PhysicalField(grid::Grid{S}, fun, ::Type{T}=Float64) where {T<:Real, S}
        # this assumes that z is along the second direction and t along the third
        y, z, t = points(grid)
        data = fun.(reshape(y, :, 1, 1), reshape(z, 1, :, 1), reshape(t, 1, 1, :))
        return new{S[1], S[2], S[3], typeof(grid), T, typeof(data)}(T.(data), grid)
    end
end

# construct from grid
PhysicalField(grid::Grid{S}, ::Type{T}=Float64) where {S, T<:Real} = PhysicalField(grid, (y,z,t)->zero(T), T)

# define interface
Base.size(::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()

# get parent array
Base.parent(u::PhysicalField) = u.data

# similar
Base.similar(u::PhysicalField, ::Type{T}=eltype(u)) where {T} = PhysicalField(u.grid, T)
Base.copy(u::PhysicalField) = (v = similar(u); v .= u; v)

# method to extract grid
get_grid(u::PhysicalField) = u.grid

# ~ BROADCASTING ~
# taken from MultiscaleArrays.jl
const PhysicalFieldStyle = Broadcast.ArrayStyle{PhysicalField}
Base.BroadcastStyle(::Type{<:PhysicalField}) = Broadcast.ArrayStyle{PhysicalField}()

# for broadcasting to construct new objects
Base.similar(bc::Base.Broadcast.Broadcasted{PhysicalFieldStyle}, ::Type{T}) where {T} =
    similar(find_field(bc))

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
