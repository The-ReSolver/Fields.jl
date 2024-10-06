# This file contains the custom type to define a scalar field in physical space
# for a rotating plane couette flow.

struct PhysicalField{G, DEALIAS, PADFACTOR} <: AbstractArray{Float64, 3}
    data::Array{Float64, 3}
    grid::G

    PhysicalField{DEALIAS, PADFACTOR}(grid::Grid, field::AbstractArray{T, 3}) where {DEALIAS, PADFACTOR, T} = new{typeof(grid), DEALIAS, PADFACTOR}(Float64.(field), grid)
end

# outer constructors
PhysicalField(grid::Grid, fun, dealias::Bool=false; pad_factor::Real=3/2) = PhysicalField{dealias, pad_factor}(grid, field_from_function(fun, points(grid), grid.dom, Val(dealias), pad_factor))
PhysicalField(grid::Grid, dealias::Bool=false; pad_factor::Real=3/2) = PhysicalField(grid, (y,z,t)->zero(Float64), dealias, pad_factor=pad_factor)

function field_from_function(fun, grid_points, dom, ::Val{true}, pad_factor)
    Nz, Nt = length.(grid_points[2:3])
    z_padded, t_padded = map(x->(0:(x[2] - 1))*(2π/(dom[x[1]]*x[2])), enumerate(padded_size(Nz, Nt, pad_factor)))
    return fun.(reshape(grid_points[1], :, 1, 1), reshape(z_padded, 1, :, 1), reshape(t_padded, 1, 1, :))
end
field_from_function(fun, grid_points, ::Any, ::Val{false}, ::Real) = fun.(reshape(grid_points[1], :, 1, 1), reshape(grid_points[2], 1, :, 1), reshape(grid_points[3], 1, 1, :))

# get parent array
Base.parent(u::PhysicalField) = u.data

# define interface
Base.size(u::PhysicalField) = size(parent(u))
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()

# similar
Base.similar(u::PhysicalField{G, DEALIAS, PADFACTOR}) where {G, DEALIAS, PADFACTOR} = PhysicalField(get_grid(u), DEALIAS, pad_factor=PADFACTOR)
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
