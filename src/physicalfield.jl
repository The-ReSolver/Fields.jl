# This file contains the custom type to define a scalar field in physical space
# for a rotating plane couette flow.

struct PhysicalField{Ny, Nz, Nt, G, T<:Real, A, DEALIAS} <: AbstractArray{T, 3}
    data::A
    grid::G

    PhysicalField{DEALIAS}(grid::Grid{S}, field::AbstractArray{T, 3}) where {DEALIAS, S, T} = new{S..., typeof(grid), T, typeof(field), DEALIAS}(field, grid)
end

# outer constructors
PhysicalField(grid::Grid{S}, fun, dealias::Bool=false, ::Type{T}=Float64) where {T, S} = PhysicalField{dealias}(grid, field_from_function(fun, points(grid), grid.dom, Val(dealias)))
PhysicalField(grid::Grid{S}, dealias::Bool=false, ::Type{T}=Float64) where {S, T<:Real} = PhysicalField(grid, (y,z,t)->zero(T), dealias, T)

function field_from_function(fun, grid_points, dom, ::Val{true})
    Nz, Nt = length.(grid_points[2:3])
    z_padded, t_padded = map(x->(0:(x[2] - 1))*(2π/(dom[x[1]]*x[2])), enumerate(padded_size(Nz, Nt)))
    return fun.(reshape(grid_points[1], :, 1, 1), reshape(z_padded, 1, :, 1), reshape(t_padded, 1, 1, :))
end
field_from_function(fun, grid_points, ::Any, ::Val{false}) = fun.(reshape(grid_points[1], :, 1, 1), reshape(grid_points[2], 1, :, 1), reshape(grid_points[3], 1, 1, :))

# get parent array
Base.parent(u::PhysicalField) = u.data

# define interface
Base.size(u::PhysicalField) = size(parent(u))
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()

# similar
Base.similar(u::PhysicalField{Ny, Nz, Nt, G, T, A, DEALIAS}, ::Type{S}=eltype(u)) where {Ny, Nz, Nt, G, T, A, DEALIAS, S} = PhysicalField(u.grid, DEALIAS, S)
Base.copy(u::PhysicalField) = (v = similar(u); v .= u; v)

# method to extract grid
get_grid(u::PhysicalField) = u.grid


function energy!(K, p::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    q = rfft(parent(p), [2])./Nz
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), ny in 1:Ny
        K[nt] += p.grid.ws[ny]*norm(q[ny, nz, nt])^2
    end
    for nt in 1:Nt, ny in 1:Ny
        K[nt] += 0.5*p.grid.ws[ny]*norm(q[ny, 1, nt])^2
    end
    β = get_β(p)
    K .*= 2π/β
    return K
end
energy(p::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = energy!(zeros(Nt), p)

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
