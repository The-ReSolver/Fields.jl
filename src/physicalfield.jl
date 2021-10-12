# This file contains the custom type to define a scalar field in physical space
# for a rotating plane couette flow.

struct PhysicalField{Ny, Nz, Nt, T} <: AbstractArray{T, 3}
    data::AbstractArray{T, 3}

    # construct from size and type
    function PhysicalField(Ny::Int, Nz::Int, Nt::Int, ::Type{T}=Float64) where {T<:Real}
        new{Ny, Nz, Nt, T}(zeros(Ny, Nz, Nt))
    end

    # construct from data
    function PhysicalField(v::V) where {T<:Real, V<:AbstractArray{T, 3}}
        shape = size(v)
        new{shape[1], shape[2], shape[3], T}(v)
    end
end

# define interface
Base.size(::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)
Base.IndexStyle(::Type{<:PhysicalField}) = Base.IndexLinear()
Base.getindex(u::PhysicalField, i::Int) = u.data[i]
Base.setindex!(u::PhysicalField, v, i::Int) = (u.data[i] = v)
