# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

# TODO: Tests
# TODO: Transform between spectral and physical space
# TODO: Grid object to make the discretisation explicit (not completely necessary unless I want to change the discretisation?)

struct SpectraField{Ny, Nz, Nt, T} <: AbstractArray{Complex{T}, 3}
    data::AbstractArray{Complex{T}, 3}

    # constrcut from size and type
    function SpectraField(Ny::Int, Nz::Int, Nt::Int, ::Type{T}=Float64) where {T}
        new{Ny, Nz, Nt, T}(zeros(Ny, Nz, Nt))
    end

    # construct from data
    function SpectraField(v::V) where {T, V<:Union{AbstractArray{Complex{T}, 3}, AbstractArray{T, 3}}}
        shape = size(v)
        new{shape[1], shape[2], shape[3], T}(v)
    end
end

# define interface
Base.size(::SpectraField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)
Base.IndexStyle(::Type{<:SpectraField}) = Base.IndexLinear()
Base.getindex(u::SpectraField, i::Int) = u.data[i]
Base.setindex!(u::SpectraField, v, i::Int) = (u.data[i] = v)
