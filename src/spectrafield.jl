# This file will contain the custom type to define a scalar field in a rotating
# plane couette flow.

# TODO: Physical and spectral fields (with transform)
# TODO: Rename package for just fields

struct SpectraField{Ny, Nz, Nt, T} <: AbstractArray{Complex{T}, 3}
    data::AbstractArray{Complex{T}, 3}

    # constrcut from size and type
    function SpectraField(Ny::Int, Nz::Int, Nt::Int, ::Type{T}=Float64) where {T}
        new{Ny, Nz, Nt, T}(zeros(Ny, Nz, Nt))
    end

    # construct from given data
    function SpectraField(v::V) where {T, V<:AbstractArray{T, 3}}
        shape = size(v)
        new{shape[1], shape[2], shape[3], T}(v)
    end
end

# define interface
Base.size(u::SpectraField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)
Base.IndexStyle(::Type{<:SpectraField}) = Base.IndexLinear()
Base.getindex(u::SpectraField, i::Int) = u.data[i]
Base.setindex!(u::SpectraField, v, i::Int) = (u.data[i] = v)
