# This file will contain the custom type to define a scalar field in a rotating
# plane couette flow.

# TODO: Physical and spectral fields (with transform)
# TODO: Rename package for just fields

struct SpectralField{Ny, Nz, Nt, T} <: AbstractArray{Complex{T}, 3}
    data::AbstractArray{Complex{T}, 3}

    # constrcut from size and type
    function SpectralField(Ny::Int, Nz::Int, Nt::Int, ::Type{T}=Float64) where {T}
        new{Ny, Nz, Nt, T}(zeros(Ny, Nz, Nt))
    end

    # construct from given data
    function SpectralField(v::V) where {T, V<:AbstractArray{T, 3}}
        shape = size(v)
        new{shape[1], shape[2], shape[3], T}(v)
    end
end

# define interface
Base.size(u::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()
Base.getindex(u::SpectralField, i::Int) = u.data[i]
Base.setindex!(u::SpectralField, v, i::Int) = (u.data[i] = v)
