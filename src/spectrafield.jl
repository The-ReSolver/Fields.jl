# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

# TODO: Tests
# TODO: Transform between spectral and physical space
# TODO: Grid object to make the discretisation explicit (not completely necessary unless I want to change the discretisation?)

struct SpectralField{Ny, Nz, Nt, T, A<:AbstractArray{Complex{T}, 3}} <: AbstractArray{Complex{T}, 3}
    data::A

    # constrcut from size and type
    function SpectralField(Ny::Int, Nz::Int, Nt::Int, ::Type{T}=Float64) where {T}
        data = zeros(Complex{T}, Ny, Nz, Nt)
        new{Ny, Nz, Nt, T, typeof(data)}(data)
    end

    # construct from data
    function SpectralField(data::A) where {T, A<:AbstractArray{Complex{T}, 3}}
        shape = size(data)
        new{shape[1], shape[2], shape[3], T, A}(dara)
    end
end

# define interface
Base.size(::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()
Base.getindex(u::SpectralField, i::Int) = u.data[i]
Base.setindex!(u::SpectralField, v, i::Int) = (u.data[i] = v)
