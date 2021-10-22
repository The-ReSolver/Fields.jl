# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

export SpectralField

struct SpectralField{Ny, Nz, Nt, G, T<:Real, A<:AbstractArray{Complex{T}, 3}} <: AbstractArray{Complex{T}, 3}
    data::A
    grid::G

    # construct from grid
    function SpectralField(grid::Grid{S}, ::Type{T}=Float64) where {S, T<:Real}
        data = zeros(Complex{T}, S[1], (S[2] >> 1) + 1, S[3])
        return new{S[1], S[2], S[3], typeof(grid), T, typeof(data)}(data, grid)
    end
end

# define interface
Base.size(::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, (Nz >> 1) + 1, Nt)
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()

# get parent array
Base.parent(U::SpectralField) = U.data

# similar
Base.similar(U::SpectralField{Ny, Nz, Nt, G, T}, ::Type{S}=T) where {Ny, Nz, Nt, G, T, S} = SpectralField(U.grid, S)

# ~ BROADCASTING ~
# taken from MultiscaleArrays.jl
const SpectralFieldStyle = Broadcast.ArrayStyle{SpectralField}
Base.BroadcastStyle(::Type{<:SpectralField}) = Broadcast.ArrayStyle{SpectralField}()

# for broadcasting to construct new objects
Base.similar(bc::Base.Broadcast.Broadcasted{SpectralFieldStyle}, ::Type{T}) where {T} =
    similar(find_field(bc))

Base.@propagate_inbounds function Base.getindex(U::SpectralField, I...)
    @boundscheck checkbounds(parent(U), I...)
    @inbounds ret = parent(U)[I...]
    return ret
end

Base.@propagate_inbounds function Base.setindex!(U::SpectralField, v, I...)
    @boundscheck checkbounds(parent(U), I...)
    @inbounds parent(U)[I...] = v
    return v
end
