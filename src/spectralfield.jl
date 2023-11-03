# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

struct SpectralField{Ny, Nz, Nt, G, T, A} <: AbstractArray{Complex{T}, 3}
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
Base.copy(U::SpectralField) = (V = similar(U); V .= U; V)

# methods to allow interface with other packages
Base.zero(U::SpectralField) = zero.(U)
Base.abs(U::SpectralField) = (A = zeros(size(U)); A .= abs.(U); return A)

# method to extract grid
get_grid(U::SpectralField) = U.grid

# inner-product and norm
function LinearAlgebra.dot(p::SpectralField{Ny, Nz, Nt}, q::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # initialise sum variable
    sum = 0.0

    # loop over top half plane exclusive of mean spanwise mode
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), ny in 1:Ny
        sum += p.grid.ws[ny]*real(dot(p[ny, nz, nt], q[ny, nz, nt]))
    end

    # loop over positive temporal modes for mean spanwise mode
    for nt in 2:((Nt >> 1) + 1), ny in 1:Ny
        sum += p.grid.ws[ny]*real(dot(p[ny, 1, nt], q[ny, 1, nt]))
    end

    # evaluate mean component contribution
    for ny in 1:Ny
        sum += 0.5*p.grid.ws[ny]*real(dot(p[ny, 1, 1], q[ny, 1, 1]))
    end

    # extract domain data for scaling
    β = get_β(p)
    ω = get_ω(p)

    return ((8π^2)/(β*ω))*sum
end
LinearAlgebra.norm(p::SpectralField) = sqrt(LinearAlgebra.dot(p, p))

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
