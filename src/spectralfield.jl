# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

struct SpectralField{Ny, Nz, Nt, G, T, PROJECTED, A} <: AbstractArray{Complex{T}, 3}
    field::A
    grid::G

    function SpectralField{PROJECTED}(field::AbstractArray{Complex{T}, 3}, grid::Grid{S, T}) where {PROJECTED, S, T}
        all(isodd.(S[2:3])) || throw(ArgumentError("Grid size must be odd!"))
        new{size(field, 1), S[2], S[3], typeof(grid), T, PROJECTED, typeof(field)}(field, grid)
    end
end

# construct field from grid
SpectralField(grid::Grid{S}, ::Type{T}=Float64) where {S, T<:Real} = SpectralField{false}(zeros(Complex{T}, S[1], (S[2] >> 1) + 1, S[3]), grid)

# construct projected field from grid and modes
SpectralField(grid::Grid{S}, modes, ::Type{T}=Float64) where {S, T<:Real} = SpectralField{true}(zeros(Complex{T}, size(modes, 2), (S[2] >> 1) + 1, S[3]), grid)

# special constructor for optimisation read-write
SpectralField(::Any, grid::Grid, modes::Array{ComplexF64, 4}, rest...) = SpectralField(grid, modes)

# define interface
Base.size(U::SpectralField) = size(parent(U))
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()

# get parent array
Base.parent(U::SpectralField) = U.field

# similar and copy
Base.similar(U::SpectralField{<:Any, <:Any, <:Any, <:Any, T, PROJECTED}, ::Type{S}=T) where {T, PROJECTED, S} = SpectralField{PROJECTED}(similar(parent(U), Complex{S}), get_grid(U))
Base.copy(U::SpectralField) = (V = similar(U); V .= U; V)

# methods to allow interface with other packages
Base.zero(U::SpectralField) = zero.(U)
Base.abs(U::SpectralField) = (A = zeros(size(U)); A .= abs.(U); return A)

# method to extract grid
get_grid(U::SpectralField) = U.grid


function LinearAlgebra.dot(p::SpectralField{M, Nz, Nt, <:Any, <:Any, true}, q::SpectralField{M, Nz, Nt, <:Any, <:Any, true}) where {M, Nz, Nt}
    # initialise sum variable
    prod = 0.0

    # loop over top half plane exclusive of mean spanwise mode
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), m in 1:M
        prod += real(dot(p[m, nz, nt], q[m, nz, nt]))
    end

    # loop over positive temporal modes for mean spanwise mode
    for nt in 2:((Nt >> 1) + 1), m in 1:M
        prod += real(dot(p[m, 1, nt], q[m, 1, nt]))
    end

    # evaluate mean component contribution
    for m in 1:M
        prod += 0.5*real(dot(p[m, 1, 1], q[m, 1, 1]))
    end

    # extract domain data for scaling
    β = get_β(p)
    ω = get_ω(p)

    return ((8π^2)/(β*ω))*prod
end

function LinearAlgebra.dot(p::SpectralField{Ny, Nz, Nt, <:Any, <:Any, false}, q::SpectralField{Ny, Nz, Nt, <:Any, <:Any, false}) where {Ny, Nz, Nt}
    # initialise sum variable
    prod = 0.0

    # loop over top half plane exclusive of mean spanwise mode
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), ny in 1:Ny
        prod += p.grid.ws[ny]*real(dot(p[ny, nz, nt], q[ny, nz, nt]))
    end

    # loop over positive temporal modes for mean spanwise mode
    for nt in 2:((Nt >> 1) + 1), ny in 1:Ny
        prod += p.grid.ws[ny]*real(dot(p[ny, 1, nt], q[ny, 1, nt]))
    end

    # evaluate mean component contribution
    for ny in 1:Ny
        prod += 0.5*p.grid.ws[ny]*real(dot(p[ny, 1, 1], q[ny, 1, 1]))
    end

    # extract domain data for scaling
    β = get_β(p)
    ω = get_ω(p)

    return ((8π^2)/(β*ω))*prod
end

LinearAlgebra.norm(p::SpectralField) = sqrt(LinearAlgebra.dot(p, p))
Base.maximum(::Function, gradient::SpectralField) = norm(gradient) # this method exists just so Optim.jl uses the correct norm in the trace


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
