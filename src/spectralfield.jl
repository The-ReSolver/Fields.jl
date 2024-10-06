# This file contains the custom type to define a scalar field in spectral space
# for a rotating plane couette flow.

struct SpectralField{G, PROJECTED} <: AbstractArray{ComplexF64, 3}
    field::Array{ComplexF64, 3}
    grid::G

    function SpectralField{PROJECTED}(field::AbstractArray{T, 3}, grid::Grid{Ny, Nz, Nt}) where {PROJECTED, T, Ny, Nz, Nt}
        all(isodd.((Nz, Nt))) || throw(ArgumentError("Grid size must be odd!"))
        new{typeof(grid), PROJECTED}(ComplexF64.(field), grid)
    end
end

# outer constructors
SpectralField(grid::Grid{Ny, Nz, Nt}) where {Ny, Nz, Nt} = SpectralField{false}(zeros(ComplexF64, Ny, (Nz >> 1) + 1, Nt), grid)
SpectralField(grid::Grid{Ny, Nz, Nt}, modes) where {Ny, Nz, Nt} = SpectralField{true}(zeros(ComplexF64, size(modes, 2), (Nz >> 1) + 1, Nt), grid)

# define interface
Base.size(U::SpectralField) = size(parent(U))
Base.IndexStyle(::Type{<:SpectralField}) = Base.IndexLinear()

# get parent array
Base.parent(U::SpectralField) = U.field

# similar and copy
Base.similar(U::SpectralField{G, PROJECTED}) where {G, PROJECTED} = SpectralField{PROJECTED}(similar(parent(U)), get_grid(U))
Base.copy(U::SpectralField) = (V = similar(U); V .= U; V)

# methods to allow interface with other packages
Base.zero(U::SpectralField) = zero.(U)
Base.abs(U::SpectralField) = (A = zeros(size(U)); A .= abs.(U); return A)

# method to extract grid
get_grid(U::SpectralField) = U.grid


function LinearAlgebra.dot(p::SpectralField{<:Grid{Ny, Nz, Nt}, true}, q::SpectralField{<:Grid{Ny, Nz, Nt}, true}) where {Ny, Nz, Nt}
    # initialise sum variable
    prod = 0.0

    # loop over top half plane exclusive of mean spanwise mode
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), m in axes(p, 1)
        prod += real(dot(p[m, nz, nt], q[m, nz, nt]))
    end

    # loop over positive temporal modes for mean spanwise mode
    for nt in 2:((Nt >> 1) + 1), m in axes(p, 1)
        prod += real(dot(p[m, 1, nt], q[m, 1, nt]))
    end

    # evaluate mean component contribution
    for m in axes(p, 1)
        prod += 0.5*real(dot(p[m, 1, 1], q[m, 1, 1]))
    end

    # extract domain data for scaling
    β = get_β(p)
    ω = get_ω(p)

    return ((8π^2)/(β*ω))*prod
end

function LinearAlgebra.dot(p::SpectralField{<:Grid{Ny, Nz, Nt}, false}, q::SpectralField{<:Grid{Ny, Nz, Nt}, false}) where {Ny, Nz, Nt}
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

function LinearAlgebra.dot(p::SpectralField{<:Grid{Ny, Nz, Nt}, false}, q::SpectralField{<:Grid{Ny, Nz, Nt}, false}, A::NormScaling) where {Ny, Nz, Nt}
    prod = 0.0
    ws = get_ws(p)
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), ny in 1:Ny
        prod += A[ny, nz, nt]*ws[ny]*real(dot(p[ny, nz, nt], q[ny, nz, nt]))
    end
    for nt in 2:((Nt >> 1) + 1), ny in 1:Ny
        prod += A[ny, 1, nt]*ws[ny]*real(dot(p[ny, 1, nt], q[ny, 1, nt]))
    end
    for ny in 1:Ny
        prod += 0.5*A[ny, 1, 1]*ws[ny]*real(dot(p[ny, 1, 1], q[ny, 1, 1]))
    end
    return ((8π^2)/(get_β(p)*get_ω(p)))*prod
end

function LinearAlgebra.dot(p::SpectralField{<:Grid{Ny, Nz, Nt}, true}, q::SpectralField{<:Grid{Ny, Nz, Nt}, true}, A::NormScaling) where {Ny, Nz, Nt}
    prod = 0.0
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), m in axes(p, 1)
        prod += A[m, nz, nt]*real(dot(p[m, nz, nt], q[m, nz, nt]))
    end
    for nt in 2:((Nt >> 1) + 1), m in axes(p, 1)
        prod += A[m, 1, nt]*real(dot(p[m, 1, nt], q[m, 1, nt]))
    end
    for m in axes(p, 1)
        prod += 0.5*A[m, 1, 1]*real(dot(p[m, 1, 1], q[m, 1, 1]))
    end
    return ((8π^2)/(get_β(p)*get_ω(p)))*prod
end

LinearAlgebra.norm(p::SpectralField) = sqrt(LinearAlgebra.dot(p, p))
LinearAlgebra.norm(p::SpectralField, A) = sqrt(LinearAlgebra.dot(p, p, A))
Base.maximum(::Function, gradient::SpectralField) = norm(gradient) # this method exists just so Optim.jl uses the correct norm in the trace

function LinearAlgebra.mul!(v::SpectralField{<:Grid{Ny, Nz, Nt}, false}, A::NormScaling, u::SpectralField{<:Grid{Ny, Nz, Nt}, false}) where {Ny, Nz, Nt}
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), ny in 1:Ny
        v[ny, nz, nt] = A[ny, nz, nt]*u[ny, nz, nt]
    end
    for nt in 2:((Nt >> 1) + 1), ny in 1:Ny
        v[ny, 1, nt] = A[ny, 1, nt]*u[ny, 1, nt]
        v[ny, 1, end - nt + 2] = A[ny, 1, nt]*u[ny, 1, end - nt + 2]
    end
    for ny in 1:Ny
        v[ny, 1, 1] = A[ny, 1, 1]*u[ny, 1, 1]
    end
    return v
end

function LinearAlgebra.mul!(v::SpectralField{<:Grid{Ny, Nz, Nt}, true}, A::NormScaling, u::SpectralField{<:Grid{Ny, Nz, Nt}, true}) where {Ny, Nz, Nt}
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), ny in axes(u, 1)
        v[ny, nz, nt] = A[ny, nz, nt]*u[ny, nz, nt]
    end
    for nt in 2:((Nt >> 1) + 1), ny in axes(u, 1)
        v[ny, 1, nt] = A[ny, 1, nt]*u[ny, 1, nt]
        v[ny, 1, end - nt + 2] = A[ny, 1, nt]*u[ny, 1, end - nt + 2]
    end
    for ny in axes(u, 1)
        v[ny, 1, 1] = A[ny, 1, 1]*u[ny, 1, 1]
    end
    return v
end

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
