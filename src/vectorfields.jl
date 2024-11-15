# This file contains the custom type to define a vector field which stores
# either spectral or physical space fields.

struct VectorField{N, S} <: AbstractVector{S}
    elements::Vector{S}

    # construct using scalar fields as arguments
    function VectorField(elements::Vararg{S, N}) where {T<:Number, D, S<:AbstractArray{T, D}, N}
        new{N, typeof(elements[1])}(collect(elements))
    end
end

# outer constructor based off grid
VectorField(grid::Grid; N::Int=3, fieldType::Type{T}=SpectralField) where {T<:Union{SpectralField, PhysicalField}} = VectorField([fieldType(grid) for i in 1:N]...)
VectorField(grid::Grid, dealias::Bool; N::Int=3, pad_factor::Real=3/2) = VectorField([PhysicalField(grid, dealias, pad_factor=pad_factor) for i in 1:N]...)

# outer constructor based on grid and functions
VectorField(grid::Grid, funcs::Vararg; dealias::Bool=false) = VectorField([PhysicalField(grid, funcs[i], dealias) for i in 1:length(funcs)]...)

# define index style
Base.IndexStyle(::Type{<:VectorField}) = Base.IndexLinear()

# extract/set i-th component
Base.getindex(q::VectorField, i::Int) = q.elements[i]
Base.setindex!(q::VectorField, v, i::Int) = (q.elements[i] = v)

# these might not be needed, but
Base.size(::VectorField{N}) where {N} = (N,)
Base.length(::VectorField{N}) where {N} = N

# construct a new object from an existing one
Base.similar(q::VectorField) = VectorField(similar.(q.elements)...)
Base.similar(q::VectorField, L::Int, ::Type{S}=SpectralField) where {S} = VectorField([S(get_grid(q)) for _ in 1:L]...)
Base.copy(q::VectorField) = copy.(q)

# methods to allow interface with other packages
Base.zero(q::VectorField) = zero.(q)
Base.vcat(q::VectorField, p::VectorField) = VectorField(q..., p...)

# inner-product and norm
LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}) where {N} = sum(LinearAlgebra.dot(q[i], p[i]) for i = 1:N)
LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}, A::NormScaling) where {N} = sum(LinearAlgebra.dot(q[i], p[i], A) for i = 1:N)
LinearAlgebra.norm(q::VectorField) = sqrt(LinearAlgebra.dot(q, q))
LinearAlgebra.norm(q::VectorField, A::NormScaling) = sqrt(LinearAlgebra.dot(q, q, A))

function LinearAlgebra.mul!(q::VectorField{N}, A::NormScaling, p::VectorField{N}) where {N}
    for i in 1:N
        LinearAlgebra.mul!(q[i], A, p[i])
    end
    return q
end

# cross product with constant vector
function cross!(vu::VectorField{3, S}, v::AbstractVector, u::VectorField{3, S}) where {S}
    @. vu[1] += v[2]*u[3] - v[3]*u[2]
    @. vu[2] += v[3]*u[1] - v[1]*u[3]
    @. vu[3] += v[1]*u[2] - v[2]*u[1]
    return vu
end

function cross_k!(ku::VectorField{3, S}, u::VectorField{3, S}, mag::Float64) where {S}
    @. ku[1] -= mag*u[2]
    @. ku[2] += mag*u[1]
end

# method to extract grid
get_grid(q::VectorField) = get_grid(q[1])

# define union type of field types
const AllFields = Union{SpectralField, PhysicalField, VectorField}

# method for grid comparison of fields
grideq(U::AllFields, V::AllFields) = (get_grid(U) == get_grid(V))

# extract grid fields from parent field
get_Dy(U::AllFields) = get_Dy(get_grid(U))
get_Dy2(U::AllFields) = get_Dy2(get_grid(U))
get_ws(U::AllFields) = get_ws(get_grid(U))
get_ω(U::AllFields) = get_ω(get_grid(U))
get_β(U::AllFields) = get_β(get_grid(U))

# ~ BROADCASTING ~
# taken from MultiscaleArrays.jl
const VectorFieldStyle = Broadcast.ArrayStyle{VectorField}
Base.BroadcastStyle(::Type{<:VectorField}) = Broadcast.ArrayStyle{VectorField}()

# for broadcasting to construct new objects
Base.similar(bc::Base.Broadcast.Broadcasted{VectorFieldStyle}, ::Type{T}) where {T} = VectorField(similar.(find_field(bc).elements)...)

# f = find_field(bc)` returns the first VectorField among the arguments
find_field(bc::Base.Broadcast.Broadcasted) = find_field(bc.args)
find_field(args::Tuple) = find_field(find_field(args[1]), Base.tail(args))
find_field(a::VectorField, rest) = a
find_field(a::Base.Broadcast.Extruded{<:VectorField}, rest) = a.x
find_field(a::SpectralField, rest) = a
find_field(a::PhysicalField, rest) = a
find_field(::Any, rest) = find_field(rest)
find_field(x) = x
find_field(::Tuple{}) = nothing

@inline function Base.copyto!(dest::VectorField{N}, bc::Broadcast.Broadcasted{VectorFieldStyle}) where {N}
    for i in 1:N
        copyto!(dest.elements[i], unpack(bc, i))
    end
    return dest
end

@inline unpack(bc::Broadcast.Broadcasted, i) = Broadcast.Broadcasted(bc.f, _unpack(i, bc.args))
@inline unpack(x::Any,         i) = x
@inline unpack(x::VectorField, i) = x.elements[i]

@inline _unpack(i, args::Tuple) = (unpack(args[1], i), _unpack(i, Base.tail(args))...)
@inline _unpack(i, args::Tuple{Any}) = (unpack(args[1], i),)
@inline _unpack(::Any, args::Tuple{}) = ()


extendDomain(u::VectorField{N, <:PhysicalField{<:Grid{Ny, Nz, Nt}}}, zpadfactor::Float64, tpadfactor::Float64) where {N, Ny, Nz, Nt} = extendDomain(u, ceil(Int, zpadfactor*Nz), ceil(Int, tpadfactor*Nt))
function extendDomain(u::VectorField{N, <:PhysicalField}, Nze::Int, Nte::Int) where {N}
    u_extended = VectorField(extendDomain(get_grid(u), Nze, Nte), fieldType=PhysicalField)
    for i in 1:N
        extendDomain!(u_extended[i], u[i])
    end
    return u_extended
end

function interpolate(u::VectorField{N}, Nz::Int, Nt::Int) where {N}
    v = VectorField(interpolate(get_grid(u), Nz, Nt))
    for n in 1:N
        interpolate!(v[n], u[n])
    end
    return v
end

extendDomain(u::PhysicalField{<:Grid{Ny, Nz, Nt}}, zpadfactor::Float64, tpadfactor::Float64) where {Ny, Nz, Nt} = extendDomain(u, ceil(Int, zpadfactor*Nz), ceil(Int, tpadfactor*Nt))
extendDomain(u::PhysicalField, Nze::Int, Nte::Int) = extendDomain!(PhysicalField(extendDomain(get_grid(u), Nze, Nte)), u)
function extendDomain!(u_extended::PhysicalField{<:Grid{Ny, Nze, Nte}}, u::PhysicalField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nze, Nt, Nte}
    for nt in 1:Nte, nz in 1:Nze
        u_extended[:, nz, nt] .= u[:, ((nz - 1) % Nz) + 1, ((nt - 1) % Nt) + 1]
    end
    return u_extended
end


# TODO: test these
function writeField(path, u::SpectralField{<:Grid{Ny, Nz, Nt}, PROJECTED}) where {Ny, Nz, Nt, PROJECTED}
    data = open(path, "w") do f
        write(f, Ny)
        write(f, Nz)
        write(f, Nt)
        write(f, points(get_grid(u))[1])
        write(f, get_Dy(u))
        write(f, get_Dy2(u))
        write(f, get_ws(u))
        write(f, get_β(u))
        write(f, get_ω(u))
        write(f, parent(u))
        write(f, PROJECTED)
    end
    return data
end

function readField(path)
    u = open(path, "r") do f
        Ny = read(f, Int)
        Nz = read(f, Int)
        Nt = read(f, Int)
        y = read!(f, Vector{Float64}(undef, Ny))
        Dy = read!(f, Matrix{Float64}(undef, Ny))
        Dy2 = read!(f, Matrix{Float64}(undef, Ny))
        ws = read!(f, Vector{Float64}(undef, Ny))
        β = read(f, Float64)
        ω = read(f, Float64)
        coeffs = read!(f, Array{ComplexF64, 3}(undef, Ny, (Nz >> 1) + 1, Nt))
        projected = read(f, Bool)
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
        return SpectralField{projected}(coeffs, grid)
    end
    return u
end

