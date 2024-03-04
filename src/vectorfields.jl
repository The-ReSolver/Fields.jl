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
VectorField(grid::Grid, dealias::Bool=true; N::Int=3) = VectorField([PhysicalField(grid, dealias) for i in 1:N]...)

# outer constructor based on grid and functions
VectorField(grid::Grid, funcs::Vararg) = VectorField([PhysicalField(grid, funcs[i]) for i in 1:length(funcs)]...)

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
LinearAlgebra.norm(q::VectorField) = sqrt(LinearAlgebra.dot(q, q))

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

# TODO: check if this broadcasts over all the arrays or goes into the underlying arrays
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
