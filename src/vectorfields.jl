# This file contains the custom type to define a vector field which stores
# either spectral or physical space fields.

# Just a wrapper around a N-tuple of fields
struct VectorField{N, S} <: AbstractVector{S}
    # elements::NTuple{N, S}
    elements::Vector{S}

    # construct using scalar fields as arguments
    # function VectorField(elements::Vararg{S, N}) where {T<:Number, D, S<:AbstractArray{T, D}, N}
    function VectorField(elements::Vararg{<:AbstractArray{T, D}, N}) where {T<:Number, D, N}
        # new{N, S}(elements)
        new{N, typeof(elements[1])}(collect(elements))
    end
end

# outer constructor based off grid
function VectorField(grid::Grid; N::Int=3, field_type::Symbol=:spectral)
    field_constructor_expr = Expr(:call, Symbol(field_type, :field), grid)
    fields = eval(:([$field_constructor_expr for i in 1:$N]))
    VectorField(fields...)
end

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
Base.copy(q::VectorField) = VectorField([copy(q[i]) for i in 1:size(q)[1]]...)

# inner-product and norm
LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}) where {N} =  sum(LinearAlgebra.dot(q[i], p[i]) for i = 1:N)
LinearAlgebra.norm(q::VectorField) = sqrt(LinearAlgebra.dot(q, q))

# method to extract grid
get_grid(q::VectorField) = get_grid(q[1])

# define union type of field types
AllFields = Union{SpectralField, PhysicalField, VectorField}

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
Base.similar(bc::Base.Broadcast.Broadcasted{VectorFieldStyle}, ::Type{T}) where {T} =
    VectorField(similar.(find_field(bc).elements)...)

# f = find_field(bc)` returns the first VectorField among the arguments
find_field(bc::Base.Broadcast.Broadcasted) = find_field(bc.args)
find_field(args::Tuple) = find_field(find_field(args[1]), Base.tail(args))
find_field(a::VectorField, rest) = a
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
