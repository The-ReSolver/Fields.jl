import LinearAlgebra

export VectorField

# TODO: Restrict this to only take subtypes of abtract arrays as type inputs to the constructor

# Just a wrapper around a N-tuple of fields
struct VectorField{N, S} <: AbstractVector{S}
    elements::NTuple{N, S}

    # construct using scalar fields as arguments
    function VectorField(elements::Vararg{S, N}) where {S, N}
        return new{N, S}(elements)
    end
end

# extract i-th component
Base.getindex(q::VectorField, i::Int) = q.elements[i]

# these might not be needed, but
Base.size(::VectorField{N}) where {N} = (N,)
Base.length(::VectorField{N}) where {N} = N

# construct a new object from an existing one
Base.similar(q::VectorField) = VectorField(similar.(q.elements)...)

# A simple norm
LinearAlgebra.norm(q::VectorField) = sqrt(dot(q, q))
LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}) where {N} =  sum(dot(q[i], p[i]) for i = 1:N)

# ~ BROADCASTING ~
# taken from MultiscaleArrays.jl
const VectorFieldStyle = Broadcast.ArrayStyle{VectorField}
Base.BroadcastStyle(::Type{<:VectorField}) = Broadcast.ArrayStyle{VectorField}()

@inline function Base.copyto!(dest::VectorField{N},
    bc::Broadcast.Broadcasted{VectorFieldStyle}) where {N}
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
