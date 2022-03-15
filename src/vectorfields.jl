export VectorField

# Just a wrapper around a N-tuple of fields
struct VectorField{N, S} <: AbstractVector{S}
    elements::NTuple{N, S}

    # construct using scalar fields as arguments
    function VectorField(elements::Vararg{S, N}) where {T<:Number, D, S<:AbstractArray{T, D}, N}
        new{N, S}(elements)
    end

    function VectorField(grid::Grid, ::Type{T}=Float64; N::Int=3, field_type::Symbol=:Spectral) where {T}
        if field_type == :Spectral
            fields = [SpectralField(grid, T) for i in 1:N]
        elseif field_type == :Physical
            fields = [PhysicalField(grid, T) for i in 1:N]
        end
        new{N, typeof(fields[1])}(Tuple(fields))
    end
end

# extract/set i-th component
Base.getindex(q::VectorField, i::Int) = q.elements[i]

# these might not be needed, but
Base.size(::VectorField{N}) where {N} = (N,)
Base.length(::VectorField{N}) where {N} = N

# construct a new object from an existing one
Base.similar(q::VectorField) = VectorField(similar.(q.elements)...)
Base.copy(q::VectorField) = VectorField([copy(q[i]) for i in 1:size(q)[1]]...)

# inner-product and norm
LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}) where {N} =  sum(LinearAlgebra.dot(q[i], p[i]) for i = 1:N)
LinearAlgebra.norm(q::VectorField) = sqrt(LinearAlgebra.dot(q, q))

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
