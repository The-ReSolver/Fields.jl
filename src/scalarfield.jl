# This file will contain the custom type to define a scalar field in a rotating
# plane couette flow.

# TODO: Make it an array subtype and define interface

struct ScalarField{T, Ny, Nz, Nt}
    data::AbstractArray{Complex{T}}

    # constrcut from size and type
    function ScalarField(Ny::Int, Nz::Int, Nt::Int, ::Type{T}=Float64) where {T}
        new{T, Ny, Nz, Nt}(zeros(Ny, Nz, Nt))
    end

    # construct from given data
    function ScalarField(v::V) where {T, V<:AbstractArray{T}}
        shape = size(v)
        new{T, shape[1], shape[2], shape[3]}(v)
    end
end
