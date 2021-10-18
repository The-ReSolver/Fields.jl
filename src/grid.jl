# This file contains the definitions for the Grid custom type that holds
# important information about how to perform operations on a given grid, such
# as inner product/norms and derivatives.

export Grid

struct Grid{S::NTuple{N, Int}, T<:Real}
    psy::Vector{T}
    Dy::Matrix{T}
    ws::Vector{T}

    function Grid(  y::Vector{T},
                    Nz::int,
                    Nt::Int,
                    Dy::Matrix{T},
                    ws::Vector{T}) where {T<:Real}
        new{(size(y)[1], Nz, Nt)}(psy, Dy, ws)
    end
end
