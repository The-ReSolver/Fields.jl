# This file contains the definitions for the Grid custom type that holds
# important information about how to perform operations on a given grid, such
# as inner product/norms and derivatives.

export Grid

struct Grid{S, T<:Real}
    psy::Vector{T}
    Dy::Matrix{T}
    ws::Vector{T}

    function Grid(  y::Vector{T},
                    Nz::Int,
                    Nt::Int,
                    Dy::Matrix{T},
                    ws::Vector{T}) where {T<:Real}
        if size(Dy)[1] != size(y)[1] || size(Dy)[2] != size(y)[1]
            throw(ArgumentError("Derivative matrix and points vector not compatible!"))
        end
        if size(ws)[1] != size(y)[1]
            throw(ArgumentError("Weights vector and points vector not compatible!"))
        end
        new{(size(y)[1], Nz, Nt), T}(y, Dy, ws)
    end
end
