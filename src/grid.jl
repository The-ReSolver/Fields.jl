# This file contains the definitions for the Grid custom type that holds
# important information about how to perform operations on a given grid, such
# as inner product/norms and derivatives.

export Grid, points

struct Grid{S, T<:AbstractFloat, M<:AbstractMatrix{T}}
    y::Vector{T}
    Dy::NTuple{2, M}
    ws::Vector{T}
    dom::NTuple{2, T}

    function Grid(  y::Vector{T},
                    Nz::Int,
                    Nt::Int,
                    Dy::AbstractMatrix{T},
                    Dy2::AbstractMatrix{T},
                    ws::Vector{T},
                    ω::T,
                    β::T) where {T<:Real}
        if size(Dy)[1] != size(y)[1] || size(Dy)[2] != size(y)[1]
            throw(ArgumentError("Derivative matrix and points vector not compatible!"))
        end
        if size(Dy2) != size(Dy)
            throw(ArgumentError("Derivative matrix and points vector not compatible!"))
        end
        if size(ws)[1] != size(y)[1]
            throw(ArgumentError("Weights vector and points vector not compatible!"))
        end
        new{(size(y)[1], Nz, Nt), T, typeof(Dy)}(y, (Dy, Dy2), ws, (ω, β))
    end
end

# get points
points(g::Grid{S}) where {S} = (g.y, ntuple(i -> (0:(S[i + 1] - 1))/(S[i + 1])*2π, 2)...)
