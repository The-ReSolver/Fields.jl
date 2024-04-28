# This file contains the definitions for the Grid custom type that holds
# important information about how to perform operations on a given grid, such
# as inner product/norms and derivatives.

struct Grid{S, T<:AbstractFloat, M<:AbstractMatrix}
    y::Vector{T}
    Dy::NTuple{2, M}
    ws::Vector{T}
    dom::Vector{T}

    function Grid(y::AbstractVector{T}, Nz::Int, Nt::Int, Dy::AbstractMatrix, Dy2::AbstractMatrix, ws::Vector, ω::Real, β::Real) where {T}
        new{(length(y), Nz, Nt), T, typeof(T.(Dy))}(y, (T.(Dy), T.(Dy2)), T.(ws), [T(β), T(ω)])
    end
end

# get points
points(g::Grid{S}) where {S} = (g.y, ntuple(i -> (0:(S[i + 1] - 1))/(S[i + 1])*(2π/g.dom[i]), 2)...)

# get other fields
get_Dy(g::Grid) = g.Dy[1]
get_Dy2(g::Grid) = g.Dy[2]
get_ws(g::Grid) = g.ws
get_β(g::Grid) = g.dom[1]
get_ω(g::Grid) = g.dom[2]

# number of points in grid
Base.size(::Grid{S}) where {S} = S

# overload equality method
Base.:(==)(x::Grid{Sx}, y::Grid{Sy}) where {Sx, Sy} = (x.y == y.y && Sx[2] == Sy[2] && Sx[3] == Sy[3])

interpolate(grid::Grid, S::NTuple{2, Int}) = Grid(grid.y, S..., grid.Dy..., grid.ws, get_ω(grid), get_β(grid))
