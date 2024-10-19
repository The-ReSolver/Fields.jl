# This file contains the definitions for the Grid custom type that holds
# important information about how to perform operations on a given grid, such
# as inner product/norms and derivatives.

struct Grid{Ny, Nz, Nt, D<:AbstractMatrix}
    y::Vector{Float64}
    Dy::NTuple{2, D}
    ws::Vector{Float64}
    dom::Vector{Float64}

    Grid(y::Vector, Nz::Int, Nt::Int, Dy::D, Dy2::D, ws::Vector, ω::Real, β::Real) where {D} = new{length(y), Nz, Nt, D}(y, (Dy, Dy2), Float64.(ws), Float64[β, ω])
end

# get points
points(g::Grid{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (g.y, ntuple(i -> (0:((Nz, Nt)[i] - 1))/((Nz, Nt)[i])*(2π/g.dom[i]), 2)...)

# get other fields
get_Dy(g::Grid) = g.Dy[1]
get_Dy2(g::Grid) = g.Dy[2]
get_ws(g::Grid) = g.ws
get_β(g::Grid) = g.dom[1]
get_ω(g::Grid) = g.dom[2]

# number of points in grid
Base.size(::Grid{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)

# overload equality method
Base.:(==)(x1::Grid{Ny1, Nz1, Nt1}, x2::Grid{Ny2, Nz2, Nt2}) where {Ny1, Nz1, Nt1, Ny2, Nz2, Nt2} = (x1.y == x2.y && Nz1 == Nz2 && Nt1 == Nt2)

interpolate(grid::Grid, Nz::Int, Nt::Int) = Grid(grid.y, Nz, Nt, get_Dy(grid), get_Dy2(grid), get_ws(grid), get_ω(grid), get_β(grid))
function extend(grid::Grid{Ny, Nz, Nt}, Nze::Int, Nte::Int) where {Ny, Nz, Nt}
    β = get_β(grid)*Nz/Nze
    ω = get_ω(grid)*Nt/Nte
    return Grid(grid.y, Nze, Nte, get_Dy(grid), get_Dy2(grid), get_ws(grid), ω, β)
end
