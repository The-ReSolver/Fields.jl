# This file contains the definitions for the Grid custom type that holds
# important information about how to perform operations on a given grid, such
# as inner product/norms and derivatives.

struct Grid{S, T<:AbstractFloat, M<:AbstractMatrix}
    y::Vector{T}
    Dy::NTuple{2, M}
    ws::Vector{T}
    dom::NTuple{2, T}

    function Grid(y::Vector{<:Real}, Nz::Int, Nt::Int, Dy::AbstractMatrix{<:Real}, Dy2::AbstractMatrix{<:Real}, ws::Vector{<:Real}, ω::Real, β::Real)
        T = eltype(y)
        new{(size(y)[1], Nz, Nt), T, typeof(T.(Dy))}(y, (T.(Dy), T.(Dy2)), T.(ws), (T(β), T(ω)))
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

# method to construct field out of grid
spectralfield(g::Grid{S, T}) where {S, T} = SpectralField(g, T)
physicalfield(g::Grid{S, T}) where {S, T} = PhysicalField(g, T)
physicalfield(g::Grid{S, T}, fun) where {S, T} = PhysicalField(g, fun, T)
vectorfield(g::Grid{S, T}; N::Int=3, field_type::Symbol=:spectral) where {S, T} = VectorField(g; N=N, field_type=field_type)
vectorfield(g::Grid, funcs...) = VectorField([physicalfield(g, funcs[i]) for i in 1:size(funcs, 1)]...)
vectorfield(g::Grid) = vectorfield(g; N=3, field_type=:spectral)
