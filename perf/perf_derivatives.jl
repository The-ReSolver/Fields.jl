using Fields
using ChebUtils
using BenchmarkTools
using FDGrids

# NOTE: the workflow that works for me:
#   - include all the source files manually
#   - add and import all the required modules for those source files
#   - run file from CL using "julia --project=./perf ./perf/perf_derivatives.jl"

# initialise constants
Ny = 64
Nz = 64
Nt = 64
y = chebpts(Ny)
# Dy = chebdiff(Ny); Dy2 = chebddiff(Ny)
Dy = DiffMatrix(y, 5, 1); Dy2 = DiffMatrix(y, 5, 2)
ω = 1.0
β = 1.0
grid = Grid(y, Nz, Nt, Dy, Dy2, zeros(Ny), ω, β)
FFT! = FFTPlan!(grid)
fun(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))

# initialise fields
u_phys = PhysicalField(grid, fun)
u = SpectralField(grid)
FFT!(u, u_phys)
dudy = SpectralField(grid)
d2udy2 = SpectralField(grid)
dudz = SpectralField(grid)
d2udz2 = SpectralField(grid)
dudt = SpectralField(grid)

# benchmark derivative functions
@btime ddy!(u, dudy)
@btime d2dy2!(u, d2udy2)
@btime ddz!(u, dudz)
@btime d2dz2!(u, d2udz2)
@btime ddt!(u, dudt)
