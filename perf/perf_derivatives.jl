using Fields
using ChebUtils
using BenchmarkTools
using Random

# initialise constants
Ny = 64
Nz = 64
Nt = 64
Dy = chebdiff(Ny)
Dy2 = chebddiff(Ny)
ω = 1.0
β = 1.0
grid = Grid(rand(Ny), Nz, Nt, Dy, Dy2, rand(Ny), ω, β)
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
