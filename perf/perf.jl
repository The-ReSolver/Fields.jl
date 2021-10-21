using Fields, BenchmarkTools

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# test broadcasting speed of single field
Ny, Nz, Nt = 12, 12, 12
grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny))

a = PhysicalField(grid)
b = PhysicalField(grid)
c = PhysicalField(grid)

foo!(a, b, c) = (@.  a = 3 * b + c / 2)

# check broadcasting on field is as fast as on the underlying array
@btime foo!($a, $b, $c)
# @btime foo!($(parent(a)), $(parent(b)), $(parent(c)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# test broadcasting speed of vector field
a = VectorField(PhysicalField(grid), PhysicalField(grid), PhysicalField(grid))
b = VectorField(PhysicalField(grid), PhysicalField(grid), PhysicalField(grid))
c = VectorField(PhysicalField(grid), PhysicalField(grid), PhysicalField(grid))

# this time should be three times the time of a single field
@btime foo!($a, $b, $c)


