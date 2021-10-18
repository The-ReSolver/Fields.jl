# This file contains the test set for the spectral scalar field type
# definition.

@testset "Physical Scalar Field         " begin
    # take random variables
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    y = rand(Float64, Ny)
    Dy = rand(Float64, (Ny, Ny))
    ws = rand(Float64, Ny)
    grid = Grid(y, Nz, Nt, Dy, ws)

    # intialise using different constructors
    @test   typeof(PhysicalField(Ny, Nz, Nt, grid)) ==
            PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}
    @test   typeof(PhysicalField(grid)) ==
            PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}
    @test   typeof(PhysicalField(ones(Ny, Nz, Nt), grid)) ==
            PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}
end
