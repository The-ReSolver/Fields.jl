# This file contains the test set for the grid custom type.

@testset "Field Grid" begin
    # generate random inputs
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    randD = rand([-2, -1, 1, 2])
    y = rand(Float64, Ny)
    D1 = rand(Float64, (Ny, Ny))
    D2 = rand(Float64, (Ny + randD, Ny + randD))
    w1 = rand(Float64, Ny)
    w2 = rand(Float64, Ny + randD)

    # initialise correct ones
    @test typeof(Grid(y, Nz, Nt, D1, w1)) == Grid{(Ny, Nz, Nt), Float64}

    # initialise bad versions to catch errors
    @test_throws ArgumentError Grid(y, Nz, Nt, D2, w1)
    @test_throws ArgumentError Grid(y, Nz, Nt, D1, w2)
    @test_throws ArgumentError Grid(y, Nz, Nt, D2, w2)
    @test_throws MethodError Grid(y, Nz, Nt, D1, rand(Int, Ny))
    @test_throws MethodError Grid(y, Nz, Nt, rand(Int, (Ny, Ny)), w1)
end