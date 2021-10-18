# This file contains the test set for the vector field type definition.


@testset "Vector Field                  " begin
    # generate random arrays
    # Ny = rand(3:50)
    # Nz = rand(3:50)
    # Nt = rand(3:50)
    Ny = 2
    Nz = 2
    Nt = 2
    u1 = rand(Float64, (Ny, Nz, Nt))
    u2 = rand(Float64, (Ny, Nz))
    v1 = rand(Float64, (Ny, Nz, Nt))
    v2 = rand(Int, (Ny, Nz, Nt))
    w1 = rand(Float64, (Ny, Nz, Nt))
    w2 = "String"

    # initialise
    @test typeof(VectorField(u1, v1)) == VectorField{2, Array{Float64, 3}}
    @test typeof(VectorField(u1, v1, w1)) == VectorField{3, Array{Float64, 3}}

    # catch errors on constructors
    @test_throws MethodError VectorField(u2, v1)
    @test_throws MethodError VectorField(u1, v2, w1)
    @test_throws MethodError VectorField(u1, v1, w2)
    @test_throws MethodError VectorField("string1", "string2")
end
