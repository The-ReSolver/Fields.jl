@testset "Vector Field Constructor      " begin
    # generate random arrays
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    u1 = rand(Float64, (Ny, Nz, Nt))
    u2 = rand(Float64, (Ny, Nz))
    v1 = rand(Float64, (Ny, Nz, Nt))
    v2 = rand(Int, (Ny, Nz, Nt))
    w1 = rand(Float64, (Ny, Nz, Nt))
    w2 = "String"
    ω = abs(randn())
    β = abs(randn())

    # initialise grid
    grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny), ω, β)

    # initialise
    @test typeof(VectorField(u1, v1)) == VectorField{2, Array{Float64, 3}}
    @test typeof(VectorField(u1, v1, w1)) == VectorField{3, Array{Float64, 3}}
    @test typeof(VectorField(grid)) == VectorField{3, SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{Complex{Float64}, 3}}}
    @test typeof(VectorField(grid; N=2)) == VectorField{2, SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{Complex{Float64}, 3}}}
    @test typeof(VectorField(grid, N=5, field_type=:Physical)) == VectorField{5, PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}}

    # catch errors on constructors
    @test_throws MethodError VectorField(u2, v1)
    @test_throws MethodError VectorField(u1, v2, w1)
    @test_throws MethodError VectorField(u1, v1, w2)
    @test_throws MethodError VectorField("string1", "string2")
end

@testset "Vector Field Broadcasting     " begin
    # initialise random variables
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    ω = abs(randn())
    β = abs(randn())

    # initialise grid
    grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny), ω, β)

    # initialise vector fields
    a = VectorField(PhysicalField(grid), PhysicalField(grid), PhysicalField(grid))
    b = VectorField(PhysicalField(grid), PhysicalField(grid), PhysicalField(grid))
    c = VectorField(PhysicalField(grid), PhysicalField(grid), PhysicalField(grid))

    # check broadcasting is done efficiently and does not allocate
    nalloc(a, b, c) = @allocated a .= 3 .* b .+ c ./ 2
    @test nalloc(a, b, c) == 0

    # test broadcasting
    @test typeof(a .+ b) == typeof(a)
end
