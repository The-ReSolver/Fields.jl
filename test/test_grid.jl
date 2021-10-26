@testset "Field Grid                    " begin
    # generate random inputs
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    randD = rand([-2, -1, 1, 2])
    y = rand(Float64, Ny)
    D1 = rand(Float64, (Ny, Ny))
    D2 = rand(Float64, (Ny + randD, Ny + randD))
    D_sec = rand(Float64, (Ny, Ny))
    w1 = rand(Float64, Ny)
    w2 = rand(Float64, Ny + randD)

    # initialise correct ones #FIXME: this test can be removed.. do not test types explicitly
    # @test typeof(Grid(y, Nz, Nt, D1, D_sec, w1)) == Grid{(Ny, Nz, Nt), Float64, Matrix{Float64}}

    # initialise bad versions to catch errors
    @test_throws ArgumentError Grid(y, Nz, Nt, D2, D_sec, w1)
    @test_throws ArgumentError Grid(y, Nz, Nt, D1, D_sec, w2)
    @test_throws ArgumentError Grid(y, Nz, Nt, D2, D_sec, w2)
    @test_throws MethodError Grid(y, Nz, Nt, D1, D_sec, rand(Int, Ny))
    @test_throws MethodError Grid(y, Nz, Nt, rand(Int, (Ny, Ny)), D_sec, w1)

    # test point generation
    g = Grid(y, Nz, Nt, D1, D_sec, w1)
    gpoints = points(g)
    @test gpoints[1] == y
    @test gpoints[2] ≈ range(0, 2π*(1 - 1/Nz), length = Nz) # the precision differences in these two operations
    @test gpoints[3] ≈ range(0, 2π*(1 - 1/Nt), length = Nt) # mean they aren't exactly equal
end
