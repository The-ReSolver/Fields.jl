function test_grid()
    @testset "Field Grid                            " begin
        # generate random inputs
        Ny = rand(3:50)
        Nz = rand(3:50)
        Nt = rand(3:50)
        randD = rand([-2, -1, 1, 2])
        y = rand(Float64, Ny)
        D1 = rand(Float32, (Ny, Ny))
        D2 = rand(Float16, (Ny + randD, Ny + randD))
        D_sec = rand(Float32, (Ny, Ny))
        w1 = rand(Int128, Ny)
        w2 = rand(Float16, Ny + randD)
        ω = abs(randn())
        β = π

        # test constructor
        g1 = @test_nowarn Grid(y, Nz, Nt, D1, D_sec, w1, ω, β)
        @test_throws MethodError Grid(y, Nz, Nt, rand(Float64, (Ny, Ny)), D_sec, ω, β)

        # test point generation
        gpoints = points(g1)
        @test gpoints[1] == y
        @test gpoints[2] ≈ range(0, 2π*(1 - 1/Nz), length = Nz)/β # precision differences in operations
        @test gpoints[3] ≈ range(0, 2π*(1 - 1/Nt), length = Nt)/ω # mean they aren't exactly equal

        # test size of grid
        @test size(g1) == (Ny, Nz, Nt)

        # test field extraction interface
        @test get_Dy(g1) == convert(Matrix{eltype(y)}, D1)
        @test get_Dy2(g1) == convert(Matrix{eltype(y)}, D_sec)
        @test get_ws(g1) == convert(Vector{eltype(y)}, w1)
        @test get_ω(g1) == eltype(y)(ω)
        @test get_β(g1) == eltype(y)(β)

        # test comparison
        @test g1 != Grid(y, Nz, Nt + 1, D1, D_sec, w1, ω, β)
        @test g1 != Grid(rand(Float64, Ny), Nz, Nt, D1, D_sec, w1, ω, β)
    end
end
