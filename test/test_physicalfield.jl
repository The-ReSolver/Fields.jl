@testset "Physical Scalar Field         " begin
        # take random variables
        Ny = rand(3:50)
        Nz = rand(3:50)
        Nt = rand(3:50)
        y = rand(Float64, Ny)
        Dy = rand(Float64, (Ny, Ny))
        Dy2 = rand(Float64, (Ny, Ny))
        ws = rand(Float64, Ny)
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws)

        # intialise using different constructors
        @test   typeof(PhysicalField(grid)) ==
                PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}

        # construct from function
        fun(y, z, t) = (1 - y^2)
        y = range(-1, stop=1, length=11)
        f = PhysicalField(Grid(collect(y), 1, 1, rand(11, 11), rand(11, 11), rand(11)), fun)
        @test vec(parent(f)) == fun.(y, 0, 0)

        # test in place broadcasting
        grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny))

        a = PhysicalField(grid)
        b = PhysicalField(grid)
        c = PhysicalField(grid)

        # check broadcasting is done efficiently and does not allocate
        nalloc(a, b, c) = @allocated a .= 3 .* b .+ c ./ 2

        @test nalloc(a, b, c) == 0

        # test broadcasting
        @test typeof(a .+ b) == typeof(a)
end
