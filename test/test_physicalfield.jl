@testset "Physical Field Constructor    " begin
        # take random variables
        Ny = rand(3:50); Nz = rand(3:50); Nt = rand(3:50)
        y = rand(Float64, Ny)
        Dy = rand(Float64, (Ny, Ny))
        Dy2 = rand(Float64, (Ny, Ny))
        ws = rand(Float64, Ny)
        ω = abs(randn())
        β = abs(randn())
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

        # intialise using different constructors
        @test isa(PhysicalField(grid), PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}})

        # construct from function
        fun(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))
        f = PhysicalField(grid, fun)
        y, z, t = points(grid)
        out = zeros(Ny, Nz, Nt)
        for nt = 1:Nt, nz in 1:Nz, ny in 1:Ny
                out[ny, nz, nt] = fun(y[ny], z[nz], t[nt])
        end
        @test f == out
end

@testset "Physical Field Broadcasting   " begin
        # take random variables
        Ny = rand(3:50); Nz = rand(3:50); Nt = rand(3:50)
        ω = abs(randn())
        β = abs(randn())

        # initialise grid
        grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny), ω, β)

        # test in place broadcasting
        a = PhysicalField(grid)
        b = PhysicalField(grid)
        c = PhysicalField(grid)

        # check broadcasting is done efficiently and does not allocate
        nalloc(a, b, c) = @allocated a .= 3 .* b .+ c ./ 2
        @test nalloc(a, b, c) == 0

        # test broadcasting
        @test typeof(a .+ b) == typeof(a)
end
