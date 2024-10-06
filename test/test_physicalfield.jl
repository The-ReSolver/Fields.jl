@testset "Physical Field Constructor            " begin
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
        @test PhysicalField(grid) isa PhysicalField{typeof(grid), false, 1.5}
        @test PhysicalField(grid, true) isa PhysicalField{typeof(grid), true, 1.5}

        # construct from function
        fun(y, z, t) = (1 - y^2)*exp(cos(β*z))*atan(sin(ω*t))
        f = PhysicalField(grid, fun)
        y, z, t = points(grid)
        out = zeros(Ny, Nz, Nt)
        for nt = 1:Nt, nz in 1:Nz, ny in 1:Ny
                out[ny, nz, nt] = fun(y[ny], z[nz], t[nt])
        end
        @test f == out

        # construct dealiased from function
        g = PhysicalField(grid, fun, true)
        out = zeros(Ny, Fields.padded_size(Nz, Nt)...)
        z_padded = (0:(size(out, 2) - 1))*(2π/(size(out, 2)*β))
        t_padded = (0:(size(out, 3) - 1))*(2π/(size(out, 3)*ω))
        for nt in eachindex(t_padded), nz in eachindex(z_padded), ny in 1:Ny
                out[ny, nz, nt] = fun(y[ny], z_padded[nz], t_padded[nt])
        end
        @test g == out
end

@testset "Physical Field Broadcasting           " begin
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

@testset "Kinetic Energy                        " begin
        # take random variables
        Ny = 32; Nz = 32; Nt = 16
        y = chebpts(Ny)
        Dy = rand(Float64, (Ny, Ny))
        Dy2 = rand(Float64, (Ny, Ny))
        ws = chebws(Ny)
        ω = abs(randn())
        β = abs(randn())
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

        _, _, t = points(grid)

        # construct from function
        fun(y, z, t) = (1 - y^2)*exp(cos(β*z))*atan(sin(ω*t))
        f = PhysicalField(grid, fun)
        g = VectorField(grid, fun, fun, fun)

        @test Fields.energy(f) ≈ repeat([2.4315*π/β], Nt).*(atan.(sin.(ω.*t)).^2) rtol=1e-4
        @test Fields.energy(g) ≈ repeat([2.4315*3π/β], Nt).*(atan.(sin.(ω.*t)).^2) rtol=1e-4
end
