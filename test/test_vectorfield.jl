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

    # initialise dummy functions
    fun1(y, z, t) = 1.0
    fun2(y, z, t) = 2.0
    fun3(y, z, t) = 3.0

    # initialise
    @test VectorField(u1, v1) isa VectorField{2, Array{Float64, 3}}
    @test VectorField(u1, v1, w1) isa VectorField{3, Array{Float64, 3}}
    @test VectorField(grid) isa VectorField{3, SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{Complex{Float64}, 3}}}
    @test VectorField(grid; N=2) isa VectorField{2, SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{Complex{Float64}, 3}}}
    @test VectorField(grid, N=5, field_type=:physical) isa VectorField{5, PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}}
    @test VectorField(grid, fun1) isa VectorField{1, PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}}
    @test VectorField(grid, fun1, fun2, fun3) isa VectorField{3, PhysicalField{Ny, Nz, Nt, typeof(grid), Float64, Array{Float64, 3}}}

    # test copy method
    a = VectorField(grid)
    @test copy(a) == a

    # # catch errors on constructors
    @test_throws MethodError VectorField(u2, v1)
    @test_throws MethodError VectorField(u1, v2, w1)
    @test_throws MethodError VectorField(u1, v1, w2)
    @test_throws MethodError VectorField("string1", "string2")
end

@testset "Vector Field grid methods     " begin
    # initialise random variables
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    ω = abs(randn())
    β = abs(randn())

    # initialise fields
    grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny), ω, β)
    a = SpectralField(grid)
    b = PhysicalField(grid)
    c = VectorField(grid)

    # test grid comparison method
    for i in [a, b, c]
        for j in [a, b, c]
            eval(:(@test grideq($i, $j)))
        end
    end
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

@testset "Vector Field Norm             " begin
    # initialise grid variables
    Ny = 64; Nz = 64; Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = chebws(Dy)
    ω = 1.0
    β = 1.0

    # initialise grid
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

    # definition of fields as functions
    u_func(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))
    v_func(y, z, t) = (cos(π*y) + 1)*exp(sin(z))*cos(sin(t))
    w_func(y, z, t) = sin(atan(y))*(cos(z)/(sin(z)^2 + 1))*exp(cos(t))

    # initialise fields
    vec_p = VectorField(PhysicalField(grid, u_func),
                        PhysicalField(grid, v_func),
                        PhysicalField(grid, w_func))
    vec_s = VectorField(grid)
    FFT! = FFTPlan!(grid; flags=ESTIMATE)
    FFT!(vec_s, vec_p)

    # test norm
    @test norm(vec_s)^2 ≈ 33.04894874 + 165.2156694 + 13.65625483 rtol=1e-5
end
