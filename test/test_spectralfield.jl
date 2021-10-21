# This file contains the test set for the spectral scalar field type
# definition.

@testset "Spectral Scalar Field         " begin
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
    @test   typeof(SpectralField(grid)) ==
            SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{Complex{Float64}, 3}}
    @test   typeof(SpectralField(ones(Complex{Float64}, Ny, (Nz >> 1) + 1, Nt), grid)) ==
            SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{Complex{Float64}, 3}}

    # test in place broadcasting
    grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny))

    a = SpectralField(grid)
    b = SpectralField(grid)
    c = SpectralField(grid)

    # check broadcasting is done efficiently and does not allocate
    nalloc(a, b, c) = @allocated a .= 3 .* b .+ c ./ 2

    @test nalloc(a, b, c) == 0

    # test broadcasting
    @test typeof(a .+ b) == typeof(a)
end
