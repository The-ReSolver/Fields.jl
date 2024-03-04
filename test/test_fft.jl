@testset "FFT Transforms Reversible             " begin
    # randon signal
    Ny = rand(3:50); Nz = rand(3:50); Nt = rand(3:50)
    ω = abs(randn())
    β = abs(randn())
    grid = Grid(rand(Float64, Ny), Nz, Nt, rand(Float64, (Ny, Ny)), rand(Float64, (Ny, Ny)), rand(Float64, Ny), ω, β)
    A1 = PhysicalField(grid); Â1 = SpectralField(grid); B1 = PhysicalField(grid)
    A2 = PhysicalField(grid); Â2 = SpectralField(grid); B2 = PhysicalField(grid)
    A1.data .= rand(Float64, Ny, Nz, Nt)
    A2.data .= rand(Float64, Ny, Nz, Nt)

    # initialise vector field
    𝐀 = VectorField(A1, A2)
    𝐀̂ = VectorField(Â1, Â2)
    𝐁 = VectorField(B1, B2)

    # create plans
    FFT = FFTPlan!(grid, false; flags=ESTIMATE)
    IFFT = IFFTPlan!(grid, false; flags=ESTIMATE)

    # is the transform invertible correctly
    FFT(Â1, A1); IFFT(B1, Â1)
    FFT(𝐀̂, 𝐀); IFFT(𝐁, 𝐀̂)
    @test A1 ≈ B1
    @test 𝐀 ≈ 𝐁
end

@testset "FFT Dealiasing                        " begin
    # initialise fields
    Ny = 16; Nz = 32; Nt = 32
    ω = abs(randn())
    β = abs(randn())
    grid = Grid(rand(Float64, Ny), Nz, Nt, rand(Float64, (Ny, Ny)), rand(Float64, (Ny, Ny)), rand(Float64, Ny), ω, β)
    A1 = PhysicalField(grid, (y, z, t)->cos(β*5*z)+sin(β*2*z)*sin(ω*2*t), true)
    A2 = PhysicalField(grid, (y, z, t)->cos(β*3*z)+sin(β*8*z)*sin(ω*5*t), true)
    Â1 = SpectralField(grid)
    Â2 = SpectralField(grid)
    B1 = PhysicalField(grid, true)
    B2 = PhysicalField(grid, true)
    𝐀 = VectorField(A1, A2)
    𝐀̂ = VectorField(Â1, Â2)
    𝐁 = VectorField(B1, B2)

    FFT = FFTPlan!(grid, true; flags=ESTIMATE)
    IFFT = IFFTPlan!(grid, true; flags=ESTIMATE)

    @test size(FFT.padded) == (16, 25, 48)
    @test size(IFFT.padded) == (16, 25, 48)

    FFT(𝐀̂, 𝐀); IFFT(𝐁, 𝐀̂)
    @test 𝐀 ≈ 𝐁
end
