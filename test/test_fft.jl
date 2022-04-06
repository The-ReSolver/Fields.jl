@testset "FFT Transforms                " begin
    # randon signal
    Ny = rand(3:50); Nz = rand(3:50); Nt = rand(3:50)
    ω = abs(randn())
    β = abs(randn())
    grid = Grid(rand(Float64, Ny), Nz, Nt,
                rand(Float64, (Ny, Ny)),
                rand(Float64, (Ny, Ny)),
                rand(Float64, Ny),
                ω, β)
    A1 = PhysicalField(grid); Â1 = SpectralField(grid); B1 = PhysicalField(grid)
    A2 = PhysicalField(grid); Â2 = SpectralField(grid); B2 = PhysicalField(grid)
    A1.data .= rand(Float64, (Ny, Nz, Nt))
    A2.data .= rand(Float64, (Ny, Nz, Nt))

    # initialise vector field
    𝐀 = VectorField(A1, A2)
    𝐀̂ = VectorField(Â1, Â2)
    𝐁 = VectorField(B1, B2)

    # create plans
    FFT = FFTPlan!(grid; flags=FFTW.ESTIMATE)
    IFFT = IFFTPlan!(grid; flags=FFTW.ESTIMATE)

    # is the transform invertible correctly
    FFT(Â1, A1)
    IFFT(B1, Â1, copy(Â1))
    FFT(𝐀̂, 𝐀)
    IFFT(𝐁, 𝐀̂, VectorField(grid; N = 2))
    @test A1 ≈ B1
    @test 𝐀 ≈ 𝐁
end
