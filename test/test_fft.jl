# This file contains the test set for the transform between spectral and
# physcial scalar fields.

@testset "FFT Transforms                " begin
    # randon signal
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    grid = Grid(rand(Float64, Ny), Nz, Nt,
                rand(Float64, (Ny, Ny)),
                rand(Float64, (Ny, Ny)),
                rand(Float64, Ny))
    A1 = PhysicalField(grid); Â1 = SpectralField(grid); B1 = PhysicalField(grid)
    A2 = PhysicalField(grid); Â2 = SpectralField(grid); B2 = PhysicalField(grid)
    A1.data .= rand(Float64, (Ny, Nz, Nt))
    A2.data .= rand(Float64, (Ny, Nz, Nt))

    # initialise vector field
    𝐀 = VectorField(A1, A2)
    𝐀̂ = VectorField(Â1, Â2)
    𝐁 = VectorField(B1, B2)

    # create plans
    FFTplan = FFTPlan!(A1, flags = FFTW.ESTIMATE)
    IFFTplan = IFFTPlan!(Â1, flags = FFTW.ESTIMATE)

    # is the transform invertible correctly
    FFTplan(Â1, A1)
    IFFTplan(B1, Â1)
    FFTplan(𝐀̂, 𝐀)
    IFFTplan(𝐁, 𝐀̂)
    @test A1 ≈ B1
    for i in 1:2
        @test 𝐀[i] ≈ 𝐁[i]
    end
end
