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
    A = PhysicalField(grid); Â = SpectraField(grid); B = PhysicalField(grid)
    A.data .= rand(Float64, (Ny, Nz, Nt))

    # create plans
    FFTplan = FFTPlan!(A, flags = FFTW.ESTIMATE)
    IFFTplan = IFFTPlan!(Â, flags = FFTW.ESTIMATE)

    # is the transform invertible correctly
    FFTplan(Â, A)
    IFFTplan(B, Â)
    @test A ≈ B
end
