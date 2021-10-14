# This file contains the test set for the transform between spectral and
# physcial scalar fields.

@testset "FFT Transforms" begin
    Ny = 3; Nz = 3; Nt = 3
    A = PhysicalField(ones(Ny, Nz, Nt))
    simple = Fields.simple_fft(A)
    println(A)
    Â = SpectraField(ones(Ny, Nz, Nt)+ones(Ny, Nz, Nt)*2im)
    Aplan = FFTPlan!(A)
    println(A)
    Aplan(Â, A)
    println(simple)
    @test true
end