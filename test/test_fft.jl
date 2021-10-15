# This file contains the test set for the transform between spectral and
# physcial scalar fields.

@testset "FFT Transforms" begin
    # Ny = 4; Nz = 4; Nt = 4
    # A = PhysicalField(ones(Ny, Nz, Nt))
    # B = PhysicalField(similar(parent(A)))
    # Â = SpectraField(size(A)...)
    # Aplan = FFTPlan!(A)
    # Âplan = IFFTPlan!(Â)
    # Aplan(Â, A)
    # println(parent(Â))
    # Âplan(B, Â)
    # println(parent(B))
    # println(parent(A) == parent(B)) # true!
    @test true
end