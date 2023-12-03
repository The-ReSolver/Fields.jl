@testset "vec2field                     " begin
    # initialise random spectral and vector fields
    # Ny = 64; Nz = 64; Nt = 64
    Ny = 2; Nz = 2; Nt = 2
    g = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny), abs(rand()), abs(rand()))
    u = SpectralField(g); u .= rand(ComplexF64, Ny, (Nz >> 1) + 1, Nt)
    v = VectorField(g); [v[i] .= rand(ComplexF64, Ny, (Nz >> 1) + 1, Nt) for i in 1:3]

    @test u == Fields.vec2field!(SpectralField(g), Fields.field2vec!(zeros(524288), u))
    @test v == Fields.vec2field!(VectorField(g), Fields.field2vec!(zeros(3*524288), v))
end