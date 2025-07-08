function test_vectorToField()
    @testset "Spectral Field to State Vector        " begin
        # initialise spectral field
        # Ny = rand(3:32); Nz = rand(3:2:33); Nt = rand(1:2:33); M = rand(1:3)
        Ny = 5; Nz = 3; Nt = 3; M = 1
        y = collect(range(-1, 1, length=Ny))
        ω0 = abs(rand())
        g = Grid(y, Nz, Nt, DiffMatrix(y, 3, 1), DiffMatrix(y, 3, 2), quadweights(y, 2), ω0, 1.0)
        modes = rand(ComplexF64, 3*Ny, M, (Nz >> 1) + 1, Nt)
        a = SpectralField(g, modes)
        a .= rand(ComplexF64, M, (Nz >> 1) + 1, Nt)
        a[:, 1, 1] .= real.(a[:, 1, 1])
        Fields.apply_symmetry!(a)
        vector = Vector{Float64}(undef, 2*Ny*((Nz >> 1) + 1)*Nt + 1)

        @test vectorToField!(similar(a), fieldToVector!(vector, a, ω0)) == a
    end
end