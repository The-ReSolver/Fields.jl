@testset "Spectral Scalar Field         " begin
        # take random variables
        Ny = 64
        Nz = 64
        Nt = 64
        y = chebpts(Ny)
        Dy = rand(Float64, (Ny, Ny))
        Dy2 = rand(Float64, (Ny, Ny))
        ws = quadweights(y, 2)
        ω = 1.0
        β = 1.0
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

        # intialise using different constructors
        @test typeof(SpectralField(grid)) == SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{Complex{Float64}, 3}}

        @test size(SpectralField(grid)) == size(parent(SpectralField(grid)))

        a = SpectralField(grid)
        b = SpectralField(grid)
        c = SpectralField(grid)

        # check broadcasting is done efficiently and does not allocate
        nalloc(a, b, c) = @allocated a .= 3 .* b .+ c ./ 2

        @test nalloc(a, b, c) == 0

        # test broadcasting
        @test typeof(a .+ b) == typeof(a)

        # test norm
        func(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))
        phys_norm = PhysicalField(grid, func)
        spec_norm = SpectralField(grid)
        FFT = FFTPlan!(phys_norm; flags=FFTW.ESTIMATE)
        FFT(spec_norm, phys_norm)
        @test LinearAlgebra.norm(spec_norm) ≈ sqrt(0.41856) rtol=1e-5
end
