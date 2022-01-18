@testset "Spectral Field Constructor    " begin
        # take random variables
        Ny = rand(3:50); Nz = rand(3:50); Nt = rand(3:50)
        y = rand(Float64, Ny)
        Dy = rand(Float64, (Ny, Ny))
        Dy2 = rand(Float64, (Ny, Ny))
        ws = rand(Float64, Ny)
        ω = abs(randn())
        β = abs(randn())

        # initialise gird
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

        # intialise using different constructors
        @test typeof(SpectralField(grid)) == SpectralField{Ny, Nz, Nt, typeof(grid), Float64, Array{Complex{Float64}, 3}}

        # test size method
        @test size(SpectralField(grid)) == size(parent(SpectralField(grid)))
end

@testset "Spectral Field Broadcasting   " begin
        # take random variables
        Ny = rand(3:50); Nz = rand(3:50); Nt = rand(3:50)
        y = rand(Float64, Ny)
        Dy = rand(Float64, (Ny, Ny))
        Dy2 = rand(Float64, (Ny, Ny))
        ws = rand(Float64, Ny)
        ω = abs(randn())
        β = abs(randn())

        # initialise grid
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

        # initialise spectral fields
        a = SpectralField(grid)
        b = SpectralField(grid)
        c = SpectralField(grid)
        d = SpectralField(grid)
        e = SpectralField(grid)

        # check broadcasting is done efficiently and does not allocate
        nalloc(a, b, c) = @allocated a .= 3 .* b .+ c ./ 2

        @test nalloc(a, b, c) == 0

        # test broadcasting
        @test typeof(a .+ b) == typeof(a)

        # test broadcasting with vectors
        vec = rand(Float64, Ny)
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), ny in 1:Ny
                d[ny, nz, nt] = a[ny, nz, nt] + vec[ny]
                e[ny, nz, nt] = a[ny, nz, nt]*vec[ny]
        end
        @test d == a .+ vec == vec .+ a
        @test e == vec.*a == a.*vec
end

@testset "Spectral Field Norm           " begin
        # initialise grid variables
        Ny = 64; Nz = 64; Nt = 64
        y = chebpts(Ny)
        Dy = chebdiff(Ny)
        Dy2 = rand(Float64, (Ny, Ny))
        ws = chebws(Dy)
        ω = 1.0
        β = 1.0

        # initialise grid
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

        # initialise function
        func(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))

        # initialise fields
        phys_norm = PhysicalField(grid, func)
        spec_norm = SpectralField(grid)
        FFT = FFTPlan!(phys_norm; flags=FFTW.ESTIMATE)
        FFT(spec_norm, phys_norm)

        # test norm
        @test norm(spec_norm) ≈ sqrt(0.4185698256) rtol=1e-5
end
