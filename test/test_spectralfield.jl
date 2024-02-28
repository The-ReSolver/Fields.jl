@testset "Spectral Field Constructor            " begin
        # take random variables
        Ny = rand(3:50); Nz = rand(3:50); Nt = rand(3:50); M = rand(1:Ny)
        y = rand(Float64, Ny)
        Dy = rand(Float64, (Ny, Ny))
        Dy2 = rand(Float64, (Ny, Ny))
        ws = rand(Float64, Ny)
        ω = abs(randn())
        β = abs(randn())

        # initialise gird
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

        # construct spectral fields
        u1 = SpectralField(grid)
        u2 = SpectralField(grid, rand(ComplexF64, 3*Ny, M, (Nz >> 1) + 1, Nt))

        # intialise using different constructors
        @test u1 isa SpectralField{Ny, Nz, Nt, typeof(grid), Float64, false, Array{Complex{Float64}, 3}}
        @test u2 isa SpectralField{M, Nz, Nt, typeof(grid), Float64, true, Array{Complex{Float64}, 3}}

        # test size method
        @test size(u1) == (Ny, (Nz >> 1) + 1, Nt)
        @test size(u2) == (M, (Nz >> 1) + 1, Nt)
end

@testset "Spectral Field Broadcasting           " begin
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

@testset "Spectral Field Dot and Norm           " begin
        # initialise grid variables
        Ny = 64; Nz = 64; Nt = 64
        y = chebpts(Ny)
        Dy = chebdiff(Ny)
        Dy2 = chebddiff(Ny)
        ws = chebws(Ny)
        ω = abs(rand())
        β = abs(rand())

        # initialise grid
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

        # initialise function
        func1(y, z, t) = (1 - y^2)*exp(cos(β*z))*cos(sin(ω*t))
        func2(y, z, t) = cos(π*y)*(1 - y^2)*exp(sin(β*z))*(cos(ω*t)^2)

        # initialise fields
        phys1 = PhysicalField(grid, func1)
        phys2 = PhysicalField(grid, func2)
        spec1 = SpectralField(grid)
        spec2 = SpectralField(grid)
        FFT! = FFTPlan!(grid; flags=ESTIMATE)
        FFT!(spec1, phys1)
        FFT!(spec2, phys2)

        # test norm
        @test dot(spec1, spec2)*β*ω ≈ 13.4066 rtol=1e-6
        @test norm(spec1)^2*β*ω ≈ 58.74334913 rtol=1e-5
end

@testset "Projected Field Norm                  " begin
        # initialise grid
        Ny = 16; Nz = 16; Nt = 16
        y = chebpts(Ny)
        Dy = rand(Ny, Ny)
        Dy2 = rand(Ny, Ny)
        ws = ones(Ny)
        ω = abs(rand())
        β = abs(rand())
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)

        # construct modes
        M = rand(1:12)
        Ψ = zeros(ComplexF64, 3*Ny, M, Nz, Nt)
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
                Ψ[:, :, nz, nt] .= @view(qr(rand(ComplexF64, 3*Ny, M)).Q[:, 1:M])
        end

        # construct vector field from modes
        u = VectorField(grid)
        a = SpectralField(grid, Ψ)
        a .= rand(ComplexF64, M, (Nz >> 1) + 1, Nt)
        expand!(u, a, Ψ)

        @test norm(u) ≈ norm(a)
end
