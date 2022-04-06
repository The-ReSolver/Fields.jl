@testset "Field Derivatives             " begin
    # define functions
    u_fun(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))
    dudy_fun(y, z, t) = -2*y*exp(cos(z))*atan(sin(t))
    d2udy2_fun(y, z, t) = -2*exp(cos(z))*atan(sin(t))
    dudz_fun(y, z, t) = -(1 - y^2)*sin(z)*exp(cos(z))*atan(sin(t))
    d2udz2_fun(y, z, t) = (1 - y^2)*(sin(z)^2 - cos(z))*exp(cos(z))*atan(sin(t))
    dudt_fun(y, z, t) = ((1 - y^2)*exp(cos(z))*cos(t))/(sin(t)^2 + 1)

    # initialise original function field
    Ny = 50; Nz = 50; Nt = 50
    Dy = chebdiff(Ny); Dy2 = chebddiff(Ny)
    y = chebpts(Ny)
    grid = Grid(y, Nz, Nt, Dy, Dy2, rand(Float64, Ny), 1.0, 1.0)
    u = PhysicalField(grid, u_fun)
    û = SpectralField(grid)
    FFT = FFTPlan!(u, flags=ESTIMATE)
    IFFT = IFFTPlan!(û, flags=ESTIMATE)
    FFT(û, u)

    # initialise fields to hold derivatives
    dûdy = SpectralField(grid)
    dudy = PhysicalField(grid)
    d2ûdy2 = SpectralField(grid)
    d2udy2 = PhysicalField(grid)
    dûdz = SpectralField(grid)
    dudz = PhysicalField(grid)
    d2ûdz2 = SpectralField(grid)
    d2udz2 = PhysicalField(grid)
    dûdt = SpectralField(grid)
    dudt = PhysicalField(grid)

    # compute derivatives
    ddy!(û, dûdy)
    d2dy2!(û, d2ûdy2)
    ddz!(û, dûdz)
    d2dz2!(û, d2ûdz2)
    ddt!(û, dûdt)

    # convert back to physical domain
    IFFT(dudy, dûdy)
    IFFT(d2udy2, d2ûdy2)
    IFFT(dudz, dûdz)
    IFFT(d2udz2, d2ûdz2)
    IFFT(dudt, dûdt)

    # correct values
    @test dudy ≈ PhysicalField(grid, dudy_fun)
    @test d2udy2 ≈ PhysicalField(grid, d2udy2_fun)
    @test dudz ≈ PhysicalField(grid, dudz_fun)
    @test d2udz2 ≈ PhysicalField(grid, d2udz2_fun)
    @test dudt ≈ PhysicalField(grid, dudt_fun)
end
