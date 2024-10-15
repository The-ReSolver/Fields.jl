@testset "Field Derivatives                     " begin
    # define functions
    u_fun(y, z, t)      = (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    dudy_fun(y, z, t)   = -2*y*exp(cos(5.8*z))*atan(sin(t))
    d2udy2_fun(y, z, t) = -2*exp(cos(5.8*z))*atan(sin(t))
    dudz_fun(y, z, t)   = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
    d2udz2_fun(y, z, t) = (5.8^2)*(1 - y^2)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(t))
    dudt_fun(y, z, t)   = ((1 - y^2)*exp(cos(5.8*z))*cos(t))/(sin(t)^2 + 1)

    v_fun(y, z, t)      = sin(π*y)*exp(sin(2.0*5.8*z))*cos(t)
    dvdy_fun(y, z, t)   = π*cos(π*y)*exp(sin(2.0*5.8*z))*cos(t)
    dvdz_fun(y, z, t)   = 2.0*5.8*sin(π*y)*cos(2.0*5.8*z)*exp(sin(2.0*5.8*z))*cos(t)

    w_fun(y, z, t)      = (1 - y^2)*cos(5.8*z)*exp(sin(t))
    dwdy_fun(y, z, t)   = -2*y*cos(5.8*z)*exp(sin(t))

    ωx_fun(y, z, t)     = dwdy_fun(y, z, t) - dvdz_fun(y, z, t)
    ωy_fun(y, z, t)     = dudz_fun(y, z, t)
    ωz_fun(y, z, t)     = -dudy_fun(y, z, t)

    # initialise original function field
    Ny = 50; Nz = 51; Nt = 51
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = chebws(Ny)
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, 1.0, 5.8)
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

    # test vectorfield methods
    @test ddy!(FFT(VectorField(grid, N=2), VectorField(grid, u_fun, v_fun)), VectorField(grid, N=2)) ≈ FFT(VectorField(grid, N=2), VectorField(grid, dudy_fun, dvdy_fun))

    # test voriticy function
    @test vorticity!(VectorField(grid), FFT(VectorField(grid), VectorField(grid, u_fun, v_fun, w_fun))) ≈ FFT(VectorField(grid), VectorField(grid, ωx_fun, ωy_fun, ωz_fun))
end
