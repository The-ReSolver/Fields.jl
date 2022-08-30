@testset "Field Derivatives             " begin
    # define functions
    u_fun(y, z, t) = (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    dudy_fun(y, z, t) = -2*y*exp(cos(5.8*z))*atan(sin(t))
    d2udy2_fun(y, z, t) = -2*exp(cos(5.8*z))*atan(sin(t))
    dudz_fun(y, z, t) = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
    d2udz2_fun(y, z, t) = (5.8^2)*(1 - y^2)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(t))
    dudt_fun(y, z, t) = ((1 - y^2)*exp(cos(5.8*z))*cos(t))/(sin(t)^2 + 1)

    # initialise original function field
    Ny = 50; Nz = 50; Nt = 50
    y = chebpts(Ny)
    Dy = chebdiff(Ny); Dy2 = chebddiff(Ny)
    # Dy = DiffMatrix(y, 5, 1); Dy2 = DiffMatrix(y, 5, 2)
    grid = Grid(y, Nz, Nt, Dy, Dy2, rand(Float64, Ny), 1.0, 5.8)
    u = physicalfield(grid, u_fun)
    û = spectralfield(grid)
    FFT = FFTPlan!(u, flags=ESTIMATE)
    IFFT = IFFTPlan!(û, flags=ESTIMATE)
    FFT(û, u)

    # initialise fields to hold derivatives
    dûdy = spectralfield(grid)
    dudy = physicalfield(grid)
    d2ûdy2 = spectralfield(grid)
    d2udy2 = physicalfield(grid)
    dûdz = spectralfield(grid)
    dudz = physicalfield(grid)
    d2ûdz2 = spectralfield(grid)
    d2udz2 = physicalfield(grid)
    dûdt = spectralfield(grid)
    dudt = physicalfield(grid)

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
    @test dudy ≈ physicalfield(grid, dudy_fun)
    @test d2udy2 ≈ physicalfield(grid, d2udy2_fun)
    @test dudz ≈ physicalfield(grid, dudz_fun)
    @test d2udz2 ≈ physicalfield(grid, d2udz2_fun)
    @test dudt ≈ physicalfield(grid, dudt_fun)
end
