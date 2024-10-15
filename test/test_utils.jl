@testset "Utility functions                     " begin
    # define velocity functions
    u_fun(y, z, t)  = y + (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
    v_fun(y, z, t)  = sin(π*y)*exp(sin(2.0*5.8*z))*cos(t)
    w_fun(y, z, t)  = (1 - y^2)*cos(5.8*z)*exp(sin(t))
    dudy_fun(y, z, t)   = 1 - 2*y*exp(cos(5.8*z))*atan(sin(t))
    dudz_fun(y, z, t)   = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
    dvdz_fun(y, z, t)   = 2.0*5.8*sin(π*y)*cos(2.0*5.8*z)*exp(sin(2.0*5.8*z))*cos(t)
    dwdy_fun(y, z, t)   = -2*y*cos(5.8*z)*exp(sin(t))
    ωx_fun(y, z, t) = dwdy_fun(y, z, t) - dvdz_fun(y, z, t)
    ωy_fun(y, z, t) = dudz_fun(y, z, t)
    ωz_fun(y, z, t) = -dudy_fun(y, z, t)

    # define grid
    Ny = 50; Nz = 51; Nt = 51
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = chebws(Ny)
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, 1.0, 5.8)
    FFT = FFTPlan!(grid, flags=ESTIMATE)
    u = FFT(VectorField(grid), VectorField(grid, u_fun, v_fun, w_fun))
    ω = FFT(VectorField(grid), VectorField(grid, ωx_fun, ωy_fun, ωz_fun))

    # perform tests
    @test energy(FFT(SpectralField(grid), PhysicalField(grid, (y,z,t)->u_fun(y,z,t)-y))) ≈ [1.317066667*(atan(sin(t))^2) for t in points(grid)[3]] rtol=1e-5
    @test vorticity!(VectorField(grid), u) ≈ ω
    @test boundaryEnergy(u[1]) ≈ (4π/5.8).*ones(Nt) rtol=1e-10
end
