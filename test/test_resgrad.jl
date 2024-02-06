# initialise field grid
Ny = 32; Nz = 32; Nt = 32; M = 5
Re = 100.0; Ro = 0.5
y = chebpts(Ny)
Dy = chebdiff(Ny)
Dy2 = chebddiff(Ny)
ω = 1.0
β = 1.0
grid = Grid(y, Nz, Nt, Dy, Dy2, rand(Ny), ω, β)
FFT! = FFTPlan!(grid, flags=ESTIMATE)

# initialise residual gradient cache
cache = ResGrad(grid, Array{ComplexF64}(undef, Ny, M, Nz, Nt), rand(Float64, Ny), Re, Ro)

# initialise functions
u_fun(y, z, t)       = (y^3)*exp(cos(z))*sin(t)
v_fun(y, z, t)       = (y^2 - 1)*sin(z)*sin(t)
w_fun(y, z, t)       = cos(2π*y)*exp(sin(z))*sin(t)
rx_fun(y, z, t)      = (y^3)*exp(sin(z))*sin(t)
ry_fun(y, z, t)      = (y^2)*cos(z)*sin(t)
rz_fun(y, z, t)      = exp(cos(y))*sin(z)*sin(t)
dudt_fun(y, z, t)    = (y^3)*exp(cos(z))*cos(t)
dvdt_fun(y, z, t)    = (y^2 - 1)*sin(z)*cos(t)
dwdt_fun(y, z, t)    = cos(2π*y)*exp(sin(z))*cos(t)
dudy_fun(y, z, t)    = (3*y^2)*exp(cos(z))*sin(t)
dvdy_fun(y, z, t)    = 2*y*sin(z)*sin(t)
dwdy_fun(y, z, t)    = -2π*sin(2π*y)*exp(sin(z))*sin(t)
dudz_fun(y, z, t)    = -(y^3)*sin(z)*exp(cos(z))*sin(t)
dvdz_fun(y, z, t)    = (y^2 - 1)*cos(z)*sin(t)
dwdz_fun(y, z, t)    = cos(2π*y)*cos(z)*exp(sin(z))*sin(t)
d2udy2_fun(y, z, t)  = 6*y*exp(cos(z))*sin(t)
d2vdy2_fun(y, z, t)  = 2*sin(z)*sin(t)
d2wdy2_fun(y, z, t)  = -4π^2*cos(2π*y)*exp(sin(z))*sin(t)
d2udz2_fun(y, z, t)  = (y^3)*(sin(z)^2 - cos(z))*exp(cos(z))*sin(t)
d2vdz2_fun(y, z, t)  = (1 - y^2)*sin(z)*sin(t)
d2wdz2_fun(y, z, t)  = cos(2π*y)*(cos(z)^2 - sin(z))*exp(sin(z))*sin(t)
vdudy_fun(y, z, t)   = v_fun(y, z, t)*dudy_fun(y, z, t)
wdudz_fun(y, z, t)   = w_fun(y, z, t)*dudz_fun(y, z, t)
vdvdy_fun(y, z, t)   = v_fun(y, z, t)*dvdy_fun(y, z, t)
wdvdz_fun(y, z, t)   = w_fun(y, z, t)*dvdz_fun(y, z, t)
vdwdy_fun(y, z, t)   = v_fun(y, z, t)*dwdy_fun(y, z, t)
wdwdz_fun(y, z, t)   = w_fun(y, z, t)*dwdz_fun(y, z, t)
drxdt_fun(y, z, t)   = (y^3)*exp(sin(z))*cos(t)
drydt_fun(y, z, t)   = (y^2)*cos(z)*cos(t)
drzdt_fun(y, z, t)   = exp(cos(y))*sin(z)*cos(t)
drxdy_fun(y, z, t)   = (3*y^2)*exp(sin(z))*sin(t)
drydy_fun(y, z, t)   = 2*y*cos(z)*sin(t)
drzdy_fun(y, z, t)   = -sin(y)*exp(cos(y))*sin(z)*sin(t)
drxdz_fun(y, z, t)   = (y^3)*cos(z)*exp(sin(z))*sin(t)
drydz_fun(y, z, t)   = -(y^2)*sin(z)*sin(t)
drzdz_fun(y, z, t)   = exp(cos(y))*cos(z)*sin(t)
d2rxdy2_fun(y, z, t) = 6*y*exp(sin(z))*sin(t)
d2rydy2_fun(y, z, t) = 2*cos(z)*sin(t)
d2rzdy2_fun(y, z, t) = (sin(y)^2 - cos(y))*exp(cos(y))*sin(z)*sin(t)
d2rxdz2_fun(y, z, t) = (y^3)*(cos(z)^2 - sin(z))*exp(sin(z))*sin(t)
d2rydz2_fun(y, z, t) = -(y^2)*cos(z)*sin(t)
d2rzdz2_fun(y, z, t) = -exp(cos(y))*sin(z)*sin(t)
vdrxdy_fun(y, z, t)  = v_fun(y, z, t)*drxdy_fun(y, z, t)
wdrxdz_fun(y, z, t)  = w_fun(y, z, t)*drxdz_fun(y, z, t)
vdrydy_fun(y, z, t)  = v_fun(y, z, t)*drydy_fun(y, z, t)
wdrydz_fun(y, z, t)  = w_fun(y, z, t)*drydz_fun(y, z, t)
vdrzdy_fun(y, z, t)  = v_fun(y, z, t)*drzdy_fun(y, z, t)
wdrzdz_fun(y, z, t)  = w_fun(y, z, t)*drzdz_fun(y, z, t)
rxdudy_fun(y, z, t)  = rx_fun(y, z, t)*dudy_fun(y, z, t)
rydvdy_fun(y, z, t)  = ry_fun(y, z, t)*dvdy_fun(y, z, t)
rzdwdy_fun(y, z, t)  = rz_fun(y, z, t)*dwdy_fun(y, z, t)
rxdudz_fun(y, z, t)  = rx_fun(y, z, t)*dudz_fun(y, z, t)
rydvdz_fun(y, z, t)  = ry_fun(y, z, t)*dvdz_fun(y, z, t)
rzdwdz_fun(y, z, t)  = rz_fun(y, z, t)*dwdz_fun(y, z, t)

@testset "Residual Gradient Velocity            " begin
    # assign velocity field
    cache.spec_cache[1] .= FFT!(SpectralField(grid), PhysicalField(grid, u_fun))
    cache.spec_cache[2] .= FFT!(SpectralField(grid), PhysicalField(grid, v_fun))
    cache.spec_cache[3] .= FFT!(SpectralField(grid), PhysicalField(grid, w_fun))

    # compute the cache
    Fields._update_vel_cache!(cache)

    # test for correctness
    @test cache.spec_cache[4] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dudt_fun))
    @test cache.spec_cache[5] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dvdt_fun))
    @test cache.spec_cache[6] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dwdt_fun))
    @test cache.spec_cache[7] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dudy_fun))
    @test cache.spec_cache[8] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dvdy_fun))
    @test cache.spec_cache[9] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dwdy_fun))
    @test cache.spec_cache[10] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dudz_fun))
    @test cache.spec_cache[11] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dvdz_fun))
    @test cache.spec_cache[12] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dwdz_fun))
    @test cache.spec_cache[13] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2udy2_fun))
    @test cache.spec_cache[14] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2vdy2_fun))
    @test cache.spec_cache[15] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2wdy2_fun))
    @test cache.spec_cache[16] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2udz2_fun))
    @test cache.spec_cache[17] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2vdz2_fun))
    @test cache.spec_cache[18] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2wdz2_fun))
    @test cache.spec_cache[19] ≈ FFT!(SpectralField(grid), PhysicalField(grid, vdudy_fun))
    @test cache.spec_cache[20] ≈ FFT!(SpectralField(grid), PhysicalField(grid, wdudz_fun))
    @test cache.spec_cache[21] ≈ FFT!(SpectralField(grid), PhysicalField(grid, vdvdy_fun))
    @test cache.spec_cache[22] ≈ FFT!(SpectralField(grid), PhysicalField(grid, wdvdz_fun))
    @test cache.spec_cache[23] ≈ FFT!(SpectralField(grid), PhysicalField(grid, vdwdy_fun))
    @test cache.spec_cache[24] ≈ FFT!(SpectralField(grid), PhysicalField(grid, wdwdz_fun))
end

@testset "Residual Gradient Residual            " begin
    # assign velocity and residual fields
    cache.spec_cache[1]  .= FFT!(SpectralField(grid), PhysicalField(grid, u_fun))
    cache.spec_cache[2]  .= FFT!(SpectralField(grid), PhysicalField(grid, v_fun))
    cache.spec_cache[3]  .= FFT!(SpectralField(grid), PhysicalField(grid, w_fun))
    cache.spec_cache[28] .= FFT!(SpectralField(grid), PhysicalField(grid, rx_fun))
    cache.spec_cache[29] .= FFT!(SpectralField(grid), PhysicalField(grid, ry_fun))
    cache.spec_cache[30] .= FFT!(SpectralField(grid), PhysicalField(grid, rz_fun))

    # update the cache
    Fields._update_vel_cache!(cache)
    Fields._update_res_cache!(cache)

    # test for correctness
    @test cache.spec_cache[31] ≈ FFT!(SpectralField(grid), PhysicalField(grid, drxdt_fun))
    @test cache.spec_cache[32] ≈ FFT!(SpectralField(grid), PhysicalField(grid, drydt_fun))
    @test cache.spec_cache[33] ≈ FFT!(SpectralField(grid), PhysicalField(grid, drzdt_fun))
    @test cache.spec_cache[34] ≈ FFT!(SpectralField(grid), PhysicalField(grid, drxdy_fun))
    @test cache.spec_cache[35] ≈ FFT!(SpectralField(grid), PhysicalField(grid, drydy_fun))
    @test cache.spec_cache[36] ≈ FFT!(SpectralField(grid), PhysicalField(grid, drzdy_fun))
    @test cache.spec_cache[37] ≈ FFT!(SpectralField(grid), PhysicalField(grid, drxdz_fun))
    @test cache.spec_cache[38] ≈ FFT!(SpectralField(grid), PhysicalField(grid, drydz_fun))
    @test cache.spec_cache[39] ≈ FFT!(SpectralField(grid), PhysicalField(grid, drzdz_fun))
    @test cache.spec_cache[40] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2rxdy2_fun))
    @test cache.spec_cache[41] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2rydy2_fun))
    @test cache.spec_cache[42] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2rzdy2_fun))
    @test cache.spec_cache[43] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2rxdz2_fun))
    @test cache.spec_cache[44] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2rydz2_fun))
    @test cache.spec_cache[45] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2rzdz2_fun))
    @test cache.spec_cache[46] ≈ FFT!(SpectralField(grid), PhysicalField(grid, vdrxdy_fun))
    @test cache.spec_cache[47] ≈ FFT!(SpectralField(grid), PhysicalField(grid, wdrxdz_fun))
    @test cache.spec_cache[48] ≈ FFT!(SpectralField(grid), PhysicalField(grid, vdrydy_fun))
    @test cache.spec_cache[49] ≈ FFT!(SpectralField(grid), PhysicalField(grid, wdrydz_fun))
    @test cache.spec_cache[50] ≈ FFT!(SpectralField(grid), PhysicalField(grid, vdrzdy_fun))
    @test cache.spec_cache[51] ≈ FFT!(SpectralField(grid), PhysicalField(grid, wdrzdz_fun))
    @test cache.spec_cache[52] ≈ FFT!(SpectralField(grid), PhysicalField(grid, rxdudy_fun))
    @test cache.spec_cache[53] ≈ FFT!(SpectralField(grid), PhysicalField(grid, rydvdy_fun))
    @test cache.spec_cache[54] ≈ FFT!(SpectralField(grid), PhysicalField(grid, rzdwdy_fun))
    @test cache.spec_cache[55] ≈ FFT!(SpectralField(grid), PhysicalField(grid, rxdudz_fun))
    @test cache.spec_cache[56] ≈ FFT!(SpectralField(grid), PhysicalField(grid, rydvdz_fun))
    @test cache.spec_cache[57] ≈ FFT!(SpectralField(grid), PhysicalField(grid, rzdwdz_fun))
end

@testset "Optimal Frequency                     " begin
    # set velocity profile
    cache.spec_cache[1] .= FFT!(SpectralField(grid), PhysicalField(grid, u_fun))
    cache.spec_cache[2] .= FFT!(SpectralField(grid), PhysicalField(grid, v_fun))
    cache.spec_cache[3] .= FFT!(SpectralField(grid), PhysicalField(grid, w_fun))

    # run the cache update
    Fields._update_vel_cache!(cache)

    # compute the optimal frequency manually
    duds = FFT!(VectorField(grid), VectorField(grid, dudt_fun, dvdt_fun, dwdt_fun))./ω
    nsx_fun(y, z, t) = -vdudy_fun(y, z, t) - wdudz_fun(y, z, t) + (1/Re)*(d2udy2_fun(y, z, t) + d2udz2_fun(y, z, t)) + Ro*v_fun(y, z, t)
    nsy_fun(y, z, t) = -vdvdy_fun(y, z, t) - wdvdz_fun(y, z, t) + (1/Re)*(d2vdy2_fun(y, z, t) + d2vdz2_fun(y, z, t)) - Ro*u_fun(y, z, t)
    nsz_fun(y, z, t) = -vdwdy_fun(y, z, t) - wdwdz_fun(y, z, t) + (1/Re)*(d2wdy2_fun(y, z, t) + d2wdz2_fun(y, z, t))
    navierStokesRHS = FFT!(VectorField(grid), VectorField(grid, nsx_fun, nsy_fun, nsz_fun))

    @test Fields.optimalFrequency(cache) ≈ dot(duds, navierStokesRHS)/(norm(duds)^2) atol=1e-6
end
