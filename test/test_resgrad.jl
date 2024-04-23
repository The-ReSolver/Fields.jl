# initialise field grid
Ny = 32; Nz = 33; Nt = 33; M = 5
Re = 100.0; Ro = 0.5
y = chebpts(Ny)
Dy = chebdiff(Ny)
Dy2 = chebddiff(Ny)
ω = 1.0
β = 1.0
grid = Grid(y, Nz, Nt, Dy, Dy2, rand(Ny), ω, β)
FFT! = FFTPlan!(grid, true, flags=ESTIMATE)
modes = Array{ComplexF64}(undef, 3*Ny, M, (Nz >> 1) + 1, Nt)

# initialise residual gradient cache
cache = ResGrad(grid, modes, y, Re, Ro)

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

# TODO: check allocated of update methods
@testset "Residual Gradient Velocity            " begin
    # assign velocity field
    cache.spec_cache[1] .= FFT!(VectorField(grid), VectorField(grid, u_fun, v_fun, w_fun, dealias=true))

    # compute the cache
    Fields._update_vel_cache!(cache)

    # test for correctness
    @test cache.spec_cache[2]  ≈ FFT!(VectorField(grid), VectorField(grid, dudt_fun, dvdt_fun, dwdt_fun,       dealias=true))
    @test cache.spec_cache[3]  ≈ FFT!(VectorField(grid), VectorField(grid, dudy_fun, dvdy_fun, dwdy_fun,       dealias=true))
    @test cache.spec_cache[4]  ≈ FFT!(VectorField(grid), VectorField(grid, dudz_fun, dvdz_fun, dwdz_fun,       dealias=true))
    @test cache.spec_cache[5]  ≈ FFT!(VectorField(grid), VectorField(grid, d2udy2_fun, d2vdy2_fun, d2wdy2_fun, dealias=true))
    @test cache.spec_cache[6]  ≈ FFT!(VectorField(grid), VectorField(grid, d2udz2_fun, d2vdz2_fun, d2wdz2_fun, dealias=true))
    @test cache.spec_cache[7]  ≈ FFT!(VectorField(grid), VectorField(grid, vdudy_fun, vdvdy_fun, vdwdy_fun,    dealias=true))
    @test cache.spec_cache[8]  ≈ FFT!(VectorField(grid), VectorField(grid, wdudz_fun, wdvdz_fun, wdwdz_fun,    dealias=true))
end

@testset "Residual Gradient Residual            " begin
    # assign velocity and residual fields
    cache.spec_cache[1]  .= FFT!(VectorField(grid), VectorField(grid, u_fun,  v_fun,  w_fun,  dealias=true))
    cache.spec_cache[10] .= FFT!(VectorField(grid), VectorField(grid, rx_fun, ry_fun, rz_fun, dealias=true))

    # update the cache
    Fields._update_vel_cache!(cache)
    Fields._update_res_cache!(cache)

    # test for correctness
    @test cache.spec_cache[11] ≈ FFT!(VectorField(grid), VectorField(grid, drxdt_fun, drydt_fun, drzdt_fun,        dealias=true))
    @test cache.spec_cache[12] ≈ FFT!(VectorField(grid), VectorField(grid, drxdy_fun, drydy_fun, drzdy_fun,        dealias=true))
    @test cache.spec_cache[13] ≈ FFT!(VectorField(grid), VectorField(grid, drxdz_fun, drydz_fun, drzdz_fun,        dealias=true))
    @test cache.spec_cache[14] ≈ FFT!(VectorField(grid), VectorField(grid, d2rxdy2_fun, d2rydy2_fun, d2rzdy2_fun,  dealias=true))
    @test cache.spec_cache[15] ≈ FFT!(VectorField(grid), VectorField(grid, d2rxdz2_fun, d2rydz2_fun, d2rzdz2_fun,  dealias=true))
    @test cache.spec_cache[16] ≈ FFT!(VectorField(grid), VectorField(grid, vdrxdy_fun, vdrydy_fun, vdrzdy_fun,     dealias=true))
    @test cache.spec_cache[17] ≈ FFT!(VectorField(grid), VectorField(grid, wdrxdz_fun, wdrydz_fun, wdrzdz_fun,     dealias=true))
    @test cache.spec_cache[18] ≈ FFT!(VectorField(grid), VectorField(grid, (y, z, t)->0.0, rxdudy_fun, rxdudz_fun, dealias=true))
    @test cache.spec_cache[19] ≈ FFT!(VectorField(grid), VectorField(grid, (y, z, t)->0.0, rydvdy_fun, rydvdz_fun, dealias=true))
    @test cache.spec_cache[20] ≈ FFT!(VectorField(grid), VectorField(grid, (y, z, t)->0.0, rzdwdy_fun, rzdwdz_fun, dealias=true))
end

@testset "Residual Gradient Symmetry            " begin
    a = SpectralField(grid, modes)
    a .= rand(ComplexF64, M, (Nz >> 1) + 1, Nt)
    a[:, 1, 1] .= real.(a[:, 1, 1])
    Fields.apply_symmetry!(a)
    cache(a)
    symmetric = true
    for i in 2:(((Nt - 1) >> 1) + 1)
        if !isapprox(cache.out[:, 1, i], cache.out[:, 1, end - i + 2], rtol=1e-8)
            symmetric = false
            break
        end
    end
    @test symmetric
end

@testset "Optimal Frequency                     " begin
    # set velocity profile
    cache.spec_cache[1] .= FFT!(VectorField(grid), VectorField(grid, u_fun, v_fun, w_fun, dealias=true))

    # run the cache update
    Fields._update_vel_cache!(cache)

    # compute the optimal frequency manually
    duds = FFT!(VectorField(grid), VectorField(grid, dudt_fun, dvdt_fun, dwdt_fun, dealias=true))./ω
    nsx_fun(y, z, t) = -vdudy_fun(y, z, t) - wdudz_fun(y, z, t) + (1/Re)*(d2udy2_fun(y, z, t) + d2udz2_fun(y, z, t)) + Ro*v_fun(y, z, t)
    nsy_fun(y, z, t) = -vdvdy_fun(y, z, t) - wdvdz_fun(y, z, t) + (1/Re)*(d2vdy2_fun(y, z, t) + d2vdz2_fun(y, z, t)) - Ro*u_fun(y, z, t)
    nsz_fun(y, z, t) = -vdwdy_fun(y, z, t) - wdwdz_fun(y, z, t) + (1/Re)*(d2wdy2_fun(y, z, t) + d2wdz2_fun(y, z, t))
    navierStokesRHS = FFT!(VectorField(grid), VectorField(grid, nsx_fun, nsy_fun, nsz_fun, dealias=true))

    @test Fields.optimalFrequency(cache) ≈ dot(duds, navierStokesRHS)/(norm(duds)^2) atol=1e-6
end
