# initialise field grid
Ny = 32; Nz = 32; Nt = 32
Re = 100; Ro = 0.5
y = chebpts(Ny)
Dy = chebdiff(Ny)
Dy2 = chebddiff(Ny)
grid = Grid(y, Nz, Nt, Dy, Dy2, rand(Ny), 1.0, 1.0)
FFT! = FFTPlan!(grid, flags=ESTIMATE)

# initialise modes
M = 5
ws = ones(Ny)
Ψ = zeros(ComplexF64, Ny, M, Nz, Nt)
for nt in 1:Nt, nz in 1:Nz
    Ψ[:, :, nz, nt] .= @view(qr(rand(ComplexF64, Ny, M)).Q[:, 1:M])
end

# initialise state fields
u_fun(y, z, t)  = (y^3)*exp(cos(z))*sin(t)
v_fun(y, z, t)  = (y^2 - 1)*sin(z)*sin(t)
w_fun(y, z, t)  = cos(2π*y)*exp(sin(z))*sin(t)

# define derivative functions
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

@testset "Residual Gradient Velocity    " begin
    # initialise residual gradient cache
    dR! = ResGrad(grid, Ψ, Re, Ro)

    # assign velocity field
    dR!.spec_cache[1] .= FFT!(SpectralField(grid), PhysicalField(grid, u_fun))
    dR!.spec_cache[2] .= FFT!(SpectralField(grid), PhysicalField(grid, v_fun))
    dR!.spec_cache[3] .= FFT!(SpectralField(grid), PhysicalField(grid, w_fun))

    # compute all the velocity terms
    Fields._update_vel_cache!(dR!)

    # test for correctness
    @test dR!.spec_cache[4] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dudt_fun))
    @test dR!.spec_cache[5] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dvdt_fun))
    @test dR!.spec_cache[6] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dwdt_fun))
    @test dR!.spec_cache[7] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dudy_fun))
    @test dR!.spec_cache[8] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dvdy_fun))
    @test dR!.spec_cache[9] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dwdy_fun))
    @test dR!.spec_cache[10] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dudz_fun))
    @test dR!.spec_cache[11] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dvdz_fun))
    @test dR!.spec_cache[12] ≈ FFT!(SpectralField(grid), PhysicalField(grid, dwdz_fun))
    @test dR!.spec_cache[13] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2udy2_fun))
    @test dR!.spec_cache[14] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2vdy2_fun))
    @test dR!.spec_cache[15] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2wdy2_fun))
    @test dR!.spec_cache[16] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2udz2_fun))
    @test dR!.spec_cache[17] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2vdz2_fun))
    @test dR!.spec_cache[18] ≈ FFT!(SpectralField(grid), PhysicalField(grid, d2wdz2_fun))
    @test dR!.spec_cache[19] ≈ FFT!(SpectralField(grid), PhysicalField(grid, vdudy_fun))
    @test dR!.spec_cache[20] ≈ FFT!(SpectralField(grid), PhysicalField(grid, wdudz_fun))
    @test dR!.spec_cache[21] ≈ FFT!(SpectralField(grid), PhysicalField(grid, vdvdy_fun))
    @test dR!.spec_cache[22] ≈ FFT!(SpectralField(grid), PhysicalField(grid, wdvdz_fun))
    @test dR!.spec_cache[23] ≈ FFT!(SpectralField(grid), PhysicalField(grid, vdwdy_fun))
    @test dR!.spec_cache[24] ≈ FFT!(SpectralField(grid), PhysicalField(grid, wdwdz_fun))
end