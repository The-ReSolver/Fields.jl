@testset "Objective constraint          " begin
    # initialise field grid
    Ny = 32; Nz = 32; Nt = 32
    Re = 100; Ro = 0.5
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    grid = Grid(y, Nz, Nt, Dy, Dy2, rand(Ny), 1.0, 1.0)
    FFT! = FFTPlan!(grid, flags=ESTIMATE)

    # initialise state fields
    u_fun(y, z, t)  = (y^3)*exp(cos(z))*sin(t)
    v_fun(y, z, t)  = (y^2 - 1)*sin(z)*sin(t)
    w_fun(y, z, t)  = cos(2π*y)*exp(sin(z))*sin(t)
    rx_fun(y, z, t) = (y^3)*exp(sin(z))*sin(t)
    ry_fun(y, z, t) = (y^2)*cos(z)*sin(t)
    rz_fun(y, z, t) = exp(cos(y))*sin(z)*sin(t)
    p_fun(y, z, t)  = (sin(π*y)^2)*cos(sin(z))*sin(t)
    q = FFT!(VectorField(grid, N=8), VectorField(grid, u_fun, v_fun, w_fun, rx_fun, ry_fun, rz_fun, p_fun, (y, z, t)->0.0))

    # define derivative functions
    dudt_fun(y, z, t)   = (y^3)*exp(cos(z))*cos(t)
    dvdt_fun(y, z, t)   = (y^2 - 1)*sin(z)*cos(t)
    dwdt_fun(y, z, t)   = cos(2π*y)*exp(sin(z))*cos(t)
    dudy_fun(y, z, t)   = (3*y^2)*exp(cos(z))*sin(t)
    dvdy_fun(y, z, t)   = 2*y*sin(z)*sin(t)
    dwdy_fun(y, z, t)   = -2π*sin(2π*y)*exp(sin(z))*sin(t)
    dudz_fun(y, z, t)   = -(y^3)*sin(z)*exp(cos(z))*sin(t)
    dvdz_fun(y, z, t)   = (y^2 - 1)*cos(z)*sin(t)
    dwdz_fun(y, z, t)   = cos(2π*y)*cos(z)*exp(sin(z))*sin(t)
    d2udy2_fun(y, z, t) = 6*y*exp(cos(z))*sin(t)
    d2vdy2_fun(y, z, t) = 2*sin(z)*sin(t)
    d2wdy2_fun(y, z, t) = -4π^2*cos(2π*y)*exp(sin(z))*sin(t)
    d2udz2_fun(y, z, t) = (y^3)*(sin(z)^2 - cos(z))*exp(cos(z))*sin(t)
    d2vdz2_fun(y, z, t) = (1 - y^2)*sin(z)*sin(t)
    d2wdz2_fun(y, z, t) = cos(2π*y)*(cos(z)^2 - sin(z))*exp(sin(z))*sin(t)
    drydy_fun(y, z, t)  = 2*y*cos(z)*sin(t)
    drzdz_fun(y, z, t)  = exp(cos(y))*cos(z)*sin(t)
    dpdy_fun(y, z, t)   = π*sin(2π*y)*cos(sin(z))*sin(t)
    dpdz_fun(y, z, t)   = -(sin(π*y)^2)*cos(z)*sin(sin(z))*sin(t)

    # initialise constraint output fields
    function out1_fun(y, z, t)
        if y ∈ [-1.0, 1.0]
            dudt_fun(y, z, t) + v_fun(y, z, t)*dudy_fun(y, z, t) + w_fun(y, z, t)*dudz_fun(y, z, t) - (1/Re)*(d2udy2_fun(y, z, t) + d2udz2_fun(y, z, t)) - Ro*v_fun(y, z, t)
        else
            dudt_fun(y, z, t) + v_fun(y, z, t)*dudy_fun(y, z, t) + w_fun(y, z, t)*dudz_fun(y, z, t) - (1/Re)*(d2udy2_fun(y, z, t) + d2udz2_fun(y, z, t)) - Ro*v_fun(y, z, t) - rx_fun(y, z, t)
        end
    end
    function out2_fun(y, z, t)
        if y ∈ [-1.0, 1.0]
            dvdt_fun(y, z, t) + v_fun(y, z, t)*dvdy_fun(y, z, t) + w_fun(y, z, t)*dvdz_fun(y, z, t) - (1/Re)*(d2vdy2_fun(y, z, t) + d2vdz2_fun(y, z, t)) + Ro*u_fun(y, z, t) + dpdy_fun(y, z, t)
        else
            dvdt_fun(y, z, t) + v_fun(y, z, t)*dvdy_fun(y, z, t) + w_fun(y, z, t)*dvdz_fun(y, z, t) - (1/Re)*(d2vdy2_fun(y, z, t) + d2vdz2_fun(y, z, t)) + Ro*u_fun(y, z, t) - ry_fun(y, z, t) + dpdy_fun(y, z, t)
        end
    end
    function out3_fun(y, z, t)
        if y ∈ [-1.0, 1.0]
            dwdt_fun(y, z, t) + v_fun(y, z, t)*dwdy_fun(y, z, t) + w_fun(y, z, t)*dwdz_fun(y, z, t) - (1/Re)*(d2wdy2_fun(y, z, t) + d2wdz2_fun(y, z, t)) + dpdz_fun(y, z, t)
        else
            dwdt_fun(y, z, t) + v_fun(y, z, t)*dwdy_fun(y, z, t) + w_fun(y, z, t)*dwdz_fun(y, z, t) - (1/Re)*(d2wdy2_fun(y, z, t) + d2wdz2_fun(y, z, t)) - rz_fun(y, z, t) + dpdz_fun(y, z, t)
        end
    end
    out4_fun(y, z, t) = dvdy_fun(y, z, t)  + dwdz_fun(y, z, t)
    out5_fun(y, z, t) = drydy_fun(y, z, t) + drzdz_fun(y, z, t)
    g = FFT!(VectorField(grid, N=5), VectorField(grid, out1_fun, out2_fun, out3_fun, out4_fun, out5_fun))

    # compute objective constraints
    G! = Constraint(grid, Re, Ro)
    out = G!(VectorField(grid, N=5), q)

    @test out ≈ g
end

@testset "Objective evolution           " begin

end
