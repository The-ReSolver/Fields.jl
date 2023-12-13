@testset "Simulation outputs            " begin
    # initialise inputs to optimisation
    Ny = 5; Nz = 3; Nt = 2; M = 1;
    y = collect(range(-1, 1, length=Ny))
    β = abs(rand())
    ω = abs(rand())
    g = Grid(y, Nz, Nt, DiffMatrix(y, 3, 1), DiffMatrix(y, 3, 2), quadweights(y, 2), ω, β)
    a = SpectralField(Grid(ones(M), Nz, Nt, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), ones(M), ω, β))
    a .= rand(ComplexF64, M, (Nz >> 1) + 1, Nt)
    modes = rand(ComplexF64, 3*Ny, M, (Nz >> 1) + 1, Nt)
    base_prof = rand(Ny)
    Re = abs(rand())
    Ro = abs(rand())
    free_mean = rand([true, false])
    options = OptOptions(maxiter=2, write_loc="./tmp/")
    try
        mkdir("./tmp")
    catch
        nothing
    end

    # generate output directory
    Fields._init_opt_dir(options, g, modes, base_prof, Re, Ro, free_mean)

    # test directory structure
    ini = read(Inifile(), "./tmp/params")
    @test isfile("./tmp/base_profile")
    @test isfile("./tmp/modes")
    @test isfile("./tmp/y")
    @test get(ini, "sim_data", "Re") == string(Re)
    @test get(ini, "sim_data", "Ro") == string(Ro)
    @test get(ini, "sim_data", "free_mean") == string(free_mean)
    @test get(ini, "grid_data", "Ny") == string(Ny)
    @test get(ini, "grid_data", "Nz") == string(Nz)
    @test get(ini, "grid_data", "Nt") == string(Nt)
    @test get(ini, "grid_data", "M") == string(M)
    @test get(ini, "grid_data", "Ny") == string(Ny)
    @test get(ini, "grid_data", "beta") == string(β)
    @test get(ini, "grid_data", "omega") == string(ω)

    # write single field instance
    i = rand(1:10)
    Fields._write_data("./tmp/", i, a)

    # check file has been successfully written
    @test isfile("./tmp/"*string(i)*"/a")
    a2 = Array{ComplexF64}(undef, M, (Nz >> 1) + 1, Nt)
    open("./tmp/"*string(i)*"/a", "r") do f
        read!(f, a2)
    end
    @test parent(a) == a2

    # tear down all the files created
    rm("./tmp/base_profile")
    rm("./tmp/modes")
    rm("./tmp/params")
    rm("./tmp/y")
    rm("./tmp/"*string(i)*"/a")
    rm("./tmp/"*string(i))
    rm("./tmp")
end
