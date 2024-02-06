@testset "Initialising Optimisation Directory   " begin
    # initialise inputs
    Ny = 5; Nz = 3; Nt = 2; M = 1;
    y = collect(range(-1, 1, length=Ny))
    β = abs(rand())
    ω = abs(rand())
    grid = Grid(y, Nz, Nt, DiffMatrix(y, 3, 1), DiffMatrix(y, 3, 2), quadweights(y, 2), ω, β)
    modes = rand(ComplexF64, 3*Ny, M, (Nz >> 1) + 1, Nt)
    a = SpectralField(grid, modes)
    a .= rand(ComplexF64, M, (Nz >> 1) + 1, Nt)
    baseProfile = rand(Ny)
    Re = abs(rand())
    Ro = abs(rand())
    ifFreeMean = rand([true, false])
    mkpath("./tmp")

    # write the parameters to the directory
    initialiseOptimisationDirectory("./tmp/", a, modes, baseProfile, Re, Ro, ifFreeMean)

    @test isfile("./tmp/parameters.jld2")
    @test isfile("./tmp/0/velCoeff")

    # read file back and test contents
    grid2, baseProfile2, modes2, Re2, Ro2, ifFreeMean2 = readOptimisationParameters("./tmp/")
    a2 = Fields._readOptimisationVelocityCoefficients("./tmp/0/", similar(a))

    @test grid == grid2
    @test baseProfile == baseProfile2
    @test modes == modes2
    @test Re == Re2
    @test Ro == Ro2
    @test ifFreeMean == ifFreeMean2

    # tear down directory
    rm("./tmp", recursive=true)
end

@testset "Load Optimisation State               " begin
    # initialise inputs
    Ny = 5; Nz = 3; Nt = 2; M = 1;
    y = collect(range(-1, 1, length=Ny))
    β = abs(rand())
    ω = abs(rand())
    grid = Grid(y, Nz, Nt, DiffMatrix(y, 3, 1), DiffMatrix(y, 3, 2), quadweights(y, 2), ω, β)
    modes = rand(ComplexF64, 3*Ny, M, (Nz >> 1) + 1, Nt)
    a = SpectralField(grid, modes)
    a .= rand(ComplexF64, M, (Nz >> 1) + 1, Nt)
    baseProfile = rand(Ny)
    Re = abs(rand())
    Ro = abs(rand())
    ifFreeMean = rand([true, false])
    trace = Fields.Trace([rand()], [rand()], [rand(1:10)], [rand()], [rand()])
    mkpath("./tmp")

    # do the write operations
    initialiseOptimisationDirectory("./tmp/", a, modes, baseProfile, Re, Ro, ifFreeMean)
    Fields.writeIteration("./tmp/0/", a, trace)

    # load state back into memory
    a2, modes2, baseProfile2, Re2, Ro2, ifFreeMean2, trace2 = @test_nowarn loadOptimisationState("./tmp/", 0)

    @test a == a2
    @test get_grid(a) == get_grid(a2)
    @test modes == modes2
    @test baseProfile == baseProfile2
    @test Re == Re2
    @test Ro == Ro2
    @test ifFreeMean == ifFreeMean2
    @test trace.value == trace2.value
    @test trace.g_norm == trace2.g_norm
    @test trace.iter == trace2.iter
    @test trace.time == trace2.time
    @test trace.step_size == trace2.step_size

    # tear down directory
    rm("./tmp", recursive=true)
end
