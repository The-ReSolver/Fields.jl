@testset "Mode Generation                       " begin
    # set parameters
    retain = rand(1:10)
    N = rand(1:16)

    # define function to generate mode for given frequencies
    generateRandomModes(kz, kt) = fill!(Matrix{ComplexF64}(undef, 3*N, retain), kz + 1im*kt)

    # construct grid
    Nz = rand(5:10)
    Nt = rand(5:10)
    β = abs(rand())
    ω = abs(rand())
    grid = Grid(Vector{Float64}(undef, N), Nz, Nt, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), Vector{Float64}(undef, 0), ω, β)

    # generate the grid of modes
    modes = generateGridOfModes(grid, retain, generateRandomModes)

    @test size(modes) == (3*N, retain, (Nz >> 1) + 1, Nt)
    ifCorrectValues = true
    for nz in 0:(Nz >> 1), nt in 0:(Nt - 1)
        modes[:, :, nz + 1, nt + 1] == (nz*β + nt*ω)*ones(3*N, retain)
    end
    @test ifCorrectValues
end
