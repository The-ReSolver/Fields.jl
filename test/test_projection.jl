@testset "Channel Profile Integration   " begin
    # construct channel profiles to be integrated
    Ny = 64
    y = chebpts(Ny)
    ws = chebws(Ny)
    u = [(x^2)*cos(π*x/2) for x in y]
    v = [exp(-5*(x^2)) for x in y]

    @test Fields.channel_int(u, ws, v) ≈ 0.0530025 rtol=1e-6
end

@testset "Projection Onto Mode Set      " begin
    # construct a set of modes
    N = 64
    M = rand(1:12)
    ws = ones(N)
    Ψ = @view(qr(rand(ComplexF64, N, M)).Q[:, 1:M])

    # generate channel profile as combination of modes
    a = rand(ComplexF64, M)
    u = Ψ*a

    @test project(u, ws, Ψ) ≈ a
end

@testset "Projection of Field           " begin
    # construct a field of modes
    Ny = 16; Nz = 16; Nt = 16
    M = rand(1:12)
    ws = ones(Ny)
    Ψ = zeros(ComplexF64, Ny, M, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz
        Ψ[:, :, nz, nt] .= @view(qr(rand(ComplexF64, Ny, M)).Q[:, 1:M])
    end

    # generate field as combination of modes
    a = rand(ComplexF64, M, Nz, Nt)
    u = zeros(ComplexF64, Ny, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz
        u[:, nz, nt] .= Ψ[:, :, nz, nt]*a[:, nz, nt]
    end

    @test project(u, ws, Ψ) ≈ a
end
