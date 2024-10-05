@testset "Channel Profile Integration           " begin
    # construct channel profiles to be integrated
    Ny = 64
    y = chebpts(Ny)
    ws = chebws(Ny)
    u = [(x^2)*cos(π*x/2) for x in y]
    v = [exp(-5*(x^2)) for x in y]

    @test Fields.channel_int(u, ws, v) ≈ 0.0530025 rtol=1e-6
end

@testset "Projection of Vector Field            " begin
    # construct a field of modes
    Ny = 16; Nz = 16; Nt = 16
    M = rand(1:12)
    ws = ones(Ny)
    Ψ = zeros(ComplexF64, 3*Ny, M, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz
        Ψ[:, :, nz, nt] .= @view(qr(rand(ComplexF64, 3*Ny, M)).Q[:, 1:M])
    end

    # generate field as combination of modes
    a = rand(ComplexF64, M, Nz, Nt)
    u = [zeros(ComplexF64, Ny, Nz, Nt) for i in 1:3]
    for nt in 1:Nt, nz in 1:Nz
        u[1][:, nz, nt] .= Ψ[1:Ny, :, nz, nt]*a[:, nz, nt]
        u[2][:, nz, nt] .= Ψ[Ny+1:2*Ny, :, nz, nt]*a[:, nz, nt]
        u[3][:, nz, nt] .= Ψ[2*Ny+1:3*Ny, :, nz, nt]*a[:, nz, nt]
    end

    @test project(u, ws, Ψ) ≈ a
end

@testset "Expansion of a Field                  " begin
    # construct the field of modes
    Ny = 16; Nz = 16; Nt = 16
    M = rand(1:12)
    ws = ones(Ny)
    Ψ = zeros(ComplexF64, 3*Ny, M, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz
        Ψ[:, :, nz, nt] .= @view(qr(rand(ComplexF64, 3*Ny, M)).Q[:, 1:M])
    end

    # construct field as a combination of the modes
    a = rand(ComplexF64, M, Nz, Nt)
    u = [zeros(ComplexF64, Ny, Nz, Nt) for i in 1:3]
    v = [zeros(ComplexF64, Ny, Nz, Nt) for i in 1:3]
    for nt in 1:Nt, nz in 1:Nz
        u[1][:, nz, nt] .= Ψ[1:Ny, :, nz, nt]*a[:, nz, nt]
        u[2][:, nz, nt] .= Ψ[Ny+1:2*Ny, :, nz, nt]*a[:, nz, nt]
        u[3][:, nz, nt] .= Ψ[2*Ny+1:3*Ny, :, nz, nt]*a[:, nz, nt]
    end

    # project and reverse to get the original field
    expand!(v, project(u, ws, Ψ), Ψ)

    @test v ≈ u
end
