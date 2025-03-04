function test_projection()
    @testset "Channel Profile Integration           " begin
        # construct channel profiles to be integrated
        Ny = 64
        y = chebpts(Ny)
        ws = chebws(Ny)
        u = ComplexF64[(x^2)*cos(π*x/2) for x in y]
        v = ComplexF64[exp(-5*(x^2)) for x in y]

        @test Fields.channel_int(u, ws, v) ≈ 0.0530025 rtol=1e-6
    end

    @testset "Projection of Vector Field            " begin
        # construct a field of modes
        Ny = 16; Nz = 17; Nt = 17
        M = rand(1:12)
        grid = Grid(chebpts(Ny), Nz, Nt, chebdiff(Ny), chebddiff(Ny), ones(Ny), 1.0, 1.0)
        Ψ = zeros(ComplexF64, 3*Ny, M, (Nz >> 1) + 1, Nt)
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
            Ψ[:, :, nz, nt] .= @view(qr(rand(ComplexF64, 3*Ny, M)).Q[:, 1:M])
        end

        # generate field as combination of modes
        a = SpectralField(grid, Ψ)
        u = VectorField(grid)
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
            u[1][:, nz, nt] .= Ψ[1:Ny, :, nz, nt]*a[:, nz, nt]
            u[2][:, nz, nt] .= Ψ[Ny+1:2*Ny, :, nz, nt]*a[:, nz, nt]
            u[3][:, nz, nt] .= Ψ[2*Ny+1:3*Ny, :, nz, nt]*a[:, nz, nt]
        end

        @test project(u, Ψ) ≈ a
    end

    @testset "Expansion of a Field                  " begin
        # construct the field of modes
        Ny = 16; Nz = 17; Nt = 17
        M = rand(1:12)
        grid = Grid(chebpts(Ny), Nz, Nt, chebdiff(Ny), chebddiff(Ny), ones(Ny), 1.0, 1.0)
        Ψ = zeros(ComplexF64, 3*Ny, M, Nz, Nt)
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
            Ψ[:, :, nz, nt] .= @view(qr(rand(ComplexF64, 3*Ny, M)).Q[:, 1:M])
        end

        # construct field as a combination of the modes
        a = SpectralField(grid, Ψ)
        u = VectorField(grid)
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
            u[1][:, nz, nt] .= Ψ[1:Ny, :, nz, nt]*a[:, nz, nt]
            u[2][:, nz, nt] .= Ψ[Ny+1:2*Ny, :, nz, nt]*a[:, nz, nt]
            u[3][:, nz, nt] .= Ψ[2*Ny+1:3*Ny, :, nz, nt]*a[:, nz, nt]
        end

        # project and reverse to get the original field
        @test expand!(VectorField(grid), project(u, Ψ), Ψ) ≈ u
    end
end
