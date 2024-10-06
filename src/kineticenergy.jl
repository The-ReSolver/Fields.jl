# Definitions to compute the kinetic energy of a scalar or vector field.


function energy!(K, p::PhysicalField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt}
    q = rfft(parent(p), [2])./Nz
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), ny in 1:Ny
        K[nt] += p.grid.ws[ny]*norm(q[ny, nz, nt])^2
    end
    for nt in 1:Nt, ny in 1:Ny
        K[nt] += 0.5*p.grid.ws[ny]*norm(q[ny, 1, nt])^2
    end
    β = get_β(p)
    K .*= 2π/β
    return K
end
energy(p::PhysicalField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt} = energy!(zeros(Nt), p)

function energy(p::VectorField{N, P}) where {N, Ny, Nz, Nt, P<:PhysicalField{<:Grid{Ny, Nz, Nt}}}
    K = zeros(Nt)
    for i in 1:N
        K .+= energy(p[i])
    end
    return K
end

function energy(u::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt}
    up = PhysicalField{false}(get_grid(u), brfft(parent(u), Nz, [2, 3]))
    return energy(up)
end

function energy(u::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{<:Grid{Ny, Nz, Nt}}}
    K = zeros(Nt)
    up = [PhysicalField{false}(get_grid(u), brfft(parent(u[i]), Nz, [2, 3])) for i in 1:N]
    for i in 1:N
        K .+= energy(up[i])
    end
    return K
end
