# Definitions to compute the kinetic energy of a scalar or vector field.


function energy!(K, p::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
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
energy(p::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = energy!(zeros(Nt), p)

function energy(p::VectorField{N, P}) where {N, Nt, P<:PhysicalField{<:Any, <:Any, Nt}}
    K = zeros(Nt)
    for i in 1:N
        energy!(K, p[i])
    end
    return K
end

function energy(u::SpectralField, IFFT::IFFTPlan!)
    up = IFFT(PhysicalField(get_grid(u)), u)
    return energy(up)
end

function energy(u::VectorField{N, S}, IFFT::IFFTPlan!) where {N, Nt, S<:SpectralField{<:Any, <:Any, Nt}}
    K = zeros(Nt)
    up = IFFT(VectorField(get_grid(u), fieldType=PhysicalField))
    for i in 1:N
        energy!(K, up[i])
    end
    return K
end
