# utility functions for some basic analysis of spectral fields

function energy(p::VectorField{N, P}) where {N, Ny, Nz, Nt, P<:PhysicalField{<:Grid{Ny, Nz, Nt}}}
    K = zeros(Nt)
    for i in 1:N
        K .+= energy(p[i])
    end
    return K
end

function energy!(K, p::PhysicalField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt}
    L = 2π/get_β(p)
    for nt in 1:Nt, ny in 1:Ny
        K[nt] += get_ws(p)[ny]*sum(norm(p[ny, :, nt])^2)*L/(2*Nz)
    end
    return K
end
energy(p::PhysicalField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt} = energy!(zeros(Nt), p)

function energy(u::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{<:Grid{Ny, Nz, Nt}}}
    K = zeros(Nt)
    up = [PhysicalField{false, 1.0}(get_grid(u), brfft(parent(u[i]), Nz, [2, 3])) for i in 1:N]
    for i in 1:N
        K .+= energy(up[i])
    end
    return K
end
energy(u::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt} = energy(PhysicalField{false, 1.0}(get_grid(u), brfft(parent(u), Nz, [2, 3])))


function vorticity!(ω::VectorField{3, S}, u::VectorField{3, S}) where {S<:SpectralField}
    dudy = ddy!(u, similar(u))
    dudz = ddz!(u, similar(u))
    ω[1] .=   dudy[3] .- dudz[2]
    ω[2] .=   dudz[1]
    ω[3] .= .-dudy[1]
    return ω
end

enstrophy(u::VectorField{3, <:SpectralField}) = 2.0.*energy(vorticity!(similar(u), u))

function boundaryEnergy(u::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt}
    I = zeros(Nt)
    L = 2π/get_β(u)
    dudy = brfft(parent(ddy!(u, similar(u))), Nz, [2, 3])
    for nt in 1:Nt
        I[nt] += sum(@view(dudy[1, :, nt]))*L/Nz
        I[nt] += sum(@view(dudy[end, :, nt]))*L/Nz
    end
    return I
end
