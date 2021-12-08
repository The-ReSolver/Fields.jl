# This file contains the definitions of a number of useful operations to be
# performed on the fields defined in this package.

# These functions are defined only for the spectral fields since the
# derivatives are obtained in the most efficient manner in this space.

export ddy!, d2dy2!, ddz!, d2dz2!, ddt!

function ddy!(u::SpectralField{Ny, Nz, Nt},
                dudy::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # take differentiation matrix from grid
    Dy = u.grid.Dy[1]

    # multiply field at every nz and nt
    @views begin
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
            LinearAlgebra.mul!(dudy[:, nz, nt], Dy, u[:, nz, nt])
        end
    end

    return dudy
end

function ddy!(u::VectorField{N, S}, dudy::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    for i in 1:N
        ddy!(u[i], dudy[i])
    end

    return dudy
end

function d2dy2!(u::SpectralField{Ny, Nz, Nt},
                d2udy2::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # take second differentiation matrix from grid
    Dy2 = u.grid.Dy[2]

    # multiply field at every nz and nt
    @views begin
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
            LinearAlgebra.mul!(d2udy2[:, nz, nt], Dy2, u[:, nz, nt])
        end
    end

    return d2udy2
end

function d2dy2!(u::VectorField{N, S}, d2udy2::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    for i in 1:N
        d2dy2!(u[i], d2udy2[i])
    end

    return d2udy2
end

function ddz!(u::SpectralField{Ny, Nz, Nt},
                dudz::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # extract spanwise domain info from grid
    β = u.grid.dom[2]

    # loop over spanwise modes multiplying by modifier
    @views begin
        for nz in 1:((Nz >> 1) + 1)
            dudz[:, nz, :] .= (1im*(nz - 1)*β).*u[:, nz, :]
        end
    end

    return dudz
end

function ddz!(u::VectorField{N, S}, dudz::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    for i in 1:N
        ddz!(u[i], dudz[i])
    end

    return dudz
end

function d2dz2!(u::SpectralField{Ny, Nz, Nt},
                d2udz2::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # extract spanwise domain info from grid
    β = u.grid.dom[2]

    # loop over spanwise modes multiplying by modifier
    @views begin
        for nz in 1:((Nz >> 1) + 1)
            d2udz2[:, nz, :] .= (-(((nz - 1)*β)^2)).*u[:, nz, :]
        end
    end

    return d2udz2
end

function d2dz2!(u::VectorField{N, S}, d2udz2::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    for i in 1:N
        d2ddz2!(u[i], d2udz2[i])
    end

    return d2udz2
end

function ddt!(u::SpectralField{Ny, Nz, Nt},
                dudt::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # extract temporal domain info from grid
    ω = u.grid.dom[1]

    # loop over positive temporal modes multiplying by modifier
    @views begin
        for nt in 1:floor(Int, Nt/2)
            dudt[:, :, nt] .= (1im*(nt - 1)*ω).*u[:, :, nt]
        end
    end

    # loop over negative temporal modes multiplying by modifier
    @views begin
        for nt in floor(Int, (Nt/2) + 1):Nt
            dudt[:, :, nt] .= (1im*(nt - 1 - Nt)*ω).*u[:, :, nt]
        end
    end

    return dudt
end

function ddt!(u::VectorField{N, S}, dudt::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    for i in 1:N
        ddt!(u[i], dudt[i])
    end

    return dudt
end
