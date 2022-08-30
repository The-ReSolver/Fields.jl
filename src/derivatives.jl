# This file contains the definitions of a number of useful operations to be
# performed on the fields defined in this package.

# These functions are defined only for the spectral fields since the
# derivatives are obtained in the most efficient manner in this space.

ddy!(u::SpectralField{Ny, Nz, Nt}, dudy::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = LinearAlgebra.mul!(dudy, get_Dy(u), u)
ddy!(u::VectorField{N, S}, dudy::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}} = ddy!.(u, dudy)

d2dy2!(u::SpectralField{Ny, Nz, Nt}, d2udy2::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt} = LinearAlgebra.mul!(d2udy2, get_Dy2(u), u)
d2dy2!(u::VectorField{N, S}, d2udy2::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}} = d2dy2!.(u, d2udy2)

function ddz!(u::SpectralField{Ny, Nz, Nt}, dudz::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # extract spanwise domain info from grid
    β = get_β(u)

    # loop over spanwise modes multiplying by modifier
    @inbounds begin
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), ny in 1:Ny
            dudz[ny, nz, nt] = (1im*(nz - 1)*β)*u[ny, nz, nt]
        end
    end

    return dudz
end
ddz!(u::VectorField{N, S}, dudz::VectorField{N, S}) where {N, S} = ddz!.(u, dudz)

function d2dz2!(u::SpectralField{Ny, Nz, Nt}, d2udz2::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # extract spanwise domain info from grid
    β = get_β(u)

    # loop over spanwise modes multiplying by modifier
    @inbounds begin
        for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), ny in 1:Ny
            d2udz2[ny, nz, nt] = (-(((nz - 1)*β)^2))*u[ny, nz, nt]
        end
    end

    return d2udz2
end
d2dz2!(u::VectorField{N, S}, d2udz2::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}} = d2dz2!.(u, d2udz2)

function ddt!(u::SpectralField{Ny, Nz, Nt}, dudt::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # extract temporal domain info from grid
    ω = get_ω(u)

    # loop over positive temporal modes multiplying by modifier
    @inbounds begin
        for nt in 1:floor(Int, Nt/2), nz in 1:((Nz >> 1) + 1), ny in 1:Ny
            dudt[ny, nz, nt] = (1im*(nt - 1)*ω)*u[ny, nz, nt]
        end
    end

    # loop over negative temporal modes multiplying by modifier
    @inbounds begin
        for nt in floor(Int, (Nt/2) + 1):Nt, nz in 1:((Nz >> 1) + 1), ny in 1:Ny
            dudt[ny, nz, nt] = (1im*(nt - 1 - Nt)*ω)*u[ny, nz, nt]
        end
    end

    return dudt
end
ddt!(u::VectorField{N, S}, dudt::VectorField{N, S}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}} = ddt!.(u, dudt)
