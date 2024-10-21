# This file contains the definitions of a number of useful operations to be
# performed on the fields defined in this package.

# These functions are defined only for the spectral fields since the
# derivatives are obtained in the most efficient manner in this space.

ddy!(u::SpectralField{<:Grid{Ny, Nz, Nt}}, dudy::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt} = LinearAlgebra.mul!(dudy, get_Dy(u), u)
function ddy!(u::VectorField{N, S}, dudy::VectorField{N, S}) where {N, S<:SpectralField}
    for i in 1:N
        @inbounds ddy!(u[i], dudy[i])
    end

    return dudy
end

d2dy2!(u::SpectralField{<:Grid{Ny, Nz, Nt}}, d2udy2::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt} = LinearAlgebra.mul!(d2udy2, get_Dy2(u), u)
function d2dy2!(u::VectorField{N, S}, d2udy2::VectorField{N, S}) where {N, S<:SpectralField}
    for i in 1:N
        @inbounds d2dy2!(u[i], d2udy2[i])
    end

    return d2udy2
end

function ddz!(u::SpectralField{<:Grid{Ny, Nz, Nt}}, dudz::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt}
    # extract spanwise domain info from grid
    β = get_β(u)

    # loop over spanwise modes multiplying by modifier
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), ny in 1:Ny
        @inbounds dudz[ny, nz, nt] = (1im*(nz - 1)*β)*u[ny, nz, nt]
    end

    return dudz
end
function ddz!(u::VectorField{N, S}, dudz::VectorField{N, S}) where {N, S<:SpectralField}
    for i in 1:N
        @inbounds ddz!(u[i], dudz[i])
    end

    return dudz
end

function d2dz2!(u::SpectralField{<:Grid{Ny, Nz, Nt}}, d2udz2::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt}
    # extract spanwise domain info from grid
    β = get_β(u)

    # loop over spanwise modes multiplying by modifier
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), ny in 1:Ny
        @inbounds d2udz2[ny, nz, nt] = (-(((nz - 1)*β)^2))*u[ny, nz, nt]
    end

    return d2udz2
end
function d2dz2!(u::VectorField{N, S}, d2udz2::VectorField{N, S}) where {N, S<:SpectralField}
    for i in 1:N
        @inbounds d2dz2!(u[i], d2udz2[i])
    end

    return d2udz2
end

function ddt!(u::SpectralField{<:Grid{Ny, Nz, Nt}}, dudt::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt}
    # extract temporal domain info from grid
    ω = get_ω(u)

    # loop over positive temporal modes multiplying by modifier
    for nt in 1:((Nt >> 1) + 1), nz in 1:((Nz >> 1) + 1), ny in 1:Ny
        @inbounds dudt[ny, nz, nt] = (1im*(nt - 1)*ω)*u[ny, nz, nt]
    end

    # loop over negative temporal modes multiplying by modifier
    if Nt > 1
        for nt in ((Nt >> 1) + 2):Nt, nz in 1:((Nz >> 1) + 1), ny in 1:Ny
            @inbounds dudt[ny, nz, nt] = (1im*(nt - 1 - Nt)*ω)*u[ny, nz, nt]
        end
    end

    return dudt
end
function ddt!(u::VectorField{N, S}, dudt::VectorField{N, S}) where {N, S<:SpectralField}
    for i in 1:N
        @inbounds ddt!(u[i], dudt[i])
    end

    return dudt
end
