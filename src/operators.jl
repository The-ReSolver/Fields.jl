# This file contains the definitions of a number of useful operations to be
# performed on the fields defined in this package.

# These functions are defined only for the spectral fields since the
# derivatives are obtained in the most efficient manner in this space.

function ddy(û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # take grid from spectral field
    # take differentiation matrix from grid
    # multiply field at every nz and nt
end

function d2dy2(û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # take grid from spectral field
    # take double differentiation matrix from grid
    # multiply field at every nz and nt
end

function ddz(û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # loop over Nz>>1+1 multiplying by i*k_z*β
end

function d2dz2(û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # loop over Nz>>1+1 multiplying by -(k_z*β)^2
end

function ddt(û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # loop over Nt multiplying by i*k_t_ω
end
