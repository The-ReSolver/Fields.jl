# This file contain the utility functions to allow the transformation of a
# spectral field into an equivalent state vector

function _velocityCoefficientsToVector!(vector, velocityCoefficients::SpectralField{<:Grid{Ny, Nz, Nt}, true}, ω) where {Ny, Nz, Nt}
    M = size(velocityCoefficients, 1)
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), m in 1:M
        vector[2*(m + M*(nz - 1) + M*((Nz >> 1) + 1)*(nt - 1)) - 1] = real(velocityCoefficients[m, nz, nt])
        vector[2*(m + M*(nz - 1) + M*((Nz >> 1) + 1)*(nt - 1))]     = imag(velocityCoefficients[m, nz, nt])
    end
    vector[end] = ω
    return vector
end

function _vectorToVelocityCoefficients!(velocityCoefficients::SpectralField{<:Grid{Ny, Nz, Nt}, true}, vector) where {Ny, Nz, Nt}
    M = size(velocityCoefficients, 1)
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1), m in 1:M
        velocityCoefficients[m, nz, nt] = vector[2*(m + M*(nz - 1) + ((Nz >> 1) + 1)*M*(nt - 1)) - 1] + 1im*vector[2*(m + M*(nz - 1) + ((Nz >> 1) + 1)*M*(nt - 1))]
    end
    velocityCoefficients.grid.dom[2] = vector[end]
    return velocityCoefficients
end
