# This file contains the function definitions to transform a vector field into
# its corresponding state vector for optimisation using Optim.jl.

function field2vec!(vec::Vector{T}, u::VectorField{N, <:SpectralField{Ny, Nz, Nt, <:Any, T}}) where {T, N, Ny, Nz, Nt}
    # get spectral spanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of the field and assign to the state vector
    for i in 1:N, nt in 1:Nt, nz in 1:Nz_spec, ny in 1:Ny
        vec[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec + (i - 1)*Ny*Nz_spec*Nt) - 1] = real(u[i][ny, nz, nt])
        vec[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec + (i - 1)*Ny*Nz_spec*Nt)]     = imag(u[i][ny, nz, nt])
    end

    return vec
end

function vec2field!(u::VectorField{N, <:SpectralField{Ny, Nz, Nt, <:Any, T}}, vec::Vector{T}) where {N, Ny, Nz, Nt, T}
    # get spectralspanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of the state vector and assign to the field
    for i in 1:N, nt in 1:Nt, nz in 1:Nz_spec, ny in 1:Ny
        u[i][ny, nz, nt] =    vec[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec + (i - 1)*Ny*Nz_spec*Nt) - 1]
                        + 1im*vec[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec + (i - 1)*Ny*Nz_spec*Nt)]
    end

    return u
end