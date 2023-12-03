# This file contains the function definitions to transform a vector field into
# its corresponding state vector for optimisation using Optim.jl.

function field2vec!(vec::AbstractVector{T}, u::SpectralField{Ny, Nz, Nt, <:Any, T}) where {T, Ny, Nz, Nt}
    # get spectral spanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of the field and assign to the state vector
    for nt in 1:Nt, nz in 1:Nz_spec, ny in 1:Ny
        vec[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec) - 1] = real(u[ny, nz, nt])
        vec[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec)]     = imag(u[ny, nz, nt])
    end

    return vec
end

function field2vec!(vec::AbstractVector{T}, u::VectorField{N, <:SpectralField{Ny, Nz, Nt, <:Any, T}}) where {T, N, Ny, Nz, Nt}
    # get spectral spanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of vector elements assigning to fields to the state vector
    for i in 1:N
        field2vec!(@view(vec[(2*(i - 1)*Ny*Nz_spec*Nt + 1):2*i*Ny*Nz_spec*Nt]), u[i])
    end

    return vec
end

function vec2field!(u::SpectralField{Ny, Nz, Nt, <:Any, T}, vec::AbstractVector{T}) where {Ny, Nz, Nt, T}
    # get spectralspanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of the state vector and assign to the field
    for nt in 1:Nt, nz in 1:Nz_spec, ny in 1:Ny
        u[ny, nz, nt] = Complex{T}(vec[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec) - 1], vec[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec)])
    end

    return u
end

function vec2field!(u::VectorField{N, <:SpectralField{Ny, Nz, Nt, <:Any, T}}, vec::AbstractVector{T}) where {N, Ny, Nz, Nt, T}
    # get spectral spanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of the state vector and assign to the field
    for i in 1:N
        vec2field!(u[i], @view(vec[(2*(i - 1)*Ny*Nz_spec*Nt + 1):2*i*Ny*Nz_spec*Nt]))
    end

    return u
end