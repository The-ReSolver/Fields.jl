# This file contains the function definitions required to optimise a flow field
# using gradient free methods.

using Optim

export gf_optimise

function flow2vec!(v::Vector{T}, u::SpectralField{Ny, Nz, Nt, <:Any, T}) where {T, Ny, Nz, Nt}
    # get spectral spanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of the scalar field and assign to state vector
    for nt in 1:Nt, nz in 1:Nz_spec, ny in 1:Ny
        v[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec) - 1] = real(u[ny, nz, nt])
        v[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec)]     = imag(u[ny, nz, nt])
    end

    return v
end

function flow2vec!(v::Vector{T}, u::VectorField{N, <:SpectralField{Ny, Nz, Nt, <:Any, T}}) where {T, N, Ny, Nz, Nt}
    # get spectral spanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of the vector field and assign to state vector
    for i in eachindex(u), nt in 1:Nt, nz in 1:Nz_spec, ny in 1:Ny
        v[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec + (i - 1)*Ny*Nz_spec*Nt) - 1] = real(u[i][ny, nz, nt])
        v[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec + (i - 1)*Ny*Nz_spec*Nt)]     = imag(u[i][ny, nz, nt])
    end

    return v
end

function vec2flow!(u::SpectralField{Ny, Nz, Nt, <:Any, T}, v::Vector{T}) where {Ny, Nz, Nt, T}
    # get spectral spanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of the vector field and assign from state vector
    for nt in 1:Nt, nz in 1:Nz_spec, ny in 1:Ny
        u[ny, nz, nt] =  Complex{T}(v[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec) - 1],
                                    v[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec)])
    end

    return u
end

function vec2flow!(u::VectorField{N, <:SpectralField{Ny, Nz, Nt, <:Any, T}}, v::Vector{T}) where {T, N, Ny, Nz, Nt}
    # get spectral spanwise size of the field
    Nz_spec = (Nz >> 1) + 1

    # loop over the elements of the vector field and assign from state vector
    for i in eachindex(u), nt in 1:Nt, nz in 1:Nz_spec, ny in 1:Ny
        u[i][ny, nz, nt] =   Complex{T}(v[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec + (i - 1)*Ny*Nz_spec*Nt) - 1],
                                        v[2*(ny + (nz - 1)*Ny + (nt - 1)*Ny*Nz_spec + (i - 1)*Ny*Nz_spec*Nt)])
    end

    return u
end

function gf_optimise(u::VectorField{3, <:SpectralField{<:Any, Nz, Nt}}, modes::Array{ComplexF64, 4}, mean::Vector{Float64}, Re::Real, Ro::Real; maxiter::Int=1000, verbose::Bool=false, show_every::Int=1) where {Nz, Nt}
    # initialise coefficient field
    M = size(modes, 2)
    a = SpectralField(Grid(ones(M), Nz, Nt, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), ones(M), get_grid(u).dom[2], get_grid(u).dom[1]))

    # project velocity field onto modes and get vector representation
    project!(a, u, get_grid(u).ws, modes)
    a_vec = flow2vec!(zeros(2*size(modes, 2)*((Nz >> 1) + 1)*Nt), a)

    # initialise the cache object to compute the residual
    dR! = ResGrad(get_grid(u), modes, mean, Re, Ro)

    # define the function to compute the residual from a vector of coefficients
    # FIXME: does some overwriting where there shouldn't be any
    residual(a_vec::Vector) = dR!(vec2flow!(a, a_vec))[2]

    # perform optimisation
    return optimize(residual, a_vec, NelderMead(), Optim.Options(iterations=maxiter, show_trace=verbose, show_every=show_every))
end
