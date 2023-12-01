# This file contains the function definitions required to optimise a flow field
# using gradient free methods.

using Optim

export gf_optimise

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
