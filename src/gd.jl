# This file contains the methods needed to perform a basic gradient descent
# optimisation of a field.

function gd!(u::VectorField{3, S}, modes::Array{ComplexF64, 4}, mean::Vector{Float64}, Re::Real, Ro::Real; eps::Float64=1e-3, α::Float64=1e-3, maxiter::Int=1000, trace::Vector{Float64}=ones(maxiter)) where {Ny, S<:SpectralField{Ny}}
    # set the mean of the velocity to zero
    u[1][:, 1, 1] .= 0.0
    u[2][:, 1, 1] .= 0.0
    u[3][:, 1, 1] .= 0.0

    # initialise the gradient function
    dR! = ResGrad(get_grid(u), modes, mean, Re, Ro)

    # obtain the projected velocity coefficients
    a = similar(dR!.out)
    project!(a, u, get_grid(u).ws, modes)

    # loop to step in descent direction
    i = 0
    while i < maxiter
        # compute the residual values
        Δa, R = dR!(a)

        # check if converges
        norm(Δa) < eps ? (println("Converged! (R = ", R, ")"); break) : nothing

        # update the velocity
        a .-= α.*Δa

        # update trace
        append!(trace, R)

        # print the current global residual
        println("Global Residual: ", R)

        # update iterator
        i += 1
    end

    # convert final result back to full-space
    # u[1] .= dR!.spec_cache[1]
    # u[2] .= dR!.spec_cache[2]
    # u[3] .= dR!.spec_cache[3]

    # final print statements
    if i == maxiter
        println("Could not converge! (maximum iterations reached)")
    end

    # return u, trace
    return VectorField(dR!.spec_cache[1], dR!.spec_cache[2], dR!.spec_cache[3]), trace, dR!
end
