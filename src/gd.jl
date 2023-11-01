# This file contains the methods needed to perform a basic gradient descent
# optimisation of a field.

function gd!(u::VectorField{3, SpectralField}, modes::Array{ComplexF64, 4}, Re::Real, Ro::Real; fix_mean::Bool=true, eps::Float64=1e-3, α::Float64=1e-3, maxiter::Int=1000)
    # initialise the gradient function
    dR! = ResGrad(u.grid, modes, u[:, 1, 1], Re, Ro, fix_mean=fix_mean)

    # obtain the projected velocity coefficients
    a = similar(dR!.out)
    project!(a[1], u[1], u.grid.ws, @view(modes[:, 1:Ny, :, :]))
    project!(a[2], u[2], u.grid.ws, @view(modes[:, (Ny + 1):2*Ny, :, :]))
    project!(a[3], u[3], u.grid.ws, @view(modes[:, (2*Ny + 1):3*Ny, :, :]))

    # loop to step in descent direction
    i = 0
    while i < maxiter
        # compute the change in velocity coefficients
        Δa = dR!(a)

        # check if converges
        norm(Δa) < eps ? println("Converged!") : nothing

        # update the velocity
        a[1] .-= α.*Δa[1]
        a[2] .-= α.*Δa[2]
        a[3] .-= α.*Δa[3]

        # print the current global residual
        println("Global Residual: ", gr(dR!))

        # update iterator
        i += 1
    end

    # convert final result back to full-space
    u[1] .= dR!.spec_cache[1]
    u[2] .= dR!.spec_cache[2]
    u[3] .= dR!.spec_cache[3]

    # final print statements
    if i == maxiter
        println("Could not converge (maximum iterations reached)!")
    end

    return u
end
