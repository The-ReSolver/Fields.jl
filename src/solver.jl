# This file contains the definitions required to optimise a velocity field to
# minimise the global residual using the DAE solver and pre-defined objective
# functions.

# NOTE: currently initialise other fields as zeros, might be better for the user to input this?

function optimise(u₀::VectorField{3, S}, Re::Real, Ro::Real; Δτ::Real=1e-3, maxiter::Int=1000, callback=()->(), verbose::Bool=false) where {S<:SpectralField}
    # initialise other state fields
    r₀ = similar(u₀)
    FFT! = FFTPlan!(get_grid(u₀), flags=ESTIMATE)
    # TODO: see if this exactly solves the NSEs
    p₀ = FFT!(SpectralField(get_grid(u₀)), PhysicalField(get_grid(u₀), (y, z, t)->(1/2)*Ro*(y^2)))
    ϕ₀ = SpectralField(get_grid(u₀))
    q1₀ = VectorField(u₀...)
    q2₀ = VectorField(r₀..., p₀, ϕ₀)
    q1 = similar(q1₀)
    q2 = similar(q2₀)

    # initialise objective functions
    F! = Evolution(get_grid(u₀), Re, Ro)
    G! = Constraint(get_grid(u₀), Re, Ro)

    # define the stopping criteria for minimal residual
    # TODO: make sure this calling the right norm method
    minres(q1, q2, i) = sqrt(norm(F!(q1, q2))^2 + norm(G!(q1, q2))^2) < 1e-3

    # optimise the flow
    solvedae!(q1, q2, q1₀, q2₀, F!, G!, Δτ, maxiter*Δτ, stopcrit=minres, verbose=verbose)

    return q
end
