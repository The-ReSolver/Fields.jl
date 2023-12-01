# This file contains the definitions required to optimise a velocity field to
# minimise the global residual using the DAE solver and pre-defined objective
# functions.

# NOTE: currently initialise other fields as zeros, might be better for the user to input this?

export Options

export optimisedae

function optimisedae(u₀::VectorField{3, S}, Re::Real, Ro::Real; Δτ::Real=1e-3, maxiter::Int=1000, callback=()->(), verbose::Bool=false, rootopts::Options=Options(verbose=false)) where {S<:SpectralField}
    # initialise other state fields
    # r₀ = similar(u₀); r₀ .= VectorField([rand(ComplexF64, size(u₀[1])...) for _ in 1:3]...); r₀[1][:, 1, 1] .= real.(r₀[1][:, 1, 1]); r₀[2][:, 1, 1] .= real.(r₀[2][:, 1, 1]); r₀[3][:, 1, 1] .= real.(r₀[3][:, 1, 1])
    # p₀ = SpectralField(get_grid(u₀)); p₀ .= rand(ComplexF64, size(u₀[1])...); p₀[:, 1, 1] .= real.(p₀[:, 1, 1])
    # ϕ₀ = SpectralField(get_grid(u₀)); ϕ₀ .= rand(ComplexF64, size(u₀[1])...); ϕ₀[:, 1, 1] .= real.(ϕ₀[:, 1, 1])
    r₀ = similar(u₀); [r₀[i][:, 1, 1] .= 1e-2 for i in 1:3]
    p₀ = SpectralField(get_grid(u₀)); p₀[:, 2, 1] .= 1e-2
    ϕ₀ = SpectralField(get_grid(u₀)); ϕ₀[:, 2, 1] .= 1e-2
    q1₀ = VectorField(u₀...)
    q2₀ = VectorField(r₀..., p₀, ϕ₀)
    q1 = similar(q1₀)
    q2 = similar(q2₀)

    # initialise objective functions
    F! = Evolution(get_grid(u₀), Re, Ro)
    G! = Constraint(get_grid(u₀), Re, Ro)

    # define the stopping criteria for minimal residual
    minres(q1, q2, i) = sqrt(norm(F!(q1, q2))^2) < 1e-3

    # optimise the flow
    solvedae!(q1, q2, q1₀, q2₀, F!, G!, Δτ, maxiter=maxiter, verbose=verbose, rootopts=rootopts)

    return q1, q2
end
