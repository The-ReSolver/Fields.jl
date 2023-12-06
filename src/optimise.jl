# This file contains the definitions required to solve the variational problem
# using Optim.jl.

const LBFGS = Optim.LBFGS
const ConjugateGradient = Optim.ConjugateGradient
const GradientDescent = Optim.GradientDescent
const MomentumGradientDescent = Optim.MomentumGradientDescent
const AcceleratedGradientDescent = Optim.AcceleratedGradientDescent

# TODO: allow optional free mean
# TODO: restart method
# TODO: expose linesearch interface so I don't have to import it

function optimise!(a::SpectralField{M, Nz, Nt, <:Any, T}, g::Grid, modes::Array{ComplexF64, 4}, mean::Vector{Float64}, Re, Ro; opts::OptOptions=OptOptions()) where {M, Nz, Nt, T}
    # initialise cache functor
    dR! = ResGrad(g, modes, mean, Re, Ro)

    # set the mean components to zero
    a[:, 1, 1] .= zero(Complex{T})

    # define objective function for optimiser
    function fg!(F, G, x)
        G === nothing ? R = dR!(x, false)[2] : (R = dR!(x, true)[2]; G .= dR!.out)

        return R
    end

    # perform optimisation
    sol = optimize(Optim.only_fg!(fg!), a, opts.alg, _gen_optim_opts(opts))

    # update input
    a .= Optim.minimizer(sol)

    return sol
end

_gen_optim_opts(opts) = Optim.Options(; g_tol=opts.g_tol,
                                        allow_f_increases=opts.allow_f_increases,
                                        iterations=opts.maxiter,
                                        show_trace=opts.show_trace,
                                        extended_trace=opts.extended_trace,
                                        show_every=opts.n_it_print,
                                        callback=opts.callback,
                                        time_limit=opts.time_limit,
                                        store_trace=opts.store_trace)
