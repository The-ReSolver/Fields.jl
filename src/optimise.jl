# This file contains the definitions required to solve the variational problem
# using Optim.jl.

const LBFGS = Optim.LBFGS
const ConjugateGradient = Optim.ConjugateGradient
const GradientDescent = Optim.GradientDescent
const MomentumGradientDescent = Optim.MomentumGradientDescent
const AcceleratedGradientDescent = Optim.AcceleratedGradientDescent

function optimise(a::SpectralField{Ny, Nz, Nt, <:Any, T}, g::Grid, modes::Array{ComplexF64, 4}, mean::Vector{Float64}, Re, Ro; opts::OptOptions=OptOptions()) where {Ny, Nz, Nt, T}
    # initialise cache functor
    dR! = ResGrad(g, modes, mean, Re, Ro)

    # initialise state vector for input
    a_vec = Vector{T}(undef, 2*Ny*((Nz >> 1) + 1)*Nt)
    field2vec!(a_vec, a)

    # define objective function for optimiser
    function fg!(F, G, x)
        # convert vector to field
        vec2field!(a, x)

        # compute objective and gradients
        G === nothing ? R = dR!(a, false)[2] : (R = dR!(a, true)[2]; field2vec!(G, dR!.out))

        return R
    end

    # perform optimisation
    sol = optimize(Optim.only_fg!(fg!), a_vec, opts.alg, _gen_optim_opts(opts))

    return _unpack_optim_sol!(sol)
end

_gen_optim_opts(opts) = Optim.Options(; g_tol=opts.g_tol,
                                        allow_f_increases=opts.allow_f_increases,
                                        iterations=opts.maxiter,
                                        show_trace=opts.show_trace,
                                        extended_trace=opts.extended_trace,
                                        show_every=opts.n_it_print,
                                        callback=opts.callback,
                                        time_limit=opts.time_limit)

function _unpack_optim_sol!(sol)
    return sol
end

struct SimOut
    alg::String
    min::Float64
    argmin::SpectralField
end
