# This file contains the definitions required to compute the rate of change of
# the variational dynamics given a set of modes to perform a Galerkin
# projection.

struct ResGrad{Ny, Nz, Nt, M, FREEMEAN, S, D, T, DEALIAS, PLAN, IPLAN}
    out::SpectralField{M, Nz, Nt, Grid{S, T, D}, T, true, Array{Complex{T}, 3}}
    modes::Array{ComplexF64, 4}
    ws::Vector{Float64}
    proj_cache::Vector{SpectralField{M, Nz, Nt, Grid{S, T, D}, T, true, Array{Complex{T}, 3}}}
    spec_cache::Vector{VectorField{3, SpectralField{Ny, Nz, Nt, Grid{S, T, D}, T, false, Array{Complex{T}, 3}}}}
    phys_cache::Vector{VectorField{3, PhysicalField{Ny, Nz, Nt, Grid{S, T, D}, T, Array{T, 3}, DEALIAS}}}
    fft::FFTPlan!{Ny, Nz, Nt, DEALIAS, PLAN}
    ifft::IFFTPlan!{Ny, Nz, Nt, DEALIAS, IPLAN}
    base::Vector{Float64}
    Re_recip::T
    Ro::T

    function ResGrad(grid::Grid{S}, ψs::Array{ComplexF64, 4}, base_prof::Vector{Float64}, Re::Real, Ro::Real, free_mean::Bool=false, dealias::Bool=true) where {S}
        # initialise output vector field
        out = SpectralField(grid, ψs)

        # create field cache
        proj_cache = [SpectralField(grid, ψs)                    for _ in 1:2]
        spec_cache = [VectorField(grid, fieldType=SpectralField) for _ in 1:23]
        phys_cache = [VectorField(grid, dealias)                 for _ in 1:13]

        # create transform plans
        FFT! = FFTPlan!(grid, dealias)
        IFFT! = IFFTPlan!(grid, dealias)

        # convert parameters to compatible type
        Re = convert(eltype(phys_cache[1][1]), Re)
        Ro = convert(eltype(phys_cache[1][1]), Ro)

        new{S...,
            size(ψs, 2),
            free_mean,
            (S[1], S[2], S[3]),
            typeof(grid.Dy[1]),
            eltype(phys_cache[1][1]),
            dealias,
            typeof(FFT!.plan),
            typeof(IFFT!.plan)}(out,
                                ψs,
                                get_ws(grid),
                                proj_cache,
                                spec_cache,
                                phys_cache,
                                FFT!,
                                IFFT!,
                                base_prof,
                                1/Re,
                                Ro)
    end
end

function (f::ResGrad{Ny, Nz, Nt, M, FREEMEAN})(a::SpectralField{M, Nz, Nt}, compute_grad::Bool=true) where {Ny, Nz, Nt, M, FREEMEAN}
    # assign aliases
    u         = f.spec_cache[1]
    dudt      = f.spec_cache[2]
    d2udy2    = f.spec_cache[5]
    d2udz2    = f.spec_cache[6]
    vdudy     = f.spec_cache[7]
    wdudz     = f.spec_cache[8]
    ns        = f.spec_cache[9]
    r         = f.spec_cache[10]
    drdt      = f.spec_cache[11]
    d2rdy2    = f.spec_cache[14]
    d2rdz2    = f.spec_cache[15]
    vdrdy     = f.spec_cache[16]
    wdrdz     = f.spec_cache[17]
    rx∇u      = f.spec_cache[18]
    ry∇v      = f.spec_cache[19]
    rz∇w      = f.spec_cache[20]
    dudτ      = f.spec_cache[21]
    crossprod = f.spec_cache[22]
    s         = f.proj_cache[1]

    # convert velocity coefficients to full-space
    expand!(u, a, f.modes)

    # set velocity field mean
    u[1][:, 1, 1] .+= f.base

    # compute all the terms with only velocity
    _update_vel_cache!(f)

    # compute the navier-stokes
    cross!(crossprod, [0, 0, 1], u)
    # TODO: just re-use the ns field
    @. ns = dudt + vdudy + wdudz - f.Re_recip*(d2udy2 + d2udz2) + f.Ro*crossprod

    # convert to residual in terms of modal basis
    expand!(r, project!(s, ns, f.ws, f.modes), f.modes)

    if compute_grad
        # compute all the terms for the variational evolution
        _update_res_cache!(f)

        # compute the RHS of the evolution equation
        # TODO: where the hell do the factors of half come from???
        @. dudτ = -vdrdy - wdrdz + rx∇u + ry∇v + rz∇w
        dudτ[1][:, 1, 2:end] .*= 0.5
        dudτ[2][:, 1, 2:end] .*= 0.5
        dudτ[3][:, 1, 2:end] .*= 0.5
        # TODO: try cross product by passing reference rather than copying values
        cross!(crossprod, [0, 0, 1], r)
        @. dudτ += -drdt - f.Re_recip*(d2rdy2 + d2rdz2) - f.Ro*crossprod
        dudτ[1][:, 1, 1] .*= 0.5
        dudτ[2][:, 1, 1] .*= 0.5
        dudτ[3][:, 1, 1] .*= 0.5

        # project to get velocity coefficient evolution
        project!(f.out, dudτ, f.ws, f.modes)

        # take off the mean profile
        if !FREEMEAN
            f.out[:, 1, 1] .= 0
        end
    end

    return f.out, gr(f)
end

function (f::ResGrad)(F, G, a::SpectralField)
    G === nothing ? F = f(a, false)[2] : (F = f(a, true)[2]; G .= f.out)
    return F
end
(f::ResGrad)(x::Vector{T}) where {T<:AbstractFloat} = f(_vectorToVelocityCoefficients!(f.proj_cache[2], x), false)[2]

function _update_vel_cache!(cache::ResGrad) 
    # assign aliases
    u       = cache.spec_cache[1]
    dudt    = cache.spec_cache[2]
    dudy    = cache.spec_cache[3]
    dudz    = cache.spec_cache[4]
    d2udy2  = cache.spec_cache[5]
    d2udz2  = cache.spec_cache[6]
    vdudy   = cache.spec_cache[7]
    wdudz   = cache.spec_cache[8]
    u_p     = cache.phys_cache[1]
    dudy_p  = cache.phys_cache[2]
    dudz_p  = cache.phys_cache[3]
    vdudy_p = cache.phys_cache[4]
    wdudz_p = cache.phys_cache[5]
    FFT!    = cache.fft
    IFFT!   = cache.ifft

    # compute all the derivatives of the field
    @sync begin
        Base.Threads.@spawn ddt!(u, dudt)
        Base.Threads.@spawn ddy!(u, dudy)
        Base.Threads.@spawn ddz!(u, dudz)
        Base.Threads.@spawn d2dy2!(u, d2udy2)
        Base.Threads.@spawn d2dz2!(u, d2udz2)
    end

    # compute the nonlinear terms
    @sync begin
        Base.Threads.@spawn IFFT!(u_p, u)
        Base.Threads.@spawn IFFT!(dudy_p, dudy)
        Base.Threads.@spawn IFFT!(dudz_p, dudz)
    end
    @sync begin
        Base.Threads.@spawn vdudy_p[1] .= u_p[2].*dudy_p[1]
        Base.Threads.@spawn vdudy_p[2] .= u_p[2].*dudy_p[2]
        Base.Threads.@spawn vdudy_p[3] .= u_p[2].*dudy_p[3]
        Base.Threads.@spawn wdudz_p[1] .= u_p[3].*dudz_p[1]
        Base.Threads.@spawn wdudz_p[2] .= u_p[3].*dudz_p[2]
        Base.Threads.@spawn wdudz_p[3] .= u_p[3].*dudz_p[3]
    end
    @sync begin
        Base.Threads.@spawn FFT!(vdudy, vdudy_p)
        Base.Threads.@spawn FFT!(wdudz, wdudz_p)
    end

    return cache
end

function _update_res_cache!(cache::ResGrad) 
    # assign aliases
    r       = cache.spec_cache[10]
    drdt    = cache.spec_cache[11]
    drdy    = cache.spec_cache[12]
    drdz    = cache.spec_cache[13]
    d2rdy2  = cache.spec_cache[14]
    d2rdz2  = cache.spec_cache[15]
    vdrdy   = cache.spec_cache[16]
    wdrdz   = cache.spec_cache[17]
    rx∇u    = cache.spec_cache[18]
    ry∇v    = cache.spec_cache[19]
    rz∇w    = cache.spec_cache[20]
    u_p     = cache.phys_cache[1]
    dudy_p  = cache.phys_cache[2]
    dudz_p  = cache.phys_cache[3]
    r_p     = cache.phys_cache[6]
    drdy_p  = cache.phys_cache[7]
    drdz_p  = cache.phys_cache[8]
    vdrdy_p = cache.phys_cache[9]
    wdrdz_p = cache.phys_cache[10]
    rx∇u_p  = cache.phys_cache[11]
    ry∇v_p  = cache.phys_cache[12]
    rz∇w_p  = cache.phys_cache[13]
    FFT!    = cache.fft
    IFFT!   = cache.ifft

    # compute the derivatives of the residual
    @sync begin
        Base.Threads.@spawn ddt!(r, drdt)
        Base.Threads.@spawn ddy!(r, drdy)
        Base.Threads.@spawn ddz!(r, drdz)
        Base.Threads.@spawn d2dy2!(r, d2rdy2)
        Base.Threads.@spawn d2dz2!(r, d2rdz2)
    end

    # compute the nonlienar terms
    @sync begin
        Base.Threads.@spawn IFFT!(r_p, r)
        Base.Threads.@spawn IFFT!(drdy_p, drdy)
        Base.Threads.@spawn IFFT!(drdz_p, drdz)
    end
    @sync begin
        Base.Threads.@spawn vdrdy_p[1] .= u_p[2].*drdy_p[1]
        Base.Threads.@spawn vdrdy_p[2] .= u_p[2].*drdy_p[2]
        Base.Threads.@spawn vdrdy_p[3] .= u_p[2].*drdy_p[3]
        Base.Threads.@spawn wdrdz_p[1] .= u_p[3].*drdz_p[1]
        Base.Threads.@spawn wdrdz_p[2] .= u_p[3].*drdz_p[2]
        Base.Threads.@spawn wdrdz_p[3] .= u_p[3].*drdz_p[3]
        Base.Threads.@spawn rx∇u_p[2] .= r_p[1].*dudy_p[1]
        Base.Threads.@spawn rx∇u_p[3] .= r_p[1].*dudz_p[1]
        Base.Threads.@spawn ry∇v_p[2] .= r_p[2].*dudy_p[2]
        Base.Threads.@spawn ry∇v_p[3] .= r_p[2].*dudz_p[2]
        Base.Threads.@spawn rz∇w_p[2] .= r_p[3].*dudy_p[3]
        Base.Threads.@spawn rz∇w_p[3] .= r_p[3].*dudz_p[3]
    end
    @sync begin
        Base.Threads.@spawn FFT!(vdrdy, vdrdy_p)
        Base.Threads.@spawn FFT!(wdrdz, wdrdz_p)
        Base.Threads.@spawn FFT!(rx∇u, rx∇u_p)
        Base.Threads.@spawn FFT!(ry∇v, ry∇v_p)
        Base.Threads.@spawn FFT!(rz∇w, rz∇w_p)
    end
end

gr(cache::ResGrad) = ((get_β(cache.spec_cache[1])*get_ω(cache.spec_cache[1]))/(16π^2))*(norm(cache.proj_cache[1])^2)

function optimalFrequency(optimisationCache)
    dudt       = optimisationCache.spec_cache[2]
    d2udy2     = optimisationCache.spec_cache[5]
    d2udz2     = optimisationCache.spec_cache[6]
    vdudy      = optimisationCache.spec_cache[7]
    wdudz      = optimisationCache.spec_cache[8]
    crossprod  = optimisationCache.spec_cache[22]
    nsOperator = optimisationCache.spec_cache[23]
    s          = optimisationCache.proj_cache[1]

    @. nsOperator = -vdudy - wdudz + optimisationCache.Re_recip*(d2udy2 + d2udz2) - optimisationCache.Ro*crossprod

    return get_ω(dudt)*dot(dudt, nsOperator)/(norm(dudt)^2)
end
