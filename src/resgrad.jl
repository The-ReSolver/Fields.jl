# This file contains the definitions required to compute the rate of change of
# the variational dynamics given a set of modes to perform a Galerkin
# projection.

# TODO: simplify this
struct ResGrad{Ny, Nz, Nt, M, FREEMEAN, INCLUDEPERIOD, NORM, S, D, T, DEALIAS, PADFACTOR, PLAN, IPLAN}
    out::SpectralField{M, Nz, Nt, Grid{S, T, D}, T, true, Array{Complex{T}, 3}}
    modes::Array{ComplexF64, 4}
    ws::Vector{Float64}
    proj_cache::Vector{SpectralField{M, Nz, Nt, Grid{S, T, D}, T, true, Array{Complex{T}, 3}}}
    spec_cache::Vector{VectorField{3, SpectralField{Ny, Nz, Nt, Grid{S, T, D}, T, false, Array{Complex{T}, 3}}}}
    phys_cache::Vector{VectorField{3, PhysicalField{Ny, Nz, Nt, Grid{S, T, D}, T, Array{T, 3}, DEALIAS, PADFACTOR}}}
    fft::FFTPlan!{Ny, Nz, Nt, DEALIAS, PLAN}
    ifft::IFFTPlan!{Ny, Nz, Nt, DEALIAS, IPLAN}
    base::Vector{Float64}
    norm::NORM
    Re_recip::T
    Ro::T

    function ResGrad(grid::Grid{S}, ψs::Array{ComplexF64, 4}, base_prof::Vector{Float64}, Re::Real, Ro::Real; free_mean::Bool=false, dealias::Bool=true, pad_factor::Real=3/2, norm::Union{NormScaling, Nothing}=FarazmandScaling(get_ω(grid), get_β(grid)), include_period::Bool=false) where {S}
        pad_factor > 1 || throw(ArgumentError("Padding factor for dealiasing must be larger than 1!"))
        pad_factor = Float64(pad_factor)

        # initialise output vector field
        out = SpectralField(grid, ψs)

        # create field cache
        proj_cache = [SpectralField(grid, ψs)                           for _ in 1:3]
        spec_cache = [VectorField(grid, fieldType=SpectralField)        for _ in 1:21]
        phys_cache = [VectorField(grid, dealias, pad_factor=pad_factor) for _ in 1:13]

        # create transform plans
        FFT! = FFTPlan!(grid, dealias, pad_factor=pad_factor)
        IFFT! = IFFTPlan!(grid, dealias, pad_factor=pad_factor)

        # convert parameters to compatible type
        Re = convert(eltype(phys_cache[1][1]), Re)
        Ro = convert(eltype(phys_cache[1][1]), Ro)

        new{S...,
            size(ψs, 2),
            free_mean,
            include_period,
            typeof(norm),
            (S[1], S[2], S[3]),
            typeof(grid.Dy[1]),
            eltype(phys_cache[1][1]),
            dealias,
            pad_factor,
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
                                norm,
                                1/Re,
                                Ro)
    end
end

function (f::ResGrad{Ny, Nz, Nt, M, FREEMEAN, INCLUDEPERIOD})(a::SpectralField{M, Nz, Nt}, compute_grad::Bool=true) where {Ny, Nz, Nt, M, FREEMEAN, INCLUDEPERIOD}
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
    s         = f.proj_cache[1]
    s̃         = f.proj_cache[3]

    # convert velocity coefficients to full-space
    expand!(u, a, f.modes)

    # set velocity field mean
    u[1][:, 1, 1] .+= f.base

    # compute all the terms with only velocity
    _update_vel_cache!(f)

    # compute the navier-stokes
    @. ns = dudt + vdudy + wdudz - f.Re_recip*(d2udy2 + d2udz2)
    cross!(ns, [0, 0, f.Ro], u)

    # convert to residual in terms of modal basis
    expand!(r, mul!(s̃, f.norm, project!(s, ns, f.ws, f.modes)), f.modes)

    if compute_grad
        # compute all the terms for the variational evolution
        _update_res_cache!(f)

        # compute the RHS of the evolution equation
        # TODO: try without the factors of half to see if the periodic optimisation works any better
        # TODO: where the hell do the factors of half come from???
        @. dudτ = -vdrdy - wdrdz + rx∇u + ry∇v + rz∇w
        dudτ[1][:, 1, 2:end] .*= 0.5
        dudτ[2][:, 1, 2:end] .*= 0.5
        dudτ[3][:, 1, 2:end] .*= 0.5
        @. dudτ += -drdt - f.Re_recip*(d2rdy2 + d2rdz2)
        cross!(dudτ, [0, 0, -f.Ro], r)
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

    if INCLUDEPERIOD
        return f.out, gr(f), frequencyGradient(f)
    else
        return f.out, gr(f)
    end
end

# TODO: I need to sort out the different scalings used and the implicit effect it has on the results
gr(cache::ResGrad) = ((get_β(cache.spec_cache[1])*get_ω(cache.spec_cache[1]))/(16π^2))*(norm(cache.proj_cache[1], cache.norm)^2)
frequencyGradient(cache::ResGrad) = dot(cache.spec_cache[2], cache.spec_cache[10])/get_ω(cache.spec_cache[1])


function (f::ResGrad{<:Any, <:Any, <:Any, <:Any, <:Any, false})(F, G, a::SpectralField)
    G === nothing ? F = f(a, false)[2] : (F = f(a, true)[2]; G .= f.out)
    return F
end

function (f::ResGrad{<:Any, <:Any, <:Any, <:Any, <:Any, true})(F, G, x::Vector{T}) where {T<:Real}
    a = _vectorToVelocityCoefficients!(f.proj_cache[2], x)
    return f(F, G, a)
end
function (f::ResGrad{<:Any, <:Any, <:Any, <:Any, <:Any, true})(F, G, a::SpectralField)
    G === nothing ? F = f(a, false)[2] : (output = f(a, true)[2:3]; _velocityCoefficientsToVector!(G, f.out); G[end] = output[2])
    return output[1]
end


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
