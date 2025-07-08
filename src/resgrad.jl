# This file contains the definitions required to compute the rate of change of
# the variational dynamics given a set of modes to perform a Galerkin
# projection.

# -----------------------------------------------------------------------------
# Core residual functions
# -----------------------------------------------------------------------------
struct ResGrad{G, M, FREEMEAN, INCLUDEPERIOD, MULTITHREADED, GRADFACTORS, NORM, D, P, T}
    modes::T
    proj_cache::Vector{SpectralField{G, true}}
    spec_cache::Vector{VectorField{3, SpectralField{G, false}}}
    phys_cache::Vector{VectorField{3, PhysicalField{G, D, P}}}
    fft::FFTPlan!{G, D}
    ifft::IFFTPlan!{G, D}
    base::Vector{Float64}
    norm::NORM
    Re_recip::Float64
    Ro::Float64
    T_offset::Int
    compute_T::Vector{Bool}
    T_relaxation::Float64

    function ResGrad(grid::Grid{Ny, Nz, Nt}, modes, base_prof::Vector{Float64}, Re::Real, Ro::Real;
                    free_mean::Bool=false, dealias::Bool=true, pad_factor::Real=3/2, norm::Union{NormScaling, Nothing}=FarazmandScaling(get_ω(grid), get_β(grid)), include_period::Bool=false, grad_factors::Bool=false, T_offset=0, T_relaxation=1) where {Ny, Nz, Nt}
        pad_factor > 1 || throw(ArgumentError("Padding factor for dealiasing must be larger than 1!"))
        pad_factor = Float64(pad_factor)

        # check if multiple threads are available
        multithreaded = Base.Threads.nthreads() > 1

        # create field cache
        proj_cache = [SpectralField(grid, modes)                        for _ in 1:4]
        spec_cache = [VectorField(grid, fieldType=SpectralField)        for _ in 1:21]
        phys_cache = [VectorField(grid, dealias, pad_factor=pad_factor) for _ in 1:13]

        # create transform plans
        FFT! = FFTPlan!(grid, dealias, pad_factor=pad_factor)
        IFFT! = IFFTPlan!(grid, dealias, pad_factor=pad_factor)

        new{typeof(grid), size(modes, 2), free_mean, include_period, multithreaded, grad_factors, typeof(norm), dealias, pad_factor, typeof(modes)}(modes, proj_cache, spec_cache, phys_cache, FFT!, IFFT!, base_prof, norm, 1/Float64(Re), Float64(Ro), T_offset, T_offset > 0 ? [false] : [true], T_relaxation)
    end
end

function (f::ResGrad{<:Grid{Ny, Nz, Nt}, M, FREEMEAN, INCLUDEPERIOD, MULTITHREADED, GRADFACTORS})(a::SpectralField{<:Grid{Ny, Nz, Nt}, true}) where {Ny, Nz, Nt, M, FREEMEAN, INCLUDEPERIOD, MULTITHREADED, GRADFACTORS}
    # assign aliases
    u         = f.spec_cache[1]
    dudt      = f.spec_cache[2]
    d2udy2    = f.spec_cache[5]
    d2udz2    = f.spec_cache[6]
    vdudy     = f.spec_cache[7]
    wdudz     = f.spec_cache[8]
    ns        = f.spec_cache[9]
    r         = f.spec_cache[10]
    s         = f.proj_cache[1]
    s̃         = f.proj_cache[2]

    # convert velocity coefficients to full-space
    expand!(u, a, f.modes)

    # set velocity field mean
    @view(u[1][:, 1, 1]) .+= f.base

    # compute all the terms with only velocity
    _update_vel_cache!(f, MULTITHREADED)

    # compute the navier-stokes
    @. ns = dudt + vdudy + wdudz - f.Re_recip*(d2udy2 + d2udz2)
    cross_k!(ns, u, f.Ro)

    # convert to residual in terms of modal basis
    expand!(r, mul!(s̃, f.norm, project!(s, ns, f.modes)), f.modes)

    if INCLUDEPERIOD
        return gr(s, f.norm), f.compute_T[1] ? f.T_relaxation*frequencyGradient(dudt, r) : 0.0
    else
        return gr(s, f.norm)
    end
end

function (f::ResGrad{<:Grid{Ny, Nz, Nt}, M, FREEMEAN, INCLUDEPERIOD, MULTITHREADED, GRADFACTORS})(dR::S, a::S) where {Ny, Nz, Nt, M, FREEMEAN, INCLUDEPERIOD, GRADFACTORS, MULTITHREADED, S<:SpectralField{<:Grid{Ny, Nz, Nt}, true}}
    # assign aliases
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

    # compute residual and frequency gradient
    output = f(a)

    # compute all the terms for the variational evolution
    _update_res_cache!(f, MULTITHREADED)

    # compute the RHS of the evolution equation
    @. dudτ = -drdt - vdrdy - wdrdz + rx∇u + ry∇v + rz∇w - f.Re_recip*(d2rdy2 + d2rdz2)
    cross_k!(dudτ, r, -f.Ro)
    dudτ[1][1:end, 1, 1] .*= 0.5
    dudτ[2][1:end, 1, 1] .*= 0.5
    dudτ[3][1:end, 1, 1] .*= 0.5

    # project to get velocity coefficient evolution
    project!(dR, dudτ, f.modes)

    # take off the mean profile
    if !FREEMEAN
        dR[:, 1, 1] .= 0
    end

    return output
end

gr(s, norm_scale) = norm(s, norm_scale)^2/2
frequencyGradient(dudt, r) = dot(dudt, r)/get_ω(r)


# -----------------------------------------------------------------------------
# Interface functions for OptimWrapper.jl
# -----------------------------------------------------------------------------
function (fg::ResGrad{G, M, FREEMEAN, false})(R, dRda, a::SpectralField) where {G, M, FREEMEAN}
    if dRda === nothing
        return fg(a)
    else
        return fg(dRda, a)
    end
end

function (fg::ResGrad{GRID, M, FREEMEAN, true})(R, G, x::Vector) where {GRID, M, FREEMEAN}
    dRda = fg.proj_cache[4]
    a = vectorToField!(fg.proj_cache[3], x)
    if G === nothing
        R = fg(a)[1]
    else
        R, dRdω = fg(dRda, a)
        fieldToVector!(G, dRda, dRdω)
    end
    return R
end

(fg::ResGrad)(x::Vector) = fg(vectorToField!(fg.proj_cache[3], x))
function (fg::ResGrad{G, M, FREEMEAN, false})(grad::V, x::V) where {G, M, FREEMEAN, V<:Vector}
    dRda = fg.proj_cache[4]
    R = fg(dRda, vectorToField!(fg.proj_cache[3], x))
    fieldToVector!(grad, dRda, 0)
    return R
end
function (fg::ResGrad{G, M, FREEMEAN, true})(grad::V, x::V) where {G, M, FREEMEAN, V<:Vector}
    dRda = fg.proj_cache[4]
    R, dRdω = fg(dRda, vectorToField!(fg.proj_cache[3], x))
    fieldToVector!(grad, dRda, dRdω)
    return R
end

_update_vel_cache!(cache::ResGrad, multithreaded::Bool) = multithreaded ? _update_vel_cache_mt!(cache) : _update_vel_cache_st!(cache)
_update_res_cache!(cache::ResGrad, multithreaded::Bool) = multithreaded ? _update_res_cache_mt!(cache) : _update_res_cache_st!(cache)

function _update_vel_cache_st!(cache::ResGrad)
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
    ddt!(u, dudt)
    ddy!(u, dudy)
    ddz!(u, dudz)
    d2dy2!(u, d2udy2)
    d2dz2!(u, d2udz2)

    # compute the nonlinear terms
    IFFT!(u_p, u)
    IFFT!(dudy_p, dudy)
    IFFT!(dudz_p, dudz)
    vdudy_p[1] .= u_p[2].*dudy_p[1]
    vdudy_p[2] .= u_p[2].*dudy_p[2]
    vdudy_p[3] .= u_p[2].*dudy_p[3]
    wdudz_p[1] .= u_p[3].*dudz_p[1]
    wdudz_p[2] .= u_p[3].*dudz_p[2]
    wdudz_p[3] .= u_p[3].*dudz_p[3]
    FFT!(vdudy, vdudy_p)
    FFT!(wdudz, wdudz_p)

    return cache
end

function _update_vel_cache_mt!(cache::ResGrad)
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
    # NOTE: for some reason including these set of IFFTs messes up the computation for multiple threads
    IFFT!(u_p, u)
    IFFT!(dudy_p, dudy)
    IFFT!(dudz_p, dudz)
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

function _update_res_cache_st!(cache::ResGrad)
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
    ddt!(r, drdt)
    ddy!(r, drdy)
    ddz!(r, drdz)
    d2dy2!(r, d2rdy2)
    d2dz2!(r, d2rdz2)

    # compute the nonlienar terms
    IFFT!(r_p, r)
    IFFT!(drdy_p, drdy)
    IFFT!(drdz_p, drdz)
    vdrdy_p[1] .= u_p[2].*drdy_p[1]
    vdrdy_p[2] .= u_p[2].*drdy_p[2]
    vdrdy_p[3] .= u_p[2].*drdy_p[3]
    wdrdz_p[1] .= u_p[3].*drdz_p[1]
    wdrdz_p[2] .= u_p[3].*drdz_p[2]
    wdrdz_p[3] .= u_p[3].*drdz_p[3]
    rx∇u_p[2] .= r_p[1].*dudy_p[1]
    rx∇u_p[3] .= r_p[1].*dudz_p[1]
    ry∇v_p[2] .= r_p[2].*dudy_p[2]
    ry∇v_p[3] .= r_p[2].*dudz_p[2]
    rz∇w_p[2] .= r_p[3].*dudy_p[3]
    rz∇w_p[3] .= r_p[3].*dudz_p[3]
    FFT!(vdrdy, vdrdy_p)
    FFT!(wdrdz, wdrdz_p)
    FFT!(rx∇u, rx∇u_p)
    FFT!(ry∇v, ry∇v_p)
    FFT!(rz∇w, rz∇w_p)

    return cache
end

function _update_res_cache_mt!(cache::ResGrad)
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

    return cache
end
