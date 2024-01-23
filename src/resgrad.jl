# This file contains the definitions required to compute the rate of change of
# the variational dynamics given a set of modes to perform a Galerkin
# projection.

struct ResGrad{Ny, Nz, Nt, M, FREEMEAN, S, D, T, PLAN, IPLAN}
    out::SpectralField{M, Nz, Nt, Grid{S, T, D}, T, true, Array{Complex{T}, 3}}
    modes::Array{ComplexF64, 4}
    ws::Vector{Float64}
    proj_cache::Vector{SpectralField{M, Nz, Nt, Grid{S, T, D}, T, true, Array{Complex{T}, 3}}}
    spec_cache::Vector{SpectralField{Ny, Nz, Nt, Grid{S, T, D}, T, false, Array{Complex{T}, 3}}}
    phys_cache::Vector{PhysicalField{Ny, Nz, Nt, Grid{S, T, D}, T, Array{T, 3}}}
    fft::FFTPlan!{Ny, Nz, Nt, PLAN}
    ifft::IFFTPlan!{Ny, Nz, Nt, IPLAN}
    base::Vector{Float64}
    Re_recip::T
    Ro::T

    function ResGrad(grid::Grid{S}, ψs::Array{ComplexF64, 4}, base_prof::Vector{Float64}, Re::Real, Ro::Real, free_mean::Bool=false) where {S}
        # initialise output vector field
        out = SpectralField(grid, ψs)

        # create field cache
        proj_cache = [SpectralField(grid, ψs) for _ in 1:1]
        spec_cache = [SpectralField(grid)     for _ in 1:69]
        phys_cache = [PhysicalField(grid)     for _ in 1:35]

        # create transform plans
        FFT! = FFTPlan!(grid)
        IFFT! = IFFTPlan!(grid)

        # convert parameters to compatible type
        Re = convert(eltype(phys_cache[1]), Re)
        Ro = convert(eltype(phys_cache[1]), Ro)

        new{S...,
            size(ψs, 2),
            free_mean,
            (S[1], S[2], S[3]),
            typeof(grid.Dy[1]),
            eltype(phys_cache[1]),
            typeof(FFT!.plan),
            typeof(IFFT!.plan)}(out,
                                ψs,
                                grid.ws,
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
    u        = f.spec_cache[1]
    v        = f.spec_cache[2]
    w        = f.spec_cache[3]
    dudt     = f.spec_cache[4]
    dvdt     = f.spec_cache[5]
    dwdt     = f.spec_cache[6]
    d2udy2   = f.spec_cache[13]
    d2vdy2   = f.spec_cache[14]
    d2wdy2   = f.spec_cache[15]
    d2udz2   = f.spec_cache[16]
    d2vdz2   = f.spec_cache[17]
    d2wdz2   = f.spec_cache[18]
    vdudy    = f.spec_cache[19]
    wdudz    = f.spec_cache[20]
    vdvdy    = f.spec_cache[21]
    wdvdz    = f.spec_cache[22]
    vdwdy    = f.spec_cache[23]
    wdwdz    = f.spec_cache[24]
    nsx      = f.spec_cache[25]
    nsy      = f.spec_cache[26]
    nsz      = f.spec_cache[27]
    rx       = f.spec_cache[28]
    ry       = f.spec_cache[29]
    rz       = f.spec_cache[30]
    drxdt    = f.spec_cache[31]
    drydt    = f.spec_cache[32]
    drzdt    = f.spec_cache[33]
    d2rxdy2  = f.spec_cache[40]
    d2rydy2  = f.spec_cache[41]
    d2rzdy2  = f.spec_cache[42]
    d2rxdz2  = f.spec_cache[43]
    d2rydz2  = f.spec_cache[44]
    d2rzdz2  = f.spec_cache[45]
    vdrxdy   = f.spec_cache[46]
    wdrxdz   = f.spec_cache[47]
    vdrydy   = f.spec_cache[48]
    wdrydz   = f.spec_cache[49]
    vdrzdy   = f.spec_cache[50]
    wdrzdz   = f.spec_cache[51]
    rxdudy   = f.spec_cache[52]
    rydvdy   = f.spec_cache[53]
    rzdwdy   = f.spec_cache[54]
    rxdudz   = f.spec_cache[55]
    rydvdz   = f.spec_cache[56]
    rzdwdz   = f.spec_cache[57]
    dwdτ     = f.spec_cache[58]
    dudτ     = f.spec_cache[59]
    dvdτ     = f.spec_cache[60]
    s        = f.proj_cache[1]
    ws       = f.ws
    ψs       = f.modes

    # convert velocity coefficients to full-space
    expand!([u, v, w], a, ψs)

    # set velocity field mean
    u[:, 1, 1] .+= f.base

    # compute all the terms with only velocity
    _update_vel_cache!(f)

    # compute the navier-stokes
    @. nsx = dudt + vdudy + wdudz - f.Re_recip*(d2udy2 + d2udz2) - f.Ro*v
    @. nsy = dvdt + vdvdy + wdvdz - f.Re_recip*(d2vdy2 + d2vdz2) + f.Ro*u
    @. nsz = dwdt + vdwdy + wdwdz - f.Re_recip*(d2wdy2 + d2wdz2)

    # convert to residual in terms of modal basis
    project!(s, [nsx, nsy, nsz], ws, ψs)
    expand!([rx, ry, rz], s, ψs)

    if compute_grad
        # compute all the terms for the variational evolution
        _update_res_cache!(f)

        # compute the RHS of the evolution equation
        @. dudτ = -drxdt - vdrxdy - wdrxdz                            - f.Re_recip*(d2rxdy2 + d2rxdz2) + f.Ro*ry
        @. dvdτ = -drydt - vdrydy - wdrydz + rxdudy + rydvdy + rzdwdy - f.Re_recip*(d2rydy2 + d2rydz2) - f.Ro*rx
        @. dwdτ = -drzdt - vdrzdy - wdrzdz + rxdudz + rydvdz + rzdwdz - f.Re_recip*(d2rzdy2 + d2rzdz2)

        # project to get velocity coefficient evolution
        project!(f.out, [dudτ, dvdτ, dwdτ], ws, ψs)

        # take off the mean profile
        if !FREEMEAN
            f.out[:, 1, 1] .= 0
        end
    end

    return f.out, gr(f)
end

function _update_vel_cache!(cache::ResGrad) 
    # assign aliases
    u        = cache.spec_cache[1]
    v        = cache.spec_cache[2]
    w        = cache.spec_cache[3]
    dudt     = cache.spec_cache[4]
    dvdt     = cache.spec_cache[5]
    dwdt     = cache.spec_cache[6]
    dudy     = cache.spec_cache[7]
    dvdy     = cache.spec_cache[8]
    dwdy     = cache.spec_cache[9]
    dudz     = cache.spec_cache[10]
    dvdz     = cache.spec_cache[11]
    dwdz     = cache.spec_cache[12]
    d2udy2   = cache.spec_cache[13]
    d2vdy2   = cache.spec_cache[14]
    d2wdy2   = cache.spec_cache[15]
    d2udz2   = cache.spec_cache[16]
    d2vdz2   = cache.spec_cache[17]
    d2wdz2   = cache.spec_cache[18]
    vdudy    = cache.spec_cache[19]
    wdudz    = cache.spec_cache[20]
    vdvdy    = cache.spec_cache[21]
    wdvdz    = cache.spec_cache[22]
    vdwdy    = cache.spec_cache[23]
    wdwdz    = cache.spec_cache[24]
    tmp1     = cache.spec_cache[58]
    tmp2     = cache.spec_cache[59]
    tmp3     = cache.spec_cache[60]
    tmp4     = cache.spec_cache[61]
    tmp5     = cache.spec_cache[62]
    tmp6     = cache.spec_cache[63]
    tmp7     = cache.spec_cache[64]
    tmp8     = cache.spec_cache[65]
    v_p      = cache.phys_cache[1]
    w_p      = cache.phys_cache[2]
    dudy_p   = cache.phys_cache[3]
    dvdy_p   = cache.phys_cache[4]
    dwdy_p   = cache.phys_cache[5]
    dudz_p   = cache.phys_cache[6]
    dvdz_p   = cache.phys_cache[7]
    dwdz_p   = cache.phys_cache[8]
    vdudy_p  = cache.phys_cache[9]
    wdudz_p  = cache.phys_cache[10]
    vdvdy_p  = cache.phys_cache[11]
    wdvdz_p  = cache.phys_cache[12]
    vdwdy_p  = cache.phys_cache[13]
    wdwdz_p  = cache.phys_cache[14]
    FFT!     = cache.fft
    IFFT!    = cache.ifft

    # compute all the derivatives of the field
    @sync begin
        Base.Threads.@spawn ddt!(u, dudt)
        Base.Threads.@spawn ddt!(v, dvdt)
        Base.Threads.@spawn ddt!(w, dwdt)
        Base.Threads.@spawn ddy!(u, dudy)
        Base.Threads.@spawn ddy!(v, dvdy)
        Base.Threads.@spawn ddy!(w, dwdy)
        Base.Threads.@spawn ddz!(u, dudz)
        Base.Threads.@spawn ddz!(v, dvdz)
        Base.Threads.@spawn ddz!(w, dwdz)
        Base.Threads.@spawn d2dy2!(u, d2udy2)
        Base.Threads.@spawn d2dy2!(v, d2vdy2)
        Base.Threads.@spawn d2dy2!(w, d2wdy2)
        Base.Threads.@spawn d2dz2!(u, d2udz2)
        Base.Threads.@spawn d2dz2!(v, d2vdz2)
        Base.Threads.@spawn d2dz2!(w, d2wdz2)
    end

    # compute the nonlinear terms
    @sync begin
        Base.Threads.@spawn IFFT!(v_p, v, tmp1)
        Base.Threads.@spawn IFFT!(w_p, w, tmp2)
        Base.Threads.@spawn IFFT!(dudy_p, dudy, tmp3)
        Base.Threads.@spawn IFFT!(dvdy_p, dvdy, tmp4)
        Base.Threads.@spawn IFFT!(dwdy_p, dwdy, tmp5)
        Base.Threads.@spawn IFFT!(dudz_p, dudz, tmp6)
        Base.Threads.@spawn IFFT!(dvdz_p, dvdz, tmp7)
        Base.Threads.@spawn IFFT!(dwdz_p, dwdz, tmp8)
    end
    @sync begin
        Base.Threads.@spawn vdudy_p .= v_p.*dudy_p
        Base.Threads.@spawn wdudz_p .= w_p.*dudz_p
        Base.Threads.@spawn vdvdy_p .= v_p.*dvdy_p
        Base.Threads.@spawn wdvdz_p .= w_p.*dvdz_p
        Base.Threads.@spawn vdwdy_p .= v_p.*dwdy_p
        Base.Threads.@spawn wdwdz_p .= w_p.*dwdz_p
    end
    @sync begin
        Base.Threads.@spawn FFT!(vdudy, vdudy_p)
        Base.Threads.@spawn FFT!(wdudz, wdudz_p)
        Base.Threads.@spawn FFT!(vdvdy, vdvdy_p)
        Base.Threads.@spawn FFT!(wdvdz, wdvdz_p)
        Base.Threads.@spawn FFT!(vdwdy, vdwdy_p)
        Base.Threads.@spawn FFT!(wdwdz, wdwdz_p)
    end

    return cache
end

function _update_res_cache!(cache::ResGrad) 
    # assign aliases
    rx       = cache.spec_cache[28]
    ry       = cache.spec_cache[29]
    rz       = cache.spec_cache[30]
    drxdt    = cache.spec_cache[31]
    drydt    = cache.spec_cache[32]
    drzdt    = cache.spec_cache[33]
    drxdy    = cache.spec_cache[34]
    drydy    = cache.spec_cache[35]
    drzdy    = cache.spec_cache[36]
    drxdz    = cache.spec_cache[37]
    drydz    = cache.spec_cache[38]
    drzdz    = cache.spec_cache[39]
    d2rxdy2  = cache.spec_cache[40]
    d2rydy2  = cache.spec_cache[41]
    d2rzdy2  = cache.spec_cache[42]
    d2rxdz2  = cache.spec_cache[43]
    d2rydz2  = cache.spec_cache[44]
    d2rzdz2  = cache.spec_cache[45]
    vdrxdy   = cache.spec_cache[46]
    wdrxdz   = cache.spec_cache[47]
    vdrydy   = cache.spec_cache[48]
    wdrydz   = cache.spec_cache[49]
    vdrzdy   = cache.spec_cache[50]
    wdrzdz   = cache.spec_cache[51]
    rxdudy   = cache.spec_cache[52]
    rydvdy   = cache.spec_cache[53]
    rzdwdy   = cache.spec_cache[54]
    rxdudz   = cache.spec_cache[55]
    rydvdz   = cache.spec_cache[56]
    rzdwdz   = cache.spec_cache[57]
    tmp1     = cache.spec_cache[61]
    tmp2     = cache.spec_cache[62]
    tmp3     = cache.spec_cache[63]
    tmp4     = cache.spec_cache[64]
    tmp5     = cache.spec_cache[65]
    tmp6     = cache.spec_cache[66]
    tmp7     = cache.spec_cache[67]
    tmp8     = cache.spec_cache[68]
    tmp9     = cache.spec_cache[69]
    v_p      = cache.phys_cache[1]
    w_p      = cache.phys_cache[2]
    dudy_p   = cache.phys_cache[3]
    dvdy_p   = cache.phys_cache[4]
    dwdy_p   = cache.phys_cache[5]
    dudz_p   = cache.phys_cache[6]
    dvdz_p   = cache.phys_cache[7]
    dwdz_p   = cache.phys_cache[8]
    rx_p     = cache.phys_cache[15]
    ry_p     = cache.phys_cache[16]
    rz_p     = cache.phys_cache[17]
    drxdy_p  = cache.phys_cache[18]
    drydy_p  = cache.phys_cache[19]
    drzdy_p  = cache.phys_cache[20]
    drxdz_p  = cache.phys_cache[21]
    drydz_p  = cache.phys_cache[22]
    drzdz_p  = cache.phys_cache[23]
    vdrxdy_p = cache.phys_cache[24]
    wdrxdz_p = cache.phys_cache[25]
    vdrydy_p = cache.phys_cache[26]
    wdrydz_p = cache.phys_cache[27]
    vdrzdy_p = cache.phys_cache[28]
    wdrzdz_p = cache.phys_cache[29]
    rxdudy_p = cache.phys_cache[30]
    rydvdy_p = cache.phys_cache[31]
    rzdwdy_p = cache.phys_cache[32]
    rxdudz_p = cache.phys_cache[33]
    rydvdz_p = cache.phys_cache[34]
    rzdwdz_p = cache.phys_cache[35]
    FFT!     = cache.fft
    IFFT!    = cache.ifft

    # compute the derivatives of the residual
    @sync begin
        Base.Threads.@spawn ddt!(rx, drxdt)
        Base.Threads.@spawn ddt!(ry, drydt)
        Base.Threads.@spawn ddt!(rz, drzdt)
        Base.Threads.@spawn ddy!(rx, drxdy)
        Base.Threads.@spawn ddy!(ry, drydy)
        Base.Threads.@spawn ddy!(rz, drzdy)
        Base.Threads.@spawn ddz!(rx, drxdz)
        Base.Threads.@spawn ddz!(ry, drydz)
        Base.Threads.@spawn ddz!(rz, drzdz)
        Base.Threads.@spawn d2dy2!(rx, d2rxdy2)
        Base.Threads.@spawn d2dy2!(ry, d2rydy2)
        Base.Threads.@spawn d2dy2!(rz, d2rzdy2)
        Base.Threads.@spawn d2dz2!(rx, d2rxdz2)
        Base.Threads.@spawn d2dz2!(ry, d2rydz2)
        Base.Threads.@spawn d2dz2!(rz, d2rzdz2)
    end

    # compute the nonlienar terms
    @sync begin
        Base.Threads.@spawn IFFT!(rx_p, rx, tmp1)
        Base.Threads.@spawn IFFT!(ry_p, ry, tmp2)
        Base.Threads.@spawn IFFT!(rz_p, rz, tmp3)
        Base.Threads.@spawn IFFT!(drxdy_p, drxdy, tmp4)
        Base.Threads.@spawn IFFT!(drydy_p, drydy, tmp5)
        Base.Threads.@spawn IFFT!(drzdy_p, drzdy, tmp6)
        Base.Threads.@spawn IFFT!(drxdz_p, drxdz, tmp7)
        Base.Threads.@spawn IFFT!(drydz_p, drydz, tmp8)
        Base.Threads.@spawn IFFT!(drzdz_p, drzdz, tmp9)
    end
    @sync begin
        Base.Threads.@spawn vdrxdy_p .= v_p.*drxdy_p
        Base.Threads.@spawn wdrxdz_p .= w_p.*drxdz_p
        Base.Threads.@spawn vdrydy_p .= v_p.*drydy_p
        Base.Threads.@spawn wdrydz_p .= w_p.*drydz_p
        Base.Threads.@spawn vdrzdy_p .= v_p.*drzdy_p
        Base.Threads.@spawn wdrzdz_p .= w_p.*drzdz_p
        Base.Threads.@spawn rxdudy_p .= rx_p.*dudy_p
        Base.Threads.@spawn rydvdy_p .= ry_p.*dvdy_p
        Base.Threads.@spawn rzdwdy_p .= rz_p.*dwdy_p
        Base.Threads.@spawn rxdudz_p .= rx_p.*dudz_p
        Base.Threads.@spawn rydvdz_p .= ry_p.*dvdz_p
        Base.Threads.@spawn rzdwdz_p .= rz_p.*dwdz_p
    end
    @sync begin
        Base.Threads.@spawn FFT!(vdrxdy, vdrxdy_p)
        Base.Threads.@spawn FFT!(wdrxdz, wdrxdz_p)
        Base.Threads.@spawn FFT!(vdrydy, vdrydy_p)
        Base.Threads.@spawn FFT!(wdrydz, wdrydz_p)
        Base.Threads.@spawn FFT!(vdrzdy, vdrzdy_p)
        Base.Threads.@spawn FFT!(wdrzdz, wdrzdz_p)
        Base.Threads.@spawn FFT!(rxdudy, rxdudy_p)
        Base.Threads.@spawn FFT!(rydvdy, rydvdy_p)
        Base.Threads.@spawn FFT!(rzdwdy, rzdwdy_p)
        Base.Threads.@spawn FFT!(rxdudz, rxdudz_p)
        Base.Threads.@spawn FFT!(rydvdz, rydvdz_p)
        Base.Threads.@spawn FFT!(rzdwdz, rzdwdz_p)
    end
end

gr(cache::ResGrad) = ((get_β(cache.spec_cache[1])*get_ω(cache.spec_cache[1]))/(16π^2))*(norm(cache.proj_cache[1])^2)
