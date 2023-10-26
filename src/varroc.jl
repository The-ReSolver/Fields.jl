# This file contains the definitions required to compute the rate of change of
# the variational dynamics given a set of modes to perform a Galerkin
# projection.

# TODO: functor for Evolution
# TODO: figure out what is going on at the mean mode with the new fields

struct ResGrad{Ny, Nz, Nt, M, G, T, PLAN, IPLAN}
    out::VectorField{3, SpectralField{Ny, Nz, Nt, G, T, Array{Complex{T}, 3}}}
    modes::Array{ComplexF64, 4}
    ws::Vector{Float64}
    proj_cache::Vector{SpectralField{Ny, Nz, Nt, G, T, Array{Complex{T}, 3}}}
    spec_cache::Vector{SpectralField{Ny, Nz, Nt, G, T, Array{Complex{T}, 3}}}
    phys_cache::Vector{PhysicalField{Ny, Nz, Nt, G, T, Array{T, 3}}}
    fft::FFTPlan!{Ny, Nz, Nt, PLAN}
    ifft::IFFTPlan!{Ny, Nz, Nt, IPLAN}
    Re_recip::T
    Ro::T

    function Evolution(grid::Grid{S}, Ψs::Array{ComplexF64, 4}, Re::Real, Ro::Real) where {S}
        # initialise output vector field
        out = VectorField(grid, N=3)

        # generate grid for projected fields
        proj_grid = Grid(ones(size(modes, 2)), S[2], S[3], grid.Dy, grid.Dy2, grid.ws, grid.ω, grid.β)

        # create field cache
        proj_cache = [SpectralField(proj_grid) for _ in 1:1]
        spec_cache = [SpectralField(grid)      for _ in 1:1]
        phys_cache = [PhysicalField(grid)      for _ in 1:1]

        # create transform plans
        FFT! = FFTPlan!(grid)
        IFFT! = IFFTPlan!(grid)

        # convert parameters to compatible type
        Re = convert(eltype(phys_cache[1]), Re)
        Ro = convert(eltype(phys_cache[1]), Ro)

        new{S...,
            size(modes, 2),
            typeof(grid),
            eltype(phys_cache[1]),
            typeof(FFT!.plan),
            typeof(IFFT!.plan)}(out,
                                Ψs,
                                repeat(grid.ws, 3),
                                proj_cache,
                                spec_cache,
                                phys_cache,
                                FFT!,
                                IFFT!,
                                1/Re,
                                Ro)
    end
end

# TODO: do the modes need to be split into component parts?
function (f::ResGrad{Ny, Nz, Nt, M})(a::VectorField{3, S}) where {Ny, Nz, Nt, M, S<:SpectralField{M, Nz, Nt}}
    # assign aliases
    ws       = f.ws
    u        = VectorField(f.spec_cache[1], f.spec_cache[2], f.spec_cache[3])
    dudτ     = f.spec_cache[4]
    dvdτ     = f.spec_cache[5]
    dwdτ     = f.spec_cache[6]
    dudt     = f.spec_cache[7]
    dvdt     = f.spec_cache[8]
    dwdt     = f.spec_cache[9]
    dudy     = f.spec_cache[10]
    dvdy     = f.spec_cache[11]
    dwdy     = f.spec_cache[12]
    dudz     = f.spec_cache[13]
    dvdz     = f.spec_cache[14]
    dwdz     = f.spec_cache[15]
    d2udy2   = f.spec_cache[16]
    d2vdy2   = f.spec_cache[17]
    d2wdy2   = f.spec_cache[18]
    d2udz2   = f.spec_cache[19]
    d2vdz2   = f.spec_cache[20]
    d2wdz2   = f.spec_cache[21]
    vdudy    = f.spec_cache[22]
    wdudz    = f.spec_cache[23]
    vdvdy    = f.spec_cache[24]
    wdvdz    = f.spec_cache[25]
    vdwdy    = f.spec_cache[26]
    wdwdz    = f.spec_cache[27]
    nsx      = f.spec_cache[28]
    nsy      = f.spec_cache[29]
    nsz      = f.spec_cache[30]
    rx       = f.spec_cache[31]
    ry       = f.spec_cache[32]
    rz       = f.spec_cache[33]
    drxdt    = cache.spec_cache[34]
    drydt    = cache.spec_cache[35]
    drzdt    = cache.spec_cache[36]
    drxdy    = cache.spec_cache[37]
    drydy    = cache.spec_cache[38]
    drzdy    = cache.spec_cache[39]
    drxdz    = cache.spec_cache[40]
    drydz    = cache.spec_cache[41]
    drzdz    = cache.spec_cache[42]
    d2rxdy2  = cache.spec_cache[43]
    d2rydy2  = cache.spec_cache[44]
    d2rzdy2  = cache.spec_cache[45]
    d2rxdz2  = cache.spec_cache[46]
    d2rydz2  = cache.spec_cache[47]
    d2rzdz2  = cache.spec_cache[48]
    vdrxdy   = cache.spec_cache[49]
    wdrxdz   = cache.spec_cache[50]
    vdrydy   = cache.spec_cache[51]
    wdrydz   = cache.spec_cache[52]
    vdrzdy   = cache.spec_cache[53]
    wdrzdz   = cache.spec_cache[54]
    rxdudy   = cache.spec_cache[55]
    rydvdy   = cache.spec_cache[56]
    rzdwdy   = cache.spec_cache[57]
    rxdudz   = cache.spec_cache[58]
    rydvdz   = cache.spec_cache[59]
    rzdwdz   = cache.spec_cache[60]
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
    vdudy_p  = cache.phys_cache[9]
    wdudz_p  = cache.phys_cache[10]
    vdvdy_p  = cache.phys_cache[11]
    wdvdz_p  = cache.phys_cache[12]
    vdwdy_p  = cache.phys_cache[13]
    wdwdz_p  = cache.phys_cache[14]
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
    sx       = proj_cache[1]
    sy       = proj_cache[2]
    sz       = proj_cache[3]
    FFT!     = cache.fft
    IFFT!    = cache.ifft

    # convert velocity coefficients to full-space
    reverse_project!(u[1], a[1], f.modes)
    reverse_project!(u[2], a[2], f.modes)
    reverse_project!(u[3], a[3], f.modes)

    # compute all the derivatives of the field
    @sync begin
        Base.Threads.@spawn ddt!(u[1], dudt)
        Base.Threads.@spawn ddt!(u[2], dvdt)
        Base.Threads.@spawn ddt!(u[3], dwdt)
        Base.Threads.@spawn ddy!(u[1], dudy)
        Base.Threads.@spawn ddy!(u[2], dvdy)
        Base.Threads.@spawn ddy!(u[3], dwdy)
        Base.Threads.@spawn ddz!(u[1], dudz)
        Base.Threads.@spawn ddz!(u[2], dvdz)
        Base.Threads.@spawn ddz!(u[3], dwdz)
        Base.Threads.@spawn d2dy2!(u[1], d2udy2)
        Base.Threads.@spawn d2dy2!(u[2], d2vdy2)
        Base.Threads.@spawn d2dy2!(u[3], d2wdy2)
        Base.Threads.@spawn d2dz2!(u[1], d2udz2)
        Base.Threads.@spawn d2dz2!(u[2], d2vdz2)
        Base.Threads.@spawn d2dz2!(u[3], d2wdz2)
        Base.Threads.@spawn ddy!(p, dpdy)
        Base.Threads.@spawn ddz!(p, dpdz)
        Base.Threads.@spawn ddy!(r[2], drydy)
        Base.Threads.@spawn ddz!(r[3], drzdz)
    end

    # compute the nonlinear terms
    @sync begin
        Base.Threads.@spawn IFFT!(v_p, u[2], tmp1)
        Base.Threads.@spawn IFFT!(w_p, u[3], tmp2)
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

    # compute the navier-stokes
    @. nsx = dudt + vdudy + wdudz - f.Re_recip*(d2udy2 + d2udz2) - f.Ro*u[2]
    @. nsy = dvdt + vdvdy + wdvdz - f.Re_recip*(d2vdy2 + d2vdz2) + f.Ro*u[1]
    @. nsz = dwdt + vdwdy + wdwdz - f.Re_recip*(d2wdy2 + d2wdz2)

    # project navier-stokes to get residual coefficients
    project!(sx, nsx, ws, modes)
    project!(sy, nsy, ws, modes)
    project!(sz, nsz, ws, modes)

    # convert the residual coefficients to full-space
    reverse_project!(rx, sx, f.modes)
    reverse_project!(ry, sy, f.modes)
    reverse_project!(rz, sz, f.modes)

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

    # compute the RHS of the evolution equation
    @. dudτ = drxdt + vdrxdy + wdrxdz                            + f.Re_recip*(d2rxdy2 + d2rxdz2) - f.Ro*ry
    @. dvdτ = drydt + vdrydy + wdrydz - rxdudy - rydvdy - rzdwdy + f.Re_recip*(d2rydy2 + d2rydz2) + f.Ro*rx
    @. dwdτ = drzdt + vdrzdy + wdrzdz - rxdudz - rydvdz - rzdwdz + f.Re_recip*(d2rzdy2 + d2rzdz2)

    # project to get velocity coefficient evolution
    project!(f.out[1], dudτ, ws, modes)
    project!(f.out[2], dvdτ, ws, modes)
    project!(f.out[3], dwdτ, ws, modes)

    return f.out
end
