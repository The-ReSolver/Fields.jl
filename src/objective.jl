# This file contains the definitions required to compute the evolution and
# constraints governing the variational evolution of flow field to minimise its
# residual.

# TODO: come up with a way for the arrays to shared
# TODO: what about the boundaries???

# =============================================================================
# DAE evolution
# =============================================================================
struct Evolution{Ny, Nz, Nt, G, T, A, PLAN, IPLAN}
    spec_cache::Vector{SpectralField{Ny, Nz, Nt, G, T, A}}
    phys_cache::Vector{PhysicalField{Ny, Nz, Nt, G, T, A}}
    fft::FFTPlan!{Ny, Nz, Nt, PLAN}
    ifft::IFFTPlan!{Ny, Nz, Nt, IPLAN}
    Re_recip::T
    Ro::T

    function Evolution(grid::Grid{S}, Re::Real, Ro::Real) where {S}
        # convert parameters to floats
        Re = convert(Float64, Re)
        Ro = convert(Float64, Ro)

        # create field caches
        spec_cache = [SpectralField(grid) for _ in 1:52]
        phys_cache = [PhysicalField(grid) for _ in 1:17]

        # create transform plans
        FFT! = FFTPlan!(grid)
        IFFT! = IFFTPlan!(grid)

        new{S...,
            typeof(grid),
            eltype(phys_cache[1]),
            typeof(parent(spec_cache[1])),
            typeof(FFT!.plan),
            typeof(IFFT!.plan)}(spec_cache,
                                phys_cache,
                                FFT!,
                                IFFT!,
                                1/Re,
                                Ro)
    end
end

function (f::Evolution{Ny, Nz, Nt})(out::VectorField{3, S}, q::VectorField{8, S}) where {Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    # unpack input
    u  = q[1]; v  = q[2]; w  = q[3]
    rx = q[4]; ry = q[5]; rz = q[6]
    p  = q[7]; ϕ  = q[8]

    # update fields for current state
    _update_evolution_cache!(f, u, r, ϕ)

    # assign aliases

    # compute output
    @. out[1] = -drxdt - vdrxdy - wdrxdz                            - f.Re_recip*(d2rxdy2 + d2rxdz2) + f.Ro*ry
    @. out[2] = -drydt - vdrydy - wdrydz + rxdudy + rydvdy + rzdwdy - f.Re_recip*(d2rydy2 + d2rydz2) - f.Ro*rx + dϕdy
    @. out[3] = -drzdt - vdrzdy - wdrzdz + rxdudz + rydvdz + rzdwdz - f.Re_recip*(d2rzdy2 + d2rzdz2)           + dϕdz

    return out
end

function _update_evolution_cache!(cache::Evolution{Ny, Nz, Nt}, u::VectorField{3, S}, r::VectorField{3, S}, ϕ::S) where {Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    # assign aliases
    dudy    = cache.spec_cache[1]
    dvdy    = cache.spec_cache[2]
    dwdy    = cache.spec_cache[3]
    dudz    = cache.spec_cache[4]
    dvdz    = cache.spec_cache[5]
    dwdz    = cache.spec_cache[6]
    drxdt   = cache.spec_cache[7]
    drydt   = cache.spec_cache[8]
    drzdt   = cache.spec_cache[9]
    drxdy   = cache.spec_cache[10]
    drydy   = cache.spec_cache[11]
    drzdy   = cache.spec_cache[12]
    drxdz   = cache.spec_cache[13]
    drydz   = cache.spec_cache[14]
    drzdz   = cache.spec_cache[15]
    d2rxdy2 = cache.spec_cache[16]
    d2rydy2 = cache.spec_cache[17]
    d2rzdy2 = cache.spec_cache[18]
    d2rxdz2 = cache.spec_cache[19]
    d2rydz2 = cache.spec_cache[20]
    d2rzdz2 = cache.spec_cache[21]
    vdrxdy  = cache.spec_cache[22]
    wdrxdz  = cache.spec_cache[23]
    vdrydy  = cache.spec_cache[24]
    wdrydz  = cache.spec_cache[25]
    vdrzdy  = cache.spec_cache[26]
    wdrzdz  = cache.spec_cache[27]
    rxdudy  = cache.spec_cache[28]
    rydvdy  = cache.spec_cache[29]
    rzdwdy  = cache.spec_cache[30]
    rxdudz  = cache.spec_cache[31]
    rydvdz  = cache.spec_cache[32]
    rzdwdz  = cache.spec_cache[33]
    dϕdy    = cache.spec_cache[34]
    dϕdz    = cache.spec_cache[35]
    tmp1    = cache.spec_cache[36]
    tmp2    = cache.spec_cache[37]
    tmp3    = cache.spec_cache[38]
    tmp4    = cache.spec_cache[39]
    tmp5    = cache.spec_cache[40]
    tmp6    = cache.spec_cache[41]
    tmp7    = cache.spec_cache[42]
    tmp8    = cache.spec_cache[43]
    tmp9    = cache.spec_cache[44]
    tmp10   = cache.spec_cache[45]
    tmp11   = cache.spec_cache[46]
    tmp12   = cache.spec_cache[47]
    tmp13   = cache.spec_cache[48]
    tmp14   = cache.spec_cache[49]
    tmp15   = cache.spec_cache[50]
    tmp16   = cache.spec_cache[51]
    tmp17   = cache.spec_cache[52]
    v_p     = cache.phys_cache[1]
    w_p     = cache.phys_cache[2]
    dudy_p  = cache.phys_cache[3]
    dvdy_p  = cache.phys_cache[4]
    dwdy_p  = cache.phys_cache[5]
    dudz_p  = cache.phys_cache[6]
    dvdz_p  = cache.phys_cache[7]
    dwdz_p  = cache.phys_cache[8]
    rx_p    = cache.phys_cache[9]
    ry_p    = cache.phys_cache[10]
    rz_p    = cache.phys_cache[11]
    drxdy_p    = cache.phys_cache[12]
    drydy_p    = cache.phys_cache[13]
    drzdy_p    = cache.phys_cache[14]
    drxdz_p    = cache.phys_cache[15]
    drydz_p    = cache.phys_cache[16]
    drzdz_p    = cache.phys_cache[17]
    FFT!    = cache.fft
    IFFT!   = cache.ifft

    # compute derivatives
    @sync begin
        Base.Threads.@spawn ddy!(u[1], dudy)
        Base.Threads.@spawn ddy!(u[2], dvdy)
        Base.Threads.@spawn ddy!(u[3], dwdy)
        Base.Threads.@spawn ddz!(u[1], dudz)
        Base.Threads.@spawn ddz!(u[2], dvdz)
        Base.Threads.@spawn ddz!(u[3], dwdz)
        Base.Threads.@spawn ddt!(r[1], drxdt)
        Base.Threads.@spawn ddt!(r[2], drydt)
        Base.Threads.@spawn ddt!(r[3], drzdt)
        Base.Threads.@spawn ddy!(r[1], drxdy)
        Base.Threads.@spawn ddy!(r[2], drydy)
        Base.Threads.@spawn ddy!(r[3], drzdy)
        Base.Threads.@spawn ddz!(r[1], drxdz)
        Base.Threads.@spawn ddz!(r[2], drydz)
        Base.Threads.@spawn ddz!(r[3], drzdz)
        Base.Threads.@spawn d2dy2!(r[1], d2rxdy2)
        Base.Threads.@spawn d2dy2!(r[2], d2rydy2)
        Base.Threads.@spawn d2dy2!(r[3], d2rzdy2)
        Base.Threads.@spawn d2dz2!(r[1], d2rxdz2)
        Base.Threads.@spawn d2dz2!(r[2], d2rydz2)
        Base.Threads.@spawn d2dz2!(r[3], d2rzdz2)
        Base.Threads.@spawn ddy!(ϕ, dϕdy)
        Base.Threads.@spawn ddy!(ϕ, dϕdz)
    end

    # compute nonlinear components
    @sync begin
        Base.Threads.@spawn IFFT!(v_p, u[2], tmp1)
        Base.Threads.@spawn IFFT!(w_p, u[3], tmp2)
        Base.Threads.@spawn IFFT!(dudy_p, dudy, tmp3)
        Base.Threads.@spawn IFFT!(dvdy_p, dvdy, tmp4)
        Base.Threads.@spawn IFFT!(dwdy_p, dwdy, tmp5)
        Base.Threads.@spawn IFFT!(dudz_p, dudz, tmp6)
        Base.Threads.@spawn IFFT!(dvdz_p, dvdz, tmp7)
        Base.Threads.@spawn IFFT!(dwdz_p, dwdz, tmp8)
        Base.Threads.@spawn IFFT!(rx_p, rx, tmp9)
        Base.Threads.@spawn IFFT!(ry_p, ry, tmp10)
        Base.Threads.@spawn IFFT!(rz_p, rz, tmp11)
        Base.Threads.@spawn IFFT!(drxdy_p, drxdy, tmp12)
        Base.Threads.@spawn IFFT!(drydy_p, drydy, tmp13)
        Base.Threads.@spawn IFFT!(drzdy_p, drzdy, tmp14)
        Base.Threads.@spawn IFFT!(drxdz_p, drxdz, tmp15)
        Base.Threads.@spawn IFFT!(drydz_p, drydz, tmp16)
        Base.Threads.@spawn IFFT!(drzdz_p, drzdz, tmp17)
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

    return cache
end


# =============================================================================
# DAE constraints
# =============================================================================
struct Constraint{Ny, Nz, Nt, G, T, A, PLAN, IPLAN}
    spec_cache::Vector{SpectralField{Ny, Nz, Nt, G, T, A}}
    phys_cache::Vector{PhysicalField{Ny, Nz, Nt, G, T, A}}
    fft::FFTPlan!{Ny, Nz, Nt, PLAN}
    ifft::IFFTPlan!{Ny, Nz, Nt, PLAN}
    Re_recip::T
    Ro::T

    function Constraint(grid::Grid{S}, Re::Real, Ro::Real) where {S}
        # convert parameters to floats
        Re = convert(Float64, Re)
        Ro = convert(Float64, Ro)

        # create field caches
        spec_cache = [SpectralField(grid) for _ in 1:34]
        phys_cache = [PhysicalField(grid) for _ in 1:8]

        # create transform plans
        FFT! = FFTPlan!(grid)
        IFFT! = IFFTPlan!(grid)

        new{S...,
            typeof(grid),
            eltype(phys_cache[1]),
            typeof(parent(spec_cache[1])),
            typeof(FFT!.plan),
            typeof(IFFT!.plan)}(spec_cache,
                                phys_cache,
                                FFT!,
                                IFFT!,
                                1/Re,
                                Ro)
    end
end

function (f::Constraint{Ny, Nz, Nt})(out::VectorField{5, S}, q::VectorField{8, S}) where {Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    # unpack input
    u  = q[1]; v  = q[2];
    rx = q[4]; ry = q[5]; rz = q[6]
    p  = q[7]

    # update fields for current state
    _update_constraint_cache!(f, u, r, p)

    # assign aliases
    dudt   = cache.spec_cache[1]
    dvdt   = cache.spec_cache[2]
    dwdt   = cache.spec_cache[3]
    dvdy   = cache.spec_cache[5]
    dwdz   = cache.spec_cache[9]
    d2udy2 = cache.spec_cache[10]
    d2vdy2 = cache.spec_cache[11]
    d2wdy2 = cache.spec_cache[12]
    d2udz2 = cache.spec_cache[13]
    d2vdz2 = cache.spec_cache[14]
    d2wdz2 = cache.spec_cache[15]
    vdudy  = cache.spec_cache[16]
    wdudz  = cache.spec_cache[17]
    vdvdy  = cache.spec_cache[18]
    wdvdz  = cache.spec_cache[19]
    vdwdy  = cache.spec_cache[20]
    wdwdz  = cache.spec_cache[21]
    dpdy   = cache.spec_cache[22]
    dpdz   = cache.spec_cache[23]
    drydy  = cache.spec_cache[25]
    drzdz  = cache.spec_cache[26]

    # compute output
    # TODO: check if rotation has correct sign
    @. out[1] = dudt  + vdudy + wdudz - f.Re_recip*(d2udy2 + d2udz2) - f.Ro*v - rx
    @. out[2] = dvdt  + vdvdy + wdvdz - f.Re_recip*(d2vdy2 + d2vdz2) + f.Ro*u - ry + dpdy
    @. out[3] = dwdt  + vdwdy + wdwdz - f.Re_recip*(d2wdy2 + d2wdz2)          - rz + dpdz
    @. out[4] = dvdy  + dwdz
    @. out[5] = drydy + drzdz

    return out
end

function _update_constraint_cache!(cache::Constraint{Ny, Nz, Nt}, u::VectorField{3, S}, r::VectorField{3, S}, p::S) where {Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    # assign aliases
    dudt   = cache.spec_cache[1]
    dvdt   = cache.spec_cache[2]
    dwdt   = cache.spec_cache[3]
    dudy   = cache.spec_cache[4]
    dvdy   = cache.spec_cache[5]
    dwdy   = cache.spec_cache[6]
    dudz   = cache.spec_cache[7]
    dvdz   = cache.spec_cache[8]
    dwdz   = cache.spec_cache[9]
    d2udy2 = cache.spec_cache[10]
    d2vdy2 = cache.spec_cache[11]
    d2wdy2 = cache.spec_cache[12]
    d2udz2 = cache.spec_cache[13]
    d2vdz2 = cache.spec_cache[14]
    d2wdz2 = cache.spec_cache[15]
    vdudy  = cache.spec_cache[16]
    wdudz  = cache.spec_cache[17]
    vdvdy  = cache.spec_cache[18]
    wdvdz  = cache.spec_cache[19]
    vdwdy  = cache.spec_cache[20]
    wdwdz  = cache.spec_cache[21]
    dpdy   = cache.spec_cache[22]
    dpdz   = cache.spec_cache[23]
    drydy  = cache.spec_cache[25]
    drzdz  = cache.spec_cache[26]
    tmp1   = cache.spec_cache[27]
    tmp2   = cache.spec_cache[28]
    tmp3   = cache.spec_cache[29]
    tmp4   = cache.spec_cache[30]
    tmp5   = cache.spec_cache[31]
    tmp6   = cache.spec_cache[32]
    tmp7   = cache.spec_cache[33]
    tmp8   = cache.spec_cache[34]
    v_p    = cache.phys_cache[1]
    w_p    = cache.phys_cache[2]
    dudy_p = cache.phys_cache[3]
    dvdy_p = cache.phys_cache[4]
    dwdy_p = cache.phys_cache[5]
    dudz_p = cache.phys_cache[6]
    dvdz_p = cache.phys_cache[7]
    dwdz_p = cache.phys_cache[8]
    FFT! = cache.fft
    IFFT! = cache.ifft

    # compute derivatives
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

    # compute nonlinear components
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

    return cache
end
