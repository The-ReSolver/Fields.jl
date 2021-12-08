# This file contains the definitions required to transform the spectral scalar
# field to a physical scalar field and vice versa

# The dimensions of the physical and spectral field are related as follows:
#   - Ny_spec = Ny_phys
#   - Nz_spec = (Nz_phys >> 1) + 1
#   - Nt_spec = Nt_phys

# What direction transforms first?
# One direction may be faster?
# profile to figure out which direction to choose!
# Profile is for a bunch of different combinations of Nz and Nt

export FFTPlan!, IFFTPlan!

struct FFTPlan!{Ny, Nz, Nt, PLAN}
    plan::PLAN

    function FFTPlan!(u::PhysicalField{Ny, Nz, Nt};
                        flags::UInt32=FFTW.EXHAUSTIVE,
                        timelimit::Real=FFTW.NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        plan = FFTW.plan_rfft(similar(parent(u)), order;
                                flags = flags, timelimit = timelimit)
        new{Ny, Nz, Nt, typeof(plan)}(plan)
    end

    function FFTPlan!(grid::Grid{S, T};
                        flags::UInt32=FFTW.EXHAUSTIVE,
                        timelimit::Real=FFTW.NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {S, T}
        plan = FFTW.plan_rfft(similar(parent(PhysicalField(grid, T))), order;
                                flags = flags, timelimit = timelimit)
        new{S[1], S[2], S[3], typeof(plan)}(plan)
    end
end

function (f::FFTPlan!{Ny, Nz, Nt})(û::SpectralField{Ny, Nz, Nt},
                                    u::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # perform transform
    FFTW.unsafe_execute!(f.plan, parent(u), parent(û))
    
    # normalise
    parent(û) .*= (1/(Nz*Nt))

    return û
end

function (f::FFTPlan!{Ny, Nz, Nt})(𝐮̂::VectorField{N, S}, 𝐮::VectorField{N, P}) where
            {Ny, Nz, Nt, N, S<:SpectralField{Ny, Nz, Nt}, P<:PhysicalField{Ny, Nz, Nt}}
    for i in 1:N
        f(𝐮̂[i], 𝐮[i])
    end

    return 𝐮̂
end

struct IFFTPlan!{Ny, Nz, Nt, PLAN}
    plan::PLAN

    function IFFTPlan!(û::SpectralField{Ny, Nz, Nt};
                        flags::UInt32=FFTW.EXHAUSTIVE,
                        timelimit::Real=FFTW.NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        plan = FFTW.plan_brfft(similar(parent(û)), Nz, order;
                                flags = flags, timelimit = timelimit)
        new{Ny, Nz, Nt, typeof(plan)}(plan)
    end

    function IFFTPlan!(grid::Grid{S, T};
                        flags::UInt32=FFTW.EXHAUSTIVE,
                        timelimit::Real=FFTW.NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {S, T}
        plan = FFTW.plan_brfft(similar(parent(SpectralField(grid, T))), S[2], order;
                                flags = flags, timelimit = timelimit)
        new{S[1], S[2], S[3], typeof(plan)}(plan)
    end
end

function (f::IFFTPlan!{Ny, Nz, Nt})(u::PhysicalField{Ny, Nz, Nt},
                                    û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # perform transform
    # FIXME: required to make copy because otherwise input spectral field is altered
    spec = copy(parent(û))
    FFTW.unsafe_execute!(f.plan, spec, parent(u))

    return u
end

function (f::IFFTPlan!{Ny, Nz, Nt})(𝐮::VectorField{N, P}, 𝐮̂::VectorField{N, S}) where
            {Ny, Nz, Nt, N, P<:PhysicalField{Ny, Nz, Nt}, S<:SpectralField{Ny, Nz, Nt}}
    for i in 1:N
        f(𝐮[i], 𝐮̂[i])
    end

    return 𝐮
end
