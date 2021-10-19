# This file contains the definitions required to transform the spectral scalar
# field to a physical scalar field and vice versa

# The dimensions of the physical and spectral field are related as follows:
#   - Ny_spec = Ny_phys
#   - Nz_spec = (Nz_phys >> 1) + 1
#   - Nt_spec = Nt_phys
# NZ AND NT MAY SWAP DEPENDING ON WHICH DIRECTION IS TRANSFORMED FIRST

# What direction transforms first?
# One direction may be faster?
# profile to figure out which direction to choose!
# Profile is for a bunch of different combinations of Nz and Nt

import FFTW
import LinearAlgebra

export FFTPlan!, IFFTPlan!

struct FFTPlan!{Ny, Nz, Nt, PLAN}
    plan::PLAN

    function FFTPlan!(  u::PhysicalField{Ny, Nz, Nt};
                        flags::UInt32=FFTW.EXHAUSTIVE,
                        timelimit::Real=FFTW.NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        plan = FFTW.plan_rfft(  similar(parent(u)), order;
                                flags = flags, timelimit = timelimit)
        new{Ny, Nz, Nt, typeof(plan)}(plan)
    end
end

function (f::FFTPlan!{Ny, Nz, Nt})( û::SpectraField{Ny, Nz, Nt},
                                    u::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # perform transform
    FFTW.unsafe_execute!(f.plan, parent(u), parent(û))
    
    # normalise
    parent(û) .*= (1/(Nz*Nt))

    return û
end

function (f::FFTPlan!{Ny, Nz, Nt})(𝐮̂::VectorField{N, S}, 𝐮::VectorField{N, P}) where
            {Ny, Nz, Nt, N, S<:SpectraField{Ny, Nz, Nt}, P<:PhysicalField{Ny, Nz, Nt}}
    for i in 1:N
        f(𝐮̂[i], 𝐮[i])
    end

    return 𝐮̂
end

struct IFFTPlan!{Ny, Nz, Nt, PLAN}
    plan::PLAN

    function IFFTPlan!( û::SpectraField{Ny, Nz, Nt};
                        flags::UInt32=FFTW.EXHAUSTIVE,
                        timelimit::Real=FFTW.NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        plan = FFTW.plan_brfft( similar(parent(û)), Nz, order;
                                flags = flags, timelimit = timelimit)
        new{Ny, Nz, Nt, typeof(plan)}(plan)
    end
end

function (f::IFFTPlan!{Ny, Nz, Nt})(u::PhysicalField{Ny, Nz, Nt},
                                    û::SpectraField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # perform transform
    FFTW.unsafe_execute!(f.plan, parent(û), parent(u))

    return u
end

function (f::IFFTPlan!{Ny, Nz, Nt})(𝐮::VectorField{N, P}, 𝐮̂::VectorField{N, S}) where
            {Ny, Nz, Nt, N, P<:PhysicalField{Ny, Nz, Nt}, S<:SpectraField{Ny, Nz, Nt}}
    for i in 1:N
        f(𝐮[i], 𝐮̂[i])
    end

    return 𝐮
end
