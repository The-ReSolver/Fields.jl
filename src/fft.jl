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

export FFTPlan!

# TODO: Normalisation
# TODO: Correct recovery size

struct FFTPlan!{Ny, Nz, Nt, PLAN}
    plan::PLAN

    function FFTPlan!(  u::PhysicalField{Ny, Nz, Nt},
                        flags::UInt32=FFTW.EXHAUSTIVE,
                        timelimit::Real=FFTW.NO_TIMELIMIT) where {Ny, Nz, Nt}
        # THE ARRAY U IS SET TO ZEROS AT THIS STEP
        plan = FFTW.plan_rfft(similar(parent(u)), [2, 3]; flags = flags, timelimit = timelimit)
        # plan_t = FFTW.plan_rfft(parent(u), [3]; flags = flags, timelimit = timelimit)
        new{Ny, Nz, Nt, typeof(plan)}(plan)
    end
end

function (f::FFTPlan!{Ny, Nz, Nt})(û::SpectraField{Ny, Nz, Nt}, u::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # which array gets input first
    FFTW.unsafe_execute!(f.plan, parent(u), parent(û))
    # return û
end

function simple_fft(u::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    return SpectraField(FFTW.rfft(u.data, [2, 3]))
end
