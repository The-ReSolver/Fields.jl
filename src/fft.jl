# This file contains the definitions required to transform the spectral scalar
# field to a physical scalar field and vice versa

# The dimensions of the physical and spectral field are related as follows:
#   - Ny_spec = Ny_phys
#   - Nz_spec = (Nz_phys >> 1) + 1
#   - Nt_spec = Nt_phys

# TODO: analyse the profiled transforms to find optimal combinations of axes

const ESTIMATE = FFTW.ESTIMATE
const EXHAUSTIVE = FFTW.EXHAUSTIVE
const MEASURE = FFTW.MEASURE
const PATIENT = FFTW.PATIENT
const WISDOM_ONLY = FFTW.WISDOM_ONLY
const NO_TIMELIMIT = FFTW.NO_TIMELIMIT

struct FFTPlan!{Ny, Nz, Nt, PLAN}
    plan::PLAN

    function FFTPlan!(u::PhysicalField{Ny, Nz, Nt};
                        flags::UInt32=EXHAUSTIVE,
                        timelimit::Real=NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        plan = FFTW.plan_rfft(similar(parent(u)), order;
                                flags=flags, timelimit=timelimit)
        new{Ny, Nz, Nt, typeof(plan)}(plan)
    end
end

FFTPlan!(grid::Grid{S, T}; flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT, order=[2, 3]) where {S, T} = FFTPlan!(PhysicalField(grid, T); flags=flags, timelimit=timelimit, order=order)

function (f::FFTPlan!{Ny, Nz, Nt})(û::SpectralField{Ny, Nz, Nt},
                                    u::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # perform transform
    FFTW.unsafe_execute!(f.plan, parent(u), parent(û))

    # normalise
    parent(û) .*= (1/(Nz*Nt))

    return û
end

function (f::FFTPlan!{Ny, Nz, Nt})(û::VectorField{N, S}, u::VectorField{N, P}) where
            {Ny, Nz, Nt, N, S<:SpectralField{Ny, Nz, Nt}, P<:PhysicalField{Ny, Nz, Nt}}
    for i in 1:N
        f(û[i], u[i])
    end

    return û
end

struct IFFTPlan!{Ny, Nz, Nt, PLAN}
    plan::PLAN

    function IFFTPlan!(û::SpectralField{Ny, Nz, Nt};
                        flags::UInt32=EXHAUSTIVE,
                        timelimit::Real=NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        plan = FFTW.plan_brfft(similar(parent(û)), Nz, order;
                                flags=flags, timelimit=timelimit)
        new{Ny, Nz, Nt, typeof(plan)}(plan)
    end
end

IFFTPlan!(grid::Grid{S, T}; flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT, order=[2, 3]) where {S, T} = IFFTPlan!(SpectralField(grid, T); flags=flags, timelimit=timelimit, order=order)

function (f::IFFTPlan!{Ny, Nz, Nt})(u::PhysicalField{Ny, Nz, Nt},
                                    û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # perform transform
    FFTW.unsafe_execute!(f.plan, parent(û), parent(u))

    return u
end

function (f::IFFTPlan!{Ny, Nz, Nt})(u::VectorField{N, P}, û::VectorField{N, S}) where
            {Ny, Nz, Nt, N, P<:PhysicalField{Ny, Nz, Nt}, S<:SpectralField{Ny, Nz, Nt}}
    for i in 1:N
        f(u[i], û[i])
    end
end

function (f::IFFTPlan!{Ny, Nz, Nt})(u::PhysicalField{Ny, Nz, Nt},
                                    û::SpectralField{Ny, Nz, Nt},
                                    û_tmp::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    # copy spectral contents to temporary field
    û_tmp .= û

    # perform transform
    FFTW.unsafe_execute!(f.plan, parent(û_tmp), parent(u))

    return u
end

function (f::IFFTPlan!{Ny, Nz, Nt})(u::VectorField{N, P}, û::VectorField{N, S}, û_tmp::VectorField{N, S}) where
            {Ny, Nz, Nt, N, P<:PhysicalField{Ny, Nz, Nt}, S<:SpectralField{Ny, Nz, Nt}}
    for i in 1:N
        f(u[i], û[i], û_tmp[i])
    end

    return u
end
