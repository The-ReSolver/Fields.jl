# This file contains the definitions required to transform the spectral scalar
# field to a physical scalar field and vice versa

# The dimensions of the physical and spectral field are related as follows:
#   - Ny_spec = Ny_phys
#   - Nz_spec = (Nz_phys >> 1) + 1
#   - Nt_spec = Nt_phys

const ESTIMATE = FFTW.ESTIMATE
const EXHAUSTIVE = FFTW.EXHAUSTIVE
const MEASURE = FFTW.MEASURE
const PATIENT = FFTW.PATIENT
const WISDOM_ONLY = FFTW.WISDOM_ONLY
const NO_TIMELIMIT = FFTW.NO_TIMELIMIT

padded_size(Nz, Nt) = (ceil(Int, 3*Nz/2), ceil(Int, 3*Nt/2))

function copy_to_truncated!(truncated, padded)
    Nz, Nt = size(truncated)[2:3]
    Nt_padded = size(padded, 3)
    if Nt > 1
        @views copyto!(truncated[:, :, 1:floor(Int, Nt/2)], padded[:, 1:Nz, 1:floor(Int, Nt/2)])
        @views copyto!(truncated[:, :, floor(Int, Nt/2 + 1):Nt], padded[:, 1:Nz, (floor(Int, Nt/2 + 1) + Nt_padded - Nt):Nt_padded])
    else
        @views copyto!(truncated[:, :, 1], padded[:, 1:Nz, 1])
    end
    return truncated
end
function copy_to_padded!(padded, truncated)
    Nz, Nt = size(truncated)[2:3]
    Nt_padded = size(padded, 3)
    if Nt > 1
        @views copyto!(padded[:, 1:Nz, 1:floor(Int, Nt/2)], truncated[:, :, 1:floor(Int, Nt/2)])
        @views copyto!(padded[:, 1:Nz, (floor(Int, Nt/2 + 1) + Nt_padded - Nt):Nt_padded], truncated[:, :, floor(Int, Nt/2 + 1):Nt])
    else
        @views copyto!(padded[:, 1:Nz, 1], truncated[:, :, 1])
    end
    return padded
end

apply_mask!(padded::Array{T, 3}) where {T} = (padded .= zero(T); return padded)


struct FFTPlan!{Ny, Nz, Nt, DEALIAS, PLAN}
    plan::PLAN
    padded::Array{ComplexF64, 3}

    function FFTPlan!(u::PhysicalField{Ny, Nz, Nt}, dealias::Bool=false;
                        flags::UInt32=EXHAUSTIVE,
                        timelimit::Real=NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        Nz_padded, Nt_padded = padded_size(Nz, Nt)
        padded = dealias ? zeros(ComplexF64, Ny, (Nz_padded >> 1) + 1, Nt_padded) : zeros(ComplexF64, 0, 0, 0)
        plan = FFTW.plan_rfft(similar(parent(u)), order; flags=flags, timelimit=timelimit)
        new{Ny, Nz, Nt, dealias, typeof(plan)}(plan, padded)
    end
end
FFTPlan!(grid::Grid{S, T}, dealias::Bool=false; flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT, order=[2, 3]) where {S, T} = FFTPlan!(PhysicalField(grid, dealias, T), dealias; flags=flags, timelimit=timelimit, order=order)

function (f::FFTPlan!{Ny, Nz, Nt, true})(û::SpectralField{Ny, Nz, Nt}, u::PhysicalField{Ny, Nz, Nt, <:Any, <:Any, <:Any, true}) where {Ny, Nz, Nt}
    FFTW.unsafe_execute!(f.plan, parent(u), f.padded)
    copy_to_truncated!(û, f.padded)
    û .*= (1/prod(size(u)[2:3]))
    return û
end

function (f::FFTPlan!{Ny, Nz, Nt, false})(û::SpectralField{Ny, Nz, Nt}, u::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    FFTW.unsafe_execute!(f.plan, parent(u), parent(û))
    û .*= (1/(Nz*Nt))
    return û
end

function (f::FFTPlan!)(û::VectorField{N, S}, u::VectorField{N, P}) where {N, S<:SpectralField, P<:PhysicalField}
    for i in 1:N
        f(û[i], u[i])
    end
    return û
end


# TODO: add ability to omit caching copy
struct IFFTPlan!{Ny, Nz, Nt, DEALIAS, PLAN}
    plan::PLAN
    padded::Array{ComplexF64, 3}

    function IFFTPlan!(û::SpectralField{Ny, Nz, Nt}, dealias::Bool=false;
                        flags::UInt32=EXHAUSTIVE,
                        timelimit::Real=NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        if dealias
            Nz_padded, Nt_padded = padded_size(Nz, Nt)
            padded = zeros(ComplexF64, Ny, (Nz_padded >> 1) + 1, Nt_padded)
            plan = FFTW.plan_brfft(similar(padded), Nz_padded, order; flags=flags, timelimit=timelimit)
        else
            padded = similar(parent(û))
            plan = FFTW.plan_brfft(similar(parent(û)), Nz, order; flags=flags, timelimit=timelimit)
        end
        new{Ny, Nz, Nt, dealias, typeof(plan)}(plan, padded)
    end
end
IFFTPlan!(grid::Grid{S, T}, dealias::Bool=false; flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT, order=[2, 3]) where {S, T} = IFFTPlan!(SpectralField(grid, T), dealias; flags=flags, timelimit=timelimit, order=order)

function (f::IFFTPlan!{Ny, Nz, Nt, true})(u::PhysicalField{Ny, Nz, Nt, <:Any, <:Any, <:Any, true}, û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    copy_to_padded!(apply_mask!(f.padded), û)
    FFTW.unsafe_execute!(f.plan, f.padded, parent(u))
    return u
end

function (f::IFFTPlan!{Ny, Nz, Nt, false})(u::PhysicalField{Ny, Nz, Nt}, û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    f.padded .= û
    FFTW.unsafe_execute!(f.plan, f.padded, parent(u))
    return u
end

function (f::IFFTPlan!)(u::VectorField{N, P}, û::VectorField{N, S}) where {N, P<:PhysicalField, S<:SpectralField}
    for i in 1:N
        f(u[i], û[i])
    end
    return u
end
