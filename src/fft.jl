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

# TODO: verify this
padded_size(Nz, Nt) = (Nz + ((Nz + 1) >> 1), Nt + ((Nt + 1) >> 1))

# TODO: make sure allocations are zero
# FIXME: copying doesn't take into account shift
copy_to_truncated!(truncated, padded) = copyto!(truncated, @view(padded[:, size(truncated)[2:3]...]))
copy_to_padded!(padded, truncated) = copyto!(@view(padded[:, size(truncated)[2:3]]), truncated)


struct FFTPlan!{Ny, Nz, Nt, DEALIAS, PLAN}
    plan::PLAN
    padded::Array{ComplexF64, 3}

    function FFTPlan!(u::PhysicalField{Ny, Nz, Nt}, dealias::Bool=false;
                        flags::UInt32=EXHAUSTIVE,
                        timelimit::Real=NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        padded = dealias ? zeros(ComplexF64, Ny, padded_size((Nz >> 1) + 1, Nt)...) : zeros(ComplexF64, 0, 0, 0)
        plan = FFTW.plan_rfft(similar(parent(u)), order; flags=flags, timelimit=timelimit)
        new{Ny, Nz, Nt, dealias, typeof(plan)}(plan, padded)
    end
end
FFTPlan!(grid::Grid{S, T}, dealias::Bool=false; flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT, order=[2, 3]) where {S, T} = FFTPlan!(PhysicalField(grid, dealias, T), dealias; flags=flags, timelimit=timelimit, order=order)

function (f::FFTPlan!{<:Any, <:Any, <:Any, true})(û::SpectralField, u::PhysicalField)
    mul!(f.padded, f.plan, parent(u)) # let FFTW do the size checks instead of prescribing from type parameters
    copy_to_truncated!(û, f.padded)
    û .*= (1/prod(size(u)[2:3]))
    return û
end

function (f::FFTPlan!{Ny, Nz, Nt, false})(û::SpectralField{Ny, Nz, Nt}, u::PhysicalField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    FFTW.unsafe_execute!(f.plan, parent(u), parent(û))
    # TODO: remove unnecessary call to parent
    parent(û) .*= (1/(Nz*Nt))
    return û
end

function (f::FFTPlan!)(û::VectorField{N, S}, u::VectorField{N, P}) where {N, S<:SpectralField, P<:PhysicalField}
    for i in 1:N
        f(û[i], u[i])
    end
    return û
end


# TODO: add ability to omit caching copy
# TODO: remove unnecessary extra cached array
struct IFFTPlan!{Ny, Nz, Nt, DEALIAS, PLAN}
    plan::PLAN
    cache::Array{ComplexF64, 3}
    padded::Array{ComplexF64, 3}

    function IFFTPlan!(û::SpectralField{Ny, Nz, Nt}, dealias::Bool=false;
                        flags::UInt32=EXHAUSTIVE,
                        timelimit::Real=NO_TIMELIMIT,
                        order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt}
        if dealias
            padded = zeros(ComplexF64, Ny, padded_size((Nz >> 1) + 1, Nt)...)
            plan = FFTW.plan_brfft(similar(padded), Nz, order; flags=flags, timelimit=timelimit)
        else
            padded = zeros(ComplexF64, 0, 0, 0)
            plan = FFTW.plan_brfft(similar(parent(û)), Nz, order; flags=flags, timelimit=timelimit)
        end
        new{Ny, Nz, Nt, dealias, typeof(plan)}(plan, similar(parent(û)), padded)
    end
end
IFFTPlan!(grid::Grid{S, T}, dealias::Bool=false; flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT, order=[2, 3]) where {S, T} = IFFTPlan!(SpectralField(grid, T), dealias; flags=flags, timelimit=timelimit, order=order)

function (f::IFFTPlan!{<:Any, <:Any, <:Any, true})(u::PhysicalField, û::SpectralField)
    copy_to_padded!(f.padded, û)
    mul!(parent(u), f.plan, f.padded) # let FFTW do the size checks instead of prescribing from type parameters
    return u
end

function (f::IFFTPlan!{Ny, Nz, Nt, false})(u::PhysicalField{Ny, Nz, Nt}, û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    f.cache .= û
    FFTW.unsafe_execute!(f.plan, f.cache, parent(u))
    return u
end

function (f::IFFTPlan!)(u::VectorField{N, P}, û::VectorField{N, S}) where {N, P<:PhysicalField, S<:SpectralField}
    for i in 1:N
        f(u[i], û[i])
    end
    return u
end
