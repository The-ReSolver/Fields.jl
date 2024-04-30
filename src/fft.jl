# This file contains the definitions required to transform the spectral scalar
# field to a physical scalar field and vice versa

# The dimensions of the physical and spectral field are related as follows:
#   - Ny_spec = Ny_phys
#   - Nz_spec = (Nz_phys >> 1) + 1
#   - Nt_spec = Nt_phys

# key to proper symmetry is the identity "((Nt - 1) >> 1) + 1"

const ESTIMATE = FFTW.ESTIMATE
const EXHAUSTIVE = FFTW.EXHAUSTIVE
const MEASURE = FFTW.MEASURE
const PATIENT = FFTW.PATIENT
const WISDOM_ONLY = FFTW.WISDOM_ONLY
const NO_TIMELIMIT = FFTW.NO_TIMELIMIT

function padded_size(Nz, Nt)
    Nz_padded = ceil(Int, Nz*(3/2))
    Nt_padded = ceil(Int, Nt*(3/2))
    Nz_padded = (Nz_padded - Nz) % 2 == 0 ? Nz_padded : Nz_padded + 1
    Nt_padded = (Nt_padded - Nt) % 2 == 0 ? Nt_padded : Nt_padded + 1
    return Nz_padded, Nt_padded
end

function copy_to_truncated!(truncated, padded)
    Nz, Nt = size(truncated)[2:3]
    Nt_padded = size(padded, 3)
    if Nt > 1
        if Nt % 2 == 0
            # FIXME: doesn't work for even grid numbers, should figure out why at some point
            @views copyto!(truncated[:, :, 1:((Nt >> 1) + 1)], padded[:, 1:Nz, 1:((Nt >> 1) + 1)])
            @views copyto!(truncated[:, :, ((Nt >> 1) + 2):Nt], padded[:, 1:Nz, ((Nt >> 1) + 2 + Nt_padded - Nt):Nt_padded])
        else
            @views copyto!(truncated[:, :, 1:((Nt >> 1) + 1)], padded[:, 1:Nz, 1:((Nt >> 1) + 1)])
            @views copyto!(truncated[:, :, ((Nt >> 1) + 2):Nt], padded[:, 1:Nz, ((Nt >> 1) + 2 + Nt_padded - Nt):Nt_padded])
        end
    else
        @views copyto!(truncated[:, :, 1], padded[:, 1:Nz, 1])
    end
    return truncated
end
function copy_to_padded!(padded, truncated)
    Nz, Nt = size(truncated)[2:3]
    Nt_padded = size(padded, 3)
    if Nt > 1
        if Nt % 2 == 0
            @views copyto!(padded[:, 1:Nz, 1:(((Nt - 1) >> 1) + 1)], truncated[:, :, 1:(((Nt - 1) >> 1) + 1)])
            @views copyto!(padded[:, 1:Nz, (((Nt - 1) >> 1) + 2 + Nt_padded - Nt):Nt_padded], truncated[:, :, (((Nt - 1) >> 1) + 2):Nt])
        else
            @views copyto!(padded[:, 1:Nz, 1:(((Nt - 1) >> 1) + 1)], truncated[:, :, 1:(((Nt - 1) >> 1) + 1)])
            @views copyto!(padded[:, 1:Nz, (((Nt - 1) >> 1) + 2 + Nt_padded - Nt):Nt_padded], truncated[:, :, (((Nt - 1) >> 1) + 2):Nt])
        end
    else
        @views copyto!(padded[:, 1:Nz, 1], truncated[:, :, 1])
    end
    return padded
end

apply_mask!(padded::Array{T}) where {T} = (padded .= zero(T); return padded)

function apply_symmetry!(u::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    for nt in 2:(((Nt - 1) >> 1) + 1), ny in 1:Ny
        pos = u[ny, 1, nt]
        neg = u[ny, 1, end - nt + 2]
        _re = 0.5*(real(pos) + real(neg))
        _im = 0.5*(imag(pos) - imag(neg))
        u[ny, 1, nt] = _re + 1im*_im
        u[ny, 1, end - nt + 2] = _re - 1im*_im
    end
    return u
end


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

# TODO: should come up with a better way to do this than just applying symmetry manually
function (f::IFFTPlan!{Ny, Nz, Nt, true})(u::PhysicalField{Ny, Nz, Nt, <:Any, <:Any, <:Any, true}, û::SpectralField{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    apply_symmetry!(û)
    copy_to_padded!(apply_mask!(f.padded), û)
    FFTW.unsafe_execute!(f.plan, f.padded, parent(u))
    return u
end

function (f::IFFTPlan!{Ny, Nz, Nt, false})(u::PhysicalField{Ny, Nz, Nt}, û::SpectralField{Ny, Nz, Nt}, safe::Bool=true) where {Ny, Nz, Nt}
    apply_symmetry!(û)
    if safe
        f.padded .= û
        FFTW.unsafe_execute!(f.plan, f.padded, parent(u))
    else
        FFTW.unsafe_execute!(f.plan, parent(û), parent(u))
    end
    return u
end

function (f::IFFTPlan!)(u::VectorField{N, P}, û::VectorField{N, S}) where {N, P<:PhysicalField, S<:SpectralField}
    for i in 1:N
        f(u[i], û[i])
    end
    return u
end


function (f::IFFTPlan!{Ny, Nz, Nt})(û::SpectralField{Ny, Nz, Nt}, padSize::NTuple{2, Int}) where {Ny, Nz, Nt}
    grid_p = interpolate(get_grid(û), padSize)
    ûp = SpectralField(grid_p)
    u = PhysicalField(grid_p)
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1)
        ûp[:, nz, nt] .= û[:, nz, nt]
    end
    for nt in 2:((Nt >> 1) + 1)
        ûp[:, 1, nt] .= û[:, 1, nt]
        ûp[:, 1, end - nt + 2] .= û[:, 1, end - nt + 2]
    end
    ûp[:, 1, 1] .= û[:, 1, 1]
    u.data .= brfft(parent(ûp), padSize[1], [2, 3])
    return u
end

function (f::IFFTPlan!{Ny, Nz, Nt})(û::VectorField{N, S}, padSize::NTuple{2, Int}) where {N, Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}}
    grid_p = interpolate(get_grid(û), padSize)
    ûp = VectorField(grid_p)
    u = VectorField(grid_p, fieldType=PhysicalField)
    for i in 1:N
        for nt in 1:Nt, nz in 2:((Nz >> 1) + 1)
            ûp[i][:, nz, nt] .= û[i][:, nz, nt]
        end
        for nt in 2:((Nt >> 1) + 1)
            ûp[i][:, 1, nt] .= û[i][:, 1, nt]
            ûp[i][:, 1, end - nt + 2] .= û[i][:, 1, end - nt + 2]
        end
        ûp[i][:, 1, 1] .= û[i][:, 1, 1]
        u[i] .= brfft(parent(ûp[i]), padSize[1], [2, 3])
    end
    return u
end
