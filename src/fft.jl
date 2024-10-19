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

function padded_size(Nz, Nt, factor=3/2)
    Nz_padded = ceil(Int, Nz*factor)
    Nt_padded = ceil(Int, Nt*factor)
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

function apply_symmetry!(u::AbstractArray{ComplexF64, 3})
    S = size(u)
    for nt in 2:(((S[3] - 1) >> 1) + 1), ny in 1:S[1]
        pos = u[ny, 1, nt]
        neg = u[ny, 1, end - nt + 2]
        _re = 0.5*(real(pos) + real(neg))
        _im = 0.5*(imag(pos) - imag(neg))
        u[ny, 1, nt] = _re + 1im*_im
        u[ny, 1, end - nt + 2] = _re - 1im*_im
    end
    return u
end


struct FFTPlan!{G, DEALIAS}
    plan::FFTW.rFFTWPlan{Float64, -1, false, 3, Vector{Int}}
    padded::Array{ComplexF64, 3}

    function FFTPlan!(u::PhysicalField{G}, dealias::Bool=false; pad_factor::Float64=3/2, flags::UInt32=EXHAUSTIVE, timelimit::Real=NO_TIMELIMIT, order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt, G<:Grid{Ny, Nz, Nt}}
        Nz_padded, Nt_padded = padded_size(Nz, Nt, pad_factor)
        padded = dealias ? zeros(ComplexF64, Ny, (Nz_padded >> 1) + 1, Nt_padded) : zeros(ComplexF64, 0, 0, 0)
        plan = FFTW.plan_rfft(similar(parent(u)), order; flags=flags, timelimit=timelimit)
        new{G, dealias}(plan, padded)
    end
end
FFTPlan!(grid::Grid, dealias::Bool=false; pad_factor::Float64=3/2, flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT, order=[2, 3]) = FFTPlan!(PhysicalField(grid, dealias, pad_factor=pad_factor), dealias; pad_factor=pad_factor, flags=flags, timelimit=timelimit, order=order)

function (f::FFTPlan!{<:Grid{Ny, Nz, Nt}, true})(û::SpectralField{<:Grid{Ny, Nz, Nt}}, u::PhysicalField{<:Grid{Ny, Nz, Nt}, true}) where {Ny, Nz, Nt}
    FFTW.unsafe_execute!(f.plan, parent(u), f.padded)
    copy_to_truncated!(û, f.padded)
    û .*= (1/prod(size(u)[2:3]))
    return û
end

function (f::FFTPlan!{<:Grid{Ny, Nz, Nt}, false})(û::SpectralField{<:Grid{Ny, Nz, Nt}}, u::PhysicalField{<:Grid{Ny, Nz, Nt}, false}) where {Ny, Nz, Nt}
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


struct IFFTPlan!{G, DEALIAS}
    plan::FFTW.rFFTWPlan{ComplexF64, 1, false, 3, Vector{Int}}
    padded::Array{ComplexF64, 3}

    function IFFTPlan!(û::SpectralField{G}, dealias::Bool=false; pad_factor::Float64=3/2, flags::UInt32=EXHAUSTIVE, timelimit::Real=NO_TIMELIMIT, order::Vector{Int}=[2, 3]) where {Ny, Nz, Nt, G<:Grid{Ny, Nz, Nt}}
        if dealias
            Nz_padded, Nt_padded = padded_size(Nz, Nt, pad_factor)
            padded = zeros(ComplexF64, Ny, (Nz_padded >> 1) + 1, Nt_padded)
            plan = FFTW.plan_brfft(similar(padded), Nz_padded, order; flags=flags, timelimit=timelimit)
        else
            padded = similar(parent(û))
            plan = FFTW.plan_brfft(similar(parent(û)), Nz, order; flags=flags, timelimit=timelimit)
        end
        new{G, dealias}(plan, padded)
    end
end
IFFTPlan!(grid::Grid, dealias::Bool=false; pad_factor::Float64=3/2, flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT, order=[2, 3]) = IFFTPlan!(SpectralField(grid), dealias; pad_factor=pad_factor, flags=flags, timelimit=timelimit, order=order)

function (f::IFFTPlan!{<:Grid{Ny, Nz, Nt}, true})(u::PhysicalField{<:Grid{Ny, Nz, Nt}, true}, û::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, Nz, Nt}
    apply_symmetry!(û)
    copy_to_padded!(apply_mask!(f.padded), û)
    FFTW.unsafe_execute!(f.plan, f.padded, parent(u))
    return u
end

function (f::IFFTPlan!{<:Grid{Ny, Nz, Nt}, false})(u::PhysicalField{<:Grid{Ny, Nz, Nt}, false}, û::SpectralField{<:Grid{Ny, Nz, Nt}}, safe::Bool=true) where {Ny, Nz, Nt}
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


function ifft(û::VectorField{N, <:SpectralField{<:Grid{Ny, Nz, Nt}}}, Nz_pad::Int, Nt_pad::Int) where {N, Ny, Nz, Nt}
    u = VectorField(interpolate(get_grid(û), Nz_pad, Nt_pad), fieldType=PhysicalField)
    for n in 1:N
        ifft!(u[n], û[n])
    end
    return u
end
ifft(û::SpectralField, Nz_pad::Int, Nt_pad::Int) = ifft!(PhysicalField(interpolate(get_grid(û), Nz_pad, Nt_pad)), û)

function ifft!(u::PhysicalField{<:Grid{Ny, NzP, NtP}}, û::SpectralField{<:Grid{Ny, Nz, Nt}}) where {Ny, NzP, Nz, NtP, Nt}
    û_interp = SpectralField(get_grid(u))
    for nz in 1:((Nz >> 1) + 1)
        û_interp[:, nz, 1] .= û[:, nz, 1]
    end
    for nt in 2:((Nt >> 1) + 1), nz in 1:((Nz >> 1) + 1)
        û_interp[:, nz, nt] .= û[:, nz, nt]
        û_interp[:, nz, end-nt+2] .= û[:, nz, end-nt+2]
    end
    parent(u) .= brfft(parent(û_interp), NzP, [2, 3])
    return u
end
