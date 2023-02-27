# This file contains the utilities required to convert DNS from Davide's solver
# (https://github.com/tb6g16/rpcf) into a spectral field type instance.

# This will allow the outputs of the DNS to be directly loaded into Julia for
# manipulation with the rest of the code.

# TODO: maybe change DNSData to be a subtype of abstract vector??? (simplet interface)

# -----------------------------------------------------------------------------
# Custom error for indexing at incorrect times
# -----------------------------------------------------------------------------
struct SnapshotTimeError{N, T<:Real} <:Exception; times::Tuple{T, Union{Nothing, T}}; end
SnapshotTimeError(time::Real) = SnapshotTimeError{1, typeof(time)}((time, nothing))
SnapshotTimeError(times::Vararg{Real, 2}) = SnapshotTimeError{2, eltype(times)}(times)
Base.showerror(io::IO, e::SnapshotTimeError{1}) = print(io, "Snapshots do not exist at: ", e.times[1])
Base.showerror(io::IO, e::SnapshotTimeError{2}) = print(io, "Snapshots do not exist between ", e.times[1], " - ", e.times[2])


# -----------------------------------------------------------------------------
# Interface for a set of spatiotemporal DNS data stored in a simulation
# directory.
# -----------------------------------------------------------------------------

# given the lcoation just use getproperty() to get the paramaeter information
struct DNSData{Ny, Nz, Nt}
    loc::String
    params::Inifile
    snaps_string::Vector{String}
end

function DNSData(loc::String)
    ini = _read_params(loc*"params")
    snaps = readdir(loc)[1:end - 1]
    DNSData{_fetch_param(ini, :Ny, Int), _fetch_param(ini, :Nz, Int), length(snaps)}(loc, ini, _sort_snaps!(snaps))
end
loadDNS(loc) = DNSData(string(loc))

function Base.getproperty(data::DNSData{Ny, Nz, Nt}, field::Symbol) where {Ny, Nz, Nt}
    if field === :Ny
        return Ny
    elseif field === :Nz
        return Nz
    elseif field === :Nt
        return Nt
    elseif field ∈ [:Re, :Ro, :L, :dt, :T, :n_it_out, :t_restart, :stretch_factor, :n_threads]
        return _getparamfield(data, field)
    elseif field === :β
        return 2π/_getparamfield(data, :L)
    elseif field === :ω
        return 2π/_getparamfield(data, :T)
    elseif field === :dt_snap
        return _getparamfield(data, :n_it_out)*_getparamfield(data, :dt)
    elseif field === :y
        step = 1/(Ny - 1)
        sf = getproperty(data, :stretch_factor)
        return tanh.(sf*((0:Ny - 1)*step .- 0.5))/tanh(0.5*sf)
    elseif field === :snaps
        return tryparse.(Float64, data.snaps_string)
    else
        return getfield(data, field)
    end
end

# Base.iterate(data::DNSData{Ny, Nz}) where {Ny, Nz} = (Snapshot(data.loc*data.snaps_string[0]*"/", Ny, Nz), 1)
Base.iterate(data::DNSData{Ny, Nz, Nt}, state::Int=1) where {Ny, Nz, Nt} = state > Nt ? nothing : (Snapshot(data.loc*data.snaps_string[state]*"/", Ny, Nz), state + 1)
Base.eltype(::Type{DNSData}) = Snapshot
Base.eltype(::DNSData) = eltype(DNSData)
Base.length(::DNSData{<:Any, <:Any, Nt}) where {Nt} = Nt
Base.size(::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)

function Base.getindex(data::DNSData{Ny, Nz}, t::Real) where {Ny, Nz}
    i = findfirst(x->x==t, data.snaps)
    isnothing(i) ? throw(SnapshotTimeError(t)) : Snapshot(data.loc*data.snaps_string[i]*"/", Ny, Nz)
end
Base.getindex(data::DNSData, ::Nothing) = data
Base.getindex(data::DNSData, range::NTuple{2, Real}) = getindex(data, range...)
function Base.getindex(data::DNSData{Ny, Nz}, start::Real, stop::Real) where {Ny, Nz}
    i_start = findfirst(x->x>=start, data.snaps)
    i_stop = findlast(x->x<=stop, data.snaps)
    isnothing(i_start) || isnothing(i_stop) ? throw(SnapshotTimeError(start, stop)) : DNSData{Ny, Nz, length(data.snaps_string[i_start:i_stop])}(data.loc, data.params, data.snaps_string[i_start:i_stop])
end
Base.firstindex(data::DNSData) = tryparse(Float64, data.snaps_string[firstindex(data.snaps_string)])
Base.lastindex(data::DNSData) = tryparse(Float64, data.snaps_string[lastindex(data.snaps_string)])

_read_params(loc::String) = read(Inifile(), loc)
_fetch_param(ini::Inifile, param::Symbol, ::Type{T}=Float64) where {T} = tryparse(T, strip(get(ini, "params", string(param)), ';'))
_getparamfield(data::DNSData, field::Symbol) = _fetch_param(getfield(data, :params), field)
function _sort_snaps!(snaps::Vector{String})
    # create a sorted list of snaps
    float_snaps = tryparse.(Float64, snaps)
    ordered_snaps = sort(float_snaps)

    # loop over the sorted snaps finding where they were and assigning to permutation vector
    p = zeros(Int, length(ordered_snaps))
    for (i, t) in enumerate(ordered_snaps)
        p[i] = findfirst(x->x==t, float_snaps)
    end

    return permute!(snaps, p)
end


# -----------------------------------------------------------------------------
# Interface for the state of a simulation stored in a single snapshot of a 
# simulation directory.
# -----------------------------------------------------------------------------

struct Snapshot{Ny, Nz}
    loc::String
    t::Float64
    K::Float64
    dKdt::Float64
    U::Array{Float64, 3}
end

function Snapshot(loc::String, Ny::Int, Nz::Int)
    # extract metadata
    (_, t, K, dKdt) = open(loc*"metadata") do f; tryparse.(Float64, lstrip.(x->isnothing(tryparse(Int, string(x))), readlines(f))); end
    t = round.(t; digits=6); K = round.(K; digits=6); dKdt = round.(dKdt; digits=6)

    # extract velocity field
    U = zeros(Float64, 3, Ny, Nz)
    open(loc*"U") do f; U .= mmap(f, Array{Float64, 3}, (3, Ny, Nz)); end

    Snapshot{Ny, Nz}(loc, t, K, dKdt, U)
end

function Base.getproperty(snap::Snapshot{Ny, Nz}, field::Symbol) where {Ny, Nz}
    if field === :omega
        omega = zeros(Float64, Ny, Nz)
        open(snap.loc*"omega") do f; omega .= mmap(f, Matrix{Float64}, (Ny, Nz)); end
        return omega
    elseif field === :psi
        psi = zeros(Float64, Ny, Nz)
        open(snap.loc*"psi") do f; psi .= mmap(f, Matrix{Float64}, (Ny, Nz)); end
        return psi
    else
        getfield(snap, field)
    end
end

Base.getindex(snap::Snapshot, i::Int) = @view(snap.U[i, :, :])

Base.iterate(snap::Snapshot) = (snap[1], Val(:V))
Base.iterate(snap::Snapshot, ::Val{:V}) = (snap[2], Val(:W))
Base.iterate(snap::Snapshot, ::Val{:W}) = (snap[3], Val(:done))
Base.iterate(::Snapshot, ::Val{:done}) = nothing

# -----------------------------------------------------------------------------
# Methods to convert simulation directory directly into fields which we know
# how to manipulate
# -----------------------------------------------------------------------------

dns2field(loc::AbstractString; times::Union{Nothing, NTuple{2, Real}}=nothing) = dns2field(DNSData(loc)[times])
function dns2field(data::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    grid = Grid(data.y, Nz, Nt, zeros(Ny, Ny), zeros(Ny, Ny), zeros(Ny), data.ω, data.β)
    u = VectorField(grid; field_type=:physical)
    U = VectorField(grid)
    FFT! = FFTPlan!(grid)
    return dns2field!(U, u, FFT!, data)
end

dns2field!(U::VectorField{3, S},
            u::VectorField{3, P},
            FFT!::FFTPlan!{Ny, Nz, Nt},
            data::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}, P<:PhysicalField{Ny, Nz, Nt}} = FFT!(U, dns2field!(u, data))

function dns2field!(U::VectorField{3, P}, data::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt, P<:PhysicalField{Ny, Nz, Nt}}
    # loop over snaps and assign each velocity component
    for (i, snaps) in enumerate(data)
        U[1][:, :, i] .= snaps[1]
        U[2][:, :, i] .= snaps[2]
        U[3][:, :, i] .= snaps[3]
    end

    return U
end
