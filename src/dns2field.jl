# This file contains the utilities required to convert DNS from Davide's solver
# (https://github.com/tb6g16/rpcf) into a spectral field type instance.

# This will allow the outputs of the DNS to be directly loaded into Julia for
# manipulation with the rest of the code.

# TODO: move this stuff out, only keeping to conversion methods in favour of external package

# -----------------------------------------------------------------------------
# Custom error for indexing at incorrect times
# -----------------------------------------------------------------------------
struct SnapshotTimeError{N, T<:Real} <:Exception; times::Tuple{T, Union{Nothing, T}}; end
SnapshotTimeError(time::Real) = SnapshotTimeError{1, typeof(time)}((time, nothing))
SnapshotTimeError(times::Vararg{Real, 2}) = SnapshotTimeError{2, eltype(times)}(times)
Base.showerror(io::IO, e::SnapshotTimeError{1}) = print(io, "Snapshot does not exist at: ", e.times[1])
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
    snaps = _filterDirectoryToSnapshots(loc)
    DNSData{_fetch_param(ini, :Ny, Int), _fetch_param(ini, :Nz, Int), length(snaps)}(loc, ini, _sort_snaps!(snaps))
end
DNSData(loc) = DNSData(string(loc))

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
Base.getindex(data::DNSData, range::NTuple{2, Real}; skip_step::Int=1) = getindex(data, range..., skip_step=skip_step)
function Base.getindex(data::DNSData{Ny, Nz}, start::Real, stop::Real; skip_step::Int=1) where {Ny, Nz}
    i_start = findfirst(x->x>=start, data.snaps); i_stop = findlast(x->x<=stop, data.snaps)
    isnothing(i_start) || isnothing(i_stop) ? throw(SnapshotTimeError(start, stop)) : DNSData{Ny, Nz, length(data.snaps_string[i_start:skip_step:i_stop])}(data.loc, data.params, data.snaps_string[i_start:skip_step:i_stop])
end
Base.firstindex(data::DNSData) = tryparse(Float64, data.snaps_string[firstindex(data.snaps_string)])
Base.lastindex(data::DNSData) = tryparse(Float64, data.snaps_string[lastindex(data.snaps_string)])

_read_params(loc::String) = read(Inifile(), loc)
_fetch_param(ini::Inifile, param::Symbol, ::Type{T}=Float64) where {T} = tryparse(T, strip(get(ini, "params", string(param)), ';'))
_getparamfield(data::DNSData, field::Symbol) = _fetch_param(getfield(data, :params), field)
_sort_snaps!(snaps::Vector{String}) = sort(snaps, by=x->tryparse(Float64, x))
_filterDirectoryToSnapshots(path) = filter!(x -> x=="params" || x=="K" || x=="t" ? false : true, readdir(path))


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
    open(loc*"U") do f; permutedims!(U, mmap(f, Array{Float64, 3}, (Nz + 1, Ny, 3))[1:end - 1, :, :], (3, 2, 1)); end

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

Base.parent(snap::Snapshot) = snap.U

Base.getindex(snap::Snapshot, i::Int) = @view(snap.U[i, :, :])

Base.iterate(snap::Snapshot) = (snap[1], Val(:V))
Base.iterate(snap::Snapshot, ::Val{:V}) = (snap[2], Val(:W))
Base.iterate(snap::Snapshot, ::Val{:W}) = (snap[3], Val(:done))
Base.iterate(::Snapshot, ::Val{:done}) = nothing


# -----------------------------------------------------------------------------
# Utility methods for the DNS data set that do not require the complete
# associated field to be loaded
# -----------------------------------------------------------------------------

function _mean!(ū::Vector{Float64}, data::DNSData{<:Any, Nz}, snap_times::Vector{Float64}) where {Nz}
    # loop over the time window of the data and compute mean
    for t in snap_times
        ū .+= dropdims(sum(data[t][1], dims=2), dims=2)./Nz
    end
    ū ./= length(snap_times)

    # add back the laminar profile
    ū .+= data.y

    return ū
end

function mean!(ū, data::DNSData; window::NTuple{2, Real}=(firstindex(data), lastindex(data)))
    # find range of snapshots inside the provided window
    snapshot_times = data.snaps
    start_ti = findfirst(x->window[1]<=x, snapshot_times)
    end_ti = findlast(x->window[2]>=x, snapshot_times)

    # overwrite the mean with zeros
    ū .= zero(Float64)

    # compute mean
    return _mean!(ū, data, snapshot_times[start_ti:end_ti])
end

mean(data::DNSData{Ny}; window::NTuple{2, Real}=(firstindex(data), lastindex(data))) where {Ny} = (ū = zeros(Float64, Ny); mean!(ū, data, window=window))


# -----------------------------------------------------------------------------
# Methods to convert simulation directory directly into fields which we know
# how to manipulate
# -----------------------------------------------------------------------------
function dnsToSpectralField(data::DNSData{Ny, Nzd, Ntd}, grid::Grid{Ny, Nzg, Ntg}) where {Ny, Nzd, Ntd, Nzg, Ntg}
    u = VectorField(grid)
    A = [Array{Float64, 3}(undef, Ny, Nzd, Ntd) for _ in 1:3]
    for n in 1:3, (i, snap) in enumerate(data)
        A[n][:, :, i] .= snap[n]
    end
    B = [rfft(A[i], [2, 3])./(Nzd*Ntd) for i in 1:3]
    for n in 1:3, nz in 1:((minimum([Nzd, Nzg]) >> 1) + 1)
        u[n][:, nz, 1] .= B[n][:, nz, 1]
    end
    for n in 1:3, nz in 1:((minimum([Nzd, Nzg]) >> 1) + 1), nt in 2:((minimum([Ntd, Ntg]) >> 1) + 1)
        u[n][:, nz, nt] .= B[n][:, nz, nt]
        u[n][:, nz, end-nt+2] .= B[n][:, nz, end-nt+2]
    end
    return u
end
dnsToSpectralField(data::DNSData{Ny, Nzd, Ntd}, grid::Grid{Ny, Nzg, Ntg}, modes) where {Ny, Nzd, Ntd, Nzg, Ntg} = project!(SpectralField(grid, modes), dnsToSpectralField(data, grid), modes)

correct_mean!(u::VectorField{3, S}, data::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt, S<:SpectralField{<:Grid{Ny, Nz, Nt}}} = (u[1][:, 1, 1] .+= data.y; return u)
function correct_mean!(u::VectorField{3, P}, data::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt, P<:PhysicalField{<:Grid{Ny, Nz, Nt}}}
    for nt in 1:Nt, nz in 1:Nz
        u[1][:, nz, nt] .+= data.y
    end
    
    return u
end
