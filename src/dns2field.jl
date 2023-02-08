# This file contains the utilities required to convert DNS from Davide's solver
# (https://github.com/tb6g16/rpcf) into a spectral field type instance.

# This will allow the outputs of the DNS to be directly loaded into Julia for
# manipulation with the rest of the code.

# TODO: add ability for DNS simulation to simulate a certain amount of time before it begins writing data

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
Base.length(data::DNSData) = length(data.snaps_string)

function Base.getindex(data::DNSData{Ny, Nz}, t::Real) where {Ny, Nz}
    i = findfirst(x->x==t, data.snaps)
    isnothing(i) ? throw(ArgumentError(string("Snapsot not available at time ", t))) : Snapshot(data.loc*data.snaps_string[i[1]]*"/", Ny, Nz)
end

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


dns2field!(U::VectorField{3, S}, u::VectorField{3, P}, FFT!::FFTPlan!{Ny, Nz, Nt}, data::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt, S<:SpectralField{Ny, Nz, Nt}, P<:PhysicalField{Ny, Nz, Nt}} =
        FFT!(U, _dns2field!(u, data))
function dns2field!(data::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt}
    grid = Grid(data.y, Nz, Nt, zeros(Ny, Ny), zeros(Ny, Ny), zeros(Ny), data.ω, data.β)
    u = VectorField(grid; field_type=:physical)
    U = VectorField(grid)
    FFT! = FFTPlan!(u)
    dns2field!(U, u, FFT!, data)
end

function dns2field!(U::VectorField{3, P}, data::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt, P<:PhysicalField{Ny, Nz, Nt}}
    # loop over snaps and assign each velocity component
    for (i, snaps) in enumerate(data)
        U[1][:, :, i] .= snaps[1]
        U[2][:, :, i] .= snaps[2]
        U[3][:, :, i] .= snaps[3]
    end

    return U
end
