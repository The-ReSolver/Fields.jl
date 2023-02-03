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
    DNSData{_fetch_param(ini, :Ny, Int), _fetch_param(ini, :Nz, Int), length(snaps)}(loc, ini, snaps)
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
        return sort!(tryparse.(Float64, data.snaps_string))
    else
        return getfield(data, field)
    end
end

function Base.getindex(data::DNSData{Ny, Nz}, t::Real) where {Ny, Nz}
    i = findall(x->x==t, data.snaps)
    length(i) == 0 ? throw(ArgumentError(string("Snapsot not available at time ", t))) : Snapshot(data.loc*data.snaps_string[i[1]]*"/", Ny, Nz)
end

_read_params(loc::String) = read(Inifile(), loc)
_fetch_param(ini::Inifile, param::Symbol, ::Type{T}=Float64) where {T} = tryparse(T, strip(get(ini, "params", string(param)), ';'))
_getparamfield(data::DNSData, field::Symbol) = _fetch_param(getfield(data, :params), field)



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

    # extract velocity field
    U = zeros(Float64, 3, Ny, Nz)
    open(loc*"U") do f; U .= mmap(f, Array{Float64, 3}, (3, Ny, Nz)); end

    Snapshot{Ny, Nz}(loc, t, K, dKdt, U)
end

function Base.getproperty(snap::Snapshot{Ny, Nz}, field::Symbol) where {Ny, Nz}
    if field === :omega
        omega = zeros(Float64, Ny, Nz)
        open(snap.loc*"omega") do f; omega .= mmap(f, Matrix{Float64}, (Ny, Nz)); end
    elseif field === :psi
        psi = zeros(Float64, Ny, Nz)
        open(snap.loc*"psi") do f; psi .= mmap(f, Matrix{Float64}, (Ny, Nz)); end
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


function dns2field!(U::SpectralField, data::DNSData) end
function _dns2physicalfield!(U::PhysicalField, data::DNSData) end
