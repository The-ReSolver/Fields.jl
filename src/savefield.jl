# Definition to read and write a field from disk.

function writeSpectralField(path, u::SpectralField{<:Any, <:Any, <:Any, <:Any, <:Any, PROJECTED}) where {PROJECTED}
    writeGrid(path, get_grid(u), PROJECTED)
    writeField(path, parent(u))
    return nothing
end

function readSpectralField(path, ::Type{T}=Float64) where {T}
    grid, PROJECTED = readGrid(path)
    return SpectralField{PROJECTED}(readField(path, T), grid)
end


function writeVectorField(path, u::VectorField{N, S}) where {N, PROJECTED, S<:SpectralField{<:Any, <:Any, <:Any, <:Any, <:Any, PROJECTED}}
    writeGrid(path, get_grid(u), PROJECTED)
    for i in 1:N
        writeField(path*string(i), parent(u[i]))
    end
    return nothing
end

function readVectorField(path)
    grid, PROJECTED = readGrid(path)
    return VectorField([SpectralField{PROJECTED}(readField(path*string(i)), grid) for i in 1:3]...)
end


function writeGrid(path, grid::Grid{S}, PROJECTED) where {S}
    open(path*"grid", "w") do f
        write(f, S[1])
        write(f, S[2])
        write(f, S[3])
        write(f, grid.y)
        write(f, get_Dy(grid))
        write(f, get_Dy2(grid))
        write(f, get_ws(grid))
        write(f, get_ω(grid))
        write(f, get_β(grid))
        write(f, PROJECTED)
    end
    return nothing
end

function readGrid(path)
    grid, PROJECTED = open(path*"grid", "r") do f
        Ny = read(f, Int)
        Nz = read(f, Int)
        Nt = read(f, Int)
        y = read!(f, Vector{Float64}(undef, Ny))
        Dy = read!(f, Matrix{Float64}(undef, Ny, Ny))
        Dy2 = read!(f, Matrix{Float64}(undef, Ny, Ny))
        ws = read!(f, Vector{Float64}(undef, Ny))
        ω = read(f, Float64)
        β = read(f, Float64)
        PROJECTED = read(f, Bool)
        return Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β), PROJECTED
    end
    return grid, PROJECTED
end

function writeField(path, field)
    open(path*"field", "w") do f
        write(f, size(field)...)
        write(f, field)
    end
end

function readField(path, ::Type{T}=Float64) where {T}
    open(path*"field", "r") do f
        Ny = read(f, Int)
        Nz = read(f, Int)
        Nt = read(f, Int)
        read!(f, Array{Complex{T}, 3}(undef, Ny, Nz, Nt))
    end
end

