# This file contains the definitions that will allow easy generation of a set
# of modes given a grid and the function to generate it.

function generateModes!(modes::Array{ComplexF64, 4}, grid::Grid{Ny, Nz, Nt}, M, Re, Ro; base::Vector{Float64}=ones(Ny), verbose::Bool=true) where {Ny, Nz, Nt}
    # get domain information
    ω = get_ω(grid)
    β = get_β(grid)
    ws = get_ws(grid)

    # intiialise resolvent operator
    H = Resolvent(Ny, get_Dy(grid), get_Dy2(grid))

    # loop over frequencies computing response modes
    for nt in 1:((Nt >> 1) + 1), nz in 1:((Nz >> 1) + 1)
        verbose && print("$nz/$((Nz >> 1) + 1), $nt/$((Nt >> 1) + 1)       \r")
        if nt == 1
            modes[:, :, nz, 1] .= svd(H((nz - 1)*β, 0, base, Re, Ro), ws, M).U
        elseif nz == 1
            modes[:, :, 1, nt] .= svd(H(0, (nt - 1)*ω, base, Re, Ro), ws, M).U
            modes[:, :, 1, end-nt+2] .= conj.(modes[:, :, 1, nt])
        else
            modes[:, :, nz, nt] = svd(H((nz - 1)*β, (nt - 1)*ω, base, Re, Ro), ws, M).U
            modes[:, :, nz, end-nt+2] = svd(H((nz - 1)*β, (1 - nt)*ω, base, Re, Ro), ws, M).U
        end
        flush(stdout)
    end

    return modes
end
generateModes(grid::Grid{Ny, Nz, Nt}, M, Re, Ro; base=ones(Ny), verbose=true) where {Ny, Nz, Nt} = generateModes!(Array{ComplexF64, 4}(undef, 3*Ny, M, (Nz >> 1) + 1, Nt), grid, M, Re, Ro, base=base, verbose=verbose)

# TODO: test this
function generateModes(file, grid::Grid{Ny, Nz, Nt}, M, Re, Ro; base::Vector{Float64}=ones(Ny), verbose::Bool=true) where {Ny, Nz, Nt}
    # get domain information
    ω = get_ω(grid)
    β = get_β(grid)
    ws = get_ws(grid)

    # intiialise resolvent operator
    H = Resolvent(Ny, get_Dy(grid), get_Dy2(grid))

    # open file for writing
    f = open(file, "w")

    # write size of mode array
    write(f, Ny)
    write(f, M)
    write(f, Nz)
    write(f, Nt)

    # loop over frequencies computing response modes
    for nt in 1:((Nt >> 1) + 1), nz in 1:((Nz >> 1) + 1)
        verbose && print("$nz/$((Nz >> 1) + 1), $nt/$((Nt >> 1) + 1)       \r")
        if nt == 1
            write(f, svd(H((nz - 1)*β, 0, base, Re, Ro), ws, M).U)
        elseif nz == 1
            write(f, svd(H(0, (nt - 1)*ω, base, Re, Ro), ws, M).U)
            write(f, conj.(modes[:, :, 1, nt]))
        else
            write(f, svd(H((nz - 1)*β, (nt - 1)*ω, base, Re, Ro), ws, M).U)
            write(f, svd(H((nz - 1)*β, (1 - nt)*ω, base, Re, Ro), ws, M).U)
        end
        flush(stdout)
    end

    close(f)
    return nothing
end

function dumpModes(file, modes)
    open(file, "w") do f
        write(f, size(modes, 1)÷3)
        write(f, size(modes, 2))
        write(f, size(modes, 3))
        write(f, size(modes, 4))
        write(f, modes)
    end
    return nothing
end

function readModes!(modes, file)
    A = _creatMemMap(file)

    # check sizes are compatible
    size(A, 1) == size(modes, 1) || throw(ArgumentError("Input modes do not have the correct number of wall-normal points!"))
    size(modes, 4) % 2 == 1 || throw(ArgumentError("Temporal grid must have an odd number of points!"))

    # loop over size and assign values
    for m in axes(modes, 2), nz in axes(modes, 3)
        @views modes[:, m, nz, 1] .= A[:, m, nz, 1]
        for nt in 1:(size(modes, 4) >> 1)
            @views modes[:, m, nz, nt + 1] .= A[:, m, nz, nt + 1]
            @views modes[:, m, nz, end - nt + 1] .= A[:, m, nz, end - nt + 1]
        end
    end

    return modes
end
readModes(file, shape) = readModes!(Array{ComplexF64, 4}(undef, shape...), file)

function _creatMemMap(file)
    f = open(file, "r")
    Ny = read(f, Int)
    M = read(f, Int)
    Nz_spec = read(f, Int)
    Nt = read(f, Int)
    A = mmap(f, Array{ComplexF64, 4}, (3*Ny, M, Nz_spec, Nt))
    close(f)
    return A
end
