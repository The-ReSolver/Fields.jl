# This file contains the definitions that will allow easy generation of a set
# of modes given a grid and the function to generate it.

function generateModes!(modes::Array{ComplexF64, 4}, grid::Grid{Ny, Nz, Nt}, M, Re, Ro; base::Vector{Float64}=ones(Ny), verbose::Bool=true) where {Ny, Nz, Nt}
    # get domain information
    ω = get_ω(grid)
    β = get_β(grid)
    ws = get_ws(grid)

    # intiialise resolvent operator
    H = Resolvent(Ny, get_Dy(grid), get_Dy2(grid))

    for nt in 1:((Nt >> 1) + 1), nz in 1:((Nz >> 1) + 1)
        verbose && print("$nz/$((Nz >> 1) + 1), $nt/$((Nt >> 1) + 1)       \r")
        if nz == nt == 1
            modes[:, :, 1, 1] = svd(H(0, 0, base, Re, Ro), ws, M).U
        elseif nt == 1
            modes[:, :, nz, 1] .= svd(H((nz - 1)*β, 0, base, Re, Ro), ws, M).U
        elseif nz == 1
            modes[:, :, 1, nt] .= svd(H(0, (nt - 1)*ω, base, Re, Ro), ws, M).U
            modes[:, :, 1, end-nt+2] .= conj.(modes[:, :, 1, nt])
        else
            modes[:, :, nz, nt] = svd(H((nz - 1)*β, (nt - 1)*ω, base, Re, Ro), ws, M).U
            modes[:, :, nz, end-nt+2] = svd(H((nz - 1)*β, (1 - nt)*ω, base, Re, Ro), ws, M).U
        end
    end

    return modes
end
generateModes(grid::Grid{Ny, Nz, Nt}, M, Re, Ro; base=ones(Ny), verbose=true) where {Ny, Nz, Nt} = generateModes!(Array{ComplexF64, 4}(undef, 3*Ny, M, (Nz >> 1) + 1, Nt), grid, M, Re, Ro, base=base, verbose=verbose)
