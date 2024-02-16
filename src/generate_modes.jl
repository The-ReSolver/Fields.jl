# This file contains the definitions that will allow easy generation of a set
# of modes given a grid and the function to generate it.

# TODO: provide corresponding function in ResolventAnalysis.jl package to give mode_function
# TODO: extra argument for modeFunction for "retain"

function generateGridOfModes(grid::Grid{S}, retain, modeFunction) where {S}
    # get the size of the grid
    Ny, Nz, Nt = length.(points(grid))

    # get domain size information
    ω = get_ω(grid)
    β = get_β(grid)

    # generate arrays to hold modes
    modes = Array{ComplexF64}(undef, 3*Ny, retain, (Nz >> 1) + 1, Nt)

    # loop over all frequencies
    for nz in 1:((Nz >> 1) + 1), nt in 1:Nt
        modes[:, :, nz, nt] .= modeFunction((nz - 1)*β, (nt - 1)*ω)
    end

    return modes
end
