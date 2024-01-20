# This file contains the definitions that will allow easy generation of a set
# of modes given a grid and the function to generate it.

# TODO: provide corresponding function in ResolventAnalysis.jl package to give mode_function
# TODO: test for this method

function generate_modes(grid::Grid{S}, retain, modeFunction) where {S}
    # get the size of the grid
    Nt, Nz, Nt = length.(points(grid))

    # generate arrays to hold modes
    modeWeights = zeros(retain, (Nz >> 1) + 1, Nt)
    modes = zeros(3*Ny, retain, (Nz >> 1) + 1, Nt)

    # loop over all frequencies
    for nz in 1:((Nz >> 1) + 1), nt in 1:Nt
        modes[:, :, nz, nt], modeWeights[:, nz, nt] = modeFunction(nz, nt)
    end

    return modes, modeWeights
end
