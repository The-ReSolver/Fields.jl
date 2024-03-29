module Fields

using FFTW, LinearAlgebra, IniFile, Mmap, RecipesBase

using DAESolve

export ddy!, d2dy2!, ddz!, d2dz2!, ddt!, divergence!, laplacian!
export FFTPlan!, IFFTPlan!
export ESTIMATE, EXHAUSTIVE, MEASURE, PATIENT, WISDOM_ONLY, NO_TIMELIMIT
export Grid, points, spectralfield, physicalfield, vectorfield
export PhysicalField
export SpectralField, norm, dot
export VectorField
export get_grid, grideq, get_Dy, get_Dy2, get_ws, get_ω, get_β
export project, project!, reverse_project!, expand!
export Evolution, Constraint
export ResGrad
export gd!, optimise!, OptOptions, Callback
export generateGridOfModes

export DNSData, loadDNS, dns2field!, dns2field, correct_mean!, mean!, mean

include("projection.jl")
include("grid.jl")
include("generate_modes.jl")
include("physicalfield.jl")
include("spectralfield.jl")
include("vectorfields.jl")
include("fft.jl")
include("derivatives.jl")
include("dns2field.jl")
include("plot_vectorfield.jl")
include("resgrad.jl")
include("vectorToField.jl")

end
