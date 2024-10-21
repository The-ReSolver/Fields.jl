module Fields

using FFTW, LinearAlgebra, IniFile, Mmap, ResolventAnalysis

export ddy!, d2dy2!, ddz!, d2dz2!, ddt!
export FFTPlan!, IFFTPlan!, apply_symmetry!
export ESTIMATE, EXHAUSTIVE, MEASURE, PATIENT, WISDOM_ONLY, NO_TIMELIMIT
export Grid, points, spectralfield, physicalfield, vectorfield
export PhysicalField
export SpectralField, norm, dot, interpolate
export VectorField
export get_grid, grideq, get_Dy, get_Dy2, get_ws, get_ω, get_β
export project, project!, expand!, project, expand
export fieldToVector!, vectorToField!
export ResGrad
export generateModes
export energy, energy!, vorticity!, enstrophy, boundaryEnergy

export DNSData, dnsToSpectralField, correct_mean!, mean!, mean

include("scaling_operator.jl")
include("grid.jl")
include("generate_modes.jl")
include("physicalfield.jl")
include("spectralfield.jl")
include("vectorfields.jl")
include("utils.jl")
include("projection.jl")
include("fft.jl")
include("derivatives.jl")
include("vectorToField.jl")
include("resgrad.jl")
include("dns2field.jl")

end
