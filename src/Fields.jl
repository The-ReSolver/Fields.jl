module Fields

using FFTW
using LinearAlgebra

export ddy!, d2dy2!, ddz!, d2dz2!, ddt!
export FFTPlan!, IFFTPlan!
export ESTIMATE, EXHAUSTIVE, MEASURE, PATIENT, WISDOM_ONLY, NO_TIMELIMIT
export Grid, points
export PhysicalField
export SpectralField, norm
export VectorField
export quadweights
export grid, grideq, get_Dy, get_Dy2, get_ws, get_ω, get_β

include("grid.jl")
include("physicalfield.jl")
include("spectralfield.jl")
include("vectorfields.jl")
include("fft.jl")
include("derivatives.jl")
include("quadweights.jl")

end
