module Fields

using FFTW
using LinearAlgebra

include("grid.jl")
include("physicalfield.jl")
include("spectralfield.jl")
include("vectorfields.jl")
include("fft.jl")
include("derivatives.jl")
include("quadweights.jl")

end
