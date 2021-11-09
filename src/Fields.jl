module Fields

import FFTW
import LinearAlgebra

include("grid.jl")
include("physicalfield.jl")
include("spectralfield.jl")
include("vectorfields.jl")
include("fft.jl")
include("operators.jl")
include("quadweights.jl")

# TODO: norms and inner product functions based off of quadrature weights

end
