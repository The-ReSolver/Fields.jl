module Fields

import FFTW
import LinearAlgebra

include("grid.jl")
include("physicalfield.jl")
include("spectralfield.jl")
include("vectorfields.jl")
include("fft.jl")
include("operators.jl")

# TODO: constructor for Laplace operator based on grid
# TODO: constructor for FFT plans based on grid
# TODO: performance of FFT for different plans and sizes
# TODO: generate weights for inner products and norms (Inside FDGrids.jl package)

end
