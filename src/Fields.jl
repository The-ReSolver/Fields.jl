module Fields

import FFTW
import LinearAlgebra

include("grid.jl")
include("physicalfield.jl")
include("spectralfield.jl")
include("vectorfields.jl")
include("fft.jl")

# TODO: Performance of FFT for different plans and sizes
# TODO: generate weights for inner products and norms (Inside FDGrids.jl package)

end
