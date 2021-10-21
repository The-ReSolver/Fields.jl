module Fields

include("grid.jl")
include("physicalfield.jl")
include("spectrafield.jl")
include("vectorfields.jl")
include("fft.jl")

# Constructor for vector field based off of grid
# TODO: Test for FFT of vector field
# TODO: Performance of FFT for different plans and sizes
# TODO: generate weights for inner products and norms (Inside FGGrids.jl package)

end
