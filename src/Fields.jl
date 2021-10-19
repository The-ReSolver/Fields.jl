module Fields

include("grid.jl")
include("physicalfield.jl")
include("spectrafield.jl")
include("vectorfields.jl")
include("fft.jl")

# TODO: Tuple for differentiation matrices in grid type
# TODO: Performance of FFT for different plans and sizes
# TODO: What is going on with "#undef" elements in SpectraField (causes problems with indexing)
# TODO: generate weights for inner products and norms (Inside FGGrids.jl package)

end
