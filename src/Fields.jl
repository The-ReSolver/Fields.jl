module Fields

include("grid.jl")
include("physicalfield.jl")
include("spectrafield.jl")
include("vectorfields.jl")
include("fft.jl")

# TODO: Tests
# TODO: Transform between spectral and physical space
# TODO: Grid object to make the discretisation explicit (not completely necessary unless I want to change the discretisation?)
# TODO: What is going on with "#undef" elements in SpectraField (causes problems with indexing)
# TODO: add differentiation operators
# TODO: add weights for inner products and norms

end
