module Fields

include("physicalfield.jl")
include("spectrafield.jl")
include("vectorfields.jl")
include("fft.jl")

# TODO: Tests
# TODO: Transform between spectral and physical space
# TODO: Grid object to make the discretisation explicit (not completely necessary unless I want to change the discretisation?)
# TODO: What is going on with "#undef" elements in SpectraField (causes problems with indexing)

end
