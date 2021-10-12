module Fields

include("physicalfield.jl")
include("spectrafield.jl")
include("vectorfields.jl")

export PhysicalField, 
       SpectralField,
       VectorField

end
