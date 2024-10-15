using Test
using Random
using LinearAlgebra
using IniFile

using Fields
using ChebUtils
using FDGrids

function testAll()
    include("test_grid.jl")
    include("test_physicalfield.jl")
    include("test_spectralfield.jl")
    include("test_vectorfield.jl")
    include("test_utils.jl")
    include("test_projection.jl")
    include("test_fft.jl")
    include("test_derivatives.jl")
    include("test_vectorToField.jl")
    include("test_resgrad.jl")
end

# TODO: this
function testArgs(args...)

end

isempty(ARGS) ? testAll() : testArg(ARGS...)
