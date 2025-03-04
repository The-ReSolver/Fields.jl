using Test
using Random
using LinearAlgebra
using IniFile

using Fields
using ChebUtils
using FDGrids

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

function testAll()
    test_grid()
    test_physicalfield()
    test_spectralfield()
    test_vectorfield()
    test_utils()
    test_projection()
    test_fft()
    test_derivatives()
    test_vectorToField()
    test_resgrad()
end

function testSome(args::Vararg{String, N}) where {N}
    if "grid" in args
        test_grid()
    end
    if "physicalfield" in args
        test_physicalfield()
    end
    if "spectralfield" in args
        test_spectralfield()
    end
    if "vectorfield" in args
        test_vectorfield()
    end
    if "utils" in args
        test_utils()
    end
    if "projection" in args
        test_projection()
    end
    if "fft" in args
        test_fft()
    end
    if "derivatives" in args
        test_derivatives()
    end
    if "vectortofield" in args
        test_vectorToField()
    end
    if "resgrad" in args
        test_resgrad()
    end
end

if isempty(ARGS) || "all" in ARGS
    testAll()
else
    testSome(ARGS...)
end
