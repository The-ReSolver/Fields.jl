using Test
using Random
using LinearAlgebra
using IniFile

using Fields
using ChebUtils
using FDGrids

# TODO: smart testing using ARGS

include("test_projection.jl")
include("test_grid.jl")
include("test_generate_modes.jl")
include("test_physicalfield.jl")
include("test_spectralfield.jl")
include("test_vectorfield.jl")
include("test_fft.jl")
include("test_derivatives.jl")
include("test_resgrad.jl")
include("test_vectorToField.jl")
