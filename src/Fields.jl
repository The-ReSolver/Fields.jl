module Fields

using FFTW, LinearAlgebra, IniFile, Mmap, RecipesBase, Printf, Optim, Parameters, LineSearches

using DAESolve

const LBFGS = Optim.LBFGS
const ConjugateGradient = Optim.ConjugateGradient
const GradientDescent = Optim.GradientDescent
const MomentumGradientDescent = Optim.MomentumGradientDescent
const AcceleratedGradientDescent = Optim.AcceleratedGradientDescent

const HagerZhang = LineSearches.HagerZhang
const MoreThuente = LineSearches.MoreThuente
const BackTracking = LineSearches.BackTracking
const StrongWolfe = LineSearches.StrongWolfe
const Static = LineSearches.Static

const InitialPrevious = LineSearches.InitialPrevious
const InitialStatic = LineSearches.InitialStatic
const InitialHagerZhang = LineSearches.InitialHagerZhang
const InitialQuadratic = LineSearches.InitialQuadratic
const InitialConstantChange = LineSearches.InitialConstantChange

export ddy!, d2dy2!, ddz!, d2dz2!, ddt!, divergence!, laplacian!
export FFTPlan!, IFFTPlan!
export ESTIMATE, EXHAUSTIVE, MEASURE, PATIENT, WISDOM_ONLY, NO_TIMELIMIT
export Grid, points, spectralfield, physicalfield, vectorfield
export PhysicalField
export SpectralField, norm
export VectorField
export get_grid, grideq, get_Dy, get_Dy2, get_ws, get_ω, get_β
export project, project!, reverse_project!, expand!
export Evolution, Constraint
export ResGrad
export gd!, optimise!, OptOptions, Callback

export DNSData, loadDNS, dns2field!, dns2field, correct_mean!, mean!, mean

include("projection.jl")
include("grid.jl")
include("physicalfield.jl")
include("spectralfield.jl")
include("vectorfields.jl")
include("fft.jl")
include("derivatives.jl")
include("dns2field.jl")
include("plot_vectorfield.jl")
include("resgrad.jl")
include("callback.jl")
include("optoptions.jl")
include("output.jl")
include("optimise.jl")

end
