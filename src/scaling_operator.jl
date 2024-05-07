# Definition for the operator used to scale the fields and norms as a form of
# preconditioning.

abstract type NormScaling end


struct UniformScaling <: NormScaling end
Base.getindex(I::UniformScaling, ::Int, ::Int, ::Int) = 1


struct FarazmandScaling{T} <: NormScaling
    ω::T
    β::T
end
Base.getindex(A::FarazmandScaling, ::Int, nz::Int, nt::Int) = 1/(1 + (A.β*nz)^2 + (A.ω*nt)^2)
