# This file contains the definitions required to compute the rate of change of
# the variational dynamics given a set of modes to perform a Galerkin
# projection.

# TODO: functor for Evolution
# TODO: intermediate methods
# TODO: store modes so they are accessed in via columns (slice earlier indexes)
# TODO: figure out what is going on at the mean mode with the new fields

struct Evolution{Ny, Nz, Nt, M, G, T, PLAN, IPLAN}
    out::VectorField{3, SpectralField{Ny, Nz, Nt, G, T, Array{Complex{T}, 3}}}
    modes::Matrix{ComplexF64, 4}
    proj_cache::Vector{SpectralField{Ny, Nz, Nt, G, T, Array{Complex{T}, 3}}}
    spec_cache::Vector{SpectralField{Ny, Nz, Nt, G, T, Array{Complex{T}, 3}}}
    phys_cache::Vector{PhysicalField{Ny, Nz, Nt, G, T, Array{T, 3}}}
    fft::FFTPlan!{Ny, Nz, Nt, PLAN}
    ifft::IFFTPlan!{Ny, Nz, Nt, IPLAN}
    Re_recip::T
    Ro::T

    modes
    proj_cache::Vector

    function Evolution(grid::Grid{S}, Ψs::Matrix{ComplexF64, 4}, Re::Real, Ro::Real) where {S}
        # initialise output vector field
        out = VectorField(grid, N=3)

        # generate grid for projected fields
        proj_grid = Grid(ones(size(modes, 2)), S[2], S[3], grid.Dy, grid.Dy2, grid.ws, grid.ω, grid.β)

        # create field cache
        proj_cache = [SpectralField(proj_grid) for _ in 1:1]
        spec_cache = [SpectralField(grid)      for _ in 1:1]
        phys_cache = [PhysicalField(grid)      for _ in 1:1]

        # create transform plans
        FFT! = FFTPlan!(grid)
        IFFT! = IFFTPlan!(grid)

        # convert parameters to compatible type
        Re = convert(eltype(phys_cache[1]), Re)
        Ro = convert(eltype(phys_cache[1]), Ro)

        new{S...,
            size(modes, 2)
            typeof(grid),
            eltype(phys_cache[1]),
            typeof(FFT!.plan),
            typeof(IFFT!.plan)}(out,
                                Ψs,
                                proj_cache,
                                spec_cache,
                                phys_cache,
                                FFT!,
                                IFFT!,
                                1/Re,
                                Ro)
    end
end

function (f::Evolution{Ny, Nz, Nt})() where {Ny, Nz, Nt} end


function _compute_lr() end
function _compute_evolution() end
function _compute_quadratic() end
