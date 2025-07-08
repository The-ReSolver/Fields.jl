# This file contain the utility functions to allow the transformation of a
# spectral field into an equivalent state vector

function fieldToVector!(vec, a::SpectralField{<:Grid{Ny, Nz, Nt}, true}, ω) where {Ny, Nz, Nt}
    j = 1
    for m in axes(a, 1)
        vec[j] = real(a[m, 1, 1])
        j += 1
    end
    for nt in 2:((Nt >> 1) + 1), m in axes(a, 1)
        vec[j]   = real(a[m, 1, nt])
        vec[j+1] = imag(a[m, 1, nt])
        j += 2
    end
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), m in axes(a, 1)
        vec[j]   = real(a[m, nz, nt])
        vec[j+1] = imag(a[m, nz, nt])
        j += 2
    end
    vec[j] = ω
    return vec
end
fieldToVector(a::SpectralField{<:Grid{Ny, Nz, Nt}}, ω) where {Ny, Nz, Nt} = fieldToVector!(zeros(size(a, 1)*(2*Nt*(Nz >> 1) + 2*(Nt >> 1) + 1) + 1), a, ω)

function vectorToField!(a::SpectralField{<:Grid{Ny, Nz, Nt}, true}, vec) where {Ny, Nz, Nt}
    j = 1
    for m in axes(a, 1)
        a[m, 1, 1] = vec[j]
        j += 1
    end
    for nt in 2:((Nt >> 1) + 1), m in axes(a, 1)
        a[m, 1, nt] = vec[j] + 1im*vec[j+1]
        a[m, 1, end-nt+2] = vec[j] - 1im*vec[j+1]
        j += 2
    end
    for nt in 1:Nt, nz in 2:((Nz >> 1) + 1), m in axes(a, 1)
        a[m, nz, nt] = vec[j] + 1im*vec[j+1]
        j += 2
    end
    a.grid.dom[2] = vec[end]
    return a
end
