# This file computes the quadrature weights for a discretisation in the wall-
# normal direction.

export quadweights

function quadweights(y::AbstractVector, order::Int)
    # number of points
    N = length(y)

    # partition y so that we have at least order+1 points
    w = zeros(N)

    # ii and ie are the initial and final indices
    ii, ie = 1, 1
    while ie < N
        ie = ii + order
        # if next interval is smaller, just go till the end
        ie = N - ie < order ? N : ie
        rng = ii:ie
        w[rng] += _quadweights(y[rng])
        ii = ie
    end

    return w
end

function _quadweights(y::AbstractVector)
    # number of points
    N = length(y)

    # find integral of polynomial of degree d from y[1] to y[N]
    b = [(y[1]^d - y[end]^d)/d for d = 1:N]

    # evaluate polynomial up to degree do on points
    A = [y[i]^d for d = 0:(N - 1), i = 1:N]

    return A\b
end
