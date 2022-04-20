# This file computes the quadrature weights for a discretisation in the wall-
# normal direction.

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

# http://www2.math.umd.edu/~dlevy/classes/amsc466/lecture-notes/integration-chap.pdf (section 6.4)
function _quadweights(y::AbstractVector)
    # number of points
    N = length(y)

    # find integral of polynomial of degree d from y[1] to y[N]
    b = [(y[1]^d - y[end]^d)/d for d = 1:N]

    # evaluate polynomial up to degree d on points
    A = [y[i]^d for d = 0:(N - 1), i = 1:N]

    return A\b
end
