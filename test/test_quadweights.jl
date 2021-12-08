@testset "Quadrature Weights            " begin
    # quadrature
    y = range(2, stop = -2, length = 121)
    I = exp(2) - exp(-2)
    @test abs(sum(exp.(y) .* quadweights(y, 1)) - I) < 1e-3
    @test abs(sum(exp.(y) .* quadweights(y, 2)) - I) < 5e-8
    @test abs(sum(exp.(y) .* quadweights(y, 3)) - I) < 2e-7
    @test abs(sum(exp.(y) .* quadweights(y, 4)) - I) < 1e-10

    # also does not depend on points
    y = [1, 0.2, -0.2, -1]
    @test abs(sum((x->x^3).(y) .* quadweights(y, 3))) < 1e-5
    y = [1, 0.2, -0.1, -1]
    @test abs(sum((x->x^3).(y) .* quadweights(y, 3))) < 1e-5
    y = [1, 0.9, -0.1, -1]
    @test abs(sum((x->x^3).(y) .* quadweights(y, 3))) < 1e-5

    # trapz
    @test Fields._quadweights([1, 0]) ≈ [1, 1]/2

    # simpson
    @test Fields._quadweights([1, 0.5, 0]) ≈ [1, 4, 1]/6

    # generic polynomial
    y = [1, 0.2, -0.2, -1]
    # @test sum((x->x^3).(y) .* Fields._quadweights(y)) ≈ 0
    @test sum((x->x^2).(y) .* Fields._quadweights(y)) ≈ 2/3
    # @test sum((x->x).(y) .* Fields._quadweights(y)) ≈ 0
    @test sum((x->1).(y) .* Fields._quadweights(y)) ≈ 2
end
