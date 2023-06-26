@testset "Channel Profile Integration" begin
    # construct channel profiles to be integrated
    Ny = 64
    y = chebpts(Ny)
    ws = chebws(Ny)
    u = [(x^2)*cos(π*x/2) for x in y]
    v = [exp(-5*(x^2)) for x in y]

    @test Fields.channel_int(u, ws, v) ≈ 0.0530025 rtol=1e-6
end