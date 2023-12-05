@testset "Optimise outputs              " begin
    # construct grid
    Ny = 16; Nz = 4; Nt = 2;
    y = chebpts(Ny);
    Dy = chebdiff(Ny);
    Dy2 = chebddiff(Ny);
    ws = chebws(Ny);
    β = 1.0;
    ω = 1.0;
    g = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β);
    FFT! = FFTPlan!(g);
    ū = y;
    dūdy = Dy*ū;
    Re = 10;
    Ro = 0.5;
    retain = 1;
    g_p = Grid(Vector{Float64}(undef, retain), Nz, Nt, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), ones(retain), ω, β);

    # get choleksy matrices
    Z = zeros(Ny, Ny)
    L =[Diagonal(sqrt.(ws)) Z                   Z                  ;
        Z                   Diagonal(sqrt.(ws)) Z                  ;
        Z                   Z                   Diagonal(sqrt.(ws));]
    L_inv =[Diagonal(1 ./ sqrt.(ws)) Z                        Z                       ;
            Z                        Diagonal(1 ./ sqrt.(ws)) Z                       ;
            Z                        Z                        Diagonal(1 ./ sqrt.(ws));]

    # generate modes
    psis = zeros(ComplexF64, 3*Ny, retain, (Nz >> 1) + 1, Nt)
    for nt in 1:Nt, nz in 1:((Nz >> 1) + 1)
        psis[:, :, nz, nt] .= qr(L*rand(ComplexF64, 3*Ny, 3*Ny)*L_inv).Q[:, 1:retain]
    end

    # generate random mode weightings
    a = SpectralField(g_p);
    a .= rand(ComplexF64, retain, (Nz >> 1) + 1, Nt);

    # compute single iteration of optimisation and get output
    res = optimise(a, g, psis, ū, Re, Ro, opts=OptOptions(; maxiter=1, alg=Fields.GradientDescent()))

    @test res isa Fields.Results
    @test res.alg == "Gradient Descent"
    @test length(res.R_trace) == 2
    @test !res.converged
    @test res.iterations == 1
    @test length(res.dR_norm_trace) == 2
end