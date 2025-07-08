function test_vectorfield()
    @testset "Vector Field Constructor              " begin
        # generate random arrays
        Ny = rand(3:50)
        Nz = rand(3:2:51)
        Nt = rand(3:2:51)
        u1 = rand(Float64, (Ny, Nz, Nt))
        u2 = rand(Float64, (Ny, Nz))
        v1 = rand(Float64, (Ny, Nz, Nt))
        v2 = rand(Int, (Ny, Nz, Nt))
        w1 = rand(Float64, (Ny, Nz, Nt))
        w2 = "String"
        ω = abs(randn())
        β = abs(randn())

        # initialise grid
        grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny), ω, β)

        # initialise dummy functions
        fun1(y, z, t) = 1.0
        fun2(y, z, t) = 2.0
        fun3(y, z, t) = 3.0

        # initialise
        @test VectorField(u1, v1) isa VectorField{2, Array{Float64, 3}}
        @test VectorField(u1, v1, w1) isa VectorField{3, Array{Float64, 3}}
        @test VectorField(grid) isa VectorField{3, SpectralField{typeof(grid), false}}
        @test VectorField(grid; N=2) isa VectorField{2, SpectralField{typeof(grid), false}}
        @test VectorField(grid, N=5, fieldType=PhysicalField) isa VectorField{5, PhysicalField{typeof(grid), false, 1.5}}
        @test VectorField(grid, true, N=5) isa VectorField{5, PhysicalField{typeof(grid), true, 1.5}}
        @test VectorField(grid, fun1) isa VectorField{1, PhysicalField{typeof(grid), false, 1.5}}
        @test VectorField(grid, fun1, fun2, fun3) isa VectorField{3, PhysicalField{typeof(grid), false, 1.5}}

        # test copy  and similar methods
        a = VectorField(grid)
        [a[i] .= rand(ComplexF64, Ny, (Nz >> 1) + 1, Nt) for i in eachindex(a)]
        @test copy(a) == a
        @test similar(a) isa VectorField{3, SpectralField{typeof(grid), false}}
        @test similar(a, 5) isa VectorField{5, SpectralField{typeof(grid), false}}

        # # catch errors on constructors
        @test_throws MethodError VectorField(u2, v1)
        @test_throws MethodError VectorField(u1, v2, w1)
        @test_throws MethodError VectorField(u1, v1, w2)
        @test_throws MethodError VectorField("string1", "string2")
    end

    @testset "Vector Field Grid Methods             " begin
        # initialise random variables
        Ny = rand(3:50)
        Nt = rand(3:2:51)
        Nz = rand(3:2:51)
        ω = abs(randn())
        β = abs(randn())

        # initialise fields
        grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny), ω, β)
        a = SpectralField(grid)
        b = PhysicalField(grid)
        c = VectorField(grid)

        # test grid comparison method
        for i in [a, b, c]
            for j in [a, b, c]
                eval(:(@test grideq($i, $j)))
            end
        end
    end

    @testset "Vector Field Broadcasting             " begin
        # initialise random variables
        Ny = rand(3:50)
        Nz = rand(3:2:51)
        Nt = rand(3:2:51)
        ω = abs(randn())
        β = abs(randn())

        # initialise grid
        grid = Grid(rand(Ny), Nz, Nt, rand(Ny, Ny), rand(Ny, Ny), rand(Ny), ω, β)

        # initialise vector fields
        a = VectorField(PhysicalField(grid), PhysicalField(grid), PhysicalField(grid))
        b = VectorField(PhysicalField(grid), PhysicalField(grid), PhysicalField(grid))
        c = VectorField(PhysicalField(grid), PhysicalField(grid), PhysicalField(grid))

        # check broadcasting is done efficiently and does not allocate
        nalloc(a, b, c) = @allocated a .= 3 .* b .+ c ./ 2
        @test nalloc(a, b, c) == 0

        # test broadcasting
        @test typeof(a .+ b) == typeof(a)
    end

    @testset "Vector Field Farazmand Scaling        " begin
        # initialise
        Ny = rand(3:50); Nz = rand(3:2:51); Nt = rand(3:2:51)
        y = rand(Float64, Ny)
        Dy = rand(Float64, (Ny, Ny))
        Dy2 = rand(Float64, (Ny, Ny))
        ws = rand(Float64, Ny)
        ω = abs(randn())
        β = abs(randn())
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
        a = VectorField(grid)
        for i in 1:3
            a[i] .= rand(ComplexF64, Ny, (Nz >> 1) + 1, Nt)
        end
        A = Fields.FarazmandScaling(ω, β)

        # compute scaling
        b = mul!(similar(a), A, a)

        passed = true
        for i in 1:3, ny in 1:Ny, nz in 2:((Nz >> 1) + 1), nt in 1:Nt
            if !(b[i][ny, nz, nt] ≈ a[i][ny, nz, nt]/(1 + (nz*β)^2 + (nt*ω)^2))
                passed = false
                break
            end
        end
        for i in 1:3, ny in 1:Ny, nt in 2:((Nt >> 1) + 1)
            if !(b[i][ny, 1, nt] ≈ a[i][ny, 1, nt]/(1 + β^2 + (nt*ω)^2))
                passed = false
                    break
            end
        end
        for i in 1:3, ny in 1:Ny
            if !(b[i][ny, 1, 1] ≈ a[i][ny, 1, 1]/(1 + β^2 + ω^2))
                passed = false
                break
            end
        end
        @test passed
    end

    @testset "Vector Field Norm                     " begin
        # initialise grid
        Ny = 64; Nz = 65; Nt = 65
        y = chebpts(Ny)
        Dy = chebdiff(Ny)
        Dy2 = chebddiff(Ny)
        ws = chebws(Ny)
        ω = abs(rand())
        β = abs(rand())
        grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
        A = Fields.FarazmandScaling(ω, β)

        # definition of fields as functions
        u_func(y, z, t) = (1 - y^2)*exp(cos(β*z))*atan(sin(ω*t))
        v_func(y, z, t) = (cos(π*y) + 1)*exp(sin(β*z))*cos(sin(ω*t))
        w_func(y, z, t) = sin(atan(y))*(cos(β*z)/(sin(β*z)^2 + 1))*exp(cos(ω*t))

        # initialise fields
        vec_p = VectorField(PhysicalField(grid, u_func),
                            PhysicalField(grid, v_func),
                            PhysicalField(grid, w_func))
        vec_s = VectorField(grid)
        FFT! = FFTPlan!(grid; flags=ESTIMATE)
        FFT!(vec_s, vec_p)

        # test norm
        @test norm(vec_s)^2 ≈ 0.41857 + 2.09248 + 0.172958 rtol=1e-5

        # test weights norm
        @test norm(vec_s, A) ≈ sqrt(dot(vec_s, mul!(similar(vec_s), A, vec_s)))
    end
end
