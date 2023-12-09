struct DummyTrace
    value::Float64
    g_norm::Float64
    iteration::Int
    metadata::Dict
end

@testset "Callback function             " begin
    # test construction
    @test Callback() isa Callback{false}
    @test Callback(write=true) isa Callback{true}
    @test Callback().value == Vector{Float64}(undef, 0)
    @test Callback().g_norm == Vector{Float64}(undef, 0)
    @test Callback().iter == Vector{Int}(undef, 0)
    @test Callback().time == Vector{Float64}(undef, 0)
    @test_nowarn Callback(rand(5), rand(5), rand(Int, 5), rand(5), rand(5))
    @test_throws ArgumentError Callback(rand(5), rand(3), rand(Int, 5), rand(5), rand(5))

    # test trace assignment
    trace = Callback()
    value = rand(); g_norm = rand(); iter = rand(1:10); metadata = Dict("time"=>rand(), "Current step size"=>rand(), "x"=>nothing);
    trace(DummyTrace(value, g_norm, iter, metadata))
    @test trace.value == [value]
    @test trace.g_norm == [g_norm]
    @test trace.iter == [iter]
    @test trace.time == [metadata["time"]]
    @test trace.step_size == [metadata["Current step size"]]
    trace(DummyTrace(2*value, 4*g_norm, 5*iter, metadata))
    @test trace.value == [value, 2*value]
    @test trace.g_norm == [g_norm, 4*g_norm]
    @test trace.iter == [iter, 5*iter]
    @test trace.time == [metadata["time"], metadata["time"]]
    @test trace.step_size == [metadata["Current step size"], metadata["Current step size"]]
end