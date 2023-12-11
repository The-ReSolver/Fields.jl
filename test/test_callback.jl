struct DummyTrace
    value::Float64
    g_norm::Float64
    iteration::Int
    metadata::Dict
end

@testset "Callback function             " begin
    # test construction
    @test Callback().trace.value == Vector{Float64}(undef, 0)
    @test Callback().trace.g_norm == Vector{Float64}(undef, 0)
    @test Callback().trace.iter == Vector{Int}(undef, 0)
    @test Callback().trace.time == Vector{Float64}(undef, 0)
    @test Callback().write == false
    @test Callback(write=true).write == true
    @test Callback(write_loc="somewhere").write_loc == "somewhere/"
    @test_nowarn Callback(Fields.Trace(rand(5), rand(5), rand(Int, 5), rand(5), rand(5)))
    @test_throws ArgumentError Callback(Fields.Trace(rand(5), rand(3), rand(Int, 5), rand(5), rand(5)))

    # test trace assignment
    cb = Callback()
    value = rand(); g_norm = rand(); iter = rand(1:10); metadata = Dict("time"=>rand(), "Current step size"=>rand(), "x"=>nothing);
    cb(DummyTrace(value, g_norm, iter, metadata))
    @test cb.trace.value == [value]
    @test cb.trace.g_norm == [g_norm]
    @test cb.trace.iter == [iter]
    @test cb.trace.time == [metadata["time"]]
    @test cb.trace.step_size == [metadata["Current step size"]]
    cb(DummyTrace(2*value, 4*g_norm, 5*iter, metadata))
    @test cb.trace.value == [value, 2*value]
    @test cb.trace.g_norm == [g_norm, 4*g_norm]
    @test cb.trace.iter == [iter, 5*iter]
    @test cb.trace.time == [metadata["time"], metadata["time"]]
    @test cb.trace.step_size == [metadata["Current step size"], metadata["Current step size"]]
end
