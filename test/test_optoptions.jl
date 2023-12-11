@testset "Optimisation Options          " begin
    @test OptOptions().write_loc == "./"
    @test_throws AssertionError OptOptions(; write_loc="/home/tom/somewhere")
    @test OptOptions(; write_loc="/home/tom/somewhere/").write_loc == "/home/tom/somewhere/"
    @test !OptOptions().callback.write
    @test OptOptions(; callback=Callback(; write=true)).callback.write
end
