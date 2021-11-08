using Fields
using BenchmarkTools
using Random

Nz_max = 64; Nt_max = 64

f1 = open("./perf/fft_results_course.txt", "w")

for nz in 1:Nz_max, nt in 1:Nt_max
    Ny = 8; Nz = nz; Nt = nt
    grid = Grid(rand(Float64, Ny), Nz, Nt,
                rand(Float64, (Ny, Ny)), rand(Float64, (Ny, Ny)),
                rand(Float64, Ny),
                1.0, 1.0)
    u = PhysicalField(grid)
    û = SpectralField(grid)
    FFT = FFTPlan!(grid)
    IFFT = IFFTPlan!(grid)

    redirect_stdout(f1) do
        @btime $FFT($û, $u)
        @btime $IFFT($u, $û)
        println()
    end

end

close(f1)

open("./perf/fft_results_course.txt") do f1
    open("./perf/fft_results.txt", "w") do f2
        for nz in 1:Nz_max, nt in 1:Nt_max
            redirect_stdout(f2) do
                println("For size: (nz=$nz, nt=$nt):")
                print("FFT:    ")
                println(readline(f1))
                print("IFFT:   ")
                println(readline(f1))
                println(readline(f1))
            end
        end
    end
end
