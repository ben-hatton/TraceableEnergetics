# Run from the TraceableEnergetics folder with:  julia -t 4 --project=. test/runtests.jl
# Runs the model suites of the work-in-progress packages against the versions in packages/.
# Cross-model (traceability) tests will live here.
using Test

const PACKAGES = joinpath(@__DIR__, "..", "packages")

@testset "TraceableEnergetics" begin
    # each suite in its own module, as both define the same configuration names
    @eval module AnelasticTests
        include(joinpath($PACKAGES, "CFAnelastics", "test", "runtests.jl"))
    end
    @eval module CompressibleTests
        include(joinpath($PACKAGES, "CFCompressible", "test", "box2d.jl"))
    end
end
