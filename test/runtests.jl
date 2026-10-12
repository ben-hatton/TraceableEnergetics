# Run from the TraceableEnergetics folder with:  julia -t 4 --project=. test/runtests.jl
# Runs the setups tests, the model suites of the work-in-progress packages against the versions in packages/, and
# the canonical setups on both models. Cross-model (traceability) tests will live here.
using Test

const PACKAGES = joinpath(@__DIR__, "..", "packages")
const TESTS = @__DIR__

@testset "TraceableEnergetics" begin
    @eval module SetupTests
        using Test, TraceableEnergetics
        using CFBoxes: Periodic, Bounded, Box2D
        using CFPlanets: Tank2D
        using ClimFluids: NonlinearBinaryFluid
        include(joinpath($TESTS, "setups.jl"))
    end
    # each suite in its own module, as both define the same configuration names
    @eval module AnelasticTests
        include(joinpath($PACKAGES, "CFAnelastics", "test", "runtests.jl"))
        include(joinpath($TESTS, "canonical.jl"))
        @testset "canonical setups, AN2D" begin
            test_canonical(TraceableEnergetics.initialize, an_tend!, an_model(mgr = VectorizedCPU()),
                           an_model(boundary = (Bounded, Bounded), walls = forced_box))
        end
    end
    @eval module CompressibleTests
        include(joinpath($PACKAGES, "CFCompressible", "test", "box2d.jl"))
        include(joinpath($TESTS, "canonical.jl"))
        @testset "canonical setups, FC2D" begin
            test_canonical((m, fields) -> TraceableEnergetics.initialize(m, fields; ptop = 1e5), fc_tend!,
                           fc_model(mgr = VectorizedCPU()), fc_model(boundary = (Bounded, Bounded), walls = forced_box))
        end
    end
    # AN2D and FC2D share their diagnostic names, apart from the model-specific ones
    @testset "shared diagnostic names" begin
        an = keys(AnelasticTests.CFAnelastics.diagnostic_info(AnelasticTests.an_model()))
        fc = keys(CompressibleTests.CFCompressible.diagnostic_info(CompressibleTests.fc_model()))
        @test Set(setdiff(an, (:buoyancy, :kinematic_pressure_anomaly))) == Set(setdiff(fc, (:dρ,)))
    end
end
