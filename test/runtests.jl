# Run from the TraceableEnergetics folder with:  julia -t 4 --project=. test/runtests.jl
# Runs the setups tests, the canonical setups on both models, and the diagnostics and outputs on both models. The model
# suites of CFAnelastics and CFCompressible are run from their own repositories. Cross-model (traceability) tests will
# live here.
using Test
using TraceableEnergetics, CookBooks
using CFBoxes: Bounded

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
        include(joinpath($TESTS, "config.jl"))
        include(joinpath($TESTS, "helpers.jl"))
        include(joinpath($TESTS, "an_model.jl"))
        include(joinpath($TESTS, "canonical.jl"))
        const make_model, make_state, tend! = an_model, an_state, an_tend!     # model-independent names, used below
        @testset "canonical setups, AN2D" begin
            test_canonical(TraceableEnergetics.initialize, tend!, make_model(mgr = VectorizedCPU()),
                           make_model(boundary = (Bounded, Bounded), walls = forced_box))
        end
    end
    @eval module CompressibleTests
        include(joinpath($TESTS, "config.jl"))
        include(joinpath($TESTS, "helpers.jl"))
        include(joinpath($TESTS, "fc_model.jl"))
        include(joinpath($TESTS, "canonical.jl"))
        const make_model, make_state, tend! = fc_model, fc_state, fc_tend!
        @testset "canonical setups, FC2D" begin
            test_canonical((m, fields) -> TraceableEnergetics.initialize(m, fields; ptop = 1e5), tend!,
                           make_model(mgr = VectorizedCPU()), make_model(boundary = (Bounded, Bounded), walls = forced_box))
        end
    end
    # AN2D and FC2D share their diagnostic names, apart from the model-specific ones
    @testset "shared diagnostic names" begin
        an = keys(AnelasticTests.CFAnelastics.diagnostic_info(AnelasticTests.an_model()))
        fc = keys(CompressibleTests.CFCompressible.diagnostic_info(CompressibleTests.fc_model()))
        @test Set(setdiff(an, (:buoyancy, :kinematic_pressure_anomaly))) == Set(setdiff(fc, (:dρ,)))
    end
    # for either model, every diagnostic without its halo matches the coordinates of its grid location, and every
    # diagnostic is defined on its full array, halo included (no uninitialised memory)
    @testset "interior diagnostics, $M" for (M, T) in (("AN2D", AnelasticTests), ("FC2D", CompressibleTests))
        model = T.make_model(boundary = (Bounded, Bounded), walls = T.forced_box)
        state = T.make_state(model)
        dstate, scratch = T.evaluate(T.tend!, model, state)
        session = open(diagnostics(model); state, dstate, scratch, model)
        for name in keys(diagnostic_info(model))
            x, z = diagnostic_coordinates(model, name)
            @test size(interior_diagnostic(session, name)) == (length(x), length(z))
            @test all(isfinite, CookBooks.get(session, name))
        end
        close(session)
    end
    # for either model, data files read back the full diagnostics, with halo, at each time, in either format, and
    # videos hold their interior
    @testset "outputs, $M" for (M, T) in (("AN2D", AnelasticTests), ("FC2D", CompressibleTests))
        model = T.make_model(boundary = (Bounded, Bounded), walls = T.forced_box)
        state = T.make_state(model)
        dstate, scratch = T.evaluate(T.tend!, model, state)
        names, book, dir = (:u, :w, :conservative_variable, :vorticity), diagnostics(model), mktempdir()
        outputs = Outputs(DataFile(joinpath(dir, "out.nc"), model, names), DataFile(joinpath(dir, "out.jld2"), model, names),
                          DiagnosticVideo(model, :vorticity))
        foreach(t -> outputs(model, state, dstate, scratch, book, t), (0.0, 0.5))
        close(outputs)
        session = open(book; state, dstate, scratch, model)
        for path in readdir(dir; join = true), name in names
            field, t = read_diagnostic(path, name)
            @test t == [0.0, 0.5]
            @test field[:, :, 1] == field[:, :, 2] == CookBooks.get(session, name)
        end
        @test outputs[3].frames == [interior_diagnostic(session, :vorticity), interior_diagnostic(session, :vorticity)]
        close(session)
    end
end
