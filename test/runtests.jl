# Run from the TraceableEnergetics folder with:  julia -t 4 --project=. test/runtests.jl
# Runs the setups tests, the canonical setups on both models, the reference state of available energy and the budgets of
# the available energy, the TEOS-10 seawater on both models (teos10.jl), and the diagnostics and outputs on both models
# (diagnostics.jl). The model suites of CFAnelastics and CFCompressible are run from their own repositories. With
# TE_EXAMPLES=true, it also runs the experiment scripts listed in EXAMPLES below in their quick variant
# (test/examples.jl). Cross-model (traceability) tests will live here.
using Test
using TraceableEnergetics, CookBooks
using CFBoxes: Bounded, Periodic
using LoopManagers: VectorizedCPU

const TESTS = @__DIR__

include(joinpath(TESTS, "..", "tools", "scales.jl"))
include(joinpath(TESTS, "..", "tools", "experiments.jl"))
include(joinpath(TESTS, "examples.jl"))

# the experiment scripts converted to the template of notes/experiments/experiment_plan.md, as examples/<script>; run
# in their quick variant only when TE_EXAMPLES=true
const EXAMPLES = String[]

@testset "TraceableEnergetics" begin
    include(joinpath(TESTS, "scales.jl"))
    include(joinpath(TESTS, "experiments.jl"))
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
            test_canonical((m, fields) -> TraceableEnergetics.initialize(m, fields; pref = 1e5), tend!,
                           make_model(mgr = VectorizedCPU()), make_model(boundary = (Bounded, Bounded), walls = forced_box))
        end
    end
    # the Lorenz reference state of available potential energy, for either model
    @eval module ReferenceStateTests
        include(joinpath($TESTS, "config.jl"))
        include(joinpath($TESTS, "reference_state.jl"))
        include(joinpath($TESTS, "available_energy_budget.jl"))
        include(joinpath($TESTS, "reference_tendency.jl"))
        include(joinpath($TESTS, "transportation.jl"))
        include(joinpath($TESTS, "exact_rearrangement.jl"))
    end
    # the TEOS-10 seawater of ClimFluids in both models, for its three conservative variables
    @eval module TEOS10Tests
        include(joinpath($TESTS, "config.jl"))
        include(joinpath($TESTS, "helpers.jl"))
        include(joinpath($TESTS, "an_model.jl"))
        include(joinpath($TESTS, "fc_model.jl"))
        include(joinpath($TESTS, "teos10.jl"))
    end
    # the diagnostics and outputs of either model
    include(joinpath(TESTS, "diagnostics.jl"))
    # the experiment scripts, opt-in: they take minutes even in their quick variant
    if get(ENV, "TE_EXAMPLES", "false") == "true"
        @testset "examples (quick)" begin
            test_examples(EXAMPLES)
        end
    end
end
