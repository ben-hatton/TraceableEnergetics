# Run from the TraceableEnergetics folder with:  julia -t 4 --project=. test/runtests.jl
# Runs the setups tests, the canonical setups on both models, and the diagnostics and outputs on both models. The model
# suites of CFAnelastics and CFCompressible are run from their own repositories. Cross-model (traceability) tests will
# live here.
using Test
using TraceableEnergetics, CookBooks
using CFBoxes: Bounded, Periodic
using LoopManagers: VectorizedCPU

const TESTS = @__DIR__

# reference state (density increasing with depth, as that of the AN2D test model) on the grid of `model`, for the diagnostics
# bytes allocated by a second call of f(args...), the arguments being concrete
allocated(f, args...) = (f(args...); @allocated f(args...))

test_reference(model) = AnelasticTests.AnelasticReference(model.domain, model.space, AnelasticTests.an_density; pref = 1e5)

include(joinpath(TESTS, "..", "tools", "scales.jl"))

@testset "TraceableEnergetics" begin
    include(joinpath(TESTS, "scales.jl"))
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
        # FC2D has no reference state: the one of the matching AN2D (same density profile) serves for the manipulated conversion
        session = open(diagnostics(model; reference = test_reference(model)); state, dstate, scratch, model)
        for name in keys(diagnostic_info(model))
            isempty(diagnostic_info(model)[name].location) && continue     # a number, not a field
            x, z = diagnostic_coordinates(model, name)
            @test size(interior_diagnostic(session, name)) == (length(x), length(z))
            @test all(isfinite, CookBooks.get(session, name))
        end
        close(session)
    end
    # energy budget of energy.tex, for either model, with periodic, closed and forced boundaries: the budget of ρP closes at
    # each cell, the closure Q = Q_ϑ + Q_q + ρε_K holds, the vortex force does no work with the energy-conserving scheme, and
    # the budget of K + P closes in the volume integral; the manipulated conversion is close to C
    @testset "energy budget, $M, $(label)" for (M, T) in (("AN2D", AnelasticTests), ("FC2D", CompressibleTests)),
                                              (label, boundary, walls) in (("periodic", (Periodic, Bounded), (; bottom = T.closed_wall(), top = T.closed_wall())),
                                                                           ("closed", (Bounded, Bounded), (; left = T.closed_wall(), right = T.closed_wall(), bottom = T.closed_wall(), top = T.closed_wall())),
                                                                           ("forced", (Bounded, Bounded), T.forced_box))
        model = T.make_model(; boundary, walls, mgr = VectorizedCPU())
        state = T.make_state(model)
        dstate, scratch = T.evaluate(T.tend!, model, state)
        session = open(diagnostics(model; reference = test_reference(model)); state, dstate, scratch, model)
        largest(name) = maximum(abs, interior_diagnostic(session, name))
        @test largest(:potential_budget_residual) < 1e-10 * largest(:dρP)
        @test largest(:heating_residual) < 1e-10 * largest(:heating)
        @test largest(:conversion_residual) < 1e-2 * largest(:conversion)
        potential = CookBooks.get(session, :integrated_potential_energy)
        @test abs(CookBooks.get(session, :integrated_budget_residual)) < 1e-12 * potential
        @test abs(CookBooks.get(session, :integrated_vortex_work)) < 1e-12 * potential     # in the integral only: VF is not zero at a cell
        label == "forced" || @test abs(CookBooks.get(session, :integrated_energy_tendency)) < 1e-12 * potential    # no energy enters
        # the kernels allocate nothing on a serial manager (the arrays belong to the book)
        divergence_args = (CookBooks.get(session, :kinetic_flux_rev_x), CookBooks.get(session, :kinetic_flux_rev_z))
        out = copy(CookBooks.get(session, :div_kinetic_flux_rev))
        @test allocated(TraceableEnergetics.divergence!, out, model, divergence_args...) == 0
        close(session)
    end
    # the origin z0 of the geopotential in the tank shifts P by -g z0 and nothing else
    @testset "geopotential origin, $M" for (M, T) in (("AN2D", AnelasticTests), ("FC2D", CompressibleTests))
        model = T.make_model(boundary = (Bounded, Bounded), walls = T.forced_box)
        tank = T.Tank2D(Lx = T.Lx, Lz = T.Lz, g = T.g, z0 = 0.3)
        # the same model in a tank with another origin: every field but `space` is the same
        shifted = typeof(model)(map(f -> f === :space ? tank : getfield(model, f), fieldnames(typeof(model)))...)
        state = T.make_state(model)
        dstate, scratch = T.evaluate(T.tend!, model, state)
        P(m) = copy(interior_diagnostic(open(diagnostics(m; reference = test_reference(m)); state, dstate, scratch, model = m), :potential_energy))
        @test P(shifted) .- P(model) ≈ fill(-T.g * 0.3, size(P(model)))
    end
    # the gauge of the anelastic pressure (a constant added to φ) changes the pressure and no tendency or budget term: the
    # divergences of ρΣu and (p - p_R)u, whose fluxes change by the constant times the mass flux, stay the same because
    # ∇·(ρ_R u) = 0 discretely
    @testset "pressure gauge, AN2D, $label" for (label, boundary, walls) in (("periodic", (Periodic, Bounded), (; bottom = AnelasticTests.closed_wall(), top = AnelasticTests.closed_wall())),
                                                                              ("forced", (Bounded, Bounded), AnelasticTests.forced_box))
        T, c = AnelasticTests, 100.0
        results = map((0.0, c)) do gauge_offset
            model = T.make_model(; boundary, walls, mgr = VectorizedCPU(), gauge_offset)
            state = T.make_state(model)
            dstate, scratch = T.evaluate(T.tend!, model, state)
            session = open(diagnostics(model); state, dstate, scratch, model)
            get(name) = copy(interior_diagnostic(session, name))
            (; model, dstate = map(copy, dstate), get = get, scalar = name -> CookBooks.get(session, name))
        end
        (zero_gauge, shifted) = results
        # differences are round-off: the tendencies subtract ε c/Δ from δφ, the budget terms are noise of about 1e-11 of the
        # largest term (the tendency of ρP), as the residual maps show
        noise = 10 * eps() * c / min(zero_gauge.model.dx, zero_gauge.model.dz)
        for name in (:u, :w, :m)
            @test maximum(abs, getproperty(shifted.dstate, name) .- getproperty(zero_gauge.dstate, name)) < noise
        end
        invariant = (:dρK, :dρP, :conversion, :conversion_consvar, :conversion_composition, :conversion_bernoulli, :heating, :heating_residual,
                     :div_kinetic_flux_rev, :div_kinetic_flux_irr, :div_potential_flux_rev, :div_potential_flux_irr,
                     :kinetic_budget_residual, :potential_budget_residual, :div_pressure_flux, :buoyancy_conversion, :pressure_dilatation,
                     :conversion_residual, :potential_energy, :vortex_work)
        tolerance = 1e-11 * maximum(abs, zero_gauge.get(:dρP))
        for name in invariant
            @test maximum(abs, shifted.get(name) .- zero_gauge.get(name)) <= tolerance
        end
        ρ_R = zero_gauge.get(:density)
        @test zero_gauge.get(:kinematic_pressure_anomaly) .+ c ≈ shifted.get(:kinematic_pressure_anomaly)
        @test zero_gauge.get(:pressure) .+ c .* ρ_R ≈ shifted.get(:pressure)
        @test zero_gauge.get(:pressure_anomaly) .+ c .* ρ_R ≈ shifted.get(:pressure_anomaly)
        for name in (:integrated_kinetic_energy, :integrated_potential_energy, :integrated_energy_tendency, :integrated_boundary_flux,
                     :integrated_vortex_work, :integrated_budget_residual)
            @test shifted.scalar(name) ≈ zero_gauge.scalar(name) atol = 1e-11 * zero_gauge.scalar(:integrated_potential_energy)
        end
    end
    # for either model, data files read back the full diagnostics, with halo, and the numbers, at each time, in either
    # format, and videos hold their interior
    @testset "outputs, $M" for (M, T) in (("AN2D", AnelasticTests), ("FC2D", CompressibleTests))
        model = T.make_model(boundary = (Bounded, Bounded), walls = T.forced_box)
        state = T.make_state(model)
        dstate, scratch = T.evaluate(T.tend!, model, state)
        names = (:u, :w, :conservative_variable, :vorticity, :integrated_kinetic_energy)
        book, dir = diagnostics(model), mktempdir()
        outputs = Outputs(DataFile(joinpath(dir, "out.nc"), model, names), DataFile(joinpath(dir, "out.jld2"), model, names),
                          DiagnosticVideo(model, :vorticity))
        foreach(t -> outputs(model, state, dstate, scratch, book, t), (0.0, 0.5))
        close(outputs)
        session = open(book; state, dstate, scratch, model)
        for path in readdir(dir; join = true), name in names
            field, t = read_diagnostic(path, name)
            @test t == [0.0, 0.5]
            expected = CookBooks.get(session, name)
            if expected isa Number
                @test field == [expected, expected]     # a vector over t
            else
                @test field[:, :, 1] == field[:, :, 2] == expected
            end
        end
        @test outputs[3].frames == [interior_diagnostic(session, :vorticity), interior_diagnostic(session, :vorticity)]
        close(session)
    end
end
