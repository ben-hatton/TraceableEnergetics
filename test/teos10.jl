# The TEOS-10 seawater of ClimFluids in both models, for its three conservative variables: finite tendencies that
# allocate nothing on a serial manager, the energy budget closed to round-off, and a certified Lorenz reference state.
# Included in a module after config.jl, helpers.jl, an_model.jl and fc_model.jl.
using CFAnelastics, CFCompressible, CookBooks
using TraceableEnergetics
using ClimFluids: TEOS10Seawater

const CONSVARS = (:conservative_temperature, :potential_temperature, :entropy)

# the tracers of config.jl (θ [K], q [kg/kg]) as the conservative variable of `fluid`
consvar_of(fluid) = (x, z) -> fluid(:p, :theta, :q).conservative_variable(fluid.p0, θ_active(x, z), q_active(x, z))

# a mass streamfunction vanishing on the walls, so that ∇·(ρ u) = 0 discretely
ψ_active(X, Z) = ρ0 * U * Lz * (sin(π * Z / Lz) * (1 + 0.5 * cos(2π * X / Lx + 0.3)) + 0.3 * sin(2π * Z / Lz) * sin(4π * X / Lx))

active_state(model::AN2D) = CFAnelastics.initialize(model; consvar = consvar_of(model.fluid), q = q_active,
                                                    velocity = CFAnelastics.divergence_free_velocity(model, ψ_active))
function active_state(model::FC2D)
    consvar = consvar_of(model.fluid)
    density = CFCompressible.balanced_density(model; consvar, q = q_active, pref = 1e5)
    velocity = CFCompressible.divergence_free_velocity(model, density, ψ_active)
    return CFCompressible.initialize(model; density, consvar, q = q_active, velocity)
end

# bytes allocated by a call of tend!, after a warm-up call, behind a function barrier
function allocated(tend!, dstate, scratch, model, state)
    tend!(dstate, scratch, model, state)
    return @allocated tend!(dstate, scratch, model, state)
end

tend!(dstate, scratch, model::AN2D, state) = an_tend!(dstate, scratch, model, state)
tend!(dstate, scratch, model::FC2D, state) = fc_tend!(dstate, scratch, model, state)

# FC2D has no reference state: that of the AN2D model serves for the manipulated conversion of the diagnostics
test_reference(model) = AnelasticReference(model.domain, model.space, an_density; pref = 1e5)

lorenz(model::AN2D) = AnelasticLorenzReference(model, ExactRearrangement())
lorenz(model::FC2D) = CompressibleLorenzReference(model, ExactRearrangement())

@testset "TEOS-10" begin
    @testset "TEOS-10, $M, $consvar" for (M, make) in (("AN2D", an_model), ("FC2D", fc_model)), consvar in CONSVARS
        model = make(; fluid = TEOS10Seawater(consvar), mgr = VectorizedCPU())
        state = active_state(model)
        dstate, scratch = evaluate(tend!, model, state)
        @test all_finite(state) && all_finite(dstate)

        # the energy budget closes to round-off, and no energy enters the closed box
        session = open(diagnostics(model; reference = test_reference(model)); state, dstate, scratch, model)
        largest(name) = maximum(abs, interior_diagnostic(session, name))
        potential = CookBooks.get(session, :integrated_potential_energy)
        @test largest(:potential_budget_residual) < 1e-10 * largest(:dρP)
        @test abs(CookBooks.get(session, :integrated_budget_residual)) < 1e-12 * abs(potential)
        @test abs(CookBooks.get(session, :integrated_energy_tendency)) < 1e-12 * abs(potential)
        close(session)

        # tendencies! allocates nothing on a serial manager, its scratch reused
        @test allocated(tend!, dstate, scratch, model, copy_state(state)) == 0

        # the exact rearrangement of the parcels is certified optimal
        ref = lorenz(model)
        reference_state!(ref, model, state)
        @test ref.column.certificate_violations == 0
    end
end
