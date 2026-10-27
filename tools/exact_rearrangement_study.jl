# Studies of the exact rearrangement of the Lorenz reference state (`ExactRearrangement`, the transportation simplex
# from the sorted placement, src/numerics/transportation.jl and src/reference_state/rearrangement.jl) against the sort
# (`SortedRearrangement`).
#
# 1. Cost: time, pivots and passes over the parcel-slot pairs of both methods, on the thermohaline lock exchange at rest
#    (examples/thermohaline_lock_exchange) on grids of M_z = 32, 64 and 128 rows (M_x = 2 M_z), and the error of the
#    sort: the underestimate of E_P - E_* that the exact rearrangement reports, relative to E_P - E_*.
# 2. Along a run of the lock exchange (AN2D and FC2D, 64 × 32, the grid of the example): at each snapshot, the
#    certificate violations of the sort, the pivots, passes and time of the exact rearrangement, the error of the sort
#    in E_P - E_* and in P_*, and the roughness in time of ∫ρΠ and of P_* for both methods (second differences between
#    successive snapshots, relative to the first differences): an exact placement can change between placements of
#    nearly equal energy but different slot volumes.
# 3. The N_*⁻² weights under strong thermobaricity at full size (K = 80, as notes/prototypes/weights.py): two
#    density-compensated families mixed in every row, with only one heated; the error of the weighted and plain means
#    against the centred difference of the slot volumes of the exact rearrangement, which includes the reordering.
# 4. Float32: the amplitude of thermobaricity γ below which Float32 no longer sees the reordering (the reduced costs
#    are below its certificate tolerance, 16 (K + 1) eps g L_z), on the active state of the tests.
#
# Results (2026-10-27, 4 threads):
# 1. The lock exchange at rest: the sort fails the certificate for 0 to 3 parcels (nearly tied, at round-off), which the
#    exact rearrangement fixes in 2 or 3 pivots and 3 or 4 passes; its error in E_P - E_* is round-off, 3e-11 of it in
#    AN2D (in FC2D the excess, a difference of the energies of the column, is -4e-10 of it, the round-off of
#    P_top L_x L_z). Time: the same as the sort when it has no violation (one pass, 1.2 ms at M_z = 64, 8 ms at 128), 2.6
#    to 3.5 times the sort with a few (0.4 ms at 32, 23 ms for FC2D at 128).
# 2. Along the run (64 × 32, 0 to 10 s): the sort fails for at most 1 (AN2D) or 2 (FC2D) parcels; the exact
#    rearrangement needs at most 2 pivots and 4 passes; it changes E_P - E_* by at most 9e-10 of it and P_* by at most
#    2e-10 Pa (AN2D) and 2e-8 Pa (FC2D). The roughness in time of E_P - E_* and of P_* is the same for both methods (the
#    FC2D E_P - E_* is rough with the sound waves, 0.55, whatever the method). In the lock exchange, whose γ is 1e-7 Pa⁻¹,
#    the sort is the Lorenz state to round-off.
# 3. The weights: the error of the N_*⁻²-weighted mean against the differences of the exact rearrangement is 0.26, 0.085
#    and 0.057 at K = 20, 40, 80, that of the plain mean 0.36, 0.35, 0.35 (the prototype, with other parameters: 0.006 and
#    0.2). Every parcel of this state is misplaced by the sort: the simplex needs 315, 1114 and 7405 pivots in 21, 38 and
#    104 passes (0.46 s at K = 80, against 0.9 ms for the sort).
# 4. Float32 sees no reordering on this state for any γ up to 1e-3 Pa⁻¹, where Float64 moves mass in 95 pivots and the
#    sort underestimates E_P - E_* by 1.6e-3 J m⁻¹: the reduced costs of the reordering are below the
#    tolerance of the certificate in Float32, 16 (K + 1) eps g L_z = 1.6e-4 J kg⁻¹ for K = 16. The exact rearrangement
#    needs Float64.
#
#   julia -t 4 --project=. tools/exact_rearrangement_study.jl
include(joinpath(@__DIR__, "..", "examples", "thermohaline_lock_exchange", "models.jl"))
using Printf
using CFBoxes: interior_cells, halo_size, ncells
using TraceableEnergetics: rearrangement_energy, reference_tendency!

method_reference(model::AN2D, method) = AnelasticLorenzReference(model, method)
method_reference(model::FC2D, method) = CompressibleLorenzReference(model, method)

# the reference state of both methods for a state: (sorted, exact) and their E_P - E_*, with the time of a second call
function both_methods(model, state)
    map((SortedRearrangement(), ExactRearrangement())) do method
        ref = method_reference(model, method)
        reference_state!(ref, model, state)
        time = @elapsed reference_state!(ref, model, state)
        (; ref, time, ΔE = rearrangement_energy(model, ref))
    end
end

#####
##### 1. Cost on the lock exchange at rest, against the grid
#####

println("1. The lock exchange at rest: cost of the exact rearrangement and error of the sort")
for Mz in (32, 64, 128)
    grid = Box2D(; Mx = 2Mz, Mz, Hx = 1, Hz = 1, boundary = (Bounded, Bounded))
    conditions = BoxBoundaryConditions(grid; left = closed, right = closed, bottom = closed, top = closed)
    fc_grid = FC2D((; physics..., domain = grid, boundary_conditions = conditions))
    ρ_grid = CFCompressible.balanced_density(fc_grid; consvar = θ_background, q = q_background, pref = 1e5)
    an_grid = AN2D((; physics..., domain = grid, boundary_conditions = conditions, buoyancy_scheme = ThermodynamicForcing(),
                    anelastic_reference = AnelasticReference(grid, space, ρ_grid[2, 2:end-1]; pref = 1e5), gauge_offset = 0.0))
    for model in (an_grid, fc_grid)
        state = initialize(model, setup(an_grid; θ = θ_lock, q = q_lock); (model isa FC2D ? (; pref = 1e5) : (;))...)
        sorted, exact = both_methods(model, state)
        @printf("  %s M_z = %3d: sort %.2e s, %d violations; exact %.2e s, %d pivots, %d passes, %d violations; underestimate of E_P - E_* by the sort %.2e of it\n",
                nameof(typeof(model)), Mz, sorted.time, sorted.ref.certificate_violations, exact.time, exact.ref.pivots,
                exact.ref.pricing_passes, exact.ref.certificate_violations, exact.ref.sorted_energy_excess / exact.ΔE)
    end
end

#####
##### 2. Along a run of the lock exchange
#####

println("\n2. Along a run of the lock exchange (64 × 32): every 0.25 s for 10 s")
# the second differences of a series, relative to its first differences: about Δt/T for a series smooth on the time
# scale T, about 2 for one that jumps
roughness(series) = sum(abs, diff(diff(series))) / sum(abs, diff(series))
for key in keys(models)
    model = models[key]
    state = deepcopy(states[key])
    scheme = CFTimeSchemes.RungeKutta4(model)
    solver_scratch = CFTimeSchemes.scratch_space(scheme, state, 0.0)
    t, Δt_out = 0.0, 0.25
    rows = []
    for snapshot in 0:40
        if snapshot > 0
            dt, steps = (model isa AN2D ? CFAnelastics : CFCompressible.Loops).cfl_condition(model, scheme, state, 0.9, Δt_out, 2.5e-3)
            state, t = CFTimeSchemes.advance!(state, CFTimeSchemes.IVPSolver(dt, scheme, solver_scratch), state, t, steps)
        end
        sorted, exact = both_methods(model, state)
        Δp = maximum(abs, exact.ref.p_star .- sorted.ref.p_star)
        push!(rows, (; t, sorted, exact, Δp))
    end
    pick(f) = map(f, rows)
    @printf("  %s: sort violations up to %d; exact: pivots up to %d, passes up to %d, time %.2e s at most (sort %.2e s), violations up to %d\n",
            key, maximum(pick(r -> r.sorted.ref.certificate_violations)), maximum(pick(r -> r.exact.ref.pivots)),
            maximum(pick(r -> r.exact.ref.pricing_passes)), maximum(pick(r -> r.exact.time)), maximum(pick(r -> r.sorted.time)),
            maximum(pick(r -> r.exact.ref.certificate_violations)))
    @printf("    underestimate of E_P - E_* by the sort: up to %.2e of it; max |P_*,exact - P_*,sorted| up to %.2e Pa\n",
            maximum(pick(r -> r.exact.ref.sorted_energy_excess / r.exact.ΔE)), maximum(pick(r -> r.Δp)))
    @printf("    roughness in time of E_P - E_*: sort %.3f, exact %.3f; of P_* (worst row): sort %.3f, exact %.3f\n",
            roughness(pick(r -> r.sorted.ΔE)), roughness(pick(r -> r.exact.ΔE)),
            maximum(j -> roughness(pick(r -> r.sorted.ref.p_star[j])), 2:length(rows[1].sorted.ref.p_star)-1),
            maximum(j -> roughness(pick(r -> r.exact.ref.p_star[j])), 2:length(rows[1].exact.ref.p_star)-1))
end

#####
##### 3. The weights under strong thermobaricity, at full size
#####

println("\n3. The N_*⁻² weights, two density-compensated families, γ = 1.3e-4 Pa⁻¹")
function weights_errors(; K, per, γ = 1.3e-4, ε = 0.02, method = ExactRearrangement())
    grid = Box2D(; Mx = 2per, Mz = K, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))
    tank = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)
    families = NonlinearBinaryFluid(:potential_temperature, (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1 / ρ0, α_T = 2e-4,
        α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 0.0, γ))
    model = AN2D((; physics..., mgr = nothing, domain = grid, space = tank, fluid = families,
                  boundary_conditions = BoxBoundaryConditions(grid; bottom = closed, top = closed),
                  buoyancy_scheme = ThermodynamicForcing(), anelastic_reference = AnelasticReference(grid, tank, z -> ρ0; pref = 1e5),
                  gauge_offset = 0.0))
    (; α_T, α_q, p0) = families
    Hx, Hz = halo_size(grid)
    p_mid = model.reference.p_R[Hz+K÷2]
    family_A(i) = i - Hx <= per
    rank(i, j) = (j - Hz - 1) * per + mod(i - Hx - 1, per)
    θ(i, j) = T0 + 4 * (1 - (rank(i, j) + 0.5) / (K * per)) + (family_A(i) ? 0 : 3)
    q(i, j) = family_A(i) ? q0 : q0 + α_T * (1 + γ * (p_mid - p0)) * 3 / α_q
    function state(δ)
        m = zeros(ncells(grid, 1), ncells(grid, 2), 2)
        for i in interior_cells(grid, 1), j in interior_cells(grid, 2)
            m[i, j, 1], m[i, j, 2] = ρ0 * (θ(i, j) + (family_A(i) ? δ : 0)), ρ0 * q(i, j)
        end
        return (; m)
    end
    consvar_rate, q_rate = zeros(ncells(grid, 1), ncells(grid, 2)), zeros(ncells(grid, 1), ncells(grid, 2))
    for i in interior_cells(grid, 1), j in interior_cells(grid, 2)
        consvar_rate[i, j] = family_A(i) ? ρ0 : 0.0
    end
    ref, ref₊, ref₋ = (method_reference(model, method) for _ in 1:3)
    reference_state!(ref₊, model, state(ε))
    reference_state!(ref₋, model, state(-ε))
    difference = (ref₊.slot_volume .- ref₋.slot_volume) ./ 2ε
    reference_state!(ref, model, state(0.0))
    reference_tendency!(ref, model, consvar_rate, q_rate)
    inner = K÷8+1:K-K÷8
    scale = sum(abs, difference[inner]) / length(inner)
    error(mean) = sqrt(sum(abs2, (mean .- difference)[inner]) / length(inner)) / scale
    return (; weighted = error(ref.slot_volume_tendency), plain = error(ref.slot_means.plain_volume_rate),
            pivots = ref.pivots, passes = ref.pricing_passes, violations = ref.certificate_violations)
end
for (K, per) in ((20, 5), (40, 5), (80, 10))
    r = weights_errors(; K, per)
    @printf("  K = %2d, %4d parcels: error of the weighted mean %.3f, of the plain mean %.3f (exact: %d pivots, %d passes, %d violations)\n",
            K, 2per * K, r.weighted, r.plain, r.pivots, r.passes, r.violations)
end

#####
##### 4. Where Float32 stops seeing the reordering
#####

println("\n4. Float32 against Float64: pivots that move mass, on an active state, against γ")
active_θ(x, z) = T0 + z / 0.5 + 0.3 * sin(2π * x) * cos(2π * z) + 0.1 * cos(4π * x + 1) * sin(4π * z)
active_q(x, z) = q0 - 0.002 * z / 0.5 + 0.001 * cos(2π * x + 0.5) * cos(2π * z)
for γ in (1e-6, 1e-5, 1e-4, 3e-4, 1e-3)
    counts = map((Float32, Float64)) do F
        grid = Box2D(; Mx = 32, Mz = 16, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))
        tank = Tank2D(Lx = F(1), Lz = F(0.5), g = F(10))
        fluid_F = NonlinearBinaryFluid(:potential_temperature, map(F, (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1 / ρ0, α_T = 2e-4,
            α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ)))
        model = AN2D((; mgr = nothing, domain = grid, space = tank, fluid = fluid_F, spatial_scheme = Centred2(),
                      vortex_scheme = VortexEnergyCons(), thermoflux_scheme = ThermoFluxConsistent(k_T = F(1e-4), k_q = F(3e-5)),
                      viscosity_scheme = ViscosityScheme(dyn_visc = F(1e-3)),
                      boundary_conditions = BoxBoundaryConditions(grid; bottom = closed, top = closed),
                      buoyancy_scheme = ThermodynamicForcing(), anelastic_reference = AnelasticReference(grid, tank, z -> F(ρ0); pref = F(1e5)),
                      gauge_offset = F(0)))
        m = zeros(F, ncells(grid, 1), ncells(grid, 2), 2)
        for i in axes(m, 1), j in axes(m, 2)
            x, z = (i - 1.5) / 32, (j - 1.5) * 0.5 / 16
            m[i, j, 1], m[i, j, 2] = F(ρ0 * active_θ(x, z)), F(ρ0 * active_q(x, z))
        end
        ref = method_reference(model, ExactRearrangement())
        reference_state!(ref, model, (; m))
        (ref.workspace.mass_pivots, ref.sorted_energy_excess)
    end
    @printf("  γ = %.0e Pa⁻¹: Float32 %3d pivots moving mass (excess %.2e J m⁻¹), Float64 %3d (excess %.2e J m⁻¹)\n",
            γ, counts[1][1], counts[1][2], counts[2][1], counts[2][2])
end
