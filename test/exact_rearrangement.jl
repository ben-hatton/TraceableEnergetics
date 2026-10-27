# Tests of the exact rearrangement of the Lorenz reference state (`ExactRearrangement`, the transportation simplex from
# the sorted placement) for AN2D and FC2D, on small grids. Included in the module of the reference-state tests, after
# test/reference_tendency.jl, whose models, states and helpers it uses.
using TraceableEnergetics: reference_tendency!

exact(model::AN2D; changes...) = AnelasticLorenzReference(model, ExactRearrangement(; changes...))
exact(model::FC2D; changes...) = CompressibleLorenzReference(model, ExactRearrangement(; changes...))

# the masses of the placement of `ref` in each parcel and each slot
parcel_masses(ref) = [sum(e -> ref.placement.parcel[e] == n ? ref.placement.mass[e] : zero(eltype(ref.mass)), 1:ref.placement.count)
                      for n in eachindex(ref.mass)]
slot_masses(ref) = [sum(e -> ref.placement.slot[e] == k ? ref.placement.mass[e] : zero(eltype(ref.mass)), 1:ref.placement.count)
                    for k in eachindex(ref.slot_mass)]

# Without thermobaricity (γ = 0) the sort is the optimum: the exact rearrangement moves no mass (its pivots, if any, are
# degenerate, from the duals of its tree), and gives the same E_P - E_* and profile
@testset "exact rearrangement, separable EOS, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64; fluid = reference_fluid(Float64; γ = 0))
    sorted, ref = lorenz(model), exact(model)
    energy_sorted = reference_energy!(sorted, model, active(model))
    energy = reference_energy!(ref, model, active(model))
    @test ref.certificate_violations == 0
    @test ref.workspace.mass_pivots == 0
    @test ref.sorted_energy_excess == 0
    @test energy.ΔE ≈ energy_sorted.ΔE rtol = 1e-14
    @test ref.p_star ≈ sorted.p_star rtol = 1e-14
end

# The brute force on N = 8 parcels of equal mass in 4 rows of 2 places each, with strong thermobaricity and
# density-compensated parcels, where the sort is not the optimum: the exact rearrangement finds it, certified
@testset "exact rearrangement, brute force, AN2D" begin
    γ = 3e-4
    model = an_box(Float64; Mx = 2, Mz = 4, ρ = z -> ρ0, fluid = reference_fluid(Float64; γ))
    p_key = model.reference.p_R[1 + 2]
    θ(n) = T0 + 4 * noise(n)
    (; α_T, α_TT, α_q, p0) = model.fluid
    q(n) = q0 + (α_T * (1 + γ * (p_key - p0)) * (θ(n) - T0) + α_TT * (θ(n) - T0)^2 / 2) / α_q + 1e-6 * noise(n + 100)
    index(x, z) = round(Int, x / model.dx + 0.5) + 2 * round(Int, z / model.dz - 0.5)
    state = tracer_state(model, (x, z) -> θ(index(x, z)), (x, z) -> q(index(x, z)))
    sorted, ref = lorenz(model), exact(model)
    energy_sorted = reference_energy!(sorted, model, state)
    energy = reference_energy!(ref, model, state)
    C = [anelastic_cost(model, ref, n, (u - 1) ÷ 2 + 1) for n in 1:8, u in 1:8]
    own = sum(n -> anelastic_cost(model, ref, n, ref.row[n]), 1:8)
    @test energy.ΔE ≈ ref.mass[1] * (own - least_assignment(C)) rtol = 1e-12
    @test ref.certificate_violations == 0
    @test sorted.certificate_violations > 0
    # the sort underestimates E_P - E_* by the excess
    @test energy.ΔE - energy_sorted.ΔE ≈ ref.sorted_energy_excess rtol = 1e-10
end

# Strong thermobaricity (γ = 3e-4 Pa⁻¹) on the active state: the sort fails its certificate, the exact rearrangement
# passes it, releases more energy (E_*,exact ≤ E_*,sorted) by the excess it reports, conserves the mass of every parcel
# and every slot, is invariant under a periodic shift of the parcels in x, and in AN2D satisfies weak duality with
# equality for its duals on the row centres. In FC2D the column fills the box
@testset "exact rearrangement, thermobaric, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64; Mx = 16, Mz = 16, fluid = reference_fluid(Float64; γ = 3e-4))
    state = active(model)
    sorted, ref = lorenz(model), exact(model)
    energy_sorted = reference_energy!(sorted, model, state)
    energy = reference_energy!(ref, model, state)
    @test sorted.certificate_violations > 0
    @test ref.certificate_violations == 0
    @test ref.pivots > 0
    @test energy.ΔE > energy_sorted.ΔE
    # (FC2D: the excess is a difference of the energies of the column G = Σ x c - P_top L_x L_z, of round-off about
    # eps P_top L_x L_z, 1e-11 J m⁻¹ here)
    @test energy.ΔE - energy_sorted.ΔE ≈ ref.sorted_energy_excess rtol = model isa AN2D ? 1e-8 : 1e-6
    @test energy.ΣmAE >= energy.ΔE
    @test all(>=(0), ref.placement.mass[1:ref.placement.count])
    @test parcel_masses(ref) ≈ ref.mass rtol = 1e-14
    @test slot_masses(ref) ≈ ref.slot_mass rtol = 1e-12
    if model isa AN2D
        on_rows = sum(n -> ref.mass[n] * (anelastic_cost(model, ref, n, ref.row[n]) - ref.u[n] - ref.slot_dual[ref.row[n]]),
                      eachindex(ref.mass))
        @test on_rows ≈ energy.ΔE rtol = 1e-8
    else
        @test abs(ref.volume_residual) < 1e-14
        @test 1 < ref.top_pressure_iterations <= ref.method.max_top_iterations
    end
    columns = interior_cells(model.domain, 1)
    shifted = (; m = copy(state.m))
    shifted.m[columns, :, :] .= circshift(state.m[columns, :, :], (3, 0, 0))
    ref_shifted = exact(model)
    energy_shifted = reference_energy!(ref_shifted, model, shifted)
    @test energy_shifted.ΔE ≈ energy.ΔE rtol = 1e-10
    @test ref_shifted.p_star ≈ ref.p_star rtol = 1e-12
end

# Degenerate states: a single column, and a fully mixed fluid (every cost tied); the placement is certified and conserves
# the masses
@testset "exact rearrangement, degenerate, $M, $label" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box)),
                                                           (label, Mx, fields) in (("Mx = 1", 1, (θ_active, q_active)),
                                                                                   ("mixed", 8, ((x, z) -> T0, (x, z) -> q0)))
    model = make(Float64; Mx, fluid = reference_fluid(Float64; γ = 3e-4))
    ref = exact(model)
    energy = reference_energy!(ref, model, tracer_state(model, fields...))
    @test ref.certificate_violations == 0
    @test parcel_masses(ref) ≈ ref.mass rtol = 1e-14
    @test slot_masses(ref) ≈ ref.slot_mass rtol = 1e-12
    @test all(isfinite, energy.AE) && all(>=(0), interior(model.domain, energy.AE))
end

# Where Float32 breaks: the reduced costs that reorder the parcels under thermobaricity, about 1e-4 J kg⁻¹ here, are below
# the tolerance of the certificate in Float32, 16 (K + 1) eps g L_z (about 2e-4 J kg⁻¹ for K = 16): the sorted placement
# that fails the certificate in Float64 passes it in Float32, and the exact rearrangement keeps it
@testset "exact rearrangement, Float32, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model32, model64 = (make(F; Mx = 16, Mz = 16, fluid = reference_fluid(F; γ = 3e-4)) for F in (Float32, Float64))
    state64 = active(model64)
    ref32, ref64 = exact(model32), exact(model64)
    reference_state!(ref32, model32, (; m = Float32.(state64.m)))
    reference_state!(ref64, model64, state64)
    @test ref64.sorted_energy_excess > 0
    @test ref32.certificate_violations == 0
    @test ref32.sorted_energy_excess == 0
end

# The pass cap scales with the slots, as the passes do: 10 per slot by default
@testset "exact rearrangement, pass cap, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64; Mx = 16, Mz = 16, fluid = reference_fluid(Float64; γ = 3e-4))
    ref = exact(model)
    @test ref.method.passes_per_slot == 10
    reference_state!(ref, model, active(model))
    @test ref.certificate_violations == 0
    @test ref.pricing_passes <= ref.method.passes_per_slot * length(ref.slot_mass)
end

# The same exact rearrangement with a vectorised, threaded manager as with a serial one, and no allocation after its
# first call (measured inside a function, so that the arguments are concrete)
state_allocations(ref, model, state) = @allocated reference_state!(ref, model, state)
@testset "exact rearrangement, managers and allocations, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    fluid = reference_fluid(Float64; γ = 3e-4)
    serial, threaded = make(Float64; Mx = 16, Mz = 16, fluid), make(Float64; Mx = 16, Mz = 16, fluid, mgr)
    state = active(serial)
    ref_serial, ref_threaded = exact(serial), exact(threaded)
    reference_state!(ref_serial, serial, state)
    reference_state!(ref_threaded, threaded, state)
    @test ref_threaded.placement.mass == ref_serial.placement.mass
    @test ref_threaded.p_star == ref_serial.p_star
    state_allocations(ref_serial, serial, state)
    @test state_allocations(ref_serial, serial, state) == 0
end

# The N_*⁻²-weighted means of the slots under strong thermobaricity: two density-compensated families of parcels at the
# mid-depth pressure (family B 3 K warmer and saltier than family A), mixed in every row, with different ∂v/∂p, of which
# only family A is heated. The derivative of the slot volumes v_*,k of the exact rearrangement, by centred differences,
# includes the reordering of the parcels; the weighted mean ⟨v̇⟩_k follows it (error 8.5 % at K = 40, 5.7 % at K = 80),
# and the plain mean does not (35 %). The sort, at a fixed placement, gives the plain mean
@testset "weights under thermobaricity, AN2D" begin
    γ, K, per = 1.3e-4, 40, 5
    model = an_box(Float64; Mx = 2per, Mz = K, ρ = z -> ρ0, fluid = reference_fluid(Float64; γ, α_TT = 0))
    (; α_T, α_q, p0) = model.fluid
    (; domain) = model
    Hx, Hz = halo_size(domain)
    p_mid = model.reference.p_R[Hz+K÷2]
    family_A(i) = i - Hx <= per
    rank(i, j) = (j - Hz - 1) * per + mod(i - Hx - 1, per)          # rank of the parcel in its family, 0 .. K per - 1
    θ(i, j) = T0 + 4 * (1 - (rank(i, j) + 0.5) / (K * per)) + (family_A(i) ? 0 : 3)
    q(i, j) = family_A(i) ? q0 : q0 + α_T * (1 + γ * (p_mid - p0)) * 3 / α_q
    function state(ε)
        m = zeros(ncells(domain, 1), ncells(domain, 2), 2)
        for i in interior_cells(domain, 1), j in interior_cells(domain, 2)
            m[i, j, 1] = ρ0 * (θ(i, j) + (family_A(i) ? ε : 0))
            m[i, j, 2] = ρ0 * q(i, j)
        end
        return (; m)
    end
    # ρ Dϑ/Dt = ρ_R for family A, zero for B
    consvar_rate, q_rate = cell_array(model), cell_array(model)
    for i in interior_cells(domain, 1), j in interior_cells(domain, 2)
        consvar_rate[i, j] = family_A(i) ? ρ0 : 0
    end
    ε = 0.02
    inner = K÷8+1:K-K÷8
    for (ref, ref₊, ref₋) in ((exact(model), exact(model), exact(model)), (lorenz(model), lorenz(model), lorenz(model)))
        reference_state!(ref₊, model, state(ε))
        reference_state!(ref₋, model, state(-ε))
        difference = (ref₊.slot_volume .- ref₋.slot_volume) ./ 2ε
        reference_state!(ref, model, state(0))
        reference_tendency!(ref, model, consvar_rate, q_rate)
        scale = sum(abs, difference[inner]) / length(inner)
        error(mean) = sqrt(sum(abs2, (mean .- difference)[inner]) / length(inner)) / scale
        if ref.method isa ExactRearrangement
            @test ref.certificate_violations == 0
            @test error(ref.slot_volume_tendency) < 0.1
            @test error(ref.slot_means.plain_volume_rate) > 0.3
        else
            @test error(ref.slot_means.plain_volume_rate) < 1e-6
        end
    end
end
