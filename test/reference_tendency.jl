# Tests of the tendency of the Lorenz reference state from the diabatic tendencies (src/reference_tendency.jl) and of the
# terms in ∂ₜP_* of the budgets of ACE and APE (src/available_energy_budget.jl), for AN2D and FC2D, on small grids.
# Included in the module of the reference-state tests, after test/available_energy_budget.jl, whose models, states and
# helpers it uses.
using TraceableEnergetics: reference_tendency!, reference_pressure_tendency, reference_pressure_tendency_at!,
                           displacement_reference_term!, slot_means!, SlotMeans, consvar_index

# the active tracers of config.jl at rest: only diffusion changes them, so their full tendency is diabatic
rest_fields = (; consvar = θ_active, q = q_active, u = z -> 0.0, ψ = (X, Z) -> 0.0)
make_state(model::AN2D, fields) = initialize(model, fields)
make_state(model::FC2D, fields) = initialize(model, fields; pref = 1e5)

# the tendencies of a state and its diabatic rates ρ Dϑ/Dt = σ_ϑ - ∇·J_ϑ, ρ Dq/Dt = -∇·J_q on the cells
function diabatic_rates(model, state)
    dstate, scratch = tendencies!(void, void, model, state, zero(model.dx))
    session = open(diagnostics(model); state, dstate, scratch, model)
    rates = (CookBooks.get(session, :conservative_variable_production) .- CookBooks.get(session, :div_consflux),
             .-CookBooks.get(session, :div_qflux))
    close(session)
    return rates, dstate, scratch
end

# the state whose densities ρϑ and ρq are moved by ε times the rates, at the same ρ
function perturbed_state(model, state, (consvar_rate, q_rate), ε)
    m = copy(state.m)
    k = consvar_index(model)
    m[:, :, k] .+= ε .* consvar_rate
    m[:, :, k+1] .+= ε .* q_rate
    return (; m)
end

# P_* at the height z of a reference state: P_R + ρ_R π_* for AN2D, with ρ_R = -∂_z P_R/g
reference_pressure(ref::AnelasticLorenzReference, model, z) =
    hermite(ref.pressure, z)[1] - hermite(ref.pressure, z)[2] / model.space.g * hermite(ref.potential, z)[1]
reference_pressure(ref::CompressibleLorenzReference, model, z) = hermite(ref.pressure, z)[1]

# heights over the whole box, with both walls and points next to them
test_heights = vcat(range(0.0, Lz, 37), [1e-3, 0.02, Lz - 0.02, Lz - 1e-3])

# Without thermobaricity (γ = 0), ∂v/∂p is the same for all the parcels of a slot, so the N_*⁻²-weighted means are the
# plain ones, which are the derivative of the reference state at a fixed placement: the tendency matches the centred
# difference of the reference state of the state moved by ±ε times the diabatic rates, for an ε small enough that no
# parcel changes slot (the order of the placement is the same). In FC2D the differences of P_* are limited by the
# round-off of P_top, whose Newton iteration stops at |R| ≤ 16 eps L_x L_z (about 4e-9 Pa here)
@testset "tendency of the reference state against differences, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64; fluid = reference_fluid(Float64; γ = 0))
    state = make_state(model, rest_fields)
    rates, _, _ = diabatic_rates(model, state)
    ref = reference_tendency!(reference_state!(lorenz(model), model, state), model, rates...)
    ε = 0.01
    r₊ = reference_state!(lorenz(model), model, perturbed_state(model, state, rates, ε))
    r₋ = reference_state!(lorenz(model), model, perturbed_state(model, state, rates, -ε))
    @test r₊.order == r₋.order == ref.order
    difference(f) = (f(r₊) .- f(r₋)) ./ 2ε
    @test ref.plain_weight_slots == 0
    @test difference(r -> r.slot_volume) ≈ ref.slot_volume_tendency rtol = 1e-6
    scale = maximum(abs, ref.p_star_tendency)
    @test scale > 0
    @test maximum(abs, difference(r -> r.p_star) .- ref.p_star_tendency) <= 1e-5 * scale
    at_heights = difference(r -> [reference_pressure(r, model, z) for z in test_heights])
    @test maximum(abs, at_heights .- [reference_pressure_tendency(ref, model, z) for z in test_heights]) <= 1e-5 * scale
    if model isa FC2D
        @test difference(r -> r.p_top) ≈ ref.p_top_tendency rtol = 1e-5
        @test difference(r -> r.pressure.z) ≈ ref.edge_height_tendency atol = 1e-6 * maximum(abs, ref.edge_height_tendency)
    else
        # the nodes of π_*, both with a zero mean over the mass of the rows
        @test difference(r -> r.potential.y) ≈ ref.potential_tendency.y atol = 1e-6 * maximum(abs, ref.potential_tendency.y)
    end
end

# The tendencies ∂ₜ(ρΠ₁), ∂ₜ(ρΠ₂) of the diagnostics, with the terms R_Π₁, R_Π₂ in ∂ₜP_*, are those of ρΠ₁, ρΠ₂ with the
# reference state recomputed, at each cell: for a state at rest (no advection, ∂ₜρ = 0) the tendency of the state is
# diabatic, and the centred difference of ρΠ over ±ε times it matches them. Without the R terms, they would not
@testset "tendencies of ρΠ₁ and ρΠ₂ against differences, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64; fluid = reference_fluid(Float64; γ = 0))
    state = make_state(model, rest_fields)
    rates, dstate, scratch = diabatic_rates(model, state)
    session = open(diagnostics(model; reference_state = lorenz(model)); state, dstate, scratch, model)
    f(name) = interior_field(session, name)
    ε = 0.01
    e₊ = reference_energy!(lorenz(model), model, perturbed_state(model, state, rates, ε))
    e₋ = reference_energy!(lorenz(model), model, perturbed_state(model, state, rates, -ε))
    ρ = interior(model.domain, model isa AN2D ? repeat(model.reference.ρ_R', size(state.m, 1)) : state.m[:, :, 1])
    for (E, R) in ((:ACE, :reference_pressure_term_ACE), (:APE, :reference_pressure_term_APE))
        difference = ρ .* (interior(model.domain, getproperty(e₊, E)) .- interior(model.domain, getproperty(e₋, E))) ./ 2ε
        tendency = f(Symbol(:dρ, E))
        scale = maximum(abs, f(:dρAPE))
        @test maximum(abs, difference .- tendency) <= 1e-4 * scale
        # the term in ∂ₜP_* matters: without it the difference is not matched
        E == :APE && @test maximum(abs, difference .- (tendency .- f(R))) > 1e-2 * scale
    end
    close(session)
end

# The budgets of ACE and APE with the terms in ∂ₜP_* close at each cell to round-off, as at a fixed reference state
@testset "closure of the budgets of ACE and APE with ∂ₜP_*, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64; Mx = 16)
    session = budget_session(model)
    f(name) = interior_field(session, name)
    for E in (:ACE, :APE)
        @test maximum(abs, f(Symbol(E, :_budget_residual))) <= 1e-9 * maximum(abs, f(:dρAPE))
    end
    @test maximum(abs, f(:reference_pressure_term_APE)) > 0
    model isa AN2D && @test all(iszero, f(:reference_pressure_term_ACE))
    close(session)
end

# FC2D: the column still fills the box, Σ_k M_k ∂ₜv_*,k = 0, so the bottom edge stays at the bottom and the top edge at
# the top, in both float types
@testset "the reference column fills the box, FC2D, $F" for F in (Float32, Float64)
    model = fc_box(F)
    state = make_state(model, rest_fields)
    rates, _, _ = diabatic_rates(model, state)
    ref = reference_tendency!(reference_state!(lorenz(model), model, state), model, rates...)
    @test abs(sum(ref.slot_mass .* ref.slot_volume_tendency)) <= 16 * eps(F) * sum(abs, ref.slot_mass .* ref.slot_volume_tendency)
    @test ref.edge_height_tendency[1] == 0
    @test abs(ref.edge_height_tendency[end]) <= 16 * eps(F) * maximum(abs, ref.edge_height_tendency)
end

# Advection rearranges the fluid without changing the distribution of mass over (ϑ, q): without diffusion or viscous
# heating, the diabatic rates are zero, and so are the tendency of the reference state and the terms in ∂ₜP_*, although
# the flow changes the available energy
@testset "no diffusion, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64; k_T = 0, k_q = 0, dyn_visc = 0)
    session = budget_session(model)
    f(name) = interior_field(session, name)
    @test all(iszero, CookBooks.get(session, :p_star_tendency))
    @test all(iszero, f(:p_star_tendency_at_z_star))
    @test all(iszero, f(:reference_pressure_term_ACE)) && all(iszero, f(:reference_pressure_term_APE))
    @test maximum(abs, f(:dρAPE)) > 0
    close(session)
end

# The terms in ∂ₜP_* exchange energy between the parcels but none with the rest: ∫R_Π dV = ∫∂ₜP_*(z) dV -
# Σ_n m_n Σ_p(z_*) ∂ₜP_*(z_*) vanishes when the reference heights of the parcels fill the box as their placement does, to
# the resolution of the slots: the reference heights of the parcels of a slot that holds a mixture differ from their
# placement by up to a slot thickness, so it converges at first order as the grid is refined, |∫R_Π|/∫|R_Π| = 2.0e-2,
# 1.2e-2, 5.8e-3, 2.3e-3 (AN2D, where R_Π = R_Π₂) and 3.4e-2, 1.1e-2, 4.2e-3, 1.6e-3 (FC2D) at M_z = 8, 16, 32, 64
@testset "∫R_Π dV = 0 to the slot resolution, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    errors = map((8, 16, 32)) do n
        model = make(Float64; Mx = n, Mz = n)
        session = budget_session(model)
        R = CookBooks.get(session, :reference_pressure_term_ACE) .+ CookBooks.get(session, :reference_pressure_term_APE)
        error = abs(domain_integral(model, R)) / domain_integral(model, abs.(R))
        close(session)
        error
    end
    @test errors[1] < 0.1
    @test errors[1] / errors[3] > 3
end

# The weighted means are convex combinations of the parcels' values, whatever the stratification: ⟨v̇⟩_k lies between the
# smallest and largest v̇ of the parcels of slot k. With thermobaricity (γ ≠ 0) the weights differ between the parcels of
# a slot, and the weighted means differ from the plain ones; without, they are the same. A nearly mixed fluid, with
# N² of round-off size and either sign, takes plain weights in its unstable slots, and stays finite
@testset "weighted means of the slots, $M, $label" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box)),
                                                        (label, γ, θ, q) in (("separable", 0, θ_active, q_active),
                                                                             ("thermobaric", 1e-4, θ_active, q_active),
                                                                             ("nearly mixed", 1e-7, (x, z) -> T0 + 1e-6 * θ_active(x, z), (x, z) -> q0))
    model = make(Float64; fluid = reference_fluid(Float64; γ))
    state = make_state(model, (; consvar = θ, q, u = z -> 0.0, ψ = (X, Z) -> 0.0))
    rates, _, _ = diabatic_rates(model, state)
    ref = reference_tendency!(reference_state!(lorenz(model), model, state), model, rates...)
    (; placement, slot_means) = ref
    K = length(ref.slot_mass)
    lowest, highest = fill(Inf, K), fill(-Inf, K)
    for e in 1:placement.count
        n, k = placement.parcel[e], placement.slot[e]
        _, _, ∂v_∂ϑ, ∂v_∂q = model.fluid(:p, :consvar, :q).volume_functions(ref.slot_pressure[k], ref.consvar[n], ref.q[n])
        v̇ = ∂v_∂ϑ * ref.consvar_rate[n] + ∂v_∂q * ref.q_rate[n]
        lowest[k], highest[k] = min(lowest[k], v̇), max(highest[k], v̇)
    end
    margin = 1e-12 * maximum(abs, vcat(lowest, highest))
    @test all(lowest .- margin .<= slot_means.volume_rate .<= highest .+ margin)
    @test all(isfinite, ref.p_star_tendency)
    if label == "separable"
        @test slot_means.volume_rate ≈ slot_means.plain_volume_rate rtol = 1e-12
    elseif label == "thermobaric"
        @test maximum(abs, slot_means.volume_rate .- slot_means.plain_volume_rate) > 1e-6 * maximum(abs, slot_means.plain_volume_rate)
    end
end

# A slot with a parcel of N² ≤ 0 takes the plain means and is counted
@testset "slots with N² ≤ 0" begin
    model = an_box(Float64)
    state = make_state(model, rest_fields)
    rates, _, _ = diabatic_rates(model, state)
    ref = reference_tendency!(reference_state!(lorenz(model), model, state), model, rates...)
    (; slot_means) = ref
    # make slot 3 unstable for all its parcels: ∂_z v_* below -g ρ ∂v/∂p
    slot_means.volume_gradient[3] = -1
    count = slot_means!(slot_means, ref.placement, ref.slot_mass, ref.slot_pressure, model.fluid, model.space.g, ref.consvar,
                        ref.q, ref.consvar_rate, ref.q_rate)
    @test count == 1
    @test slot_means.unstable == (1:length(ref.slot_mass) .== 3)
    @test slot_means.volume_rate[3] == slot_means.plain_volume_rate[3]
end

# AN2D: the constant of ∂ₜπ_* is free; adding one to it changes ∂ₜP_* = ρ_R ∂ₜπ_*, but not R_Π₂
@testset "constant of ∂ₜπ_*, AN2D" begin
    model = an_box(Float64)
    state = make_state(model, rest_fields)
    rates, _, _ = diabatic_rates(model, state)
    ref = reference_tendency!(reference_state!(lorenz(model), model, state), model, rates...)
    z_star = reference_heights!(cell_array(model), model, ref)
    R₂(ref) = displacement_reference_term!(cell_array(model), model, ref, z_star,
                                           reference_pressure_tendency_at!(cell_array(model), model, ref, z_star))
    before = R₂(ref)
    c = 1e-3
    ref.potential_tendency.y .+= c
    ref.p_star_tendency .+= c .* model.reference.ρ_R
    after = R₂(ref)
    @test maximum(abs, after .- before) <= 1e-12 * c * ρ0
    @test maximum(abs, before) > 0
end

# Allocations: none after the first call, for either model
allocations(ref, model, consvar_rate, q_rate) = @allocated reference_tendency!(ref, model, consvar_rate, q_rate)
@testset "allocations of reference_tendency!, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64)
    state = make_state(model, rest_fields)
    rates, _, _ = diabatic_rates(model, state)
    ref = reference_state!(lorenz(model), model, state)
    allocations(ref, model, rates...)
    @test allocations(ref, model, rates...) == 0
end

# Float32 against Float64, from the same diabatic rates (those of the Float64 model: the Poisson solver of AN2D fails in
# Float32, as an FFTW plan is applied to a misaligned array): the tendency of P_* is a difference of the volume changes of
# the parcels (heat and salt are conserved, so ∂ₜP_top comes from the nonlinearity of the equation of state alone), and
# Float32 keeps about 3 digits of it on these grids
@testset "precision of the tendency of the reference state, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    rates, _, _ = let model = make(Float64)
        diabatic_rates(model, make_state(model, rest_fields))
    end
    tendency(F) = let model = make(F), state = make_state(model, rest_fields)
        reference_tendency!(reference_state!(lorenz(model), model, state), model, map(r -> F.(r), rates)...).p_star_tendency
    end
    single, double = tendency(Float32), tendency(Float64)
    @test eltype(single) == Float32
    @test maximum(abs, single .- double) <= 1e-2 * maximum(abs, double)
end
