# Studies of the tendency ∂ₜP_* of the Lorenz reference state from the diabatic tendencies (src/reference_state/tendency.jl),
# on a stratified state at rest with lateral structure, which diffusion changes.
#
# 1. FC2D: ∂ₜP_* at fixed height, as the derivative of the monotone cubic P_* with respect to the motion of its nodes (dual
#    numbers), against the centred difference of the reference state of the state moved by ±ε times the diabatic rates,
#    with γ = 0 so that the weighted means are the plain ones (the derivative at a fixed placement). Three methods were
#    compared on 2026-10-27 (decided with the user: dual numbers):
#      - dual numbers and a hand-written tangent of the interpolant agreed to 1e-16 and matched the differences to their
#        floor wherever the placement did not change (1e-8 relative at M_z = 16, ε = 0.1; 2e-5 at M_z = 32 and 64,
#        ε = 1e-4, the round-off of P);
#      - the continuous formula ∂ₜP_top + g ρ_*(z) ∂ₜz|_m, linear in z between the edges, is not the derivative of the
#        P_* of Π: 7.0e-3, 5.3e-3, 3.6e-3 of max|∂ₜP_*| at M_z = 16, 32, 64;
#      - all cost 16 to 23 ns per height; reference_tendency! takes 6 μs at M_z = M_x = 32 and 88 μs at 128 (reference_state!:
#        190 μs and 9.5 ms), with no allocations.
#    Nearly equal parcels change slots for ε as small as 1e-3 at M_z = 64, so the differences need ε ≤ 1e-4 there; P_top
#    is uncertain by about 16 eps L_x L_z/|Σ x ∂v/∂p| (its Newton iteration), so the errors are of ∂ₜ(P_* - P_top).
# 2. AN2D: the change of the compressibility b_e of the end slots, neglected between the end rows and the walls, where
#    π_*(z) = π_*(z_e) - v_e ΔP - b_e ΔP²/2 - g (z - z_e): the neglected term of ∂ₜπ_* is -∂ₜb_e ΔP²/2, with
#    ∂ₜb_e = Σ_n x_ne (∂²v/∂p∂ϑ ϑ̇_n + ∂²v/∂p∂q q̇_n)/M_e, which for NonlinearBinaryFluid is v0 α_T γ ⟨ϑ̇⟩_e (∂v/∂p is
#    linear in ϑ and independent of q). Measured against the centred difference at the walls, and its effect on R_Π₂, for
#    γ from that of the tests (1e-7 Pa⁻¹) to 3e-4 Pa⁻¹. Results (2026-10-27): the difference less the formula at the walls
#    is the neglected term to 3 or 4 digits; relative to max|∂ₜπ_*| it is 1e-6 at γ = 1e-7, 1e-4 at 1e-5, 9e-4 at 1e-4
#    and 2.6e-3 at 3e-4 (M_z = 16), and 5 times smaller at M_z = 64; it changes R_Π₂ by at most as much, only for the
#    parcels whose z_* is in the end half rows, and ∫R_Π₂ by at most 1.3e-8 W m⁻¹ against ∫|R_Π₂| = 8.7e-5 W m⁻¹.
#
#   julia -t 4 --project=. tools/reference_tendency_study.jl
using TraceableEnergetics, CFCompressible, CFAnelastics, ClimFluids, CFBoxes, CFDiffusionSchemes, CookBooks
using CFPlanets: Tank2D
using CFBoxes: cellpoint, interior_cells, interior
using CFTimeSchemes: tendencies!
using MutatingOrNot: void
using Printf
using TraceableEnergetics: reference_tendency!, reference_pressure_tendency, reference_pressure_tendency_at!,
                           displacement_reference_term!, reference_heights!, hermite, consvar_index, domain_integral

const Lx, Lz, g, T0, q0, ρ0 = 1.0, 0.5, 10.0, 280.0, 0.035, 1000.0
fluid_params(F; γ) = map(F, (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1e-3, α_T = 2e-4, α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2,
                             R = 8.31446261815324, α_TT = 1e-5, γ))
closed_wall() = Wall(; tangential = FreeSlip(), tracers = (Flux(0.0), Flux(0.0)))
schemes(F) = (; spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(), thermoflux_scheme = ThermoFluxConsistent(k_T = F(1e-4), k_q = F(1e-5)),
              viscosity_scheme = ViscosityScheme(dyn_visc = F(1e-3)))

function fc_box(F; Mx = 16, Mz = 16, γ = 0.0)
    domain = Box2D(; Mx, Mz)
    fluid = NonlinearBinaryFluid(:potential_temperature, fluid_params(F; γ))
    return FC2D((; mgr = nothing, domain, space = Tank2D(Lx = F(Lx), Lz = F(Lz), g = F(g)), fluid, schemes(F)...,
        boundary_conditions = BoxBoundaryConditions(domain; bottom = closed_wall(), top = closed_wall())))
end
function an_box(F; Mx = 16, Mz = 16, γ = 0.0)
    domain = Box2D(; Mx, Mz)
    space = Tank2D(Lx = F(Lx), Lz = F(Lz), g = F(g))
    fluid = NonlinearBinaryFluid(:potential_temperature, fluid_params(F; γ))
    return AN2D((; mgr = nothing, domain, space, fluid, schemes(F)..., buoyancy_scheme = ThermodynamicForcing(),
        anelastic_reference = AnelasticReference(domain, space, z -> F(ρ0 * (1 + 0.1 * (Lz - z) / Lz)); pref = F(1e5)),
        boundary_conditions = BoxBoundaryConditions(domain; bottom = closed_wall(), top = closed_wall()), gauge_offset = F(0)))
end
lorenz(model::AN2D) = AnelasticLorenzReference(model)
lorenz(model::FC2D) = CompressibleLorenzReference(model)

# a stratified state at rest with lateral structure
θ(x, z) = T0 + z / Lz + 0.3 * sin(2π * x / Lx) * cos(π * z / Lz) + 0.1 * cos(4π * x / Lx + 1) * sin(2π * z / Lz)
q(x, z) = q0 - 0.002 * z / Lz + 0.001 * cos(2π * x / Lx + 0.5) * cos(π * z / Lz)
fields = (; consvar = θ, q, u = z -> 0.0, ψ = (X, Z) -> 0.0)
make_state(model::AN2D) = initialize(model, fields)
make_state(model::FC2D) = initialize(model, fields; pref = 1e5)

# the state, and the diabatic rates ρ Dϑ/Dt = σ_ϑ - ∇·J_ϑ, ρ Dq/Dt = -∇·J_q on the cells
function state_and_rates(model)
    state = make_state(model)
    dstate, scratch = tendencies!(void, void, model, state, zero(model.dx))
    session = open(diagnostics(model); state, dstate, scratch, model)
    rates = (CookBooks.get(session, :conservative_variable_production) .- CookBooks.get(session, :div_consflux),
             .-CookBooks.get(session, :div_qflux))
    close(session)
    return state, rates
end

# the reference state of the state whose ρϑ, ρq are moved by ε times the rates
function perturbed_reference(model, state, (consvar_rate, q_rate), ε)
    m = copy(state.m)
    k = consvar_index(model)
    m[:, :, k] .+= ε .* consvar_rate
    m[:, :, k+1] .+= ε .* q_rate
    return reference_state!(lorenz(model), model, (; m))
end

# n heights spread over the box, the same in every run (a golden-ratio sequence)
spread_heights(F, n) = [F(Lz * mod(0.6180339887498949 * k + 0.1, 1.0)) for k in 1:n]

## 1. FC2D: ∂ₜP_* against differences

function fc_study(F; Mz = 16, ε = 1.0)
    model = fc_box(F; Mz, Mx = Mz)
    state, rates = state_and_rates(model)
    ref = reference_tendency!(reference_state!(lorenz(model), model, state), model, rates...)
    edges = ref.pressure.z
    sets = (; rows = [cellpoint(model.domain, model.dz, 2, j) for j in interior_cells(model.domain, 2)],
            spread = spread_heights(F, 200), walls = F.(vcat(range(0, edges[2], 9), range(edges[end-1], Lz, 9))))
    r₊, r₋ = perturbed_reference(model, state, rates, F(ε)), perturbed_reference(model, state, rates, -F(ε))
    top = (r₊.p_top - r₋.p_top) / 2F(ε)
    scale = maximum(abs, ref.p_star_tendency)
    @printf("\nFC2D %s, M_z = %d, ε = %g (placement %s): max|∂ₜP_*| = %.3e Pa/s, ∂ₜP_top %.6e (difference %.6e)\n", F, Mz, ε,
            r₊.order == r₋.order ? "fixed" : "changed", scale, ref.p_top_tendency, top)
    for (name, zs) in pairs(sets)
        difference = [(hermite(r₊.pressure, z)[1] - hermite(r₋.pressure, z)[1]) / 2F(ε) for z in zs]
        formula = [reference_pressure_tendency(ref, model, z) for z in zs]
        @printf("  %-7s error of ∂ₜ(P_* - P_top) / max|∂ₜP_*| = %.2e\n", name,
                maximum(abs, (formula .- ref.p_top_tendency) .- (difference .- top)) / scale)
    end
end

## 2. AN2D: the neglected change of the compressibility of the end slots

function wall_study(; Mz = 16, γ)
    model = an_box(Float64; Mz, Mx = Mz, γ)
    (; fluid) = model
    state, rates = state_and_rates(model)
    ref = reference_tendency!(reference_state!(lorenz(model), model, state), model, rates...)
    K = length(ref.slot_mass)
    ε = 1e-2
    r₊, r₋ = perturbed_reference(model, state, rates, ε), perturbed_reference(model, state, rates, -ε)
    scale = maximum(abs, ref.potential_tendency.y)
    @printf("\nAN2D, M_z = %d, γ = %.0e Pa⁻¹ (placement %s): max|∂ₜπ_*| = %.3e J kg⁻¹ s⁻¹\n", Mz, γ,
            r₊.order == r₋.order ? "fixed" : "changed", scale)
    neglected = zeros(2)
    for (side, (wall, row, e)) in enumerate(((1, 2, K), (K + 2, K + 1, 1)))
        # the increment of π_* from the end row to the wall, by differences and by the formula with the plain mean of the
        # end slot (the differences are at a fixed placement), and the neglected term -∂ₜb_e ΔP²/2
        difference = ((r₊.potential.y[wall] - r₊.potential.y[row]) - (r₋.potential.y[wall] - r₋.potential.y[row])) / 2ε
        ΔP = hermite(ref.pressure, ref.potential.z[wall])[1] - ref.slot_pressure[e]
        formula = -ref.slot_means.plain_volume_rate[e] * ΔP
        ḃ_e = sum(i -> ref.placement.slot[i] == e ? ref.placement.mass[i] * fluid.v0 * fluid.α_T * γ * ref.consvar_rate[ref.placement.parcel[i]] : 0.0,
                  1:ref.placement.count) / ref.slot_mass[e]
        neglected[side] = -ḃ_e * ΔP^2 / 2
        @printf("  %s wall: difference - formula = %.3e, -∂ₜb_e ΔP²/2 = %.3e, relative to max|∂ₜπ_*| %.2e, to the increment %.2e\n",
                side == 1 ? "bottom" : "top", difference - formula, neglected[side], abs(neglected[side]) / scale,
                abs(neglected[side] / formula))
    end
    # its effect on R_Π₂: add the neglected term to the nodes of ∂ₜπ_* at the walls, -∂ₜb_e ΔP²/2 to their values and
    # -∂ₜb_e ΔP ∂_z P_R to their slopes
    z_star = reference_heights!(zeros(size(state.m, 1), size(state.m, 2)), model, ref)
    R₂(ref) = displacement_reference_term!(similar(z_star), model, ref, z_star, reference_pressure_tendency_at!(similar(z_star), model, ref, z_star))
    before = R₂(ref)
    for (side, (wall, e)) in enumerate(((1, K), (K + 2, 1)))
        P_wall, ∂P_wall, _ = hermite(ref.pressure, ref.potential.z[wall])
        ΔP = P_wall - ref.slot_pressure[e]
        ref.potential_tendency.y[wall] += neglected[side]
        ref.potential_tendency.d[wall] += 2neglected[side] / ΔP * ∂P_wall
    end
    after = R₂(ref)
    ends = map(z -> z < ref.slot_height[K] || z > ref.slot_height[1], interior(model.domain, z_star))
    change = interior(model.domain, after .- before)
    @printf("  R_Π₂: max change %.2e of max|R_Π₂| (parcels with z_* in the end half rows: %d); ∫R_Π₂ %.4e -> %.4e, ∫|R_Π₂| %.4e W m⁻¹\n",
            maximum(abs, change) / maximum(abs, interior(model.domain, before)), count(ends),
            domain_integral(model, before), domain_integral(model, after), domain_integral(model, abs.(before)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    for (Mz, ε) in ((16, 0.1), (32, 1e-4), (64, 1e-4))
        fc_study(Float64; Mz, ε)
    end
    for γ in (1e-7, 1e-5, 1e-4, 3e-4), Mz in (16, 64)
        wall_study(; Mz, γ)
    end
end
