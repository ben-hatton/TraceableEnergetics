# Energy budget of a warm bubble rising in the anelastic (AN2D) and fully compressible (FC2D) models, from the same
# initial fields, diagnosed and plotted locally (maps of every term at the end of the run) and globally (time series of
# the volume integrals). With K the kinetic and P the potential energy per unit mass, the local budgets are
#     ∂ₜ(ρK) = -∇·(J_K^rev + J_K^irr) + C - ρε_K + VF,
#     ∂ₜ(ρP) = -∇·(J_P^rev + J_P^irr) - C - Q_ϑ - Q_q + Q,
# with the reversible fluxes J_K^rev = ρKu, J_P^rev = ρΣu, the irreversible fluxes J_K^irr = -u·τ, J_P^irr = Σ_ϑ J_ϑ + Σ_q J_q,
# the conversion C = C_ϑ + C_q + C_Σ = -ρϑu·∇Σ_ϑ - ρqu·∇Σ_q - ρu·∇(Σ - ϑΣ_ϑ - qΣ_q) from P to K, the viscous dissipation
# ρε_K = ∇·(u·τ) - u·∇·τ, the heatings Q_ϑ = -J_ϑ·∇Σ_ϑ, Q_q = -J_q·∇Σ_q, Q = Σ_ϑ σ_ϑ, and the vortex force work
# VF = -ρu·(ω×u); ρ is the reference density ρ_R(z) for AN2D. The heatings close the budget, Q = Q_ϑ + Q_q + ρε_K, and the
# conversion is also written as C = -∇·[(p - p_R)u] + C_b + C_D, with the buoyancy conversion C_b = ρ g (ρ_R v - 1) w and
# the pressure dilatation C_D = (p - p_R) ∇·u (zero for AN2D, where v = v(ϑ, q, p_R)). The global budget is
#     d/dt ∫ρ(K + P) dV = -∮(J_K^rev + J_K^irr + J_P^rev + J_P^irr)·dS + ∫VF dV,
# zero for the closed walls here, and its residual is zero to round-off for the energy-consistent schemes.
#
# The available energy is relative to the Lorenz reference state, the rearrangement of the parcels (the cells, each
# keeping its mass, ϑ and q) of least potential energy, computed at each output: its pressure P_*(z) and the reference
# height z_* of each parcel, the minimiser of the parcel's static energy along P_*. The available energy (AE) of a parcel
# at height z and pressure p is
#     Π = Σ(z, p) - Σ(z_*, P_*(z_*)) - (p - P_*(z)) Σ_p(z, p) = Π₁ + Π₂,
# the sum of the available compressible energy (ACE) Π₁ = Σ(z, p) - Σ(z, P_*(z)) - (p - P_*(z)) Σ_p(z, p), the
# compression from P_*(z) to p at fixed z (h(p) - h(P_*(z)) - (p - P_*(z)) v for FC2D, zero for AN2D, where Σ is linear in
# p), and the available potential energy (APE) Π₂ = Σ(z, P_*(z)) - Σ(z_*, P_*(z_*)), the displacement from z_* to z
# through the reference state. Each is computed parcel by parcel and integrated over the box, ∫ρΠ dV = Σ_n m_n Π_n, and
# the background potential energy is ∫ρ(P - Π) dV. With the potentials Σ_ϑ, Σ_q at the intermediate state (z, P_*(z))
# (Σ̃) and at the reference state (z_*, P_*(z_*)) (Σ*), the budgets at a fixed reference state are
#     ∂ₜ(ρΠ₁) = -∇·(J_ACE^rev + J_ACE^irr) - C_ACE - (Q_ϑ - Q̃_ϑ) - (Q_q - Q̃_q) + Q - Q̃,
#     ∂ₜ(ρΠ₂) = -∇·(J_APE^rev + J_APE^irr) - C_APE - (Q̃_ϑ - Q*_ϑ) - (Q̃_q - Q*_q) + Q̃ - Q*,
# with the heatings Q̃_ϑ = -J_ϑ·∇Σ̃_ϑ, Q*_ϑ = -J_ϑ·∇Σ*_ϑ, ..., and the conversions to K, C_ACE (= C_b,1* + C_D* - ∇·J_C*
# in the continuous equations, with J_C* = (p - P_*)u) and C_APE (= C_b,2*, the buoyancy conversion relative to the
# reference state), which leave the conversion C_bg = C - C_ACE - C_APE of the background, zero in the continuous
# equations. The terms in ∂ₜP_*, R_Π₁ and R_Π₂, are not diagnosed yet; ∫R_Π dV = 0. With closed walls, the global budgets
# are then
#     d/dt ∫ρΠ dV = -∫(C_ACE + C_APE) dV + G,    d/dt ∫ρ(P - Π) dV = -∫C_bg dV + ∫(Q* - Q*_ϑ - Q*_q) dV,
# with G = ∫(-Q_ϑ + Q*_ϑ - Q_q + Q*_q + Q - Q*) dV the diabatic generation of available energy. They are checked
# against d/dt ∫ρΠ dV from differences of successive outputs, with the rates averaged between them (both second order in
# the output interval; the same estimate for ∫ρK, against its exact tendency, measures the error of the differences).
# The available energy of the neutral state at rest (the same state without the bubble), zero for an exact reference
# state, is printed for comparison.
#
#   julia -t 4 examples/energy_budget/run.jl
#
# The data files and figures are saved to examples/energy_budget/output/.
include("../../tools/setup.jl")

using TraceableEnergetics, CFAnelastics, CFCompressible
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes
using CFPlanets: Tank2D
using CookBooks: CookBooks
using MutatingOrNot: void
using CairoMakie
include("../../tools/budget_figures.jl")

#####
##### Models: a small box, closed by free-slip insulating walls (see examples/kelvin_helmholtz for each choice)
#####

space = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)
T0, q0, ρ0 = 280.0, 0.035, 1000.0
fluid = NonlinearBinaryFluid(:potential_temperature, (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1 / ρ0, α_T = 2e-4,
    α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ = 1e-7))

mgr    = SIMD()
domain = Box2D(Mx = 64, Mz = 32, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))
closed = Wall(tangential = FreeSlip(), tracers = (Flux(0.0), Flux(0.0)))
boundary_conditions = BoxBoundaryConditions(domain; bottom = closed, top = closed)

# viscosity and diffusion make every irreversible term of the budget (ρε_K, Q_ϑ, Q_q, Q) nonzero
physics = (; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(),
    thermoflux_scheme = ThermoFluxConsistent(k_T = 1e-4, k_q = 1e-5), viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3),
    boundary_conditions)

an = AN2D((; physics..., buoyancy_scheme = ThermodynamicForcing(),
    anelastic_reference = AnelasticReference(domain, space, ρ0; pref = 1e5), gauge_offset = 0.0))
fc = FC2D(physics)

# a bubble 2 K warmer than its neutral surroundings, the same fields for both models
fields = rising_bubble(an; θ₀ = T0, Δθ = 2.0, q₀ = q0)
models = (an = an, fc = fc)
states = (an = initialize(an, fields), fc = initialize(fc, fields; pref = 1e5))

# the manipulated conversion needs a reference state ρ_R(z), p_R(z): AN2D has its own, FC2D takes that of AN2D (same grid)
references = (an = an.reference, fc = an.reference)
# the Lorenz reference state of available energy of each model, computed at each output by the exact rearrangement
# of the parcels (the transportation simplex, from their sorted placement)
lorenz = (an = AnelasticLorenzReference(an, ExactRearrangement()), fc = CompressibleLorenzReference(fc, ExactRearrangement()))
loops = (an = CFAnelastics.loop, fc = CFCompressible.loop)

#####
##### Diagnostics recorded at t = 0 and at the end of each time slice
#####

# the local budgets of ρK, ρP, ACE and APE at the cells, their integrals, and the available energies (`budget_names`)
names = map(budget_names, models)

params = (; Nslice = 40, slice_size = 0.5, dt_max = 0.1, cfl = 0.9)
dir = mkpath(joinpath(@__DIR__, "output"))
# NetCDF (.nc) or JLD2 (.jld2), read back identically
paths = map(label -> joinpath(dir, "energy_budget_$label.nc"), (an = "an", fc = "fc"))

#####
##### Runs
#####

# the outputs are recorded from the book of `TraceableEnergetics.diagnostics`, which adds the energy budget and the
# Lorenz reference state to the diagnostics of the model package (the default book of `loop`)
for key in keys(models)
    model = models[key]
    println("\n", nameof(typeof(model)))
    book = diagnostics(model; reference = references[key], reference_state = lorenz[key])
    # a data file, and a video of the conversion from P to K
    outputs = Outputs(DataFile(paths[key], model, names[key]; reference_state = lorenz[key]),
                      DiagnosticVideo(model, :conversion; file = joinpath(dir, "conversion_$key.mp4"), colormap = :balance))
    loops[key](states[key], model, CFTimeSchemes.RungeKutta4(model), params; outputs, book)
end

#####
##### Reading back, and the figures (tools/budget_figures.jl)
#####

results = read_results(models, paths)
files = String[]
save_figure(name, figure) = (file = joinpath(dir, name); save(file, figure); push!(files, file))
save_figure("global_budget.png", energy_budget_figure(results))
foreach(key -> save_figure("local_budget_$key.png", local_energy_budget_figure(results, key)), keys(models))
save_figure("available_energy_budget.png", available_energy_budget_figure(results))
save_figure("generation.png", generation_figure(results))
save_figure("reference_tendency.png", reference_tendency_figure(results))
foreach(key -> save_figure("local_available_energy_budget_$key.png", local_available_energy_budget_figure(results, key)), keys(models))
save_figure("available_energy.png", available_energy_maps(results))

#####
##### Available energy at rest: the same neutral state without the bubble
#####

# ∫ρΠ dV, ∫ρΠ₁ dV and ∫ρΠ₂ dV of the state at rest, zero for an exact reference state: what the discretisation of the
# reference state leaves is the level below which the available energy has no meaning
resting_fields = rising_bubble(an; θ₀ = T0, Δθ = 0.0, q₀ = q0)
resting_states = (an = initialize(an, resting_fields), fc = initialize(fc, resting_fields; pref = 1e5))
at_rest = map(models, resting_states, references, lorenz) do model, state, reference, ref
    dstate, scratch = CFTimeSchemes.tendencies!(void, void, model, state, 0.0)
    session = open(diagnostics(model; reference, reference_state = ref); state, dstate, scratch, model)
    integrated = (:integrated_AE, :integrated_ACE, :integrated_APE)
    energies = NamedTuple{integrated}(map(name -> CookBooks.get(session, name), integrated))
    close(session)
    energies
end

print_summary(results; at_rest)
println("figures saved to $(join(files, ", "))")
