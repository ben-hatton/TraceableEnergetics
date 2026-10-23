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
#   julia -t 4 examples/energy_budget/run.jl
#
# The data files and figures are saved to examples/energy_budget/output/.
include("../../tools/setup.jl")

using TraceableEnergetics, CFAnelastics, CFCompressible
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes
using CFPlanets: Tank2D
using CairoMakie

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
loops = (an = CFAnelastics.loop, fc = CFCompressible.loop)

#####
##### Diagnostics recorded at t = 0 and at the end of each time slice
#####

# local budget of ρK, of ρP, and the manipulated conversion, at the cells [W m⁻³]
kinetic_terms   = (:dρK, :div_kinetic_flux_rev, :div_kinetic_flux_irr, :conversion, :viscous_dissipation, :vortex_work,
                   :kinetic_budget_residual)
potential_terms = (:dρP, :div_potential_flux_rev, :div_potential_flux_irr, :heating, :heating_consvar, :heating_composition,
                   :potential_budget_residual, :heating_residual)
conversion_terms = (:div_pressure_flux, :buoyancy_conversion, :pressure_dilatation, :conversion_residual)
# global budget: integrals over the box, per unit length along y [J m⁻¹ or W m⁻¹]
integrals = (:integrated_kinetic_energy, :integrated_potential_energy, :integrated_energy_tendency, :integrated_boundary_flux,
             :integrated_vortex_work, :integrated_budget_residual)
names = (kinetic_terms..., potential_terms..., conversion_terms..., integrals...)

params = (; Nslice = 40, slice_size = 0.5, dt_max = 0.1, cfl = 0.9)
dir = mkpath(joinpath(@__DIR__, "output"))
# NetCDF (.nc) or JLD2 (.jld2), read back identically
paths = map(label -> joinpath(dir, "energy_budget_$label.nc"), (an = "an", fc = "fc"))

#####
##### Runs
#####

# the outputs are recorded from the book of `TraceableEnergetics.diagnostics`, which adds the energy budget to the
# diagnostics of the model package (the default book of `loop`)
for key in keys(models)
    model = models[key]
    println("\n", nameof(typeof(model)))
    book = diagnostics(model; reference = references[key])
    # a data file, and a video of the conversion from P to K
    outputs = Outputs(DataFile(paths[key], model, names),
                      DiagnosticVideo(model, :conversion; file = joinpath(dir, "conversion_$key.mp4"), colormap = :balance))
    loops[key](states[key], model, CFTimeSchemes.RungeKutta4(model), params; outputs, book)
end

#####
##### Reading back
#####

# every recorded diagnostic of each model: a (x, z, t) array with halo for a field, a vector over t for a number
data = map(paths) do path
    NamedTuple{names}(map(name -> first(read_diagnostic(path, name)), names))
end
times = last(read_diagnostic(paths.an, :dρK))

# the interior of the field `name` of `model` at slice n, and its integral over the box ∫ field dx dz [W m⁻¹]
field(key, name, n) = interior(models[key].domain, view(data[key][name], :, :, n))
integral(key, name) = [sum(field(key, name, n)) * models[key].dx * models[key].dz for n in eachindex(times)]

#####
##### Global budget: the integrals over the box against time, one column per model
#####

figure = Figure(size = (1100, 1200))
for (column, key) in enumerate(keys(models))
    d = data[key]
    label = nameof(typeof(models[key]))
    # energies, as changes from t = 0 (∫ρP is about 10¹⁰ times larger than ∫ρK)
    ax = Axis(figure[1, column]; title = "$label: energy change from t = 0", ylabel = "J m⁻¹")
    ΔK = d.integrated_kinetic_energy .- d.integrated_kinetic_energy[1]
    ΔP = d.integrated_potential_energy .- d.integrated_potential_energy[1]
    lines!(ax, times, ΔK; label = "∫ρK")
    lines!(ax, times, ΔP; label = "∫ρP")
    lines!(ax, times, ΔK .+ ΔP; label = "∫ρ(K + P)", linestyle = :dash, color = :black)
    axislegend(ax; position = :lt)
    # budget of ∫ρK: d/dt ∫ρK = -∮J_K·dS + ∫C - ∫ρε_K + ∫VF
    ax = Axis(figure[2, column]; title = "$label: budget of ∫ρK", ylabel = "W m⁻¹")
    lines!(ax, times, integral(key, :dρK); label = "d/dt ∫ρK", color = :black)
    lines!(ax, times, integral(key, :conversion); label = "∫C")
    lines!(ax, times, -integral(key, :viscous_dissipation); label = "-∫ρε_K")
    lines!(ax, times, -integral(key, :div_kinetic_flux_rev) .- integral(key, :div_kinetic_flux_irr); label = "-∮J_K·dS")
    lines!(ax, times, d.integrated_vortex_work; label = "∫VF")
    axislegend(ax; position = :rt)
    # budget of ∫ρP: d/dt ∫ρP = -∮J_P·dS - ∫C + ∫(Q - Q_ϑ - Q_q)
    ax = Axis(figure[3, column]; title = "$label: budget of ∫ρP", ylabel = "W m⁻¹")
    lines!(ax, times, integral(key, :dρP); label = "d/dt ∫ρP", color = :black)
    lines!(ax, times, -integral(key, :conversion); label = "-∫C")
    lines!(ax, times, integral(key, :heating) .- integral(key, :heating_consvar) .- integral(key, :heating_composition);
           label = "∫(Q - Q_ϑ - Q_q)")
    lines!(ax, times, -integral(key, :div_potential_flux_rev) .- integral(key, :div_potential_flux_irr); label = "-∮J_P·dS")
    axislegend(ax; position = :rt)
    # closure of the global budget: d/dt ∫ρ(K + P) + ∮(J_K + J_P)·dS - ∫VF, against the scale of the exchanges
    ax = Axis(figure[4, column]; title = "$label: residual of the budget of ∫ρ(K + P)", xlabel = "t [s]",
              ylabel = "W m⁻¹")
    lines!(ax, times, d.integrated_budget_residual; label = "residual", color = :black)
    lines!(ax, times, d.integrated_energy_tendency; label = "d/dt ∫ρ(K + P)")
    lines!(ax, times, d.integrated_boundary_flux; label = "∮(J_K + J_P)·dS")
    axislegend(ax; position = :rt)
end
global_file = joinpath(dir, "global_budget.png")
save(global_file, figure)

#####
##### Local budgets: maps of each term at the end of the run, one figure per model
#####

# each term is shown as its contribution to the tendency (with its sign in the budget), on its own symmetric colour range:
# the terms span many orders of magnitude (the divergence of ρΣu is large at a cell, but integrates to the wall flux),
# and the residuals are of round-off size
function budget_row!(figure, row, key, title, panels)
    x, z = diagnostic_coordinates(models[key], first(first(panels)))
    n = lastindex(times)
    Label(figure[row, 1], title; rotation = π / 2, tellheight = false)
    for (k, (name, sign, label)) in enumerate(panels)
        map = sign * field(key, name, n)
        # a zero field (C_D for AN2D) gets the smallest nonzero range
        limit = max(maximum(abs, map), floatmin(eltype(map)))
        cell = figure[row, k + 1] = GridLayout()
        ax = Axis(cell[1, 1]; title = label, aspect = DataAspect())
        heatmap!(ax, x, z, map; colormap = :balance, colorrange = (-limit, limit))
        hidedecorations!(ax)
        Colorbar(cell[2, 1]; colormap = :balance, limits = (-limit, limit), vertical = false, flipaxis = false,
                 label = "W m⁻³", ticklabelsize = 10, ticks = Makie.WilkinsonTicks(3))
    end
end

local_files = map(keys(models)) do key
    figure = Figure(size = (2000, 800))
    # (diagnostic, sign in the budget, title)
    budget_row!(figure, 1, key, "budget of ρK",
        ((:dρK, 1, "∂ₜ(ρK)"), (:div_kinetic_flux_rev, -1, "-∇·J_K^rev"), (:div_kinetic_flux_irr, -1, "-∇·J_K^irr"),
         (:conversion, 1, "C"), (:viscous_dissipation, -1, "-ρε_K"), (:vortex_work, 1, "VF"),
         (:kinetic_budget_residual, 1, "residual (zero in ∫ only)")))
    budget_row!(figure, 2, key, "budget of ρP",
        ((:dρP, 1, "∂ₜ(ρP)"), (:div_potential_flux_rev, -1, "-∇·J_P^rev"), (:div_potential_flux_irr, -1, "-∇·J_P^irr"),
         (:conversion, -1, "-C"), (:heating_consvar, -1, "-Q_ϑ"), (:heating_composition, -1, "-Q_q"),
         (:heating, 1, "Q"), (:potential_budget_residual, 1, "residual")))
    budget_row!(figure, 3, key, "conversion",
        ((:conversion, 1, "C"), (:div_pressure_flux, -1, "-∇·J_C"), (:buoyancy_conversion, 1, "C_b"),
         (:pressure_dilatation, 1, "C_D"),
         (:conversion_residual, 1, "C + ∇·J_C - C_b - C_D")))
    Label(figure[0, :], "$(nameof(typeof(models[key]))): local energy budget at t = $(times[end]) s"; fontsize = 18)
    file = joinpath(dir, "local_budget_$key.png")
    save(file, figure)
    file
end

#####
##### Summary
#####

println()
for key in keys(models)
    d = data[key]
    scale = maximum(abs, integral(key, :conversion))
    println("$(nameof(typeof(models[key]))): max |global residual| = $(maximum(abs, d.integrated_budget_residual)) W m⁻¹, ",
            "max |∫C| = $scale W m⁻¹, max |∫VF| = $(maximum(abs, d.integrated_vortex_work)) W m⁻¹")
end
println("figures saved to $global_file, $(join(local_files, ", "))")
