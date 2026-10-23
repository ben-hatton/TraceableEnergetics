## DIAGNOSTICS OF EITHER MODEL
#
# The diagnostics of AN2D and FC2D are those of the model packages and the energy budget of energy.tex (kernels in
# energetics.jl). They are full arrays with halo, for post-processing; for plotting, `interior_diagnostic` strips the halo
# and `diagnostic_coordinates` gives the coordinates of the remaining points, from the grid location of the diagnostic.

"""
    book = diagnostics(model; reference = default)
    info = diagnostic_info(model)

`CookBook` of the diagnostics of an AN2D or FC2D model, opened with `open(book; state, dstate, scratch, model)`, and
the long name, units and grid location of each diagnostic: one `Face()`/`Cell()` per dimension, e.g. `(Cell(), Cell())`
cells, `(Face(), Cell())` u-faces, `()` for a number. They are those of `CFAnelastics.diagnostics` and
`CFCompressible.diagnostics`, and the energy budget of energy.tex, listed in `energetics_entries` below. K and P are
specific energies [J kg⁻¹]; ρK, ρP and every term of the budget are per unit volume [W m⁻³] (ρ = ρ_R for AN2D); the
`integrated_` diagnostics are their integrals over the interior cells, per unit length along y.

The geopotential is g (z - z0), with z0 the origin `model.space.z0` [m] of the tank. The manipulated conversion needs the
reference state (`AnelasticReference` with the density ρ_R(z) and the hydrostatic pressure p_R(z)): the model's own for
AN2D, the keyword `reference` (usually that of the matching AN2D, on the same grid) for FC2D, which has none.

Build the book once, as each session of it overwrites the same output arrays.
"""
function diagnostics(model::AN2D{F}; reference = model.reference) where F
    book = CFAnelastics.diagnostics(model)
    return add_energetics!(book, F, model, reference)
end
function diagnostics(model::FC2D{F}; reference = nothing) where F
    book = CFCompressible.diagnostics(model)
    return add_energetics!(book, F, model, reference)
end
diagnostic_info(model::AN2D) = merge(CFAnelastics.diagnostic_info(model), map(first, energetics_entries(model, nothing, nothing, nothing)))
diagnostic_info(model::FC2D) = merge(CFCompressible.diagnostic_info(model), map(first, energetics_entries(model, nothing, nothing, nothing)))

# Adds the recipes of `energetics_entries` to `book`. The output array of each diagnostic is allocated on its first use, on
# its grid location; `faces` are the scratch arrays on the x and z faces shared by the terms built from face products.
function add_energetics!(book, F, model, reference)
    (; mgr, domain) = model
    entries = energetics_entries(model, nothing, nothing, nothing)
    cache = Dict{Symbol, Any}()
    out(name) = get!(() -> alloc(F, mgr, domain, first(entries[name]).location), cache, name)::Matrix{F}
    faces = (alloc(F, mgr, domain, (Face(), Cell())), alloc(F, mgr, domain, (Cell(), Face())))
    merge!(book.recipes, pairs(map(last, energetics_entries(model, out, faces, reference))))
    return book
end

#=
Energy budget of energy.tex: each entry is `entry(long_name, units, location, recipe)`, below the mathematical expression
of the diagnostic. Σ is the potential of the model (Σ = h + g z for FC2D, Σ = h(p_R) + (p - p_R)/ρ_R + g z for AN2D), with
Σ_ϑ = ∂Σ/∂ϑ, Σ_q = ∂Σ/∂q the conjugate variable and the modified chemical potential, J_ϑ, J_q the diffusive fluxes, σ_ϑ
the production of the conservative variable, τ the viscous stress and ρε_K = ∇·(u·τ) - u·∇·τ the viscous dissipation.
=#

energetics_entries(model::AN2D, out, faces, reference) = let (; mgr) = model, cells = (Cell(), Cell()), x_faces = (Face(), Cell()), z_faces = (Cell(), Face())
    (;
        # energies, per unit mass: h(ϑ, q, p_R), P = h - p_R/ρ_R + g (z - z0), Σ - ϑ Σ_ϑ - q Σ_q
        specific_enthalpy = entry("specific enthalpy h at p_R", "J kg⁻¹", cells,
            (model, scratch) -> specific_enthalpy!(out(:specific_enthalpy), model, model.reference.p_R, scratch.thermo)),
        potential_energy = entry("potential energy P = h - p_R/ρ_R + g (z - z0)", "J kg⁻¹", cells,
            (model, specific_enthalpy) -> potential_energy!(out(:potential_energy), model, specific_enthalpy, model.reference.ρ_R, model.reference.p_R)),
        bernoulli_potential = entry("Σ - ϑ Σ_ϑ - q Σ_q, B + g (z - z0) + φ", "J kg⁻¹", cells,
            (model, specific_enthalpy, scratch) -> bernoulli_potential!(out(:bernoulli_potential), model, specific_enthalpy, scratch)),
        # energy densities ρ_R K and ρ_R P
        ρK = entry("kinetic energy density ρ K", "J m⁻³", cells,
            (density, kinetic_energy) -> pointwise!(out(:ρK), mgr, *, density, kinetic_energy)),
        ρP = entry("potential energy density ρ P", "J m⁻³", cells,
            (density, potential_energy) -> pointwise!(out(:ρP), mgr, *, density, potential_energy)),
        # tendency of K, ∂ₜK = ℑx(u ∂ₜu) + ℑz(w ∂ₜw), and of ρ_R K
        kinetic_energy_rate = entry("tendency of K", "W kg⁻¹", cells,
            (model, u, w, du, dw) -> kinetic_energy_rate!(out(:kinetic_energy_rate), model, faces, (u, w), (du, dw))),
        dρK = entry("tendency of ρ K", "W m⁻³", cells,
            (density, kinetic_energy_rate) -> pointwise!(out(:dρK), mgr, *, density, kinetic_energy_rate)),
        # tendency of ρ_R P = Σ_ϑ ∂ₜ(ρ_R ϑ) + Σ_q ∂ₜ(ρ_R q)
        dρP = entry("tendency of ρ P", "W m⁻³", cells,
            (model, conjugate_variable, modified_chemical_potential, dρconsvar, dρq) -> potential_energy_tendency!(out(:dρP), model, conjugate_variable, modified_chemical_potential, dρconsvar, dρq)),
        # reversible flux of K, J_K^rev = ρ K u
        kinetic_flux_rev_x = entry("reversible flux of K along x, ρ K u", "W m⁻²", x_faces,
            (model, mass_flux_x, kinetic_energy) -> advective_flux_x!(out(:kinetic_flux_rev_x), model, mass_flux_x, kinetic_energy)),
        kinetic_flux_rev_z = entry("reversible flux of K along z, ρ K w", "W m⁻²", z_faces,
            (model, mass_flux_z, kinetic_energy) -> advective_flux_z!(out(:kinetic_flux_rev_z), model, mass_flux_z, kinetic_energy)),
        # irreversible flux of K, J_K^irr = -u·τ
        kinetic_flux_irr_x = entry("irreversible flux of K along x, -(u·τ)_x", "W m⁻²", x_faces,
            (model, scratch) -> stress_work_flux_x!(out(:kinetic_flux_irr_x), model, scratch.irreversible.uτ_x)),
        kinetic_flux_irr_z = entry("irreversible flux of K along z, -(u·τ)_z", "W m⁻²", z_faces,
            (model, scratch) -> stress_work_flux_z!(out(:kinetic_flux_irr_z), model, scratch.irreversible.uτ_z)),
        # reversible flux of P, J_P^rev = ρ Σ u = ρu (Σ - ϑ Σ_ϑ - q Σ_q) + ρϑu Σ_ϑ + ρqu Σ_q
        potential_flux_rev_x = entry("reversible flux of P along x, ρ Σ u", "W m⁻²", x_faces,
            (model, scratch, mass_flux_x, bernoulli_potential, conjugate_variable, modified_chemical_potential) ->
                potential_flux_rev_x!(out(:potential_flux_rev_x), model, mass_flux_x, scratch.advection.mu, 1, bernoulli_potential, conjugate_variable, modified_chemical_potential)),
        potential_flux_rev_z = entry("reversible flux of P along z, ρ Σ w", "W m⁻²", z_faces,
            (model, scratch, mass_flux_z, bernoulli_potential, conjugate_variable, modified_chemical_potential) ->
                potential_flux_rev_z!(out(:potential_flux_rev_z), model, mass_flux_z, scratch.advection.mw, 1, bernoulli_potential, conjugate_variable, modified_chemical_potential)),
        # irreversible flux of P, J_P^irr = Σ_ϑ J_ϑ + Σ_q J_q
        potential_flux_irr_x = entry("irreversible flux of P along x, Σ_ϑ J_ϑ + Σ_q J_q", "W m⁻²", x_faces,
            (model, consflux_x, qflux_x, conjugate_variable, modified_chemical_potential) ->
                potential_flux_irr_x!(out(:potential_flux_irr_x), model, consflux_x, qflux_x, conjugate_variable, modified_chemical_potential)),
        potential_flux_irr_z = entry("irreversible flux of P along z, Σ_ϑ J_ϑ + Σ_q J_q", "W m⁻²", z_faces,
            (model, consflux_z, qflux_z, conjugate_variable, modified_chemical_potential) ->
                potential_flux_irr_z!(out(:potential_flux_irr_z), model, consflux_z, qflux_z, conjugate_variable, modified_chemical_potential)),
        # divergences ∇·J_K^rev, ∇·J_K^irr, ∇·J_P^rev, ∇·J_P^irr
        div_kinetic_flux_rev = entry("divergence of the reversible flux of K", "W m⁻³", cells,
            (model, kinetic_flux_rev_x, kinetic_flux_rev_z) -> divergence!(out(:div_kinetic_flux_rev), model, kinetic_flux_rev_x, kinetic_flux_rev_z)),
        div_kinetic_flux_irr = entry("divergence of the irreversible flux of K", "W m⁻³", cells,
            (model, kinetic_flux_irr_x, kinetic_flux_irr_z) -> divergence!(out(:div_kinetic_flux_irr), model, kinetic_flux_irr_x, kinetic_flux_irr_z)),
        div_potential_flux_rev = entry("divergence of the reversible flux of P", "W m⁻³", cells,
            (model, potential_flux_rev_x, potential_flux_rev_z) -> divergence!(out(:div_potential_flux_rev), model, potential_flux_rev_x, potential_flux_rev_z)),
        div_potential_flux_irr = entry("divergence of the irreversible flux of P", "W m⁻³", cells,
            (model, potential_flux_irr_x, potential_flux_irr_z) -> divergence!(out(:div_potential_flux_irr), model, potential_flux_irr_x, potential_flux_irr_z)),
        # conversion from P to K, C_ϑ = -ρ ϑ u·∇Σ_ϑ
        conversion_consvar = entry("conversion C_ϑ from P to K", "W m⁻³", cells,
            (model, scratch, conjugate_variable) -> gradient_product!(out(:conversion_consvar), model, faces, tracer_flux(scratch.advection, 1), conjugate_variable)),
        # C_q = -ρ q u·∇Σ_q
        conversion_composition = entry("conversion C_q from P to K", "W m⁻³", cells,
            (model, scratch, modified_chemical_potential) -> gradient_product!(out(:conversion_composition), model, faces, tracer_flux(scratch.advection, 2), modified_chemical_potential)),
        # C_Σ = -ρ u·∇(Σ - ϑ Σ_ϑ - q Σ_q)
        conversion_bernoulli = entry("conversion C_Σ from P to K", "W m⁻³", cells,
            (model, mass_flux_x, mass_flux_z, bernoulli_potential) -> gradient_product!(out(:conversion_bernoulli), model, faces, (mass_flux_x, mass_flux_z), bernoulli_potential)),
        # C = C_ϑ + C_q + C_Σ
        conversion = entry("conversion C = C_ϑ + C_q + C_Σ from P to K", "W m⁻³", cells,
            (conversion_consvar, conversion_composition, conversion_bernoulli) -> pointwise!(out(:conversion), mgr, +, conversion_consvar, conversion_composition, conversion_bernoulli)),
        # heating Q_ϑ = -J_ϑ·∇Σ_ϑ
        heating_consvar = entry("heating Q_ϑ = -J_ϑ·∇Σ_ϑ", "W m⁻³", cells,
            (model, consflux_x, consflux_z, conjugate_variable) -> gradient_product!(out(:heating_consvar), model, faces, (consflux_x, consflux_z), conjugate_variable)),
        # Q_q = -J_q·∇Σ_q
        heating_composition = entry("heating Q_q = -J_q·∇Σ_q", "W m⁻³", cells,
            (model, qflux_x, qflux_z, modified_chemical_potential) -> gradient_product!(out(:heating_composition), model, faces, (qflux_x, qflux_z), modified_chemical_potential)),
        # Q = Σ_ϑ σ_ϑ
        heating = entry("heating Q = Σ_ϑ σ_ϑ", "W m⁻³", cells,
            (conjugate_variable, conservative_variable_production) -> pointwise!(out(:heating), mgr, *, conjugate_variable, conservative_variable_production)),
        # closure Q - Q_ϑ - Q_q - ρ ε_K = 0
        heating_residual = entry("closure residual Q - Q_ϑ - Q_q - ρ ε_K", "W m⁻³", cells,
            (model, heating, heating_consvar, heating_composition, viscous_dissipation) -> heating_residual!(out(:heating_residual), model, heating, heating_consvar, heating_composition, viscous_dissipation)),
        # vortex force work VF = -ρ u·(ω×u)
        vortex_work = entry("vortex force work VF = -ρ u·(ω×u)", "W m⁻³", cells,
            (model, scratch, mass_flux_x, mass_flux_z) -> vortex_work!(out(:vortex_work), model, faces, scratch.advection, mass_flux_x, mass_flux_z)),
        # ∂ₜ(ρK) + ∇·(J_K^rev + J_K^irr) - C + ρ ε_K - VF: zero in the volume integral only
        kinetic_budget_residual = entry("residual of the budget of ρ K", "W m⁻³", cells,
            (model, dρK, div_kinetic_flux_rev, div_kinetic_flux_irr, conversion, viscous_dissipation, vortex_work) ->
                kinetic_budget_residual!(out(:kinetic_budget_residual), model, dρK, div_kinetic_flux_rev, div_kinetic_flux_irr, conversion, viscous_dissipation, vortex_work)),
        # ∂ₜ(ρP) + ∇·(J_P^rev + J_P^irr) + C + Q_ϑ + Q_q - Q: zero at each cell
        potential_budget_residual = entry("residual of the budget of ρ P", "W m⁻³", cells,
            (model, dρP, div_potential_flux_rev, div_potential_flux_irr, conversion, heating_consvar, heating_composition, heating) ->
                potential_budget_residual!(out(:potential_budget_residual), model, dρP, div_potential_flux_rev, div_potential_flux_irr, conversion, heating_consvar, heating_composition, heating)),
        # manipulated conversion C = -∇·J_C + C_b + C_D, with p - p_R
        pressure_anomaly = entry("pressure anomaly p - p_R", "Pa", cells,
            (model, pressure) -> pressure_anomaly!(out(:pressure_anomaly), model, require_reference(reference), pressure)),
        # ∇·J_C, J_C = (p - p_R) u
        div_pressure_flux = entry("divergence of J_C = (p - p_R) u", "W m⁻³", cells,
            (model, u, w, pressure_anomaly) -> pressure_flux_divergence!(out(:div_pressure_flux), model, faces, u, w, pressure_anomaly)),
        # b = g (ρ_R v - 1), v = v(ϑ, q, p_R)
        reference_buoyancy = entry("buoyancy g (ρ_R v - 1) of the reference state", "m s⁻²", cells,
            (model, state, scratch) -> reference_buoyancy!(out(:reference_buoyancy), model, require_reference(reference), state, scratch)),
        # C_b = ρ_R b w
        buoyancy_conversion = entry("buoyancy conversion C_b = ρ b w", "W m⁻³", cells,
            (model, mass_flux_z, reference_buoyancy) -> buoyancy_conversion!(out(:buoyancy_conversion), model, faces, mass_flux_z, reference_buoyancy)),
        # C_D = (p - p_R) ∇·u, zero for AN2D (part of C_b)
        pressure_dilatation = entry("pressure dilatation C_D = (p - p_R) ∇·u (zero for AN2D)", "W m⁻³", cells,
            (model, pressure_anomaly, divergence) -> pressure_dilatation!(out(:pressure_dilatation), model, pressure_anomaly, divergence)),
        # C + ∇·J_C - C_b - C_D
        conversion_residual = entry("residual of C = -∇·J_C + C_b + C_D", "W m⁻³", cells,
            (model, conversion, div_pressure_flux, buoyancy_conversion, pressure_dilatation) -> conversion_residual!(out(:conversion_residual), model, conversion, div_pressure_flux, buoyancy_conversion, pressure_dilatation)),
        # integrals over the interior cells, ∫ρK dV, ∫ρP dV
        integrated_kinetic_energy = entry("integral of ρ K", "J m⁻¹", (), (model, ρK) -> domain_integral(model, ρK)),
        integrated_potential_energy = entry("integral of ρ P", "J m⁻¹", (), (model, ρP) -> domain_integral(model, ρP)),
        # ∫(∂ₜ(ρK) + ∂ₜ(ρP)) dV
        integrated_energy_tendency = entry("integral of the tendency of ρ (K + P)", "W m⁻¹", (),
            (model, dρK, dρP) -> domain_integral(model, dρK) + domain_integral(model, dρP)),
        # ∮(J_K^rev + J_K^irr + J_P^rev + J_P^irr)·dS, the sum of the divergences (telescoping to the boundary fluxes)
        integrated_boundary_flux = entry("integral of the divergences of the fluxes of K and P", "W m⁻¹", (),
            (model, div_kinetic_flux_rev, div_kinetic_flux_irr, div_potential_flux_rev, div_potential_flux_irr) ->
                domain_integral(model, div_kinetic_flux_rev) + domain_integral(model, div_kinetic_flux_irr) + domain_integral(model, div_potential_flux_rev) + domain_integral(model, div_potential_flux_irr)),
        # ∫VF dV
        integrated_vortex_work = entry("integral of the vortex force work", "W m⁻¹", (), (model, vortex_work) -> domain_integral(model, vortex_work)),
        # d/dt ∫(K + P) ρ dV + ∮(J_K + J_P)·dS - ∫VF dV: zero to round-off for the energy-consistent schemes
        integrated_budget_residual = entry("residual of the budget of ∫ρ (K + P) dV", "W m⁻¹", (),
            (integrated_energy_tendency, integrated_boundary_flux, integrated_vortex_work) -> integrated_energy_tendency + integrated_boundary_flux - integrated_vortex_work),
    )
end

energetics_entries(model::FC2D, out, faces, reference) = let (; mgr) = model, cells = (Cell(), Cell()), x_faces = (Face(), Cell()), z_faces = (Cell(), Face())
    (;
        # energies, per unit mass: h(ϑ, q, p), P = h - p/ρ + g (z - z0), Σ - ϑ Σ_ϑ - q Σ_q
        specific_enthalpy = entry("specific enthalpy h", "J kg⁻¹", cells,
            (model, scratch) -> specific_enthalpy!(out(:specific_enthalpy), model, scratch.thermo.p, scratch.thermo)),
        potential_energy = entry("potential energy P = h - p/ρ + g (z - z0)", "J kg⁻¹", cells,
            (model, specific_enthalpy, density, scratch) -> potential_energy!(out(:potential_energy), model, specific_enthalpy, density, scratch.thermo.p)),
        bernoulli_potential = entry("Σ - ϑ Σ_ϑ - q Σ_q, B + g (z - z0)", "J kg⁻¹", cells,
            (model, specific_enthalpy, scratch) -> bernoulli_potential!(out(:bernoulli_potential), model, specific_enthalpy, scratch)),
        # energy densities ρK and ρP
        ρK = entry("kinetic energy density ρ K", "J m⁻³", cells,
            (density, kinetic_energy) -> pointwise!(out(:ρK), mgr, *, density, kinetic_energy)),
        ρP = entry("potential energy density ρ P", "J m⁻³", cells,
            (density, potential_energy) -> pointwise!(out(:ρP), mgr, *, density, potential_energy)),
        # tendency of K, ∂ₜK = ℑx(u ∂ₜu) + ℑz(w ∂ₜw), and of ρK = ρ ∂ₜK + K ∂ₜρ
        kinetic_energy_rate = entry("tendency of K", "W kg⁻¹", cells,
            (model, u, w, du, dw) -> kinetic_energy_rate!(out(:kinetic_energy_rate), model, faces, (u, w), (du, dw))),
        dρK = entry("tendency of ρ K", "W m⁻³", cells,
            (model, density, kinetic_energy, kinetic_energy_rate, dρ) -> kinetic_energy_tendency!(out(:dρK), model, density, kinetic_energy, kinetic_energy_rate, dρ)),
        # tendency of ρP = (Σ - ϑ Σ_ϑ - q Σ_q) ∂ₜρ + Σ_ϑ ∂ₜ(ρϑ) + Σ_q ∂ₜ(ρq)
        dρP = entry("tendency of ρ P", "W m⁻³", cells,
            (model, bernoulli_potential, conjugate_variable, modified_chemical_potential, dρ, dρconsvar, dρq) ->
                potential_energy_tendency!(out(:dρP), model, bernoulli_potential, conjugate_variable, modified_chemical_potential, dρ, dρconsvar, dρq)),
        # reversible flux of K, J_K^rev = ρ K u
        kinetic_flux_rev_x = entry("reversible flux of K along x, ρ K u", "W m⁻²", x_faces,
            (model, mass_flux_x, kinetic_energy) -> advective_flux_x!(out(:kinetic_flux_rev_x), model, mass_flux_x, kinetic_energy)),
        kinetic_flux_rev_z = entry("reversible flux of K along z, ρ K w", "W m⁻²", z_faces,
            (model, mass_flux_z, kinetic_energy) -> advective_flux_z!(out(:kinetic_flux_rev_z), model, mass_flux_z, kinetic_energy)),
        # irreversible flux of K, J_K^irr = -u·τ
        kinetic_flux_irr_x = entry("irreversible flux of K along x, -(u·τ)_x", "W m⁻²", x_faces,
            (model, scratch) -> stress_work_flux_x!(out(:kinetic_flux_irr_x), model, scratch.irreversible.uτ_x)),
        kinetic_flux_irr_z = entry("irreversible flux of K along z, -(u·τ)_z", "W m⁻²", z_faces,
            (model, scratch) -> stress_work_flux_z!(out(:kinetic_flux_irr_z), model, scratch.irreversible.uτ_z)),
        # reversible flux of P, J_P^rev = ρ Σ u = ρu (Σ - ϑ Σ_ϑ - q Σ_q) + ρϑu Σ_ϑ + ρqu Σ_q
        potential_flux_rev_x = entry("reversible flux of P along x, ρ Σ u", "W m⁻²", x_faces,
            (model, scratch, mass_flux_x, bernoulli_potential, conjugate_variable, modified_chemical_potential) ->
                potential_flux_rev_x!(out(:potential_flux_rev_x), model, mass_flux_x, scratch.advection.mu, 2, bernoulli_potential, conjugate_variable, modified_chemical_potential)),
        potential_flux_rev_z = entry("reversible flux of P along z, ρ Σ w", "W m⁻²", z_faces,
            (model, scratch, mass_flux_z, bernoulli_potential, conjugate_variable, modified_chemical_potential) ->
                potential_flux_rev_z!(out(:potential_flux_rev_z), model, mass_flux_z, scratch.advection.mw, 2, bernoulli_potential, conjugate_variable, modified_chemical_potential)),
        # irreversible flux of P, J_P^irr = Σ_ϑ J_ϑ + Σ_q J_q
        potential_flux_irr_x = entry("irreversible flux of P along x, Σ_ϑ J_ϑ + Σ_q J_q", "W m⁻²", x_faces,
            (model, consflux_x, qflux_x, conjugate_variable, modified_chemical_potential) ->
                potential_flux_irr_x!(out(:potential_flux_irr_x), model, consflux_x, qflux_x, conjugate_variable, modified_chemical_potential)),
        potential_flux_irr_z = entry("irreversible flux of P along z, Σ_ϑ J_ϑ + Σ_q J_q", "W m⁻²", z_faces,
            (model, consflux_z, qflux_z, conjugate_variable, modified_chemical_potential) ->
                potential_flux_irr_z!(out(:potential_flux_irr_z), model, consflux_z, qflux_z, conjugate_variable, modified_chemical_potential)),
        # divergences ∇·J_K^rev, ∇·J_K^irr, ∇·J_P^rev, ∇·J_P^irr
        div_kinetic_flux_rev = entry("divergence of the reversible flux of K", "W m⁻³", cells,
            (model, kinetic_flux_rev_x, kinetic_flux_rev_z) -> divergence!(out(:div_kinetic_flux_rev), model, kinetic_flux_rev_x, kinetic_flux_rev_z)),
        div_kinetic_flux_irr = entry("divergence of the irreversible flux of K", "W m⁻³", cells,
            (model, kinetic_flux_irr_x, kinetic_flux_irr_z) -> divergence!(out(:div_kinetic_flux_irr), model, kinetic_flux_irr_x, kinetic_flux_irr_z)),
        div_potential_flux_rev = entry("divergence of the reversible flux of P", "W m⁻³", cells,
            (model, potential_flux_rev_x, potential_flux_rev_z) -> divergence!(out(:div_potential_flux_rev), model, potential_flux_rev_x, potential_flux_rev_z)),
        div_potential_flux_irr = entry("divergence of the irreversible flux of P", "W m⁻³", cells,
            (model, potential_flux_irr_x, potential_flux_irr_z) -> divergence!(out(:div_potential_flux_irr), model, potential_flux_irr_x, potential_flux_irr_z)),
        # conversion from P to K, C_ϑ = -ρ ϑ u·∇Σ_ϑ
        conversion_consvar = entry("conversion C_ϑ from P to K", "W m⁻³", cells,
            (model, scratch, conjugate_variable) -> gradient_product!(out(:conversion_consvar), model, faces, tracer_flux(scratch.advection, 2), conjugate_variable)),
        # C_q = -ρ q u·∇Σ_q
        conversion_composition = entry("conversion C_q from P to K", "W m⁻³", cells,
            (model, scratch, modified_chemical_potential) -> gradient_product!(out(:conversion_composition), model, faces, tracer_flux(scratch.advection, 3), modified_chemical_potential)),
        # C_Σ = -ρ u·∇(Σ - ϑ Σ_ϑ - q Σ_q)
        conversion_bernoulli = entry("conversion C_Σ from P to K", "W m⁻³", cells,
            (model, mass_flux_x, mass_flux_z, bernoulli_potential) -> gradient_product!(out(:conversion_bernoulli), model, faces, (mass_flux_x, mass_flux_z), bernoulli_potential)),
        # C = C_ϑ + C_q + C_Σ
        conversion = entry("conversion C = C_ϑ + C_q + C_Σ from P to K", "W m⁻³", cells,
            (conversion_consvar, conversion_composition, conversion_bernoulli) -> pointwise!(out(:conversion), mgr, +, conversion_consvar, conversion_composition, conversion_bernoulli)),
        # heating Q_ϑ = -J_ϑ·∇Σ_ϑ
        heating_consvar = entry("heating Q_ϑ = -J_ϑ·∇Σ_ϑ", "W m⁻³", cells,
            (model, consflux_x, consflux_z, conjugate_variable) -> gradient_product!(out(:heating_consvar), model, faces, (consflux_x, consflux_z), conjugate_variable)),
        # Q_q = -J_q·∇Σ_q
        heating_composition = entry("heating Q_q = -J_q·∇Σ_q", "W m⁻³", cells,
            (model, qflux_x, qflux_z, modified_chemical_potential) -> gradient_product!(out(:heating_composition), model, faces, (qflux_x, qflux_z), modified_chemical_potential)),
        # Q = Σ_ϑ σ_ϑ
        heating = entry("heating Q = Σ_ϑ σ_ϑ", "W m⁻³", cells,
            (conjugate_variable, conservative_variable_production) -> pointwise!(out(:heating), mgr, *, conjugate_variable, conservative_variable_production)),
        # closure Q - Q_ϑ - Q_q - ρ ε_K = 0
        heating_residual = entry("closure residual Q - Q_ϑ - Q_q - ρ ε_K", "W m⁻³", cells,
            (model, heating, heating_consvar, heating_composition, viscous_dissipation) -> heating_residual!(out(:heating_residual), model, heating, heating_consvar, heating_composition, viscous_dissipation)),
        # vortex force work VF = -ρ u·(ω×u)
        vortex_work = entry("vortex force work VF = -ρ u·(ω×u)", "W m⁻³", cells,
            (model, scratch, mass_flux_x, mass_flux_z) -> vortex_work!(out(:vortex_work), model, faces, scratch.advection, mass_flux_x, mass_flux_z)),
        # ∂ₜ(ρK) + ∇·(J_K^rev + J_K^irr) - C + ρ ε_K - VF: zero in the volume integral only
        kinetic_budget_residual = entry("residual of the budget of ρ K", "W m⁻³", cells,
            (model, dρK, div_kinetic_flux_rev, div_kinetic_flux_irr, conversion, viscous_dissipation, vortex_work) ->
                kinetic_budget_residual!(out(:kinetic_budget_residual), model, dρK, div_kinetic_flux_rev, div_kinetic_flux_irr, conversion, viscous_dissipation, vortex_work)),
        # ∂ₜ(ρP) + ∇·(J_P^rev + J_P^irr) + C + Q_ϑ + Q_q - Q: zero at each cell
        potential_budget_residual = entry("residual of the budget of ρ P", "W m⁻³", cells,
            (model, dρP, div_potential_flux_rev, div_potential_flux_irr, conversion, heating_consvar, heating_composition, heating) ->
                potential_budget_residual!(out(:potential_budget_residual), model, dρP, div_potential_flux_rev, div_potential_flux_irr, conversion, heating_consvar, heating_composition, heating)),
        # manipulated conversion C = -∇·J_C + C_b + C_D, with p - p_R from the reference state
        pressure_anomaly = entry("pressure anomaly p - p_R", "Pa", cells,
            (model, pressure) -> pressure_anomaly!(out(:pressure_anomaly), model, require_reference(reference), pressure)),
        # ∇·J_C, J_C = (p - p_R) u
        div_pressure_flux = entry("divergence of J_C = (p - p_R) u", "W m⁻³", cells,
            (model, u, w, pressure_anomaly) -> pressure_flux_divergence!(out(:div_pressure_flux), model, faces, u, w, pressure_anomaly)),
        # b = g (ρ_R v - 1), v = 1/ρ
        reference_buoyancy = entry("buoyancy g (ρ_R v - 1) of the reference state", "m s⁻²", cells,
            (model, state, scratch) -> reference_buoyancy!(out(:reference_buoyancy), model, require_reference(reference), state, scratch)),
        # C_b = ρ b w
        buoyancy_conversion = entry("buoyancy conversion C_b = ρ b w", "W m⁻³", cells,
            (model, mass_flux_z, reference_buoyancy) -> buoyancy_conversion!(out(:buoyancy_conversion), model, faces, mass_flux_z, reference_buoyancy)),
        # C_D = (p - p_R) ∇·u
        pressure_dilatation = entry("pressure dilatation C_D = (p - p_R) ∇·u", "W m⁻³", cells,
            (model, pressure_anomaly, divergence) -> pressure_dilatation!(out(:pressure_dilatation), model, pressure_anomaly, divergence)),
        # C + ∇·J_C - C_b - C_D
        conversion_residual = entry("residual of C = -∇·J_C + C_b + C_D", "W m⁻³", cells,
            (model, conversion, div_pressure_flux, buoyancy_conversion, pressure_dilatation) -> conversion_residual!(out(:conversion_residual), model, conversion, div_pressure_flux, buoyancy_conversion, pressure_dilatation)),
        # integrals over the interior cells, ∫ρK dV, ∫ρP dV
        integrated_kinetic_energy = entry("integral of ρ K", "J m⁻¹", (), (model, ρK) -> domain_integral(model, ρK)),
        integrated_potential_energy = entry("integral of ρ P", "J m⁻¹", (), (model, ρP) -> domain_integral(model, ρP)),
        # ∫(∂ₜ(ρK) + ∂ₜ(ρP)) dV
        integrated_energy_tendency = entry("integral of the tendency of ρ (K + P)", "W m⁻¹", (),
            (model, dρK, dρP) -> domain_integral(model, dρK) + domain_integral(model, dρP)),
        # ∮(J_K^rev + J_K^irr + J_P^rev + J_P^irr)·dS, the sum of the divergences (telescoping to the boundary fluxes)
        integrated_boundary_flux = entry("integral of the divergences of the fluxes of K and P", "W m⁻¹", (),
            (model, div_kinetic_flux_rev, div_kinetic_flux_irr, div_potential_flux_rev, div_potential_flux_irr) ->
                domain_integral(model, div_kinetic_flux_rev) + domain_integral(model, div_kinetic_flux_irr) + domain_integral(model, div_potential_flux_rev) + domain_integral(model, div_potential_flux_irr)),
        # ∫VF dV
        integrated_vortex_work = entry("integral of the vortex force work", "W m⁻¹", (), (model, vortex_work) -> domain_integral(model, vortex_work)),
        # d/dt ∫(K + P) ρ dV + ∮(J_K + J_P)·dS - ∫VF dV: zero to round-off for the energy-consistent schemes
        integrated_budget_residual = entry("residual of the budget of ∫ρ (K + P) dV", "W m⁻¹", (),
            (integrated_energy_tendency, integrated_boundary_flux, integrated_vortex_work) -> integrated_energy_tendency + integrated_boundary_flux - integrated_vortex_work),
    )
end

"""
    field = interior_diagnostic(session, name)
    x, z = diagnostic_coordinates(model, name)

The diagnostic `name` of an open diagnostics session without its halo (a view), for plotting, and the coordinates
[m] of its points, from the grid location of the diagnostic (e.g. both wall faces for `w` in a bounded box).
"""
interior_diagnostic(session, name) = interior(session.model.domain, CookBooks.get(session, name))
diagnostic_coordinates(model, name) =
    coordinates(model.domain, (model.dx, model.dz), diagnostic_info(model)[name].location)
