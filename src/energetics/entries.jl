## DIAGNOSTICS OF THE ENERGY BUDGET
#
# The entries of the energy budget of energy.tex (kernels in kernels.jl), merged into the book of `diagnostics`
# (`add_energetics!`).

"""
    entries = energetics_entries(model, out, faces, reference)

The diagnostics of the energy budget of energy.tex, as a named tuple of `entry(long_name, units, location, recipe)`,
each below the mathematical expression of the diagnostic. The recipes write into the arrays `out(name)`, with the
scratch arrays `faces` on the x and z faces; with `out = nothing`, only the information of the entries is used. Σ is
the potential of the model (Σ = h + g z for FC2D, Σ = h(p_R) + (p - p_R)/ρ_R + g z for AN2D), with Σ_ϑ = ∂Σ/∂ϑ,
Σ_q = ∂Σ/∂q the conjugate variable and the modified chemical potential, J_ϑ, J_q the diffusive fluxes, σ_ϑ the
production of the conservative variable, τ the viscous stress and ρε_K = ∇·(u·τ) - u·∇·τ the viscous dissipation.
"""
energetics_entries(model, out, faces, reference) = let (; mgr) = model, cells = (Cell(), Cell()), x_faces = (Face(), Cell()),
                                                       z_faces = (Cell(), Face()), k = consvar_index(model)
    merge((;
        # energies, per unit mass: h(ϑ, q, p), P = h - p/ρ + g (z - z0), Σ - ϑ Σ_ϑ - q Σ_q, at the pressure and density of
        # the energy (`energy_pressure`, `energy_density`): p_R and ρ_R for AN2D, those of the state for FC2D
        specific_enthalpy = entry(model isa AN2D ? "specific enthalpy h at p_R" : "specific enthalpy h", "J kg⁻¹", cells,
            (model, scratch) -> specific_enthalpy!(out(:specific_enthalpy), model, energy_pressure(model, scratch), scratch.thermo)),
        potential_energy = entry(model isa AN2D ? "potential energy P = h - p_R/ρ_R + g (z - z0)" : "potential energy P = h - p/ρ + g (z - z0)",
                                 "J kg⁻¹", cells,
            (model, specific_enthalpy, density, scratch) ->
                potential_energy!(out(:potential_energy), model, specific_enthalpy, energy_density(model, density), energy_pressure(model, scratch))),
        bernoulli_potential = entry(model isa AN2D ? "Σ - ϑ Σ_ϑ - q Σ_q, B + g (z - z0) + φ" : "Σ - ϑ Σ_ϑ - q Σ_q, B + g (z - z0)", "J kg⁻¹", cells,
            (model, specific_enthalpy, scratch) -> bernoulli_potential!(out(:bernoulli_potential), model, specific_enthalpy, scratch)),
        # energy densities ρK and ρP
        ρK = entry("kinetic energy density ρ K", "J m⁻³", cells,
            (density, kinetic_energy) -> pointwise!(out(:ρK), mgr, *, density, kinetic_energy)),
        ρP = entry("potential energy density ρ P", "J m⁻³", cells,
            (density, potential_energy) -> pointwise!(out(:ρP), mgr, *, density, potential_energy)),
        # tendency of K, ∂ₜK = ℑx(u ∂ₜu) + ℑz(w ∂ₜw)
        kinetic_energy_rate = entry("tendency of K", "W kg⁻¹", cells,
            (model, u, w, du, dw) -> kinetic_energy_rate!(out(:kinetic_energy_rate), model, faces, (u, w), (du, dw))),
        # reversible flux of K, J_K^rev = ρ K u
        kinetic_flux_rev_x = entry("reversible flux of K along x, ρ K u", "W m⁻²", x_faces,
            (model, mass_flux_x, kinetic_energy) -> advective_flux!(out(:kinetic_flux_rev_x), model, Val(1), mass_flux_x, kinetic_energy)),
        kinetic_flux_rev_z = entry("reversible flux of K along z, ρ K w", "W m⁻²", z_faces,
            (model, mass_flux_z, kinetic_energy) -> advective_flux!(out(:kinetic_flux_rev_z), model, Val(2), mass_flux_z, kinetic_energy)),
        # irreversible flux of K, J_K^irr = -u·τ
        kinetic_flux_irr_x = entry("irreversible flux of K along x, -(u·τ)_x", "W m⁻²", x_faces,
            (model, scratch) -> stress_work_flux!(out(:kinetic_flux_irr_x), model, Val(1), scratch.irreversible.uτ_x)),
        kinetic_flux_irr_z = entry("irreversible flux of K along z, -(u·τ)_z", "W m⁻²", z_faces,
            (model, scratch) -> stress_work_flux!(out(:kinetic_flux_irr_z), model, Val(2), scratch.irreversible.uτ_z)),
        # reversible flux of P, J_P^rev = ρ Σ u = ρu (Σ - ϑ Σ_ϑ - q Σ_q) + ρϑu Σ_ϑ + ρqu Σ_q
        potential_flux_rev_x = entry("reversible flux of P along x, ρ Σ u", "W m⁻²", x_faces,
            (model, scratch, mass_flux_x, bernoulli_potential, conjugate_variable, modified_chemical_potential) ->
                potential_flux_rev!(out(:potential_flux_rev_x), model, Val(1), mass_flux_x, scratch.advection.mu, k, bernoulli_potential, conjugate_variable, modified_chemical_potential)),
        potential_flux_rev_z = entry("reversible flux of P along z, ρ Σ w", "W m⁻²", z_faces,
            (model, scratch, mass_flux_z, bernoulli_potential, conjugate_variable, modified_chemical_potential) ->
                potential_flux_rev!(out(:potential_flux_rev_z), model, Val(2), mass_flux_z, scratch.advection.mw, k, bernoulli_potential, conjugate_variable, modified_chemical_potential)),
        # irreversible flux of P, J_P^irr = Σ_ϑ J_ϑ + Σ_q J_q
        potential_flux_irr_x = entry("irreversible flux of P along x, Σ_ϑ J_ϑ + Σ_q J_q", "W m⁻²", x_faces,
            (model, consflux_x, qflux_x, conjugate_variable, modified_chemical_potential) ->
                potential_flux_irr!(out(:potential_flux_irr_x), model, Val(1), consflux_x, qflux_x, conjugate_variable, modified_chemical_potential)),
        potential_flux_irr_z = entry("irreversible flux of P along z, Σ_ϑ J_ϑ + Σ_q J_q", "W m⁻²", z_faces,
            (model, consflux_z, qflux_z, conjugate_variable, modified_chemical_potential) ->
                potential_flux_irr!(out(:potential_flux_irr_z), model, Val(2), consflux_z, qflux_z, conjugate_variable, modified_chemical_potential)),
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
            (model, scratch, conjugate_variable) -> gradient_product!(out(:conversion_consvar), model, faces, tracer_flux(scratch.advection, k), conjugate_variable)),
        # C_q = -ρ q u·∇Σ_q
        conversion_composition = entry("conversion C_q from P to K", "W m⁻³", cells,
            (model, scratch, modified_chemical_potential) -> gradient_product!(out(:conversion_composition), model, faces, tracer_flux(scratch.advection, k + 1), modified_chemical_potential)),
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
        # b = g (ρ_R v - 1), v = 1/ρ (FC2D), v = v(ϑ, q, p_R) (AN2D)
        reference_buoyancy = entry("buoyancy g (ρ_R v - 1) of the reference state", "m s⁻²", cells,
            (model, state, scratch) -> reference_buoyancy!(out(:reference_buoyancy), model, require_reference(reference), state, scratch)),
        # C_b = ρ b w
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
    ), energy_tendency_entries(model, out))
end

"""
    p = energy_pressure(model, scratch)
    ρ = energy_density(model, density)

The pressure and density of the energies P = h - p/ρ + g (z - z0) and Σ: the reference profiles p_R(z), ρ_R(z) for
AN2D, the pressure of the state and its `density` diagnostic for FC2D.
"""
energy_pressure(model::AN2D, scratch) = model.reference.p_R
energy_pressure(model::FC2D, scratch) = scratch.thermo.p
energy_density(model::AN2D, density) = model.reference.ρ_R
energy_density(model::FC2D, density) = density

"""
    entries = energy_tendency_entries(model, out)

The entries of the tendencies of ρK and ρP, which differ between the models: ∂ₜρ = 0 for AN2D, where ρ = ρ_R.
"""
energy_tendency_entries(model::AN2D, out) = let (; mgr) = model, cells = (Cell(), Cell())
    (;
        # tendency of ρ_R K = ρ_R ∂ₜK
        dρK = entry("tendency of ρ K", "W m⁻³", cells,
            (density, kinetic_energy_rate) -> pointwise!(out(:dρK), mgr, *, density, kinetic_energy_rate)),
        # tendency of ρ_R P = Σ_ϑ ∂ₜ(ρ_R ϑ) + Σ_q ∂ₜ(ρ_R q)
        dρP = entry("tendency of ρ P", "W m⁻³", cells,
            (model, conjugate_variable, modified_chemical_potential, dρconsvar, dρq) ->
                potential_energy_tendency!(out(:dρP), model, conjugate_variable, modified_chemical_potential, dρconsvar, dρq)),
    )
end
energy_tendency_entries(model::FC2D, out) = let cells = (Cell(), Cell())
    (;
        # tendency of ρK = ρ ∂ₜK + K ∂ₜρ
        dρK = entry("tendency of ρ K", "W m⁻³", cells,
            (model, density, kinetic_energy, kinetic_energy_rate, dρ) -> kinetic_energy_tendency!(out(:dρK), model, density, kinetic_energy, kinetic_energy_rate, dρ)),
        # tendency of ρP = (Σ - ϑ Σ_ϑ - q Σ_q) ∂ₜρ + Σ_ϑ ∂ₜ(ρϑ) + Σ_q ∂ₜ(ρq)
        dρP = entry("tendency of ρ P", "W m⁻³", cells,
            (model, bernoulli_potential, conjugate_variable, modified_chemical_potential, dρ, dρconsvar, dρq) ->
                potential_energy_tendency!(out(:dρP), model, bernoulli_potential, conjugate_variable, modified_chemical_potential, dρ, dρconsvar, dρq)),
    )
end

"""
    k = consvar_index(model)

Index of the advected density ρϑ among those of the advection scratch of the tendencies: (ρ_R ϑ, ρ_R q) for AN2D,
(ρ, ρ ϑ, ρ q) for FC2D. That of ρq is k + 1.
"""
consvar_index(::AN2D) = 1
consvar_index(::FC2D) = 2
