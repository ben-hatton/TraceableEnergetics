## DIAGNOSTICS OF EITHER MODEL
#
# The diagnostics of AN2D and FC2D are those of the model packages, the energy budget of energy.tex (kernels in
# energetics.jl) and the Lorenz reference state of available potential energy with the available energy of its parcels
# (anelastic_reference.jl and anelastic_available_energy.jl for AN2D, compressible_reference.jl and
# compressible_available_energy.jl for FC2D). They are full arrays with halo, for post-processing; for plotting,
# `interior_diagnostic` strips the halo and `diagnostic_coordinates` gives the coordinates of the remaining points, from
# the grid location of the diagnostic.

"""
    book = diagnostics(model; reference = default, reference_state = nothing)
    info = diagnostic_info(model; reference_state = nothing)

`CookBook` of the diagnostics of an AN2D or FC2D model, opened with `open(book; state, dstate, scratch, model)`, and
the long name, units and grid location of each diagnostic: one `Face()`/`Cell()` per dimension, e.g. `(Cell(), Cell())`
cells, `(Face(), Cell())` u-faces, `()` for a number. They are those of `CFAnelastics.diagnostics` and
`CFCompressible.diagnostics`, and the energy budget of energy.tex, listed in `energetics_entries` below. K and P are
specific energies [J kg⁻¹]; ρK, ρP and every term of the budget are per unit volume [W m⁻³] (ρ = ρ_R for AN2D); the
`integrated_` diagnostics are their integrals over the interior cells, per unit length along y.

The geopotential is g (z - z0), with z0 the origin `model.space.z0` [m] of the tank. The manipulated conversion needs the
reference state (`AnelasticReference` with the density ρ_R(z) and the hydrostatic pressure p_R(z)): the model's own for
AN2D, the keyword `reference` (usually that of the matching AN2D, on the same grid) for FC2D, which has none.

With a `reference_state` (an `AnelasticLorenzReference` or a `CompressibleLorenzReference` of the model), the book
also computes it for the state of each session, once, and has its diagnostics and those of the available energy of its
parcels (`reference_state_entries`): on the cells, profiles on the z axis (location `(nothing, Cell())`), and
numbers.

Build the book once, as each session of it overwrites the same output arrays.
"""
function diagnostics(model::AN2D{F}; reference = model.reference, reference_state = nothing) where F
    book = CFAnelastics.diagnostics(model)
    add_energetics!(book, F, model, reference)
    return add_reference_state!(book, F, model, reference_state)
end
function diagnostics(model::FC2D{F}; reference = nothing, reference_state = nothing) where F
    book = CFCompressible.diagnostics(model)
    add_energetics!(book, F, model, reference)
    return add_reference_state!(book, F, model, reference_state)
end
diagnostic_info(model::AN2D; reference_state = nothing) =
    merge(CFAnelastics.diagnostic_info(model), map(first, energetics_entries(model, nothing, nothing, nothing)), reference_state_info(model, reference_state))
diagnostic_info(model::FC2D; reference_state = nothing) =
    merge(CFCompressible.diagnostic_info(model), map(first, energetics_entries(model, nothing, nothing, nothing)), reference_state_info(model, reference_state))
reference_state_info(model, ::Nothing) = (;)
reference_state_info(model, ref) =
    map(first, merge(reference_state_entries(model, ref, nothing), available_energy_budget_entries(model, nothing, nothing),
                     reference_tendency_entries(model, nothing)))

"""
    entry = entry(long_name, units, location, recipe)

A diagnostic: its `info = (; long_name, units, location)` and its `recipe` for a `CookBook`.
"""
entry(long_name, units, location, recipe) = ((; long_name, units, location), recipe)

"""
    book = add_energetics!(book, F, model, reference)

Adds the recipes of `energetics_entries` to `book`. The output array of each diagnostic is allocated on its first use,
on its grid location, in the float type `F`; `faces` are the scratch arrays on the x and z faces shared by the terms
built from face products.
"""
function add_energetics!(book, F, model, reference)
    (; mgr, domain) = model
    entries = energetics_entries(model, nothing, nothing, nothing)
    cache = Dict{Symbol, Any}()
    out(name) = get!(() -> alloc(F, mgr, domain, first(entries[name]).location), cache, name)::Matrix{F}
    faces = (alloc(F, mgr, domain, (Face(), Cell())), alloc(F, mgr, domain, (Cell(), Face())))
    merge!(book.recipes, pairs(map(last, energetics_entries(model, out, faces, reference))))
    return book
end

"""
    book = add_reference_state!(book, F, model, ref)

Adds to `book` the recipe `reference_state`, which computes the reference state `ref` for the state of the session, the
recipe `reference_state_tendency`, which computes its tendency from the diabatic rates (`reference_tendency!`), and
the recipes of `reference_state_entries`, `available_energy_budget_entries` and `reference_tendency_entries`; nothing if
`ref` is `nothing`. The output array of each diagnostic on the cells is allocated on its first use, in the float type
`F`, as for the energetics.
"""
add_reference_state!(book, F, model, ::Nothing) = book
function add_reference_state!(book, F, model, ref)
    (; mgr, domain) = model
    entries = merge(reference_state_entries(model, ref, nothing), available_energy_budget_entries(model, nothing, nothing),
                    reference_tendency_entries(model, nothing))
    cache = Dict{Symbol, Any}()
    out(name) = get!(() -> alloc(F, mgr, domain, first(entries[name]).location), cache, name)::Matrix{F}
    # scratch arrays on the x and z faces of the terms built from face products, apart from those of the energetics
    faces = (alloc(F, mgr, domain, (Face(), Cell())), alloc(F, mgr, domain, (Cell(), Face())))
    book.recipes[:reference_state] = (model, state) -> reference_state!(ref, model, state)
    # the tendency of the reference state, from the N_*⁻²-weighted means of v̇ over the slots, after the reference state
    book.recipes[:reference_state_tendency] = (model, reference_state, consvar_diabatic_rate, composition_diabatic_rate) ->
        reference_tendency!(reference_state, model, consvar_diabatic_rate, composition_diabatic_rate)
    merge!(book.recipes, pairs(map(last, reference_state_entries(model, ref, out))))
    merge!(book.recipes, pairs(map(last, available_energy_budget_entries(model, out, faces))))
    merge!(book.recipes, pairs(map(last, reference_tendency_entries(model, out))))
    return book
end

"""
    entries = energetics_entries(model, out, faces, reference)

The diagnostics of the energy budget of energy.tex, as a named tuple of `entry(long_name, units, location, recipe)`,
each below the mathematical expression of the diagnostic. The recipes write into the arrays `out(name)`, with the
scratch arrays `faces` on the x and z faces; with `out = nothing`, only the information of the entries is used. Σ is
the potential of the model (Σ = h + g z for FC2D, Σ = h(p_R) + (p - p_R)/ρ_R + g z for AN2D), with Σ_ϑ = ∂Σ/∂ϑ,
Σ_q = ∂Σ/∂q the conjugate variable and the modified chemical potential, J_ϑ, J_q the diffusive fluxes, σ_ϑ the
production of the conservative variable, τ the viscous stress and ρε_K = ∇·(u·τ) - u·∇·τ the viscous dissipation.
"""
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
    entries = reference_state_entries(model, ref, out)

The diagnostics of the Lorenz reference state `ref` of available potential energy and of the available energy of its
parcels, as a named tuple of `entry(long_name, units, location, recipe)`: on the cells, on the z axis (location
`(nothing, Cell())`) or a number (location `()`). The available energy (AE) of a parcel at height z and pressure p, of
reference height z_*, is
    Π = Σ(ϑ, q, z, p) - Σ(ϑ, q, z_*, P_*(z_*)) - (p - P_*(z)) Σ_p(ϑ, q, z, p) = Π₁ + Π₂   [J kg⁻¹],
with the available compressible energy (ACE)
    Π₁ = Σ(ϑ, q, z, p) - Σ(ϑ, q, z, P_*(z)) - (p - P_*(z)) Σ_p(ϑ, q, z, p),
the compression from P_*(z) to p at fixed z (zero for AN2D), and the available potential energy (APE)
    Π₂ = Σ(ϑ, q, z, P_*(z)) - Σ(ϑ, q, z_*, P_*(z_*)),
the displacement from z_* to z through the reference state; Σ is the potential of the model (h(p) + g z for FC2D,
h(P_R) + (p - P_R)/ρ_R + g z for AN2D) and Σ_p = ∂Σ/∂p = 1/ρ. The diagnostics are AE, ACE and APE of each parcel
[J kg⁻¹], their integrals Σ_n m_n Π_n, Σ_n m_n Π₁_n, Σ_n m_n Π₂_n [J m⁻¹] (`integrated_AE`, `integrated_ACE`,
`integrated_APE`), computed parcel by parcel, and the background potential energy ∫ρ (P - Π) dV [J m⁻¹]. The recipes
read the reference state, computed once per diagnostics session by the recipe `reference_state`
(`add_reference_state!`); those on the cells write into the arrays `out(name)`
(with `out = nothing`, only the information of the entries is used).
"""
function reference_state_entries(model::AN2D{F}, ref, out) where F
    cells, rows = (Cell(), Cell()), (nothing, Cell())
    return (;
        # z_* = argmin S(z) over 0 ≤ z ≤ L_z, with S(z) = h(P_R(z), ϑ, q) + π_*(z) + g z
        z_star = entry("reference height z_*, minimiser of the static energy along P_*", "m", cells,
            (model, reference_state) -> reference_heights!(out(:z_star), model, reference_state)),
        wall_minima = entry("number of parcels whose reference height is at a wall", "1", (),
            (model, z_star) -> F(count_wall_minima(model, z_star))),
        # ACE: Π₁ = Σ(z, p) - Σ(z, P_*(z)) - (p - P_*(z))/ρ_R = 0, as Σ is linear in p
        ACE = entry("available compressible energy (ACE) Π₁, zero (Σ linear in p)", "J kg⁻¹", cells,
            (model, reference_state) -> available_compressible_energy!(out(:ACE), model, reference_state)),
        # APE: Π₂ = h(P_R(z)) - h(P_R(z_*)) + π_*(z) - π_*(z_*) + g (z - z_*) = S(z) - S(z_*)
        APE = entry("available potential energy (APE) Π₂ = S(z) - S(z_*)", "J kg⁻¹", cells,
            (model, reference_state, z_star) -> available_potential_energy!(out(:APE), model, reference_state, z_star)),
        # AE: Π = Π₁ + Π₂
        AE = entry("available energy (AE) Π = Π₁ + Π₂", "J kg⁻¹", cells, (model, ACE, APE) -> pointwise!(out(:AE), model.mgr, +, ACE, APE)),
        # Σ_n m_n Π₁_n, Σ_n m_n Π₂_n, Σ_n m_n Π_n
        integrated_ACE = entry("integral of ρ Π₁", "J m⁻¹", (), (model, reference_state, ACE) -> parcel_integral(model, reference_state, ACE)),
        integrated_APE = entry("integral of ρ Π₂", "J m⁻¹", (), (model, reference_state, APE) -> parcel_integral(model, reference_state, APE)),
        integrated_AE = entry("integral of ρ Π", "J m⁻¹", (), (model, reference_state, AE) -> parcel_integral(model, reference_state, AE)),
        p_star = entry("reference pressure P_*", "Pa", rows, reference_state -> reference_state.p_star),
        v_star = entry("reference specific volume v_*", "m³ kg⁻¹", rows, reference_state -> reference_state.v_star),
        # background energy ∫ρ (P - Π) dV = ∫ρ P dV - Σ_n m_n Π_n, with ∫ρ P dV from the energetics
        background_potential_energy = entry("background potential energy ∫ρ (P - Π) dV", "J m⁻¹", (),
            (integrated_potential_energy, integrated_AE) -> integrated_potential_energy - integrated_AE),
        certificate_violations = entry("number of parcels with a negative reduced cost", "1", (),
            reference_state -> F(reference_state.certificate_violations)),
        # the transportation simplex of ExactRearrangement (no pivots and one pass for SortedRearrangement)
        pivots = entry("number of pivots of the transportation simplex", "1", (), reference_state -> F(reference_state.pivots)),
        pricing_passes = entry("number of passes over the parcel-slot pairs", "1", (),
            reference_state -> F(reference_state.pricing_passes)),
        # Σ_nk x_nk c_nk of the sorted placement less that of the placement, zero for SortedRearrangement
        sorted_energy_excess = entry("underestimate of E_P - E_* by the sorted placement", "J m⁻¹", (),
            reference_state -> reference_state.sorted_energy_excess),
    )
end
function reference_state_entries(model::FC2D{F}, ref, out) where F
    cells, rows = (Cell(), Cell()), (nothing, Cell())
    return (;
        # z_* = argmin S(z) over 0 ≤ z ≤ L_z, with S(z) = h(P_*(z), ϑ, q) + g z
        z_star = entry("reference height z_*, minimiser of the static energy along P_*", "m", cells,
            (model, reference_state) -> reference_heights!(out(:z_star), model, reference_state)),
        wall_minima = entry("number of parcels whose reference height is at a wall", "1", (),
            (model, z_star) -> F(count_wall_minima(model, z_star))),
        # ACE: Π₁ = h(p) - h(P_*(z)) - (p - P_*(z)) v
        ACE = entry("available compressible energy (ACE) Π₁ = h(p) - h(P_*(z)) - (p - P_*(z)) v", "J kg⁻¹", cells,
            (model, reference_state) -> available_compressible_energy!(out(:ACE), model, reference_state)),
        # APE: Π₂ = h(P_*(z)) - h(P_*(z_*)) + g (z - z_*) = S(z) - S(z_*)
        APE = entry("available potential energy (APE) Π₂ = S(z) - S(z_*)", "J kg⁻¹", cells,
            (model, reference_state, z_star) -> available_potential_energy!(out(:APE), model, reference_state, z_star)),
        # AE: Π = Π₁ + Π₂
        AE = entry("available energy (AE) Π = Π₁ + Π₂", "J kg⁻¹", cells, (model, ACE, APE) -> pointwise!(out(:AE), model.mgr, +, ACE, APE)),
        # Σ_n m_n Π₁_n, Σ_n m_n Π₂_n, Σ_n m_n Π_n
        integrated_ACE = entry("integral of ρ Π₁", "J m⁻¹", (), (model, reference_state, ACE) -> parcel_integral(model, reference_state, ACE)),
        integrated_APE = entry("integral of ρ Π₂", "J m⁻¹", (), (model, reference_state, APE) -> parcel_integral(model, reference_state, APE)),
        integrated_AE = entry("integral of ρ Π", "J m⁻¹", (), (model, reference_state, AE) -> parcel_integral(model, reference_state, AE)),
        p_star = entry("reference pressure P_*", "Pa", rows, reference_state -> reference_state.p_star),
        v_star = entry("reference specific volume v_*", "m³ kg⁻¹", rows, reference_state -> reference_state.v_star),
        p_top = entry("pressure P_top at the top of the reference column", "Pa", (), reference_state -> reference_state.p_top),
        # (Σ_k V_k/L_x - L_z)/L_z
        volume_residual = entry("relative excess of the height of the reference column over L_z", "1", (),
            reference_state -> reference_state.volume_residual),
        # background energy ∫ρ (P - Π) dV = ∫ρ P dV - Σ_n m_n Π_n, with ∫ρ P dV from the energetics
        background_potential_energy = entry("background potential energy ∫ρ (P - Π) dV", "J m⁻¹", (),
            (integrated_potential_energy, integrated_AE) -> integrated_potential_energy - integrated_AE),
        certificate_violations = entry("number of parcels with a negative reduced cost", "1", (),
            reference_state -> F(reference_state.certificate_violations)),
        # the transportation simplex of ExactRearrangement (no pivots and one pass for SortedRearrangement)
        pivots = entry("number of pivots of the transportation simplex", "1", (), reference_state -> F(reference_state.pivots)),
        pricing_passes = entry("number of passes over the parcel-slot pairs", "1", (),
            reference_state -> F(reference_state.pricing_passes)),
        top_pressure_iterations = entry("number of placements alternating with the top pressure", "1", (),
            reference_state -> F(reference_state.top_pressure_iterations)),
        # G = Σ_nk x_nk c_nk - P_top L_x L_z of the sorted placement less that of the placement, zero for SortedRearrangement
        sorted_energy_excess = entry("underestimate of E_P - E_* by the sorted placement", "J m⁻¹", (),
            reference_state -> reference_state.sorted_energy_excess),
    )
end

"""
    k = consvar_index(model)

Index of the advected density ρϑ among those of the advection scratch of the tendencies: (ρ_R ϑ, ρ_R q) for AN2D,
(ρ, ρ ϑ, ρ q) for FC2D. That of ρq is k + 1.
"""
consvar_index(::AN2D) = 1
consvar_index(::FC2D) = 2

"""
    entries = available_energy_budget_entries(model, out, faces)

The diagnostics of the budgets of the available compressible energy (ACE) Π₁ and the available potential energy (APE) Π₂
(kernels in available_energy_budget.jl), as a named tuple of `entry(long_name, units, location, recipe)`, the same for
AN2D and FC2D (the ACE terms are zero for AN2D, apart from those of the pressure flux J_C* = (p - P_*) u). With the
potentials Σ_ϑ, Σ_q of the model at the parcel's state (z, p), at the intermediate state (z, P_*(z)) (Σ̃, `_at_p_star`)
and at the reference state (z_*, P_*(z_*)) (Σ*, `_at_z_star`), their differences Δ₁Σ = Σ - Σ̃ (`_ACE`) and
Δ₂Σ = Σ̃ - Σ* (`_APE`), and the Bernoulli coefficients B₁ = Π₁ + (p - P_*)/ρ - ϑ Δ₁Σ_ϑ - q Δ₁Σ_q,
B₂ = Π₂ - ϑ Δ₂Σ_ϑ - q Δ₂Σ_q, the budgets at a fixed reference state are, at each cell to round-off,
    ∂ₜ(ρΠ₁) = -∇·(J_ACE^rev + J_ACE^irr) - C_ACE - (Q_ϑ - Q̃_ϑ) - (Q_q - Q̃_q) + Q - Q̃,
    ∂ₜ(ρΠ₂) = -∇·(J_APE^rev + J_APE^irr) - C_APE - (Q̃_ϑ - Q*_ϑ) - (Q̃_q - Q*_q) + Q̃ - Q*,
and C = C_ACE + C_APE + C_bg. The terms in ∂ₜP_*, and the full tendencies, are those of `reference_tendency_entries`.
The forms of the energy notes, C_b,1*, C_b,2*,
C_D* and ∇·J_C*, are diagnosed with the residuals of the manipulation, C_ACE - C_b,1* - C_D* + ∇·J_C* and
C_APE - C_b,2*. The recipes write into the arrays `out(name)`, with the scratch arrays `faces` on the x and z faces; with
`out = nothing`, only the information of the entries is used.
"""
available_energy_budget_entries(model, out, faces) = let (; mgr) = model, cells = (Cell(), Cell()), x_faces = (Face(), Cell()),
                                                         z_faces = (Cell(), Face()), k = consvar_index(model)
    (;
        # Δ₁Σ_ϑ = Σ_ϑ(z, p) - Σ_ϑ(z, P_*(z)) = ∫ ∂v/∂ϑ dp from P_*(z) to p, and the same for Σ_q
        conjugate_variable_ACE = entry("Σ_ϑ - Σ̃_ϑ, from the intermediate state (z, P_*(z)) to (z, p)", "J kg⁻¹ / consvar", cells,
            (model, reference_state) -> compression_potential!(out(:conjugate_variable_ACE), model, reference_state, 3)),
        modified_chemical_potential_ACE = entry("Σ_q - Σ̃_q, from the intermediate state (z, P_*(z)) to (z, p)", "J kg⁻¹", cells,
            (model, reference_state) -> compression_potential!(out(:modified_chemical_potential_ACE), model, reference_state, 4)),
        # Δ₂Σ_ϑ = Σ_ϑ(z, P_*(z)) - Σ_ϑ(z_*, P_*(z_*)) = ∫ ∂v/∂ϑ dp from P_*(z_*) to P_*(z), and the same for Σ_q
        conjugate_variable_APE = entry("Σ̃_ϑ - Σ*_ϑ, from the reference state (z_*, P_*(z_*)) to (z, P_*(z))", "J kg⁻¹ / consvar", cells,
            (model, reference_state, z_star) -> displacement_potential!(out(:conjugate_variable_APE), model, reference_state, z_star, 3)),
        modified_chemical_potential_APE = entry("Σ̃_q - Σ*_q, from the reference state (z_*, P_*(z_*)) to (z, P_*(z))", "J kg⁻¹", cells,
            (model, reference_state, z_star) -> displacement_potential!(out(:modified_chemical_potential_APE), model, reference_state, z_star, 4)),
        # Σ̃_ϑ = Σ_ϑ - Δ₁Σ_ϑ, Σ̃_q = Σ_q - Δ₁Σ_q
        conjugate_variable_at_p_star = entry("conjugate variable Σ̃_ϑ at the intermediate state (z, P_*(z))", "J kg⁻¹ / consvar", cells,
            (conjugate_variable, conjugate_variable_ACE) -> pointwise!(out(:conjugate_variable_at_p_star), mgr, -, conjugate_variable, conjugate_variable_ACE)),
        modified_chemical_potential_at_p_star = entry("modified chemical potential Σ̃_q at the intermediate state (z, P_*(z))", "J kg⁻¹", cells,
            (modified_chemical_potential, modified_chemical_potential_ACE) ->
                pointwise!(out(:modified_chemical_potential_at_p_star), mgr, -, modified_chemical_potential, modified_chemical_potential_ACE)),
        # Σ*_ϑ = Σ̃_ϑ - Δ₂Σ_ϑ, Σ*_q = Σ̃_q - Δ₂Σ_q
        conjugate_variable_at_z_star = entry("conjugate variable Σ*_ϑ at the reference state (z_*, P_*(z_*))", "J kg⁻¹ / consvar", cells,
            (conjugate_variable_at_p_star, conjugate_variable_APE) -> pointwise!(out(:conjugate_variable_at_z_star), mgr, -, conjugate_variable_at_p_star, conjugate_variable_APE)),
        modified_chemical_potential_at_z_star = entry("modified chemical potential Σ*_q at the reference state (z_*, P_*(z_*))", "J kg⁻¹", cells,
            (modified_chemical_potential_at_p_star, modified_chemical_potential_APE) ->
                pointwise!(out(:modified_chemical_potential_at_z_star), mgr, -, modified_chemical_potential_at_p_star, modified_chemical_potential_APE)),
        # p - P_*(z)
        pressure_anomaly_star = entry("pressure anomaly p - P_* from the reference pressure", "Pa", cells,
            (model, reference_state, pressure) -> pressure_anomaly!(out(:pressure_anomaly_star), model, (; p_R = reference_state.p_star), pressure)),
        # B₁ = Π₁ + (p - P_*)/ρ - ϑ Δ₁Σ_ϑ - q Δ₁Σ_q, B₂ = Π₂ - ϑ Δ₂Σ_ϑ - q Δ₂Σ_q
        bernoulli_potential_ACE = entry("B₁ = Π₁ + (p - P_*)/ρ - ϑ (Σ_ϑ - Σ̃_ϑ) - q (Σ_q - Σ̃_q)", "J kg⁻¹", cells,
            (model, scratch, ACE, pressure_anomaly_star, density, conjugate_variable_ACE, modified_chemical_potential_ACE) ->
                compression_bernoulli!(out(:bernoulli_potential_ACE), model, ACE, pressure_anomaly_star, density, scratch.thermo,
                                       conjugate_variable_ACE, modified_chemical_potential_ACE)),
        bernoulli_potential_APE = entry("B₂ = Π₂ - ϑ (Σ̃_ϑ - Σ*_ϑ) - q (Σ̃_q - Σ*_q)", "J kg⁻¹", cells,
            (model, scratch, APE, conjugate_variable_APE, modified_chemical_potential_APE) ->
                displacement_bernoulli!(out(:bernoulli_potential_APE), model, APE, scratch.thermo, conjugate_variable_APE, modified_chemical_potential_APE)),
        # reversible fluxes, J_ACE^rev = ρu B₁ + ρϑu Δ₁Σ_ϑ + ρqu Δ₁Σ_q (= ρΠ₁u + J_C*) and J_APE^rev (= ρΠ₂u)
        ACE_flux_rev_x = entry("reversible flux of ACE along x, ρu B₁ + ρϑu Δ₁Σ_ϑ + ρqu Δ₁Σ_q", "W m⁻²", x_faces,
            (model, scratch, mass_flux_x, bernoulli_potential_ACE, conjugate_variable_ACE, modified_chemical_potential_ACE) ->
                potential_flux_rev_x!(out(:ACE_flux_rev_x), model, mass_flux_x, scratch.advection.mu, k, bernoulli_potential_ACE,
                                      conjugate_variable_ACE, modified_chemical_potential_ACE)),
        ACE_flux_rev_z = entry("reversible flux of ACE along z, ρw B₁ + ρϑw Δ₁Σ_ϑ + ρqw Δ₁Σ_q", "W m⁻²", z_faces,
            (model, scratch, mass_flux_z, bernoulli_potential_ACE, conjugate_variable_ACE, modified_chemical_potential_ACE) ->
                potential_flux_rev_z!(out(:ACE_flux_rev_z), model, mass_flux_z, scratch.advection.mw, k, bernoulli_potential_ACE,
                                      conjugate_variable_ACE, modified_chemical_potential_ACE)),
        APE_flux_rev_x = entry("reversible flux of APE along x, ρu B₂ + ρϑu Δ₂Σ_ϑ + ρqu Δ₂Σ_q", "W m⁻²", x_faces,
            (model, scratch, mass_flux_x, bernoulli_potential_APE, conjugate_variable_APE, modified_chemical_potential_APE) ->
                potential_flux_rev_x!(out(:APE_flux_rev_x), model, mass_flux_x, scratch.advection.mu, k, bernoulli_potential_APE,
                                      conjugate_variable_APE, modified_chemical_potential_APE)),
        APE_flux_rev_z = entry("reversible flux of APE along z, ρw B₂ + ρϑw Δ₂Σ_ϑ + ρqw Δ₂Σ_q", "W m⁻²", z_faces,
            (model, scratch, mass_flux_z, bernoulli_potential_APE, conjugate_variable_APE, modified_chemical_potential_APE) ->
                potential_flux_rev_z!(out(:APE_flux_rev_z), model, mass_flux_z, scratch.advection.mw, k, bernoulli_potential_APE,
                                      conjugate_variable_APE, modified_chemical_potential_APE)),
        # irreversible fluxes, J_ACE^irr = Δ₁Σ_ϑ J_ϑ + Δ₁Σ_q J_q, J_APE^irr = Δ₂Σ_ϑ J_ϑ + Δ₂Σ_q J_q
        ACE_flux_irr_x = entry("irreversible flux of ACE along x, Δ₁Σ_ϑ J_ϑ + Δ₁Σ_q J_q", "W m⁻²", x_faces,
            (model, consflux_x, qflux_x, conjugate_variable_ACE, modified_chemical_potential_ACE) ->
                potential_flux_irr_x!(out(:ACE_flux_irr_x), model, consflux_x, qflux_x, conjugate_variable_ACE, modified_chemical_potential_ACE)),
        ACE_flux_irr_z = entry("irreversible flux of ACE along z, Δ₁Σ_ϑ J_ϑ + Δ₁Σ_q J_q", "W m⁻²", z_faces,
            (model, consflux_z, qflux_z, conjugate_variable_ACE, modified_chemical_potential_ACE) ->
                potential_flux_irr_z!(out(:ACE_flux_irr_z), model, consflux_z, qflux_z, conjugate_variable_ACE, modified_chemical_potential_ACE)),
        APE_flux_irr_x = entry("irreversible flux of APE along x, Δ₂Σ_ϑ J_ϑ + Δ₂Σ_q J_q", "W m⁻²", x_faces,
            (model, consflux_x, qflux_x, conjugate_variable_APE, modified_chemical_potential_APE) ->
                potential_flux_irr_x!(out(:APE_flux_irr_x), model, consflux_x, qflux_x, conjugate_variable_APE, modified_chemical_potential_APE)),
        APE_flux_irr_z = entry("irreversible flux of APE along z, Δ₂Σ_ϑ J_ϑ + Δ₂Σ_q J_q", "W m⁻²", z_faces,
            (model, consflux_z, qflux_z, conjugate_variable_APE, modified_chemical_potential_APE) ->
                potential_flux_irr_z!(out(:APE_flux_irr_z), model, consflux_z, qflux_z, conjugate_variable_APE, modified_chemical_potential_APE)),
        # ∇·J_ACE^rev, ∇·J_ACE^irr, ∇·J_APE^rev, ∇·J_APE^irr
        div_ACE_flux_rev = entry("divergence of the reversible flux of ACE", "W m⁻³", cells,
            (model, ACE_flux_rev_x, ACE_flux_rev_z) -> divergence!(out(:div_ACE_flux_rev), model, ACE_flux_rev_x, ACE_flux_rev_z)),
        div_ACE_flux_irr = entry("divergence of the irreversible flux of ACE", "W m⁻³", cells,
            (model, ACE_flux_irr_x, ACE_flux_irr_z) -> divergence!(out(:div_ACE_flux_irr), model, ACE_flux_irr_x, ACE_flux_irr_z)),
        div_APE_flux_rev = entry("divergence of the reversible flux of APE", "W m⁻³", cells,
            (model, APE_flux_rev_x, APE_flux_rev_z) -> divergence!(out(:div_APE_flux_rev), model, APE_flux_rev_x, APE_flux_rev_z)),
        div_APE_flux_irr = entry("divergence of the irreversible flux of APE", "W m⁻³", cells,
            (model, APE_flux_irr_x, APE_flux_irr_z) -> divergence!(out(:div_APE_flux_irr), model, APE_flux_irr_x, APE_flux_irr_z)),
        # C_ACE = -ρu·∇B₁ - ρϑu·∇(Δ₁Σ_ϑ) - ρqu·∇(Δ₁Σ_q), C_APE the same with B₂, Δ₂Σ
        conversion_ACE = entry("conversion C_ACE from ACE to K, -ρu·∇B₁ - ρϑu·∇Δ₁Σ_ϑ - ρqu·∇Δ₁Σ_q", "W m⁻³", cells,
            (model, scratch, mass_flux_x, mass_flux_z, bernoulli_potential_ACE, conjugate_variable_ACE, modified_chemical_potential_ACE) ->
                bernoulli_conversion!(out(:conversion_ACE), model, faces, (mass_flux_x, mass_flux_z), (scratch.advection.mu, scratch.advection.mw), k,
                                      bernoulli_potential_ACE, conjugate_variable_ACE, modified_chemical_potential_ACE)),
        conversion_APE = entry("conversion C_APE from APE to K, -ρu·∇B₂ - ρϑu·∇Δ₂Σ_ϑ - ρqu·∇Δ₂Σ_q", "W m⁻³", cells,
            (model, scratch, mass_flux_x, mass_flux_z, bernoulli_potential_APE, conjugate_variable_APE, modified_chemical_potential_APE) ->
                bernoulli_conversion!(out(:conversion_APE), model, faces, (mass_flux_x, mass_flux_z), (scratch.advection.mu, scratch.advection.mw), k,
                                      bernoulli_potential_APE, conjugate_variable_APE, modified_chemical_potential_APE)),
        # C_bg = C - C_ACE - C_APE, zero in the continuous equations
        background_conversion = entry("conversion C_bg = C - C_ACE - C_APE from the background to K (zero when continuous)", "W m⁻³", cells,
            (conversion, conversion_ACE, conversion_APE) ->
                pointwise!(out(:background_conversion), mgr, (c, c₁, c₂) -> c - c₁ - c₂, conversion, conversion_ACE, conversion_APE)),
        # heatings at the intermediate state, Q̃_ϑ = -J_ϑ·∇Σ̃_ϑ, Q̃_q = -J_q·∇Σ̃_q, Q̃ = Σ̃_ϑ σ_ϑ
        heating_consvar_at_p_star = entry("heating Q̃_ϑ = -J_ϑ·∇Σ̃_ϑ at the intermediate state", "W m⁻³", cells,
            (model, consflux_x, consflux_z, conjugate_variable_at_p_star) ->
                gradient_product!(out(:heating_consvar_at_p_star), model, faces, (consflux_x, consflux_z), conjugate_variable_at_p_star)),
        heating_composition_at_p_star = entry("heating Q̃_q = -J_q·∇Σ̃_q at the intermediate state", "W m⁻³", cells,
            (model, qflux_x, qflux_z, modified_chemical_potential_at_p_star) ->
                gradient_product!(out(:heating_composition_at_p_star), model, faces, (qflux_x, qflux_z), modified_chemical_potential_at_p_star)),
        heating_at_p_star = entry("heating Q̃ = Σ̃_ϑ σ_ϑ at the intermediate state", "W m⁻³", cells,
            (conjugate_variable_at_p_star, conservative_variable_production) ->
                pointwise!(out(:heating_at_p_star), mgr, *, conjugate_variable_at_p_star, conservative_variable_production)),
        # heatings at the reference state, Q*_ϑ = -J_ϑ·∇Σ*_ϑ, Q*_q = -J_q·∇Σ*_q, Q* = Σ*_ϑ σ_ϑ
        heating_consvar_at_z_star = entry("heating Q*_ϑ = -J_ϑ·∇Σ*_ϑ at the reference state", "W m⁻³", cells,
            (model, consflux_x, consflux_z, conjugate_variable_at_z_star) ->
                gradient_product!(out(:heating_consvar_at_z_star), model, faces, (consflux_x, consflux_z), conjugate_variable_at_z_star)),
        heating_composition_at_z_star = entry("heating Q*_q = -J_q·∇Σ*_q at the reference state", "W m⁻³", cells,
            (model, qflux_x, qflux_z, modified_chemical_potential_at_z_star) ->
                gradient_product!(out(:heating_composition_at_z_star), model, faces, (qflux_x, qflux_z), modified_chemical_potential_at_z_star)),
        heating_at_z_star = entry("heating Q* = Σ*_ϑ σ_ϑ at the reference state", "W m⁻³", cells,
            (conjugate_variable_at_z_star, conservative_variable_production) ->
                pointwise!(out(:heating_at_z_star), mgr, *, conjugate_variable_at_z_star, conservative_variable_production)),
        # the forms of the energy notes: ∇·J_C*, J_C* = (p - P_*) u
        div_pressure_flux_star = entry("divergence of J_C* = (p - P_*) u", "W m⁻³", cells,
            (model, u, w, pressure_anomaly_star) -> pressure_flux_divergence!(out(:div_pressure_flux_star), model, faces, u, w, pressure_anomaly_star)),
        # b₁ = g ρ_* (v - ṽ), b₂ = g ρ_* (ṽ - v_*) (FC2D); b₁ = 0, b₂ = g ρ_R (v(P_R) - v_*) (AN2D)
        buoyancy_ACE = entry("buoyancy b₁ = g ρ_* (v - v(P_*)) of the compression from P_* to p (zero for AN2D)", "m s⁻²", cells,
            (model, reference_state) -> compression_buoyancy!(out(:buoyancy_ACE), model, reference_state)),
        buoyancy_APE = entry("buoyancy b₂ = g ρ_* (v(P_*) - v_*) relative to the reference state", "m s⁻²", cells,
            (model, reference_state) -> displacement_buoyancy!(out(:buoyancy_APE), model, reference_state)),
        # C_b,1* = ρ b₁ w, C_b,2* = ρ b₂ w
        buoyancy_conversion_ACE = entry("buoyancy conversion C_b,1* = ρ b₁ w", "W m⁻³", cells,
            (model, mass_flux_z, buoyancy_ACE) -> buoyancy_conversion!(out(:buoyancy_conversion_ACE), model, faces, mass_flux_z, buoyancy_ACE)),
        buoyancy_conversion_APE = entry("buoyancy conversion C_b,2* = ρ b₂ w", "W m⁻³", cells,
            (model, mass_flux_z, buoyancy_APE) -> buoyancy_conversion!(out(:buoyancy_conversion_APE), model, faces, mass_flux_z, buoyancy_APE)),
        # C_D* = (p - P_*) ∇·u, zero for AN2D (part of C_b,2*)
        pressure_dilatation_star = entry("pressure dilatation C_D* = (p - P_*) ∇·u (zero for AN2D)", "W m⁻³", cells,
            (model, pressure_anomaly_star, divergence) -> pressure_dilatation!(out(:pressure_dilatation_star), model, pressure_anomaly_star, divergence)),
        # C_ACE + ∇·J_C* - C_b,1* - C_D*, C_APE - C_b,2*: zero in the continuous equations
        ACE_conversion_residual = entry("residual of C_ACE = C_b,1* + C_D* - ∇·J_C*", "W m⁻³", cells,
            (model, conversion_ACE, div_pressure_flux_star, buoyancy_conversion_ACE, pressure_dilatation_star) ->
                conversion_residual!(out(:ACE_conversion_residual), model, conversion_ACE, div_pressure_flux_star, buoyancy_conversion_ACE, pressure_dilatation_star)),
        APE_conversion_residual = entry("residual of C_APE = C_b,2*", "W m⁻³", cells,
            (conversion_APE, buoyancy_conversion_APE) -> pointwise!(out(:APE_conversion_residual), mgr, -, conversion_APE, buoyancy_conversion_APE)),
    )
end

"""
    entries = reference_tendency_entries(model, out)

The diagnostics of the tendency of the Lorenz reference state from the diabatic tendencies (`reference_tendency!`, in
reference_tendency.jl) and of the terms it adds to the budgets of the available compressible energy (ACE) Π₁ and the
available potential energy (APE) Π₂, as a named tuple of `entry(long_name, units, location, recipe)`, the same for AN2D
and FC2D (with ∂ₜP_top for FC2D):
- the diabatic rates ρ Dϑ/Dt = σ_ϑ - ∇·J_ϑ, ρ Dq/Dt = -∇·J_q, from which the recipe `reference_state_tendency`
  (`add_reference_state!`) computes the tendency of the reference state once per diagnostics session;
- ∂ₜP_* on the z axis and at the reference height z_* of each parcel [Pa s⁻¹], ∂ₜP_top (FC2D), and the number of slots
  whose means have plain weights;
- the terms in ∂ₜP_*, R_Π₁ = ρ (Σ_p(z, p) - Σ_p(z, P_*(z))) ∂ₜP_*(z) and
  R_Π₂ = ρ Σ_p(z, P_*(z)) ∂ₜP_*(z) - ρ Σ_p(z_*, P_*(z_*)) ∂ₜP_*(z_*) [W m⁻³];
- the tendencies ∂ₜ(ρΠ₁) = B₁ ∂ₜρ + Δ₁Σ_ϑ ∂ₜ(ρϑ) + Δ₁Σ_q ∂ₜ(ρq) + R_Π₁ and ∂ₜ(ρΠ₂) = B₂ ∂ₜρ + Δ₂Σ_ϑ ∂ₜ(ρϑ) + Δ₂Σ_q ∂ₜ(ρq) + R_Π₂
  [W m⁻³], and the residuals of their budgets,
      ∂ₜ(ρΠ₁) + ∇·(J_ACE^rev + J_ACE^irr) + C_ACE + (Q_ϑ - Q̃_ϑ) + (Q_q - Q̃_q) - (Q - Q̃) - R_Π₁,
      ∂ₜ(ρΠ₂) + ∇·(J_APE^rev + J_APE^irr) + C_APE + (Q̃_ϑ - Q*_ϑ) + (Q̃_q - Q*_q) - (Q̃ - Q*) - R_Π₂,
  zero at each cell to round-off.
The recipes write into the arrays `out(name)`; with `out = nothing`, only the information of the entries is used.
"""
reference_tendency_entries(model, out) = let (; mgr) = model, cells = (Cell(), Cell()), rows = (nothing, Cell())
    entries = (;
        # ρ Dϑ/Dt = σ_ϑ - ∇·J_ϑ, ρ Dq/Dt = -∇·J_q
        consvar_diabatic_rate = entry("diabatic rate ρ Dϑ/Dt = σ_ϑ - ∇·J_ϑ", "kg m⁻³ s⁻¹ × consvar", cells,
            (conservative_variable_production, div_consflux) -> pointwise!(out(:consvar_diabatic_rate), mgr, -, conservative_variable_production, div_consflux)),
        composition_diabatic_rate = entry("diabatic rate ρ Dq/Dt = -∇·J_q", "kg m⁻³ s⁻¹", cells,
            div_qflux -> pointwise!(out(:composition_diabatic_rate), mgr, -, div_qflux)),
        p_star_tendency = entry("tendency of the reference pressure ∂ₜP_* at fixed height", "Pa s⁻¹", rows,
            reference_state_tendency -> reference_state_tendency.p_star_tendency),
        p_star_tendency_at_z_star = entry("tendency of the reference pressure ∂ₜP_*(z_*) at the reference height", "Pa s⁻¹", cells,
            (model, reference_state_tendency, z_star) -> reference_pressure_tendency_at!(out(:p_star_tendency_at_z_star), model, reference_state_tendency, z_star)),
        plain_weight_slots = entry("number of slots whose means have plain weights (N² ≤ 0 for one of their parcels)", "1", (),
            reference_state_tendency -> eltype(reference_state_tendency.mass)(reference_state_tendency.plain_weight_slots)),
        # R_Π₁ = ρ (Σ_p(z, p) - Σ_p(z, P_*(z))) ∂ₜP_*(z), R_Π₂ = ρ Σ_p(z, P_*(z)) ∂ₜP_*(z) - ρ Σ_p(z_*, P_*(z_*)) ∂ₜP_*(z_*)
        reference_pressure_term_ACE = entry("term R_Π₁ of ∂ₜP_* in the tendency of ρΠ₁ (zero for AN2D)", "W m⁻³", cells,
            (model, reference_state_tendency) -> compression_reference_term!(out(:reference_pressure_term_ACE), model, reference_state_tendency)),
        reference_pressure_term_APE = entry("term R_Π₂ of ∂ₜP_* in the tendency of ρΠ₂", "W m⁻³", cells,
            (model, reference_state_tendency, z_star, p_star_tendency_at_z_star) ->
                displacement_reference_term!(out(:reference_pressure_term_APE), model, reference_state_tendency, z_star, p_star_tendency_at_z_star)),
        # ∂ₜ(ρΠ) = B ∂ₜρ + ΔΣ_ϑ ∂ₜ(ρϑ) + ΔΣ_q ∂ₜ(ρq) + R
        dρACE = entry("tendency of ρ Π₁", "W m⁻³", cells,
            (model, dstate, bernoulli_potential_ACE, conjugate_variable_ACE, modified_chemical_potential_ACE, dρconsvar, dρq, reference_pressure_term_ACE) ->
                available_energy_tendency!(out(:dρACE), model, dstate, bernoulli_potential_ACE, conjugate_variable_ACE, modified_chemical_potential_ACE,
                                           dρconsvar, dρq, reference_pressure_term_ACE)),
        dρAPE = entry("tendency of ρ Π₂", "W m⁻³", cells,
            (model, dstate, bernoulli_potential_APE, conjugate_variable_APE, modified_chemical_potential_APE, dρconsvar, dρq, reference_pressure_term_APE) ->
                available_energy_tendency!(out(:dρAPE), model, dstate, bernoulli_potential_APE, conjugate_variable_APE, modified_chemical_potential_APE,
                                           dρconsvar, dρq, reference_pressure_term_APE)),
        # the residuals of the budgets of ρΠ₁ and ρΠ₂
        ACE_budget_residual = entry("residual of the budget of ρ Π₁", "W m⁻³", cells,
            (model, dρACE, div_ACE_flux_rev, div_ACE_flux_irr, conversion_ACE, heating_consvar, heating_composition, heating,
             heating_consvar_at_p_star, heating_composition_at_p_star, heating_at_p_star, reference_pressure_term_ACE) ->
                available_energy_budget_residual!(out(:ACE_budget_residual), model, dρACE, div_ACE_flux_rev, div_ACE_flux_irr, conversion_ACE,
                                                  (heating_consvar, heating_composition, heating),
                                                  (heating_consvar_at_p_star, heating_composition_at_p_star, heating_at_p_star), reference_pressure_term_ACE)),
        APE_budget_residual = entry("residual of the budget of ρ Π₂", "W m⁻³", cells,
            (model, dρAPE, div_APE_flux_rev, div_APE_flux_irr, conversion_APE, heating_consvar_at_p_star, heating_composition_at_p_star,
             heating_at_p_star, heating_consvar_at_z_star, heating_composition_at_z_star, heating_at_z_star, reference_pressure_term_APE) ->
                available_energy_budget_residual!(out(:APE_budget_residual), model, dρAPE, div_APE_flux_rev, div_APE_flux_irr, conversion_APE,
                                                  (heating_consvar_at_p_star, heating_composition_at_p_star, heating_at_p_star),
                                                  (heating_consvar_at_z_star, heating_composition_at_z_star, heating_at_z_star), reference_pressure_term_APE)),
    )
    model isa FC2D || return entries
    return merge(entries, (;
        p_top_tendency = entry("tendency of the pressure P_top at the top of the reference column", "Pa s⁻¹", (),
            reference_state_tendency -> reference_state_tendency.p_top_tendency),
    ))
end

"""
    count = count_wall_minima(model, z_star)

The number of parcels whose reference height `z_star` [m] is at the bottom (z_* = 0) or top (z_* = L_z) wall.
"""
count_wall_minima(model, z_star) = count(z -> z == 0 || z == model.space.Lz, interior(model.domain, z_star))

"""
    field = interior_diagnostic(session, name)
    x, z = diagnostic_coordinates(model, name)

The diagnostic `name` of an open diagnostics session without its halo (a view), for plotting, and the coordinates
[m] of its points, from the grid location of the diagnostic (e.g. both wall faces for `w` in a bounded box).
"""
interior_diagnostic(session, name) = interior(session.model.domain, CookBooks.get(session, name))
diagnostic_coordinates(model, name) =
    coordinates(model.domain, (model.dx, model.dz), diagnostic_info(model)[name].location)
