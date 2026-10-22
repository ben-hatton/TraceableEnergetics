## ENERGETICS DIAGNOSTICS OF EITHER MODEL
#
# The energy budget of energy.tex as diagnostics of AN2D and FC2D, added to the diagnostics of the model packages
# (`energetics_diagnostics`, merged into `diagnostics` of diagnostics.jl). K and P are specific energies [J kg⁻¹], the
# energy densities ρK, ρP [J m⁻³] and every term of the budget are per unit volume [W m⁻³], with ρ = ρ_R for AN2D.
# Fields on the cells carry no suffix; their components on the x and z faces end in `_x` and `_z`.
#
# Every term is built from the model's own operators (`ℑ`, `δ` of the spatial scheme): a face product -mass_flux ·
# ∇(potential) is averaged to the cells, as the model does for ρε. With the product rule of these operators,
#     δ(ℑ(a) b) = a δb + ℑ(b δa),
# the budget of ρP closes at each cell, and that of ρK in the volume integral (`energy_budget`).

"""
    book = energetics_diagnostics(model; reference = default_reference(model))

`CookBook` of the energetics diagnostics of an AN2D or FC2D `model`, to be merged into its diagnostics (see
`diagnostics`); the names, long names, units and grid locations are those of `energetics_info`. The geopotential is
g (z - z0), with z0 the origin `model.space.z0` [m] of the tank. The reference state (`AnelasticReference` with the density ρ_R(z) and the
hydrostatic pressure p_R(z)) is that of the model for AN2D; FC2D has none, so the diagnostics of the manipulated
conversion (`pressure_flux`, `buoyancy_conversion`, `pressure_dilatation`, `conversion_residual`) need a `reference`,
usually that of the matching AN2D, on the same grid.

The book owns the output array of each diagnostic, allocated on its first use, which every session overwrites.
"""
function energetics_diagnostics(model::Union{AN2D{F}, FC2D{F}}; reference = default_reference(model)) where F
    (; mgr, domain) = model
    info = energetics_info(model)
    cache = Dict{Symbol, Any}()
    # output array of the diagnostic `name`, on its grid location, allocated on first use
    out(name) = get!(() -> alloc(F, mgr, domain, info[name].location), cache, name)::Matrix{F}
    book = CookBook(;
        # energies: the specific enthalpy h(ϑ, q, p) at the pressure of the energy (p_R for AN2D), P = h - p/ρ + g (z - z0)
        specific_enthalpy            = (model, state, scratch) -> specific_enthalpy!(out(:specific_enthalpy), model, energy_pressure(model, state, scratch), scratch.thermo),
        potential_energy             = (model, specific_enthalpy, state, scratch) -> potential_energy!(out(:potential_energy), model, specific_enthalpy, energy_density(model, state, scratch), energy_pressure(model, state, scratch)),
        bernoulli_potential          = (model, specific_enthalpy, scratch) -> bernoulli_potential!(out(:bernoulli_potential), model, specific_enthalpy, scratch),
        ρK                           = (density, kinetic_energy) -> pointwise!(out(:ρK), mgr, *, density, kinetic_energy),
        ρP                           = (density, potential_energy) -> pointwise!(out(:ρP), mgr, *, density, potential_energy),
        # reversible and irreversible fluxes of K and P, and their divergences
        kinetic_flux_rev_xz          = (model, mass_flux_x, mass_flux_z, kinetic_energy) -> advective_flux!(out(:kinetic_flux_rev_x), out(:kinetic_flux_rev_z), model, mass_flux_x, mass_flux_z, kinetic_energy),
        kinetic_flux_rev_x           = kinetic_flux_rev_xz -> kinetic_flux_rev_xz[1],
        kinetic_flux_rev_z           = kinetic_flux_rev_xz -> kinetic_flux_rev_xz[2],
        kinetic_flux_irr_xz          = (model, scratch) -> stress_work_flux!(out(:kinetic_flux_irr_x), out(:kinetic_flux_irr_z), model, scratch.irreversible),
        kinetic_flux_irr_x           = kinetic_flux_irr_xz -> kinetic_flux_irr_xz[1],
        kinetic_flux_irr_z           = kinetic_flux_irr_xz -> kinetic_flux_irr_xz[2],
        potential_flux_rev_xz        = (model, scratch, mass_flux_x, mass_flux_z, bernoulli_potential, conjugate_variable, modified_chemical_potential) ->
            potential_flux_rev!(out(:potential_flux_rev_x), out(:potential_flux_rev_z), model, scratch.advection, (mass_flux_x, mass_flux_z), bernoulli_potential, conjugate_variable, modified_chemical_potential),
        potential_flux_rev_x         = potential_flux_rev_xz -> potential_flux_rev_xz[1],
        potential_flux_rev_z         = potential_flux_rev_xz -> potential_flux_rev_xz[2],
        potential_flux_irr_xz        = (model, scratch, conjugate_variable, modified_chemical_potential) ->
            potential_flux_irr!(out(:potential_flux_irr_x), out(:potential_flux_irr_z), model, scratch.irreversible, conjugate_variable, modified_chemical_potential),
        potential_flux_irr_x         = potential_flux_irr_xz -> potential_flux_irr_xz[1],
        potential_flux_irr_z         = potential_flux_irr_xz -> potential_flux_irr_xz[2],
        div_kinetic_flux_rev         = (model, kinetic_flux_rev_x, kinetic_flux_rev_z) -> divergence!(out(:div_kinetic_flux_rev), model, kinetic_flux_rev_x, kinetic_flux_rev_z),
        div_kinetic_flux_irr         = (model, kinetic_flux_irr_x, kinetic_flux_irr_z) -> divergence!(out(:div_kinetic_flux_irr), model, kinetic_flux_irr_x, kinetic_flux_irr_z),
        div_potential_flux_rev       = (model, potential_flux_rev_x, potential_flux_rev_z) -> divergence!(out(:div_potential_flux_rev), model, potential_flux_rev_x, potential_flux_rev_z),
        div_potential_flux_irr       = (model, potential_flux_irr_x, potential_flux_irr_z) -> divergence!(out(:div_potential_flux_irr), model, potential_flux_irr_x, potential_flux_irr_z),
        # reversible conversions C_ϑ, C_q, C_Σ = -(mass flux of the tracer) · ∇(potential), on the faces and at the cells
        conversion_consvar_xz        = (model, scratch, conjugate_variable) -> gradient_product!(out(:conversion_consvar_x), out(:conversion_consvar_z), model, tracer_flux(model, scratch.advection, 1), conjugate_variable),
        conversion_consvar_x         = conversion_consvar_xz -> conversion_consvar_xz[1],
        conversion_consvar_z         = conversion_consvar_xz -> conversion_consvar_xz[2],
        conversion_composition_xz    = (model, scratch, modified_chemical_potential) -> gradient_product!(out(:conversion_composition_x), out(:conversion_composition_z), model, tracer_flux(model, scratch.advection, 2), modified_chemical_potential),
        conversion_composition_x     = conversion_composition_xz -> conversion_composition_xz[1],
        conversion_composition_z     = conversion_composition_xz -> conversion_composition_xz[2],
        conversion_bernoulli_xz      = (model, mass_flux_x, mass_flux_z, bernoulli_potential) -> gradient_product!(out(:conversion_bernoulli_x), out(:conversion_bernoulli_z), model, (mass_flux_x, mass_flux_z), bernoulli_potential),
        conversion_bernoulli_x       = conversion_bernoulli_xz -> conversion_bernoulli_xz[1],
        conversion_bernoulli_z       = conversion_bernoulli_xz -> conversion_bernoulli_xz[2],
        conversion_consvar           = (model, conversion_consvar_x, conversion_consvar_z) -> faces_to_cells!(out(:conversion_consvar), model, conversion_consvar_x, conversion_consvar_z),
        conversion_composition       = (model, conversion_composition_x, conversion_composition_z) -> faces_to_cells!(out(:conversion_composition), model, conversion_composition_x, conversion_composition_z),
        conversion_bernoulli         = (model, conversion_bernoulli_x, conversion_bernoulli_z) -> faces_to_cells!(out(:conversion_bernoulli), model, conversion_bernoulli_x, conversion_bernoulli_z),
        conversion                   = (conversion_consvar, conversion_composition, conversion_bernoulli) -> pointwise!(out(:conversion), mgr, +, conversion_consvar, conversion_composition, conversion_bernoulli),
        # heating: Q_ϑ, Q_q = -(diffusive flux) · ∇(potential), Q = conjvar σ_cons, and the closure Q = Q_ϑ + Q_q + ρε
        heating_consvar_xz           = (model, scratch, conjugate_variable) -> gradient_product!(out(:heating_consvar_x), out(:heating_consvar_z), model, (scratch.irreversible.consflux_x, scratch.irreversible.consflux_z), conjugate_variable),
        heating_consvar_x            = heating_consvar_xz -> heating_consvar_xz[1],
        heating_consvar_z            = heating_consvar_xz -> heating_consvar_xz[2],
        heating_composition_xz       = (model, scratch, modified_chemical_potential) -> gradient_product!(out(:heating_composition_x), out(:heating_composition_z), model, (scratch.irreversible.qflux_x, scratch.irreversible.qflux_z), modified_chemical_potential),
        heating_composition_x        = heating_composition_xz -> heating_composition_xz[1],
        heating_composition_z        = heating_composition_xz -> heating_composition_xz[2],
        heating_consvar              = (model, heating_consvar_x, heating_consvar_z) -> faces_to_cells!(out(:heating_consvar), model, heating_consvar_x, heating_consvar_z),
        heating_composition          = (model, heating_composition_x, heating_composition_z) -> faces_to_cells!(out(:heating_composition), model, heating_composition_x, heating_composition_z),
        heating                      = (conjugate_variable, conservative_variable_production) -> pointwise!(out(:heating), mgr, *, conjugate_variable, conservative_variable_production),
        heating_residual             = (model, heating, heating_consvar, heating_composition, viscous_dissipation) -> heating_residual!(out(:heating_residual), model, heating, heating_consvar, heating_composition, viscous_dissipation),
        # vortex force work VF = -ρ u·(ω×u)
        vortex_work_xz               = (model, scratch, mass_flux_x, mass_flux_z) -> vortex_work!(out(:vortex_work_x), out(:vortex_work_z), model, scratch.advection, mass_flux_x, mass_flux_z),
        vortex_work_x                = vortex_work_xz -> vortex_work_xz[1],
        vortex_work_z                = vortex_work_xz -> vortex_work_xz[2],
        vortex_work                  = (model, vortex_work_x, vortex_work_z) -> faces_to_cells!(out(:vortex_work), model, vortex_work_x, vortex_work_z),
        # budgets: tendency of ρK, ρP, and the part of it not given by the terms of energy.tex
        kinetic_budget_residual      = (model, dρK, div_kinetic_flux_rev, div_kinetic_flux_irr, conversion, viscous_dissipation, vortex_work) ->
            kinetic_budget_residual!(out(:kinetic_budget_residual), model, dρK, div_kinetic_flux_rev, div_kinetic_flux_irr, conversion, viscous_dissipation, vortex_work),
        potential_budget_residual    = (model, dρP, div_potential_flux_rev, div_potential_flux_irr, conversion, heating_consvar, heating_composition, heating) ->
            potential_budget_residual!(out(:potential_budget_residual), model, dρP, div_potential_flux_rev, div_potential_flux_irr, conversion, heating_consvar, heating_composition, heating),
    )
    merge!(book.recipes, pairs(model_recipes(model, out)), pairs(manipulated_recipes(model, out, reference)))
    return book
end

default_reference((; reference)::AN2D) = reference
default_reference(::FC2D) = nothing

#=
Model-specific recipes. AN2D has the reference density ρ_R(z) as mass weight (no ∂ₜρ), and its energy of P is evaluated
at the reference pressure p_R(z); FC2D has the density and pressure of the state.
=#
model_recipes(model::AN2D, out) = let (; mgr) = model
    (;
        kinetic_energy_rate_xz = (model, u, w, du, dw) -> velocity_work!(out(:kinetic_energy_rate_x), out(:kinetic_energy_rate_z), model, (u, w), (du, dw)),
        kinetic_energy_rate_x  = kinetic_energy_rate_xz -> kinetic_energy_rate_xz[1],
        kinetic_energy_rate_z  = kinetic_energy_rate_xz -> kinetic_energy_rate_xz[2],
        kinetic_energy_rate = (model, kinetic_energy_rate_x, kinetic_energy_rate_z) -> faces_to_cells!(out(:kinetic_energy_rate), model, kinetic_energy_rate_x, kinetic_energy_rate_z),
        dρK                 = (density, kinetic_energy_rate) -> pointwise!(out(:dρK), mgr, *, density, kinetic_energy_rate),
        dρP                 = (model, conjugate_variable, modified_chemical_potential, dρconsvar, dρq) -> potential_energy_tendency!(out(:dρP), model, conjugate_variable, modified_chemical_potential, dρconsvar, dρq),
    )
end
model_recipes(model::FC2D, out) = let (; mgr) = model
    (;
        kinetic_energy_rate_xz = (model, u, w, du, dw) -> velocity_work!(out(:kinetic_energy_rate_x), out(:kinetic_energy_rate_z), model, (u, w), (du, dw)),
        kinetic_energy_rate_x  = kinetic_energy_rate_xz -> kinetic_energy_rate_xz[1],
        kinetic_energy_rate_z  = kinetic_energy_rate_xz -> kinetic_energy_rate_xz[2],
        kinetic_energy_rate = (model, kinetic_energy_rate_x, kinetic_energy_rate_z) -> faces_to_cells!(out(:kinetic_energy_rate), model, kinetic_energy_rate_x, kinetic_energy_rate_z),
        dρK                 = (model, density, kinetic_energy, kinetic_energy_rate, dρ) -> kinetic_energy_tendency!(out(:dρK), model, density, kinetic_energy, kinetic_energy_rate, dρ),
        dρP                 = (model, bernoulli_potential, conjugate_variable, modified_chemical_potential, dρ, dρconsvar, dρq) -> potential_energy_tendency!(out(:dρP), model, bernoulli_potential, conjugate_variable, modified_chemical_potential, dρ, dρconsvar, dρq),
    )
end

# Manipulated conversion C = -∇·J_C + C_b + C_D, with J_C = (p - p_R) u, needs the reference state
manipulated_recipes(model, out, ::Nothing) = (;)
manipulated_recipes(model, out, reference) = let (; mgr) = model
    (;
        pressure_anomaly      = (model, pressure) -> pressure_anomaly!(out(:pressure_anomaly), model, reference, pressure),
        pressure_flux_xz      = (model, u, w, pressure_anomaly) -> advective_flux!(out(:pressure_flux_x), out(:pressure_flux_z), model, u, w, pressure_anomaly),
        pressure_flux_x       = pressure_flux_xz -> pressure_flux_xz[1],
        pressure_flux_z       = pressure_flux_xz -> pressure_flux_xz[2],
        div_pressure_flux     = (model, pressure_flux_x, pressure_flux_z) -> divergence!(out(:div_pressure_flux), model, pressure_flux_x, pressure_flux_z),
        reference_buoyancy    = (model, state, scratch) -> reference_buoyancy!(out(:reference_buoyancy), model, reference, state, scratch),
        buoyancy_conversion_zc = (model, mass_flux_z, reference_buoyancy) -> buoyancy_conversion!(out(:buoyancy_conversion_z), out(:buoyancy_conversion), model, mass_flux_z, reference_buoyancy),
        buoyancy_conversion_z = buoyancy_conversion_zc -> buoyancy_conversion_zc[1],
        buoyancy_conversion   = buoyancy_conversion_zc -> buoyancy_conversion_zc[2],
        pressure_dilatation   = (model, pressure_anomaly, divergence) -> pressure_dilatation!(out(:pressure_dilatation), model, pressure_anomaly, divergence),
        conversion_residual   = (model, conversion, div_pressure_flux, buoyancy_conversion, pressure_dilatation) -> conversion_residual!(out(:conversion_residual), model, conversion, div_pressure_flux, buoyancy_conversion, pressure_dilatation),
    )
end

# density and pressure of the energy P = h(ϑ, q, p) - p/ρ + g (z - z0): the reference profiles (AN2D) or the state (FC2D)
energy_density((; reference)::AN2D, state, scratch) = reference.ρ_R
energy_pressure((; reference)::AN2D, state, scratch) = reference.p_R
energy_density(::FC2D, (; m), scratch) = view(m, :, :, 1)
energy_pressure(::FC2D, state, scratch) = scratch.thermo.p

# index of the consvar flux (then q) among the advected densities of the state: (ρ_R consvar, ρ_R q) or (ρ, ρ consvar, ρ q)
consvar_index(::AN2D) = 1
consvar_index(::FC2D) = 2
"""
    mu, mw = tracer_flux(model, advection, k)

Advective flux ρ_R ϑ u, ρ_R ϑ w (k = 1) or ρ_R q u, ρ_R q w (k = 2), or with ρ for FC2D, on the x and z faces, from the
`advection` scratch of the tendencies.
"""
tracer_flux(model, (; mu, mw), k) = (view(mu, :, :, consvar_index(model) + k - 1), view(mw, :, :, consvar_index(model) + k - 1))

"""
    info = energetics_info(model)

Long name, units and grid location of each diagnostic of `energetics_diagnostics`, as a named tuple of
`(; long_name, units, location)`, in the style of the model packages.
"""
energetics_info(::Union{AN2D, FC2D}) = let cells = (Cell(), Cell()), x_faces = (Face(), Cell()), z_faces = (Cell(), Face())
    (;
        specific_enthalpy         = (long_name = "specific enthalpy h (at p_R for AN2D)",                         units = "J kg⁻¹",   location = cells),
        potential_energy          = (long_name = "potential energy P = h - p/ρ + g (z - z0)",                    units = "J kg⁻¹",   location = cells),
        bernoulli_potential       = (long_name = "Σ - ϑ Σ_ϑ - q Σ_q, B + g (z - z0) (+ φ for AN2D)",             units = "J kg⁻¹",   location = cells),
        ρK                        = (long_name = "kinetic energy density ρ K",                                    units = "J m⁻³",    location = cells),
        ρP                        = (long_name = "potential energy density ρ P",                                  units = "J m⁻³",    location = cells),
        dρK                       = (long_name = "tendency of ρ K",                                               units = "W m⁻³",    location = cells),
        dρP                       = (long_name = "tendency of ρ P",                                               units = "W m⁻³",    location = cells),
        kinetic_energy_rate_x     = (long_name = "u ∂ₜu on the x-faces",                                          units = "W kg⁻¹",   location = x_faces),
        kinetic_energy_rate_z     = (long_name = "w ∂ₜw on the z-faces",                                          units = "W kg⁻¹",   location = z_faces),
        kinetic_energy_rate       = (long_name = "tendency of K",                                                 units = "W kg⁻¹",   location = cells),
        kinetic_flux_rev_x        = (long_name = "reversible flux of K along x, ρ K u",                           units = "W m⁻²",    location = x_faces),
        kinetic_flux_rev_z        = (long_name = "reversible flux of K along z, ρ K w",                           units = "W m⁻²",    location = z_faces),
        kinetic_flux_irr_x        = (long_name = "irreversible flux of K along x, -(u·τ)_x",                      units = "W m⁻²",    location = x_faces),
        kinetic_flux_irr_z        = (long_name = "irreversible flux of K along z, -(u·τ)_z",                      units = "W m⁻²",    location = z_faces),
        potential_flux_rev_x      = (long_name = "reversible flux of P along x, ρ Σ u",                           units = "W m⁻²",    location = x_faces),
        potential_flux_rev_z      = (long_name = "reversible flux of P along z, ρ Σ w",                           units = "W m⁻²",    location = z_faces),
        potential_flux_irr_x      = (long_name = "irreversible flux of P along x, Σ_ϑ J_ϑ + Σ_q J_q",            units = "W m⁻²",    location = x_faces),
        potential_flux_irr_z      = (long_name = "irreversible flux of P along z, Σ_ϑ J_ϑ + Σ_q J_q",            units = "W m⁻²",    location = z_faces),
        div_kinetic_flux_rev      = (long_name = "divergence of the reversible flux of K",                        units = "W m⁻³",    location = cells),
        div_kinetic_flux_irr      = (long_name = "divergence of the irreversible flux of K",                      units = "W m⁻³",    location = cells),
        div_potential_flux_rev    = (long_name = "divergence of the reversible flux of P",                        units = "W m⁻³",    location = cells),
        div_potential_flux_irr    = (long_name = "divergence of the irreversible flux of P",                      units = "W m⁻³",    location = cells),
        conversion_consvar_x      = (long_name = "conversion C_ϑ along x, -ρ ϑ u ∂ₓΣ_ϑ",                         units = "W m⁻³",    location = x_faces),
        conversion_consvar_z      = (long_name = "conversion C_ϑ along z, -ρ ϑ w ∂_zΣ_ϑ",                        units = "W m⁻³",    location = z_faces),
        conversion_composition_x  = (long_name = "conversion C_q along x, -ρ q u ∂ₓΣ_q",                         units = "W m⁻³",    location = x_faces),
        conversion_composition_z  = (long_name = "conversion C_q along z, -ρ q w ∂_zΣ_q",                        units = "W m⁻³",    location = z_faces),
        conversion_bernoulli_x    = (long_name = "conversion C_Σ along x, -ρ u ∂ₓ(Σ - ϑΣ_ϑ - qΣ_q)",             units = "W m⁻³",    location = x_faces),
        conversion_bernoulli_z    = (long_name = "conversion C_Σ along z, -ρ w ∂_z(Σ - ϑΣ_ϑ - qΣ_q)",            units = "W m⁻³",    location = z_faces),
        conversion_consvar        = (long_name = "conversion C_ϑ from P to K",                                    units = "W m⁻³",    location = cells),
        conversion_composition    = (long_name = "conversion C_q from P to K",                                    units = "W m⁻³",    location = cells),
        conversion_bernoulli      = (long_name = "conversion C_Σ from P to K",                                    units = "W m⁻³",    location = cells),
        conversion                = (long_name = "conversion C = C_ϑ + C_q + C_Σ from P to K",                    units = "W m⁻³",    location = cells),
        heating_consvar_x         = (long_name = "heating Q_ϑ along x, -J_ϑ ∂ₓΣ_ϑ",                              units = "W m⁻³",    location = x_faces),
        heating_consvar_z         = (long_name = "heating Q_ϑ along z, -J_ϑ ∂_zΣ_ϑ",                             units = "W m⁻³",    location = z_faces),
        heating_composition_x     = (long_name = "heating Q_q along x, -J_q ∂ₓΣ_q",                              units = "W m⁻³",    location = x_faces),
        heating_composition_z     = (long_name = "heating Q_q along z, -J_q ∂_zΣ_q",                             units = "W m⁻³",    location = z_faces),
        heating_consvar           = (long_name = "heating Q_ϑ = -J_ϑ·∇Σ_ϑ",                                      units = "W m⁻³",    location = cells),
        heating_composition       = (long_name = "heating Q_q = -J_q·∇Σ_q",                                      units = "W m⁻³",    location = cells),
        heating                   = (long_name = "heating Q = Σ_ϑ σ_ϑ",                                          units = "W m⁻³",    location = cells),
        heating_residual          = (long_name = "closure residual Q - Q_ϑ - Q_q - ρ ε_K",                      units = "W m⁻³",    location = cells),
        vortex_work_x             = (long_name = "vortex force work along x, -ρ u (ω×u)_x",                       units = "W m⁻³",    location = x_faces),
        vortex_work_z             = (long_name = "vortex force work along z, -ρ w (ω×u)_z",                       units = "W m⁻³",    location = z_faces),
        vortex_work               = (long_name = "vortex force work VF = -ρ u·(ω×u)",                             units = "W m⁻³",    location = cells),
        kinetic_budget_residual   = (long_name = "residual of the budget of ρ K",                                 units = "W m⁻³",    location = cells),
        potential_budget_residual = (long_name = "residual of the budget of ρ P",                                 units = "W m⁻³",    location = cells),
        pressure_anomaly          = (long_name = "pressure anomaly p - p_R",                                      units = "Pa",       location = cells),
        pressure_flux_x           = (long_name = "flux J_C along x, (p - p_R) u",                                 units = "W m⁻²",    location = x_faces),
        pressure_flux_z           = (long_name = "flux J_C along z, (p - p_R) w",                                 units = "W m⁻²",    location = z_faces),
        div_pressure_flux         = (long_name = "divergence of J_C",                                             units = "W m⁻³",    location = cells),
        reference_buoyancy        = (long_name = "buoyancy g (ρ_R v - 1) of the reference state",                 units = "m s⁻²",    location = cells),
        buoyancy_conversion_z     = (long_name = "buoyancy conversion C_b, ρ b w",                                units = "W m⁻³",    location = z_faces),
        buoyancy_conversion       = (long_name = "buoyancy conversion C_b = ρ b w",                               units = "W m⁻³",    location = cells),
        pressure_dilatation       = (long_name = "pressure dilatation C_D = (p - p_R) ∇·u (zero for AN2D)",       units = "W m⁻³",    location = cells),
        conversion_residual       = (long_name = "residual of C = -∇·J_C + C_b + C_D",                            units = "W m⁻³",    location = cells),
    )
end

#=
Kernels. Cell fields are computed at the interior cells and face fields at the interior faces (both walls of a bounded
dimension), then their halo is filled: zero-gradient for cells, periodic for faces (the wall faces are values, not halo). The mass flux, the tracer fluxes and the vortex force are
those of the tendencies, so every term has the discretization of the model.
=#

"""
    out = pointwise!(out, mgr, f, a, b[, c])

out = f(a, b[, c]) pointwise on arrays of the same grid, halo included, by the manager `mgr`.
"""
function pointwise!(out, mgr, f::Fun, a, b) where Fun
    @with mgr, let (irange, jrange) = (axes(out, 1), axes(out, 2))
        @inbounds @vec for i in irange, j in jrange
            out[i, j] = f(a[i, j], b[i, j])
        end
    end
    return out
end
function pointwise!(out, mgr, f::Fun, a, b, c) where Fun
    @with mgr, let (irange, jrange) = (axes(out, 1), axes(out, 2))
        @inbounds @vec for i in irange, j in jrange
            out[i, j] = f(a[i, j], b[i, j], c[i, j])
        end
    end
    return out
end

"""
    h = specific_enthalpy!(h, model, p, (; consvar, q))

Specific enthalpy h(p, consvar, q) [J kg⁻¹] at the interior cells, at the pressure `p`: an array, or the profile p_R(z).
"""
function specific_enthalpy!(h, model, p, (; consvar, q))
    (; mgr, domain, fluid, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            # ℑ with no direction reads an array at (i, j), a profile at j
            h[i, j] = fluid(:p, :consvar, :q).specific_enthalpy(ℑ(p, (), (), i, j), consvar[i, j], q[i, j])
        end
    end
    return fill_halo!(domain, h, ZeroGradient())
end

"""
    P = potential_energy!(P, model, h, ρ, p)

Potential energy P = h - p/ρ + g (z - z0) [J kg⁻¹] at the interior cells (see `energy_density`, `energy_pressure`):
h - p v + g z, with v = 1/ρ, for FC2D; h(p_R) - p_R/ρ_R + g z for AN2D.
"""
function potential_energy!(P, model, h, ρ, p)
    (; mgr, domain, space, dz, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    (; g, z0) = space
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            P[i, j] = h[i, j] - ℑ(p, (), (), i, j) / ℑ(ρ, (), (), i, j) + g * (cellpoint(domain, dz, 2, j) - z0)
        end
    end
    return fill_halo!(domain, P, ZeroGradient())
end

"""
    Σ_B = bernoulli_potential!(Σ_B, model, h, scratch)

Σ - ϑ Σ_ϑ - q Σ_q = h - consvar conjvar - q chempot + g (z - z0) [J kg⁻¹] at the interior cells, plus the kinematic
pressure anomaly φ = (p - p_R)/ρ_R for AN2D. With the thermodynamic forcing of the model it is B + g (z - z0) (+ φ).
"""
function bernoulli_potential!(Σ_B, model::FC2D, h, (; thermo))
    (; mgr, domain, space, dz) = model
    (; g, z0) = space
    (; consvar, q, conjvar, chempot) = thermo
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            Σ_B[i, j] = h[i, j] - consvar[i, j] * conjvar[i, j] - q[i, j] * chempot[i, j] + g * (cellpoint(domain, dz, 2, j) - z0)
        end
    end
    return fill_halo!(domain, Σ_B, ZeroGradient())
end
function bernoulli_potential!(Σ_B, model::AN2D, h, (; thermo, φ))
    (; mgr, domain, space, dz) = model
    (; g, z0) = space
    (; consvar, q, conjvar, chempot) = thermo
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            Σ_B[i, j] = h[i, j] - consvar[i, j] * conjvar[i, j] - q[i, j] * chempot[i, j] + φ[i, j] + g * (cellpoint(domain, dz, 2, j) - z0)
        end
    end
    return fill_halo!(domain, Σ_B, ZeroGradient())
end

"""
    Jx, Jz = advective_flux!(Jx, Jz, model, flux_x, flux_z, a)

Flux of the cell quantity `a` carried by a flux on the faces, J = flux ℑ(a), with `a` averaged to the faces. For the
mass flux ρu: the reversible flux of K, ρ K u (`a` = K); with the velocity: J_C = (p - p_R) u (`a` = p - p_R).
"""
function advective_flux!(Jx, Jz, model, flux_x, flux_z, a)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_faces(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            Jx[I, j] = flux_x[I, j] * ℑ(a, (1,), (Face(),), I, j)
        end
    end
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            Jz[i, J] = flux_z[i, J] * ℑ(a, (2,), (Face(),), i, J)
        end
    end
    periodize!(domain, (Jx, Jz))
    return Jx, Jz
end

"""
    Jx, Jz = stress_work_flux!(Jx, Jz, model, (; uτ_x, uτ_z))

Irreversible flux of K on the faces, -u·τ, from the work of the viscous stress of the tendencies.
"""
function stress_work_flux!(Jx, Jz, model, (; uτ_x, uτ_z))
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_faces(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            Jx[I, j] = - uτ_x[I, j]
        end
    end
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            Jz[i, J] = - uτ_z[i, J]
        end
    end
    periodize!(domain, (Jx, Jz))
    return Jx, Jz
end

"""
    Jx, Jz = potential_flux_rev!(Jx, Jz, model, advection, (mass_flux_x, mass_flux_z), Σ_B, conjvar, chempot)

Reversible flux of P on the faces, ρ Σ u = ρu ℑ(Σ_B) + ρϑu ℑ(Σ_ϑ) + ρqu ℑ(Σ_q), from the mass flux and the tracer fluxes
of the tendencies.
"""
function potential_flux_rev!(Jx, Jz, model, advection, (mass_flux_x, mass_flux_z), Σ_B, conjvar, chempot)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    consvar_x, consvar_z = tracer_flux(model, advection, 1)
    q_x, q_z = tracer_flux(model, advection, 2)
    @with mgr, let (irange, jrange) = (interior_faces(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            Jx[I, j] = mass_flux_x[I, j] * ℑ(Σ_B, (1,), (Face(),), I, j) + consvar_x[I, j] * ℑ(conjvar, (1,), (Face(),), I, j) + q_x[I, j] * ℑ(chempot, (1,), (Face(),), I, j)
        end
    end
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            Jz[i, J] = mass_flux_z[i, J] * ℑ(Σ_B, (2,), (Face(),), i, J) + consvar_z[i, J] * ℑ(conjvar, (2,), (Face(),), i, J) + q_z[i, J] * ℑ(chempot, (2,), (Face(),), i, J)
        end
    end
    periodize!(domain, (Jx, Jz))
    return Jx, Jz
end

"""
    Jx, Jz = potential_flux_irr!(Jx, Jz, model, irreversible, conjvar, chempot)

Irreversible flux of P on the faces, Σ_ϑ J_ϑ + Σ_q J_q = ℑ(conjvar) consflux + ℑ(chempot) qflux. On a wall face, ℑ is the
value of the cell (zero-gradient halo), as for the energy the wall delivers in the tendencies.
"""
function potential_flux_irr!(Jx, Jz, model, (; consflux_x, consflux_z, qflux_x, qflux_z), conjvar, chempot)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_faces(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            Jx[I, j] = consflux_x[I, j] * ℑ(conjvar, (1,), (Face(),), I, j) + qflux_x[I, j] * ℑ(chempot, (1,), (Face(),), I, j)
        end
    end
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            Jz[i, J] = consflux_z[i, J] * ℑ(conjvar, (2,), (Face(),), i, J) + qflux_z[i, J] * ℑ(chempot, (2,), (Face(),), i, J)
        end
    end
    periodize!(domain, (Jx, Jz))
    return Jx, Jz
end

"""
    div = divergence!(div, model, Fx, Fz)

Divergence δx Fx / Δx + δz Fz / Δz of a vector on the faces, at the interior cells, with a zero-gradient halo.
"""
function divergence!(div, model, Fx, Fz)
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; δ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            div[i, j] = inv_dx * δ(Fx, (1,), (Cell(),), i, j) + inv_dz * δ(Fz, (2,), (Cell(),), i, j)
        end
    end
    return fill_halo!(domain, div, ZeroGradient())
end

"""
    Fx, Fz = gradient_product!(Fx, Fz, model, (flux_x, flux_z), Σ)

-flux·∇Σ on the faces, F = -flux δΣ / Δ along each direction, for a flux on the faces and a potential Σ at the cells.
It is a conversion C (mass or tracer flux of the tendencies, potential Σ_ϑ, Σ_q or Σ_B) or a heating Q (diffusive flux).
"""
function gradient_product!(Fx, Fz, model, (flux_x, flux_z), Σ)
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; δ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_faces(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            Fx[I, j] = - flux_x[I, j] * inv_dx * δ(Σ, (1,), (Face(),), I, j)
        end
    end
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            Fz[i, J] = - flux_z[i, J] * inv_dz * δ(Σ, (2,), (Face(),), i, J)
        end
    end
    periodize!(domain, (Fx, Fz))
    return Fx, Fz
end

"""
    C = faces_to_cells!(C, model, Fx, Fz)

C = ℑx(Fx) + ℑz(Fz): the terms on the faces averaged to the interior cells, as the model does for ρε.
"""
function faces_to_cells!(C, model, Fx, Fz)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            C[i, j] = ℑ(Fx, (1,), (Cell(),), i, j) + ℑ(Fz, (2,), (Cell(),), i, j)
        end
    end
    return fill_halo!(domain, C, ZeroGradient())
end

"""
    Fx, Fz = vortex_work!(Fx, Fz, model, advection, mass_flux_x, mass_flux_z)

Work of the vortex force on the faces, -ρu·(ω×u): minus the mass flux times the vortex force (vortex_x, vortex_z) of the
tendencies.
"""
function vortex_work!(Fx, Fz, model, (; vortex_x, vortex_z), mass_flux_x, mass_flux_z)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_faces(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            Fx[I, j] = - mass_flux_x[I, j] * vortex_x[I, j]
        end
    end
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            Fz[i, J] = - mass_flux_z[i, J] * vortex_z[i, J]
        end
    end
    periodize!(domain, (Fx, Fz))
    return Fx, Fz
end

"""
    Fx, Fz = velocity_work!(Fx, Fz, model, (u, w), (du, dw))

u ∂ₜu and w ∂ₜw [W kg⁻¹] on the faces (both walls of a bounded dimension); their average to the cells (`faces_to_cells!`)
is the tendency of K = (ℑx(u²) + ℑz(w²))/2. The tendencies du, dw have no halo faces in a periodic dimension, so the
product is periodized rather than read there.
"""
function velocity_work!(Fx, Fz, model, (u, w), (du, dw))
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_faces(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            Fx[I, j] = u[I, j] * du[I, j]
        end
    end
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            Fz[i, J] = w[i, J] * dw[i, J]
        end
    end
    periodize!(domain, (Fx, Fz))
    return Fx, Fz
end

"""
    dρK = kinetic_energy_tendency!(dρK, model::FC2D, ρ, K, dK, dρ)

Tendency of ρ K, ρ dK + K dρ [W m⁻³] at the interior cells (for AN2D, ρ_R dK).
"""
function kinetic_energy_tendency!(dρK, model, ρ, K, dK, dρ)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            dρK[i, j] = ρ[i, j] * dK[i, j] + K[i, j] * dρ[i, j]
        end
    end
    return fill_halo!(domain, dρK, ZeroGradient())
end

"""
    dρP = potential_energy_tendency!(dρP, model::AN2D, conjvar, chempot, dρconsvar, dρq)
    dρP = potential_energy_tendency!(dρP, model::FC2D, Σ_B, conjvar, chempot, dρ, dρconsvar, dρq)

Tendency of ρ P [W m⁻³] at the interior cells from those of the densities, with d(ρ P) = Σ_B dρ + Σ_ϑ d(ρϑ) + Σ_q d(ρq)
(from d(ρ h - p) = h dρ + ρ (v dp + h_ϑ dϑ + h_q dq) - dp); dρ = 0 for AN2D, where ρ_R P depends on ϑ, q only.
"""
function potential_energy_tendency!(dρP, model::AN2D, conjvar, chempot, dρconsvar, dρq)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            dρP[i, j] = conjvar[i, j] * dρconsvar[i, j] + chempot[i, j] * dρq[i, j]
        end
    end
    return fill_halo!(domain, dρP, ZeroGradient())
end
function potential_energy_tendency!(dρP, model::FC2D, Σ_B, conjvar, chempot, dρ, dρconsvar, dρq)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            dρP[i, j] = Σ_B[i, j] * dρ[i, j] + conjvar[i, j] * dρconsvar[i, j] + chempot[i, j] * dρq[i, j]
        end
    end
    return fill_halo!(domain, dρP, ZeroGradient())
end

"""
    res = heating_residual!(res, model, Q, Q_ϑ, Q_q, ρε)

Closure of the energy budget, Q - Q_ϑ - Q_q - ρ ε_K [W m⁻³] at the interior cells.
"""
function heating_residual!(res, model, Q, Q_ϑ, Q_q, ρε)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            res[i, j] = Q[i, j] - Q_ϑ[i, j] - Q_q[i, j] - ρε[i, j]
        end
    end
    return fill_halo!(domain, res, ZeroGradient())
end

"""
    res = kinetic_budget_residual!(res, model, dρK, ∇J_rev, ∇J_irr, C, ρε, VF)

∂ₜ(ρK) + ∇·(J_K^rev + J_K^irr) - C + ρ ε_K - VF [W m⁻³] at the interior cells: not zero at a cell, but its sum over the
box is (see `energy_budget`).
"""
function kinetic_budget_residual!(res, model, dρK, ∇J_rev, ∇J_irr, C, ρε, VF)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            res[i, j] = dρK[i, j] + ∇J_rev[i, j] + ∇J_irr[i, j] - C[i, j] + ρε[i, j] - VF[i, j]
        end
    end
    return fill_halo!(domain, res, ZeroGradient())
end

"""
    res = potential_budget_residual!(res, model, dρP, ∇J_rev, ∇J_irr, C, Q_ϑ, Q_q, Q)

∂ₜ(ρP) + ∇·(J_P^rev + J_P^irr) + C + Q_ϑ + Q_q - Q [W m⁻³] at the interior cells: zero to round-off, by the product rule
of the operators (for AN2D, up to the divergence of ρ_R u).
"""
function potential_budget_residual!(res, model, dρP, ∇J_rev, ∇J_irr, C, Q_ϑ, Q_q, Q)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            res[i, j] = dρP[i, j] + ∇J_rev[i, j] + ∇J_irr[i, j] + C[i, j] + Q_ϑ[i, j] + Q_q[i, j] - Q[i, j]
        end
    end
    return fill_halo!(domain, res, ZeroGradient())
end

#=
Manipulated conversion, C = -∇·J_C + C_b + C_D, which is exact in the continuous equations only
=#

"""
    pa = pressure_anomaly!(pa, model, reference, pressure)

p - p_R(z) [Pa] at the interior cells.
"""
function pressure_anomaly!(pa, model, (; p_R), pressure)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            pa[i, j] = pressure[i, j] - p_R[j]
        end
    end
    return fill_halo!(domain, pa, ZeroGradient())
end

"""
    b = reference_buoyancy!(b, model, reference, state, scratch)

g (ρ_R v - 1) [m s⁻²] at the interior cells, with the specific volume v = 1/ρ of the state (FC2D) or v(ϑ, q, p_R) at the
reference pressure (AN2D).
"""
function reference_buoyancy!(b, model::AN2D, (; ρ_R, p_R), state, (; thermo))
    (; mgr, domain, space, fluid) = model
    (; g) = space
    (; consvar, q) = thermo
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            b[i, j] = g * (ρ_R[j] * fluid(:p, :consvar, :q).specific_volume(p_R[j], consvar[i, j], q[i, j]) - 1)
        end
    end
    return fill_halo!(domain, b, ZeroGradient())
end
function reference_buoyancy!(b, model::FC2D, (; ρ_R), (; m), scratch)
    (; mgr, domain, space) = model
    (; g) = space
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            b[i, j] = g * (ρ_R[j] / m[i, j, 1] - 1)
        end
    end
    return fill_halo!(domain, b, ZeroGradient())
end

"""
    Cb_z, Cb = buoyancy_conversion!(Cb_z, Cb, model, mass_flux_z, b)

C_b = ρ b w: the mass flux times the buoyancy averaged to the w-faces, and its average to the interior cells.
"""
function buoyancy_conversion!(Cb_z, Cb, model, mass_flux_z, b)
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            Cb_z[i, J] = mass_flux_z[i, J] * ℑ(b, (2,), (Face(),), i, J)
        end
    end
    periodize!(domain, Cb_z)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            Cb[i, j] = ℑ(Cb_z, (2,), (Cell(),), i, j)
        end
    end
    fill_halo!(domain, Cb, ZeroGradient())
    return Cb_z, Cb
end

"""
    C_D = pressure_dilatation!(C_D, model, pressure_anomaly, divergence)

C_D = (p - p_R) ∇·u [W m⁻³] for FC2D; for AN2D it is zero (left as the zero array: the dilatation is part of C_b, see
energy.tex).
"""
pressure_dilatation!(C_D, ::AN2D, pressure_anomaly, divergence) = C_D
pressure_dilatation!(C_D, model::FC2D, pressure_anomaly, divergence) = pointwise!(C_D, model.mgr, *, pressure_anomaly, divergence)

"""
    res = conversion_residual!(res, model, C, ∇J_C, C_b, C_D)

C + ∇·J_C - C_b - C_D [W m⁻³] at the interior cells: the error of the manipulations beyond discrete exactness.
"""
function conversion_residual!(res, model, C, ∇J_C, C_b, C_D)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            res[i, j] = C[i, j] + ∇J_C[i, j] - C_b[i, j] - C_D[i, j]
        end
    end
    return fill_halo!(domain, res, ZeroGradient())
end

"""
    budget = energy_budget(session)

Volume integrals of the energy budget of an open diagnostics session, from the energetics diagnostics (see
`energetics_diagnostics`), over the interior cells (per unit length along y) as a named tuple:
- `kinetic`, `potential`: ∫ρK dV and ∫ρP dV [J m⁻¹],
- `tendency`: ∫(∂ₜ(ρK) + ∂ₜ(ρP)) dV [W m⁻¹],
- `boundary_flux`: ∮(J_K^rev + J_K^irr + J_P^rev + J_P^irr)·dS, the sum of the divergences of the fluxes (it telescopes
  to the fluxes through the boundary) [W m⁻¹],
- `vortex_work`: ∫VF dV [W m⁻¹],
- `residual`: tendency + boundary_flux - vortex_work, zero to round-off for the energy-consistent schemes.
"""
function energy_budget(session)
    (; domain, dx, dz) = session.model
    integral(name) = sum(interior(domain, CookBooks.get(session, name))) * dx * dz
    kinetic, potential = integral(:ρK), integral(:ρP)
    tendency = integral(:dρK) + integral(:dρP)
    boundary_flux = integral(:div_kinetic_flux_rev) + integral(:div_kinetic_flux_irr) + integral(:div_potential_flux_rev) + integral(:div_potential_flux_irr)
    vortex_work = integral(:vortex_work)
    return (; kinetic, potential, tendency, boundary_flux, vortex_work, residual = tendency + boundary_flux - vortex_work)
end
