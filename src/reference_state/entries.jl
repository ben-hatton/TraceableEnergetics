## DIAGNOSTICS OF THE LORENZ REFERENCE STATE
#
# The entries of the reference state, the available energy of its parcels, the budgets of the available compressible and
# potential energies and the tendency of the reference state, merged into the book of `diagnostics` with a reference
# state (`add_reference_state!`).

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
function reference_state_entries(model::Union{AN2D{F}, FC2D{F}}, ref, out) where F
    cells, rows = (Cell(), Cell()), (nothing, Cell())
    entries = (;
        # z_* = argmin S(z) over 0 ≤ z ≤ L_z, with S(z) = h(P_*(z), ϑ, q) + g z (FC2D), h(P_R(z), ϑ, q) + π_*(z) + g z (AN2D)
        z_star = entry("reference height z_*, minimiser of the static energy along P_*", "m", cells,
            (model, reference_state) -> reference_heights!(out(:z_star), model, reference_state)),
        wall_minima = entry("number of parcels whose reference height is at a wall", "1", (),
            (model, z_star) -> F(count_wall_minima(model, z_star))),
        # ACE: Π₁ = h(p) - h(P_*(z)) - (p - P_*(z)) v (FC2D); Σ(z, p) - Σ(z, P_*(z)) - (p - P_*(z))/ρ_R = 0, as Σ is
        # linear in p (AN2D)
        ACE = entry(model isa AN2D ? "available compressible energy (ACE) Π₁, zero (Σ linear in p)" :
                                     "available compressible energy (ACE) Π₁ = h(p) - h(P_*(z)) - (p - P_*(z)) v", "J kg⁻¹", cells,
            (model, reference_state) -> available_compressible_energy!(out(:ACE), model, reference_state)),
        # APE: Π₂ = S(z) - S(z_*), h(P_*(z)) - h(P_*(z_*)) + g (z - z_*) (FC2D),
        # h(P_R(z)) - h(P_R(z_*)) + π_*(z) - π_*(z_*) + g (z - z_*) (AN2D)
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
            reference_state -> F(reference_state.column.certificate_violations)),
        # the transportation simplex of ExactRearrangement (no pivots and one pass for SortedRearrangement)
        pivots = entry("number of pivots of the transportation simplex", "1", (), reference_state -> F(reference_state.column.pivots)),
        pricing_passes = entry("number of passes over the parcel-slot pairs", "1", (),
            reference_state -> F(reference_state.column.pricing_passes)),
        # the cost of the sorted placement less that of the placement, zero for SortedRearrangement: Σ_nk x_nk c_nk (AN2D),
        # G = Σ_nk x_nk c_nk - P_top L_x L_z (FC2D)
        sorted_energy_excess = entry("underestimate of E_P - E_* by the sorted placement", "J m⁻¹", (),
            reference_state -> reference_state.column.sorted_energy_excess),
    )
    model isa FC2D || return entries
    # the top of the reference column of FC2D
    return merge(entries, (;
        p_top = entry("pressure P_top at the top of the reference column", "Pa", (), reference_state -> reference_state.p_top),
        # (Σ_k V_k/L_x - L_z)/L_z
        volume_residual = entry("relative excess of the height of the reference column over L_z", "1", (),
            reference_state -> reference_state.volume_residual),
        top_pressure_iterations = entry("number of placements alternating with the top pressure", "1", (),
            reference_state -> F(reference_state.column.top_pressure_iterations)),
    ))
end

"""
    entries = available_energy_budget_entries(model, out, faces)

The diagnostics of the budgets of the available compressible energy (ACE) Π₁ and the available potential energy (APE) Π₂
(kernels in budget.jl), as a named tuple of `entry(long_name, units, location, recipe)`, the same for
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
tendency.jl) and of the terms it adds to the budgets of the available compressible energy (ACE) Π₁ and the
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
