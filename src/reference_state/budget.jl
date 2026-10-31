## KERNELS OF THE BUDGETS OF THE AVAILABLE ENERGY
#
# The budgets of the available compressible energy (ACE) Π₁ and the available potential energy (APE) Π₂ of the energy
# notes, as diagnostics of AN2D and FC2D (recipes in entries.jl), computed after the reference state and the reference
# heights z_* of the same state. With the potential Σ(ϑ, q, z, p) of the model (h(p) + g z for FC2D,
# h(P_R) + (p - P_R)/ρ_R + g z for AN2D), its derivatives Σ_ϑ, Σ_q evaluated at three states of a parcel,
#     the state itself (z, p):                Σ_ϑ,   Σ_q    (the model's conjugate variable and chemical potential),
#     the intermediate state (z, P_*(z)):     Σ̃_ϑ,   Σ̃_q    (`_at_p_star`),
#     the reference state (z_*, P_*(z_*)):    Σ*_ϑ,  Σ*_q   (`_at_z_star`),
# and their differences, Δ₁Σ_ϑ = Σ_ϑ - Σ̃_ϑ (`_ACE`, zero for AN2D, where Σ_ϑ does not depend on p) and
# Δ₂Σ_ϑ = Σ̃_ϑ - Σ*_ϑ (`_APE`), the material derivatives of the energy notes give, at a fixed reference state,
#     ∂ₜ(ρΠ₁) = B₁ ∂ₜρ + Δ₁Σ_ϑ ∂ₜ(ρϑ) + Δ₁Σ_q ∂ₜ(ρq),   B₁ = Π₁ + (p - P_*) Σ_p - ϑ Δ₁Σ_ϑ - q Δ₁Σ_q,
#     ∂ₜ(ρΠ₂) = B₂ ∂ₜρ + Δ₂Σ_ϑ ∂ₜ(ρϑ) + Δ₂Σ_q ∂ₜ(ρq),   B₂ = Π₂ - ϑ Δ₂Σ_ϑ - q Δ₂Σ_q,
# with Σ_p = 1/ρ (1/ρ_R for AN2D, where ∂ₜρ = 0). These are written as the budget of ρP is (energetics/kernels.jl), in
# Bernoulli form, from the fluxes of the tendencies and the operators of the model: with the reversible flux
#     J^rev = ρu ℑ(B) + ρϑu ℑ(Δ Σ_ϑ) + ρqu ℑ(Δ Σ_q),
# the irreversible flux J^irr = ℑ(Δ Σ_ϑ) J_ϑ + ℑ(Δ Σ_q) J_q, the conversion
#     C_B = -ρu·∇B - ρϑu·∇(Δ Σ_ϑ) - ρqu·∇(Δ Σ_q)
# and the heatings Q_ϑ = -J_ϑ·∇Σ_ϑ, Q_q = -J_q·∇Σ_q, Q = Σ_ϑ σ_ϑ at each state, the product rule of the operators,
# δ(ℑ(a) b) = a δb + ℑ(b δa), gives at each cell, to round-off (for AN2D, up to the divergence of ρ_R u),
#     ∂ₜ(ρΠ₁) = -∇·(J_ACE^rev + J_ACE^irr) - C_ACE - (Q_ϑ - Q̃_ϑ) - (Q_q - Q̃_q) + Q - Q̃,
#     ∂ₜ(ρΠ₂) = -∇·(J_APE^rev + J_APE^irr) - C_APE - (Q̃_ϑ - Q*_ϑ) - (Q̃_q - Q*_q) + Q̃ - Q*,
# plus the terms in ∂ₜP_* (R_Π₁, R_Π₂ of the energy notes, below). In the continuous equations
#     J_ACE^rev = ρΠ₁u + J_C*,   C_ACE = C_b,1* + C_D* - ∇·J_C*,   J_APE^rev = ρΠ₂u,   C_APE = C_b,2*,
# with J_C* = (p - P_*) u, the buoyancy conversions C_b,1* = ρ g ρ_* (v - ṽ) w, C_b,2* = ρ g ρ_* (ṽ - v_*) w (FC2D;
# ṽ = v(ϑ, q, P_*(z))), C_b,1* = 0, C_b,2* = ρ_R g ρ_R (v(ϑ, q, P_R) - v_*) w (AN2D), and the pressure dilatation
# C_D* = (p - P_*) ∇·u (zero for AN2D). These forms of the energy notes are diagnosed too, with the residuals of the
# manipulation. The conversion C from P to K of the energetics is C_ACE + C_APE + C_bg, with the conversion C_bg of the
# background, which is zero in the continuous equations (Σ(ϑ, q, z_*, P_*(z_*)) depends on x through ϑ, q only, at the
# level of neutral buoyancy).
#
# The differences of the potentials are integrals over the pressure by quadrature (`potential_difference`), which keep the
# precision of the float type. Fields on the cells are computed at the interior cells, with a zero-gradient halo.

"""
    P = intermediate_pressure(model, ref, j)

The pressure [Pa] of the intermediate state at the height of the row of cell index j, at which the potentials Σ̃_ϑ, Σ̃_q
are evaluated: P_*(z_j) for FC2D, P_R(z_j) for AN2D (whose Σ_ϑ, Σ_q depend on the pressure through P_R only).
"""
@inline intermediate_pressure(model::FC2D, ref::CompressibleLorenzReference, j) = ref.p_star[j]
@inline intermediate_pressure(model::AN2D, ref::AnelasticLorenzReference, j) = model.reference.p_R[j]

"""
    ΔΣ = compression_potential!(ΔΣ, model, ref, c)

Δ₁Σ = Σ_c(ϑ, q, z, p) - Σ_c(ϑ, q, z, P_*(z)), the change of the potential Σ_ϑ (c = 3) or Σ_q (c = 4) from the
intermediate state to the parcel's own pressure p, by quadrature (`potential_difference`), at the interior cells: the
potential of the available compressible energy. Zero for AN2D, whose potentials depend on p through P_R only.
"""
compression_potential!(ΔΣ, model::AN2D, ref::AnelasticLorenzReference, c) = fill!(ΔΣ, 0)
function compression_potential!(ΔΣ, model::FC2D, ref::CompressibleLorenzReference, c)
    (; mgr, domain, fluid) = model
    (; consvar, q, pressure) = ref.parcels
    (; p_star) = ref
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            # from P_*(z) to the parcel's own pressure p
            ΔΣ[i, j] = potential_difference(fluid, p_star[j], pressure[n] - p_star[j], consvar[n], q[n], c)
        end
    end
    return fill_halo!(domain, ΔΣ, ZeroGradient())
end

"""
    ΔΣ = displacement_potential!(ΔΣ, model, ref, z_star, c)

Δ₂Σ = Σ_c(ϑ, q, z, P_*(z)) - Σ_c(ϑ, q, z_*, P_*(z_*)), the change of the potential Σ_ϑ (c = 3) or Σ_q (c = 4) from the
reference state to the intermediate state, by quadrature (`potential_difference`) from the pressure P_*(z_*) (P_R(z_*)
for AN2D) to the intermediate pressure, at the interior cells: the potential of the available potential energy.
"""
function displacement_potential!(ΔΣ, model, ref, z_star, c)
    (; mgr, domain, fluid) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        for j in jrange, i in irange
            n = parcel_index(domain, i, j)
            P = intermediate_pressure(model, ref, j)                # P_*(z), or P_R(z) for AN2D
            P_ref = hermite(ref.pressure, z_star[i, j])[1]          # P_*(z_*), or P_R(z_*) for AN2D
            ΔΣ[i, j] = potential_difference(fluid, P_ref, P - P_ref, ref.parcels.consvar[n], ref.parcels.q[n], c)
        end
    end
    return fill_halo!(domain, ΔΣ, ZeroGradient())
end

"""
    B₁ = compression_bernoulli!(B₁, model, ACE, pressure_anomaly, density, (; consvar, q), Δϑ, Δq)
    B₂ = displacement_bernoulli!(B₂, model, APE, (; consvar, q), Δϑ, Δq)

The coefficients of ∂ₜρ in the tendencies of ρΠ₁ and ρΠ₂ at a fixed reference state [J kg⁻¹], at the interior cells:
    B₁ = Π₁ + (p - P_*)/ρ - ϑ Δ₁Σ_ϑ - q Δ₁Σ_q,   B₂ = Π₂ - ϑ Δ₂Σ_ϑ - q Δ₂Σ_q,
with ρ = ρ_R for AN2D, where B₁ = (p - P_*)/ρ_R. Their gradients along the mass flux, with those of the differences of
the potentials along the tracer fluxes, are the conversions C_ACE and C_APE (`bernoulli_conversion!`).
"""
compression_bernoulli!(B₁, model, ACE, pressure_anomaly, density, (; consvar, q), Δϑ, Δq) =
    cellwise!(B₁, model, (Π₁, pa, ρ, ϑ, q, Δϑ, Δq) -> Π₁ + pa / ρ - ϑ * Δϑ - q * Δq, ACE, pressure_anomaly, density, consvar, q, Δϑ, Δq)
displacement_bernoulli!(B₂, model, APE, (; consvar, q), Δϑ, Δq) =
    cellwise!(B₂, model, (Π₂, ϑ, q, Δϑ, Δq) -> Π₂ - ϑ * Δϑ - q * Δq, APE, consvar, q, Δϑ, Δq)

"""
    C = bernoulli_conversion!(C, model, faces, (mass_flux_x, mass_flux_z), (mu, mw), k, B, Δϑ, Δq)
    F = face_bernoulli_product!(F, model, dir, mass_flux, tracer_flux, k, B, Δϑ, Δq)

The conversion C_B = -ρu·∇B - ρϑu·∇(Δ Σ_ϑ) - ρqu·∇(Δ Σ_q) [W m⁻³] at the interior cells: on the faces,
F = -(ρu δB + ρϑu δ(Δ Σ_ϑ) + ρqu δ(Δ Σ_q))/Δ along each direction, with the mass flux and the tracer fluxes
`mu[:, :, k]` (ϑ) and `mu[:, :, k + 1]` (q) of the tendencies (`face_bernoulli_product!`, on the faces of direction
`dir`), averaged to the cells. With the Bernoulli coefficient B and the potentials of the ACE or the APE, it is C_ACE or
C_APE.
"""
function bernoulli_conversion!(C, model, (Fx, Fz), (mass_flux_x, mass_flux_z), (mu, mw), k, B, Δϑ, Δq)
    face_bernoulli_product!(Fx, model, Val(1), mass_flux_x, mu, k, B, Δϑ, Δq)
    face_bernoulli_product!(Fz, model, Val(2), mass_flux_z, mw, k, B, Δϑ, Δq)
    return faces_to_cells!(C, model, Fx, Fz)
end
function face_bernoulli_product!(F, model, dir, mass_flux, tracer_flux, k, B, Δϑ, Δq)
    (; mgr, domain, spatial_scheme) = model
    (; δ) = operators(spatial_scheme)
    inv_Δ = inv_spacing(model, dir)
    @with mgr, let (irange, jrange) = face_ranges(domain, dir)
        @inbounds @vec for i in irange, j in jrange
            F[i, j] = - inv_Δ * (mass_flux[i, j] * δ(B, along(dir), (Face(),), i, j) +
                                 tracer_flux[i, j, k] * δ(Δϑ, along(dir), (Face(),), i, j) +
                                 tracer_flux[i, j, k+1] * δ(Δq, along(dir), (Face(),), i, j))
        end
    end
    return periodize!(domain, F)
end

"""
    b₁ = compression_buoyancy!(b₁, model, ref)
    b₂ = displacement_buoyancy!(b₂, model, ref)

The buoyancies of the conversions of the energy notes [m s⁻²], at the interior cells: b₁ = g ρ_* (v - ṽ), from the
compression from P_*(z) to p, and b₂ = g ρ_* (ṽ - v_*), relative to the reference state at the intermediate pressure, with
v the parcel's specific volume, ṽ = v(ϑ, q, P_*(z)) and ρ_* = 1/v_* at the parcel's height (FC2D); b₁ = 0 and
b₂ = g ρ_R (v(ϑ, q, P_R) - v_*) (AN2D). C_b,1* = ρ b₁ w and C_b,2* = ρ b₂ w (`buoyancy_conversion!`).
"""
compression_buoyancy!(b₁, model::AN2D, ref::AnelasticLorenzReference) = fill!(b₁, 0)
function compression_buoyancy!(b₁, model::FC2D, ref::CompressibleLorenzReference)
    (; mgr, domain, fluid, space) = model
    (; consvar, q, volume) = ref.parcels
    (; p_star, v_star) = ref
    g = space.g
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ṽ = fluid(:p, :consvar, :q).specific_volume(p_star[j], consvar[n], q[n])    # v(ϑ, q, P_*(z))
            b₁[i, j] = g * (volume[n] - ṽ) / v_star[j]
        end
    end
    return fill_halo!(domain, b₁, ZeroGradient())
end
function displacement_buoyancy!(b₂, model::FC2D, ref::CompressibleLorenzReference)
    (; mgr, domain, fluid, space) = model
    (; consvar, q) = ref.parcels
    (; p_star, v_star) = ref
    g = space.g
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ṽ = fluid(:p, :consvar, :q).specific_volume(p_star[j], consvar[n], q[n])    # v(ϑ, q, P_*(z))
            b₂[i, j] = g * (ṽ / v_star[j] - 1)
        end
    end
    return fill_halo!(domain, b₂, ZeroGradient())
end
function displacement_buoyancy!(b₂, model::AN2D, ref::AnelasticLorenzReference)
    (; mgr, domain, space) = model
    (; ρ_R) = model.reference
    (; volume) = ref.parcels                                       # v(ϑ, q, P_R(z))
    (; v_star) = ref
    g = space.g
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            b₂[i, j] = g * ρ_R[j] * (volume[n] - v_star[j])
        end
    end
    return fill_halo!(domain, b₂, ZeroGradient())
end

## Terms in the tendency of the reference pressure
#
# With P_* changing in time, the tendencies of ρΠ₁ and ρΠ₂ at a fixed height gain the terms of the energy notes
#     R_Π₁ = ρ (Σ_p(ϑ, q, z, p) - Σ_p(ϑ, q, z, P_*(z))) ∂ₜP_*(z),
#     R_Π₂ = ρ Σ_p(ϑ, q, z, P_*(z)) ∂ₜP_*(z) - ρ Σ_p(ϑ, q, z_*, P_*(z_*)) ∂ₜP_*(z_*),
# the derivatives of Π₁ = Σ(z, p) - Σ(z, P_*) - (p - P_*) Σ_p(z, p) and Π₂ = Σ(z, P_*(z)) - Σ(z_*, P_*(z_*)) with respect
# to P_* (the motion of z_* does not enter, at the minimum of the static energy). With Σ_p = v for FC2D and 1/ρ_R for
# AN2D,
#     FC2D:  R_Π₁ = (1 - ρ ṽ) ∂ₜP_*(z),   R_Π₂ = ρ ṽ ∂ₜP_*(z) - ρ v(ϑ, q, P_*(z_*)) ∂ₜP_*(z_*),   ṽ = v(ϑ, q, P_*(z)),
#     AN2D:  R_Π₁ = 0,                     R_Π₂ = ∂ₜP_*(z) - ρ_R(z)/ρ_R(z_*) ∂ₜP_*(z_*),
# where v(ϑ, q, P_*(z_*)) is v_*(z_*) at an interior minimum (level of neutral buoyancy), but not at a wall. Their sum
# R_Π = ∂ₜP_*(z) - ρ Σ_p(z_*) ∂ₜP_*(z_*) integrates to zero over the box by the volume constraint, to the accuracy with which
# the reference heights fill the box.

"""
    ∂ₜP = reference_pressure_tendency_at!(∂ₜP, model, ref, z_star)

∂ₜP_*(z_*) [Pa s⁻¹] at the reference height `z_star` [m] of the parcel of each cell, at the interior cells, after
`reference_tendency!` (`reference_pressure_tendency`).
"""
function reference_pressure_tendency_at!(∂ₜP, model, ref, z_star)
    (; mgr, domain) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        for j in jrange, i in irange
            ∂ₜP[i, j] = reference_pressure_tendency(ref, model, z_star[i, j])
        end
    end
    return fill_halo!(domain, ∂ₜP, ZeroGradient())
end

"""
    R₁ = compression_reference_term!(R₁, model, ref)
    R₂ = displacement_reference_term!(R₂, model, ref, z_star, ∂ₜP_at_z_star)

The terms of the tendencies of ρΠ₁ and ρΠ₂ [W m⁻³] in the tendency of the reference pressure ∂ₜP_*, at the interior cells,
after `reference_tendency!`: R_Π₁ = (1 - ρ v(ϑ, q, P_*(z))) ∂ₜP_*(z) (FC2D; zero for AN2D) and
R_Π₂ = ρ v(ϑ, q, P_*(z)) ∂ₜP_*(z) - ρ v(ϑ, q, P_*(z_*)) ∂ₜP_*(z_*) (FC2D), ∂ₜP_*(z) - ρ_R(z)/ρ_R(z_*) ∂ₜP_*(z_*) (AN2D,
with ρ_R(z_*) = -∂_z P_R(z_*)/g), with ∂ₜP_*(z_*) from `∂ₜP_at_z_star` (`reference_pressure_tendency_at!`).
"""
compression_reference_term!(R₁, model::AN2D, ref::AnelasticLorenzReference) = fill!(R₁, 0)
function compression_reference_term!(R₁, model::FC2D, ref::CompressibleLorenzReference)
    (; mgr, domain, fluid, dx, dz) = model
    (; mass, consvar, q) = ref.parcels
    (; p_star, p_star_tendency) = ref
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ρ = mass[n] / (dx * dz)
            ṽ = fluid(:p, :consvar, :q).specific_volume(p_star[j], consvar[n], q[n])  # v(ϑ, q, P_*(z))
            R₁[i, j] = (1 - ρ * ṽ) * p_star_tendency[j]
        end
    end
    return fill_halo!(domain, R₁, ZeroGradient())
end
function displacement_reference_term!(R₂, model::FC2D, ref::CompressibleLorenzReference, z_star, ∂ₜP_at_z_star)
    (; mgr, domain, fluid, dx, dz) = model
    (; mass, consvar, q) = ref.parcels
    (; p_star, p_star_tendency) = ref
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        for j in jrange, i in irange
            n = parcel_index(domain, i, j)
            ρ = mass[n] / (dx * dz)
            ṽ = fluid(:p, :consvar, :q).specific_volume(p_star[j], consvar[n], q[n])                  # v(ϑ, q, P_*(z))
            P_ref = hermite(ref.pressure, z_star[i, j])[1]                                          # P_*(z_*)
            v_ref = fluid(:p, :consvar, :q).specific_volume(P_ref, consvar[n], q[n])                 # v(ϑ, q, P_*(z_*))
            R₂[i, j] = ρ * (ṽ * p_star_tendency[j] - v_ref * ∂ₜP_at_z_star[i, j])
        end
    end
    return fill_halo!(domain, R₂, ZeroGradient())
end
function displacement_reference_term!(R₂, model::AN2D, ref::AnelasticLorenzReference, z_star, ∂ₜP_at_z_star)
    (; mgr, domain, space) = model
    (; ρ_R) = model.reference
    (; p_star_tendency) = ref
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        for j in jrange, i in irange
            ρ_ref = -hermite(ref.pressure, z_star[i, j])[2] / space.g                              # ρ_R(z_*)
            R₂[i, j] = p_star_tendency[j] - ρ_R[j] / ρ_ref * ∂ₜP_at_z_star[i, j]
        end
    end
    return fill_halo!(domain, R₂, ZeroGradient())
end

"""
    dρΠ = available_energy_tendency!(dρΠ, model, dstate, B, Δϑ, Δq, dρconsvar, dρq, R)

The tendency of ρΠ₁ or ρΠ₂ [W m⁻³] at the interior cells, from those of the model's densities and of the reference
pressure: ∂ₜ(ρΠ) = B ∂ₜρ + ΔΣ_ϑ ∂ₜ(ρϑ) + ΔΣ_q ∂ₜ(ρq) + R, with the Bernoulli coefficient B, the differences of the
potentials ΔΣ_ϑ, ΔΣ_q and the term R in ∂ₜP_* of that energy; ∂ₜρ from the tendencies `dstate` (FC2D; zero for AN2D).
"""
available_energy_tendency!(dρΠ, model::FC2D, dstate, B, Δϑ, Δq, dρconsvar, dρq, R) =
    cellwise!(dρΠ, model, (B, dρ, Δϑ, dρϑ, Δq, dρq, R) -> B * dρ + Δϑ * dρϑ + Δq * dρq + R, B, view(dstate.m, :, :, 1), Δϑ, dρconsvar, Δq, dρq, R)
available_energy_tendency!(dρΠ, model::AN2D, dstate, B, Δϑ, Δq, dρconsvar, dρq, R) =
    cellwise!(dρΠ, model, (Δϑ, dρϑ, Δq, dρq, R) -> Δϑ * dρϑ + Δq * dρq + R, Δϑ, dρconsvar, Δq, dρq, R)

"""
    res = available_energy_budget_residual!(res, model, dρΠ, ∇J_rev, ∇J_irr, C, (Q_ϑ, Q_q, Q), (Q′_ϑ, Q′_q, Q′), R)

The residual of the budget of ρΠ₁ or ρΠ₂ [W m⁻³] at the interior cells,
    ∂ₜ(ρΠ) + ∇·(J^rev + J^irr) + C + (Q_ϑ - Q′_ϑ) + (Q_q - Q′_q) - (Q - Q′) - R,
with the heatings at the two states of the energy, (Q_ϑ, Q_q, Q) at the parcel's state and (Q′_ϑ, Q′_q, Q′) at the
intermediate state for Π₁, at the intermediate and reference states for Π₂: zero to round-off by the product rule of the
operators (for the APE of AN2D, up to the divergence of ρ_R u).
"""
available_energy_budget_residual!(res, model, dρΠ, ∇J_rev, ∇J_irr, C, (Q_ϑ, Q_q, Q), (Q′_ϑ, Q′_q, Q′), R) =
    cellwise!(res, model, (dρΠ, ∇J_rev, ∇J_irr, C, Q_ϑ, Q_q, Q, Q′_ϑ, Q′_q, Q′, R) ->
                  dρΠ + ∇J_rev + ∇J_irr + C + (Q_ϑ - Q′_ϑ) + (Q_q - Q′_q) - (Q - Q′) - R,
              dρΠ, ∇J_rev, ∇J_irr, C, Q_ϑ, Q_q, Q, Q′_ϑ, Q′_q, Q′, R)
