## AVAILABLE ENERGY OF THE PARCELS OF AN2D
#
# Diagnostics of an `AnelasticLorenzReference` (anelastic_reference.jl), computed after `reference_state!` and
# `reference_heights!` for the same state: the available energy of each parcel and its two parts. With the potential
# Σ(ϑ, q, z, p) = h(ϑ, q, P_R(z)) + (p - P_R(z))/ρ_R(z) + g z of AN2D, the available energy (AE) of a parcel at height z
# and pressure p, of reference height z_*, is
#     Π = Σ(ϑ, q, z, p) - Σ(ϑ, q, z_*, P_*(z_*)) - (p - P_*(z)) Σ_p(ϑ, q, z, p) = Π₁ + Π₂,
# with Σ_p = 1/ρ_R, and its two parts: the available compressible energy (ACE)
#     Π₁ = Σ(ϑ, q, z, p) - Σ(ϑ, q, z, P_*(z)) - (p - P_*(z)) Σ_p(ϑ, q, z, p) = 0,
# the compression from P_*(z) to p at fixed z, zero as Σ is linear in p, and the available potential energy (APE)
#     Π₂ = Σ(ϑ, q, z, P_*(z)) - Σ(ϑ, q, z_*, P_*(z_*))
#        = h(ϑ, q, P_R(z)) - h(ϑ, q, P_R(z_*)) + π_*(z) - π_*(z_*) + g (z - z_*) ≥ 0,
# the displacement from z_* to z through the reference state, with π_* = (P_* - P_R)/ρ_R. Π₂ = S(z) - S(z_*), with
# S(z) = h(P_R(z), ϑ, q) + π_*(z) + g z the static energy of the parcel along the reference profile, measured from the
# parcel's own height (`static_energy`), so Π₂ = -(S(z_*) - S(z)).

"""
    Π₁ = available_compressible_energy!(Π₁, model::AN2D, ref)

The available compressible energy (ACE) Π₁ = Σ(z, p) - Σ(z, P_*(z)) - (p - P_*(z))/ρ_R [J kg⁻¹], on the cells (interior
and halo): zero, as the potential Σ of AN2D is linear in the pressure p.
"""
available_compressible_energy!(Π₁, model::AN2D, ref::AnelasticLorenzReference) = fill!(Π₁, 0)

"""
    Π₂ = available_potential_energy!(Π₂, model::AN2D, ref, z_star)

The available potential energy (APE)
    Π₂ = h(P_R(z), ϑ, q) - h(P_R(z_*), ϑ, q) + π_*(z) - π_*(z_*) + g (z - z_*) = S(z) - S(z_*) ≥ 0   [J kg⁻¹]
of the parcel of each cell, at its own height z and its reference height `z_star` (`reference_heights!`), on the cells
(interior and zero-gradient halo), by the model's manager.
"""
function available_potential_energy!(Π₂, model::AN2D, ref::AnelasticLorenzReference, z_star)
    (; mgr, domain, fluid, space) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        for j in jrange, i in irange
            n = parcel_index(domain, i, j)
            # S is measured from the parcel's own height, S(z) = 0, so Π₂ = -S(z_*)
            Π₂[i, j] = -static_energy(ref, fluid, space.g, n, z_star[i, j])
        end
    end
    fill_halo!(domain, Π₂, ZeroGradient())
    return Π₂
end

"""
    ΔE = rearrangement_energy(model::AN2D, ref)

The potential energy released by the rearrangement of the parcels into the placement of `ref`, E_P - E_* =
-Σ_n Σ_k x_nk c_nk [J m⁻¹], the objective of the rearrangement, with the costs measured from each parcel's own row o,
c_nk = h(P_R(z_k), ϑ_n, q_n) - h(P_R(z_o), ϑ_n, q_n) + g (z_k - z_o) (`anelastic_cost`). The slots are rows, so the other
terms of the potential energy E_P = Σ_n m_n (h(P_R(z_n), ϑ_n, q_n) + g (z_n - z0)) - P_R(z_n) Δx Δz
(`integrated_potential_energy` of the energetics), -g z0 m_n and -P_R Δx Δz, are the same in E_* and cancel. It is not
the available energy, which is computed parcel by parcel (Σ_n m_n Π_n); the two are equal for the optimal placement
when every reference height is at a row centre.
"""
function rearrangement_energy(model::AN2D, ref::AnelasticLorenzReference)
    (; placement) = ref
    total = zero(eltype(ref.mass))
    for e in 1:placement.count
        total -= placement.mass[e] * anelastic_cost(model, ref, placement.parcel[e], placement.slot[e])
    end
    return total
end
