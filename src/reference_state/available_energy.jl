## AVAILABLE ENERGY OF THE PARCELS
#
# The available energy Π = Π₁ + Π₂ of each parcel, its available compressible energy (ACE) Π₁ and its available
# potential energy (APE) Π₂, and their integrals over the parcels.

"""
    total = parcel_integral(model, ref, X)

Σ_n m_n X_n [J m⁻¹ for X in J kg⁻¹], the integral of ρ X over the interior cells, with m_n the mass of parcel n of the
reference state `ref` (ρ = ρ_R for AN2D) and X a field on the cells.
"""
function parcel_integral(model, ref, X)
    (; domain) = model
    total = zero(eltype(X))
    for j in interior_cells(domain, 2), i in interior_cells(domain, 1)
        total += ref.parcels.mass[parcel_index(domain, i, j)] * X[i, j]
    end
    return total
end

"""
    Π₂ = available_potential_energy!(Π₂, model, ref, z_star)

The available potential energy (APE) Π₂ = S(z) - S(z_*) ≥ 0 [J kg⁻¹] of the parcel of each cell, at its own height z and
its reference height `z_star` (`reference_heights!`), with S its static energy along the reference profile
(`static_energy`): h(P_R(z), ϑ, q) - h(P_R(z_*), ϑ, q) + π_*(z) - π_*(z_*) + g (z - z_*) for AN2D,
h(P_*(z), ϑ, q) - h(P_*(z_*), ϑ, q) + g (z - z_*) for FC2D. On the cells (interior and zero-gradient halo), by the model's
manager. Both values of S are measured from the parcel's own pressure and height, as when z_* is found, so that Π₂ ≥ 0
holds exactly (S(z) = 0 for AN2D, where z is a level).
"""
function available_potential_energy!(Π₂, model, ref::AbstractLorenzReference, z_star)
    (; mgr, domain, fluid, space) = model
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        for j in jrange, i in irange
            n = parcel_index(domain, i, j)
            Π₂[i, j] = static_energy(ref, fluid, space.g, n, ref.parcels.height[n]) - static_energy(ref, fluid, space.g, n, z_star[i, j])
        end
    end
    fill_halo!(domain, Π₂, ZeroGradient())
    return Π₂
end

## AN2D
#
# Diagnostics of an `AnelasticLorenzReference`, computed after `reference_state!` and
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
# parcel's own height (`static_energy`).

"""
    Π₁ = available_compressible_energy!(Π₁, model::AN2D, ref)

The available compressible energy (ACE) Π₁ = Σ(z, p) - Σ(z, P_*(z)) - (p - P_*(z))/ρ_R [J kg⁻¹], on the cells (interior
and halo): zero, as the potential Σ of AN2D is linear in the pressure p.
"""
available_compressible_energy!(Π₁, model::AN2D, ref::AnelasticLorenzReference) = fill!(Π₁, 0)

"""
    ΔE = rearrangement_energy(model::AN2D, ref)

The potential energy released by the rearrangement of the parcels into the placement of `ref`, E_P - E_* =
-Σ_n Σ_k x_nk c_nk [J m⁻¹], the objective of the rearrangement, with the costs measured from each parcel's own row o,
c_nk = h(P_R(z_k), ϑ_n, q_n) - h(P_R(z_o), ϑ_n, q_n) + g (z_k - z_o) (`parcel_cost`). The slots are rows, so the other
terms of the potential energy E_P = Σ_n m_n (h(P_R(z_n), ϑ_n, q_n) + g (z_n - z0)) - P_R(z_n) Δx Δz
(`integrated_potential_energy` of the energetics), -g z0 m_n and -P_R Δx Δz, are the same in E_* and cancel. It is not
the available energy, which is computed parcel by parcel (Σ_n m_n Π_n); the two are equal for the optimal placement
when every reference height is at a row centre.
"""
function rearrangement_energy(model::AN2D, ref::AnelasticLorenzReference)
    (; placement) = ref.column
    total = zero(eltype(ref.parcels.mass))
    for e in 1:placement.count
        total -= placement.mass[e] * parcel_cost(ref, model, placement.parcel[e], placement.slot[e])
    end
    return total
end

## FC2D
#
# Diagnostics of a `CompressibleLorenzReference`, computed after `reference_state!` and
# `reference_heights!` for the same state: the available energy of each parcel and its two parts. With the potential
# Σ(ϑ, q, z, p) = h(ϑ, q, p) + g z of FC2D, the available energy (AE) of a parcel at height z and pressure p, of specific
# volume v = 1/ρ and reference height z_*, is
#     Π = Σ(ϑ, q, z, p) - Σ(ϑ, q, z_*, P_*(z_*)) - (p - P_*(z)) Σ_p(ϑ, q, z, p)
#       = h(p) + g z - h(P_*(z_*)) - g z_* - (p - P_*(z)) v = Π₁ + Π₂,
# with Σ_p = v, and its two parts, each nonnegative: the available compressible energy (ACE)
#     Π₁ = Σ(ϑ, q, z, p) - Σ(ϑ, q, z, P_*(z)) - (p - P_*(z)) Σ_p(ϑ, q, z, p) = h(p) - h(P_*(z)) - (p - P_*(z)) v,
# the compression from P_*(z) to p at fixed z, nonnegative as h is concave in p (∂h/∂p = v, ∂v/∂p < 0), and the
# available potential energy (APE)
#     Π₂ = Σ(ϑ, q, z, P_*(z)) - Σ(ϑ, q, z_*, P_*(z_*)) = h(P_*(z)) - h(P_*(z_*)) + g (z - z_*) = S(z) - S(z_*),
# the displacement from z_* to z through the reference state, with S(z) = h(P_*(z), ϑ, q) + g z the static energy of the
# parcel along the reference profile, nonnegative as z_* is its minimiser.

"""
    Π₁ = available_compressible_energy!(Π₁, model::FC2D, ref)

The available compressible energy (ACE)
    Π₁ = h(p, ϑ, q) - h(P_*(z), ϑ, q) - (p - P_*(z)) v = -∫ (s - P_*(z)) ∂v/∂p(s, ϑ, q) ds ≥ 0   [J kg⁻¹],
the integral from P_*(z) to p, of the parcel of each cell, at its own height z, pressure p and specific volume
v = v(p, ϑ, q), by quadrature (`available_compressible_energy`), on the cells (interior and zero-gradient halo), by the
model's manager.
"""
function available_compressible_energy!(Π₁, model::FC2D, ref::CompressibleLorenzReference)
    (; mgr, domain, fluid) = model
    (; consvar, q, pressure) = ref.parcels
    (; p_star) = ref
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            P_star = p_star[j]                                      # P_*(z) at the centre of the parcel's row
            Π₁[i, j] = available_compressible_energy(fluid, P_star, pressure[n] - P_star, consvar[n], q[n])
        end
    end
    fill_halo!(domain, Π₁, ZeroGradient())
    return Π₁
end

"""
    ΔE = rearrangement_energy(model::FC2D, ref)

The potential energy released by the rearrangement of the parcels into the placement of `ref` [J m⁻¹], the objective
of the rearrangement,
    E_P - E_* = Σ_n [m_n g z_n - (p_n - P_top) Δx Δz] - Σ_n Σ_k x_nk (h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n)),
from the potential energy of the parcels at their own level (`integrated_potential_energy` of the energetics),
    E_P = Σ_n m_n (h(p_n, ϑ_n, q_n) + g (z_n - z0)) - p_n Δx Δz,
and that of the column, E_* = Σ_nk x_nk (h(P_k, ϑ_n, q_n) + g (z_k - z0)) - Σ_k P_k V_k. With the slot pressures
P_k = P_top + g (Σ_{l<k} M_l + M_k/2)/L_x, the slots of thickness V_k/L_x stacked from the bottom, and Σ_k V_k = L_x L_z,
Σ_k M_k g z_k - Σ_k P_k V_k = -P_top L_x L_z (both sums are g Σ_k (Σ_{l<k} M_l + M_k/2) V_k/L_x), so
E_* = Σ_nk x_nk h(P_k, ϑ_n, q_n) - g z0 M - P_top L_x L_z: the terms -g z0 M and -P_top L_x L_z = -Σ_n P_top Δx Δz
cancel, and m_n h(p_n) = Σ_k x_nk h(p_n) makes the differences of h the costs (`parcel_cost`). It is not the
available energy, which is computed parcel by parcel (Σ_n m_n Π_n).
"""
function rearrangement_energy(model::FC2D, ref::CompressibleLorenzReference)
    (; space, dx, dz) = model
    (; mass, height, pressure) = ref.parcels
    (; placement) = ref.column
    total = zero(eltype(mass))
    # the parcels at their own level
    for n in eachindex(mass)
        total += mass[n] * space.g * height[n] - (pressure[n] - ref.p_top) * dx * dz
    end
    # the parcels in the slots
    for e in 1:placement.count
        total -= placement.mass[e] * parcel_cost(ref, model, placement.parcel[e], placement.slot[e])
    end
    return total
end
