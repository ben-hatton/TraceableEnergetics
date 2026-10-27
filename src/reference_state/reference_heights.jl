## REFERENCE HEIGHTS OF THE PARCELS
#
# The reference height z_* of a parcel is the global minimiser over the box of its static energy S along the reference
# profile; at an interior minimum, its level of neutral buoyancy, the parcel has the specific volume of the reference
# state.

## AN2D
#
# Along the reference profile, a parcel of conserved properties (ϑ, q) has the static energy
#     S(z) = h(P_R(z), ϑ, q) + π_*(z) + g z,                      with π_* = (P_* - P_R)/ρ_R,
# measured from its value at the parcel's own height z_o, with the difference of h by quadrature,
#     S(z) - S(z_o) = ∫ v dp from P_R(z_o) to P_R(z) + π_*(z) - π_*(z_o) + g (z - z_o),
# and its reference height z_* is the global minimiser of S over the box, 0 ≤ z ≤ L_z. At an interior minimum the
# residual of the level of neutral buoyancy (LNB)
#     r(z) = -dS/dz = g ρ_R (v(P_R(z), ϑ, q) - v_*(z))            (from ∂_z P_R = -g ρ_R, ∂_z π_* = g (ρ_R v_* - 1))
# vanishes, and N_*² = d²S/dz² ≥ 0 there; r > 0 means the parcel is lighter than the reference state at z.

"""
    S = static_energy(ref::AnelasticLorenzReference, fluid, g, n, z)
    r = buoyancy_residual(ref::AnelasticLorenzReference, fluid, g, ϑ, q, z)
    N² = stratification(ref::AnelasticLorenzReference, fluid, g, ϑ, q, z)

Along the reference profile of `ref` at the height z [m]:
- the static energy of parcel n, measured from its own row o = `ref.row[n]` (node c = K + 2 - o of π_*):
  S(z) - S(z_o) = ∫ v dp from P_R(z_o) to P_R(z) + π_*(z) - π_*(z_o) + g (z - z_o) [J kg⁻¹] (`enthalpy_difference`);
- for a parcel of conserved variable ϑ and composition q, the LNB residual r = -dS/dz = -(v ∂_z P_R + g + ∂_z π_*)
  [m s⁻²], which is g ρ_R (v - v_*);
- and N_*² = d²S/dz² = ∂v/∂p (∂_z P_R)² + v ∂²_z P_R + ∂²_z π_* [s⁻²].
"""
@inline function static_energy(ref::AnelasticLorenzReference, fluid, g, n, z)
    c = length(ref.potential.z) - ref.row[n]                    # the node of the parcel's own row
    P_own, z_own, π_own = ref.node_pressure[c], ref.potential.z[c], ref.potential.y[c]
    P_R = hermite(ref.pressure, z)[1]
    π = hermite(ref.potential, z)[1]
    return enthalpy_difference(fluid, P_own, P_R - P_own, ref.consvar[n], ref.q[n]) + (π - π_own) + g * (z - z_own)
end
@inline function buoyancy_residual(ref::AnelasticLorenzReference, fluid, g, ϑ, q, z)
    P_R, ∂P_R, _ = hermite(ref.pressure, z)
    ∂π = hermite(ref.potential, z)[2]
    v = fluid(:p, :consvar, :q).specific_volume(P_R, ϑ, q)
    return -(v * ∂P_R + g + ∂π)
end
@inline function stratification(ref::AnelasticLorenzReference, fluid, g, ϑ, q, z)
    P_R, ∂P_R, ∂²P_R = hermite(ref.pressure, z)
    ∂²π = hermite(ref.potential, z)[3]
    v, ∂v_∂p, _, _ = fluid(:p, :consvar, :q).volume_functions(P_R, ϑ, q)
    return ∂v_∂p * ∂P_R^2 + v * ∂²P_R + ∂²π
end

"""
    z_star = reference_height(ref::AnelasticLorenzReference, fluid, g, Lz, n, ζ)

Reference height z_* [m] of parcel n, from the level ζ [m], among the walls and the row centres (the nodes of π_*), at
which its static energy is lowest: the root of the LNB residual next to it, to within sqrt(eps) L_z, if the static
energy is lower there (`refine_minimum`).
"""
function reference_height(ref::AnelasticLorenzReference, fluid, g, Lz, n, ζ)
    S(z) = static_energy(ref, fluid, g, n, z)
    r(z) = buoyancy_residual(ref, fluid, g, ref.consvar[n], ref.q[n], z)
    c = searchsortedfirst(ref.potential.z, ζ)                   # the index of the level
    return refine_minimum(S, r, ref.potential.z, c, S(ζ), sqrt(eps(Lz)) * Lz)[1]
end

"""
    z_star = reference_heights!(z_star, model::AN2D, ref)

Reference height z_* [m] of the parcel of each cell, the minimiser of its static energy S along the profile of `ref`
over the box, on the cells (interior and zero-gradient halo), by the model's manager, in two passes:
1. the level of lowest S among the walls and the row centres (the nodes of π_*), with S measured from the parcel's own
   row from the values of P_R and π_* there, vectorised over the parcels of a row;
2. the root of the LNB residual next to that level, to within sqrt(eps) L_z, if S is lower there (`refine_minimum`).
A local minimum of S next to another level is not refined: it can be lower than the one found only if the two differ
by less than the change of S over a row.
"""
function reference_heights!(z_star, model::AN2D, ref::AnelasticLorenzReference)
    (; mgr, domain, fluid, space) = model
    (; consvar, q, node_pressure) = ref
    (; z, y) = ref.potential                                    # the levels, and π_* there
    g = space.g
    Hz = halo_size(domain)[2]
    # 1. the level of lowest S, the lowest level among equal values, with
    # S(z_c) - S(z_o) = ∫ v dp from P_R(z_o) to P_R(z_c) + π_*(z_c) - π_*(z_o) + g (z_c - z_o)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ϑ_n, q_n = consvar[n], q[n]
            o = j - Hz + 1                                      # the node of the parcel's own row
            P_own, z_own, π_own = node_pressure[o], z[o], y[o]
            lowest = enthalpy_difference(fluid, P_own, node_pressure[1] - P_own, ϑ_n, q_n) + (y[1] - π_own) + g * (z[1] - z_own)
            level = zero(lowest) + z[1]
            for c in 2:length(z)
                S = enthalpy_difference(fluid, P_own, node_pressure[c] - P_own, ϑ_n, q_n) + (y[c] - π_own) + g * (z[c] - z_own)
                level = ifelse(S < lowest, z[c], level)            # ifelse, not ? :, so that it is vectorised
                lowest = min(lowest, S)
            end
            z_star[i, j] = level
        end
    end
    # 2. the minimum next to that level
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        for j in jrange, i in irange
            n = parcel_index(domain, i, j)
            z_star[i, j] = reference_height(ref, fluid, g, space.Lz, n, z_star[i, j])
        end
    end
    fill_halo!(domain, z_star, ZeroGradient())
    return z_star
end

## FC2D
#
# Along the reference profile, a parcel of conserved properties (ϑ, q) has the static energy
#     S(z) = h(P_*(z), ϑ, q) + g z,
# measured from the energy h(p_n, ϑ, q) + g z_n of the parcel at its own pressure and height, with the difference of h
# by quadrature,
#     S(z) - h(p_n, ϑ, q) - g z_n = ∫ v dp from p_n to P_*(z) + g (z - z_n),
# and its reference height z_* is the global minimiser of S over the box, 0 ≤ z ≤ L_z. At an interior minimum the
# residual of the level of neutral buoyancy (LNB)
#     r(z) = -dS/dz = g (ρ_* v(P_*(z), ϑ, q) - 1)                 (from ∂_z P_* = -g ρ_*)
# vanishes, and N_*² = d²S/dz² ≥ 0 there; r > 0 means the parcel is lighter than the reference state at z.

"""
    S = static_energy(ref::CompressibleLorenzReference, fluid, g, n, z)
    r = buoyancy_residual(ref::CompressibleLorenzReference, fluid, g, ϑ, q, z)
    N² = stratification(ref::CompressibleLorenzReference, fluid, g, ϑ, q, z)

Along the reference profile of `ref` at the height z [m]:
- the static energy of parcel n, measured from its energy at its own pressure p_n and height z_n:
  S(z) - h(p_n, ϑ_n, q_n) - g z_n = ∫ v dp from p_n to P_*(z) + g (z - z_n) [J kg⁻¹] (`enthalpy_difference`);
- for a parcel of conserved variable ϑ and composition q, the LNB residual r = -dS/dz = -(v ∂_z P_* + g) [m s⁻²], which
  is g (ρ_* v - 1);
- and N_*² = d²S/dz² = ∂v/∂p (∂_z P_*)² + v ∂²_z P_* [s⁻²].
"""
@inline function static_energy(ref::CompressibleLorenzReference, fluid, g, n, z)
    p = ref.parcel_pressure[n]
    P = hermite(ref.pressure, z)[1]
    return enthalpy_difference(fluid, p, P - p, ref.consvar[n], ref.q[n]) + g * (z - ref.height[n])
end
@inline function buoyancy_residual(ref::CompressibleLorenzReference, fluid, g, ϑ, q, z)
    P, ∂P, _ = hermite(ref.pressure, z)
    v = fluid(:p, :consvar, :q).specific_volume(P, ϑ, q)
    return -(v * ∂P + g)
end
@inline function stratification(ref::CompressibleLorenzReference, fluid, g, ϑ, q, z)
    P, ∂P, ∂²P = hermite(ref.pressure, z)
    v, ∂v_∂p, _, _ = fluid(:p, :consvar, :q).volume_functions(P, ϑ, q)
    return ∂v_∂p * ∂P^2 + v * ∂²P
end

"""
    z_star = reference_height(ref::CompressibleLorenzReference, fluid, g, Lz, n, ζ)

Reference height z_* [m] of parcel n, from the level ζ [m], among the walls and the interior slot edges, at which its
static energy is lowest: the root of the LNB residual next to it, to within sqrt(eps) L_z, if the static energy is
lower there (`refine_minimum`), or the parcel's own height z_n if the static energy is lower still there. With
S(z_*) ≤ S(z_n), the available potential energy (APE) Π₂ = S(z_n) - S(z_*) is nonnegative; at z_* = z_n it is zero, and
the available energy Π is the available compressible energy (ACE) Π₁ = h(p_n) - h(P_*(z_n)) - (p_n - P_*(z_n)) v_n ≥ 0
(h is concave in p). z_n is not among the levels, and where S is nearly flat (a well-mixed fluid) it can be the lowest
point found.
"""
function reference_height(ref::CompressibleLorenzReference, fluid, g, Lz, n, ζ)
    S(z) = static_energy(ref, fluid, g, n, z)
    r(z) = buoyancy_residual(ref, fluid, g, ref.consvar[n], ref.q[n], z)
    c = searchsortedfirst(ref.levels, ζ)                        # the index of the level
    z_star, S_star = refine_minimum(S, r, ref.levels, c, S(ζ), sqrt(eps(Lz)) * Lz)
    z_n = ref.height[n]
    return S(z_n) < S_star ? z_n : z_star
end

"""
    z_star = reference_heights!(z_star, model::FC2D, ref)

Reference height z_* [m] of the parcel of each cell, the minimiser of its static energy S along the profile of `ref`
over the box, on the cells (interior and zero-gradient halo), by the model's manager, in two passes:
1. the level of lowest S among the walls and the interior slot edges, with S measured from the parcel's own pressure
   and height, from the values of P_* there, vectorised over the parcels of a row;
2. the root of the LNB residual next to that level, to within sqrt(eps) L_z, if S is lower there (`refine_minimum`).
A local minimum of S next to another level is not refined: it can be lower than the one found only if the two differ
by less than the change of S over a slot.
"""
function reference_heights!(z_star, model::FC2D, ref::CompressibleLorenzReference)
    (; mgr, domain, fluid, space) = model
    (; consvar, q, parcel_pressure, height, levels, level_pressure) = ref
    g = space.g
    # 1. the level of lowest S, the lowest level among equal values, with
    # S(z_c) - h(p_n) - g z_n = ∫ v dp from p_n to P_*(z_c) + g (z_c - z_n)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ϑ_n, q_n, p_n, z_n = consvar[n], q[n], parcel_pressure[n], height[n]
            lowest = enthalpy_difference(fluid, p_n, level_pressure[1] - p_n, ϑ_n, q_n) + g * (levels[1] - z_n)
            level = zero(lowest) + levels[1]
            for c in 2:length(levels)
                S = enthalpy_difference(fluid, p_n, level_pressure[c] - p_n, ϑ_n, q_n) + g * (levels[c] - z_n)
                level = ifelse(S < lowest, levels[c], level)       # ifelse, not ? :, so that it is vectorised
                lowest = min(lowest, S)
            end
            z_star[i, j] = level
        end
    end
    # 2. the minimum next to that level
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        for j in jrange, i in irange
            n = parcel_index(domain, i, j)
            z_star[i, j] = reference_height(ref, fluid, g, space.Lz, n, z_star[i, j])
        end
    end
    fill_halo!(domain, z_star, ZeroGradient())
    return z_star
end

"""
    count = count_wall_minima(model, z_star)

The number of parcels whose reference height `z_star` [m] is at the bottom (z_* = 0) or top (z_* = L_z) wall.
"""
count_wall_minima(model, z_star) = count(z -> z == 0 || z == model.space.Lz, interior(model.domain, z_star))
