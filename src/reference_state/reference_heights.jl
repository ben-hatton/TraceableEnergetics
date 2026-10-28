## REFERENCE HEIGHTS OF THE PARCELS
#
# The reference height z_* of a parcel is the global minimiser over the box of its static energy S along the reference
# profile; at an interior minimum, its level of neutral buoyancy, the parcel has the specific volume of the reference
# state. Along the reference profile, a parcel of conserved properties (ϑ, q) has the static energy
#     S(z) = h(P(z), ϑ, q) + π(z) + g z,
# with P the pressure profile of `ref.pressure` and π a reduced potential of the reference pressure:
#     AN2D:  P = P_R, the reference pressure of the model, and π = π_* = (P_* - P_R)/ρ_R;
#     FC2D:  P = P_*, and π = 0.
# S is measured from its value at the parcel's own pressure p_n and height z_n (P_R(z_n) in AN2D), with the difference
# of h by quadrature,
#     S(z) - S(z_n) = ∫ v dp from p_n to P(z) + π(z) - π(z_n) + g (z - z_n),
# and z_* is the global minimiser of S over the box, 0 ≤ z ≤ L_z. At an interior minimum the residual of the level of
# neutral buoyancy (LNB)
#     r(z) = -dS/dz = -(v ∂_z P + g + ∂_z π),
# which is g ρ_R (v(P_R(z), ϑ, q) - v_*(z)) for AN2D (from ∂_z P_R = -g ρ_R, ∂_z π_* = g (ρ_R v_* - 1)) and
# g (ρ_* v(P_*(z), ϑ, q) - 1) for FC2D (from ∂_z P_* = -g ρ_*), vanishes, and N_*² = d²S/dz² ≥ 0 there; r > 0 means the
# parcel is lighter than the reference state at z. The search starts from the levels of the column: the walls and the
# row centres (the nodes of π_*) for AN2D, the walls and the interior slot edges for FC2D.

"""
    levels = level_heights(ref)
    (π, ∂_z π, ∂²_z π) = reduced_potential(ref, z)
    π = level_potential(ref, c)
    π = row_potential(ref, domain, j)
    π = own_potential(ref, n)

The levels of the search for reference heights [m], from the bottom: the walls and the row centres for AN2D, the walls
and the interior slot edges for FC2D; the reduced potential π [J kg⁻¹] of the static energy, π_* for AN2D and 0 for
FC2D, with its derivatives at the height z [m], at the level c, at the row of cell index j and at the own row of parcel
n.
"""
level_heights(ref::AnelasticLorenzReference) = ref.potential.z
level_heights(ref::CompressibleLorenzReference) = ref.levels
@inline reduced_potential(ref::AnelasticLorenzReference, z) = hermite(ref.potential, z)
@inline reduced_potential(ref::CompressibleLorenzReference, z) = (zero(z), zero(z), zero(z))
@inline level_potential(ref::AnelasticLorenzReference, c) = ref.potential.y[c]
@inline level_potential(ref::CompressibleLorenzReference{F}, c) where F = zero(F)
@inline row_potential(ref::AnelasticLorenzReference, domain, j) = ref.potential.y[j-halo_size(domain)[2]+1]
@inline row_potential(ref::CompressibleLorenzReference{F}, domain, j) where F = zero(F)
@inline own_potential(ref::AnelasticLorenzReference, n) = ref.potential.y[length(ref.potential.z)-ref.parcels.row[n]]
@inline own_potential(ref::CompressibleLorenzReference{F}, n) where F = zero(F)

"""
    S = static_energy(ref, fluid, g, n, z)
    r = buoyancy_residual(ref, fluid, g, ϑ, q, z)
    N² = stratification(ref, fluid, g, ϑ, q, z)

Along the reference profile of `ref` at the height z [m]:
- the static energy of parcel n, measured from its value at the parcel's own pressure p_n and height z_n,
  S(z) - S(z_n) = ∫ v dp from p_n to P(z) + π(z) - π(z_n) + g (z - z_n) [J kg⁻¹] (`enthalpy_difference`);
- for a parcel of conserved variable ϑ and composition q, the LNB residual r = -dS/dz = -(v ∂_z P + g + ∂_z π) [m s⁻²];
- and N_*² = d²S/dz² = ∂v/∂p (∂_z P)² + v ∂²_z P + ∂²_z π [s⁻²],
with P = P_R, π = π_* for AN2D and P = P_*, π = 0 for FC2D.
"""
@inline function static_energy(ref::AbstractLorenzReference, fluid, g, n, z)
    (; pressure, height, consvar, q) = ref.parcels
    p_n = pressure[n]
    P = hermite(ref.pressure, z)[1]
    π = reduced_potential(ref, z)[1]
    return enthalpy_difference(fluid, p_n, P - p_n, consvar[n], q[n]) + (π - own_potential(ref, n)) + g * (z - height[n])
end
@inline function buoyancy_residual(ref::AbstractLorenzReference, fluid, g, ϑ, q, z)
    P, ∂P, _ = hermite(ref.pressure, z)
    ∂π = reduced_potential(ref, z)[2]
    v = fluid(:p, :consvar, :q).specific_volume(P, ϑ, q)
    return -(v * ∂P + g + ∂π)
end
@inline function stratification(ref::AbstractLorenzReference, fluid, g, ϑ, q, z)
    P, ∂P, ∂²P = hermite(ref.pressure, z)
    ∂²π = reduced_potential(ref, z)[3]
    v, ∂v_∂p, _, _ = fluid(:p, :consvar, :q).volume_functions(P, ϑ, q)
    return ∂v_∂p * ∂P^2 + v * ∂²P + ∂²π
end

"""
    z_star = reference_height(ref, fluid, g, Lz, n, ζ)

Reference height z_* [m] of parcel n, from the level ζ [m] at which its static energy is lowest (`level_heights`): the
root of the LNB residual next to it, to within sqrt(eps) L_z, if the static energy is lower there (`refine_minimum`), or
the parcel's own height z_n if the static energy is lower still there. With S(z_*) ≤ S(z_n), the available potential
energy (APE) Π₂ = S(z_n) - S(z_*) is nonnegative; at z_* = z_n it is zero, and the available energy Π is the available
compressible energy (ACE) Π₁ ≥ 0. In FC2D, z_n is not among the levels, and where S is nearly flat (a well-mixed fluid)
it can be the lowest point found; in AN2D it is a level.
"""
function reference_height(ref::AbstractLorenzReference, fluid, g, Lz, n, ζ)
    S(z) = static_energy(ref, fluid, g, n, z)
    r(z) = buoyancy_residual(ref, fluid, g, ref.parcels.consvar[n], ref.parcels.q[n], z)
    levels = level_heights(ref)
    c = searchsortedfirst(levels, ζ)                            # the index of the level
    z_star, S_star = refine_minimum(S, r, levels, c, S(ζ), sqrt(eps(Lz)) * Lz)
    z_n = ref.parcels.height[n]
    return S(z_n) < S_star ? z_n : z_star
end

"""
    z_star = reference_heights!(z_star, model, ref)

Reference height z_* [m] of the parcel of each cell, the minimiser of its static energy S along the profile of `ref`
over the box, on the cells (interior and zero-gradient halo), by the model's manager, in two passes:
1. the level of lowest S among the levels of the column (`level_heights`), with S measured from the parcel's own pressure
   and height, from the pressures `ref.level_pressure` and the reduced potential there, vectorised over the parcels of a
   row;
2. the root of the LNB residual next to that level, to within sqrt(eps) L_z, if S is lower there (`reference_height`).
A local minimum of S next to another level is not refined: it can be lower than the one found only if the two differ
by less than the change of S between two levels.
"""
function reference_heights!(z_star, model, ref::AbstractLorenzReference)
    (; mgr, domain, fluid, space) = model
    (; consvar, q, pressure, height) = ref.parcels
    levels, level_pressure = level_heights(ref), ref.level_pressure
    g = space.g
    # 1. the level of lowest S, the lowest level among equal values, with
    # S(z_c) - S(z_n) = ∫ v dp from p_n to P(z_c) + π(z_c) - π(z_n) + g (z_c - z_n)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ϑ_n, q_n, p_n, z_n = consvar[n], q[n], pressure[n], height[n]
            π_n = row_potential(ref, domain, j)
            lowest = enthalpy_difference(fluid, p_n, level_pressure[1] - p_n, ϑ_n, q_n) + (level_potential(ref, 1) - π_n) +
                     g * (levels[1] - z_n)
            level = zero(lowest) + levels[1]
            for c in 2:length(levels)
                S = enthalpy_difference(fluid, p_n, level_pressure[c] - p_n, ϑ_n, q_n) + (level_potential(ref, c) - π_n) +
                    g * (levels[c] - z_n)
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
