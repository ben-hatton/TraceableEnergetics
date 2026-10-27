## LORENZ REFERENCE STATE OF AN2D
#
# The slots of the Lorenz column are the model rows, k = 1..M_z from the top, of mass M_k = M_x ρ_R Δx Δz. A parcel n
# of the row o placed in slot k has the cost, measured from its own row,
#     c_nk = h(P_R(z_k), ϑ_n, q_n) - h(P_R(z_o), ϑ_n, q_n) + g (z_k - z_o)        [J kg⁻¹]
# with the difference of h as the integral of v over the pressure (`enthalpy_difference`), which keeps the precision of
# the float type (h itself is about 10⁶ J kg⁻¹). A constant per parcel changes neither the placement nor the slot duals,
# and the cost of a parcel in its own row is zero. The reduced reference pressure π_* = (P_* - P_R)/ρ_R, defined up to a
# constant, follows from the mean specific volume v_*,k of the parcels of each slot,
#     ∂_z π_* = g (ρ_R v_* - 1),   P_* = P_R + ρ_R π_*;
# the dual potentials of the slots, π_* = -ψ_k for the optimal placement, certify the placement.
# The reference height z_* of each parcel, the minimiser of its static energy along this profile, is computed here too
# (`reference_heights!`); the available energy of the parcels, in anelastic_available_energy.jl. Heights z are measured
# from the bottom of the box; masses are per unit length along y [kg m⁻¹].

"""
    ref = AnelasticLorenzReference(model::AN2D, method = SortedRearrangement())
    reference_state!(ref, model, state)

Lorenz reference state of the AN2D `model`, allocated once and computed in place for a `state` by `reference_state!`
with the `method` of rearrangement (an `AbstractRearrangement`). Its slots are the model rows, from the top.

After `reference_state!`, it holds:
- the placement of the parcels (`placement`);
- the slot duals ψ_k and the parcel duals u_n [J kg⁻¹];
- the mean specific volume v_*,k of the parcels of each slot [m³ kg⁻¹], and their mean compressibility
  b_k = Σ_n x_nk ∂v/∂p(P_R(z_k), ϑ_n, q_n)/M_k [m³ kg⁻¹ Pa⁻¹], with which v_* follows the parcels of the end slots
  between the end rows and the walls;
- the profiles P_R(z) and π_*(z) (`pressure`, `potential`);
- P_* [Pa] and v_* [m³ kg⁻¹] at the cell centres of the z axis (`p_star`, `v_star`);
- the smallest reduced cost of each parcel, min_k (c_nk - u_n - ψ_k) [J kg⁻¹] (`reduced_cost`), the slot where it is
  least (`best_slot`, as a float, so that the pass is vectorised), and the number of parcels for which it is
  negative (`certificate_violations`): zero if the placement is optimal;
- for `ExactRearrangement`, the number of pivots of the simplex and of passes over the pairs (`pivots`,
  `pricing_passes`), and the decrease of the cost Σ_nk x_nk c_nk from the sorted placement, by which the sort
  underestimates E_P - E_* [J m⁻¹] (`sorted_energy_excess`); no pivots, one pass and zero for `SortedRearrangement`.
"""
mutable struct AnelasticLorenzReference{F, M <: AbstractRearrangement, W}
    const method::M
    const workspace::W                  # the workspace of the method (`rearrangement_workspace`)
    # the parcels, one per interior cell, numbered by `parcel_index`
    const mass::Vector{F}               # m_n = ρ_R Δx Δz [kg m⁻¹]
    const consvar::Vector{F}            # ϑ_n
    const q::Vector{F}                  # q_n [kg/kg]
    const row::Vector{Int}              # slot k of the parcel's own row
    # the rearrangement
    const key::Vector{F}                # sort key of each parcel [m³ kg⁻¹]
    const order::Vector{Int}            # the parcels in the order of the placement
    const scratch::Vector{Int}          # scratch of the sort
    const placement::Placement{F}       # the masses x_nk
    const u::Vector{F}                  # parcel duals u_n [J kg⁻¹]
    const reduced_cost::Vector{F}       # min_k (c_nk - u_n - ψ_k) of each parcel [J kg⁻¹]
    const best_slot::Vector{F}          # the slot k of the least reduced cost of each parcel, as a float (vectorised)
    # the slots, the model rows from the top
    const slot_height::Vector{F}        # z_k, the row centre [m]
    const slot_pressure::Vector{F}      # P_R(z_k) [Pa]
    const slot_mass::Vector{F}          # M_k = M_x ρ_R Δx Δz [kg m⁻¹]
    const slot_dual::Vector{F}          # ψ_k [J kg⁻¹]
    const slot_volume::Vector{F}        # v_*,k = Σ_n x_nk v(P_R(z_k), ϑ_n, q_n)/M_k [m³ kg⁻¹]
    const slot_compressibility::Vector{F}   # b_k = Σ_n x_nk ∂v/∂p(P_R(z_k), ϑ_n, q_n)/M_k [m³ kg⁻¹ Pa⁻¹]
    # the profiles
    const pressure::HermiteProfile{F}   # P_R(z) [Pa], through the cell centres of the z axis
    const potential::HermiteProfile{F}  # π_*(z) [J kg⁻¹], through both walls and the row centres, from the bottom
    const node_pressure::Vector{F}      # P_R at the nodes of π_* [Pa]
    const p_star::Vector{F}             # P_* at the cell centres of the z axis [Pa]
    const v_star::Vector{F}             # v_* at the cell centres of the z axis [m³ kg⁻¹]
    # the tendency of the reference state from the diabatic tendencies (`reference_tendency!`)
    const consvar_rate::Vector{F}       # Dϑ/Dt of each parcel [consvar s⁻¹]
    const q_rate::Vector{F}             # Dq/Dt of each parcel [s⁻¹]
    const slot_means::SlotMeans{F}      # the N_*⁻²-weighted means of the parcels of each slot
    const slot_volume_tendency::Vector{F}   # ∂ₜv_*,k [m³ kg⁻¹ s⁻¹]
    const potential_tendency::HermiteProfile{F} # ∂ₜπ_*(z) [J kg⁻¹ s⁻¹], at the nodes of π_*
    const p_star_tendency::Vector{F}    # ∂ₜP_* at the cell centres of the z axis [Pa s⁻¹]
    certificate_violations::Int
    plain_weight_slots::Int             # slots whose means have plain weights, as N² ≤ 0 for one of their parcels
    pivots::Int                         # pivots of the transportation simplex
    pricing_passes::Int                 # passes over the parcel-slot pairs
    sorted_energy_excess::F             # Σ_nk x_nk c_nk of the sorted placement less that of the placement [J m⁻¹]
end

function AnelasticLorenzReference(model::AN2D{F}, method::AbstractRearrangement = SortedRearrangement()) where F
    (; domain, dx, dz, space, reference) = model
    (; ρ_R, p_R) = reference
    (; g) = space
    Mx, Mz = dims(domain)
    Hz = halo_size(domain)[2]
    N, K = Mx * Mz, Mz
    nz = ncells(domain, 2)
    vector() = zeros(F, N)
    ref = AnelasticLorenzReference(method, rearrangement_workspace(method, F, N, K), vector(), vector(), vector(), zeros(Int, N),
                                   vector(), collect(1:N), zeros(Int, N), Placement(F, N, K), vector(), vector(), vector(),
                                   zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K),
                                   HermiteProfile(F, nz), HermiteProfile(F, K + 2), zeros(F, K + 2), zeros(F, nz), zeros(F, nz),
                                   vector(), vector(), SlotMeans(F, K), zeros(F, K), HermiteProfile(F, K + 2), zeros(F, nz), 0, 0, 0, 0, zero(F))
    # P_R through the cell centres of the whole z axis, with the slopes ∂_z P_R = -g ρ_R
    for j in 1:nz
        ref.pressure.z[j] = cellpoint(domain, dz, 2, j)
        ref.pressure.y[j] = p_R[j]
        ref.pressure.d[j] = -g * ρ_R[j]
    end
    # the slots: slot k is the row of cell index j = Hz + Mz + 1 - k
    for k in 1:K
        j = Hz + Mz + 1 - k
        ref.slot_height[k] = cellpoint(domain, dz, 2, j)
        ref.slot_pressure[k] = p_R[j]
        ref.slot_mass[k] = Mx * ρ_R[j] * dx * dz
    end
    # the heights of the nodes of π_*: the bottom wall, the row centres from the bottom, the top wall
    ref.potential.z[1] = 0
    for k in 1:K
        ref.potential.z[K+2-k] = ref.slot_height[k]
    end
    ref.potential.z[K+2] = space.Lz
    ref.potential_tendency.z .= ref.potential.z
    for c in 1:K+2
        ref.node_pressure[c] = hermite(ref.pressure, ref.potential.z[c])[1]
    end
    # the parcels: their mass and their own row
    for j in interior_cells(domain, 2), i in interior_cells(domain, 1)
        n = parcel_index(domain, i, j)
        ref.mass[n] = ρ_R[j] * dx * dz
        ref.row[n] = Hz + Mz + 1 - j
    end
    return ref
end

"""
    c = anelastic_cost(model::AN2D, ref, n, k)

Energy per unit mass [J kg⁻¹] of parcel n placed in slot k, measured from its own row o = `ref.row[n]`:
c_nk = h(P_R(z_k), ϑ_n, q_n) - h(P_R(z_o), ϑ_n, q_n) + g (z_k - z_o), with the difference of h by quadrature
(`enthalpy_difference`).
"""
@inline function anelastic_cost(model::AN2D, ref::AnelasticLorenzReference, n, k)
    o = ref.row[n]
    P_own = ref.slot_pressure[o]
    Δh = enthalpy_difference(model.fluid, P_own, ref.slot_pressure[k] - P_own, ref.consvar[n], ref.q[n])
    return Δh + model.space.g * (ref.slot_height[k] - ref.slot_height[o])
end

"""
    ref = reference_state!(ref::AnelasticLorenzReference, model::AN2D, state)

Computes the Lorenz reference state `ref` of `state` in place: the parcels, their placement into the rows and its duals,
the profile π_*(z), P_* and v_* on the z axis, and the certificate of the placement.
"""
function reference_state!(ref::AnelasticLorenzReference{F}, model::AN2D, state) where F
    (; mgr, domain, fluid, space) = model
    (; ρ_R) = model.reference
    (; m) = state
    g = space.g
    K = length(ref.slot_mass)
    Hz = halo_size(domain)[2]

    # 1. The parcels: ϑ_n and q_n from the densities ρ_R ϑ and ρ_R q of the state
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ref.consvar[n] = m[i, j, 1] / ρ_R[j]
            ref.q[n] = m[i, j, 2] / ρ_R[j]
        end
    end

    # 2. The placement of the parcels into the slots, and its duals ψ_k and u_n, by the method of the reference state
    rearrange!(ref, model)

    # 3. The mean specific volume and compressibility of the parcels of each slot,
    # v_*,k = Σ_n x_nk v(P_R(z_k), ϑ_n, q_n)/M_k and b_k = Σ_n x_nk ∂v/∂p(P_R(z_k), ϑ_n, q_n)/M_k
    fill!(ref.slot_volume, 0)
    fill!(ref.slot_compressibility, 0)
    for e in 1:ref.placement.count
        n, k = ref.placement.parcel[e], ref.placement.slot[e]
        v, ∂v_∂p, _, _ = fluid(:p, :consvar, :q).volume_functions(ref.slot_pressure[k], ref.consvar[n], ref.q[n])
        ref.slot_volume[k] += ref.placement.mass[e] * v
        ref.slot_compressibility[k] += ref.placement.mass[e] * ∂v_∂p
    end
    ref.slot_volume ./= ref.slot_mass
    ref.slot_compressibility ./= ref.slot_mass

    # 4. The nodes of π_* at the row centres, from the slot volumes: the slopes ∂_z π_* = g (ρ_R v_*,k - 1), and the values
    # from ∂_z π_* = -v_* ∂_z P_R - g, with v_* linear in the pressure between successive rows (the trapezoidal rule in P_R),
    #     π_*(z_c) - π_*(z_c-1) = -(v_*,c-1 + v_*,c)/2 (P_R(z_c) - P_R(z_c-1)) - g (z_c - z_c-1),
    # exact when v is linear in p, as for a neutral fluid of NonlinearBinaryFluid, whose static energy is then flat at the
    # nodes. Node K + 2 - k is slot k, as the nodes go up from the bottom wall. The duals ψ_k give π_* = -ψ_k too (for the
    # optimal placement), but they change abruptly in time when the parcel split between two rows changes, and a parcel
    # split between two rows has the same static energy at both, so that its reference height jumps between them: they
    # serve the certificate only
    (; z, y, d) = ref.potential
    for k in 1:K
        j = Hz + K + 1 - k                                          # cell index of the row of slot k
        d[K+2-k] = g * (ρ_R[j] * ref.slot_volume[k] - 1)
    end
    y[2] = 0                                                        # the bottom row, before the constant is fixed
    for c in 3:K+1
        v₋, v₊ = ref.slot_volume[K+3-c], ref.slot_volume[K+2-c]    # v_* of the rows of nodes c - 1 and c
        y[c] = y[c-1] - (v₋ + v₊) / 2 * (ref.node_pressure[c] - ref.node_pressure[c-1]) - g * (z[c] - z[c-1])
    end
    # π_* is defined up to a constant, which changes neither the reference heights nor the available energy: its mean
    # over the mass of the rows, Σ_k M_k π_*(z_k) / Σ_k M_k, is zero
    mean_π = sum(k -> ref.slot_mass[k] * y[K+2-k], 1:K) / sum(ref.slot_mass)
    for c in 2:K+1
        y[c] -= mean_π
    end
    # between the end rows and the walls, v_* follows the compression of the parcels of the end slot e,
    # v_*(z) = v_e + b_e (P_R(z) - P_R(z_e)), so that a parcel of that slot keeps its static energy: a constant v_* would
    # lower it towards both walls by b_e (g ρ_R Δz/2)²/2, and make a wall the reference height of every parcel of a
    # neutral fluid. From ∂_z π_* = g (ρ_R v_* - 1) and ∂_z P_R = -g ρ_R, with ΔP = P_R(z) - P_R(z_e),
    #     π_*(z) = π_*(z_e) - v_e ΔP - b_e ΔP²/2 - g (z - z_e),   ∂_z π_*(z) = -(v_e + b_e ΔP) ∂_z P_R(z) - g.
    # Bottom wall, node 1, from the bottom row (node 2, slot K); top wall, node K + 2, from the top row (node K + 1, slot 1)
    for (wall, row, e) in ((1, 2, K), (K + 2, K + 1, 1))
        v_e, b_e = ref.slot_volume[e], ref.slot_compressibility[e]
        P_wall, ∂P_wall, _ = hermite(ref.pressure, z[wall])
        ΔP = P_wall - ref.slot_pressure[e]                          # P_R(wall) - P_R(z_e)
        y[wall] = y[row] - (v_e + b_e * ΔP / 2) * ΔP - g * (z[wall] - z[row])
        d[wall] = -(v_e + b_e * ΔP) * ∂P_wall - g
    end

    # 5. P_* = P_R + ρ_R π_* and v_* = (∂_z π_*/g + 1)/ρ_R at the cell centres of the z axis
    for j in eachindex(ref.p_star)
        π, ∂π, _ = hermite(ref.potential, ref.pressure.z[j])
        ref.p_star[j] = model.reference.p_R[j] + ρ_R[j] * π
        ref.v_star[j] = (∂π / g + 1) / ρ_R[j]
    end

    # 6. The certificate: the parcels whose smallest reduced cost c_nk - u_n - ψ_k over the slots, from the last pass of
    # the rearrangement (`reduced_costs!`), is negative beyond the round-off of the duals
    tolerance = certificate_tolerance(F, K, g, space.Lz)
    ref.certificate_violations = count(r -> r < -tolerance, ref.reduced_cost)
    return ref
end

"""
    ref = sort_parcels!(ref::AnelasticLorenzReference, model::AN2D)

The parcels, sorted by their specific volume v(P_R(L_z/2), ϑ_n, q_n) at mid-depth [m³ kg⁻¹], lightest first, fill the
rows from the top (`sorted_placement!`): the placement of `SortedRearrangement`, and the start of `ExactRearrangement`.
"""
function sort_parcels!(ref::AnelasticLorenzReference, model::AN2D)
    (; mgr, domain, fluid) = model
    (; key, consvar, q) = ref
    # the sort key, by the model's manager
    p_key = hermite(ref.pressure, model.space.Lz / 2)[1]
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            key[n] = fluid(:p, :consvar, :q).specific_volume(p_key, consvar[n], q[n])
        end
    end
    sortperm!(ref.order, ref.key; rev = true, scratch = ref.scratch)
    sorted_placement!(ref.placement, ref.order, ref.mass, ref.slot_mass)
    return ref
end

"""
    ref = rearrange!(ref::AnelasticLorenzReference{<:Any, SortedRearrangement}, model::AN2D)

The sorted rearrangement of AN2D (`sort_parcels!`); the duals ψ_k and u_n [J kg⁻¹] follow along the staircase of the
placement (`staircase_duals!`), and the reduced costs from one pass over the pairs (`reduced_costs!`).
"""
function rearrange!(ref::AnelasticLorenzReference{<:Any, SortedRearrangement}, model::AN2D)
    sort_parcels!(ref, model)
    staircase_duals!(ref.slot_dual, ref.u, ref.placement, ref.order, (n, k) -> anelastic_cost(model, ref, n, k))
    reduced_costs!(ref, model)
    ref.pivots, ref.pricing_passes, ref.sorted_energy_excess = 0, 1, 0
    return ref
end

"""
    ref = rearrange!(ref::AnelasticLorenzReference{<:Any, ExactRearrangement}, model::AN2D)

The exact rearrangement of AN2D: the least-cost placement by the transportation simplex (`optimal_placement!`), from the
sorted placement (`sort_parcels!`) completed into a tree (`complete_staircase!`), with the costs `anelastic_cost` and
the passes over the pairs `reduced_costs!`. The masses of the parcels and of the rows do not depend on the state's
motion, so every tree of the simplex is a feasible placement.
"""
function rearrange!(ref::AnelasticLorenzReference{F, ExactRearrangement}, model::AN2D) where F
    sort_parcels!(ref, model)
    complete_staircase!(ref.placement, ref.order)
    cost(n, k) = anelastic_cost(model, ref, n, k)
    price!() = reduced_costs!(ref, model)
    tolerance = certificate_tolerance(F, length(ref.slot_mass), model.space.g, model.space.Lz)
    ref.pricing_passes, ref.sorted_energy_excess =
        optimal_placement!(ref.placement, ref.workspace, ref.slot_dual, ref.u, ref.reduced_cost, ref.best_slot, cost, price!,
                           tolerance, ref.method.passes_per_slot * length(ref.slot_mass))
    ref.pivots = ref.workspace.pivots
    return ref
end

"""
    ref = reduced_costs!(ref::AnelasticLorenzReference, model::AN2D)

The smallest reduced cost c_nk - u_n - ψ_k of each parcel over the slots (`ref.reduced_cost`) [J kg⁻¹], and the slot
where it is least (`ref.best_slot`), with the duals `ref.u` and `ref.slot_dual` and the costs from the parcel's own row o,
c_nk = h(P_R(z_k), ϑ_n, q_n) - h(P_R(z_o), ϑ_n, q_n) + g (z_k - z_o), by the model's manager (vectorised over the
parcels of a row). All N K pairs: the certificate of the placement, and the pricing of the transportation simplex.
"""
function reduced_costs!(ref::AnelasticLorenzReference{F}, model::AN2D) where F
    (; mgr, domain, fluid, space) = model
    (; consvar, q, u, slot_pressure, slot_height, slot_dual, reduced_cost, best_slot) = ref
    g = space.g
    K = length(slot_pressure)
    Hz = halo_size(domain)[2]
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ϑ_n, q_n, u_n = consvar[n], q[n], u[n]
            o = Hz + K + 1 - j                                      # the slot of the parcel's own row
            P_own, z_own = slot_pressure[o], slot_height[o]
            reduced = enthalpy_difference(fluid, P_own, slot_pressure[1] - P_own, ϑ_n, q_n) + g * (slot_height[1] - z_own) - u_n - slot_dual[1]
            best = zero(reduced) + 1                                # a float, as `reduced`, so that it is vectorised
            for k in 2:K
                c = enthalpy_difference(fluid, P_own, slot_pressure[k] - P_own, ϑ_n, q_n) + g * (slot_height[k] - z_own)
                r = c - u_n - slot_dual[k]
                best = ifelse(r < reduced, F(k), best)              # ifelse, not ? :, so that it is vectorised
                reduced = min(reduced, r)
            end
            reduced_cost[n] = reduced
            best_slot[n] = best
        end
    end
    return ref
end

## Reference heights
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
