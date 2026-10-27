## THE LORENZ REFERENCE STATE
#
# The Lorenz reference state places the parcels of a snapshot (the interior cells, each keeping its mass m_n and its
# conserved ϑ_n, q_n) into a column of K slots of masses M_k, numbered from the top, so that the potential energy
# Σ_nk x_nk c_nk is least, with x_nk the mass of parcel n in slot k and c_nk its energy per unit mass there [J kg⁻¹]. Here
# are the methods of the placement, the reference state of each model and the steps of `reference_state!`; the placement
# itself is in rearrangement.jl, the reference heights of the parcels in reference_heights.jl, their available energy in
# available_energy.jl, the tendency of the reference state in tendency.jl.

"""
    AbstractRearrangement

Supertype of the methods that compute the placement of the parcels into the slots of the Lorenz reference column, with
the dual potentials ψ_k of the slots and u_n of the parcels [J kg⁻¹]. A method `M <: AbstractRearrangement` is the
`method` of an `AnelasticLorenzReference` or a `CompressibleLorenzReference`, and provides a method of `rearrange!`
for it.
"""
abstract type AbstractRearrangement end

"""
    SortedRearrangement()

Method of the Lorenz reference state (`AbstractRearrangement`): sorts the parcels by their specific volume at a key
pressure, lightest first, and fills the slots from the top in that order (`sorted_placement!`). It is exact when
v(p, ϑ, q) = f(ϑ, q) + w(p) (no thermobaricity). Otherwise the reduced costs c_nk - u_n - ψ_k of the duals along the
staircase (`staircase_duals!`), nonnegative for an optimal placement, report whether it is optimal.
"""
struct SortedRearrangement <: AbstractRearrangement end

"""
    ExactRearrangement(; passes_per_slot = 10, max_top_iterations = 20)

Method of the Lorenz reference state (`AbstractRearrangement`): the least-cost placement for any equation of state, by
the transportation simplex (`optimal_placement!`), from the sorted placement of `SortedRearrangement`. Its tree is the
staircase of the sorted placement completed by entries of zero mass (`complete_staircase!`); the pairs that enter it are
found by passes over all the parcel-slot pairs, the last of which is the certificate of the placement. The simplex ends
(its tree is kept strongly feasible), after a number of passes that grows with how far the sort is from the optimum:
a few where the sort fails for a few parcels, about K (1.0 to 1.3 K for K = 20 to 80) where every parcel is misplaced
(two density-compensated families mixed in every row): each pass moves a misplaced parcel about one slot nearer its
level. The passes per placement are at most `passes_per_slot` K, which only guards against round-off, and scales with
the resolution as the passes do; a placement cut short by it fails its certificate. In FC2D, the costs depend on the top
pressure P_top, which depends on the placement: the placement at fixed P_top and P_top at fixed placement
(`top_pressure!`) alternate, from the sorted placement, until the placement no longer changes, at most
`max_top_iterations` times.
"""
Base.@kwdef struct ExactRearrangement <: AbstractRearrangement
    passes_per_slot::Int = 10
    max_top_iterations::Int = 20
end

"""
    workspace = rearrangement_workspace(method, F, N, K)

The workspace of the rearrangement `method` for N parcels and K slots, allocated once with the reference state:
`nothing` for a method without one (`SortedRearrangement`), the tree of the transportation simplex
(`TransportationTree`) for `ExactRearrangement`.
"""
rearrangement_workspace(::AbstractRearrangement, F, N, K) = nothing
rearrangement_workspace(::ExactRearrangement, F, N, K) = TransportationTree(F, N, K)

"""
    tolerance = certificate_tolerance(F, K, g, Lz)

The tolerance [J kg⁻¹] below which a negative reduced cost c_nk - u_n - ψ_k is round-off, for K slots in a box of height
L_z [m] under gravity g [m s⁻²]: the duals accumulate K differences of costs of size up to g L_z, 16 (K + 1) eps g L_z.
"""
certificate_tolerance(F, K, g, Lz) = 16 * (K + 1) * eps(F) * g * Lz

"""
    n = parcel_index(domain, i, j)

Index n of the parcel of the interior cell (i, j): n = (j - Hz - 1) M_x + (i - Hx), row by row from the bottom.
"""
@inline function parcel_index(domain, i, j)
    Hx, Hz = halo_size(domain)
    Mx = dims(domain)[1]
    return (j - Hz - 1) * Mx + (i - Hx)
end

"""
    means = SlotMeans(F, K)

Workspace of the means over the parcels of each of K slots that give the tendency of the reference state
(`slot_means!`): the slot density ρ_k [kg m⁻³] and the gradient ∂_z v_*(z_k) [m² kg⁻¹] of the reference specific
volume at the slot, set before the means, and the means ⟨v̇⟩_k [m³ kg⁻¹ s⁻¹] and ⟨∂v/∂p⟩_k [m³ kg⁻¹ Pa⁻¹], with the
sums of their weights and whether the slot holds a parcel with N² ≤ 0.
"""
struct SlotMeans{F}
    density::Vector{F}              # ρ_k [kg m⁻³]
    volume_gradient::Vector{F}      # ∂_z v_*(z_k) [m² kg⁻¹]
    weight::Vector{F}               # Σ_n x_nk/N²_nk [kg m⁻¹ s²]
    volume_rate::Vector{F}          # ⟨v̇⟩_k [m³ kg⁻¹ s⁻¹]
    compressibility::Vector{F}      # ⟨∂v/∂p⟩_k [m³ kg⁻¹ Pa⁻¹]
    plain_volume_rate::Vector{F}    # Σ_n x_nk v̇_nk [m⁴ s⁻¹], then the plain mean
    plain_compressibility::Vector{F}    # Σ_n x_nk ∂v/∂p_nk, then the plain mean
    unstable::Vector{Bool}          # N²_nk ≤ 0 for a parcel of the slot
end
SlotMeans(F::Type, K) = SlotMeans(zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(Bool, K))

## AN2D
#
# The slots of the Lorenz column are the model rows, k = 1..M_z from the top, of mass M_k = M_x ρ_R Δx Δz. A parcel n
# of the row o placed in slot k has the cost, measured from its own row,
#     c_nk = h(P_R(z_k), ϑ_n, q_n) - h(P_R(z_o), ϑ_n, q_n) + g (z_k - z_o)        [J kg⁻¹]
# with the difference of h as the integral of v over the pressure (`enthalpy_difference`), which keeps the precision of
# the float type (h itself is about 10⁶ J kg⁻¹). A constant per parcel changes neither the placement nor the slot duals,
# and the cost of a parcel in its own row is zero. The reduced reference pressure π_* = (P_* - P_R)/ρ_R, defined up to a
# constant, follows from the mean specific volume v_*,k of the parcels of each slot,
#     ∂_z π_* = g (ρ_R v_* - 1),   P_* = P_R + ρ_R π_*;
# the dual potentials of the slots, π_* = -ψ_k for the optimal placement, certify the placement. Heights z are measured
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

## FC2D
#
# The slots of the Lorenz column are the model rows, k = 1..M_z from the top, each of the mass M_k of its row in the
# state (the sum of the masses of its parcels), so that a state at rest is its own reference state. The column is
# hydrostatic: slot k is at the pressure
#     P_k = P_top + g (Σ_{l<k} M_l + M_k/2)/L_x                        [Pa]
# and the top pressure P_top makes the column fill the box, Σ_nk x_nk v(P_k, ϑ_n, q_n) = L_x L_z. A parcel n placed in
# slot k has the cost, measured from its own pressure p_n,
#     c_nk = h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n)                       [J kg⁻¹]
# with the difference of h as the integral of v over the pressure (`enthalpy_difference`), which keeps the precision of
# the float type (h itself is about 10⁶ J kg⁻¹). A constant per parcel changes neither the placement nor the slot
# duals, and the geopotential of the column, Σ_k M_k g z_k, does not depend on the placement once P_top is fixed. Slot k
# has the thickness V_k/L_x, with V_k = Σ_n x_nk v(P_k, ϑ_n, q_n) its volume, stacked from the bottom, and the reference
# pressure P_*(z) is the monotone cubic through the pressures of the slot edges, P_top + g/L_x times the mass above the
# edge. Heights z are measured from the bottom of the box; masses are per unit length along y [kg m⁻¹].

"""
    ref = CompressibleLorenzReference(model::FC2D, method = SortedRearrangement())
    reference_state!(ref, model, state)

Lorenz reference state of the FC2D `model`, allocated once and computed in place for a `state` by `reference_state!`
with the `method` of rearrangement (an `AbstractRearrangement`). Its slots are the model rows, from the top, each of the
mass of its row in the state.

After `reference_state!`, it holds:
- the parcels, with their own pressure p_n [Pa] and specific volume v_n = 1/ρ_n [m³ kg⁻¹];
- the placement of the parcels (`placement`);
- the slot duals ψ_k and the parcel duals u_n [J kg⁻¹];
- the pressure P_k [Pa], the mean specific volume v_*,k [m³ kg⁻¹] and the height of the centre [m] of each slot;
- the top pressure `p_top` [Pa] of the column, and its `volume_residual`, (Σ_k V_k/L_x - L_z)/L_z;
- the profile P_*(z) (`pressure`);
- P_* [Pa] and v_* = -g/∂_z P_* [m³ kg⁻¹] at the cell centres of the z axis (`p_star`, `v_star`);
- the smallest reduced cost of each parcel, min_k (c_nk - u_n - ψ_k) [J kg⁻¹] (`reduced_cost`), the slot where it is
  least (`best_slot`, as a float, so that the pass is vectorised), and the number of parcels for which it is
  negative (`certificate_violations`): zero if the placement is optimal for this P_top;
- for `ExactRearrangement`, the number of pivots of the simplex and of passes over the pairs (`pivots`,
  `pricing_passes`), of placements alternating with P_top (`top_pressure_iterations`), and the decrease from the sorted
  placement of the energy of the column G = Σ_nk x_nk c_nk - P_top L_x L_z, by which the sort underestimates
  E_P - E_* [J m⁻¹] (`sorted_energy_excess`); no pivots, one pass, one iteration and zero for `SortedRearrangement`.
"""
mutable struct CompressibleLorenzReference{F, M <: AbstractRearrangement, W}
    const method::M
    const workspace::W                  # the workspace of the method (`rearrangement_workspace`)
    # the parcels, one per interior cell, numbered by `parcel_index`
    const mass::Vector{F}               # m_n = ρ Δx Δz [kg m⁻¹]
    const consvar::Vector{F}            # ϑ_n
    const q::Vector{F}                  # q_n [kg/kg]
    const parcel_pressure::Vector{F}    # p_n = p(v_n, ϑ_n, q_n) from the equation of state [Pa]
    const parcel_volume::Vector{F}      # v_n = 1/ρ [m³ kg⁻¹]
    const height::Vector{F}             # z_n, the centre of the parcel's row [m]
    const row::Vector{Int}              # slot k of the parcel's own row
    # the rearrangement
    const key::Vector{F}                # sort key of each parcel [m³ kg⁻¹]
    const order::Vector{Int}            # the parcels in the order of the placement
    const scratch::Vector{Int}          # scratch of the sort
    const placement::Placement{F}       # the masses x_nk
    const u::Vector{F}                  # parcel duals u_n [J kg⁻¹]
    const reduced_cost::Vector{F}       # min_k (c_nk - u_n - ψ_k) of each parcel [J kg⁻¹]
    const best_slot::Vector{F}          # the slot k of the least reduced cost of each parcel, as a float (vectorised)
    # the slots, from the top
    const slot_mass::Vector{F}          # M_k, the mass of the row [kg m⁻¹]
    const slot_pressure::Vector{F}      # P_k = P_top + g (Σ_{l<k} M_l + M_k/2)/L_x [Pa]
    const slot_dual::Vector{F}          # ψ_k [J kg⁻¹]
    const slot_volume::Vector{F}        # v_*,k = Σ_n x_nk v(P_k, ϑ_n, q_n)/M_k [m³ kg⁻¹]
    const slot_height::Vector{F}        # the height of the centre of slot k [m]
    # the profiles
    const pressure::HermiteProfile{F}   # P_*(z) [Pa], through the K + 1 slot edges, from the bottom
    const secant::Vector{F}             # secant slopes of P_* between the edges, -g/v_*,k [Pa m⁻¹]
    const levels::Vector{F}             # the walls and the interior slot edges, from the bottom [m]
    const level_pressure::Vector{F}     # P_* at the levels [Pa]
    const p_star::Vector{F}             # P_* at the cell centres of the z axis [Pa]
    const v_star::Vector{F}             # v_* at the cell centres of the z axis [m³ kg⁻¹]
    # the tendency of the reference state from the diabatic tendencies (`reference_tendency!`)
    const consvar_rate::Vector{F}       # Dϑ/Dt of each parcel [consvar s⁻¹]
    const q_rate::Vector{F}             # Dq/Dt of each parcel [s⁻¹]
    const slot_means::SlotMeans{F}      # the N_*⁻²-weighted means of the parcels of each slot
    const slot_volume_tendency::Vector{F}   # ∂ₜv_*,k at fixed mass coordinate [m³ kg⁻¹ s⁻¹]
    const edge_height_tendency::Vector{F}   # ∂ₜz_e of the slot edges, from the bottom [m s⁻¹]
    const dual_pressure::HermiteProfile{Dual{ReferenceTendencyTag, F, 1}}  # P_*(z), with ∂ₜP_* at fixed z as the partial
    const dual_secant::Vector{Dual{ReferenceTendencyTag, F, 1}}            # the secants of P_*, with their tendencies
    const p_star_tendency::Vector{F}    # ∂ₜP_* at the cell centres of the z axis [Pa s⁻¹]
    p_top::F                            # P_top [Pa]
    volume_residual::F                  # (Σ_k V_k/L_x - L_z)/L_z
    p_top_tendency::F                   # ∂ₜP_top [Pa s⁻¹]
    certificate_violations::Int
    plain_weight_slots::Int             # slots whose means have plain weights, as N² ≤ 0 for one of their parcels
    pivots::Int                         # pivots of the transportation simplex
    pricing_passes::Int                 # passes over the parcel-slot pairs
    top_pressure_iterations::Int        # placements alternating with P_top
    sorted_energy_excess::F             # G = Σ_nk x_nk c_nk - P_top L_x L_z of the sorted placement less that of the placement [J m⁻¹]
end

function CompressibleLorenzReference(model::FC2D{F}, method::AbstractRearrangement = SortedRearrangement()) where F
    (; domain, dz) = model
    Mx, Mz = dims(domain)
    Hz = halo_size(domain)[2]
    N, K = Mx * Mz, Mz
    nz = ncells(domain, 2)
    vector() = zeros(F, N)
    ref = CompressibleLorenzReference(method, rearrangement_workspace(method, F, N, K), vector(), vector(), vector(), vector(),
                                      vector(), vector(), zeros(Int, N),
                                      vector(), collect(1:N), zeros(Int, N), Placement(F, N, K), vector(), vector(), vector(),
                                      zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K),
                                      HermiteProfile(F, K + 1), zeros(F, K), zeros(F, K + 1), zeros(F, K + 1),
                                      zeros(F, nz), zeros(F, nz), vector(), vector(), SlotMeans(F, K), zeros(F, K),
                                      zeros(F, K + 1), HermiteProfile(Dual{ReferenceTendencyTag, F, 1}, K + 1),
                                      zeros(Dual{ReferenceTendencyTag, F, 1}, K), zeros(F, nz), zero(F), zero(F), zero(F), 0, 0, 0, 0, 0, zero(F))
    # the parcels' own heights and rows: slot k is the row of cell index j = Hz + Mz + 1 - k
    for j in interior_cells(domain, 2), i in interior_cells(domain, 1)
        n = parcel_index(domain, i, j)
        ref.height[n] = cellpoint(domain, dz, 2, j)
        ref.row[n] = Hz + Mz + 1 - j
    end
    return ref
end

"""
    c = compressible_cost(model::FC2D, ref, n, k)

Energy per unit mass [J kg⁻¹] of parcel n placed in slot k, measured from its own pressure p_n:
c_nk = h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n), by quadrature (`enthalpy_difference`).
"""
@inline function compressible_cost(model::FC2D, ref::CompressibleLorenzReference, n, k)
    p = ref.parcel_pressure[n]
    return enthalpy_difference(model.fluid, p, ref.slot_pressure[k] - p, ref.consvar[n], ref.q[n])
end

"""
    ref = reference_state!(ref::CompressibleLorenzReference, model::FC2D, state)

Computes the Lorenz reference state `ref` of `state` in place: the parcels, their placement into the slots, the top
pressure of the column and the duals of the placement, the slot volumes and heights, the profile P_*(z), P_* and v_* on
the z axis, and the certificate of the placement.
"""
function reference_state!(ref::CompressibleLorenzReference{F}, model::FC2D, state) where F
    (; mgr, domain, fluid, space, dx, dz) = model
    (; g, Lx, Lz) = space
    (; m) = state
    K = length(ref.slot_mass)

    # 1. The parcels: ρ, ϑ_n and q_n from the densities ρ, ρ ϑ and ρ q of the state, m_n = ρ Δx Δz, v_n = 1/ρ, and their
    # pressure p_n = p(v_n, ϑ_n, q_n) from the equation of state
    (; mass, consvar, q, parcel_volume, parcel_pressure) = ref
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ρ = m[i, j, 1]
            v = inv(ρ)
            ϑ_n = m[i, j, 2] * v
            q_n = m[i, j, 3] * v
            mass[n] = ρ * dx * dz
            consvar[n] = ϑ_n
            q[n] = q_n
            parcel_volume[n] = v
            parcel_pressure[n] = fluid(:v, :consvar, :q).pressure(v, ϑ_n, q_n)
        end
    end
    # the mass M_k of each slot, the sum of the masses of the parcels of its row
    fill!(ref.slot_mass, 0)
    for n in eachindex(mass)
        ref.slot_mass[ref.row[n]] += mass[n]
    end

    # 2. The placement of the parcels into the slots, the top pressure P_top and the slot pressures P_k, and the duals
    # ψ_k and u_n, by the method of the reference state
    rearrange!(ref, model)

    # 3. The mean specific volume of the parcels of each slot, v_*,k = Σ_n x_nk v(P_k, ϑ_n, q_n)/M_k
    fill!(ref.slot_volume, 0)
    for e in 1:ref.placement.count
        n, k = ref.placement.parcel[e], ref.placement.slot[e]
        v = fluid(:p, :consvar, :q).specific_volume(ref.slot_pressure[k], consvar[n], q[n])
        ref.slot_volume[k] += ref.placement.mass[e] * v
    end
    ref.slot_volume ./= ref.slot_mass

    # 4. The slots stacked from the bottom, each of thickness V_k/L_x = v_*,k M_k/L_x, and P_* at their edges: edge 1 is
    # the bottom, edge K + 1 the top, and edge e is below slot K + 1 - e, so P_*(z_e) = P_top + g (Σ_{k ≤ K+1-e} M_k)/L_x,
    # from the top down. Between the edges, P_* is the monotone cubic of secant slopes -g/v_*,k
    (; z, y, d) = ref.pressure
    z[1] = 0
    for e in 1:K
        k = K + 1 - e                                               # the slot above edge e
        z[e+1] = z[e] + ref.slot_volume[k] * ref.slot_mass[k] / Lx
        ref.slot_height[k] = (z[e] + z[e+1]) / 2
        ref.secant[e] = -g / ref.slot_volume[k]
    end
    y[K+1] = ref.p_top
    for e in K:-1:1
        y[e] = y[e+1] + g * ref.slot_mass[K+1-e] / Lx              # the weight of the slot above edge e
    end
    monotone_slopes!(d, z, ref.secant)
    # the column fills the box up to the round-off of P_top
    ref.volume_residual = (z[K+1] - Lz) / Lz
    # the levels of the search for reference heights: the walls, exactly, and the interior edges, with P_* there
    ref.levels[1] = 0
    for e in 2:K
        ref.levels[e] = z[e]
    end
    ref.levels[K+1] = Lz
    for c in 1:K+1
        ref.level_pressure[c] = hermite(ref.pressure, ref.levels[c])[1]
    end

    # 5. P_* and v_* = 1/ρ_* = -g/∂_z P_* at the cell centres of the z axis
    for j in eachindex(ref.p_star)
        P, ∂P, _ = hermite(ref.pressure, cellpoint(domain, dz, 2, j))
        ref.p_star[j] = P
        ref.v_star[j] = -g / ∂P
    end

    # 6. The certificate: the parcels whose smallest reduced cost c_nk - u_n - ψ_k over the slots, from the last pass of
    # the rearrangement (`reduced_costs!`), is negative beyond the round-off of the duals
    tolerance = certificate_tolerance(F, K, g, Lz)
    ref.certificate_violations = count(r -> r < -tolerance, ref.reduced_cost)
    return ref
end
