## THE LORENZ REFERENCE STATE
#
# The Lorenz reference state places the parcels of a snapshot (the interior cells, each keeping its mass m_n and its
# conserved ϑ_n, q_n) into a column of K slots of masses M_k, numbered from the top, so that the potential energy
# Σ_nk x_nk c_nk is least, with x_nk the mass of parcel n in slot k and c_nk its energy per unit mass there [J kg⁻¹]. Here
# are the methods of the placement, the parts shared by the reference states of both models (parcels, slots, placement),
# the reference state of each model and the steps of `reference_state!`; the placement itself is in rearrangement.jl, the
# reference heights of the parcels in reference_heights.jl, their available energy in available_energy.jl, the tendency
# of the reference state in tendency.jl.

"""
    AbstractRearrangement

Supertype of the methods that compute the placement of the parcels into the slots of the Lorenz reference column, with
the dual potentials ψ_k of the slots and u_n of the parcels [J kg⁻¹]. A method `M <: AbstractRearrangement` is the
`method` of the `ColumnPlacement` of an `AnelasticLorenzReference` or a `CompressibleLorenzReference`, and provides a
method of `rearrange!` for it.
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

## The parts shared by the reference states of both models

"""
    AbstractLorenzReference{F, M}

Supertype of the Lorenz reference states, `AnelasticLorenzReference` and `CompressibleLorenzReference`, in the float
type F and with the method of rearrangement `M <: AbstractRearrangement`. Each holds its parcels (`parcels`, a
`Parcels`), its slots (`slots`, a `Slots`), the placement of the parcels into the slots (`column`, a `ColumnPlacement`),
and the profiles of the reference state of its model, of which `pressure` is a `HermiteProfile` of a pressure through
the levels of the column: P_R(z) for AN2D, P_*(z) for FC2D.
"""
abstract type AbstractLorenzReference{F, M <: AbstractRearrangement} end

"""
    parcels = Parcels(F, N)

The N parcels of a snapshot, one per interior cell, numbered by `parcel_index`: each keeps its mass m_n [kg m⁻¹], its
conserved variable ϑ_n and composition q_n [kg/kg], and has the pressure p_n [Pa] of its own thermodynamics, P_R(z_n) for
AN2D and p(v_n, ϑ_n, q_n) for FC2D, the specific volume v_n = v(p_n, ϑ_n, q_n) [m³ kg⁻¹] there, and the height z_n [m]
of the centre of its own row, the slot `row` of the column. `consvar_rate` and `q_rate` are its diabatic rates Dϑ/Dt
[consvar s⁻¹] and Dq/Dt [s⁻¹], for the tendency of the reference state.
"""
struct Parcels{F}
    mass::Vector{F}             # m_n [kg m⁻¹]
    consvar::Vector{F}          # ϑ_n
    q::Vector{F}                # q_n [kg/kg]
    pressure::Vector{F}         # p_n [Pa]
    volume::Vector{F}           # v_n = v(p_n, ϑ_n, q_n) [m³ kg⁻¹]
    height::Vector{F}           # z_n, the centre of the parcel's row [m]
    row::Vector{Int}            # slot k of the parcel's own row
    consvar_rate::Vector{F}     # Dϑ/Dt [consvar s⁻¹]
    q_rate::Vector{F}           # Dq/Dt [s⁻¹]
end
Parcels(F::Type, N) = Parcels(zeros(F, N), zeros(F, N), zeros(F, N), zeros(F, N), zeros(F, N), zeros(F, N), zeros(Int, N),
                              zeros(F, N), zeros(F, N))

"""
    slots = Slots(F, K)

The K slots of the Lorenz column, numbered from the top: the mass M_k [kg m⁻¹], the height of the centre z_k [m] and the
pressure P_k [Pa] of each, the mean specific volume v_*,k = Σ_n x_nk v(P_k, ϑ_n, q_n)/M_k [m³ kg⁻¹] and compressibility
b_k = Σ_n x_nk ∂v/∂p(P_k, ϑ_n, q_n)/M_k [m³ kg⁻¹ Pa⁻¹] of their parcels (`slot_volumes!`), and the tendency ∂ₜv_*,k
[m³ kg⁻¹ s⁻¹] from the diabatic tendencies.
"""
struct Slots{F}
    mass::Vector{F}             # M_k [kg m⁻¹]
    height::Vector{F}           # z_k [m]
    pressure::Vector{F}         # P_k [Pa]
    volume::Vector{F}           # v_*,k [m³ kg⁻¹]
    compressibility::Vector{F}  # b_k [m³ kg⁻¹ Pa⁻¹]
    volume_tendency::Vector{F}  # ∂ₜv_*,k [m³ kg⁻¹ s⁻¹]
end
Slots(F::Type, K) = Slots(zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K))

"""
    column = ColumnPlacement(method, F, N, K)

The placement of N parcels into the K slots of the Lorenz column by the rearrangement `method` (an
`AbstractRearrangement`), with its workspace (`rearrangement_workspace`). After `reference_state!`, it holds:
- the sort key of each parcel [m³ kg⁻¹] and the order of the sorted placement (`key`, `order`, with the `scratch` of
  the sort);
- the placement of the parcels (`placement`, the masses x_nk);
- the parcel duals u_n and the slot duals ψ_k [J kg⁻¹] (`parcel_dual`, `slot_dual`);
- the smallest reduced cost of each parcel, min_k (c_nk - u_n - ψ_k) [J kg⁻¹] (`reduced_cost`), the slot where it is
  least (`best_slot`, as a float, so that the pass is vectorised), and the number of parcels for which it is negative
  (`certificate_violations`): zero if the placement is optimal;
- the number of pivots of the transportation simplex, of passes over the parcel-slot pairs and of placements alternating
  with the top pressure of FC2D (`pivots`, `pricing_passes`, `top_pressure_iterations`), and the decrease of the energy
  of the column from the sorted placement, by which the sort underestimates E_P - E_* [J m⁻¹] (`sorted_energy_excess`):
  no pivots, one pass, one placement and zero for `SortedRearrangement`.
"""
mutable struct ColumnPlacement{F, M <: AbstractRearrangement, W}
    const method::M
    const workspace::W                  # the workspace of the method (`rearrangement_workspace`)
    const key::Vector{F}                # sort key of each parcel [m³ kg⁻¹]
    const order::Vector{Int}            # the parcels in the order of the sorted placement
    const scratch::Vector{Int}          # scratch of the sort
    const placement::Placement{F}       # the masses x_nk
    const parcel_dual::Vector{F}        # u_n [J kg⁻¹]
    const slot_dual::Vector{F}          # ψ_k [J kg⁻¹]
    const reduced_cost::Vector{F}       # min_k (c_nk - u_n - ψ_k) of each parcel [J kg⁻¹]
    const best_slot::Vector{F}          # the slot k of the least reduced cost of each parcel, as a float (vectorised)
    certificate_violations::Int
    pivots::Int                         # pivots of the transportation simplex
    pricing_passes::Int                 # passes over the parcel-slot pairs
    top_pressure_iterations::Int        # placements alternating with P_top (FC2D)
    sorted_energy_excess::F             # the energy of the column of the sorted placement less that of the placement [J m⁻¹]
end
ColumnPlacement(method::AbstractRearrangement, F::Type, N, K) =
    ColumnPlacement(method, rearrangement_workspace(method, F, N, K), zeros(F, N), collect(1:N), zeros(Int, N),
                    Placement(F, N, K), zeros(F, N), zeros(F, K), zeros(F, N), zeros(F, N), 0, 0, 0, 0, zero(F))

"""
    c = parcel_cost(ref, model, n, k)
    c = slot_cost(ref, fluid, g, p_n, z_n, ϑ_n, q_n, k)

Energy per unit mass [J kg⁻¹] of parcel n placed in slot k, measured from its own pressure p_n and height z_n:
    c_nk = h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n) + g (z_k - z_n)   (AN2D, where the slots are the rows),
    c_nk = h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n)                   (FC2D, where the geopotential of the column does not
                                                                   depend on the placement once P_top is fixed),
with the difference of h by quadrature (`enthalpy_difference`). `slot_cost` takes the parcel's properties, so that the
pass over the pairs (`reduced_costs!`) reads them once for all the slots.
"""
@inline parcel_cost(ref::AbstractLorenzReference, model, n, k) =
    let (; pressure, height, consvar, q) = ref.parcels
        slot_cost(ref, model.fluid, model.space.g, pressure[n], height[n], consvar[n], q[n], k)
    end
@inline slot_cost(ref::AbstractLorenzReference, fluid, g, p_n, z_n, ϑ_n, q_n, k) =
    enthalpy_difference(fluid, p_n, ref.slots.pressure[k] - p_n, ϑ_n, q_n) + geopotential_cost(ref, g, k, z_n)

"""
    ref = slot_volumes!(ref, fluid)

The mean specific volume v_*,k = Σ_n x_nk v(P_k, ϑ_n, q_n)/M_k [m³ kg⁻¹] and compressibility
b_k = Σ_n x_nk ∂v/∂p(P_k, ϑ_n, q_n)/M_k [m³ kg⁻¹ Pa⁻¹] of the parcels of each slot, at the slot pressures P_k, from the
placement of `ref`.
"""
function slot_volumes!(ref::AbstractLorenzReference, fluid)
    (; slots, parcels) = ref
    (; placement) = ref.column
    fill!(slots.volume, 0)
    fill!(slots.compressibility, 0)
    for e in 1:placement.count
        n, k, x = placement.parcel[e], placement.slot[e], placement.mass[e]
        v, ∂v_∂p, _, _ = fluid(:p, :consvar, :q).volume_functions(slots.pressure[k], parcels.consvar[n], parcels.q[n])
        slots.volume[k] += x * v
        slots.compressibility[k] += x * ∂v_∂p
    end
    slots.volume ./= slots.mass
    slots.compressibility ./= slots.mass
    return ref
end

"""
    ref = certify!(ref, model)

The certificate of the placement: the number of parcels whose smallest reduced cost c_nk - u_n - ψ_k over the slots,
from the last pass of the rearrangement (`reduced_costs!`), is negative beyond the round-off of the duals
(`certificate_tolerance`).
"""
function certify!(ref::AbstractLorenzReference{F}, model) where F
    tolerance = certificate_tolerance(F, length(ref.slots.mass), model.space.g, model.space.Lz)
    ref.column.certificate_violations = count(r -> r < -tolerance, ref.column.reduced_cost)
    return ref
end

## AN2D
#
# The slots of the Lorenz column are the model rows, k = 1..M_z from the top, of mass M_k = M_x ρ_R Δx Δz and pressure
# P_k = P_R(z_k). A parcel n of the row o placed in slot k has the cost, measured from its own row,
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
with the `method` of rearrangement (an `AbstractRearrangement`). Its slots are the model rows, from the top, at the
pressures P_R(z_k); its parcels are at their own P_R(z_n).

After `reference_state!`, it holds:
- the parcels (`parcels`), the slots (`slots`), with the mean compressibility b_k of the parcels of each slot, with which
  v_* follows the parcels of the end slots between the end rows and the walls, and the placement of the parcels into
  the slots, with its duals and its certificate (`column`);
- the profiles P_R(z) and π_*(z) (`pressure`, `potential`), and P_R at the nodes of π_*, the levels of the search for
  reference heights (`level_pressure`);
- P_* [Pa] and v_* [m³ kg⁻¹] at the cell centres of the z axis (`p_star`, `v_star`).
After `reference_tendency!`, it holds the tendency of the reference state (`slot_means`, `potential_tendency`,
`p_star_tendency`, `plain_weight_slots`).
"""
mutable struct AnelasticLorenzReference{F, M <: AbstractRearrangement, W} <: AbstractLorenzReference{F, M}
    const parcels::Parcels{F}
    const slots::Slots{F}
    const column::ColumnPlacement{F, M, W}
    # the profiles
    const pressure::HermiteProfile{F}   # P_R(z) [Pa], through the cell centres of the z axis
    const potential::HermiteProfile{F}  # π_*(z) [J kg⁻¹], through both walls and the row centres, from the bottom
    const level_pressure::Vector{F}     # P_R at the nodes of π_* [Pa]
    const p_star::Vector{F}             # P_* at the cell centres of the z axis [Pa]
    const v_star::Vector{F}             # v_* at the cell centres of the z axis [m³ kg⁻¹]
    # the tendency of the reference state from the diabatic tendencies (`reference_tendency!`)
    const slot_means::SlotMeans{F}      # the N_*⁻²-weighted means of the parcels of each slot
    const potential_tendency::HermiteProfile{F} # ∂ₜπ_*(z) [J kg⁻¹ s⁻¹], at the nodes of π_*
    const p_star_tendency::Vector{F}    # ∂ₜP_* at the cell centres of the z axis [Pa s⁻¹]
    plain_weight_slots::Int             # slots whose means have plain weights, as N² ≤ 0 for one of their parcels
end

function AnelasticLorenzReference(model::AN2D{F}, method::AbstractRearrangement = SortedRearrangement()) where F
    (; domain, dx, dz, space, reference) = model
    (; ρ_R, p_R) = reference
    (; g) = space
    Mx, Mz = dims(domain)
    Hz = halo_size(domain)[2]
    N, K = Mx * Mz, Mz
    nz = ncells(domain, 2)
    ref = AnelasticLorenzReference(Parcels(F, N), Slots(F, K), ColumnPlacement(method, F, N, K), HermiteProfile(F, nz),
                                   HermiteProfile(F, K + 2), zeros(F, K + 2), zeros(F, nz), zeros(F, nz), SlotMeans(F, K),
                                   HermiteProfile(F, K + 2), zeros(F, nz), 0)
    (; parcels, slots) = ref
    # P_R through the cell centres of the whole z axis, with the slopes ∂_z P_R = -g ρ_R
    for j in 1:nz
        ref.pressure.z[j] = cellpoint(domain, dz, 2, j)
        ref.pressure.y[j] = p_R[j]
        ref.pressure.d[j] = -g * ρ_R[j]
    end
    # the slots: slot k is the row of cell index j = Hz + Mz + 1 - k
    for k in 1:K
        j = Hz + Mz + 1 - k
        slots.height[k] = cellpoint(domain, dz, 2, j)
        slots.pressure[k] = p_R[j]
        slots.mass[k] = Mx * ρ_R[j] * dx * dz
    end
    # the heights of the nodes of π_*: the bottom wall, the row centres from the bottom, the top wall
    ref.potential.z[1] = 0
    for k in 1:K
        ref.potential.z[K+2-k] = slots.height[k]
    end
    ref.potential.z[K+2] = space.Lz
    ref.potential_tendency.z .= ref.potential.z
    for c in 1:K+2
        ref.level_pressure[c] = hermite(ref.pressure, ref.potential.z[c])[1]
    end
    # the parcels: their mass, their own row, its height and P_R there
    for j in interior_cells(domain, 2), i in interior_cells(domain, 1)
        n = parcel_index(domain, i, j)
        parcels.mass[n] = ρ_R[j] * dx * dz
        parcels.row[n] = Hz + Mz + 1 - j
        parcels.height[n] = cellpoint(domain, dz, 2, j)
        parcels.pressure[n] = p_R[j]
    end
    return ref
end

"""
    c = geopotential_cost(ref, g, k, z_n)

The part of the cost c_nk of a parcel at the height z_n placed in slot k that comes from the geopotential [J kg⁻¹]:
g (z_k - z_n) for AN2D, whose slots are the rows; zero for FC2D, where the geopotential of the column does not depend on
the placement once P_top is fixed.
"""
@inline geopotential_cost(ref::AnelasticLorenzReference, g, k, z_n) = g * (ref.slots.height[k] - z_n)

"""
    ref = reference_state!(ref::AnelasticLorenzReference, model::AN2D, state)

Computes the Lorenz reference state `ref` of `state` in place: the parcels, their placement into the rows and its duals,
the profile π_*(z), P_* and v_* on the z axis, and the certificate of the placement.
"""
function reference_state!(ref::AnelasticLorenzReference, model::AN2D, state)
    (; mgr, domain, fluid, space) = model
    (; ρ_R, p_R) = model.reference
    (; m) = state
    (; parcels, slots) = ref
    g = space.g
    K = length(slots.mass)
    Hz = halo_size(domain)[2]

    # 1. The parcels: ϑ_n and q_n from the densities ρ_R ϑ and ρ_R q of the state, and v_n = v(P_R(z_n), ϑ_n, q_n)
    (; consvar, q, volume) = parcels
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ϑ_n = m[i, j, 1] / ρ_R[j]
            q_n = m[i, j, 2] / ρ_R[j]
            consvar[n] = ϑ_n
            q[n] = q_n
            volume[n] = fluid(:p, :consvar, :q).specific_volume(p_R[j], ϑ_n, q_n)
        end
    end

    # 2. The placement of the parcels into the slots, and its duals ψ_k and u_n, by the method of the reference state
    rearrange!(ref, model)

    # 3. The mean specific volume and compressibility of the parcels of each slot, at P_R(z_k)
    slot_volumes!(ref, fluid)

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
        d[K+2-k] = g * (ρ_R[j] * slots.volume[k] - 1)
    end
    y[2] = 0                                                        # the bottom row, before the constant is fixed
    for c in 3:K+1
        v₋, v₊ = slots.volume[K+3-c], slots.volume[K+2-c]           # v_* of the rows of nodes c - 1 and c
        y[c] = y[c-1] - (v₋ + v₊) / 2 * (ref.level_pressure[c] - ref.level_pressure[c-1]) - g * (z[c] - z[c-1])
    end
    # π_* is defined up to a constant, which changes neither the reference heights nor the available energy: its mean
    # over the mass of the rows, Σ_k M_k π_*(z_k) / Σ_k M_k, is zero
    mean_π = sum(k -> slots.mass[k] * y[K+2-k], 1:K) / sum(slots.mass)
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
        v_e, b_e = slots.volume[e], slots.compressibility[e]
        P_wall, ∂P_wall, _ = hermite(ref.pressure, z[wall])
        ΔP = P_wall - slots.pressure[e]                             # P_R(wall) - P_R(z_e)
        y[wall] = y[row] - (v_e + b_e * ΔP / 2) * ΔP - g * (z[wall] - z[row])
        d[wall] = -(v_e + b_e * ΔP) * ∂P_wall - g
    end

    # 5. P_* = P_R + ρ_R π_* and v_* = (∂_z π_*/g + 1)/ρ_R at the cell centres of the z axis
    for j in eachindex(ref.p_star)
        π, ∂π, _ = hermite(ref.potential, ref.pressure.z[j])
        ref.p_star[j] = p_R[j] + ρ_R[j] * π
        ref.v_star[j] = (∂π / g + 1) / ρ_R[j]
    end

    # 6. The certificate of the placement
    return certify!(ref, model)
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
- the parcels (`parcels`), at their own pressure p_n and specific volume v_n = 1/ρ_n, the slots (`slots`), with their
  pressure P_k and the height of their centre, and the placement of the parcels into the slots, with its duals and its
  certificate (`column`); the certificate holds for the top pressure of the placement;
- the top pressure `p_top` [Pa] of the column, and its `volume_residual`, (Σ_k V_k/L_x - L_z)/L_z;
- the profile P_*(z) (`pressure`), the walls and the interior slot edges, the levels of the search for reference heights
  (`levels`), and P_* there (`level_pressure`);
- P_* [Pa] and v_* = -g/∂_z P_* [m³ kg⁻¹] at the cell centres of the z axis (`p_star`, `v_star`).
After `reference_tendency!`, it holds the tendency of the reference state (`slot_means`, `edge_height_tendency`,
`dual_pressure`, `p_star_tendency`, `p_top_tendency`, `plain_weight_slots`).
"""
mutable struct CompressibleLorenzReference{F, M <: AbstractRearrangement, W} <: AbstractLorenzReference{F, M}
    const parcels::Parcels{F}
    const slots::Slots{F}
    const column::ColumnPlacement{F, M, W}
    # the profiles
    const pressure::HermiteProfile{F}   # P_*(z) [Pa], through the K + 1 slot edges, from the bottom
    const secant::Vector{F}             # secant slopes of P_* between the edges, -g/v_*,k [Pa m⁻¹]
    const levels::Vector{F}             # the walls and the interior slot edges, from the bottom [m]
    const level_pressure::Vector{F}     # P_* at the levels [Pa]
    const p_star::Vector{F}             # P_* at the cell centres of the z axis [Pa]
    const v_star::Vector{F}             # v_* at the cell centres of the z axis [m³ kg⁻¹]
    # the tendency of the reference state from the diabatic tendencies (`reference_tendency!`)
    const slot_means::SlotMeans{F}      # the N_*⁻²-weighted means of the parcels of each slot
    const edge_height_tendency::Vector{F}   # ∂ₜz_e of the slot edges, from the bottom [m s⁻¹]
    const dual_pressure::HermiteProfile{Dual{ReferenceTendencyTag, F, 1}}  # P_*(z), with ∂ₜP_* at fixed z as the partial
    const dual_secant::Vector{Dual{ReferenceTendencyTag, F, 1}}            # the secants of P_*, with their tendencies
    const p_star_tendency::Vector{F}    # ∂ₜP_* at the cell centres of the z axis [Pa s⁻¹]
    p_top::F                            # P_top [Pa]
    volume_residual::F                  # (Σ_k V_k/L_x - L_z)/L_z
    p_top_tendency::F                   # ∂ₜP_top [Pa s⁻¹]
    plain_weight_slots::Int             # slots whose means have plain weights, as N² ≤ 0 for one of their parcels
end

function CompressibleLorenzReference(model::FC2D{F}, method::AbstractRearrangement = SortedRearrangement()) where F
    (; domain, dz) = model
    Mx, Mz = dims(domain)
    Hz = halo_size(domain)[2]
    N, K = Mx * Mz, Mz
    nz = ncells(domain, 2)
    D = Dual{ReferenceTendencyTag, F, 1}
    ref = CompressibleLorenzReference(Parcels(F, N), Slots(F, K), ColumnPlacement(method, F, N, K), HermiteProfile(F, K + 1),
                                      zeros(F, K), zeros(F, K + 1), zeros(F, K + 1), zeros(F, nz), zeros(F, nz),
                                      SlotMeans(F, K), zeros(F, K + 1), HermiteProfile(D, K + 1), zeros(D, K), zeros(F, nz),
                                      zero(F), zero(F), zero(F), 0)
    # the parcels' own heights and rows: slot k is the row of cell index j = Hz + Mz + 1 - k
    for j in interior_cells(domain, 2), i in interior_cells(domain, 1)
        n = parcel_index(domain, i, j)
        ref.parcels.height[n] = cellpoint(domain, dz, 2, j)
        ref.parcels.row[n] = Hz + Mz + 1 - j
    end
    return ref
end

@inline geopotential_cost(ref::CompressibleLorenzReference, g, k, z_n) = zero(z_n)

"""
    ref = reference_state!(ref::CompressibleLorenzReference, model::FC2D, state)

Computes the Lorenz reference state `ref` of `state` in place: the parcels, their placement into the slots, the top
pressure of the column and the duals of the placement, the slot volumes and heights, the profile P_*(z), P_* and v_* on
the z axis, and the certificate of the placement.
"""
function reference_state!(ref::CompressibleLorenzReference, model::FC2D, state)
    (; mgr, domain, fluid, space, dx, dz) = model
    (; g, Lx, Lz) = space
    (; m) = state
    (; parcels, slots) = ref
    K = length(slots.mass)

    # 1. The parcels: ρ, ϑ_n and q_n from the densities ρ, ρ ϑ and ρ q of the state, m_n = ρ Δx Δz, v_n = 1/ρ, and their
    # pressure p_n = p(v_n, ϑ_n, q_n) from the equation of state
    (; mass, consvar, q, volume, pressure) = parcels
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
            volume[n] = v
            pressure[n] = fluid(:v, :consvar, :q).pressure(v, ϑ_n, q_n)
        end
    end
    # the mass M_k of each slot, the sum of the masses of the parcels of its row
    fill!(slots.mass, 0)
    for n in eachindex(mass)
        slots.mass[parcels.row[n]] += mass[n]
    end

    # 2. The placement of the parcels into the slots, the top pressure P_top and the slot pressures P_k, and the duals
    # ψ_k and u_n, by the method of the reference state
    rearrange!(ref, model)

    # 3. The mean specific volume and compressibility of the parcels of each slot, at P_k
    slot_volumes!(ref, fluid)

    # 4. The slots stacked from the bottom, each of thickness V_k/L_x = v_*,k M_k/L_x, and P_* at their edges: edge 1 is
    # the bottom, edge K + 1 the top, and edge e is below slot K + 1 - e, so P_*(z_e) = P_top + g (Σ_{k ≤ K+1-e} M_k)/L_x,
    # from the top down. Between the edges, P_* is the monotone cubic of secant slopes -g/v_*,k
    (; z, y, d) = ref.pressure
    z[1] = 0
    for e in 1:K
        k = K + 1 - e                                               # the slot above edge e
        z[e+1] = z[e] + slots.volume[k] * slots.mass[k] / Lx
        slots.height[k] = (z[e] + z[e+1]) / 2
        ref.secant[e] = -g / slots.volume[k]
    end
    y[K+1] = ref.p_top
    for e in K:-1:1
        y[e] = y[e+1] + g * slots.mass[K+1-e] / Lx                  # the weight of the slot above edge e
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

    # 6. The certificate of the placement
    return certify!(ref, model)
end
