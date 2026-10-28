## THE PLACEMENT OF THE PARCELS INTO THE SLOTS
#
# The step of `reference_state!` that depends on the method of the reference state (`AbstractRearrangement`): the
# placement of the parcels into the slots and its dual potentials, ψ_k of the slots and u_n of the parcels [J kg⁻¹]. Both
# methods start from the parcels sorted by their specific volume, which fill the slots from the top
# (`sort_parcels!`); the sorted method keeps that placement, and the exact method improves it by the transportation
# simplex. The reduced costs of every parcel-slot pair (`reduced_costs!`) are the certificate of either. In FC2D, the
# slot pressures follow from the top pressure of the column (`fit_column!`), which depends on the placement.

"""
    ref = rearrange!(ref, model)

The step of the reference state `ref` of `model` that depends on its method `ref.column.method` (an
`AbstractRearrangement`): the placement of the parcels into the slots and its dual potentials, ψ_k of the slots and u_n
of the parcels [J kg⁻¹].
"""
function rearrange! end

"""
    p = sort_pressure(ref, model)

The pressure [Pa] at which the specific volumes of the parcels are compared to sort them (`sort_parcels!`): P_R(L_z/2)
at mid-depth for AN2D, the mean pressure p̄ of the parcels for FC2D.
"""
sort_pressure(ref::AnelasticLorenzReference, model::AN2D) = hermite(ref.pressure, model.space.Lz / 2)[1]
sort_pressure(ref::CompressibleLorenzReference, model::FC2D) = sum(ref.parcels.pressure) / length(ref.parcels.pressure)

"""
    ref = fit_column!(ref, model)

The slot pressures P_k [Pa] of the placement of `ref`: fixed for AN2D, whose slots are the rows at P_R(z_k); for FC2D,
from the top pressure of the column that fills the box (`top_pressure!`).
"""
fit_column!(ref::AnelasticLorenzReference, model::AN2D) = ref
fit_column!(ref::CompressibleLorenzReference, model::FC2D) = top_pressure!(ref, model)

"""
    ref = sort_parcels!(ref, model)

The parcels, sorted by their specific volume v(p, ϑ_n, q_n) at the pressure `sort_pressure` [m³ kg⁻¹], lightest first,
fill the slots from the top (`sorted_placement!`): the placement of `SortedRearrangement`, and the start of
`ExactRearrangement`.
"""
function sort_parcels!(ref::AbstractLorenzReference, model)
    (; mgr, domain, fluid) = model
    (; consvar, q) = ref.parcels
    (; key, order, scratch, placement) = ref.column
    # the sort key, by the model's manager
    p_key = sort_pressure(ref, model)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            key[n] = fluid(:p, :consvar, :q).specific_volume(p_key, consvar[n], q[n])
        end
    end
    sortperm!(order, key; rev = true, scratch)
    sorted_placement!(placement, order, ref.parcels.mass, ref.slots.mass)
    return ref
end

"""
    ref = rearrange!(ref::AbstractLorenzReference{<:Any, SortedRearrangement}, model)

The sorted rearrangement (`sort_parcels!`), with the slot pressures of the placement (`fit_column!`); the duals ψ_k and
u_n [J kg⁻¹] follow along the staircase of the placement (`staircase_duals!`) with the costs `parcel_cost`, and the
reduced costs from one pass over the pairs (`reduced_costs!`).
"""
function rearrange!(ref::AbstractLorenzReference{<:Any, SortedRearrangement}, model)
    column = ref.column
    sort_parcels!(ref, model)
    fit_column!(ref, model)
    staircase_duals!(column.slot_dual, column.parcel_dual, column.placement, column.order, (n, k) -> parcel_cost(ref, model, n, k))
    reduced_costs!(ref, model)
    column.pivots, column.pricing_passes, column.top_pressure_iterations, column.sorted_energy_excess = 0, 1, 1, 0
    return ref
end

"""
    (passes, decrease) = simplex!(ref, model, max_passes)

The least-cost placement at the current slot pressures by the transportation simplex (`optimal_placement!`), from the
tree of the placement of `ref`, with the costs `parcel_cost` and the passes over the pairs `reduced_costs!`, in at most
`max_passes` passes; adds its pivots to `ref.column.pivots`, and returns its passes and the decrease of Σ_nk x_nk c_nk.
"""
function simplex!(ref::AbstractLorenzReference{F}, model, max_passes) where F
    column = ref.column
    cost(n, k) = parcel_cost(ref, model, n, k)
    price!() = reduced_costs!(ref, model)
    tolerance = certificate_tolerance(F, length(ref.slots.mass), model.space.g, model.space.Lz)
    passes, decrease = optimal_placement!(column.placement, column.workspace, column.slot_dual, column.parcel_dual,
                                          column.reduced_cost, column.best_slot, cost, price!, tolerance, max_passes)
    column.pivots += column.workspace.pivots
    return passes, decrease
end

"""
    ref = rearrange!(ref::AnelasticLorenzReference{<:Any, ExactRearrangement}, model::AN2D)

The exact rearrangement of AN2D: the least-cost placement by the transportation simplex (`simplex!`), from the sorted
placement (`sort_parcels!`) completed into a tree (`complete_staircase!`). The masses of the parcels and of the rows do
not depend on the state's motion, so every tree of the simplex is a feasible placement, and the decrease of the cost is
the underestimate of E_P - E_* by the sort.
"""
function rearrange!(ref::AnelasticLorenzReference{<:Any, ExactRearrangement}, model::AN2D)
    column = ref.column
    sort_parcels!(ref, model)
    complete_staircase!(column.placement, column.order)
    column.pivots = 0
    column.pricing_passes, column.sorted_energy_excess = simplex!(ref, model, column.method.passes_per_slot * length(ref.slots.mass))
    column.top_pressure_iterations = 1
    return ref
end

"""
    ref = rearrange!(ref::CompressibleLorenzReference{<:Any, ExactRearrangement}, model::FC2D)

The exact rearrangement of FC2D, the saddle point of the energy of the column G(x, P_top) = Σ_nk x_nk c_nk(P_top) -
P_top L_x L_z, the least over the placements x of its greatest value over P_top, which is where the column fills the
box. From the sorted placement (`sort_parcels!`) completed into a tree (`complete_staircase!`), two steps alternate:
1. P_top at fixed placement (`top_pressure!`);
2. the least-cost placement at fixed P_top by the transportation simplex (`simplex!`), from the tree of the previous one,
   with the costs at the slot pressures P_k(P_top);
until the placement no longer changes (no pivot moves mass), at most `max_top_iterations` times: the last placement is
then only priced, and its certificate fails.
"""
function rearrange!(ref::CompressibleLorenzReference{<:Any, ExactRearrangement}, model::FC2D)
    column = ref.column
    (; max_top_iterations, passes_per_slot) = column.method
    sort_parcels!(ref, model)
    complete_staircase!(column.placement, column.order)
    top_pressure!(ref, model)
    G_sorted = column_energy(ref, model)
    column.pivots, column.pricing_passes = 0, 0
    for iteration in 1:max_top_iterations+1
        # the last iteration only prices the placement
        max_passes = iteration > max_top_iterations ? 1 : passes_per_slot * length(ref.slots.mass)
        passes, _ = simplex!(ref, model, max_passes)
        column.pricing_passes += passes
        column.top_pressure_iterations = iteration
        column.workspace.mass_pivots == 0 && break
        top_pressure!(ref, model)
    end
    column.sorted_energy_excess = G_sorted - column_energy(ref, model)
    return ref
end

"""
    G = column_energy(ref::CompressibleLorenzReference, model::FC2D)

The energy of the column of the placement of `ref` at its top pressure P_top, G = Σ_nk x_nk c_nk - P_top L_x L_z
[J m⁻¹], with the costs c_nk = h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n) (`parcel_cost`): the reference energy E_*, up to terms
that do not depend on the placement.
"""
function column_energy(ref::CompressibleLorenzReference, model::FC2D)
    (; placement) = ref.column
    G = -ref.p_top * model.space.Lx * model.space.Lz
    for e in 1:placement.count
        G += placement.mass[e] * parcel_cost(ref, model, placement.parcel[e], placement.slot[e])
    end
    return G
end

"""
    ref = reduced_costs!(ref, model)

The smallest reduced cost c_nk - u_n - ψ_k of each parcel over the slots (`ref.column.reduced_cost`) [J kg⁻¹], and the
slot where it is least (`ref.column.best_slot`), with the duals of `ref.column` and the costs from the parcel's own
pressure p_n and height z_n (`slot_cost`), c_nk = h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n) (+ g (z_k - z_n) for AN2D), by
the model's manager (vectorised over the parcels of a row). All N K pairs: the certificate of the placement, and the
pricing of the transportation simplex.
"""
function reduced_costs!(ref::AbstractLorenzReference{F}, model) where F
    (; mgr, domain, fluid, space) = model
    (; consvar, q, pressure, height) = ref.parcels
    (; parcel_dual, slot_dual, reduced_cost, best_slot) = ref.column
    g = space.g
    K = length(slot_dual)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ϑ_n, q_n, u_n, p_n, z_n = consvar[n], q[n], parcel_dual[n], pressure[n], height[n]
            reduced = slot_cost(ref, fluid, g, p_n, z_n, ϑ_n, q_n, 1) - u_n - slot_dual[1]
            best = zero(reduced) + 1                                # a float, as `reduced`, so that it is vectorised
            for k in 2:K
                r = slot_cost(ref, fluid, g, p_n, z_n, ϑ_n, q_n, k) - u_n - slot_dual[k]
                best = ifelse(r < reduced, F(k), best)              # ifelse, not ? :, so that it is vectorised
                reduced = min(reduced, r)
            end
            reduced_cost[n] = reduced
            best_slot[n] = best
        end
    end
    return ref
end

"""
    ref = top_pressure!(ref::CompressibleLorenzReference, model::FC2D)

The top pressure `ref.p_top` [Pa] of the column of the placement, and the slot pressures
P_k = P_top + g (Σ_{l<k} M_l + M_k/2)/L_x [Pa], by Newton's method on the volume residual
    R(P_top) = Σ_nk x_nk v(P_k, ϑ_n, q_n) - L_x L_z,   dR/dP_top = Σ_nk x_nk ∂v/∂p(P_k, ϑ_n, q_n) < 0.
It starts from the mean pressure of the parcels of the top row less the weight of half a row, p_n - g Δz/(2v_n), and
stops when |R| ≤ 16 eps L_x L_z or |R| no longer decreases.
"""
function top_pressure!(ref::CompressibleLorenzReference{F}, model::FC2D) where F
    (; fluid, space, dz) = model
    (; g, Lx, Lz) = space
    (; parcels, slots) = ref
    (; placement) = ref.column
    Mx, Mz = dims(model.domain)
    K = length(slots.mass)
    # the first guess, from the parcels of the top row, n = M_x (M_z - 1) + 1 .. M_x M_z
    ref.p_top = sum(n -> parcels.pressure[n] - g * dz / (2 * parcels.volume[n]), Mx*(Mz-1)+1:Mx*Mz) / Mx
    box_volume = Lx * Lz
    residual = F(Inf)
    for _ in 1:50
        above = zero(F)                                             # the mass above slot k
        for k in 1:K
            slots.pressure[k] = ref.p_top + g * (above + slots.mass[k] / 2) / Lx
            above += slots.mass[k]
        end
        R, ∂R = -box_volume, zero(F)
        for e in 1:placement.count
            n, k = placement.parcel[e], placement.slot[e]
            v, ∂v_∂p, _, _ = fluid(:p, :consvar, :q).volume_functions(slots.pressure[k], parcels.consvar[n], parcels.q[n])
            R += placement.mass[e] * v
            ∂R += placement.mass[e] * ∂v_∂p
        end
        # stop when |R| is round-off, or no longer decreases (the slot pressures are those of the last P_top)
        abs(R) >= residual && break
        residual = abs(R)
        residual <= 16 * eps(F) * box_volume && break
        ref.p_top -= R / ∂R
    end
    return ref
end
