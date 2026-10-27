## THE PLACEMENT OF THE PARCELS INTO THE SLOTS
#
# The step of `reference_state!` that depends on the method of the reference state (`AbstractRearrangement`): the
# placement of the parcels into the slots and its dual potentials, ψ_k of the slots and u_n of the parcels [J kg⁻¹]. Both
# methods start from the parcels sorted by their specific volume, which fill the slots from the top
# (`sort_parcels!`); the sorted method keeps that placement, and the exact method improves it by the transportation
# simplex. The reduced costs of every parcel-slot pair (`reduced_costs!`) are the certificate of either.

"""
    ref = rearrange!(ref, model)

The step of the reference state `ref` of `model` that depends on its method `ref.method` (an `AbstractRearrangement`):
the placement of the parcels into the slots and its dual potentials, ψ_k of the slots and u_n of the parcels [J kg⁻¹].
"""
function rearrange! end

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
    ref = sort_parcels!(ref::CompressibleLorenzReference, model::FC2D)

The parcels, sorted by their specific volume v(p̄, ϑ_n, q_n) at the mean pressure p̄ of the parcels [m³ kg⁻¹], lightest
first, fill the slots from the top (`sorted_placement!`): the placement of `SortedRearrangement`, and the start of
`ExactRearrangement`.
"""
function sort_parcels!(ref::CompressibleLorenzReference, model::FC2D)
    (; mgr, domain, fluid) = model
    (; key, consvar, q) = ref
    # the sort key, by the model's manager
    p_key = sum(ref.parcel_pressure) / length(ref.parcel_pressure)
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
    ref = rearrange!(ref::CompressibleLorenzReference{<:Any, SortedRearrangement}, model::FC2D)

The sorted rearrangement of FC2D (`sort_parcels!`); the top pressure P_top of the column follows from its volume
(`top_pressure!`), the duals ψ_k and u_n [J kg⁻¹] along the staircase of the placement (`staircase_duals!`) with the
costs at the slot pressures P_k, and the reduced costs from one pass over the pairs (`reduced_costs!`).
"""
function rearrange!(ref::CompressibleLorenzReference{<:Any, SortedRearrangement}, model::FC2D)
    sort_parcels!(ref, model)
    top_pressure!(ref, model)
    staircase_duals!(ref.slot_dual, ref.u, ref.placement, ref.order, (n, k) -> compressible_cost(model, ref, n, k))
    reduced_costs!(ref, model)
    ref.pivots, ref.pricing_passes, ref.top_pressure_iterations, ref.sorted_energy_excess = 0, 1, 1, 0
    return ref
end

"""
    ref = rearrange!(ref::CompressibleLorenzReference{<:Any, ExactRearrangement}, model::FC2D)

The exact rearrangement of FC2D, the saddle point of the energy of the column G(x, P_top) = Σ_nk x_nk c_nk(P_top) -
P_top L_x L_z, the least over the placements x of its greatest value over P_top, which is where the column fills the
box. From the sorted placement (`sort_parcels!`) completed into a tree (`complete_staircase!`), two steps alternate:
1. P_top at fixed placement (`top_pressure!`);
2. the least-cost placement at fixed P_top by the transportation simplex (`optimal_placement!`), from the tree of the
   previous one, with the costs `compressible_cost` at the slot pressures P_k(P_top) and the passes `reduced_costs!`;
until the placement no longer changes (no pivot moves mass), at most `max_top_iterations` times: the last placement is
then only priced, and its certificate fails.
"""
function rearrange!(ref::CompressibleLorenzReference{F, ExactRearrangement}, model::FC2D) where F
    (; space) = model
    tree = ref.workspace
    sort_parcels!(ref, model)
    complete_staircase!(ref.placement, ref.order)
    top_pressure!(ref, model)
    G_sorted = column_energy(ref, model)
    cost(n, k) = compressible_cost(model, ref, n, k)
    price!() = reduced_costs!(ref, model)
    tolerance = certificate_tolerance(F, length(ref.slot_mass), space.g, space.Lz)
    ref.pivots, ref.pricing_passes = 0, 0
    for iteration in 1:ref.method.max_top_iterations+1
        # the last iteration only prices the placement
        max_passes = iteration > ref.method.max_top_iterations ? 1 : ref.method.passes_per_slot * length(ref.slot_mass)
        passes, _ = optimal_placement!(ref.placement, tree, ref.slot_dual, ref.u, ref.reduced_cost, ref.best_slot, cost,
                                       price!, tolerance, max_passes)
        ref.pivots += tree.pivots
        ref.pricing_passes += passes
        ref.top_pressure_iterations = iteration
        tree.mass_pivots == 0 && break
        top_pressure!(ref, model)
    end
    ref.sorted_energy_excess = G_sorted - column_energy(ref, model)
    return ref
end

"""
    G = column_energy(ref::CompressibleLorenzReference, model::FC2D)

The energy of the column of the placement of `ref` at its top pressure P_top, G = Σ_nk x_nk c_nk - P_top L_x L_z
[J m⁻¹], with the costs c_nk = h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n) (`compressible_cost`): the reference energy E_*, up to
terms that do not depend on the placement.
"""
function column_energy(ref::CompressibleLorenzReference, model::FC2D)
    (; placement) = ref
    G = -ref.p_top * model.space.Lx * model.space.Lz
    for e in 1:placement.count
        G += placement.mass[e] * compressible_cost(model, ref, placement.parcel[e], placement.slot[e])
    end
    return G
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

"""
    ref = reduced_costs!(ref::CompressibleLorenzReference, model::FC2D)

The smallest reduced cost c_nk - u_n - ψ_k of each parcel over the slots (`ref.reduced_cost`) [J kg⁻¹], and the slot
where it is least (`ref.best_slot`), with the duals `ref.u` and `ref.slot_dual` and the costs from the parcel's own
pressure, c_nk = h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n), by the model's manager (vectorised over the parcels of a row). All
N K pairs: the certificate of the placement, and the pricing of the transportation simplex.
"""
function reduced_costs!(ref::CompressibleLorenzReference{F}, model::FC2D) where F
    (; mgr, domain, fluid) = model
    (; consvar, q, u, parcel_pressure, slot_pressure, slot_dual, reduced_cost, best_slot) = ref
    K = length(slot_pressure)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ϑ_n, q_n, u_n, p_n = consvar[n], q[n], u[n], parcel_pressure[n]
            reduced = enthalpy_difference(fluid, p_n, slot_pressure[1] - p_n, ϑ_n, q_n) - u_n - slot_dual[1]
            best = zero(reduced) + 1                                # a float, as `reduced`, so that it is vectorised
            for k in 2:K
                c = enthalpy_difference(fluid, p_n, slot_pressure[k] - p_n, ϑ_n, q_n)
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
    (; placement, slot_pressure) = ref
    Mx, Mz = dims(model.domain)
    K = length(ref.slot_mass)
    # the first guess, from the parcels of the top row, n = M_x (M_z - 1) + 1 .. M_x M_z
    ref.p_top = sum(n -> ref.parcel_pressure[n] - g * dz / (2 * ref.parcel_volume[n]), Mx*(Mz-1)+1:Mx*Mz) / Mx
    box_volume = Lx * Lz
    residual = F(Inf)
    for _ in 1:50
        above = zero(F)                                             # the mass above slot k
        for k in 1:K
            slot_pressure[k] = ref.p_top + g * (above + ref.slot_mass[k] / 2) / Lx
            above += ref.slot_mass[k]
        end
        R, ∂R = -box_volume, zero(F)
        for e in 1:placement.count
            n, k = placement.parcel[e], placement.slot[e]
            v, ∂v_∂p, _, _ = fluid(:p, :consvar, :q).volume_functions(slot_pressure[k], ref.consvar[n], ref.q[n])
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
