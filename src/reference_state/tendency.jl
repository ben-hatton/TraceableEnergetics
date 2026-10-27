## TENDENCY OF THE LORENZ REFERENCE STATE FROM THE DIABATIC TENDENCIES
#
# The Lorenz reference state depends only on the distribution of mass over (ϑ, q): advection rearranges the fluid without
# changing it, so P_* changes only through the diabatic rates of the fluid elements,
#     ϑ̇ = Dϑ/Dt = (σ_ϑ - ∇·J_ϑ)/ρ,   q̇ = Dq/Dt = -∇·J_q/ρ   (ρ = ρ_R for AN2D),
# which change the specific volume of a parcel at fixed pressure at v̇ = ∂v/∂ϑ ϑ̇ + ∂v/∂q q̇. Following a parcel at its
# level of neutral buoyancy, v(P_*(z_*), ϑ, q) = v_*(z_*), and requiring that the mass at each level stays the same, gives
# the tendency of the reference specific volume as the mean of v̇ over the parcels at the level, weighted by 1/N_*²
# (`slot_means!`):
#     AN2D:  ∂ₜv_*(z) = ⟨v̇⟩,   ∂_z(∂ₜπ_*) = g ρ_R ∂ₜv_*,   ∂ₜP_* = ρ_R ∂ₜπ_*,
#     FC2D:  ∂ₜv_*|_m = ⟨v̇⟩ + ⟨∂v/∂p⟩ ∂ₜP_top,   with ∂ₜP_top such that the column still fills the box,
# at fixed mass coordinate m (the mass above a level, per unit length along x) in FC2D. Discretely, the parcels at a
# level are those of a slot, with the masses of the placement, and the tendencies are those of the slot profile: of the
# nodes of π_* in AN2D, of the slot edges in FC2D, from which follows ∂ₜP_* at fixed height.

"""
    count = slot_means!(means::SlotMeans, placement, slot_mass, slot_pressure, fluid, g, consvar, q, consvar_rate, q_rate)

The means over the parcels of each slot k, weighted by the inverse of their stratification at the slot,
    ⟨a⟩_k = Σ_n w_nk a_nk / Σ_n w_nk,   w_nk = x_nk/N²_nk,   N²_nk = g ρ_k (∂_z v_*(z_k) + g ρ_k ∂v/∂p_nk),
of the rate of change of the specific volume at the slot pressure P_k, v̇_nk = ∂v/∂ϑ ϑ̇_n + ∂v/∂q q̇_n [m³ kg⁻¹ s⁻¹], and of
the compressibility ∂v/∂p_nk [m³ kg⁻¹ Pa⁻¹], with v and its derivatives at (P_k, ϑ_n, q_n), the parcel rates ϑ̇_n, q̇_n
(`consvar_rate`, `q_rate`), and ρ_k and ∂_z v_*(z_k) from `means`. A parcel that is weakly stratified at its level moves
further for a given change of its volume, and displaces more of the others. Where N²_nk ≤ 0 for a parcel of the slot,
the weights are undefined and the slot takes the plain means, w_nk = x_nk. Returns the number of such slots.
"""
function slot_means!(means::SlotMeans{F}, placement, slot_mass, slot_pressure, fluid, g, consvar, q, consvar_rate, q_rate) where F
    (; density, volume_gradient, weight, volume_rate, compressibility, plain_volume_rate, plain_compressibility, unstable) = means
    foreach(a -> fill!(a, 0), (weight, volume_rate, compressibility, plain_volume_rate, plain_compressibility))
    fill!(unstable, false)
    for e in 1:placement.count
        n, k, x = placement.parcel[e], placement.slot[e], placement.mass[e]
        x > 0 || continue                                           # an entry of zero mass of the exact placement
        _, ∂v_∂p, ∂v_∂ϑ, ∂v_∂q = fluid(:p, :consvar, :q).volume_functions(slot_pressure[k], consvar[n], q[n])
        v̇ = ∂v_∂ϑ * consvar_rate[n] + ∂v_∂q * q_rate[n]                # at fixed pressure
        N² = g * density[k] * (volume_gradient[k] + g * density[k] * ∂v_∂p)
        unstable[k] |= N² <= 0
        w = x / N²
        weight[k] += w
        volume_rate[k] += w * v̇
        compressibility[k] += w * ∂v_∂p
        plain_volume_rate[k] += x * v̇
        plain_compressibility[k] += x * ∂v_∂p
    end
    for k in eachindex(weight)
        plain_volume_rate[k] /= slot_mass[k]
        plain_compressibility[k] /= slot_mass[k]
        volume_rate[k] = unstable[k] ? plain_volume_rate[k] : volume_rate[k] / weight[k]
        compressibility[k] = unstable[k] ? plain_compressibility[k] : compressibility[k] / weight[k]
    end
    return count(unstable)
end

## AN2D

"""
    ref = reference_tendency!(ref::AnelasticLorenzReference, model::AN2D, consvar_rate, q_rate)

The tendency of the Lorenz reference state `ref` of AN2D (computed by `reference_state!`) from the diabatic tendencies
ρ_R Dϑ/Dt = σ_ϑ - ∇·J_ϑ (`consvar_rate`, [kg m⁻³ s⁻¹ × consvar]) and ρ_R Dq/Dt = -∇·J_q (`q_rate`, [kg m⁻³ s⁻¹]) on the
cells. With the rates of the parcels ϑ̇_n, q̇_n, it computes
- ∂ₜv_*,k = ⟨v̇⟩_k [m³ kg⁻¹ s⁻¹], the mean over the parcels of slot k weighted by 1/N²_nk (`slot_means!`), with v̇ and
  ∂v/∂p at P_R(z_k), ρ_k = ρ_R(z_k), and N²_nk = g ρ_R (∂_z v_* + g ρ_R ∂v/∂p) (`slot_volume_tendency`);
- ∂ₜπ_*(z) [J kg⁻¹ s⁻¹] (`potential_tendency`) at the nodes of π_*, by the rule of π_*: the slopes g ρ_R ∂ₜv_*,k and the
  values from the trapezoidal rule in P_R, ∂ₜπ_*(z_c) - ∂ₜπ_*(z_c-1) = -(∂ₜv_*,c-1 + ∂ₜv_*,c)/2 (P_R(z_c) - P_R(z_c-1)),
  with a zero mean over the mass of the rows; between the end rows and the walls, from ∂ₜv_* = ∂ₜv_e, the tendency of the
  volume of the end slot e (the change of its compressibility b_e is neglected);
- ∂ₜP_* = ρ_R ∂ₜπ_* [Pa s⁻¹] at the cell centres of the z axis (`p_star_tendency`);
- the number of slots with plain weights (`plain_weight_slots`), where N² ≤ 0 for one of their parcels.
"""
function reference_tendency!(ref::AnelasticLorenzReference{F}, model::AN2D, consvar_rate, q_rate) where F
    (; mgr, domain, fluid, space) = model
    (; ρ_R) = model.reference
    g = space.g
    K = length(ref.slot_mass)
    Hz = halo_size(domain)[2]

    # 1. The rates of the parcels, ϑ̇_n = (ρ_R Dϑ/Dt)/ρ_R and q̇_n = (ρ_R Dq/Dt)/ρ_R
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ref.consvar_rate[n] = consvar_rate[i, j] / ρ_R[j]
            ref.q_rate[n] = q_rate[i, j] / ρ_R[j]
        end
    end

    # 2. The density and the gradient of v_* at each slot: from v_* = (∂_z π_*/g + 1)/ρ_R = -(∂_z π_* + g)/∂_z P_R,
    # ∂_z v_* = -(∂²_z π_* ∂_z P_R - (∂_z π_* + g) ∂²_z P_R)/(∂_z P_R)²
    (; slot_means) = ref
    for k in 1:K
        _, ∂P, ∂²P = hermite(ref.pressure, ref.slot_height[k])
        _, ∂π, ∂²π = hermite(ref.potential, ref.slot_height[k])
        slot_means.density[k] = ρ_R[Hz+K+1-k]
        slot_means.volume_gradient[k] = -(∂²π * ∂P - (∂π + g) * ∂²P) / ∂P^2
    end

    # 3. ∂ₜv_*,k = ⟨v̇⟩_k, with v̇ at P_R(z_k)
    ref.plain_weight_slots = slot_means!(slot_means, ref.placement, ref.slot_mass, ref.slot_pressure, fluid, g, ref.consvar,
                                         ref.q, ref.consvar_rate, ref.q_rate)
    ref.slot_volume_tendency .= slot_means.volume_rate

    # 4. The nodes of ∂ₜπ_*, by the rule of the nodes of π_*: node K + 2 - k is slot k
    (; z, y, d) = ref.potential_tendency
    for k in 1:K
        d[K+2-k] = g * ρ_R[Hz+K+1-k] * ref.slot_volume_tendency[k]
    end
    y[2] = 0
    for c in 3:K+1
        v̇₋, v̇₊ = ref.slot_volume_tendency[K+3-c], ref.slot_volume_tendency[K+2-c]    # ∂ₜv_* of the rows of nodes c - 1 and c
        y[c] = y[c-1] - (v̇₋ + v̇₊) / 2 * (ref.node_pressure[c] - ref.node_pressure[c-1])
    end
    # the constant of ∂ₜπ_*, which changes no budget term, keeps the mean of π_* over the mass of the rows zero
    mean_π̇ = sum(k -> ref.slot_mass[k] * y[K+2-k], 1:K) / sum(ref.slot_mass)
    for c in 2:K+1
        y[c] -= mean_π̇
    end
    # between the end rows and the walls, π_*(z) = π_*(z_e) - v_e ΔP - b_e ΔP²/2 - g (z - z_e), with ΔP = P_R(z) - P_R(z_e):
    # with b_e held fixed, ∂ₜπ_*(z) = ∂ₜπ_*(z_e) - ∂ₜv_e ΔP and ∂_z ∂ₜπ_* = -∂ₜv_e ∂_z P_R
    for (wall, row, e) in ((1, 2, K), (K + 2, K + 1, 1))
        v̇_e = ref.slot_volume_tendency[e]
        P_wall, ∂P_wall, _ = hermite(ref.pressure, z[wall])
        y[wall] = y[row] - v̇_e * (P_wall - ref.slot_pressure[e])
        d[wall] = -v̇_e * ∂P_wall
    end

    # 5. ∂ₜP_* = ρ_R ∂ₜπ_* at the cell centres of the z axis
    for j in eachindex(ref.p_star_tendency)
        ref.p_star_tendency[j] = ρ_R[j] * hermite(ref.potential_tendency, ref.pressure.z[j])[1]
    end
    return ref
end

"""
    ∂ₜP = reference_pressure_tendency(ref::AnelasticLorenzReference, model::AN2D, z)

∂ₜP_*(z) = ρ_R(z) ∂ₜπ_*(z) [Pa s⁻¹] at the height z [m], with ρ_R(z) = -∂_z P_R(z)/g, after `reference_tendency!`.
"""
@inline function reference_pressure_tendency(ref::AnelasticLorenzReference, model::AN2D, z)
    ρ_R = -hermite(ref.pressure, z)[2] / model.space.g
    return ρ_R * hermite(ref.potential_tendency, z)[1]
end

## FC2D

"""
    ref = reference_tendency!(ref::CompressibleLorenzReference, model::FC2D, consvar_rate, q_rate)

The tendency of the Lorenz reference state `ref` of FC2D (computed by `reference_state!`) from the diabatic tendencies
ρ Dϑ/Dt = σ_ϑ - ∇·J_ϑ (`consvar_rate`, [kg m⁻³ s⁻¹ × consvar]) and ρ Dq/Dt = -∇·J_q (`q_rate`, [kg m⁻³ s⁻¹]) on the cells,
at fixed slot masses. With the rates of the parcels ϑ̇_n, q̇_n and the means ⟨⋅⟩_k over the parcels of slot k weighted by
1/N²_nk (`slot_means!`), with v̇ and ∂v/∂p at the slot pressure P_k, ρ_k = 1/v_*,k and N²_nk = g ρ_k (∂_z v_* + g ρ_k ∂v/∂p),
it computes
- ∂ₜP_top = -Σ_k M_k ⟨v̇⟩_k / Σ_k M_k ⟨∂v/∂p⟩_k [Pa s⁻¹], so that the column still fills the box (`p_top_tendency`);
- ∂ₜv_*,k = ⟨v̇⟩_k + ⟨∂v/∂p⟩_k ∂ₜP_top [m³ kg⁻¹ s⁻¹] at fixed mass coordinate (`slot_volume_tendency`), with
  Σ_k M_k ∂ₜv_*,k = 0;
- the motion of the slot edges, ∂ₜz_e = Σ_{k below e} M_k ∂ₜv_*,k/L_x [m s⁻¹] (`edge_height_tendency`), zero at both
  walls, whose pressures change at ∂ₜP_top;
- the profile P_*(z) as dual numbers (`dual_pressure`), whose partial at a fixed height is ∂ₜP_* there: the derivative of
  the monotone cubic through the edges with respect to the motion of its nodes (`reference_pressure_tendency`);
- ∂ₜP_* [Pa s⁻¹] at fixed height at the cell centres of the z axis (`p_star_tendency`);
- the number of slots with plain weights (`plain_weight_slots`), where N² ≤ 0 for one of their parcels.
"""
function reference_tendency!(ref::CompressibleLorenzReference{F}, model::FC2D, consvar_rate, q_rate) where F
    (; mgr, domain, fluid, space, dx, dz) = model
    (; g, Lx) = space
    K = length(ref.slot_mass)

    # 1. The rates of the parcels, ϑ̇_n = (ρ Dϑ/Dt)/ρ and q̇_n = (ρ Dq/Dt)/ρ, with ρ = m_n/(Δx Δz)
    (; mass) = ref
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            volume = dx * dz / mass[n]                                  # 1/ρ
            ref.consvar_rate[n] = consvar_rate[i, j] * volume
            ref.q_rate[n] = q_rate[i, j] * volume
        end
    end

    # 2. The density and the gradient of v_* at the centre of each slot: from v_* = -g/∂_z P_*, ∂_z v_* = g ∂²_z P_*/(∂_z P_*)²
    (; slot_means) = ref
    for k in 1:K
        _, ∂P, ∂²P = hermite(ref.pressure, ref.slot_height[k])
        slot_means.density[k] = 1 / ref.slot_volume[k]
        slot_means.volume_gradient[k] = g * ∂²P / ∂P^2
    end

    # 3. ⟨v̇⟩_k and ⟨∂v/∂p⟩_k at P_k, ∂ₜP_top and ∂ₜv_*,k
    ref.plain_weight_slots = slot_means!(slot_means, ref.placement, ref.slot_mass, ref.slot_pressure, fluid, g, ref.consvar,
                                         ref.q, ref.consvar_rate, ref.q_rate)
    ref.p_top_tendency = -sum(k -> ref.slot_mass[k] * slot_means.volume_rate[k], 1:K) /
                          sum(k -> ref.slot_mass[k] * slot_means.compressibility[k], 1:K)
    for k in 1:K
        ref.slot_volume_tendency[k] = slot_means.volume_rate[k] + slot_means.compressibility[k] * ref.p_top_tendency
    end

    # 4. The nodes of P_* as dual numbers, value + ε × tendency: edge e, below slot K + 1 - e, moves at ∂ₜz_e, the sum of the
    # changes of thickness M_k ∂ₜv_*,k/L_x of the slots below it; its pressure P_top + g (mass above)/L_x changes at
    # ∂ₜP_top; the secant -g/v_*,k of the slot above it changes at g ∂ₜv_*,k/v_*,k². The slopes follow by `monotone_slopes!`
    (; edge_height_tendency, dual_pressure, dual_secant) = ref
    edge_height_tendency[1] = 0
    for e in 1:K
        k = K + 1 - e                                               # the slot above edge e
        edge_height_tendency[e+1] = edge_height_tendency[e] + ref.slot_mass[k] * ref.slot_volume_tendency[k] / Lx
        dual_secant[e] = Dual{ReferenceTendencyTag}(ref.secant[e], g * ref.slot_volume_tendency[k] / ref.slot_volume[k]^2)
    end
    for e in 1:K+1
        dual_pressure.z[e] = Dual{ReferenceTendencyTag}(ref.pressure.z[e], edge_height_tendency[e])
        dual_pressure.y[e] = Dual{ReferenceTendencyTag}(ref.pressure.y[e], ref.p_top_tendency)
    end
    monotone_slopes!(dual_pressure.d, dual_pressure.z, dual_secant)

    # 5. ∂ₜP_* at fixed height at the cell centres of the z axis
    for j in eachindex(ref.p_star_tendency)
        ref.p_star_tendency[j] = reference_pressure_tendency(ref, model, cellpoint(domain, dz, 2, j))
    end
    return ref
end

"""
    ∂ₜP = reference_pressure_tendency(ref::CompressibleLorenzReference, model::FC2D, z)

∂ₜP_*(z) [Pa s⁻¹] at the fixed height z [m], after `reference_tendency!`: the partial of the interpolant of dual numbers
`ref.dual_pressure` at z, the derivative of the monotone cubic P_* with respect to the motion of its nodes. At an edge it
is ∂ₜP_top + g ρ_*(z_e) ∂ₜz_e, with ρ_* = -∂_z P_*/g.
"""
@inline reference_pressure_tendency(ref::CompressibleLorenzReference, model::FC2D, z) = partials(hermite(ref.dual_pressure, z)[1], 1)
