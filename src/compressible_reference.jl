## LORENZ REFERENCE STATE OF FC2D
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
# # duals, and the geopotential of the column, Σ_k M_k g z_k, does not depend on the placement once P_top is fixed. Slot k
# has the thickness V_k/L_x, with V_k = Σ_n x_nk v(P_k, ϑ_n, q_n) its volume, stacked from the bottom, and the reference
# pressure P_*(z) is the monotone cubic through the pressures of the slot edges, P_top + g/L_x times the mass above the
# edge.
# The reference height z_* of each parcel, the minimiser of its static energy along this profile, is computed here too
# (`reference_heights!`); the available energy of the parcels, in compressible_available_energy.jl. Heights z are
# measured from the bottom of the box; masses are per unit length along y [kg m⁻¹].

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
- the smallest reduced cost of each parcel, min_k (c_nk - u_n - ψ_k) [J kg⁻¹] (`reduced_cost`), and the number of
  parcels for which it is negative (`certificate_violations`): zero if the placement is optimal for this P_top.
"""
mutable struct CompressibleLorenzReference{F, M <: AbstractRearrangement}
    const method::M
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
    p_top::F                            # P_top [Pa]
    volume_residual::F                  # (Σ_k V_k/L_x - L_z)/L_z
    certificate_violations::Int
end

function CompressibleLorenzReference(model::FC2D{F}, method::AbstractRearrangement = SortedRearrangement()) where F
    (; domain, dz) = model
    Mx, Mz = dims(domain)
    Hz = halo_size(domain)[2]
    N, K = Mx * Mz, Mz
    nz = ncells(domain, 2)
    vector() = zeros(F, N)
    ref = CompressibleLorenzReference(method, vector(), vector(), vector(), vector(), vector(), vector(), zeros(Int, N),
                                      vector(), collect(1:N), zeros(Int, N), Placement(F, N, K), vector(), vector(),
                                      zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K), zeros(F, K),
                                      HermiteProfile(F, K + 1), zeros(F, K), zeros(F, K + 1), zeros(F, K + 1),
                                      zeros(F, nz), zeros(F, nz), zero(F), zero(F), 0)
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

    # 6. The certificate: the smallest reduced cost c_nk - u_n - ψ_k of each parcel over the slots, with
    # c_nk = h(P_k, ϑ_n, q_n) - h(p_n, ϑ_n, q_n), by the model's manager (vectorised over the parcels of a row)
    (; u, slot_pressure, slot_dual, reduced_cost) = ref
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            n = parcel_index(domain, i, j)
            ϑ_n, q_n, u_n, p_n = consvar[n], q[n], u[n], parcel_pressure[n]
            reduced = enthalpy_difference(fluid, p_n, slot_pressure[1] - p_n, ϑ_n, q_n) - u_n - slot_dual[1]
            for k in 2:K
                c = enthalpy_difference(fluid, p_n, slot_pressure[k] - p_n, ϑ_n, q_n)
                reduced = min(reduced, c - u_n - slot_dual[k])
            end
            reduced_cost[n] = reduced
        end
    end
    # the parcels with a negative reduced cost, beyond the round-off of the duals, which accumulate K differences of
    # costs of size up to g L_z
    tolerance = 16 * (K + 1) * eps(F) * g * Lz
    ref.certificate_violations = count(n -> reduced_cost[n] < -tolerance, eachindex(reduced_cost))
    return ref
end

"""
    ref = rearrange!(ref::CompressibleLorenzReference{<:Any, SortedRearrangement}, model::FC2D)

The sorted rearrangement of FC2D: the parcels, sorted by their specific volume v(p̄, ϑ_n, q_n) at the mean pressure p̄ of
the parcels [m³ kg⁻¹], lightest first, fill the slots from the top (`sorted_placement!`); the top pressure
P_top of the column follows from its volume (`top_pressure!`), and the duals ψ_k and u_n [J kg⁻¹] follow along the
staircase of the placement (`staircase_duals!`) with the costs at the slot pressures P_k.
"""
function rearrange!(ref::CompressibleLorenzReference{<:Any, SortedRearrangement}, model::FC2D)
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
    top_pressure!(ref, model)
    staircase_duals!(ref.slot_dual, ref.u, ref.placement, ref.order, (n, k) -> compressible_cost(model, ref, n, k))
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

## Reference heights
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
