## METHODS OF THE LORENZ REFERENCE STATE
#
# The Lorenz reference state places the parcels of a snapshot (the interior cells, each keeping its mass m_n and its
# conserved ϑ_n, q_n) into a column of K slots of masses M_k, numbered from the top, so that the potential energy
# Σ_nk x_nk c_nk is least, with x_nk the mass of parcel n in slot k and c_nk its energy per unit mass there [J kg⁻¹].
# The reference state of AN2D is in anelastic_reference.jl, that of FC2D in compressible_reference.jl.

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
the resolution as the passes do; a placement cut short by it fails its certificate. In FC2D, the costs depend on the top pressure P_top, which depends on the placement: the placement at
fixed P_top and P_top at fixed placement (`top_pressure!`) alternate, from the sorted placement, until the placement no
longer changes, at most `max_top_iterations` times.
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
    ref = rearrange!(ref, model)

The step of the reference state `ref` of `model` that depends on its method `ref.method` (an `AbstractRearrangement`):
the placement of the parcels into the slots and its dual potentials, ψ_k of the slots and u_n of the parcels [J kg⁻¹].
"""
function rearrange! end

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
    total = parcel_integral(model, ref, X)

Σ_n m_n X_n [J m⁻¹ for X in J kg⁻¹], the integral of ρ X over the interior cells, with m_n the mass of parcel n of the
reference state `ref` (ρ = ρ_R for AN2D) and X a field on the cells.
"""
function parcel_integral(model, ref, X)
    (; domain) = model
    total = zero(eltype(X))
    for j in interior_cells(domain, 2), i in interior_cells(domain, 1)
        total += ref.mass[parcel_index(domain, i, j)] * X[i, j]
    end
    return total
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
