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
