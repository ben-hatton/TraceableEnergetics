## PLACEMENT OF MASSES BY THE NORTH-WEST CORNER RULE
#
# N parcels of masses m_n are placed into K slots of masses M_k (Σ_n m_n = Σ_k M_k): with x_nk ≥ 0 the mass of parcel n
# in slot k and c_nk its cost per unit mass, the least-cost placement solves
#     minimise Σ_nk x_nk c_nk   subject to   Σ_k x_nk = m_n for every parcel,   Σ_n x_nk = M_k for every slot.
# Its dual has a potential u_n per parcel and ψ_k per slot, and a placement is optimal if the reduced costs satisfy
#     c_nk - u_n - ψ_k ≥ 0 for every pair, = 0 wherever x_nk > 0.

"""
    placement = Placement(F, N, K)

A placement of N parcels into K slots, as the list of its positive entries (parcel n, slot k, mass x_nk), at most
N + K - 1 of them (`count`), in the order in which the parcels were placed: the entries of the parcel of rank r are
`first[r]:first[r+1]-1`. Never an N × K matrix.
"""
mutable struct Placement{F}
    const parcel::Vector{Int}       # parcel n of each entry
    const slot::Vector{Int}         # slot k of each entry
    const mass::Vector{F}           # mass x_nk of each entry
    const first::Vector{Int}        # first entry of the parcel of each rank, and one past the last entry
    count::Int                      # number of entries
end
Placement(F::Type, N, K) = Placement(zeros(Int, N + K), zeros(Int, N + K), zeros(F, N + K), zeros(Int, N + 1), 0)

"""
    placement = sorted_placement!(placement, order, m, M)

Fills the slots of masses `M`, in order, with the parcels of masses `m` in the given `order`: the north-west corner
rule. A parcel that does not fit in what remains of a slot fills it and goes on into the next ones. A slot whose
remaining mass is below the round-off of its sum (about one ulp of M_k per entry) is full, so that a parcel that ends
at a slot boundary is not split by round-off: the boundary is then degenerate (no parcel across it). The last slot
takes whatever mass remains.
"""
function sorted_placement!(placement::Placement{F}, order, m, M) where F
    (; parcel, slot, mass, first) = placement
    K = length(M)
    e, k = 0, 1                     # last entry, current slot
    room = M[1]                     # mass still free in slot k
    entries = 0                     # entries in slot k so far
    for (r, n) in enumerate(order)
        first[r] = e + 1
        left = m[n]                 # mass of parcel n still to place
        while true
            slack = 4 * (entries + 1) * eps(F) * M[k]   # round-off of the sum of the entries of slot k
            if left <= room + slack || k == K
                # the rest of the parcel fits in slot k
                e += 1
                parcel[e], slot[e], mass[e] = n, k, left
                room -= left
                entries += 1
                # slot k is full: the next parcel starts slot k + 1
                if room <= slack && k < K
                    k += 1
                    room, entries = M[k], 0
                end
                break
            else
                # the parcel fills slot k and goes on into slot k + 1
                e += 1
                parcel[e], slot[e], mass[e] = n, k, room
                left -= room
                k += 1
                room, entries = M[k], 0
            end
        end
    end
    first[length(order)+1] = e + 1
    placement.count = e
    return placement
end

"""
    (ψ, u) = staircase_duals!(ψ, u, placement, order, cost)

Dual potentials of a placement filled in order (`sorted_placement!`), whose entries form a staircase, from u_n + ψ_k =
c_nk on its entries, with ψ_1 = 0; `cost(n, k)` is c_nk. Walking down the staircase, the step of ψ across the boundary
between slots k and k + 1 is
- c_{a,k+1} - c_{a,k}, for the parcel a split across the boundary;
- if no parcel is split there (a degenerate boundary), the middle of the interval allowed by the last parcel a of slot k
  and the first parcel b of slot k + 1, ½[(c_{a,k+1} - c_{a,k}) + (c_{b,k+1} - c_{b,k})]: the choice is reproducible.
Then u_n = c_{n,k} - ψ_k for any slot k of parcel n.
"""
function staircase_duals!(ψ, u, placement, order, cost::Cost) where Cost
    (; parcel, slot, first) = placement
    ψ[1] = 0
    for e in 2:placement.count
        k = slot[e-1]
        slot[e] == k && continue                # same slot: no boundary
        a, b = parcel[e-1], parcel[e]           # the parcels on either side of the boundary
        step_a = cost(a, k + 1) - cost(a, k)
        ψ[k+1] = ψ[k] + (a == b ? step_a : (step_a + cost(b, k + 1) - cost(b, k)) / 2)
    end
    for (r, n) in enumerate(order)
        k = slot[first[r]]
        u[n] = cost(n, k) - ψ[k]
    end
    return ψ, u
end
