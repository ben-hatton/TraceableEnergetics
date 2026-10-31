## THE TRANSPORTATION SIMPLEX
#
# The least-cost placement of N parcels of masses m_n into K slots of masses M_k (Σ_n m_n = Σ_k M_k),
#     minimise Σ_nk x_nk c_nk   subject to   Σ_k x_nk = m_n for every parcel,   Σ_n x_nk = M_k for every slot,   x ≥ 0,
# by the simplex method on the network of N + K nodes, the parcels 1..N and the slots N+1..N+K, with an arc from each
# parcel to each slot that carries the mass x_nk. A basis is a spanning tree of N + K - 1 arcs, the entries of a
# `Placement`, some of which may carry no mass (a degenerate basis). Its dual potentials, u_n of the parcels and ψ_k of
# the slots, solve u_n + ψ_k = c_nk on its arcs, with ψ_1 = 0. A pair (n, k) outside the tree with a negative reduced cost
# c_nk - u_n - ψ_k enters it: it closes a cycle in the tree, around which a mass θ moves, +θ on the arcs oriented as the
# entering one and -θ on the others, as much as keeps the masses nonnegative, and an arc emptied by it leaves the tree.
# When no pair has a negative reduced cost, the placement is optimal.
#
# The tree is rooted at the first slot and kept strongly feasible (Cunningham 1976): a positive mass can be sent from
# every node to the root along the tree, that is, every arc of zero mass goes from a parcel up to its parent slot. Then
# a pivot that moves no mass (θ = 0) still makes progress, and the method cannot cycle. The rule that keeps it so: the
# leaving arc is the last of the arcs that block θ met going round the cycle from its apex (the node of the cycle
# nearest the root) in the direction of the entering arc.

"""
    tree = TransportationTree(F, N, K)

Workspace of the transportation simplex (`tree_duals!`, `pivot!`) for N parcels and K slots, allocated once: the
number N of parcels, the cost c_nk [J kg⁻¹] of each arc of the tree (the entries of the `Placement`), and the tree rooted
at the first slot (node N + 1), as the parent, the arc to the parent, the depth and the children (a list of siblings) of
each node (nodes 1..N are the parcels, N+1..N+K the slots), the arcs at each node and the queue of a walk of the tree;
the candidates of a pass of the simplex and the scratch space of their sort; and the number of pivots made and of pivots
that moved mass.
"""
mutable struct TransportationTree{F}
    const parcels::Int                  # N
    const entry_cost::Vector{F}         # c_nk of each entry of the placement
    const parent::Vector{Int}           # parent node of each node, 0 at the root
    const parent_entry::Vector{Int}     # the entry joining each node to its parent
    const depth::Vector{Int}            # number of arcs from each node to the root
    const first_child::Vector{Int}      # the first child of each node, 0 for none
    const next_sibling::Vector{Int}     # the next and the previous child of the parent of each node, 0 for none
    const previous_sibling::Vector{Int}
    const adjacency_start::Vector{Int}  # the arcs at node a are adjacency[adjacency_start[a]:adjacency_start[a+1]-1]
    const adjacency::Vector{Int}        # the entries at each node, node by node
    const queue::Vector{Int}            # the nodes in the order of a walk of the tree
    const candidates::Vector{Int}       # the parcels of negative reduced cost in a pass, the most negative first
    const candidate_scratch::Vector{Int}
    pivots::Int                         # pivots made
    mass_pivots::Int                    # pivots that moved mass (θ > 0)
end
TransportationTree(F::Type, N, K) =
    TransportationTree(N, zeros(F, N + K), zeros(Int, N + K), zeros(Int, N + K), zeros(Int, N + K), zeros(Int, N + K),
                       zeros(Int, N + K), zeros(Int, N + K), zeros(Int, N + K + 1), zeros(Int, 2 * (N + K)), zeros(Int, N + K),
                       zeros(Int, N), zeros(Int, N), 0, 0)

"""
    attach!(tree, c, a)
    detach!(tree, c)

Makes node c the first child of node a in the lists of children of `tree`, or removes c from the children of its parent
`tree.parent[c]`; the parents themselves are left to the caller.
"""
function attach!(tree::TransportationTree, c, a)
    (; first_child, next_sibling, previous_sibling) = tree
    next = first_child[a]
    next_sibling[c], previous_sibling[c] = next, 0
    next == 0 || (previous_sibling[next] = c)
    first_child[a] = c
    return tree
end
function detach!(tree::TransportationTree, c)
    (; parent, first_child, next_sibling, previous_sibling) = tree
    previous, next = previous_sibling[c], next_sibling[c]
    previous == 0 ? (first_child[parent[c]] = next) : (next_sibling[previous] = next)
    next == 0 || (previous_sibling[next] = previous)
    return tree
end

"""
    placement = complete_staircase!(placement, order)

Completes a placement filled in order (`sorted_placement!`), whose entries form a staircase, into a spanning tree of the
parcels and the slots, with N + K - 1 entries: at a boundary between slots k and k + 1 across which no parcel is split
(a degenerate boundary), the first parcel b of slot k + 1 gets an entry of zero mass in slot k. Rooted at the first
slot, the tree is then strongly feasible: the arc of zero mass goes from b up to slot k, its parent, as slot k + 1 hangs
from b. The entries of `placement` beyond those of the staircase are no longer in the order of the parcels.
"""
function complete_staircase!(placement::Placement{F}, order) where F
    (; parcel, slot, mass) = placement
    count = placement.count
    for e in 2:placement.count
        k = slot[e-1]
        # a new slot starts at entry e with a new parcel b: the boundary between k and k + 1 is degenerate
        if slot[e] != k && parcel[e] != parcel[e-1]
            count += 1
            parcel[count], slot[count], mass[count] = parcel[e], k, zero(F)
        end
    end
    placement.count = count
    return placement
end

"""
    (ψ, u) = tree_duals!(ψ, u, placement, tree)

Builds the tree of the entries of `placement`, rooted at the first slot, by a breadth-first walk: the parent, the arc to
the parent, the depth and the children of each node of `tree`, and along the way the dual potentials, ψ_k of the slots and u_n of the
parcels [J kg⁻¹], from ψ_1 = 0 and u_n + ψ_k = c_nk on the arcs, with the costs `tree.entry_cost`. Costs O(N + K), with
no evaluation of costs.
"""
function tree_duals!(ψ, u, placement::Placement, tree::TransportationTree)
    (; parcel, slot) = placement
    (; entry_cost, parent, parent_entry, depth, first_child, adjacency_start, adjacency, queue) = tree
    N, K = tree.parcels, length(ψ)
    E = placement.count
    # the arcs at each node, by counting: node n of a parcel, node N + k of a slot
    fill!(adjacency_start, 0)
    for e in 1:E
        adjacency_start[parcel[e]+1] += 1
        adjacency_start[N+slot[e]+1] += 1
    end
    adjacency_start[1] = 1
    for a in 1:N+K
        adjacency_start[a+1] += adjacency_start[a]
    end
    # fill the arcs of each node, using `queue` as the next free place of each node
    for a in 1:N+K
        queue[a] = adjacency_start[a]
    end
    for e in 1:E
        a, b = parcel[e], N + slot[e]
        adjacency[queue[a]] = e
        queue[a] += 1
        adjacency[queue[b]] = e
        queue[b] += 1
    end
    # the breadth-first walk from the root, the first slot
    root = N + 1
    fill!(parent, -1)                   # -1: not reached yet
    fill!(first_child, 0)
    parent[root], parent_entry[root], depth[root] = 0, 0, 0
    ψ[1] = 0
    queue[1] = root
    head, tail = 1, 1
    while head <= tail
        a = queue[head]
        head += 1
        for i in adjacency_start[a]:adjacency_start[a+1]-1
            e = adjacency[i]
            b = a > N ? parcel[e] : N + slot[e]         # the node at the other end of arc e
            parent[b] == -1 || continue
            parent[b], parent_entry[b], depth[b] = a, e, depth[a] + 1
            attach!(tree, b, a)
            # u_n + ψ_k = c_nk on arc e
            if b > N
                ψ[b-N] = entry_cost[e] - u[parcel[e]]
            else
                u[b] = entry_cost[e] - ψ[slot[e]]
            end
            tail += 1
            queue[tail] = b
        end
    end
    tail == N + K || error("the entries of the placement do not span the parcels and the slots")
    return ψ, u
end

"""
    θ = pivot!(placement, tree, ψ, u, n, k, c_nk, r_nk)

Brings the pair of parcel n and slot k, of cost c_nk and negative reduced cost r_nk = c_nk - u_n - ψ_k [J kg⁻¹], into
the tree of `placement` (built by `tree_duals!`), and updates the tree and the duals ψ, u. The cycle it closes runs from
parcel n up the tree to the apex, the deepest common ancestor of n and slot k, and down to slot k. Going round it from
the apex down to n, across the new arc to k and up to the apex, the arcs oriented from parcel to slot in that direction
gain the mass θ and the others lose it, θ being the least mass of those that lose it. The leaving arc is the last of
these with mass θ met in that order, which keeps the tree strongly feasible; its entry takes (n, k, θ, c_nk).

Without the leaving arc, the subtree below it hangs from the rest of the tree by the new arc instead: it is rerooted
at the end q of the new arc that it holds (n, or slot k), and its depths and duals change, at a cost of the size of the
subtree rather than of the tree. The duals of the rest of the tree are unchanged, so that u_n + ψ_k = c_nk on the new
arc holds when the duals of the subtree shift by r_nk: u + r_nk for its parcels and ψ - r_nk for its slots if q = n,
u - r_nk and ψ + r_nk if q is slot k, which keeps u + ψ = c on the arcs within it. Returns θ [kg m⁻¹], zero for a
degenerate pivot.
"""
function pivot!(placement::Placement{F}, tree::TransportationTree, ψ, u, n, k, c_nk, r_nk) where F
    (; mass) = placement
    (; parent, parent_entry, depth, first_child, next_sibling, queue) = tree
    N = tree.parcels
    # 1. the apex, and θ. Going round the cycle, an arc of the parcel side (from the apex down to n) is crossed from
    # parent to child, so it loses mass when the child is a parcel (the arc goes from the child up to its parent slot,
    # against the direction of the round); an arc of the slot side (from k up to the apex) is crossed from child to
    # parent, so it loses mass when the child is a slot
    a, b = n, N + k                     # the walks up the parcel side and the slot side
    θ = typemax(F)
    while a != b
        if depth[a] >= depth[b]
            a <= N && (θ = min(θ, mass[parent_entry[a]]))
            a = parent[a]
        else
            b > N && (θ = min(θ, mass[parent_entry[b]]))
            b = parent[b]
        end
    end
    apex = a
    # 2. the leaving arc: the last arc with mass θ that loses mass, in the order of the round. On the slot side, that is
    # the one nearest the apex; if there is none, on the parcel side, the one nearest n. The subtree below it holds the
    # end q of the new arc on that side, and the other end o stays in the rest of the tree
    leaving, below = 0, 0               # the leaving arc and the node below it
    c = N + k
    while c != apex
        e = parent_entry[c]
        c > N && mass[e] == θ && ((leaving, below) = (e, c))
        c = parent[c]
    end
    q, o = N + k, n
    if leaving == 0
        q, o = n, N + k
        c = n
        while leaving == 0 && c != apex
            e = parent_entry[c]
            c <= N && mass[e] == θ && ((leaving, below) = (e, c))
            c = parent[c]
        end
    end
    # 3. θ moves round the cycle: x - θ ≥ 0 exactly for x ≥ θ, and the leaving arc is emptied exactly
    c = n
    while c != apex
        e = parent_entry[c]
        mass[e] = c <= N ? mass[e] - θ : mass[e] + θ
        c = parent[c]
    end
    c = N + k
    while c != apex
        e = parent_entry[c]
        mass[e] = c > N ? mass[e] - θ : mass[e] + θ
        c = parent[c]
    end
    # 4. the entering pair takes the place of the leaving arc
    placement.parcel[leaving], placement.slot[leaving], mass[leaving] = n, k, θ
    tree.entry_cost[leaving] = c_nk
    # 5. the subtree below the leaving arc, rerooted at q: along the path from q up to `below`, each node becomes the
    # parent of its former parent, by the arc that joined them, and q hangs from o by the new arc
    c, new_parent, new_entry = q, o, leaving
    while true
        old_parent, old_entry = parent[c], parent_entry[c]
        detach!(tree, c)
        parent[c], parent_entry[c] = new_parent, new_entry
        attach!(tree, c, new_parent)
        c == below && break
        c, new_parent, new_entry = old_parent, c, old_entry
    end
    # 6. the depths and the duals of the subtree, by a walk from q: its parcels shift by δu and its slots by -δu
    δu = q <= N ? r_nk : -r_nk
    queue[1], tail = q, 1
    while tail > 0
        a = queue[tail]
        tail -= 1
        depth[a] = depth[parent[a]] + 1
        a <= N ? (u[a] += δu) : (ψ[a-N] -= δu)
        c = first_child[a]
        while c != 0
            tail += 1
            queue[tail] = c
            c = next_sibling[c]
        end
    end
    tree.pivots += 1
    tree.mass_pivots += θ > 0
    return θ
end

"""
    (passes, decrease) = optimal_placement!(placement, tree, ψ, u, reduced_cost, best_slot, cost, price!, tolerance, max_passes)

The least-cost placement of the parcels into the slots by the transportation simplex, from the spanning tree of the
entries of `placement` (strongly feasible, rooted at the first slot, e.g. `complete_staircase!`), with `cost(n, k)` the
cost c_nk [J kg⁻¹] of parcel n in slot k. In each pass:
1. the duals `u` and `ψ` of the tree are computed from the costs of its arcs (`tree_duals!`), and `price!()` fills
   `reduced_cost[n]`, the least reduced cost c_nk - u_n - ψ_k of parcel n over the slots, and `best_slot[n]`, the slot
   where it is least (an integer, or a float of integer value), with these duals (a pass over all N K pairs);
2. if no reduced cost is below -`tolerance`, the placement is optimal, certified by the duals, and the passes stop;
3. otherwise each parcel n of the pass with reduced_cost[n] < -tolerance, the most negative first (sorted once, the
   lower index first among equal reduced costs), is a candidate: the pair (n, best_slot[n]) is priced again with the
   current duals, and enters the tree (`pivot!`, which updates the tree and the duals) if its reduced cost is still
   below -tolerance.
After `max_passes` passes with violations, it stops, leaving them in `reduced_cost`. Returns the number of passes and
the decrease of the cost Σ_nk x_nk c_nk from the starting placement [J m⁻¹ for masses in kg m⁻¹], the sum over the pivots
of -θ (c_nk - u_n - ψ_k), which is exact to the round-off of the reduced costs rather than of the total cost.
"""
function optimal_placement!(placement::Placement{F}, tree::TransportationTree, ψ, u, reduced_cost, best_slot,
                            cost::Cost, price!::Price, tolerance, max_passes) where {F, Cost, Price}
    (; candidates, candidate_scratch) = tree
    for e in 1:placement.count
        tree.entry_cost[e] = cost(placement.parcel[e], placement.slot[e])
    end
    tree.pivots, tree.mass_pivots = 0, 0
    decrease = zero(F)
    passes = 0
    while true
        # the duals from the costs of the arcs, rather than from the shifts of the pivots, so that the certificate has
        # the round-off of the costs only
        tree_duals!(ψ, u, placement, tree)
        price!()
        passes += 1
        count = 0
        for n in eachindex(reduced_cost)
            reduced_cost[n] < -tolerance && (candidates[count+=1] = n)
        end
        count == 0 && break
        passes == max_passes && break
        # the candidates, the most negative first: the sort is stable, so the lower index comes first among equals
        pass = view(candidates, 1:count)
        sort!(pass; by = n -> reduced_cost[n], scratch = candidate_scratch)
        for n in pass
            k = Int(best_slot[n])
            c = cost(n, k)
            r = c - u[n] - ψ[k]                 # with the duals of the current tree
            r < -tolerance || continue
            θ = pivot!(placement, tree, ψ, u, n, k, c, r)
            decrease -= θ * r
        end
    end
    return passes, decrease
end
