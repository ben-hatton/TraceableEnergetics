# Tests of the transportation simplex (src/numerics/transportation.jl) on cost matrices, independent of the models.
# Included in the module of the reference-state tests, after test/reference_state.jl,
# whose brute force `least_assignment` and pseudo-random `noise` it uses.
using TraceableEnergetics: Placement, TransportationTree, sorted_placement!, complete_staircase!, optimal_placement!

# The least-cost placement of parcels of masses m into slots of masses M with the costs C[n, k], from the placement
# filled in the order `order`, with a pass over all the pairs as the pricing; returns the placement, the duals, the
# number of passes, the decrease of the cost reported by the simplex, the costs of the start and of the result, and the
# tree
function simplex(C::Matrix{F}, m, M; order = collect(1:size(C, 1)), tolerance = 64 * eps(F) * maximum(abs, C),
                 max_passes = 10_000) where F
    N, K = size(C)
    placement, tree = Placement(F, N, K), TransportationTree(F, N, K)
    sorted_placement!(placement, order, m, M)
    complete_staircase!(placement, order)
    ψ, u, reduced_cost, best_slot = zeros(F, K), zeros(F, N), zeros(F, N), zeros(Int, N)
    function price!()
        for n in 1:N
            reduced_cost[n], best_slot[n] = findmin(k -> C[n, k] - u[n] - ψ[k], 1:K)
        end
    end
    total() = sum(e -> placement.mass[e] * C[placement.parcel[e], placement.slot[e]], 1:placement.count)
    start = total()
    passes, decrease = optimal_placement!(placement, tree, ψ, u, reduced_cost, best_slot, (n, k) -> C[n, k], price!,
                                          tolerance, max_passes)
    return (; placement, ψ, u, passes, decrease, start, cost = total(), tree, tolerance)
end

# the placement as an N × K matrix
function placement_matrix(placement::Placement{F}, N, K) where F
    X = zeros(F, N, K)
    for e in 1:placement.count
        X[placement.parcel[e], placement.slot[e]] += placement.mass[e]
    end
    return X
end

# Optimality, checked independently of the simplex: the placement is feasible (nonnegative, the masses of every parcel
# and every slot) and its duals prove it optimal (every reduced cost nonnegative, and zero on the entries with a mass:
# then Σ x c = Σ m u + Σ M ψ, the dual objective, which bounds every feasible cost from below)
function check_optimal(C::Matrix{F}, m, M, result) where F
    (; placement, ψ, u, tolerance) = result
    N, K = size(C)
    X = placement_matrix(placement, N, K)
    scale = sum(m)
    @test placement.count == N + K - 1
    @test all(>=(0), X)
    @test maximum(abs, vec(sum(X; dims = 2)) .- m) <= 64 * eps(F) * scale
    @test maximum(abs, vec(sum(X; dims = 1)) .- M) <= 64 * eps(F) * scale
    @test minimum(C .- u .- ψ') >= -tolerance
    @test all(e -> abs(C[placement.parcel[e], placement.slot[e]] - u[placement.parcel[e]] - ψ[placement.slot[e]]) <= tolerance,
              1:placement.count)
    @test result.cost ≈ sum(m .* u) + sum(M .* ψ) rtol = 64 * eps(F) atol = 64 * eps(F) * scale * maximum(abs, C)
    # the decrease reported by the simplex, from its pivots, is that of the cost
    @test result.start - result.cost ≈ result.decrease atol = 64 * eps(F) * scale * maximum(abs, C)
end

# Random problems with unequal masses, from a placement in an arbitrary order
@testset "transportation simplex, random" begin
    for trial in 1:40
        N, K = 1 + trial % 11, 1 + trial % 5
        C = [noise(100trial + 10n + k) for n in 1:N, k in 1:K]
        m = [0.1 + noise(1000trial + n) for n in 1:N]
        M = [0.1 + noise(2000trial + k) for k in 1:K]
        M .*= sum(m) / sum(M)
        check_optimal(C, m, M, simplex(C, m, M))
    end
end

# Equal masses and as many slots as parcels: the linear assignment problem, whose least cost the brute force gives
@testset "transportation simplex, assignment" begin
    for trial in 1:30
        N = 1 + trial % 7
        C = [noise(300trial + 10n + k) for n in 1:N, k in 1:N]
        result = simplex(C, ones(N), ones(N))
        check_optimal(C, ones(N), ones(N), result)
        @test result.cost ≈ least_assignment(C) rtol = 1e-13
    end
end

# Degenerate problems: all costs equal (no pivot), one slot, one parcel, a start that is already optimal (costs
# c_nk = a_n b_k with a and b increasing, filled in order: no pivot moves mass), and integer costs with many ties from
# a start in the wrong order, highly degenerate, on which a simplex without its strongly feasible tree can cycle
@testset "transportation simplex, degenerate" begin
    result = simplex(ones(10, 4), ones(10), fill(2.5, 4))
    check_optimal(ones(10, 4), ones(10), fill(2.5, 4), result)
    @test result.tree.pivots == 0
    for (N, K) in ((5, 1), (1, 4))
        C = [noise(10n + k) for n in 1:N, k in 1:K]
        m, M = fill(K / N, N), ones(K)
        result = simplex(C, m, M)
        check_optimal(C, m, M, result)
        @test result.tree.pivots == 0
    end
    C = [n * k / 10 for n in 1:12, k in 1:4]
    result = simplex(C, ones(12), fill(3.0, 4); order = 12:-1:1)
    check_optimal(C, ones(12), fill(3.0, 4), result)
    @test result.tree.mass_pivots == 0
    C = [floor(4 * noise(10n + k)) for n in 1:200, k in 1:20]
    result = simplex(C, ones(200), fill(10.0, 20); order = reverse(1:200))
    check_optimal(C, ones(200), fill(10.0, 20), result)
    @test result.tree.pivots > result.tree.mass_pivots > 0
end

# The pass cap only stops the simplex: cut short after one pass, which only prices the starting placement, no pivot is
# made and the negative reduced costs stay
@testset "transportation simplex, cut short" begin
    C = [n * k / 10 for n in 1:12, k in 1:4]
    result = simplex(C, ones(12), fill(3.0, 4); max_passes = 1)
    @test result.passes == 1
    @test result.tree.pivots == 0
    @test result.decrease == 0
    @test minimum(C .- result.u .- result.ψ') < -result.tolerance
end
