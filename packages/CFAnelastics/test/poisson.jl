#=
Poisson solver for the anelastic pressure: pressure!(φ, solver, mgr, dU) finds φ such that the corrected tendencies
dU - ∇φ satisfy ∇·(ρ_R (dU - ∇φ)) = 0, with no flow through walls. Checked in 2D (periodic and bounded in x) and 3D.
=#

using CFAnelastics: PoissonSolver, pressure!

# box, spacing, reference density ρ_R(z) = 1 + z/2 at cell centers, and face arrays for the tendencies
function poisson_setup(topology, M; L = map(_ -> 1.0, M))
    N = length(M)
    domain = N == 2 ? Box2D(Mx = M[1], Mz = M[2], boundary = topology) :
                      Box3D(Mx = M[1], My = M[2], Mz = M[3], boundary = topology)
    spacing = L ./ M
    H = halo_size(domain)
    ρ(z) = 1 + z / 2
    ρ_R = [ρ((k - H[N] - 0.5) * spacing[N]) for k in 1:2H[N]+M[N]]
    primal = map((m, h) -> m + 2h, M, H)
    dual = map((m, h, t) -> m + 2h + (t isa Bounded), M, H, boundary_topology(domain))
    faces(d) = zeros(ntuple(e -> e == d ? dual[e] : primal[e], N)...)
    return (; domain, spacing, ρ_R, faces = ntuple(faces, N), φ = zeros(primal...),
            solver = PoissonSolver(domain, spacing, ρ_R))
end

# coordinates of face index I along d (dual along d, primal elsewhere), or of a cell center (d = 0)
coords(setup, d, I) = ntuple(e -> (I[e] - halo_size(setup.domain)[e] - (e == d ? 1.0 : 0.5)) * setup.spacing[e], length(I))

# corrected tendency dU_d - ∂_d φ on every face of the interior cells (walls included)
function corrected(setup, dU)
    (; domain, spacing, φ) = setup
    N = length(spacing)
    M, H = dims(domain), halo_size(domain)
    return ntuple(N) do d
        e = CartesianIndex(ntuple(k -> k == d ? 1 : 0, N))
        faces = CartesianIndices(ntuple(k -> k == d ? (H[k]+1:H[k]+M[k]+1) : (H[k]+1:H[k]+M[k]), N))
        [dU[d][f] - (φ[f] - φ[f - e]) / spacing[d] for f in faces]
    end
end

# Σ_d δ_d(ρ_R c_d)/Δ_d on interior cells (ρ_R averaged to the w-faces), relative to the size of its terms
function relative_divergence(setup, c)
    (; ρ_R, spacing, domain) = setup
    N, H = length(spacing), halo_size(domain)
    div = zeros(size(c[N])[1:N-1]..., size(c[N], N) - 1)
    scale = 0.0
    for I in CartesianIndices(div), d in 1:N
        hi = I + CartesianIndex(ntuple(k -> k == d ? 1 : 0, N))
        k = I[N] + H[N]
        ρlo, ρhi = d == N ? (ℑ(ρ_R, (2,), (Face(),), k), ℑ(ρ_R, (2,), (Face(),), k+1)) : (ρ_R[k], ρ_R[k])
        div[I] += (ρhi * c[d][hi] - ρlo * c[d][I]) / spacing[d]
        scale = max(scale, abs(ρhi * c[d][hi]) / spacing[d])
    end
    return maximum(abs, div) / scale
end

# bytes allocated by a call, after a warm-up call that compiles it
allocations(f, args...) = (f(args...); @allocated f(args...))

# (topology, grid, check second-order convergence); odd and unequal sizes catch index mix-ups between directions
const poisson_cases = (((Periodic, Bounded), (16, 8), true), ((Bounded, Bounded), (12, 10), false),
                       ((Bounded, Bounded, Bounded), (6, 7, 5), true))

@testset "Poisson solver" begin
    for (topology, M, convergence) in poisson_cases
        @testset "$(length(M))D $(nameof.(topology))" begin
            #=
            random tendencies: the corrected tendencies satisfy the constraint, with no flow through walls (pressure!
            zeros the wall faces of dU)
            =#
            setup = poisson_setup(topology, M)
            dU = map(a -> rand!(a) .- 0.5, setup.faces)
            for d in eachindex(topology)       # tendencies are periodic in periodic directions, as in the models
                topology[d] === Periodic && foreach(a -> CFBoxes.periodize!(setup.domain, a), dU)
            end
            pressure!(setup.φ, setup.solver, mgr, dU)
            c = corrected(setup, dU)
            @test relative_divergence(setup, c) < 1e-12                          # ∇·(ρ_R c) = 0 to round-off
            for d in eachindex(topology)
                if topology[d] === Bounded
                    @test maximum(abs, selectdim(c[d], d, 1)) < 1e-12 * maximum(abs, c[d])               # lower wall
                    @test maximum(abs, selectdim(c[d], d, size(c[d], d))) < 1e-12 * maximum(abs, c[d])   # upper wall
                end
            end

            # manufactured solution: dU = ∇φ_exact gives back φ_exact (up to a constant), at second order
            φ_exact(x...) = cos(2π * x[1]) * (1 + x[end]^2) + (length(x) == 3 ? sin(2π * x[2]) * x[end] : 0.0) + x[end]^3
            ∂φ_exact(d, x...) = d == 1 ? -2π * sin(2π * x[1]) * (1 + x[end]^2) :
                                d == length(x) ? cos(2π * x[1]) * 2x[end] + (length(x) == 3 ? sin(2π * x[2]) : 0.0) + 3x[end]^2 :
                                2π * cos(2π * x[2]) * x[end]
            function error(m)                  # ‖φ - φ_exact‖∞ on an m grid, after removing the mean difference
                s = poisson_setup(topology, m)
                g = ntuple(d -> [∂φ_exact(d, coords(s, d, Tuple(I))...) for I in CartesianIndices(s.faces[d])], length(m))
                pressure!(s.φ, s.solver, mgr, g)
                cells = CartesianIndices(map((k, h) -> h+1:h+k, m, halo_size(s.domain)))
                diff = [s.φ[I] - φ_exact(coords(s, 0, Tuple(I))...) for I in cells]
                return maximum(abs, diff .- sum(diff) / length(diff))
            end
            if convergence
                e1, e2 = error(M), error(2 .* M)
                @test 3.5 < e1 / e2 < 4.5      # halving the spacing divides the error by ≈ 4
            end

            # no allocations (single-threaded manager)
            @test allocations(pressure!, setup.φ, setup.solver, VectorizedCPU(), dU) == 0
        end
    end
end
