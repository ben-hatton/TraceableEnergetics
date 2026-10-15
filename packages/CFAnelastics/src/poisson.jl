## POISSON SOLVER FOR THE ANELASTIC PRESSURE
#
# Given momentum tendencies dU = (dU, dW) in 2D or (dU, dV, dW) in 3D on the faces of a C-grid, find the modified
# pressure φ = (p - p_R)/ρ_R at cell centres such that the corrected tendencies dU - ∇φ satisfy the anelastic
# constraint
#       ∇·(ρ_R (dU - ∇φ)) = 0          on every interior cell,
#       (dU - ∇φ)·n = 0                on every wall face (impermeable walls),
# with φ periodic in periodic directions. Discretely, with δ_d the difference across a cell along dimension d, Δ_d
# the spacing and dU_d the component of dU along d (d as in the code: dimensions are numbered d = 1, …, N),
#       Σ_d δ_d(ρ_R ∂_d φ) / Δ_d = Σ_d δ_d(ρ_R dU_d) / Δ_d,
# where wall faces carry no flux on either side (the Neumann condition ∂φ/∂n = dU·n is satisfied exactly).
# ρ_R depends on z only: ρ_R at cell centres weights horizontal faces, its average over the two cells weights w-faces.
#
# Method: the horizontal operators are diagonalized by a real Fourier transform (periodic directions, FFTW R2HC)
# or a cosine transform (walls, FFTW REDFT10); each horizontal mode then solves a tridiagonal system in z, which
# is bounded by walls. The mean mode is singular (φ is defined up to a constant): it is pinned by φ = 0 in the
# bottom cell. Plans, eigenvalues and tridiagonal factors are computed once, at construction.

using FFTW: FFTW

"""
    solver = PoissonSolver(domain, spacing, ρ_R)

Poisson solver for the anelastic pressure on a `Box2D` or `Box3D`, with grid `spacing` (dx, dz) or (dx, dy, dz).
The vertical direction must be bounded; horizontal directions may be periodic or bounded, following the box
topology. `ρ_R` is the reference density at cell centres along the full z axis, halo included.
Apply with `pressure!`.
"""
struct PoissonSolver{N, F, Domain, Columns, Forward, Backward}
    domain::Domain
    spacing::NTuple{N, F}
    ρ_R::Vector{F}
    work::Array{F, N}       # values on interior cells, transformed in place
    columns::Columns        # work as a matrix (horizontal mode, z level)
    forward::Forward
    backward::Backward
    scale::F                # backward ∘ forward = scale × identity
    below::Vector{F}        # coupling of level j to j-1, ρ_R on the face / Δz² (zero at the bottom wall)
    upper::Matrix{F}        # Thomas factors per mode: modified coupling to j+1 ...
    inv_pivot::Matrix{F}    # ... and inverse pivots
end

function PoissonSolver(domain, spacing::NTuple{N, F}, ρ_R) where {N, F}
    M, H, topology = dims(domain), halo_size(domain), boundary_topology(domain)
    check_vertical(topology[N])
    horizontal = ntuple(identity, N - 1)

    # plans for one horizontal level, executed on every level (in parallel, see `solve!`)
    work, level = zeros(F, M...), zeros(F, M[1:N-1]...)
    flags = FFTW.ESTIMATE | FFTW.UNALIGNED
    forward = FFTW.plan_r2r!(level, map(d -> transform_kinds(topology[d])[1], horizontal), horizontal; flags)
    backward = FFTW.plan_r2r!(level, map(d -> transform_kinds(topology[d])[2], horizontal), horizontal; flags)
    scale = F(prod(d -> normalization(topology[d], M[d]), horizontal))

    # eigenvalues of the horizontal second differences, summed over horizontal dimensions (first mode: zero)
    eigenvalues(d) = reshape([horizontal_eigenvalue(topology[d], m, M[d], spacing[d]) for m in 0:M[d]-1], ntuple(e -> e == d ? M[d] : 1, N - 1))
    λ = vec(mapreduce(eigenvalues, (a, b) -> a .+ b, horizontal))

    # vertical couplings ρ_R / Δz² through the interior w-faces (ρ_R averaged to the face); walls carry no flux
    Mz, Hz, Δz = M[N], H[N], spacing[N]
    below = [j == 1 ? zero(F) : ℑ(ρ_R, (2,), (Face(),), Hz+j) / Δz^2 for j in 1:Mz]
    above = [j == Mz ? zero(F) : ℑ(ρ_R, (2,), (Face(),), Hz+j+1) / Δz^2 for j in 1:Mz]

    # Thomas factorization of  belowⱼ φⱼ₋₁ + (λ ρ_R(zⱼ) - belowⱼ - aboveⱼ) φⱼ + aboveⱼ φⱼ₊₁ = Rⱼ
    upper, inv_pivot = zeros(F, length(λ), Mz), zeros(F, length(λ), Mz)
    for mode in eachindex(λ), j in 1:Mz
        diagonal, coupling = λ[mode] * ρ_R[Hz+j] - below[j] - above[j], above[j]
        if mode == 1 && j == 1      # mean mode: replace the first equation by φ = 0 (pinning)
            diagonal, coupling = one(F), zero(F)
        end
        pivot = j == 1 ? diagonal : diagonal - below[j] * upper[mode, j-1]
        upper[mode, j], inv_pivot[mode, j] = coupling / pivot, inv(pivot)
    end
    columns = reshape(work, length(λ), Mz)
    return PoissonSolver(domain, spacing, collect(F, ρ_R), work, columns, forward, backward, scale,
                         below, upper, inv_pivot)
end

check_vertical(::Bounded) = nothing
check_vertical(::Periodic) = error("PoissonSolver: the vertical direction must be bounded.")

# horizontal transforms (forward, backward) and the normalization of their product: real Fourier transform in
# half-complex order for periodic directions, cosine transform (zero flux through the walls) for bounded ones
transform_kinds(::Periodic) = (FFTW.R2HC, FFTW.HC2R)
transform_kinds(::Bounded) = (FFTW.REDFT10, FFTW.REDFT01)
normalization(::Periodic, M) = M
normalization(::Bounded, M) = 2M

# eigenvalue of the second difference (φₘ₊₁ - 2φₘ + φₘ₋₁)/Δ² for mode m of M
horizontal_eigenvalue(::Periodic, m, M, Δ) = -(2 * sin(π * min(m, M - m) / M) / Δ)^2
horizontal_eigenvalue(::Bounded, m, M, Δ) = -(2 * sin(π * m / (2M)) / Δ)^2

"""
    pressure!(φ, solver::PoissonSolver, mgr, dU)

Solve for the modified pressure `φ` (cell centres, halo included) such that `dU - ∇φ` satisfies the discrete
anelastic constraint, `dU = (dU, dW)` or `(dU, dV, dW)` being the momentum tendencies on faces. Also fills the
halo of `φ`: periodic, or on walls such that ∂φ/∂n = dU·n on the wall face. Returns `φ`.
"""
function pressure!(φ, solver::PoissonSolver{N}, mgr, dU::NTuple{N}) where N
    divergence!(solver, dU)
    solve!(solver, mgr)
    (; work, domain, scale) = solver
    H = halo_size(domain)
    for I in CartesianIndices(work)
        φ[I + CartesianIndex(H)] = work[I] / scale
    end
    pressure_halo!(φ, solver, dU)
    return φ
end

# work = Σ_d δ_d(ρ_R dU_d) / Δ_d on interior cells, then without the fluxes through wall faces
function divergence!(solver::PoissonSolver{N}, dU) where N
    (; work, domain, spacing, ρ_R) = solver
    H = CartesianIndex(halo_size(domain))
    for I in CartesianIndices(work)
        cell = I + H
        work[I] = sum(ntuple(Val(N)) do d
            hi = cell + CartesianIndex(ntuple(==(d), Val(N)))
            ρ_lo, ρ_hi = face_densities(Val(d), Val(N), ρ_R, cell[N])
            (ρ_hi * dU[d][hi] - ρ_lo * dU[d][cell]) / spacing[d]
        end)
    end
    CFBoxes.foreach_dim(D -> remove_wall_fluxes!(CFBoxes.pick(boundary_topology(domain), D), D, solver, dU), dU)
end

# densities on the two faces of a cell at level k across dimension d: ρ_R averaged to the w-faces, ρ_R on horizontal
# faces
face_densities(::Val{N}, ::Val{N}, ρ_R, k) where N = (ℑ(ρ_R, (2,), (Face(),), k), ℑ(ρ_R, (2,), (Face(),), k+1))
face_densities(_, _, ρ_R, k) = (ρ_R[k], ρ_R[k])

# walls carry no flux: remove the wall-face terms from the cells next to the walls
remove_wall_fluxes!(::Periodic, D, solver, dU) = nothing
function remove_wall_fluxes!(::Bounded, D::Val{d}, (; work, domain, spacing, ρ_R)::PoissonSolver{N}, dU) where {d, N}
    M, H = dims(domain), halo_size(domain)
    for I in CFBoxes.slice(work, D)
        lo, hi = CFBoxes.across(I, D, 1), CFBoxes.across(I, D, M[d])
        cell_lo, cell_hi = lo + CartesianIndex(H), hi + CartesianIndex(H)
        ρ_lo = face_densities(D, Val(N), ρ_R, cell_lo[N])[1]
        ρ_hi = face_densities(D, Val(N), ρ_R, cell_hi[N])[2]
        work[lo] += ρ_lo * dU[d][cell_lo] / spacing[d]
        work[hi] -= ρ_hi * dU[d][cell_hi + CartesianIndex(ntuple(==(d), Val(N)))] / spacing[d]
    end
end

# transform horizontally, solve the tridiagonal system of each mode, transform back (times `scale`)
function solve!(solver::PoissonSolver, mgr)
    (; work, columns, forward, backward, below, upper, inv_pivot) = solver
    transform_levels!(forward, work, mgr)
    columns[1, 1] = 0       # pinned mean mode
    nmodes, Mz = size(columns)
    # Thomas sweeps up and down in z, all modes at once (inner loop over the contiguous mode index)
    @with mgr, let modes = 1:nmodes
        @inbounds @simd for mode in modes
            columns[mode, 1] *= inv_pivot[mode, 1]
        end
        for j in 2:Mz
            @inbounds @simd for mode in modes
                columns[mode, j] = (columns[mode, j] - below[j] * columns[mode, j-1]) * inv_pivot[mode, j]
            end
        end
        for j in Mz-1:-1:1
            @inbounds @simd for mode in modes
                columns[mode, j] -= upper[mode, j] * columns[mode, j+1]
            end
        end
    end
    transform_levels!(backward, work, mgr)
    return work
end

# apply a horizontal transform to each level of work (the last dimension), levels in parallel
function transform_levels!(plan, work::AbstractArray{F, N}, mgr) where {F, N}
    @with mgr, let levels = axes(work, N)
        for k in levels
            plan * selectdim(work, N, k)
        end
    end
    return work
end

# halo of φ: periodic, or on walls ghost values such that ∂φ/∂n = dU·n on the wall face
function pressure_halo!(φ, (; domain, spacing)::PoissonSolver, dU)
    CFBoxes.foreach_dim(domain) do D
        topology = CFBoxes.pick(boundary_topology(domain), D)
        CFBoxes.fill_halo!(topology, φ, D, domain, Cell(), neumann_rules(topology, D, domain, dU), CFBoxes.pick(spacing, D))
    end
    return φ
end

# ghost rule: φ_ghost = φ + sign × dist × (dU·n̂) on the wall face k, with n̂ the axis direction
struct Neumann{D, A}
    dU::A
    k::Int
    sign::Int
end
neumann_rules(::Periodic, D, domain, dU) = nothing
function neumann_rules(::Bounded, D::Val{d}, domain, dU) where d
    M, H = dims(domain)[d], halo_size(domain)[d]
    return (Neumann{d, typeof(dU[d])}(dU[d], H+1, -1), Neumann{d, typeof(dU[d])}(dU[d], H+M+1, 1))
end
@inline CFBoxes.ghost(n::Neumann{d}, v, I, dist) where d = v + n.sign * dist * n.dU[CFBoxes.across(I, Val(d), n.k)]
