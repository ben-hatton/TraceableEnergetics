## POISSON SOLVER FOR THE ANELASTIC PRESSURE
#
# Problem. Given the momentum tendencies dU = (du, dw) in 2D or (du, dv, dw) in 3D on the faces of a C-grid, find the
# modified pressure φ = (p - p_R)/ρ_R at the cell centres such that the corrected tendencies dU - ∇φ satisfy
#       ∇·(ρ_R (dU - ∇φ)) = 0               in the fluid (anelastic constraint),
#       (dU - ∇φ)·n = 0                     on the walls (impermeable),
# with φ periodic in periodic directions. Writing the wall condition as dU·n = 0 and ∂φ/∂n = 0 separates the two:
#       ∇·(ρ_R ∇φ) = ∇·(ρ_R dU)             with ∂φ/∂n = 0 on the walls.
# Discretely, with δ_d the difference across a cell along dimension d (d = 1, …, N, z last) and Δ_d the spacing,
#       Σ_d δ_d(ρ_d ∂_d φ) / Δ_d = Σ_d δ_d(ρ_d dU_d) / Δ_d,
# where ρ_d is ρ_R at the cell centre on horizontal faces and ρ_w = ℑ(ρ_R) (average of the two cells) on w-faces.
# ρ_R depends on z only, so the horizontal part of the operator is ρ_R(z) times a constant-coefficient Laplacian.
#
# Method, in the order of `pressure!`:
#   1. zero dU·n on the wall faces;
#   2. right-hand side r = ∇·(ρ dU) on the interior cells;
#   3. transform each level horizontally: Fourier transform along periodic directions, real-to-complex (FFTW r2c,
#      which keeps the M/2 + 1 non-negative frequencies of the first periodic direction); cosine transform (FFTW
#      REDFT10, i.e. DCT-II) along bounded ones, applied to the real and imaginary parts separately. These
#      diagonalise the horizontal second differences, periodic or with zero normal gradient, with eigenvalues λ ≤ 0
#      (`horizontal_eigenvalue`). Without a periodic direction, the coefficients are real cosine coefficients;
#   4. for each real number of the coefficients (a real or an imaginary part), solve the tridiagonal system in z
#      (walls at bottom and top) by the Thomas algorithm: the system is real, so both parts are solved alike;
#   5. transform back;
#   6. fill the halo of φ: periodic copy, or zero normal gradient on walls.
# The mean mode (λ = 0) is singular, as φ is defined up to a constant: its bottom equation is replaced by φ = 0, so
# that the horizontal mean of φ vanishes on the bottom level (the gauge). Everything that does not depend on dU (plans,
# eigenvalues, Thomas factors) is computed once, at construction.

using FFTW: FFTW
using LinearAlgebra: mul!

"""
    solver = PoissonSolver(domain, spacing, ρ_R; flags = FFTW.MEASURE)

Poisson solver for the anelastic pressure on a `Box2D` or `Box3D`, with grid `spacing` (dx, dz) or (dx, dy, dz) in m.
The vertical direction must be bounded; horizontal directions may be periodic or bounded, following the box
topology. `ρ_R` (kg m⁻³) is the reference density at cell centres along the full z axis, halo included. `flags` are
the FFTW planner flags: `FFTW.MEASURE` times candidate algorithms at construction and keeps the fastest, so results
may differ at round-off level between runs; `FFTW.ESTIMATE` plans at once, reproducibly, with slower transforms.
Apply with `pressure!`.

The tridiagonal system of a row of `modes_by_level` (horizontal eigenvalue λ) at level j = 1, …, Mz (interior levels)
reads

    a_j φ_{j-1} + b_j φ_j + c_j φ_{j+1} = r_j,
    a_j = ρ_w(j - 1/2) / Δz²  (0 at the bottom wall),   c_j = ρ_w(j + 1/2) / Δz²  (0 at the top wall),
    b_j = λ ρ_R(j) - a_j - c_j,

and is solved by the Thomas algorithm, with pivots p_1 = b_1, p_j = b_j - a_j c_{j-1} / p_{j-1}:

    forward elimination:   y_1 = r_1 / p_1,   y_j = (r_j - a_j y_{j-1}) / p_j,
    back substitution:     φ_Mz = y_Mz,       φ_j = y_j - (c_j / p_j) φ_{j+1}.

Fields:
- `domain`: the box
- `inv_spacing`: (1/Δx, 1/Δz) or (1/Δx, 1/Δy, 1/Δz)  [m⁻¹]
- `ρ_R`: reference density at cell centres, full z axis (halo included)  [kg m⁻³]
- `ρ_w`: reference density on w-faces, ℑ(ρ_R), full z face axis  [kg m⁻³]
- `rhs`: interior cells (Mx[, My], Mz): r/`scale` before the forward transform, φ after the backward one
- `modes`: the transform coefficients of each level, as real numbers, last dimension the level. With a periodic
  direction they are complex and stored with real and imaginary parts interleaved along the first dimension, of size
  (2 × first coefficient count, other coefficient counts…, Mz); otherwise (Mx[, My], Mz) cosine coefficients
- `spectral`: `modes` as the transforms see it: complex numbers (a reinterpretation of the same memory), or `modes`
  itself when no direction is periodic
- `modes_by_level`: the same memory as `modes`, seen as (row, level): each row, a real or imaginary part of one
  coefficient, is one tridiagonal system
- `forward_plan`: transform of one level, `rhs` → `spectral`: r2c along the periodic directions, or DCT-II along all
  directions when none is periodic
- `forward_cosine_plan`: in place on one level of `spectral`: DCT-II along the bounded directions, after the r2c;
  `nothing` when no direction is bounded or none is periodic
- `backward_cosine_plan`: in place, the inverse of `forward_cosine_plan` up to its normalisation (DCT-III), or `nothing`
- `backward_plan`: one level, `spectral` → `rhs`, the inverse of `forward_plan` up to its normalisation: c2r (which
  overwrites its input), or DCT-III
- `scale`: backward ∘ forward = `scale` × identity
- `sub_diagonal`: a_j per level  [kg m⁻⁵]
- `thomas_factor`: c_j / p_j per (row, level)
- `inv_pivot`: 1 / p_j per (row, level)
"""
struct PoissonSolver{N, F, Domain, Spectral, Forward, ForwardCosine, BackwardCosine, Backward}
    domain::Domain
    inv_spacing::NTuple{N, F}
    ρ_R::Vector{F}
    ρ_w::Vector{F}
    rhs::Array{F, N}
    modes::Array{F, N}
    spectral::Spectral
    modes_by_level::Matrix{F}
    forward_plan::Forward
    forward_cosine_plan::ForwardCosine
    backward_cosine_plan::BackwardCosine
    backward_plan::Backward
    scale::F
    sub_diagonal::Vector{F}
    thomas_factor::Matrix{F}
    inv_pivot::Matrix{F}
end

function PoissonSolver(domain, spacing::NTuple{N, F}, ρ_R; flags = FFTW.MEASURE) where {N, F}
    M, H, topology = dims(domain), halo_size(domain), boundary_topology(domain)
    check_vertical(topology[N])
    Mz, Hz, Δz = M[N], H[N], spacing[N]
    horizontal, Mh, Δh = Base.front(topology), Base.front(M), Base.front(spacing)

    # 3./5. the coefficients of a level, their storage, and the transforms of one level, applied to every level
    layout = spectral_layout(horizontal)
    Mc = coefficient_counts(layout, horizontal, Mh)
    modes, spectral = spectral_arrays(layout, F, Mc, Mz)
    forward_plan, forward_cosine_plan, backward_cosine_plan, backward_plan = horizontal_plans(layout, F, horizontal, Mh, Mc, flags)
    scale = F(prod(map(normalization, horizontal, Mh)))

    # eigenvalue λ of the horizontal second differences for each row of `modes_by_level`
    λ = horizontal_eigenvalues(layout, horizontal, Mh, Mc, Δh)

    # 4. the tridiagonal systems in z and their Thomas factors
    ρ_w = face_densities(ρ_R)
    a, c = vertical_coefficients(ρ_w, Mz, Hz, Δz)
    thomas_factor, inv_pivot = thomas_factors(λ, ρ_R[Hz+1:Hz+Mz], a, c)

    rhs = zeros(F, M...)
    modes_by_level = reshape(modes, length(λ), Mz)
    return PoissonSolver(domain, inv.(spacing), collect(F, ρ_R), ρ_w, rhs, modes, spectral, modes_by_level, forward_plan,
                         forward_cosine_plan, backward_cosine_plan, backward_plan, scale, a, thomas_factor, inv_pivot)
end

check_vertical(::Bounded) = nothing
check_vertical(::Periodic) = error("PoissonSolver: the vertical direction must be bounded.")

# the coefficients of a transformed level: real cosine coefficients when every horizontal direction is bounded,
# complex Fourier(-cosine) coefficients as soon as one is periodic
struct RealCoefficients end
struct ComplexCoefficients end
spectral_layout(::Tuple{Vararg{Bounded}}) = RealCoefficients()
spectral_layout(_) = ComplexCoefficients()

# number of coefficients along each horizontal direction: M, except M÷2 + 1 along the first periodic direction, whose
# negative frequencies the r2c transform leaves out (they are the complex conjugates)
coefficient_counts(::RealCoefficients, horizontal, M) = M
function coefficient_counts(::ComplexCoefficients, horizontal, M)
    halved = findfirst(t -> t isa Periodic, horizontal)
    return ntuple(d -> d == halved ? M[d] ÷ 2 + 1 : M[d], length(M))
end

# `modes` (real numbers, levels last) and its view `spectral` by the transforms
function spectral_arrays(::RealCoefficients, F, Mc, Mz)
    modes = zeros(F, Mc..., Mz)
    return modes, modes
end
function spectral_arrays(::ComplexCoefficients, F, Mc, Mz)
    modes = zeros(F, 2Mc[1], Base.tail(Mc)..., Mz)          # real and imaginary parts interleaved
    return modes, reinterpret(Complex{F}, modes)
end

# FFTW plans transforming one level: forward, forward cosine, backward cosine, backward (see `PoissonSolver`), made on
# fresh arrays, which planning overwrites, and applied to the levels of `rhs` and `spectral`
function horizontal_plans(::RealCoefficients, F, horizontal, M, Mc, flags)
    level, directions = zeros(F, M...), ntuple(identity, length(M))
    flags    = flags | alignment_flag(F, M)
    forward  = FFTW.plan_r2r(level, FFTW.REDFT10, directions; flags)
    backward = FFTW.plan_r2r(level, FFTW.REDFT01, directions; flags)
    return forward, nothing, nothing, backward
end
function horizontal_plans(::ComplexCoefficients, F, horizontal, M, Mc, flags)
    periodic = Tuple(d for d in eachindex(M) if horizontal[d] isa Periodic)
    bounded  = Tuple(d for d in eachindex(M) if horizontal[d] isa Bounded)
    forward  = FFTW.plan_rfft(zeros(F, M...), periodic; flags = flags | alignment_flag(F, M))
    backward = FFTW.plan_brfft(zeros(Complex{F}, Mc...), M[first(periodic)], periodic; flags = flags | alignment_flag(F, M))
    forward_cosine, backward_cosine = cosine_plans(zeros(Complex{F}, Mc...), bounded, flags)
    return forward, forward_cosine, backward_cosine, backward
end
cosine_plans(level, ::Tuple{}, flags) = (nothing, nothing)
cosine_plans(level, bounded, flags) = (FFTW.plan_r2r!(level, FFTW.REDFT10, bounded; flags),
                                       FFTW.plan_r2r!(level, FFTW.REDFT01, bounded; flags)) # complex levels: aligned

# FFTW's SIMD algorithms need the data to start on a 16-byte boundary, as the arrays the plans are made on do. Level k
# of an array starts (k - 1) × (bytes per level) after its first element: every level of `rhs` is aligned when a level
# of M real numbers fills a whole number of 16 bytes (every complex level is), otherwise the plans that read or write
# `rhs` must be UNALIGNED, about twice slower. FFTW checks the alignment whenever a plan is applied.
alignment_flag(F, M) = iszero(prod(M) * sizeof(F) % 16) ? 0 : FFTW.UNALIGNED

# factor of backward ∘ forward along one direction of M points: Fourier (periodic) or cosine (bounded) transforms
normalization(::Periodic, M) = M
normalization(::Bounded, M) = 2M

# eigenvalue of the second difference (φ_{m+1} - 2φ_m + φ_{m-1})/Δ² for the coefficient m = 0, 1, … along one
# direction of M points: frequency min(m, M-m) of the Fourier transform (periodic), or the cosine mode m (bounded)
horizontal_eigenvalue(::Periodic, m, M, Δ) = -(2 * sin(π * min(m, M - m) / M) / Δ)^2
horizontal_eigenvalue(::Bounded, m, M, Δ) = -(2 * sin(π * m / (2M)) / Δ)^2

# eigenvalue of each row of `modes_by_level`: for each coefficient (counts Mc), in memory order, the sum over
# horizontal directions; complex coefficients have two rows, their real and imaginary parts, with the same λ. The
# first coefficient (all m = 0) is the horizontal mean, with λ = 0.
horizontal_eigenvalues(::RealCoefficients, horizontal, M, Mc, spacing) = vec(coefficient_eigenvalues(horizontal, M, Mc, spacing))
horizontal_eigenvalues(::ComplexCoefficients, horizontal, M, Mc, spacing) =
    repeat(vec(coefficient_eigenvalues(horizontal, M, Mc, spacing)), inner = 2)
function coefficient_eigenvalues(horizontal, M, Mc, spacing)
    λ = zeros(eltype(spacing), Mc...)
    for I in CartesianIndices(λ), d in eachindex(M)
        λ[I] += horizontal_eigenvalue(horizontal[d], I[d] - 1, M[d], spacing[d])
    end
    return λ
end

# ρ_w on the faces J of the full z axis: ρ_w[J] = ℑ(ρ_R) between cells J-1 and J. The two outermost faces, beyond the
# halo, have no cell on one side and are left at zero (never read).
function face_densities(ρ_R)
    ρ_w = zeros(eltype(ρ_R), length(ρ_R) + 1)
    for J in 2:length(ρ_R)
        ρ_w[J] = ℑ(ρ_R, (2,), (Face(),), J)
    end
    return ρ_w
end

# couplings of interior level j to the levels below (a_j) and above (c_j): ρ_w / Δz² through the face between them,
# zero through the walls (no flux). Face Hz+j is the lower face of cell Hz+j.
function vertical_coefficients(ρ_w, Mz, Hz, Δz)
    a = [ρ_w[Hz+j] / Δz^2 for j in 1:Mz]
    c = [ρ_w[Hz+j+1] / Δz^2 for j in 1:Mz]
    a[1], c[Mz] = 0, 0
    return a, c
end

# Thomas factors c_j / p_j and 1 / p_j of every row (see `PoissonSolver`), with ρ the reference density of the
# interior levels. The rows of the mean (λ = 0: its real part, and its imaginary part, which is zero) are singular:
# their first equation becomes φ_1 = 0 (the gauge), by setting both factors of (row, level 1) to zero, so that y_1 = 0
# and level 2 sees a known φ_1.
function thomas_factors(λ, ρ, a, c)
    Mz = length(a)
    thomas_factor, inv_pivot = zeros(eltype(λ), length(λ), Mz), zeros(eltype(λ), length(λ), Mz)
    for row in eachindex(λ)
        for j in 1:Mz
            b = λ[row] * ρ[j] - a[j] - c[j]
            pivot = j == 1 ? b : b - a[j] * thomas_factor[row, j-1]
            thomas_factor[row, j], inv_pivot[row, j] = c[j] / pivot, inv(pivot)
            if iszero(λ[row]) && j == 1                  # mean: φ_1 = 0, before level 2 uses the factor
                thomas_factor[row, 1], inv_pivot[row, 1] = 0, 0
            end
        end
    end
    return thomas_factor, inv_pivot
end

"""
    pressure!(φ, solver::PoissonSolver, mgr, dU)

Solve for the modified pressure `φ` = (p - p_R)/ρ_R (m² s⁻², cell centres, halo included) such that `dU - ∇φ`
satisfies the discrete anelastic constraint ∇·(ρ_R (dU - ∇φ)) = 0 with no flow through walls, `dU = (du, dw)` or
`(du, dv, dw)` being the momentum tendencies on faces (m s⁻²), periodized in periodic directions. Sets the normal
components of `dU` on wall faces to zero. Fills the halo of `φ`: periodic, or zero normal gradient on walls. The
horizontal mean of `φ` on the bottom level is zero. Returns `φ`.
"""
function pressure!(φ, solver::PoissonSolver, mgr, dU)
    (; domain, rhs, scale) = solver
    # 1. impermeable walls: dU·n = 0 on the wall faces
    CFBoxes.foreach_dim(D -> zero_walls!(domain, CFBoxes.pick(dU, D), D), domain)
    # 2. r = ∇·(ρ dU) on the interior cells, written into rhs with 1/scale folded in
    divergence!(rhs, solver, mgr, dU, inv(scale))
    # 3.–5. transform, solve in z, transform back: rhs now holds φ
    solve!(solver, mgr)
    copy_to_interior!(φ, solver, mgr)
    # 6. halo
    fill_halo!(φ, domain, ZeroGradient())
    return φ
end

# r = ∇·(ρ dU) on the interior cells, in 2D and 3D: Σ_d δ_d(ρ_d dU_d) / Δ_d, ρ_R(z) on horizontal faces and ρ_w on w-faces,
# times `factor`, written into `rhs` (interior cells only: cell i of the domain is i - H in `rhs`, H the halo width).
# Wall faces carry dU·n = 0, hence no flux.
function divergence!(rhs, (; domain, inv_spacing, ρ_R, ρ_w)::PoissonSolver{2}, mgr, (du, dw), factor)
    inv_dx, inv_dz = inv_spacing
    Hx, Hz = halo_size(domain)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            rhs[i-Hx, j-Hz] = factor * (inv_dx * ρ_R[j] * δ(du, (1,), (Cell(),), i, j) + inv_dz * δ((ρ_w, dw), (2,), (Cell(),), i, j))
        end
    end
    return rhs
end
function divergence!(rhs, (; domain, inv_spacing, ρ_R, ρ_w)::PoissonSolver{3}, mgr, (du, dv, dw), factor)
    inv_dx, inv_dy, inv_dz = inv_spacing
    Hx, Hy, Hz = halo_size(domain)
    @with mgr, let (irange, jrange, krange) = (interior_cells(domain, 1), interior_cells(domain, 2), interior_cells(domain, 3))
        for k in krange, j in jrange
            @inbounds @vec for i in irange
                rhs[i-Hx, j-Hy, k-Hz] = factor * (inv_dx * ρ_R[k] * δ(du, (1,), (Cell(),), i, j, k) + inv_dy * ρ_R[k] * δ(dv, (2,), (Cell(),), i, j, k) +
                                                  inv_dz * δ((ρ_w, dw), (3,), (Cell(),), i, j, k))
            end
        end
    end
    return rhs
end

# φ = rhs on the interior cells of the halo-included φ, after `solve!`
function copy_to_interior!(φ, (; domain, rhs)::PoissonSolver{2}, mgr)
    Hx, Hz = halo_size(domain)
    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            φ[i, j] = rhs[i-Hx, j-Hz]
        end
    end
    return φ
end
function copy_to_interior!(φ, (; domain, rhs)::PoissonSolver{3}, mgr)
    Hx, Hy, Hz = halo_size(domain)
    @with mgr, let (irange, jrange, krange) = (interior_cells(domain, 1), interior_cells(domain, 2), interior_cells(domain, 3))
        for k in krange, j in jrange
            @inbounds @vec for i in irange
                φ[i, j, k] = rhs[i-Hx, j-Hy, k-Hz]
            end
        end
    end
    return φ
end

# Steps 3–5: transform the levels of `rhs` into `modes`, solve the tridiagonal system of each row (see
# `PoissonSolver`), transform back into `rhs`. The Thomas sweeps run over the levels j, all rows at once (contiguous
# inner loop).
function solve!(solver::PoissonSolver, mgr)
    (; modes_by_level, sub_diagonal, thomas_factor, inv_pivot) = solver
    a, y = sub_diagonal, modes_by_level
    nrows, Mz = size(y)

    # 3. horizontal transform of each level
    forward_transform!(solver, mgr)

    # 4. Thomas algorithm; y overwrites r, then φ overwrites y
    @with mgr, let rows = 1:nrows
        # forward elimination: y_1 = r_1 / p_1,   y_j = (r_j - a_j y_{j-1}) / p_j
        @inbounds @vec for row in rows
            y[row, 1] = y[row, 1] * inv_pivot[row, 1]
        end
        for j in 2:Mz
            @inbounds @vec for row in rows
                y[row, j] = (y[row, j] - a[j] * y[row, j-1]) * inv_pivot[row, j]
            end
        end
        # back substitution: φ_Mz = y_Mz,   φ_j = y_j - (c_j / p_j) φ_{j+1}
        for j in Mz-1:-1:1
            @inbounds @vec for row in rows
                y[row, j] = y[row, j] - thomas_factor[row, j] * y[row, j+1]
            end
        end
    end

    # 5. backward transform of each level
    backward_transform!(solver, mgr)
    return solver.rhs
end

# 3. each level k (the last dimension): `rhs` → `spectral`, then the cosine transforms of the bounded directions in
# place. Levels are contiguous in memory, so the one-level plans apply to them, and independent: they run in parallel
# under a threaded manager.
function forward_transform!((; rhs, spectral, forward_plan, forward_cosine_plan)::PoissonSolver{N}, mgr) where {N}
    @with mgr, let levels = axes(rhs, N)
        for k in levels
            coefficients = selectdim(spectral, N, k)
            mul!(coefficients, forward_plan, selectdim(rhs, N, k))
            transform_in_place!(forward_cosine_plan, coefficients)
        end
    end
    return spectral
end
# 5. each level k: the inverse cosine transforms in place, then `spectral` → `rhs` (overwriting `spectral`)
function backward_transform!((; rhs, spectral, backward_cosine_plan, backward_plan)::PoissonSolver{N}, mgr) where {N}
    @with mgr, let levels = axes(rhs, N)
        for k in levels
            coefficients = selectdim(spectral, N, k)
            transform_in_place!(backward_cosine_plan, coefficients)
            mul!(selectdim(rhs, N, k), backward_plan, coefficients)
        end
    end
    return rhs
end
transform_in_place!(::Nothing, level) = level
transform_in_place!(plan, level) = plan * level
