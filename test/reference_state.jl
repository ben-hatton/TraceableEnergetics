# Tests of the Lorenz reference state of available potential energy and the available energy of its parcels, by
# sorting, with its certificate, for AN2D (src/anelastic_reference.jl, src/anelastic_available_energy.jl) and FC2D
# (src/compressible_reference.jl, src/compressible_available_energy.jl), on small grids. Included in a module after
# config.jl.
using CFAnelastics, CFCompressible, CookBooks
using TraceableEnergetics
using CFBoxes: Cell, cellpoint, ncells, interior_cells, dims, halo_size
using CFTimeSchemes: tendencies!
using TraceableEnergetics: interior, anelastic_cost, buoyancy_residual, stratification, reference_heights!,
                           available_compressible_energy!, available_potential_energy!, parcel_integral,
                           rearrangement_energy, available_compressible_energy, enthalpy_difference, hermite

## Models and states, generic in the float type F

# the fluid of config.jl with some parameters changed, in the float type F
reference_fluid(F; changes...) = NonlinearBinaryFluid(:potential_temperature, map(F, merge(fluid_params, NamedTuple(changes))))

tank(F; z0 = 0) = Tank2D(Lx = F(Lx), Lz = F(Lz), g = F(g), z0 = F(z0))

# AN2D in a box periodic in x, with the reference density ρ(z) (by default that of the AN2D test model, 10 % denser at
# the bottom) and the pressure 1e5 Pa at the origin z0 of the geopotential. Not in BigFloat: its Poisson solver uses FFTs.
function an_box(F; Mx = 8, Mz = 8, ρ = z -> ρ0 * (1 + 0.1 * (Lz - z) / Lz), fluid = reference_fluid(F), mgr = nothing, z0 = 0)
    domain = Box2D(; Mx, Mz)
    space = tank(F; z0)
    return AN2D((; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(),
        buoyancy_scheme = ThermodynamicForcing(), thermoflux_scheme = ThermoFluxConsistent(k_T = F(1e-4), k_q = F(1e-5)),
        viscosity_scheme = ViscosityScheme(dyn_visc = F(1e-3)),
        anelastic_reference = AnelasticReference(domain, space, z -> F(ρ(z)); pref = F(1e5)),
        boundary_conditions = BoxBoundaryConditions(domain; bottom = closed_wall(), top = closed_wall()), gauge_offset = F(0)))
end

# FC2D in the same box. Its fluid can be built in BigFloat
function fc_box(F; Mx = 8, Mz = 8, fluid = reference_fluid(F), mgr = nothing, z0 = 0)
    domain = Box2D(; Mx, Mz)
    return FC2D((; mgr, domain, space = tank(F; z0), fluid, spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(),
        thermoflux_scheme = ThermoFluxConsistent(k_T = F(1e-4), k_q = F(1e-5)), viscosity_scheme = ViscosityScheme(dyn_visc = F(1e-3)),
        boundary_conditions = BoxBoundaryConditions(domain; bottom = closed_wall(), top = closed_wall())))
end

# the state (only its densities m, which the reference state reads) with the tracers θ(x, z), q(x, z): AN2D at the
# reference density, FC2D at the density in hydrostatic balance in each column, with pressure 1e5 Pa at z = 0
function tracer_state(model::AN2D{F}, θ, q) where F
    (; domain, dx, dz, reference) = model
    m = zeros(F, ncells(domain, 1), ncells(domain, 2), 2)
    for i in axes(m, 1), j in axes(m, 2)
        x, z = cellpoint(domain, dx, 1, i), cellpoint(domain, dz, 2, j)
        m[i, j, 1], m[i, j, 2] = reference.ρ_R[j] * F(θ(x, z)), reference.ρ_R[j] * F(q(x, z))
    end
    return (; m)
end
function tracer_state(model::FC2D{F}, θ, q) where F
    (; domain, dx, dz) = model
    consvar(x, z) = F(θ(x, z))
    composition(x, z) = F(q(x, z))
    ρ = CFCompressible.balanced_density(model; consvar, q = composition, pref = F(1e5))
    m = zeros(F, ncells(domain, 1), ncells(domain, 2), 3)
    for i in axes(m, 1), j in axes(m, 2)
        x, z = cellpoint(domain, dx, 1, i), cellpoint(domain, dz, 2, j)
        m[i, j, 1], m[i, j, 2], m[i, j, 3] = ρ[i, j], ρ[i, j] * consvar(x, z), ρ[i, j] * composition(x, z)
    end
    return (; m)
end

resting(model) = tracer_state(model, (x, z) -> θ_rest(z), (x, z) -> q_rest(z))
active(model) = tracer_state(model, θ_active, q_active)

# the Lorenz reference state of a model
lorenz(model::AN2D) = AnelasticLorenzReference(model)
lorenz(model::FC2D) = CompressibleLorenzReference(model)

# an array on the cells of the model, halo included
cell_array(model) = zeros(typeof(model.dx), ncells(model.domain, 1), ncells(model.domain, 2))

# the reference state of `state` and the available energy of its parcels, as the diagnostics compute them: z_*, the
# available compressible energy ACE = Π₁, the available potential energy APE = Π₂ and the available energy AE = Π₁ + Π₂ on
# the cells, Σ m Π, and the potential energy released by the rearrangement, ΔE = E_P - E_*
function reference_energy!(ref, model, state)
    reference_state!(ref, model, state)
    z_star, ACE, APE = cell_array(model), cell_array(model), cell_array(model)
    reference_heights!(z_star, model, ref)
    available_compressible_energy!(ACE, model, ref)
    available_potential_energy!(APE, model, ref, z_star)
    AE = ACE .+ APE
    return (; z_star, ACE, APE, AE, ΣmAE = parcel_integral(model, ref, AE), ΔE = rearrangement_energy(model, ref))
end

# scales of the tolerances: the energy g L_z and the specific enthalpy Cp0 T0 per unit mass, and the mass of the box
energy_scale(model) = model.space.g * model.space.Lz
enthalpy_scale(model) = model.fluid.Cp0 * model.fluid.T0
box_mass(ref) = sum(ref.mass)

# the change of Π over half a row of a parcel that expands upwards in a column of constant v_*: ∫ r dz over Δz/2, with
# r ≈ g ρ |∂v/∂p| g ρ (z - z_row) (the error of AN2D before v_* followed the end slots' parcels beyond the end rows)
expansion_bound(model) = (model.space.g * 1.1ρ0 * model.dz)^2 * model.fluid.v0 * model.fluid.α_p

# the parcels' own heights, on the interior cells, and a result on the cells, per parcel (in the order of the parcels)
heights(model) = [cellpoint(model.domain, model.dz, 2, j) for i in interior_cells(model.domain, 1), j in interior_cells(model.domain, 2)]
parcel_values(model, field) = vec(interior(model.domain, field))

# whether each parcel's reference height is at a wall
at_wall(model, energy) = map(z -> z == 0 || z == model.space.Lz, parcel_values(model, energy.z_star))

# f(ref, fluid, g, ϑ, q, z_*) at the reference height of each parcel: the LNB residual, N_*²
function at_reference_height(f, ref, model, energy)
    z_star = parcel_values(model, energy.z_star)
    return map(n -> f(ref, model.fluid, model.space.g, ref.consvar[n], ref.q[n], z_star[n]), eachindex(ref.mass))
end

# whether each parcel is placed, unsplit, in the slot of its own row
in_own_rows(ref) = all(enumerate(ref.order)) do (r, n)
    e = ref.placement.first[r]
    ref.placement.first[r+1] == e + 1 && ref.placement.slot[e] == ref.row[n]
end

## Tests of AN2D

# A horizontally uniform, stably stratified state at rest is its own Lorenz reference state: each parcel stays in its own
# row, unsplit, with E_* = E_P exactly, and the duals certify it. Its reference heights stay at the row centres, but in
# the top and bottom rows: between their centres and the walls v_* follows the compression of their own parcels, so the
# static energy of these parcels is flat there and z_* is anywhere in that half row. Π is round-off of h, eps Cp0 T0,
# everywhere, where a v_* constant beyond the end rows made it O(Δz²) in the end rows
@testset "resting state, AN2D" begin
    model = an_box(Float64)
    ref = lorenz(model)
    energy = reference_energy!(ref, model, resting(model))
    @test in_own_rows(ref)
    @test energy.ΔE == 0
    @test ref.certificate_violations == 0
    z_star, Π = interior(model.domain, energy.z_star), interior(model.domain, energy.AE)
    @test z_star[:, 2:end-1] ≈ heights(model)[:, 2:end-1] rtol = 1e-6
    @test all(abs.(z_star[:, [1, end]] .- heights(model)[:, [1, end]]) .<= model.dz / 2)
    @test all(Π .>= 0)
    @test maximum(Π) <= 64eps() * enthalpy_scale(model)
end

# Fully mixed (neutral) fluid: every parcel is the same, so every placement is optimal and E_P - E_* is round-off. The
# static energy of a parcel is flat along the whole profile (N_*² = 0, z_* undetermined), up to the walls, as v_* follows
# the compression of the end slots' parcels beyond the end rows (a constant v_* there lowered the walls by O(Δz²) and gave
# every parcel Π = O(Δz²) at a wall, `expansion_bound`). What remains of Π is the error of the cubic interpolant of π_*:
# - for a uniform ρ_R, where π_* is quadratic in z, none: Π is at the resolution of the pressures, v eps p0, in Float32 as
#   in Float64;
# - for a ρ_R that varies with height, O(Δz⁴), converging at fourth order.
mixed(model) = tracer_state(model, (x, z) -> T0, (x, z) -> q0)
function mixed_energy(model)
    ref = lorenz(model)
    energy = reference_energy!(ref, model, mixed(model))
    @test abs(energy.ΔE) <= 1e-13 * enthalpy_scale(model) * box_mass(ref)
    @test ref.certificate_violations == 0
    @test all(>=(0), interior(model.domain, energy.AE))
    N2 = at_reference_height(stratification, ref, model, energy)
    # |N_*²| against the scale of its two cancelling terms, g² ρ² |∂v/∂p|
    @test all(n -> abs(n) <= 1e-2 * (model.space.g * ρ0)^2 * model.fluid.v0 * model.fluid.α_p, N2[.!at_wall(model, energy)])
    return maximum(interior(model.domain, energy.AE))
end
@testset "fully mixed, AN2D, uniform ρ_R, $F" for F in (Float32, Float64)
    model = an_box(F; ρ = z -> ρ0)
    largest = mixed_energy(model)
    @test largest <= 2 * model.fluid.v0 * eps(F) * model.fluid.p0
    @test largest <= expansion_bound(model) / 10
end
@testset "fully mixed, AN2D, ρ_R varying with height" begin
    largest = map(Mz -> mixed_energy(an_box(Float64; Mz)), (8, 16))
    @test largest[1] <= 1e-5 * expansion_bound(an_box(Float64))       # five orders below the constant v_*
    @test largest[1] / largest[2] > 12                                  # fourth order: 16 per halving of Δz
end

# Separable equation of state (γ = 0), at uniform ρ_R so that the rows have equal masses: the Lorenz state of an inverted
# column, or of a two-layer step that is warm below, is the column with its rows rearranged by decreasing
# f(θ) = v0 (α_T (θ - T0) + α_TT (θ - T0)²/2), the part of v that depends on θ, and the energy released is
#     E_P - E_* = Σ_r M_r f(θ_r) (p_R,r - p_R,σ(r)),
# since h depends on θ and p together only through f(θ) (p - p0), and the rest of the costs cancel in a permutation
@testset "analytic E_P - E_*, separable EOS, AN2D, $label" for (label, θ) in (("inverted column", z -> T0 + 2 * (1 - z / Lz)),
                                                                       ("two-layer step", z -> z < Lz / 2 ? T0 + 1 : T0))
    model = an_box(Float64; ρ = z -> ρ0, fluid = reference_fluid(Float64; γ = 0))
    ref = lorenz(model)
    energy = reference_energy!(ref, model, tracer_state(model, (x, z) -> θ(z), (x, z) -> q0))
    (; v0, α_T, α_TT) = model.fluid
    (Mx, Mz), Hz = dims(model.domain), halo_size(model.domain)[2]
    θ_rows = [θ(cellpoint(model.domain, model.dz, 2, Hz + r)) for r in 1:Mz]
    f = @. v0 * (α_T * (θ_rows - T0) + α_TT * (θ_rows - T0)^2 / 2)
    destination = sortperm(f)                               # rows from the bottom, the lightest at the top
    p_R = model.reference.p_R[Hz+1:Hz+Mz]
    M = Mx * ρ0 * model.dx * model.dz
    expected = sum(r -> M * f[destination[r]] * (p_R[destination[r]] - p_R[r]), 1:Mz)
    @test expected > 0
    @test energy.ΔE ≈ expected rtol = 1e-6
    @test ref.certificate_violations == 0
end

# assignment of N parcels of equal mass to N places of least cost, by dynamic programming over the subsets of the places
function least_assignment(C)
    N = size(C, 1)
    best = fill(typemax(eltype(C)), 2^N)
    best[1] = 0
    for mask in 0:2^N-1
        n = count_ones(mask) + 1                            # the next parcel
        n > N && continue
        for k in 1:N
            mask & (1 << (k - 1)) == 0 || continue
            next = mask | (1 << (k - 1))
            best[next+1] = min(best[next+1], best[mask+1] + C[n, k])
        end
    end
    return best[end]
end

# deterministic pseudo-random numbers in [0, 1)
noise(n) = mod(sin(12.9898n) * 43758.5453, 1)

# Brute force on N = 8 parcels of equal mass in 4 rows of 2 places each (uniform ρ_R): with γ = 0 the sort is the
# optimum, certified, for random parcels; with strong thermobaricity and density-compensated parcels it is not, and the
# reduced costs detect it
@testset "brute force, AN2D, $label" for (label, γ, compensated) in (("separable", 0, false), ("thermobaric", 3e-4, true))
    model = an_box(Float64; Mx = 2, Mz = 4, ρ = z -> ρ0, fluid = reference_fluid(Float64; γ))
    p_key = model.reference.p_R[1 + 2]                      # P_R at mid-depth, between rows 2 and 3
    θ(n) = T0 + 4 * noise(n)
    # with `compensated`, q makes v(p_key) nearly the same for all parcels
    (; α_T, α_TT, α_q, p0) = model.fluid
    q(n) = compensated ? q0 + (α_T * (1 + γ * (p_key - p0)) * (θ(n) - T0) + α_TT * (θ(n) - T0)^2 / 2) / α_q + 1e-6 * noise(n + 100) :
                         q0 + 0.002 * noise(n + 100)
    index(x, z) = round(Int, x / model.dx + 0.5) + 2 * round(Int, z / model.dz - 0.5)
    ref = lorenz(model)
    energy = reference_energy!(ref, model, tracer_state(model, (x, z) -> θ(index(x, z)), (x, z) -> q(index(x, z))))
    # place u of the 8 is in row (u - 1) ÷ 2 + 1 from the top
    C = [anelastic_cost(model, ref, n, (u - 1) ÷ 2 + 1) for n in 1:8, u in 1:8]
    own = sum(n -> anelastic_cost(model, ref, n, ref.row[n]), 1:8)
    least = ref.mass[1] * (own - least_assignment(C))       # E_P - E_* of the exact optimum
    if γ == 0
        @test energy.ΔE ≈ least rtol = 1e-6
        @test ref.certificate_violations == 0
    else
        @test energy.ΔE < least - 1e-3 * abs(least)
        @test ref.certificate_violations > 0
    end
end

# A single column: the placement conserves the mass of every parcel and fills every row
@testset "degenerate sizes, AN2D, Mx = 1" begin
    model = an_box(Float64; Mx = 1)
    ref = lorenz(model)
    energy = reference_energy!(ref, model, active(model))
    (; placement) = ref
    entries = 1:placement.count
    for n in eachindex(ref.mass)
        @test sum(e -> placement.mass[e], filter(e -> placement.parcel[e] == n, entries)) ≈ ref.mass[n] rtol = 1e-14
    end
    for k in eachindex(ref.slot_mass)
        @test sum(e -> placement.mass[e], filter(e -> placement.slot[e] == k, entries)) ≈ ref.slot_mass[k] rtol = 1e-12
    end
    @test all(isfinite, energy.z_star) && all(isfinite, energy.AE) && all(isfinite, ref.p_star)
    @test all(>=(0), interior(model.domain, energy.AE))
end

# Invariance: a periodic shift of the state in x permutes the parcels and changes neither the reference state nor
# E_P - E_*; changing the origin z0 of the geopotential changes neither the reference heights nor any available energy
@testset "invariance, AN2D" begin
    model = an_box(Float64)
    state = active(model)
    ref = lorenz(model)
    energy = reference_energy!(ref, model, state)
    # the interior columns shifted periodically by 3 cells
    columns = interior_cells(model.domain, 1)
    shifted = (; m = copy(state.m))
    shifted.m[columns, :, :] .= circshift(state.m[columns, :, :], (3, 0, 0))
    ref_shifted = lorenz(model)
    energy_shifted = reference_energy!(ref_shifted, model, shifted)
    @test energy_shifted.ΔE ≈ energy.ΔE rtol = 1e-6
    @test ref_shifted.p_star ≈ ref.p_star rtol = 1e-14
    @test interior(model.domain, energy_shifted.z_star) ≈ circshift(interior(model.domain, energy.z_star), (3, 0)) rtol = 1e-8
    # origin of the geopotential: the same model (and P_R) in a tank with another origin
    space = tank(Float64; z0 = 0.3)
    raised = typeof(model)(map(f -> f === :space ? space : getfield(model, f), fieldnames(typeof(model)))...)
    ref_raised = lorenz(raised)
    energy_raised = reference_energy!(ref_raised, raised, state)
    @test energy_raised.ΔE == energy.ΔE
    @test energy_raised.z_star == energy.z_star
    @test energy_raised.AE == energy.AE
end

# Weak duality: for any profile, Σ_n m_n Π_n with the global minimisers z_* is at least E_P - E_*, with equality for the
# duals of the optimal placement with z_* restricted to the row centres: Σ_n m_n (c_n,own - u_n - ψ_own). With γ = 0 the
# sort is the optimum
@testset "weak duality, AN2D" begin
    model = an_box(Float64; fluid = reference_fluid(Float64; γ = 0))
    ref = lorenz(model)
    energy = reference_energy!(ref, model, active(model))
    @test energy.ΔE > 0
    @test ref.certificate_violations == 0
    @test energy.ΣmAE >= energy.ΔE
    on_rows = sum(eachindex(ref.mass)) do n
        k = ref.row[n]
        ref.mass[n] * (anelastic_cost(model, ref, n, k) - ref.u[n] - ref.slot_dual[k])
    end
    @test on_rows ≈ energy.ΔE rtol = 1e-6
end

## Tests of FC2D

# A horizontally uniform, stably stratified state at rest is its own Lorenz reference state: each parcel stays in its own
# row, unsplit, as the slots are the rows with their masses. E_P - E_* is then the difference between the model's
# discrete hydrostatic balance and that of the column (in mass, P_k = P_top + g (Σ_{l<k} M_l + M_k/2)/L_x), O(Δz⁴) and
# about 1e-18 of M g L_z for M_z = 8 (in BigFloat), so round-off in Float64. P_* is the cubic through the slot edges, so
# the reference heights are the parcels' own heights to O(Δz²), and Π is round-off of h, eps Cp0 T0
@testset "resting state, FC2D" begin
    offsets = map((8, 16, 32)) do Mz
        model = fc_box(Float64; Mx = 2, Mz)
        ref = lorenz(model)
        energy = reference_energy!(ref, model, resting(model))
        @test in_own_rows(ref)
        @test ref.certificate_violations == 0
        @test abs(energy.ΔE) <= 1e-14 * box_mass(ref) * energy_scale(model)
        Π = interior(model.domain, energy.AE)
        @test all(Π .>= 0) && maximum(Π) <= 64eps() * enthalpy_scale(model)
        maximum(abs, interior(model.domain, energy.z_star) .- heights(model))
    end
    @test offsets[1] < 1e-3 * Lz / 8                        # a thousandth of a row
    @test offsets[1] / offsets[2] > 3.5 && offsets[2] / offsets[3] > 3.5
end

# Fully mixed fluid: every parcel is the same, so every placement is optimal and E_P - E_* is round-off. The static
# energy of a parcel is flat along P_* to the error of the cubic (N_*² ≈ 0, z_* undetermined), and Π is small and, with
# the parcel's own height as a candidate for z_*, nonnegative
@testset "fully mixed, FC2D" begin
    model = fc_box(Float64)
    ref = lorenz(model)
    energy = reference_energy!(ref, model, tracer_state(model, (x, z) -> T0, (x, z) -> q0))
    @test abs(energy.ΔE) <= 1e-14 * box_mass(ref) * energy_scale(model)
    @test ref.certificate_violations == 0
    Π = interior(model.domain, energy.AE)
    @test all(Π .>= 0) && maximum(Π) <= expansion_bound(model)
    N2 = at_reference_height(stratification, ref, model, energy)
    @test all(n -> abs(n) <= 1e-2 * (model.space.g * ρ0)^2 * model.fluid.v0 * model.fluid.α_p, N2[.!at_wall(model, energy)])
end

# Degenerate sizes: a single column, two rows, and a strongly compressible fluid (sound speed √10 m s⁻¹,
# g L_z/c² = 0.5). The placement conserves the mass of every parcel and fills every slot, and the column fills the box
@testset "degenerate sizes, FC2D, $label" for (label, model) in (("Mx = 1", fc_box(Float64; Mx = 1)),
                                                                ("Mz = 2", fc_box(Float64; Mz = 2)),
                                                                ("strongly compressible", fc_box(Float64; fluid = reference_fluid(Float64; α_p = 1e-4))))
    ref = lorenz(model)
    energy = reference_energy!(ref, model, active(model))
    (; placement) = ref
    entries = 1:placement.count
    for n in eachindex(ref.mass)
        @test sum(e -> placement.mass[e], filter(e -> placement.parcel[e] == n, entries)) ≈ ref.mass[n] rtol = 1e-14
    end
    for k in eachindex(ref.slot_mass)
        @test sum(e -> placement.mass[e], filter(e -> placement.slot[e] == k, entries)) ≈ ref.slot_mass[k] rtol = 1e-12
    end
    @test all(isfinite, energy.z_star) && all(isfinite, energy.AE) && all(isfinite, ref.p_star)
    @test all(>=(0), interior(model.domain, energy.AE))
    @test abs(ref.volume_residual) < 1e-14
end

# Invariance: a periodic shift of the state in x permutes the parcels and changes neither the reference state nor
# E_P - E_*; changing the origin z0 of the geopotential changes neither the reference heights nor any available energy
@testset "invariance, FC2D" begin
    model = fc_box(Float64)
    state = active(model)
    ref = lorenz(model)
    energy = reference_energy!(ref, model, state)
    # the interior columns shifted periodically by 3 cells
    columns = interior_cells(model.domain, 1)
    shifted = (; m = copy(state.m))
    shifted.m[columns, :, :] .= circshift(state.m[columns, :, :], (3, 0, 0))
    ref_shifted = lorenz(model)
    energy_shifted = reference_energy!(ref_shifted, model, shifted)
    @test energy_shifted.ΔE ≈ energy.ΔE rtol = 1e-6
    @test ref_shifted.p_top ≈ ref.p_top rtol = 1e-14
    @test ref_shifted.p_star ≈ ref.p_star rtol = 1e-14
    @test interior(model.domain, energy_shifted.z_star) ≈ circshift(interior(model.domain, energy.z_star), (3, 0)) rtol = 1e-8
    # origin of the geopotential: the same model in a tank with another origin
    space = tank(Float64; z0 = 0.3)
    raised = typeof(model)(map(f -> f === :space ? space : getfield(model, f), fieldnames(typeof(model)))...)
    ref_raised = lorenz(raised)
    energy_raised = reference_energy!(ref_raised, raised, state)
    @test energy_raised.ΔE == energy.ΔE
    @test energy_raised.z_star == energy.z_star
    @test energy_raised.AE == energy.AE
end

# Weak duality: for any profile, Σ_n m_n Π_n with the global minimisers z_* is at least E_P - E_*. With γ = 0 the sort is
# the optimum
@testset "weak duality, FC2D" begin
    model = fc_box(Float64; fluid = reference_fluid(Float64; γ = 0))
    ref = lorenz(model)
    energy = reference_energy!(ref, model, active(model))
    @test energy.ΔE > 0
    @test ref.certificate_violations == 0
    @test energy.ΣmAE >= energy.ΔE
end

# Precision: the same reference state in BigFloat as in Float64, to the precision of Float64. The state is that of
# Float64, converted, so that the differences are those of the reference state alone
@testset "precision, FC2D, BigFloat" begin
    state = active(fc_box(Float64))
    model64, model = fc_box(Float64), fc_box(BigFloat)
    ref64, ref = lorenz(model64), lorenz(model)
    energy64 = reference_energy!(ref64, model64, state)
    energy = reference_energy!(ref, model, (; m = BigFloat.(state.m)))
    @test eltype(energy.z_star) == BigFloat
    @test ref.certificate_violations == 0
    @test abs(ref.volume_residual) < 1e-60
    @test Float64(energy.ΔE) ≈ energy64.ΔE rtol = 1e-6
    @test Float64(energy.ΣmAE) ≈ energy64.ΣmAE rtol = 1e-6
    @test Float64.(ref.p_star) ≈ ref64.p_star rtol = 1e-14
    @test Float64.(energy.z_star) ≈ energy64.z_star atol = 1e-7 * Lz
end

## Tests of either model

# The level of neutral buoyancy: at an interior minimum r(z_*) = 0 and N_*² ≥ 0; Π ≥ 0 everywhere; at a wall minimum r
# has the outward sign
@testset "level of neutral buoyancy, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64)
    ref = lorenz(model)
    energy = reference_energy!(ref, model, active(model))
    r, N2 = at_reference_height(buoyancy_residual, ref, model, energy), at_reference_height(stratification, ref, model, energy)
    wall = at_wall(model, energy)
    z_star, Π = parcel_values(model, energy.z_star), parcel_values(model, energy.AE)
    @test maximum(abs, r[.!wall]) <= 1e-9 * model.space.g
    @test all(>=(0), N2[.!wall])
    @test all(>=(0), Π)
    @test all(((z, r),) -> (z == 0 && r <= 0) || (z == model.space.Lz && r >= 0), zip(z_star[wall], r[wall]))
end

# The available energy Π of a parcel at height z and pressure p, of reference height z_*, against its definition
#     Π = Σ(z, p) - Σ(z_*, P_*(z_*)) - (p - P_*(z)) Σ_p(z, p),
# evaluated directly: h(P_R(z)) - h(P_R(z_*)) + π_*(z) - π_*(z_*) + g (z - z_*) for AN2D (Σ_p = 1/ρ_R), and
# h(p) - h(P_*(z_*)) + g (z - z_*) - (p - P_*(z)) v for FC2D (Σ_p = v). The available compressible energy Π₁, from P_*(z)
# to p at fixed z, is zero for AN2D (Σ linear in p) and nonnegative for FC2D (h concave in p); the available potential
# energy Π₂, from z_* to z
# along the reference state, is nonnegative (z_* minimises S); and the integrals add up
@testset "parts of the available energy, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64)
    (; fluid, space) = model
    ref = lorenz(model)
    energy = reference_energy!(ref, model, active(model))
    Π₁, Π₂, Π = parcel_values(model, energy.ACE), parcel_values(model, energy.APE), parcel_values(model, energy.AE)
    z, z_star = vec(heights(model)), parcel_values(model, energy.z_star)
    direct = map(eachindex(ref.mass)) do n
        ϑ, q = ref.consvar[n], ref.q[n]
        if model isa AN2D
            P, P_star = hermite(ref.pressure, z[n])[1], hermite(ref.pressure, z_star[n])[1]          # P_R(z), P_R(z_*)
            Δπ = hermite(ref.potential, z[n])[1] - hermite(ref.potential, z_star[n])[1]           # π_*(z) - π_*(z_*)
            enthalpy_difference(fluid, P_star, P - P_star, ϑ, q) + Δπ + space.g * (z[n] - z_star[n])
        else
            p, v = ref.parcel_pressure[n], ref.parcel_volume[n]
            P, P_star = hermite(ref.pressure, z[n])[1], hermite(ref.pressure, z_star[n])[1]          # P_*(z), P_*(z_*)
            enthalpy_difference(fluid, P_star, p - P_star, ϑ, q) + space.g * (z[n] - z_star[n]) - (p - P) * v
        end
    end
    @test Π ≈ direct atol = 1e-12 * energy_scale(model)
    @test all(>=(0), Π₁) && all(>=(0), Π₂)
    @test model isa AN2D ? all(iszero, Π₁) : maximum(Π₁) > 0
    @test energy.ΣmAE ≈ parcel_integral(model, ref, energy.ACE) + parcel_integral(model, ref, energy.APE) rtol = 1e-14
end

# The available compressible energy Π₁ = h(p) - h(P) - (p - P) v(p) by quadrature, -∫ (s - P) ∂v/∂p(s) ds from P to
# p = P + Δp, against its exact value -∂v/∂p Δp²/2 for this fluid, whose v is linear in p, for Δp from 10⁻² Pa to 10⁴ Pa
# of either sign, in Float32, Float64 and BigFloat: it keeps the relative precision of the float type, and is zero at
# Δp = 0. The difference h(p) - h(P) - Δp v(p) of the definition loses it: in Float32 it is off by more than 1 % below
# Δp ≈ 10 Pa
@testset "available compressible energy by quadrature, $F" for F in (Float32, Float64, BigFloat)
    fluid = reference_fluid(F)
    P, ϑ, q = F(1e5), F(T0 + 1), F(q0 - 0.001)
    ∂v_∂p = fluid(:p, :consvar, :q).volume_functions(P, ϑ, q)[2]
    @test available_compressible_energy(fluid, P, zero(F), ϑ, q) == 0
    for Δp in F.((1e-2, 1, 1e2, 1e4)), sign in (1, -1)
        exact = -∂v_∂p * Δp^2 / 2
        @test available_compressible_energy(fluid, P, sign * Δp, ϑ, q) ≈ exact rtol = 16eps(F)
    end
    difference(Δp) = enthalpy_difference(fluid, P, Δp, ϑ, q) - Δp * fluid(:p, :consvar, :q).specific_volume(P + Δp, ϑ, q)
    F == Float32 && @test !isapprox(difference(F(1)), -∂v_∂p / 2; rtol = 0.01)
end

# Where Float32 fails: the differences of h are integrals of v (about 10⁻³ m³ kg⁻¹) over differences of pressure, which
# keep the precision of Float32, but the pressures themselves (about 10⁵ Pa) are resolved to about 10⁻² Pa, and in FC2D
# E_P - E_* is the difference of sums of about M g L_z. E_P - E_* and Σ m Π of the active state, with its perturbation
# scaled by a, are compared with those of Float64, of the same state converted: they hold to 1 % down to some
# amplitude, below which they are noise. In AN2D, below a ≈ 0.1 every parcel stays in its own row, and E_P - E_* = 0
# exactly in both
@testset "Float32 breaks below an amplitude, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    amplitudes = 10.0 .^ (1:-0.5:-3)
    failing = map(amplitudes) do a
        state(model) = tracer_state(model, (x, z) -> θ_rest(z) + a * (θ_active(x, z) - θ_rest(z)),
                                           (x, z) -> q_rest(z) + a * (q_active(x, z) - q_rest(z)))
        model32, model64 = make(Float32), make(Float64)
        energy32 = reference_energy!(lorenz(model32), model32, (; m = Float32.(state(model64).m)))
        energy64 = reference_energy!(lorenz(model64), model64, state(model64))
        !isapprox(energy32.ΔE, energy64.ΔE; rtol = 0.01) || !isapprox(energy32.ΣmAE, energy64.ΣmAE; rtol = 0.01)
    end
    threshold = amplitudes[findfirst(failing)]
    @info "Float32, $M: E_P - E_* or Σ m Π is off by more than 1 % below the amplitude $threshold of the active state"
    @test !any(failing[1:3])                                # down to the active state itself, a = 1
    @test any(failing)
end

# The loops over the parcels run on the model's manager: serial, vectorised over the parcels of a row, or on threads, with
# the same results up to round-off
@testset "managers, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    energies = map((nothing, VectorizedCPU(), MultiThread(VectorizedCPU()))) do mgr
        model = make(Float64; mgr)
        ref = lorenz(model)
        (; ref, energy = reference_energy!(ref, model, active(model)))
    end
    serial = first(energies)
    for (; ref, energy) in energies[2:end]
        @test ref.order == serial.ref.order
        @test ref.reduced_cost ≈ serial.ref.reduced_cost atol = 1e-9
        @test ref.certificate_violations == serial.ref.certificate_violations
        @test energy.z_star ≈ serial.energy.z_star atol = 1e-8 * Lz
        @test energy.AE ≈ serial.energy.AE atol = 1e-9
        @test energy.ΔE == serial.energy.ΔE
    end
end

# Allocations: after the first call, `reference_state!` and the kernels of the available energy allocate a small, bounded
# amount, the same on a larger grid (no state-sized arrays), on a serial manager
@testset "allocations, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    bytes = map((8, 16)) do Mz
        model = make(Float64; Mx = 2Mz, Mz, mgr = VectorizedCPU())
        ref, state = lorenz(model), active(model)
        z_star, ACE, APE = cell_array(model), cell_array(model), cell_array(model)
        compute!() = (reference_state!(ref, model, state); reference_heights!(z_star, model, ref);
                      available_compressible_energy!(ACE, model, ref); available_potential_energy!(APE, model, ref, z_star))
        compute!()
        @allocated compute!()
    end
    @test bytes[1] == bytes[2]
    @test bytes[2] <= 1024
end

# A method of the Lorenz reference state is an `AbstractRearrangement` with its own `rearrange!`: one without it is
# refused rather than computed by another method
struct UnimplementedRearrangement <: AbstractRearrangement end
@testset "methods of the rearrangement, $M" for (M, make, Reference) in (("AN2D", an_box, AnelasticLorenzReference),
                                                                         ("FC2D", fc_box, CompressibleLorenzReference))
    model = make(Float64)
    @test lorenz(model).method isa SortedRearrangement
    ref = Reference(model, UnimplementedRearrangement())
    @test_throws MethodError reference_state!(ref, model, active(model))
end

# the potential energy of the parcels at their own level [J m⁻¹], from the parcels of a reference state:
#     E_P = Σ_n m_n (h(p_n, ϑ_n, q_n) + g (z_n - z0)) - p_n Δx Δz,
# with p_n = P_R(z_n) in AN2D, the parcel's own pressure in FC2D
function parcel_potential_energy(model::AN2D, ref)
    return sum(eachindex(ref.mass)) do n
        k = ref.row[n]
        h = model.fluid(:p, :consvar, :q).specific_enthalpy(ref.slot_pressure[k], ref.consvar[n], ref.q[n])
        ref.mass[n] * (h + model.space.g * (ref.slot_height[k] - model.space.z0)) - ref.slot_pressure[k] * model.dx * model.dz
    end
end
function parcel_potential_energy(model::FC2D, ref)
    return sum(eachindex(ref.mass)) do n
        p = ref.parcel_pressure[n]
        h = model.fluid(:p, :consvar, :q).specific_enthalpy(p, ref.consvar[n], ref.q[n])
        ref.mass[n] * (h + model.space.g * (ref.height[n] - model.space.z0)) - p * model.dx * model.dz
    end
end

# the state of a setup with the active tracers, at rest
active_setup(model::AN2D) = initialize(model, (; consvar = θ_active, q = q_active, u = z -> 0.0, ψ = (X, Z) -> 0.0))
active_setup(model::FC2D) = initialize(model, (; consvar = θ_active, q = q_active, u = z -> 0.0, ψ = (X, Z) -> 0.0); pref = 1e5)

# The book: the reference state is computed once per session, and its diagnostics are recorded in data files, the
# profiles on z, and read back. The background potential energy is ∫ρ (P - Π) dV, the integrated potential energy E_P of
# the energetics, which needs the tendencies, less Σ m Π
@testset "diagnostics and data files, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64)
    state = active_setup(model)
    dstate, scratch = tendencies!(void, void, model, state, 0.0)
    ref = lorenz(model)
    info = diagnostic_info(model; reference_state = ref)
    @test info.p_star.location == (nothing, Cell()) && info.z_star.location == (Cell(), Cell())
    session = open(diagnostics(model; reference_state = ref); state, dstate, scratch, model)
    # the energies and the available energy, separate diagnostics of the reference state
    energy = reference_energy!(lorenz(model), model, state)
    E_P = CookBooks.get(session, :integrated_potential_energy)
    @test CookBooks.get(session, :background_potential_energy) == E_P - energy.ΣmAE
    @test E_P ≈ parcel_potential_energy(model, ref) rtol = 1e-14
    # ACE = Π₁, APE = Π₂ and AE = Π₁ + Π₂ on the cells, and their integrals
    @test CookBooks.get(session, :ACE) == energy.ACE && CookBooks.get(session, :APE) == energy.APE
    @test CookBooks.get(session, :AE) == energy.AE
    @test CookBooks.get(session, :integrated_AE) == energy.ΣmAE
    @test CookBooks.get(session, :integrated_ACE) == parcel_integral(model, ref, energy.ACE)
    @test CookBooks.get(session, :integrated_APE) == parcel_integral(model, ref, energy.APE)
    @test CookBooks.get(session, :wall_minima) == count(at_wall(model, energy))
    names = model isa FC2D ? (:z_star, :ACE, :AE, :integrated_APE, :p_star, :background_potential_energy, :wall_minima, :p_top) :
                             (:z_star, :ACE, :AE, :integrated_APE, :p_star, :background_potential_energy, :wall_minima)
    dir = mktempdir()
    for path in (joinpath(dir, "reference.nc"), joinpath(dir, "reference.jld2"))
        file = DataFile(path, model, names; reference_state = ref)
        record!(file, session, 0.0)
        close(file)
        for name in names
            value, t = read_diagnostic(path, name)
            expected = CookBooks.get(session, name)
            @test t == [0.0]
            @test (expected isa Number ? value == [expected] : selectdim(value, ndims(value), 1) == expected)
        end
    end
    close(session)
end
