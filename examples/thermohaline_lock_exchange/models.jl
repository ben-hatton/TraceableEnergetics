# Models, initial state and outputs of the thermohaline lock exchange (see run.jl), shared by run.jl, which runs both
# models, and figures.jl, which draws the figures from the files of a run.
include("../../tools/setup.jl")

using TraceableEnergetics, CFAnelastics, CFCompressible
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes
using CFPlanets: Tank2D
using CookBooks: CookBooks
using CairoMakie
include("../../tools/budget_figures.jl")

#####
##### Models: the box and fluid of examples/energy_budget, closed by free-slip insulating walls on all four sides
#####

space = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)
T0, q0, ρ0 = 280.0, 0.035, 1000.0
fluid = NonlinearBinaryFluid(:potential_temperature, (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1 / ρ0, α_T = 2e-4,
    α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ = 1e-7))

mgr    = SIMD()
domain = Box2D(Mx = 64, Mz = 32, Hx = 1, Hz = 1, boundary = (Bounded, Bounded))
closed = Wall(tangential = FreeSlip(), tracers = (Flux(0.0), Flux(0.0)))
boundary_conditions = BoxBoundaryConditions(domain; left = closed, right = closed, bottom = closed, top = closed)

# heat diffuses faster than salt, as in water
physics = (; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(),
    thermoflux_scheme = ThermoFluxConsistent(k_T = 1e-4, k_q = 3e-5), viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3),
    boundary_conditions)
fc = FC2D(physics)

#####
##### Initial state: the two waters at the density of the background
#####

# the background: uniform composition q0, and θ increasing upwards by Γ = N²/(g α_T) [K m⁻¹], N² = 1e-3 s⁻²
Γ = 1e-3 / (space.g * 2e-4)
θ_background(x, z) = T0 + Γ * (z - space.Lz / 2)
q_background(x, z) = q0
# the lock: warmer by Δθ and saltier by Δq on the left, colder and fresher on the right, with a front of width δ at the
# middle; the salt offsets 0.8 × 2Δq / (2e-4 × 2Δθ) = 40 % of the difference of density due to heat
Δθ, Δq, δ = 3.0, 3e-4, 0.05
side(x) = -tanh((x - space.Lx / 2) / δ)                  # 1 on the left, -1 on the right
θ_lock(x, z) = θ_background(x, z) + Δθ * side(x)
q_lock(x, z) = q_background(x, z) + Δq * side(x)

# the density of the background, in hydrostatic balance with the pressure 1e5 Pa at the bottom, on the cells (halo
# included); it is horizontally uniform, and is AN2D's reference density ρ_R(z)
ρ_background = CFCompressible.balanced_density(fc; consvar = θ_background, q = q_background, pref = 1e5)
Hx, Hz = CFBoxes.halo_size(domain)
ρ_column = ρ_background[Hx+1, Hz+1:end-Hz]                 # the interior cells of a column, from the bottom
an = AN2D((; physics..., buoyancy_scheme = ThermodynamicForcing(),
    anelastic_reference = AnelasticReference(domain, space, ρ_column; pref = 1e5), gauge_offset = 0.0))
models = (an = an, fc = fc)

# AN2D: (ρ_R ϑ, ρ_R q) of the lock at rest
fields = setup(an; θ = θ_lock, q = q_lock)
state_an = initialize(an, fields)
# FC2D: the same densities, (ρ, ρϑ, ρq) = (ρ_R, ρ_R ϑ, ρ_R q) at rest, rather than a density in hydrostatic balance with
# the tracers of each column; the consvar of this fluid is the potential temperature
state_fc = initialize(fc, fields; pref = 1e5)
for j in axes(state_fc.m, 2), i in axes(state_fc.m, 1)
    x, z = CFBoxes.cellpoint(domain, fc.dx, 1, i), CFBoxes.cellpoint(domain, fc.dz, 2, j)
    ρ = ρ_background[i, j]
    state_fc.m[i, j, 1], state_fc.m[i, j, 2], state_fc.m[i, j, 3] = ρ, ρ * θ_lock(x, z), ρ * q_lock(x, z)
end
states = (an = state_an, fc = state_fc)

# the manipulated conversion needs a reference state ρ_R(z), p_R(z): AN2D has its own, FC2D takes that of AN2D (same grid)
references = (an = an.reference, fc = an.reference)
# the Lorenz reference state of available energy of each model, computed at each output by the exact rearrangement
# of the parcels (the transportation simplex, from their sorted placement)
lorenz = (an = AnelasticLorenzReference(an, ExactRearrangement()), fc = CompressibleLorenzReference(fc, ExactRearrangement()))
loops = (an = CFAnelastics.loop, fc = CFCompressible.loop)

#####
##### Diagnostics: integrals over the box every 2.5 ms, fields every 0.5 s
#####

T, interval, every = 20.0, 2.5e-3, 200
params = (; Nslice = round(Int, T / interval), slice_size = interval, dt_max = interval, cfl = 0.9)
dir = mkpath(joinpath(@__DIR__, "output"))
paths = map(key -> joinpath(dir, "lock_exchange_$key.nc"), (an = :an, fc = :fc))
series_paths = map(key -> joinpath(dir, "lock_exchange_series_$key.jld2"), (an = :an, fc = :fc))
