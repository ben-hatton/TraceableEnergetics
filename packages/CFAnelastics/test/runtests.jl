#=
Property tests for AN2D: the Poisson solver for the anelastic pressure, then the tendencies (memory safety,
impermeable walls, conservation, energy conservation and budget, no allocations, anelastic constraint).
Run with Pkg.test("CFAnelastics"), or from TraceableEnergetics with  julia -t 4 --project=. test/runtests.jl
=#
using Test

using LoopManagers: MultiThread, VectorizedCPU
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes, ForwardDiff
using CFPlanets: Tank2D
import MutatingOrNot
using MutatingOrNot: void
using Random: rand!

include("helpers.jl")

# Configuration: small box, all physics switched on, closed (free-slip, insulating) walls

const mgr = MultiThread(VectorizedCPU())
const space = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)
const Lx, Lz, g = space.Lx, space.Lz, space.g
const T0, q0, ρ0, U = 280.0, 0.035, 1000.0, 0.1

# nonlinear EOS with compressibility, thermobaricity, cabbeling and haline contraction
const fluid_params = (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1e-3,
    α_T = 2e-4, α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ = 1e-7)
const fluid = NonlinearBinaryFluid(:potential_temperature, fluid_params)

const thermoflux = ThermoFluxConsistent(k_T = 1e-4, k_q = 1e-5)

# closed (impermeable, insulating, no salt flux) walls
closed_wall(tangential = FreeSlip()) = Wall(; tangential, tracers = (Flux(0.0), Flux(0.0)))   # (consvar, q)

# smooth, x-asymmetric profiles; z = 0 at the bottom
θ_rest(z) = T0 + 1.0 * z / Lz
q_rest(z) = q0 - 0.002 * z / Lz
θ_active(x, z) = θ_rest(z) + 0.3 * sin(2π * x / Lx) * cos(π * z / Lz) + 0.1 * cos(4π * x / Lx + 1) * sin(2π * z / Lz)
q_active(x, z) = q_rest(z) + 0.001 * cos(2π * x / Lx + 0.5) * cos(π * z / Lz)

include("poisson.jl")
include("anelastic.jl")
