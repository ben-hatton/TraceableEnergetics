# Configuration shared by the AN2D and FC2D test modules: a small box, all physics switched on. Mirrors the
# configuration of the model suites of CFAnelastics and CFCompressible, which are tested in their own repositories.
using Test

using LoopManagers: MultiThread, VectorizedCPU
using CFBoxes, CFDiffusionSchemes, ClimFluids
using CFPlanets: Tank2D
import MutatingOrNot
using MutatingOrNot: void

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

# walls exercising every kind of condition (consvar = θ for this fluid)
const forced_box = (;
    left   = Wall(tangential = TangentialVelocity(z -> 0.02 * z), tracers = (Value(T0 + 0.5), Value(q0))),
    right  = Wall(tangential = NoSlip(), tracers = (Value(z -> T0 - 0.5 + 0.1 * z), Flux(1e-6))),
    bottom = Wall(tangential = TangentialVelocity(0.05), tracers = (HeatFlux(x -> 40.0 + 10.0 * cos(2π * x / Lx)), Flux(0.0))),
    top    = Wall(tangential = Stress(x -> 2e-4 * sin(2π * x / Lx)), tracers = (Flux(1e-3), Flux(-1e-6))))
