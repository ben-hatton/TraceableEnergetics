# Canonical example: a Kelvin-Helmholtz shear layer, run with the anelastic (AN2D) and the fully compressible (FC2D)
# models from the same model-independent initial fields, and shown as a video of a chosen diagnostic, the two models
# side by side. Shows every piece needed to build and run a model.
#
#   julia -t 4 experiments/kelvin_helmholtz/run.jl
#
# The video is saved to experiments/kelvin_helmholtz/output/kelvin_helmholtz.mp4.
include("../../tools/setup.jl")

using TraceableEnergetics, CFAnelastics, CFCompressible
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes
using CFPlanets: Tank2D
using CairoMakie

#####
##### Physical choices
#####

# the physical box: side lengths [m] and gravity [m s⁻²]
space = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)

# reference values: temperature [K], composition [kg/kg], density [kg m⁻³], velocity jump across the shear [m s⁻¹]
T0, q0, ρ0, Δu = 280.0, 0.035, 1000.0, 0.1

# a seawater-like binary fluid, with potential temperature as the conservative variable
fluid = NonlinearBinaryFluid(:potential_temperature, (;
    p0   = 1e5,                     # reference pressure [Pa]
    T0, q0,
    Cp0  = 4000.0,                  # heat capacity [J kg⁻¹ K⁻¹]
    v0   = 1 / ρ0,                  # specific volume [m³ kg⁻¹]
    α_T  = 2e-4,                    # thermal expansion [K⁻¹]
    α_q  = 0.8,                     # haline contraction [(kg/kg)⁻¹]
    α_p  = 1e-6,                    # compressibility [m s² kg⁻¹]
    M_s  = 3.14038218e-2,           # mean atomic weight of the solute [kg mol⁻¹]
    R    = 8.31446261815324,        # gas constant [J K⁻¹ mol⁻¹]
    α_TT = 1e-5,                    # cabbeling [K⁻²]
    γ    = 1e-7,                    # thermobaricity [Pa⁻¹]
))

# irreversible processes
viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3)    # dynamic viscosity [Pa s]
thermoflux_scheme  = ThermoFluxConsistent(k_T = 1e-4, k_q = 1e-5)           # thermal and composition diffusivities [m² s⁻¹]

#####
##### Numerical choices
#####

mgr    = SIMD()                     # vectorised loops; tSIMD() (multi-threaded) pays off only on larger grids
domain = Box2D(Mx = 256, Mz = 128, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))
spatial_scheme = Centred2()
vortex_scheme = VortexEnergyCons()

# walls at the bottom and top: free slip, no flux of consvar or q; tracer conditions are positional (consvar, q)
closed = Wall(tangential = FreeSlip(), tracers = (Flux(0.0), Flux(0.0)))
boundary_conditions = BoxBoundaryConditions(domain; bottom = closed, top = closed)

#####
##### Models
#####

# fully compressible
fc = FC2D((; mgr, domain, space, fluid, spatial_scheme, vortex_scheme, thermoflux_scheme, viscosity_scheme, boundary_conditions))

# anelastic, about a reference density (a number, a function of the height z, or a vector of cell values) with
# pressure pref at the height z0 where the geopotential vanishes (the bottom by default); the constant of the kinematic
# pressure (gauge_offset) changes no tendency
anelastic_reference = AnelasticReference(domain, space, ρ0; pref = 1e5)
an = AN2D((; mgr, domain, space, fluid, spatial_scheme, vortex_scheme, buoyancy_scheme = ThermodynamicForcing(), thermoflux_scheme,
    viscosity_scheme, anelastic_reference, boundary_conditions, gauge_offset = 0.0))

#####
##### Initial condition, shared by both models
#####

# tanh steps of θ and q across a shear layer at mid-depth, with a mode-1 perturbation: the fields (consvar, q, u, ψ)
# of the initialisation, identical for both models (same fluid, domain and space)
fields = kelvin_helmholtz(an; θ₀ = T0, Δθ = 0.1, q₀ = q0, Δu, ρ₀ = ρ0)

#####
##### Run
#####

# the diagnostic shown (any diagnostic of both models, see `diagnostic_info(an)`), and the time slices (one frame each)
# and step limits (cfl: fraction of the stability limit of the time scheme); FC2D resolves sound waves, so its run takes several minutes
diagnostic = :conservative_variable
params = (; Nslice = 120, slice_size = 0.25, dt_max = 0.05, cfl = 0.9)

# initial states; FC2D also needs the pressure at the top, to build its balanced initial density
states = (an = initialize(an, fields), fc = initialize(fc, fields; pref = 1e5))

# run each model, recording a frame of the diagnostic at t = 0 and at each time slice (see experiments/recording for
# the other outputs: NetCDF and JLD2 data files)
an_outputs = Outputs(DiagnosticVideo(an, diagnostic))
CFAnelastics.loop(states.an, an, CFTimeSchemes.RungeKutta4(an), params; outputs = an_outputs)
fc_outputs = Outputs(DiagnosticVideo(fc, diagnostic))
CFCompressible.loop(states.fc, fc, CFTimeSchemes.RungeKutta4(fc), params; outputs = fc_outputs)

# the two models side by side (a single video saves itself, with `DiagnosticVideo(...; file)`; two runs are joined here)
file = save(joinpath(mkpath(joinpath(@__DIR__, "output")), "kelvin_helmholtz.mp4"), an_outputs[1], fc_outputs[1])
println("\nvideo saved to $file")
