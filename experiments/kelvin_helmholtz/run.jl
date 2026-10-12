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
viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3, bulk_visc = 0.0)    # dynamic and bulk viscosity [Pa s]
thermoflux_scheme  = ThermoFluxConsistent(k_T = 1e-4, k_q = 1e-5)           # thermal and composition diffusivities [m² s⁻¹]

#####
##### Numerical choices
#####

mgr    = tSIMD()                                                         # multi-threaded, vectorised loops
domain = Box2D(Mx = 64, Mz = 32, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))
advection_scheme = AdvEnergyCons()

# walls at the bottom and top: free slip, no flux of consvar or q; tracer conditions are positional (consvar, q)
closed = Wall(tangential = FreeSlip(), tracers = (Flux(0.0), Flux(0.0)))
boundary_conditions = BoxBoundaryConditions(domain; bottom = closed, top = closed)

#####
##### Models
#####

# fully compressible
fc = FC2D((; mgr, domain, space, fluid, advection_scheme, thermoflux_scheme, viscosity_scheme, boundary_conditions))

# anelastic, about a reference density (a number, a function of the height z, or a vector of cell values) with
# pressure ptop at the top
anelastic_reference = AnelasticReference(domain, space, ρ0; ptop = 1e5)
an = AN2D((; mgr, domain, space, fluid, advection_scheme, buoyancy_scheme = BuoyThermo(), thermoflux_scheme,
    viscosity_scheme, anelastic_reference, boundary_conditions))

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
# and step limits; FC2D resolves sound waves, so its run takes several minutes
diagnostic = :conservative_variable
params = (; Nslice = 120, slice_size = 0.25, dt_max = 0.05, cfl = 0.5)

# initial states; FC2D also needs the pressure at the top, to build its balanced initial density
states = (an = initialize(an, fields), fc = initialize(fc, fields; ptop = 1e5))

# run each model, recording a frame of the diagnostic at each time slice
an_video = DiagnosticVideo(an, states.an, diagnostic)
CFAnelastics.loop(an, states.an, CFTimeSchemes.RungeKutta4(an), params; plot_func = record_frame!, plot_obj = an_video)
fc_video = DiagnosticVideo(fc, states.fc, diagnostic)
CFCompressible.loop(fc, states.fc, CFTimeSchemes.RungeKutta4(fc), params; plot_func = record_frame!, plot_obj = fc_video)

# the two models side by side
file = save(joinpath(mkpath(joinpath(@__DIR__, "output")), "kelvin_helmholtz.mp4"), an_video, fc_video)
println("\nvideo saved to $file")
