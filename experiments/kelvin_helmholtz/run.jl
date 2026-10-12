# Canonical example: a Kelvin-Helmholtz shear layer, run with the anelastic (AN2D) and the fully compressible (FC2D)
# models from the same model-independent initial fields. Shows every piece needed to build and run a model, and checks that
# the total energy is conserved (closed walls: viscous dissipation and diffusion only convert energy).
#
#   julia -t 4 experiments/kelvin_helmholtz/run.jl
#
# Set `video = true` (below, under Run) to record a video of each model in experiments/kelvin_helmholtz/output/.
include("../../tools/setup.jl")

using TraceableEnergetics, CFAnelastics, CFCompressible
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes
using CFPlanets: Tank2D
using MutatingOrNot: void
using CookBooks: CookBooks

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

# initial states; FC2D also needs the pressure at the top, to build its balanced initial density
states = (an = initialize(an, fields), fc = initialize(fc, fields; ptop = 1e5))

# optional video of the potential temperature θ (a diagnostic of both models), one file per model, one frame per time slice. A video runs longer,
# for the billows to grow and roll up (FC2D resolves sound waves, so its run takes several minutes)
video = true
video && @eval using CairoMakie

# time slices and step limits
params = video ? (; Nslice = 120, slice_size = 0.25, dt_max = 0.05, cfl = 0.5) :
                 (; Nslice = 5, slice_size = 0.4, dt_max = 0.05, cfl = 0.5)

# record the total energy (kinetic + potential + internal) at each slice
energies!(model::AN2D, state) = CFAnelastics.energies!((void, void, void), state, model)
energies!(model::FC2D, state) = CFCompressible.energies!((void, void, void), state, model)
function record_energy(model, state, dstate, scratch, cookbook, energies, t_iter, params)
    _, _, te = energies!(model, state)
    push!(energies, sum(te[xrange_interior(model.domain), zrange_interior(model.domain)]) * model.dx * model.dz)
    return energies
end

# a diagnostic on the interior cells, e.g. :potential_temperature (see `CFAnelastics.describe_diagnostics(an)` for the list)
diagnostics(model::AN2D) = CFAnelastics.diagnostics(model)
diagnostics(model::FC2D) = CFCompressible.diagnostics(model)
function interior_diagnostic(name, model, state, dstate, scratch)
    session = open(diagnostics(model); state, dstate, scratch, model)
    field = interior(model.domain, CookBooks.get(session, name))
    close(session)
    return field
end

# video: a heatmap of θ, with colours fixed by the initial range; one frame recorded per call of `record_frame`
function start_video(name, model, state)
    (; domain, dx, dz) = model
    x = [xpoint(domain, dx, i) for i in xrange_interior(domain)]
    z = [zpoint(domain, dz, j) for j in zrange_interior(domain)]
    dstate, scratch = CFTimeSchemes.tendencies!(void, void, model, deepcopy(state), 0.0)
    field, time = Observable(interior_diagnostic(:potential_temperature, model, state, dstate, scratch)), Observable(0.0)
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1]; title = lift(t -> "$name   t = $(round(t, digits = 2)) s", time), xlabel = "x [m]",
              ylabel = "z [m]", aspect = DataAspect())
    hm = heatmap!(ax, x, z, field; colormap = :thermal, colorrange = extrema(field[]))
    Colorbar(fig[1, 2], hm; label = "θ [K]")
    stream = VideoStream(fig; framerate = 20)
    recordframe!(stream)
    return (; stream, field, time)
end
function record_frame(model, state, dstate, scratch, cookbook, video, t_iter, params)
    video.field[] = interior_diagnostic(:potential_temperature, model, state, dstate, scratch)
    video.time[] = t_iter * params.slice_size
    recordframe!(video.stream)
    return video
end

for (name, model, state, loop) in (("AN2D", an, states.an, CFAnelastics.loop), ("FC2D", fc, states.fc, CFCompressible.loop))
    energies = Float64[]
    frames = video ? (; plot_func = record_frame, plot_obj = start_video(name, model, state)) : (;)
    loop(model, state, CFTimeSchemes.RungeKutta4(model), params; write_func = record_energy, write_obj = energies, frames...)
    println("\n$name: relative change of total energy over $(params.Nslice) slices = ",
        (energies[end] - energies[1]) / abs(energies[1]))
    if video
        file = joinpath(mkpath(joinpath(@__DIR__, "output")), "kelvin_helmholtz_$name.mp4")
        save(file, frames.plot_obj.stream)
        println("$name: video saved to $file")
    end
end
