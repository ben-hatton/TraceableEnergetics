# Recording a run: the outputs of `loop`, here for a warm bubble rising in the anelastic (AN2D) model. Every output is
# recorded at t = 0 and at the end of each time slice, from a single diagnostics session shared by all outputs, and
# `loop` closes the data files and saves the videos when the run ends:
#   - data files, NetCDF (.nc) or JLD2 (.jld2), with the full arrays (halo included) of chosen diagnostics, for
#     post-processing; `read_diagnostic` reads either format back to the same array;
#   - videos, with the interior of one diagnostic, for viewing.
#
#   julia -t 4 examples/recording/run.jl
#
# The outputs are saved to examples/recording/output/.
include("../../tools/setup.jl")

using TraceableEnergetics, CFAnelastics
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes
using CFPlanets: Tank2D

#####
##### Model: a small anelastic box (see examples/kelvin_helmholtz for each choice)
#####

space = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)
T0, q0, ρ0 = 280.0, 0.035, 1000.0
fluid = NonlinearBinaryFluid(:potential_temperature, (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1 / ρ0, α_T = 2e-4,
    α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ = 1e-7))

mgr    = SIMD()      # vectorised loops; tSIMD() (multi-threaded) pays off only on larger grids
domain = Box2D(Mx = 64, Mz = 32, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))
closed = Wall(tangential = FreeSlip(), tracers = (Flux(0.0), Flux(0.0)))

an = AN2D((; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(), buoyancy_scheme = ThermodynamicForcing(),
    thermoflux_scheme = ThermoFluxConsistent(k_T = 1e-4, k_q = 1e-5),
    viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3),
    anelastic_reference = AnelasticReference(domain, space, ρ0; pref = 1e5), gauge_offset = 0.0,
    boundary_conditions = BoxBoundaryConditions(domain; bottom = closed, top = closed)))

# a bubble 2 K warmer than its neutral surroundings: buoyancy g α_T Δθ ≈ 4e-3 m s⁻², enough to rise ~0.1 m in 20 s
state = initialize(an, rising_bubble(an; θ₀ = T0, Δθ = 2.0, q₀ = q0))

#####
##### Outputs
#####

# the diagnostics saved to the data files: any subset of `keys(diagnostic_info(an))`, each on its own grid location
# (e.g. w on the w-faces, (Cell(), Face()), and the vorticity on the vertices, (Face(), Face()))
names = (:u, :w, :conservative_variable, :vorticity)
dir = mkpath(joinpath(@__DIR__, "output"))

path_nc     = joinpath(dir, "recording.nc")
path_jld2   = joinpath(dir, "recording.jld2")
path_θ      = joinpath(dir, "recording_potential_temperature.mp4")
path_ζ      = joinpath(dir, "recording_vorticity.mp4")

# the format of a data file is chosen from its extension, and a video is saved to its `file`, with any keywords of
# `save`; any combination of outputs is recorded in one pass
outputs = Outputs(
    DataFile(path_nc, an, names),        # NetCDF, e.g. for xarray or ncview
    DataFile(path_jld2, an, names),      # JLD2, native Julia
    DiagnosticVideo(an, :potential_temperature; file = path_θ),             # interior only, for viewing
    DiagnosticVideo(an, :vorticity; file = path_ζ, colormap = :balance),
)

#####
##### Run
#####

params = (; Nslice = 40, slice_size = 0.5, dt_max = 0.1, cfl = 0.9)
CFAnelastics.loop(state, an, CFTimeSchemes.RungeKutta4(an), params; outputs)

#####
##### Reading back
#####

# either format reads back to the same (x, z, t) array, with halo, and the times from t = 0
println()
for name in names
    field_nc, t_nc = read_diagnostic(path_nc, name)
    field_jld2, t_jld2 = read_diagnostic(path_jld2, name)
    @assert field_nc == field_jld2 && t_nc == t_jld2
    println("$name: size $(size(field_nc)) on $(diagnostic_info(an)[name].location), t = $(t_nc[1]) … $(t_nc[end]) s")
end
println("outputs saved to $dir")
