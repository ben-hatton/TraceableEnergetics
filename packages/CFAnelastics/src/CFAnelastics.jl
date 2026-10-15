module CFAnelastics
# Anelastic (binary) fluid model

# ClimFlows modules
using CFPlanets: Tank2D
using CFBoxes
using CFBoxes: slice, across
using CFDiffusionSchemes
using CFTimeSchemes: CFTimeSchemes
using ManagedLoops: @vec, @with
using MutatingOrNot: void
using CookBooks: CookBooks, open, close, CookBook


# Exports
export AnelasticReference
export PoissonSolver, pressure!
export AN2D
export BuoyancyScheme, BuoyDyn, BuoyDynNew, BuoyDynOld, BuoyThermo

include("reference.jl")
include("buoyancy.jl")
include("poisson.jl")

## MODEL

"""
    model = AN2D((; mgr, domain, space, fluid, spatial_scheme, vortex_scheme, buoyancy_scheme, thermoflux_scheme,
                    viscosity_scheme, anelastic_reference, boundary_conditions))

Anelastic binary fluid in a 2D box (x, z): prognostic m = ρ_R (consvar, q) at cell centres and the velocity (u, w)
on faces, with ∇·(ρ_R u) = 0. `spatial_scheme` (e.g. `Centred2()`) sets the interpolations and differences of the
C-grid, `vortex_scheme` the vortex force, `buoyancy_scheme` the momentum forcing (see `BuoyancyScheme`).
"""
struct AN2D{F, Vec, Manager, Domain<:Box2D, Boundary<:BoxBoundaryConditions, Fluid, Spatial<:SpatialScheme, Vortex<:VortexScheme,
            Buoyancy<:BuoyancyScheme, ThermoFlux<:ThermoFluxScheme{F}, Poisson<:PoissonSolver}
    mgr::Manager
    domain::Domain
    space::Tank2D{F}
    boundary::Boundary
    fluid::Fluid
    spatial_scheme::Spatial
    vortex_scheme::Vortex
    buoyancy_scheme::Buoyancy
    thermoflux::ThermoFlux
    viscosity::ViscosityScheme{F}
    reference::AnelasticReference{F, Vec}
    poisson::Poisson
    dx::F
    dz::F
    inv_dx::F
    inv_dz::F
end
function AN2D(params)
    (; mgr, domain, space, fluid, spatial_scheme, vortex_scheme, buoyancy_scheme, thermoflux_scheme, viscosity_scheme, anelastic_reference,
       boundary_conditions) = params
    check_halo(domain, spatial_scheme)
    Mx, Mz = dims(domain)
    dx, dz = space.Lx / Mx, space.Lz / Mz
    check_tracers(boundary_conditions, 2)       # tracers (consvar, q)
    boundary = discretize(boundary_conditions, domain, (dx, dz))
    poisson = PoissonSolver(domain, (dx, dz), anelastic_reference.ρ_R)
    return AN2D(mgr, domain, space, boundary, fluid, spatial_scheme, vortex_scheme, buoyancy_scheme, thermoflux_scheme, viscosity_scheme,
                anelastic_reference, poisson, dx, dz, 1/dx, 1/dz)
end

include("boundaries.jl")
include("initialize.jl")
include("diagnostics.jl")
include("tendencies.jl")
include("loop.jl")
include("energetics.jl")

end # module CFAnelastics