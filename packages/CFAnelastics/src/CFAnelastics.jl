module CFAnelastics
# Anelastic (binary) fluid model

# ClimFlows modules
using CFPlanets: Tank2D
using CFBoxes
using CFBoxes: foreach_dim, pick, slice, across
using CFDiffusionSchemes
using CFTimeSchemes: CFTimeSchemes
using ManagedLoops: @vec, @with
using MutatingOrNot: void
using CookBooks: CookBooks, open, close, CookBook


# Exports
export AnelasticReference
export PoissonSolver, project!
export AN2D
export BuoyancyScheme, BuoyDyn, BuoyDynNew, BuoyDynOld, BuoyThermo

include("reference.jl")
include("buoyancy.jl")
include("poisson.jl")

## MODEL
struct AN2D{F, Vec, Manager, Domain<:Box2D, Boundary<:BoxBoundaryConditions, Fluid, Advection<:AdvectionScheme,
            Buoyancy<:BuoyancyScheme, ThermoFlux<:ThermoFluxScheme{F}, Poisson<:PoissonSolver}
    mgr::Manager
    domain::Domain
    space::Tank2D{F}
    boundary::Boundary
    fluid::Fluid
    advection::Advection
    buoyancy::Buoyancy
    thermoflux::ThermoFlux
    viscosity::ViscosityScheme{F}
    reference::AnelasticReference{F, Vec}
    poisson::Poisson
    dx::F
    dz::F
    inv_dx::F
    inv_dz::F
end
function AN2D((; mgr, domain, space, fluid, advection_scheme, buoyancy_scheme, thermoflux_scheme, viscosity_scheme, anelastic_reference, boundary_conditions))
    Mx, Mz = dims(domain)
    dx, dz = space.Lx / Mx, space.Lz / Mz
    check_tracers(boundary_conditions, 2)       # tracers (consvar, q)
    boundary = discretize(boundary_conditions, domain, (dx, dz))
    poisson = PoissonSolver(domain, (dx, dz), anelastic_reference.ρ_R)
    return AN2D(mgr, domain, space, boundary, fluid, advection_scheme, buoyancy_scheme, thermoflux_scheme, viscosity_scheme,
                anelastic_reference, poisson, dx, dz, 1/dx, 1/dz)
end

include("boundaries.jl")
include("initialize.jl")
include("diagnostics.jl")
include("tendencies.jl")
include("loop.jl")
include("energetics.jl")

end # module CFAnelastics