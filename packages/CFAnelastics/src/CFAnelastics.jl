"""
    CFAnelastics

Anelastic binary fluid in a box: the AN2D model (`AN2D`), its reference state (`AnelasticReference`), the Poisson
solver for its pressure (`PoissonSolver`) and its momentum forcing (`BuoyancyScheme`).
"""
module CFAnelastics

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
export BuoyancyScheme, BuoyancyForce, ConsistentBuoyancy, LegacyBuoyancy, ThermodynamicForcing

include("reference.jl")
include("buoyancy.jl")
include("poisson.jl")

# Model

"""
    model = AN2D((; mgr, domain, space, fluid, spatial_scheme, vortex_scheme, buoyancy_scheme, thermoflux_scheme,
                    viscosity_scheme, anelastic_reference, boundary_conditions))

Anelastic binary fluid in a 2D box (x, z): prognostic m = ρ_R (consvar, q) at cell centers and the velocity (u, w)
on faces, with ∇·(ρ_R u) = 0. `spatial_scheme` (e.g. `Centred2()`) sets the interpolations and differences of the
C-grid, `vortex_scheme` the vortex force, `buoyancy_scheme` the momentum forcing (see `BuoyancyScheme`).
"""
struct AN2D{F, Vec, Manager, Domain<:Box2D, Boundary<:BoxBoundaryConditions, Fluid, Spatial<:SpatialScheme, Vortex<:VortexScheme,
            Buoyancy<:BuoyancyScheme, ThermoFlux<:ThermoFluxScheme{F}, Poisson<:PoissonSolver}
    "loop manager running the kernels (e.g. `MultiThread(VectorizedCPU())`), or `nothing`"
    mgr::Manager
    "the box (grid sizes, halo widths, topology)"
    domain::Domain
    "the physical box: lengths Lx, Lz [m] and gravity g [m s⁻²]"
    space::Tank2D{F}
    "wall conditions, discretized on the grid"
    boundary::Boundary
    "equation of state of the binary fluid"
    fluid::Fluid
    "interpolations and differences of the C-grid (`operators`)"
    spatial_scheme::Spatial
    "discretization of the vortex force ω×u"
    vortex_scheme::Vortex
    "momentum forcing by the thermodynamic state (see `BuoyancyScheme`)"
    buoyancy_scheme::Buoyancy
    "diffusion of consvar and q (coefficients k_T, k_q [m² s⁻¹])"
    thermoflux::ThermoFlux
    "viscosity (dynamic viscosity μ [Pa s])"
    viscosity::ViscosityScheme{F}
    "anelastic reference state ρ_R, p_R"
    reference::AnelasticReference{F, Vec}
    "solver for the anelastic pressure"
    poisson::Poisson
    "grid spacing along x [m]"
    dx::F
    "grid spacing along z [m]"
    dz::F
    "1/dx [m⁻¹]"
    inv_dx::F
    "1/dz [m⁻¹]"
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