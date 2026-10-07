module CFAnelastics
# Anelastic (binary) fluid model

# ClimFlows modules
using CFPlanets: Tank2D
using CFBoxes
using CFDiffusionSchemes
using CFTimeSchemes: CFTimeSchemes
using ManagedLoops: @vec, @with
using MutatingOrNot: Void, void, similar!
using CookBooks: CookBooks, open, close, CookBook

# Other modules
using LinearAlgebra: mul!
using FFTW: plan_fft!, plan_ifft!, ifft, fft

# Exports
export AnelasticReference
export AN2D
export BuoyancyScheme, BuoyDyn, BuoyDynNew, BuoyDynOld, BuoyThermo, bouyancy
export AvailableReference, vSorted, vpotSorted, SimplePDF, BinarySorted, BinaryPDF
export energetics, available_energetics, construct_state!

include("reference.jl")
include("buoyancy.jl")

## MODEL
struct AN2D{F, Vec, Manager, Fluid}
    mgr::Manager
    domain::Box2D
    space::Tank2D{F}
    boundary::BoundaryConditions2D
    fluid::Fluid
    advection::AdvectionScheme
    buoyancy::BuoyancyScheme
    heatflux::HeatFluxScheme{F}
    viscosity::ViscosityScheme{F}
	reference::AnelasticReference{F, Vec}
    dx::F
    dz::F
    inv_dx::F
    inv_dz::F
end
function AN2D((; mgr, domain, space, fluid, advection_scheme, buoyancy_scheme, heatflux_scheme, viscosity_scheme, anelastic_reference, boundary_conditions))
    Mx, Mz = dims(domain)
    dx, dz = space.Lx / Mx, space.Lz / Mz
    return AN2D(mgr, domain, space, boundary_conditions, fluid, advection_scheme, buoyancy_scheme, heatflux_scheme, viscosity_scheme, anelastic_reference, dx, dz, 1/dx, 1/dz)
end

include("boundaries.jl")
include("initialize.jl")
include("diagnostics.jl")
include("tendencies.jl")
include("pressure.jl")
include("loop.jl")
include("energetics.jl")
include("lorenz_reference_state.jl")
include("available_energetics.jl")

end # module CFAnelastics