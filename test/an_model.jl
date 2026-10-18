# AN2D test model and states
using CFAnelastics
using CFTimeSchemes: tendencies!


# reference density increasing with depth Lz - z
an_density(z) = ρ0 * (1 + 0.1 * (Lz - z) / Lz)

#=
32 × 16 box with every physical term on: buoyancy, vortex force, thermodynamic diffusion and viscosity, using the
energy-conserving schemes
=#
function an_model(; mgr = mgr, boundary = (Periodic, Bounded), walls = (; bottom = closed_wall(), top = closed_wall()))
    domain = Box2D(; Mx = 32, Mz = 16, boundary)
    anelastic_reference = AnelasticReference(domain, space, an_density; ptop = 1e5)
    boundary_conditions = BoxBoundaryConditions(domain; walls...)
    return AN2D((; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(),
        buoyancy_scheme = ThermodynamicForcing(), thermoflux_scheme = thermoflux, viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3),
        anelastic_reference, boundary_conditions))
end

# the model's tendencies!, in the argument order of the generic checks (helpers.jl)
an_tend!(dstate, scratch, model, state) = tendencies!(dstate, scratch, model, state, 0.0)

#=
active state (m = ρ_R(θ, q), u, w), with no symmetry, so that every term of the tendencies is non-zero; velocities
from a mass streamfunction ψ vanishing on the walls, so that ∇·(ρ_R u) = 0 discretely
=#
function an_state(model)
    # with walls in x, ψ also vanishes on x = 0, Lx
    wall_x(X) = boundary_topology(model.domain)[1] isa Bounded ? sin(π * X / Lx) : 1.0
    ψ(X, Z) = ρ0 * U * Lz * wall_x(X) * (sin(π * Z / Lz) * (1 + 0.5 * cos(2π * X / Lx + 0.3)) +
                                        0.3 * sin(2π * Z / Lz) * sin(4π * X / Lx))
    return CFAnelastics.initialize(model; consvar = θ_active, q = q_active,
                                   velocity = CFAnelastics.divergence_free_velocity(model, ψ))
end
