# FC2D test model and states
using CFCompressible
using CFCompressible: CFTimeSchemes


#=
32 × 16 box with every physical term on: vortex force, thermodynamic diffusion and viscosity, using the
energy-conserving schemes
=#
function fc_model(; mgr = mgr, boundary = (Periodic, Bounded), walls = (; bottom = closed_wall(), top = closed_wall()))
    domain = Box2D(; Mx = 32, Mz = 16, boundary)
    boundary_conditions = BoxBoundaryConditions(domain; walls...)
    return FC2D((; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(),
        thermoflux_scheme = thermoflux, viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3), boundary_conditions))
end

# the model's tendencies!, in the argument order of the generic checks (box_helpers.jl)
fc_tend!(dstate, scratch, model, state) = CFCompressible.tendencies!(dstate, scratch, model, state, 0.0)

#=
active state (m = (ρ, ρθ, ρq), u, w), with no symmetry, so that every term of the tendencies is non-zero;
w vanishes on the walls
=#
function fc_state(model)
    (; domain, dx, dz) = model
    # near-hydrostatic: ρ ≈ ρ0 (1 + g (Lz - z) / c²) with c² = v0 / α_p, plus a small perturbation
    c² = fluid.v0 / fluid.α_p
    ρ(x, z) = ρ0 * (1 + g * (Lz - z) / c² + 1e-4 * sin(2π * x / Lx + 0.7) * cos(π * z / Lz))
    x(i) = cellpoint(domain, dx, 1, i); z(j) = cellpoint(domain, dz, 2, j)
    X(I) = facepoint(domain, dx, 1, I); Z(J) = facepoint(domain, dz, 2, J)
    nx, nz, nX, nZ = ncells(domain, 1), ncells(domain, 2), nfaces(domain, 1), nfaces(domain, 2)
    density = [ρ(x(i), z(j)) for i in 1:nx, j in 1:nz]
    u = [U * (cos(π * z(j) / Lz) + 0.3 * sin(2π * X(I) / Lx) * sin(π * z(j) / Lz)) for I in 1:nX, j in 1:nz]
    w = [0.5U * sin(π * Z(J) / Lz) * cos(2π * x(i) / Lx + 0.2) for i in 1:nx, J in 1:nZ]
    return CFCompressible.initialize(model; density, consvar = θ_active, q = q_active, velocity = (u, w))   # also zeroes the wall faces
end
