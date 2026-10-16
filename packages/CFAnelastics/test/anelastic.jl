using CFAnelastics
using CFTimeSchemes: tendencies!

#####
##### AN2D configuration and states
#####

# reference density increasing with depth Lz - z
an_density(z) = ρ0 * (1 + 0.1 * (Lz - z) / Lz)

# 32 × 16 box with every physical term on: buoyancy, vortex force, thermodynamic diffusion and viscosity, using the
# energy-conserving schemes
function an_model(; mgr = mgr, boundary = (Periodic, Bounded), walls = (; bottom = closed_wall(), top = closed_wall()))
    domain = test_box(32, 16; boundary)
    anelastic_reference = AnelasticReference(domain, space, an_density; ptop = 1e5)
    boundary_conditions = BoxBoundaryConditions(domain; walls...)
    return AN2D((; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(),
        buoyancy_scheme = BuoyThermo(), thermoflux_scheme = thermoflux, viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3),
        anelastic_reference, boundary_conditions))
end

# the model's tendencies!, in the argument order of the generic checks (helpers.jl)
an_tend!(model, dstate, scratch, state) = tendencies!(dstate, scratch, model, state, 0.0)

# active state (m = ρ_R(θ, q), u, w), with no symmetry, so that every term of the tendencies is non-zero; velocities
# from a mass streamfunction ψ vanishing on the walls, so that ∇·(ρ_R u) = 0 discretely: u = ρ_R⁻¹ δz ψ / Δz,
# w = -ρ_R⁻¹ δx ψ / Δx with ψ at cell vertices and ρ_R averaged to the w-faces
function an_state(model)
    (; domain, dx, dz, reference) = model
    (; ρ_R) = reference
    θ, q = θ_active, q_active
    # with walls in x, ψ also vanishes on x = 0, Lx
    wall_x(X) = boundary_topology(domain)[1] isa Bounded ? sin(π * X / Lx) : 1.0
    ψ(X, Z) = ρ0 * U * Lz * wall_x(X) * (sin(π * Z / Lz) * (1 + 0.5 * cos(2π * X / Lx + 0.3)) +
                                        0.3 * sin(2π * Z / Lz) * sin(4π * X / Lx))
    x(i) = cellpoint(domain, dx, 1, i); z(j) = cellpoint(domain, dz, 2, j)
    X(I) = facepoint(domain, dx, 1, I); Z(J) = facepoint(domain, dz, 2, J)

    m = alloc(Float64, nothing, domain, (Cell(), Cell()), 2)
    u = alloc(Float64, nothing, domain, (Face(), Cell()))
    w = alloc(Float64, nothing, domain, (Cell(), Face()))
    for i in axes(m, 1), j in axes(m, 2)
        m[i, j, 1] = ρ_R[j] * θ(x(i), z(j))
        m[i, j, 2] = ρ_R[j] * q(x(i), z(j))
    end
    for I in axes(u, 1), j in axes(u, 2)
        u[I, j] = (ψ(X(I), Z(j + 1)) - ψ(X(I), Z(j))) / (dz * ρ_R[j])
    end
    for i in axes(w, 1), J in faces(domain, 2)
        w[i, J] = -(ψ(X(i + 1), Z(J)) - ψ(X(i), Z(J))) / (dx * ℑ(ρ_R, (2,), (Face(),), J))
    end
    # impermeable walls
    w[:, interior_xZ(domain)[2][[1, end]]] .= 0
    boundary_topology(domain)[1] isa Bounded && (u[interior_Xz(domain)[1][[1, end]], :] .= 0)
    return (; m, u, w)
end

# per-cell energy ΔxΔz [ρ_R K + ρ_R (h(p_R, s, q) + Φ) - p_R],  K = (ℑ(u², (1,), (Cell(),)) + ℑ(w², (2,), (Cell(),))) / 2
function an_cell_energies(model, (; m, u, w))
    (; domain, dx, dz, fluid, reference) = model
    (; ρ_R, p_R) = reference
    ic, jc = interior_xz(domain)
    return [begin
        K = (u[i, j]^2 + u[east(domain, i), j]^2 + w[i, j]^2 + w[i, j+1]^2) / 4
        h = fluid(:p, :consvar, :q).specific_enthalpy(p_R[j], m[i, j, 1] / ρ_R[j], m[i, j, 2] / ρ_R[j])
        Φ = g * (cellpoint(domain, dz, 2, j) - Lz)
        (ρ_R[j] * (K + h + Φ) - p_R[j]) * dx * dz
    end for i in ic, j in jc]
end

# walls exercising every kind of condition (consvar = θ for this fluid)
const forced_box = (;
    left   = Wall(tangential = TangentialVelocity(z -> 0.02 * z), tracers = (Value(T0 + 0.5), Value(q0))),
    right  = Wall(tangential = NoSlip(), tracers = (Value(z -> T0 - 0.5 + 0.1 * z), Flux(1e-6))),
    bottom = Wall(tangential = TangentialVelocity(0.05), tracers = (HeatFlux(x -> 40.0 + 10.0 * cos(2π * x / Lx)), Flux(0.0))),
    top    = Wall(tangential = Stress(x -> 2e-4 * sin(2π * x / Lx)), tracers = (Flux(1e-3), Flux(-1e-6))))

# discrete anelastic divergence ∇·(ρ_R u) at interior cells, and the size of its terms
function an_divergence(model, u, w)
    (; domain, dx, dz) = model
    (; ρ_R) = model.reference
    ic, jc = interior_xz(domain)
    div = [ρ_R[j] * (u[east(domain, i), j] - u[i, j]) / dx + (ℑ(ρ_R, (2,), (Face(),), j+1) * w[i, j+1] - ℑ(ρ_R, (2,), (Face(),), j) * w[i, j]) / dz for i in ic, j in jc]
    scale = max(maximum(abs, ρ_R[j] * u[i, j] / dx for i in ic, j in jc), maximum(abs, ℑ(ρ_R, (2,), (Face(),), j) * w[i, j] / dz for i in ic, j in jc), floatmin())
    return div, scale
end

#####
##### Tests
#####

@testset "AN2D" begin
    model = an_model(mgr = VectorizedCPU())         # closed walls, periodic in x; serial, for the allocation test
    active = an_state(model)
    box = an_model(boundary = (Bounded, Bounded), walls = forced_box)    # threaded: memory safety also catches races

    # energy-conserving configuration: every flux through the walls vanishes, so mass, tracers and energy are conserved
    @testset "closed walls, periodic in x" begin
        div, scale = an_divergence(model, active.u, active.w)
        @test maximum(abs, div) / scale < 1e-13                              # the test state is divergence-free
        test_memory_safety(an_tend!, model, active)
        test_walls(an_tend!, model, active)
        test_conservation(an_tend!, model, active, (; mass = nothing, consvar = 1, q = 2))   # m = (ρ_R θ, ρ_R q)
        r = test_energy(an_tend!, an_cell_energies, model, active)
        @info "AN2D relative dE/dt residual" r
        ds, _ = evaluate(an_tend!, model, active)
        div, scale = an_divergence(model, ds.u, ds.w)
        @test maximum(abs, div) / scale < 1e-11                              # anelastic constraint ∇·(ρ_R ∂ₜu) = 0
        test_no_allocations(an_tend!, model, active)
    end

    # every kind of wall condition: the wall fluxes and the work of the wall stresses enter the budgets exactly
    @testset "forced walls, closed box, threaded" begin
        s = an_state(box)
        test_memory_safety(an_tend!, box, s)
        test_walls(an_tend!, box, s)
        test_conservation(an_tend!, box, s, (; mass = nothing, consvar = 1, q = 2))
        r = test_energy_budget(an_tend!, an_cell_energies, box, s)           # dE/dt = wall input
        @test abs(r.input) > 1e-8                                            # the forcing is felt
        ds, scratch = evaluate(an_tend!, box, s)
        @test wall_input(box, scratch, 2, 1).heat ≈ 40.0 * Lx rtol = 1e-12     # bottom HeatFlux delivers exactly ∫F dx
        div, scale = an_divergence(box, ds.u, ds.w)
        @test maximum(abs, div) / scale < 1e-11                              # anelastic constraint
    end
end
