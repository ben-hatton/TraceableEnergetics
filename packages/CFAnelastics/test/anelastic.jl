using CFAnelastics
using CFTimeSchemes: tendencies!

#####
##### AN2D configuration and states
#####

# reference density increasing with depth Lz - z
an_density(z) = ρ0 * (1 + 0.1 * (Lz - z) / Lz)

function an_model(; Mx = 32, Mz = 16, mgr = mgr, buoyancy_scheme = BuoyThermo(), vortex_scheme = VortexEnergyCons(),
                   thermoflux_scheme = thermoflux, boundary = (Periodic, Bounded), walls = (; bottom = closed_wall(), top = closed_wall()))
    domain = test_box(Mx, Mz; boundary)
    anelastic_reference = AnelasticReference(domain, space, an_density; ptop = 1e5)
    boundary_conditions = BoxBoundaryConditions(domain; walls...)
    return AN2D((; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme, buoyancy_scheme, thermoflux_scheme,
        viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3, bulk_visc = 0.0), anelastic_reference, boundary_conditions))
end

an_tend!(model, dstate, scratch, state) = tendencies!(dstate, scratch, model, state, 0.0)

# state (m = ρ_R(θ, q), u, w); velocities from a mass streamfunction ψ vanishing on the walls, so that
# ∇·(ρ_R u) = 0 discretely: u = ρ_R⁻¹ δz ψ / Δz, w = -ρ_R⁻¹ δx ψ / Δx with ψ at cell vertices and ρ_R averaged to
# the w-faces
function an_state(model; active = true)
    (; domain, dx, dz, reference) = model
    (; ρ_R) = reference
    θ(x, z) = active ? θ_active(x, z) : θ_rest(z)
    q(x, z) = active ? q_active(x, z) : q_rest(z)
    # with walls in x, ψ also vanishes on x = 0, Lx
    wall_x(X) = boundary_topology(domain)[1] isa Bounded ? sin(π * X / Lx) : 1.0
    ψ(X, Z) = active ? ρ0 * U * Lz * wall_x(X) * (sin(π * Z / Lz) * (1 + 0.5 * cos(2π * X / Lx + 0.3)) +
                                                 0.3 * sin(2π * Z / Lz) * sin(4π * X / Lx)) : 0.0
    x(i) = cellpoint(domain, dx, 1, i); z(j) = cellpoint(domain, dz, 2, j)
    X(I) = facepoint(domain, dx, 1, I); Z(J) = facepoint(domain, dz, 2, J)

    m = alloc(Float64, domain, (Cell(), Cell()), 2)
    u = alloc(Float64, domain, (Face(), Cell()))
    w = alloc(Float64, domain, (Cell(), Face()))
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

function an_kinetic(model, (; u, w))
    (; domain, dx, dz) = model
    ρ_R = model.reference.ρ_R
    ic, jc = interior_xz(domain)
    return sum(ρ_R[j] * (u[i, j]^2 + u[east(domain, i), j]^2 + w[i, j]^2 + w[i, j+1]^2) / 4 * dx * dz for i in ic, j in jc)
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
    model = an_model(mgr = VectorizedCPU())         # closed walls, periodic in x
    active = an_state(model)
    box = an_model(boundary = (Bounded, Bounded), walls = forced_box)    # threaded: memory safety also catches races

    @testset "closed walls, periodic in x" begin
        div, scale = an_divergence(model, active.u, active.w)
        @test maximum(abs, div) / scale < 1e-13                              # test state is divergence-free
        test_memory_safety(an_tend!, model, active)
        test_symmetries(an_tend!, model, active)
        test_walls(an_tend!, model, active)
        test_conservation(an_tend!, model, active, (; mass = nothing, consvar = 1, q = 2))
        r = test_energy(an_tend!, an_cell_energies, model, active)
        ds, _ = evaluate(an_tend!, model, active)
        div, scale = an_divergence(model, ds.u, ds.w)
        @test maximum(abs, div) / scale < 1e-11                              # anelastic constraint ∇·(ρ_R ∂ₜu) = 0
        test_no_allocations(an_tend!, model, active)
        @info "AN2D relative dE/dt residual" r
    end

    @testset "forced walls, closed box, threaded" begin
        s = an_state(box)
        test_memory_safety(an_tend!, box, s)
        test_walls(an_tend!, box, s)
        test_conservation(an_tend!, box, s, (; mass = nothing, consvar = 1, q = 2))   # wall fluxes enter the budgets
        r = test_energy_budget(an_tend!, an_cell_energies, box, s)
        @test abs(r.input) > 1e-8                                            # the forcing is felt
        _, scratch = evaluate(an_tend!, box, s)
        @test wall_input(box, scratch, 2, 1).heat ≈ 40.0 * Lx rtol = 1e-12     # HeatFlux delivers exactly ∫F dx
        ds, _ = evaluate(an_tend!, box, s)
        div, scale = an_divergence(box, ds.u, ds.w)
        @test maximum(abs, div) / scale < 1e-11
    end

    @testset "schemes" begin
        # the other advection and heat flux schemes, together (independent code paths): same properties, except
        # energy conservation, which they don't have by design (VortexEnstrophyCons conserves enstrophy; ThermoFluxDowngradient
        # has no heating, σ = 0)
        variant = an_model(mgr = VectorizedCPU(), vortex_scheme = VortexEnstrophyCons(), thermoflux_scheme = ThermoFluxDowngradient(k_T = 1e-4, k_q = 1e-5))
        test_memory_safety(an_tend!, variant, active; reuse = false)
        test_symmetries(an_tend!, variant, active)
        test_walls(an_tend!, variant, active)
        test_conservation(an_tend!, variant, active, (; mass = nothing, consvar = 1, q = 2))
        # dynamic buoyancy (BuoyDynNew): energy residual O(Δz²)
        function residual(m)
            s = an_state(m)
            dE, scale = energy_tendency(an_cell_energies, m, s, first(evaluate(an_tend!, m, s)))
            return abs(dE) / scale
        end
        new = (residual(an_model(mgr = VectorizedCPU(), buoyancy_scheme = BuoyDynNew())),
               residual(an_model(mgr = VectorizedCPU(), buoyancy_scheme = BuoyDynNew(), Mx = 64, Mz = 32)))
        @info "AN2D BuoyDynNew relative dE/dt residual (Δ, Δ/2)" new
        @test new[2] / new[1] < 0.4
    end

    @testset "reference state" begin
        domain = test_box()
        ic = interior_cells(domain, 2)
        dz = Lz / dims(domain)[2]
        zc = [(k - 1/2) * dz for k in 1:dims(domain)[2]]
        # a vector of cell values gives the same interior profiles as the function it samples
        from_function = AnelasticReference(domain, space, an_density; ptop = 1e5)
        from_vector = AnelasticReference(domain, space, an_density.(zc); ptop = 1e5)
        @test from_vector.ρ_R[ic] == from_function.ρ_R[ic]
        @test from_vector.p_R[ic] ≈ from_function.p_R[ic] rtol = 1e-14
        @test_throws ErrorException AnelasticReference(domain, space, an_density.(zc[2:end]); ptop = 1e5)
    end

    @testset "initial states" begin
        # at rest, stratified, between closed walls: stays at rest
        s = CFAnelastics.initialize(model; consvar = (x, z) -> θ_rest(z), q = (x, z) -> q_rest(z))
        ds, _ = evaluate(an_tend!, model, s)
        @test maximum(abs, ds.u[interior_Xz(model.domain)...]) < momentum_roundoff(model)
        @test maximum(abs, ds.w[interior_xZ(model.domain)...]) < momentum_roundoff(model)
        # perturbation from a streamfunction vanishing on all walls: discretely divergence-free, no flow through walls;
        # with a mean flow when periodic in x
        ψ(X, Z) = 1e-3 * ρ0 * U * Lz * sin(2π * X / Lx) * sin(π * Z / Lz)
        for (m, mean_flow) in ((model, z -> U * tanh(z / Lz - 1/2)), (box, z -> zero(z)))
            u′, w′ = CFAnelastics.divergence_free_velocity(m, ψ)
            @test maximum(abs, w′[:, interior_xZ(m.domain)[2][[1, end]]]) < 1e-15 * maximum(abs, w′)
            ū = [mean_flow(cellpoint(m.domain, m.dz, 2, j)) for I in axes(u′, 1), j in axes(u′, 2)]
            s = CFAnelastics.initialize(m; consvar = θ_active, q = q_active, velocity = (u′ .+ ū, w′))
            div, scale = an_divergence(m, s.u, s.w)
            @test all_finite(s) && maximum(abs, div) <= 1e-13 * scale
        end
    end

    @testset "RK4 integration" begin
        s = test_integration(model, active, 0.01, 20, an_cell_energies, (2,), an_kinetic)
        div, scale = an_divergence(model, s.u, s.w)
        @test maximum(abs, div) / scale < 1e-11
    end

    @testset "diagnostics" begin
        test_diagnostics((; book = CFAnelastics.diagnostics(model), internal = CFAnelastics.internal_diagnostics(model),
                          info = CFAnelastics.diagnostic_info(model)), an_tend!, model, active, fluid_params)
        # the fused mass divergence equals the discrete anelastic divergence ∇·(ρ_R u)
        session = open(CFAnelastics.diagnostics(model); state = active, model)
        div, scale = an_divergence(model, active.u, active.w)
        @test maximum(abs, CookBooks.get(session, :mass_divergence)[interior_xz(model.domain)...] - div) / scale < 1e-13
        _, scratch = evaluate(an_tend!, model, active)
        es = CFAnelastics.energetics_tendencies!(void, model, copy_state(active), scratch)
        @test sum(es.energies.te[interior_xz(model.domain)...]) * model.dx * model.dz ≈ sum(an_cell_energies(model, active)) rtol = 1e-12
    end

    @testset "driver loop" begin
        params = (; Nslice = 2, slice_size = 0.02, dt_max = 0.01, cfl = 0.5)
        @test all_finite(CFAnelastics.loop(model, active, CFTimeSchemes.RungeKutta4(model), params))
    end
end
