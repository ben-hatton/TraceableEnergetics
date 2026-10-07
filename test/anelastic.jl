using CFAnelastic
using CFTimeSchemes: tendencies!

#####
##### AN2D configuration and states
#####

# reference density increasing with depth z′ = Lz - z (`density_profile` takes depth)
an_density(depth) = ρ0 * (1 + 0.1 * depth / Lz)

function an_model(; Mx = 32, Mz = 16, buoyancy_scheme = BuoyThermo())
    domain = test_box(Mx, Mz)
    anelastic_reference = AnelasticReference(; domain, space, density_profile = an_density, ptop = 1e5)
    boundary_conditions = BoundaryConditions2D(domain, (; u = neumann0(), w = dirichlet0(), T = neumann0(), q = neumann0()))
    return AN2D((; mgr, domain, space, fluid, advection_scheme = AdvEnergyCons(), buoyancy_scheme,
        heatflux_scheme = heatflux, viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3, bulk_visc = 0.0),
        anelastic_reference, boundary_conditions))
end

an_tend!(model, dstate, scratch, state) = tendencies!(dstate, scratch, model, state, 0.0)

# state (m = ρ_R(θ, q), u, w); velocities from a mass streamfunction ψ vanishing on the walls, so that
# ∇·(ρ_R u) = 0 discretely: u = ρ_R⁻¹ δz ψ / Δz, w = -ρ_R_J⁻¹ δx ψ / Δx with ψ at cell vertices
function an_state(model; active = true)
    (; domain, dx, dz, reference) = model
    (; ρ_R, ρ_R_J) = reference
    θ(x, z) = active ? θ_active(x, z) : θ_rest(z)
    q(x, z) = active ? q_active(x, z) : q_rest(z)
    ψ(X, Z) = active ? ρ0 * U * Lz * (sin(π * Z / Lz) * (1 + 0.5 * cos(2π * X / Lx + 0.3)) +
                                     0.3 * sin(2π * Z / Lz) * sin(4π * X / Lx)) : 0.0
    x(i) = xpoint(domain, dx, i); z(j) = zpoint(domain, dz, j)
    X(I) = Xpoint(domain, dx, I); Z(J) = Zpoint(domain, dz, J)

    m = alloc_xz(Float64, domain, 2)
    u = alloc_Xz(Float64, domain)
    w = alloc_xZ(Float64, domain)
    for i in axes(m, 1), j in axes(m, 2)
        m[i, j, 1] = ρ_R[j] * θ(x(i), z(j))
        m[i, j, 2] = ρ_R[j] * q(x(i), z(j))
    end
    for I in axes(u, 1), j in axes(u, 2)
        u[I, j] = (ψ(X(I), Z(j + 1)) - ψ(X(I), Z(j))) / (dz * ρ_R[j])
    end
    for i in axes(w, 1), J in axes(w, 2)
        w[i, J] = -(ψ(X(i + 1), Z(J)) - ψ(X(i), Z(J))) / (dx * ρ_R_J[J])
    end
    w[:, wfaces(domain)[2][[1, end]]] .= 0      # impermeable walls
    return (; m, u, w)
end

# per-cell energy ΔxΔz [ρ_R K + ρ_R (h(p_R, s, q) + Φ) - p_R],  K = (avg_x u² + avg_z w²) / 2
function an_cell_energies(model, (; m, u, w))
    (; domain, dx, dz, fluid, reference) = model
    (; ρ_R, p_R) = reference
    ic, jc = cells(domain)
    return [begin
        K = (u[i, j]^2 + u[east(domain, i), j]^2 + w[i, j]^2 + w[i, j+1]^2) / 4
        h = fluid(:p, :consvar, :q).specific_enthalpy(p_R[j], m[i, j, 1] / ρ_R[j], m[i, j, 2] / ρ_R[j])
        Φ = g * (zpoint(domain, dz, j) - Lz)
        (ρ_R[j] * (K + h + Φ) - p_R[j]) * dx * dz
    end for i in ic, j in jc]
end

function an_kinetic(model, (; u, w))
    (; domain, dx, dz) = model
    ρ_R = model.reference.ρ_R
    ic, jc = cells(domain)
    return sum(ρ_R[j] * (u[i, j]^2 + u[east(domain, i), j]^2 + w[i, j]^2 + w[i, j+1]^2) / 4 * dx * dz for i in ic, j in jc)
end

# discrete anelastic divergence ∇·(ρ_R u) at interior cells, and the size of its terms
function an_divergence(model, u, w)
    (; domain, dx, dz) = model
    (; ρ_R, ρ_R_J) = model.reference
    ic, jc = cells(domain)
    div = [ρ_R[j] * (u[east(domain, i), j] - u[i, j]) / dx + (ρ_R_J[j+1] * w[i, j+1] - ρ_R_J[j] * w[i, j]) / dz for i in ic, j in jc]
    scale = max(maximum(abs, ρ_R[j] * u[i, j] / dx for i in ic, j in jc), maximum(abs, ρ_R_J[j] * w[i, j] / dz for i in ic, j in jc), floatmin())
    return div, scale
end

#####
##### Tests
#####

@testset "AN2D" begin
    model = an_model()
    active = an_state(model)
    rest = an_state(model; active = false)

    @testset "initial state is discretely divergence-free" begin
        div, scale = an_divergence(model, active.u, active.w)
        @test maximum(abs, div) / scale < 1e-13
    end

    @testset "memory safety" begin test_memory_safety(an_tend!, model, active) end
    @testset "x-shift / x-mirror symmetry" begin test_symmetries(an_tend!, model, active) end

    @testset "rest state stays at rest" begin
        dstate, _ = evaluate(an_tend!, model, rest)
        d = model.domain
        @test maximum(abs, dstate.u[ufaces(d)...]) < momentum_roundoff(model)
        @test maximum(abs, dstate.w[wfaces(d)...]) < momentum_roundoff(model)
        dm = dstate.m[cells(d)..., :]
        @test maximum(abs, dm .- dm[1:1, :, :]) <= 1e-9 * maximum(abs, dm)   # horizontally uniform
    end

    @testset "anelastic constraint ∇·(ρ_R ∂ₜu) = 0" begin
        dstate, _ = evaluate(an_tend!, model, active)
        div, scale = an_divergence(model, dstate.u, dstate.w)
        @test maximum(abs, div) / scale < 1e-11
    end

    @testset "impermeable walls" begin test_walls(an_tend!, model, active) end
    @testset "conservation" begin test_conservation(an_tend!, model, active, (2,), 1) end

    @testset "energy conservation (BuoyThermo)" begin
        r = test_energy(an_tend!, an_cell_energies, model, active)
        @info "AN2D BuoyThermo relative dE/dt residual" r
    end

    @testset "energy residual of dynamic buoyancy forms" begin
        function res(scheme, Mx, Mz)
            m = an_model(; Mx, Mz, buoyancy_scheme = scheme)
            s = an_state(m)
            ds, _ = evaluate(an_tend!, m, s)
            total, scale = energy_tendency(an_cell_energies, m, s, ds)
            return abs(total) / scale
        end
        new = (res(BuoyDynNew(), 32, 16), res(BuoyDynNew(), 64, 32))
        old = (res(BuoyDynOld(), 32, 16), res(BuoyDynOld(), 64, 32))
        @info "AN2D dynamic buoyancy relative dE/dt residual (Δ, Δ/2)" BuoyDynNew = new BuoyDynOld = old
        @test new[2] / new[1] < 0.4   # O(Δz²) convergence expected for BuoyDynNew
    end

    @testset "RK4 integration" begin
        s = test_integration(model, active, 0.01, 20, an_cell_energies, (2,), an_kinetic)
        div, scale = an_divergence(model, s.u, s.w)
        @test maximum(abs, div) / scale < 1e-11
    end

    @testset "no allocations" begin test_no_allocations(an_tend!, model, active) end

    @testset "diagnostics" begin test_diagnostics(CFAnelastic.diagnostics(model), an_tend!, model, active) end

    @testset "energetics diagnostics match test energy" begin
        s = copy_state(active)
        _, scratch = evaluate(an_tend!, model, s)
        es = CFAnelastic.energetics_tendencies!(void, model, s, scratch)
        te = sum(es.energies.te[cells(model.domain)...]) * model.dx * model.dz
        @test te ≈ sum(an_cell_energies(model, active)) rtol = 1e-12
    end

    @testset "driver loop" begin
        ic = (; u = (x, z) -> 0.0, w = (x, z) -> 0.0, consvar = θ_active, q = q_active)
        params = (; Nslice = 2, slice_size = 0.02, dt_max = 0.01, cfl = 0.5)
        @test (CFAnelastic.loop(model, ic, CFTimeSchemes.RungeKutta4(model), params); true)
    end
end
