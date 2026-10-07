using CFCompressible

#####
##### FC2D configuration and states
#####

function fc_model(; Mx = 32, Mz = 16)
    domain = test_box(Mx, Mz)
    boundary_conditions = BoundaryConditions2D(domain,
        (; u = neumann0(), w = dirichlet0(), T = neumann0(), q = neumann0(), p = neumann0()))
    return FC2D((; mgr, domain, space, fluid, advection_scheme = AdvEnergyCons(), heatflux_scheme = heatflux,
        viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3, bulk_visc = 1e-3), boundary_conditions))
end

fc_tend!(model, dstate, scratch, state) = CFCompressible.tendencies!(dstate, scratch, model, state, 0.0)

# state (m = (ρ, ρθ, ρq), u, w); w vanishes on the walls
function fc_state(model; active = true)
    (; domain, dx, dz) = model
    a = active ? 1.0 : 0.0
    # near-hydrostatic: ρ ≈ ρ0 (1 + g (Lz - z) / c²) with c² = v0 / α_p, plus a small perturbation
    c² = fluid.v0 / fluid.α_p
    ρ(x, z) = ρ0 * (1 + g * (Lz - z) / c² + a * 1e-4 * sin(2π * x / Lx + 0.7) * cos(π * z / Lz))
    θ(x, z) = active ? θ_active(x, z) : θ_rest(z)
    q(x, z) = active ? q_active(x, z) : q_rest(z)
    x(i) = xpoint(domain, dx, i); z(j) = zpoint(domain, dz, j)
    X(I) = Xpoint(domain, dx, I); Z(J) = Zpoint(domain, dz, J)

    m = alloc_xz(Float64, domain, 3)
    u = alloc_Xz(Float64, domain)
    w = alloc_xZ(Float64, domain)
    for i in axes(m, 1), j in axes(m, 2)
        r = ρ(x(i), z(j))
        m[i, j, 1] = r
        m[i, j, 2] = r * θ(x(i), z(j))
        m[i, j, 3] = r * q(x(i), z(j))
    end
    for I in axes(u, 1), j in axes(u, 2)
        u[I, j] = a * U * (cos(π * z(j) / Lz) + 0.3 * sin(2π * X(I) / Lx) * sin(π * z(j) / Lz))
    end
    for i in axes(w, 1), J in axes(w, 2)
        w[i, J] = a * 0.5U * sin(π * Z(J) / Lz) * cos(2π * x(i) / Lx + 0.2)
    end
    w[:, wfaces(domain)[2][[1, end]]] .= 0      # impermeable walls
    return (; m, u, w)
end

# per-cell energy ΔxΔz ρ (K + e(v, θ, q) + Φ), with e = h - p v and K = (avg_x u² + avg_z w²) / 2
function fc_cell_energies(model, (; m, u, w))
    (; domain, dx, dz, fluid) = model
    ic, jc = cells(domain)
    return [begin
        ρ, θ, q = m[i, j, 1], m[i, j, 2] / m[i, j, 1], m[i, j, 3] / m[i, j, 1]
        v = inv(ρ)
        p = fluid(:v, :consvar, :q).pressure(v, θ, q)
        e = fluid(:p, :consvar, :q).specific_enthalpy(p, θ, q) - p * v
        K = (u[i, j]^2 + u[east(domain, i), j]^2 + w[i, j]^2 + w[i, j+1]^2) / 4
        ρ * (K + e + g * zpoint(domain, dz, j)) * dx * dz
    end for i in ic, j in jc]
end

function fc_kinetic(model, (; m, u, w))
    (; domain, dx, dz) = model
    ic, jc = cells(domain)
    return sum(m[i, j, 1] * (u[i, j]^2 + u[east(domain, i), j]^2 + w[i, j]^2 + w[i, j+1]^2) / 4 * dx * dz for i in ic, j in jc)
end

#####
##### Tests
#####

@testset "FC2D" begin
    model = nothing
    @testset "construction" begin
        model = fc_model()
        @test model isa FC2D
    end

    if model === nothing
        @warn "FC2D could not be constructed; remaining FC2D tests skipped"
    else
        active = fc_state(model)
        rest = fc_state(model; active = false)

        @testset "memory safety" begin test_memory_safety(fc_tend!, model, active) end
        @testset "x-shift / x-mirror symmetry" begin test_symmetries(fc_tend!, model, active) end

        @testset "rest state: no horizontal forcing, no mass flux" begin
            dstate, _ = evaluate(fc_tend!, model, rest)
            d = model.domain
            @test maximum(abs, dstate.u[ufaces(d)...]) < momentum_roundoff(model)
            @test maximum(abs, dstate.m[cells(d)..., 1]) == 0
            dm = dstate.m[cells(d)..., :]
            @test maximum(abs, dm .- dm[1:1, :, :]) <= 1e-9 * maximum(abs, dm)     # horizontally uniform
            dw = dstate.w[wfaces(d)...]
            @test maximum(abs, dw .- dw[1:1, :]) < momentum_roundoff(model)
        end

        @testset "impermeable walls" begin test_walls(fc_tend!, model, active) end
        @testset "conservation" begin test_conservation(fc_tend!, model, active, (1, 3), 2) end

        @testset "energy conservation" begin
            r = test_energy(fc_tend!, fc_cell_energies, model, active)
            @info "FC2D relative dE/dt residual" r
        end

        @testset "RK4 integration" begin
            test_integration(model, active, 2e-4, 20, fc_cell_energies, (1, 3), fc_kinetic)
        end

        @testset "no allocations" begin test_no_allocations(fc_tend!, model, active) end

        @testset "diagnostics" begin test_diagnostics(CFCompressible.diagnostics(model), fc_tend!, model, active) end

        @testset "driver loop" begin
            params = (; rest_variable = :θ, experiment = "shear_sin", rest_profile = "tanh", ptop = 1e5,
                T₀ = T0, ΔT = 1.0, δ_T = 0.1, q₀ = q0, Δq = 0.0, δ_q = 0.1, Δu = U, δ_u = 0.1, kick_size = 0.01,
                λ_pert = Lx, Nslice = 2, slice_size = 1e-3, dt_max = 2e-4, cfl = 0.5)
            @test (CFCompressible.loop(model, CFTimeSchemes.RungeKutta4(model), params); true)
        end
    end
end
