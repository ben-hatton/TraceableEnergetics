# Linear stability of the vortex-force schemes (diagnostic L): eigenvalues of the Jacobian of the inviscid,
# non-diffusive tendencies of AN2D and FC2D, about base states that are neutral in the continuum:
#
#   uniform      u = U, homogeneous fluid
#   stratified   u = U, uniform stratification (Hollingsworth et al. 1983's basic state, without rotation)
#   shear        u = U tanh((z - Lz/2)/δ) over θ = θ₀ + Δθ/2 tanh((z - Lz/2)/δ), Richardson number > 1/4 everywhere
#
# Any growth (Re λ > 0) is computational. The Jacobian is formed by centred differences over the interior degrees of
# freedom (m, u, w), the walls excluded; the pressure modes of AN2D have λ = 0. The base states have many degenerate
# eigenvalues (every level is advected at the same speed), which round-off splits by ~ √(noise): a spurious Re λ that
# falls as the difference step ε grows (as ε^(-1/2)). Each case is computed with two steps; a genuine growth rate is
# the same with both.
#
#   julia -t 4 experiments/vortex_schemes/linear.jl
#
# Output: a table on screen and in output/linear.txt, and the eigenvalues of the stratified case in output/linear.png.
include("../../tools/setup.jl")

using TraceableEnergetics, CFAnelastics, CFCompressible
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes
using CFPlanets: Tank2D
using MutatingOrNot: void
using LinearAlgebra, Printf, CairoMakie

#####
##### Models: inviscid, non-diffusive, periodic in x, free-slip insulating walls in z
#####

space = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)
T0, q0, ρ0 = 280.0, 0.035, 1000.0
fluid = NonlinearBinaryFluid(:potential_temperature, (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1 / ρ0, α_T = 2e-4,
    α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ = 1e-7))

function models(vortex_scheme; Mx = 16, Mz = 8)
    mgr = plain()
    domain = Box2D(; Mx, Mz, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))
    closed = Wall(tangential = FreeSlip(), tracers = (Flux(0.0), Flux(0.0)))
    boundary_conditions = BoxBoundaryConditions(domain; bottom = closed, top = closed)
    common = (; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme,
              thermoflux_scheme = ThermoFluxConsistent(k_T = 0.0, k_q = 0.0),
              viscosity_scheme = ViscosityScheme(dyn_visc = 0.0), boundary_conditions)
    an = AN2D((; common..., buoyancy_scheme = BuoyThermo(), anelastic_reference = AnelasticReference(domain, space, ρ0; ptop = 1e5)))
    fc = FC2D(common)
    return an, fc
end

#####
##### Base states, as setups (fields for `initialize`)
#####

Lz = space.Lz
base_state(model, ::Val{:uniform}, U) =
    (; consvar = consvar_from_theta(fluid, T0, q0), q = (x, z) -> q0, u = z -> U, ψ = (X, Z) -> zero(X))
base_state(model, ::Val{:stratified}, U; Δθ = 1.0) =
    (; consvar = consvar_from_theta(fluid, linear_profile(T0, T0 + Δθ, Lz), q0), q = (x, z) -> q0, u = z -> U,
       ψ = (X, Z) -> zero(X))
base_state(model, ::Val{:shear}, U; Δθ = 1.0, δ = Lz / 8) =
    (; consvar = consvar_from_theta(fluid, tanh_profile(T0 - Δθ / 2, T0 + Δθ / 2, Lz / 2, δ), q0), q = (x, z) -> q0,
       u = z -> U * tanh((z - Lz / 2) / δ), ψ = (X, Z) -> zero(X))
# minimum Richardson number N² / u_z² of the shear state, N² ≈ g α_T θ_z
richardson_shear(U; Δθ = 1.0, δ = Lz / 8) = space.g * 2e-4 * (Δθ / 2δ) / (U / δ)^2

state0(model::AN2D, fields) = initialize(model, fields)
state0(model::FC2D, fields) = initialize(model, fields; ptop = 1e5)

#####
##### Jacobian over the interior degrees of freedom
#####

# (field, index) of each degree of freedom: tracers at the interior cells, u on the interior faces, w on the interior
# faces off the walls
function dofs((; domain), state)
    x, z, X = interior_cells(domain, 1), interior_cells(domain, 2), interior_faces(domain, 1)
    Z = interior_faces(domain, 2)[2:end-1]
    return [[(:m, CartesianIndex(i, j, n)) for n in axes(state.m, 3) for i in x for j in z];
            [(:u, CartesianIndex(I, j)) for I in X for j in z];
            [(:w, CartesianIndex(i, J)) for i in x for J in Z]]
end

tendency(model, state) = CFTimeSchemes.tendencies!(void, void, model, state, 0.0)[1]
perturbed(state, (field, k), ε) = (s = map(copy, state); getproperty(s, field)[k] += ε; s)

function jacobian(model, state; rel)
    list = dofs(model, state)
    # step: a small fraction of each field's magnitude (per tracer for m)
    scale(field, k) = field == :m ? maximum(abs, view(state.m, :, :, k[3])) : max(maximum(abs, state.u), 1e-2)
    J = zeros(length(list), length(list))
    for (col, dof) in enumerate(list)
        ε = rel * scale(dof...)
        plus, minus = tendency(model, perturbed(state, dof, ε)), tendency(model, perturbed(state, dof, -ε))
        J[:, col] .= [(getproperty(plus, f)[k] - getproperty(minus, f)[k]) / 2ε for (f, k) in list]
    end
    residual = maximum(abs, tendency(model, state).u)          # the base state should be steady
    return J, residual
end

#####
##### Sweep
#####

# max Re λ with two difference steps (relative to each field's magnitude; the tendencies are quadratic in u and
# smooth in m, so large steps are accurate)
function analyse(name, model, base, U; steps = (1e-3, 1e-2))
    state = state0(model, base_state(model, Val(base), U))
    (J_1, residual), (J_2, _) = jacobian(model, state; rel = steps[1]), jacobian(model, state; rel = steps[2])
    λ, λ_2 = eigvals(J_1), eigvals(J_2)
    growth, growth_2 = maximum(real, λ), maximum(real, λ_2)
    return (; name, base, U, growth, growth_2, scale = maximum(abs, λ_2), residual, λ = λ_2)
end

results = []
for scheme in (VortexEnergyCons(), VortexEnstrophyCons(), VortexEnergyEnstrophyCons()), (Mx, Mz) in ((16, 8), (16, 16), (8, 16))
    an, fc = models(scheme; Mx, Mz)
    bases = (Mx, Mz) == (16, 8) ? (:uniform, :stratified, :shear) : (:stratified,)
    for (name, model) in (("AN2D", an), ("FC2D", fc)), base in bases, U in (0.002, 0.005, 0.01)
        r = analyse(name, model, base, U)
        push!(results, (; r..., scheme = nameof(typeof(scheme)), grid = "$(Mx)×$(Mz)"))
    end
end

out = mkpath(joinpath(@__DIR__, "output"))
open(joinpath(out, "linear.txt"), "w") do io
    for stream in (stdout, io)
        @printf(stream, "%-26s %-5s %-6s %-11s %6s %6s %12s %12s %12s %10s %8s\n", "scheme", "model", "grid", "base",
                "U", "Ri", "Re λ, ε₁", "Re λ, ε₂", "max |λ|", "residual", "verdict")
        for r in results
            Ri = r.base == :shear ? richardson_shear(r.U) : Inf
            # noise falls with the step (ratio ~ √10); growth above 1e-9 max|λ| that does not is genuine
            verdict = r.growth_2 < 1e-9 * r.scale ? "neutral" : r.growth_2 < 0.6 * r.growth ? "noise" : "GROWTH"
            @printf(stream, "%-26s %-5s %-6s %-11s %6.3f %6.1f %12.3e %12.3e %12.3e %10.1e %8s\n", r.scheme, r.name,
                    r.grid, r.base, r.U, Ri, r.growth, r.growth_2, r.scale, r.residual, verdict)
        end
    end
end

# eigenvalues of the stratified case, largest U, square cells, per model and scheme
fig = Figure(size = (1300, 700))
for (row, name) in enumerate(("AN2D", "FC2D")), (col, scheme) in enumerate((:VortexEnergyCons, :VortexEnstrophyCons, :VortexEnergyEnstrophyCons))
    r = only(filter(r -> r.name == name && r.scheme == scheme && r.base == :stratified && r.U == 0.01 && r.grid == "16×8", results))
    ax = Axis(fig[row, col]; title = "$name, $scheme", xlabel = "Re λ [s⁻¹]", ylabel = "Im λ [s⁻¹]")
    scatter!(ax, real.(r.λ), imag.(r.λ); markersize = 4)
end
save(joinpath(out, "linear.png"), fig)
println("\nsaved to $out")
