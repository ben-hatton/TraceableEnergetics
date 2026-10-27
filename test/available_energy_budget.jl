# Tests of the budgets of the available compressible energy (ACE) Π₁ and the available potential energy (APE) Π₂
# (src/reference_state/budget.jl), for AN2D and FC2D, on small grids. Included in the module of the reference-state
# tests, after test/reference_state.jl, whose models, states and helpers it uses.
using TraceableEnergetics: potential_difference, domain_integral

# the active tracers of config.jl and a cellular flow, ψ = 0.01 sin(2πx/Lx) sin²(πz/Lz) [m² s⁻¹], zero at the walls
flow_fields = (; consvar = θ_active, q = q_active, u = z -> 0.0, ψ = (X, Z) -> 0.01 * sin(2π * X / Lx) * sin(π * Z / Lz)^2)
flow_state(model::AN2D) = initialize(model, flow_fields)
flow_state(model::FC2D) = initialize(model, flow_fields; pref = 1e5)

# an open diagnostics session of the book with the Lorenz reference state, for the state with flow, and its fields on the
# interior cells
function budget_session(model)
    state = flow_state(model)
    dstate, scratch = tendencies!(void, void, model, state, zero(model.dx))
    return open(diagnostics(model; reference_state = lorenz(model)); state, dstate, scratch, model)
end
interior_field(session, name) = interior(session.model.domain, CookBooks.get(session, name))
mean_abs(a) = sum(abs, a) / length(a)

# The tendencies of ρΠ₁ and ρΠ₂ at a fixed reference state, from those of the model's densities: B ∂ₜρ + ΔΣ_ϑ ∂ₜ(ρϑ) +
# ΔΣ_q ∂ₜ(ρq), with ∂ₜρ = 0 for AN2D. The budgets close at each cell to round-off, by the product rule of the operators,
# the APE budget of AN2D up to the divergence of ρ_R u: their residuals are measured against the largest of their terms
@testset "closure of the budgets of ACE and APE, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    model = make(Float64; Mx = 16)
    session = budget_session(model)
    f(name) = interior_field(session, name)
    dρ = model isa FC2D ? f(:dρ) : zero(f(:dρconsvar))
    for (E, at_state, at_other) in ((:ACE, "", "_at_p_star"), (:APE, "_at_p_star", "_at_z_star"))
        # the Bernoulli coefficient and the differences of the potentials of this energy
        B, Δϑ, Δq = f(Symbol(:bernoulli_potential_, E)), f(Symbol(:conjugate_variable_, E)), f(Symbol(:modified_chemical_potential_, E))
        heating(name, at) = f(Symbol(name, at))
        terms = (B .* dρ, Δϑ .* f(:dρconsvar), Δq .* f(:dρq), f(Symbol(:div_, E, :_flux_rev)), f(Symbol(:div_, E, :_flux_irr)),
                 f(Symbol(:conversion_, E)), heating(:heating_consvar, at_state) .- heating(:heating_consvar, at_other),
                 heating(:heating_composition, at_state) .- heating(:heating_composition, at_other),
                 heating(:heating, at_other) .- heating(:heating, at_state))
        residual = reduce(.+, terms)
        scale = maximum(t -> maximum(abs, t), terms)
        @test maximum(abs, residual) <= 1e-6 * scale
        @test scale > 0
    end
    # the conversion from P to K is that from ACE and APE and that from the background, C = C_ACE + C_APE + C_bg
    @test f(:conversion) ≈ f(:conversion_ACE) .+ f(:conversion_APE) .+ f(:background_conversion) atol = 1e-12 * maximum(abs, f(:conversion))
    close(session)
end

# AN2D: Σ_ϑ, Σ_q depend on the pressure through P_R only, so the potentials of the ACE are zero, those at the intermediate
# state are the model's own, and so are the heatings there; the conversion from ACE is that of the pressure flux,
# C_ACE = -ρ_R u·∇((p - P_*)/ρ_R) ≈ -∇·J_C*, and integrates to zero over the closed box
@testset "ACE terms of AN2D" begin
    model = an_box(Float64; Mx = 16)
    session = budget_session(model)
    f(name) = interior_field(session, name)
    @test all(iszero, f(:conjugate_variable_ACE)) && all(iszero, f(:modified_chemical_potential_ACE))
    @test all(iszero, f(:buoyancy_ACE)) && all(iszero, f(:pressure_dilatation_star))
    @test f(:conjugate_variable_at_p_star) == f(:conjugate_variable)
    @test f(:heating_consvar_at_p_star) == f(:heating_consvar) && f(:heating_at_p_star) == f(:heating)
    @test abs(domain_integral(model, CookBooks.get(session, :conversion_ACE))) <= 1e-12 * domain_integral(model, abs.(CookBooks.get(session, :conversion)))
    close(session)
end

# The forms of the energy notes, C_ACE = C_b,1* + C_D* - ∇·J_C* and C_APE = C_b,2*, and C_bg = 0, hold in the continuous
# equations: their residuals converge with the grid, the ACE one at second order, the APE one and C_bg at about second
# order in the mean once the grid resolves the flow (the reference specific volume v_* is only continuous, so the
# reference heights z_* have kinks in (ϑ, q), and the maximum converges at first order)
@testset "convergence of the forms of the energy notes, $M" for (M, make) in (("AN2D", an_box), ("FC2D", fc_box))
    errors = map((8, 16, 32)) do Mz
        session = budget_session(make(Float64; Mx = 2Mz, Mz))
        f(name) = interior_field(session, name)
        e = (ACE = mean_abs(f(:ACE_conversion_residual)) / mean_abs(f(:conversion)),
             APE = mean_abs(f(:APE_conversion_residual)) / mean_abs(f(:conversion_APE)),
             background = mean_abs(f(:background_conversion)) / mean_abs(f(:conversion)))
        close(session)
        e
    end
    # at least first order over the two halvings of Δ (between first and second order on these coarse grids, second order
    # beyond)
    for name in (:ACE, :APE, :background)
        @test errors[1][name] / errors[3][name] > 4
    end
    @test errors[3].ACE < 1e-3 && errors[3].APE < 0.03 && errors[3].background < 0.03
end

# The differences of the potentials by quadrature, ∫ ∂v/∂ϑ dp and ∫ ∂v/∂q dp, keep the relative precision of Float64,
# for Δp from 10⁻² Pa to 10⁴ Pa of either sign: they are compared with the differences of the conjugate variable and the
# modified chemical potential of ClimFluids, computed in BigFloat at four times the precision of Float64, as the
# difference of two values of h_ϑ (about 4000 J kg⁻¹ K⁻¹) for a change of about 10⁻⁹ loses twelve digits
@testset "differences of the potentials by quadrature" begin
    fluid = reference_fluid(Float64)
    P, ϑ, q = 1e5, T0 + 1, q0 - 0.001
    for Δp in (1e-2, 1, 1e2, 1e4), sign in (1, -1), (c, potential) in ((3, :conjugate_variable), (4, :modified_chemical_potential))
        Δ = sign * Δp
        # over the interval from P to P + Δ exactly, which P + Δ in Float64 would round
        exact = setprecision(BigFloat, 4 * precision(Float64)) do
            Σ = getproperty(reference_fluid(BigFloat)(:p, :consvar, :q), potential)
            Σ(big(P) + big(Δ), big(ϑ), big(q)) - Σ(big(P), big(ϑ), big(q))
        end
        @test potential_difference(fluid, P, Δ, ϑ, q, c) ≈ exact rtol = 64eps()
    end
    @test potential_difference(fluid, P, 0.0, ϑ, q, 3) == 0
end
