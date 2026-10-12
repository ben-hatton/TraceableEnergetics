# Model-independent initial conditions: profiles, flows, streamfunctions, canonical setups

@testset "setups" begin
    @test linear_profile(1.0, 3.0, 1.0)(0.3, 0.5) ≈ 2.0
    @test tanh_profile(1.0, 3.0, 0.5, 0.1)(0.0, 0.5) ≈ 2.0
    @test shear(2.0, 0.5, 0.1)(0.5) == 0 && couette(1.0, 1.0)(1.0) ≈ 1.0 && jet(1.0, 0.5, 0.1)(0.5) == 1.0
    # consvar from θ by ClimFluids, for consvar = θ and consvar = entropy; independent of pressure (entropic)
    params = (; p0 = 1e5, T0 = 280.0, q0 = 0.035, Cp0 = 4000.0, v0 = 1e-3, α_T = 2e-4, α_q = 0.8, α_p = 1e-6,
              M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ = 1e-7)
    fluid_θ, fluid_s = NonlinearBinaryFluid(:potential_temperature, params), NonlinearBinaryFluid(:entropy, params)
    @test consvar_from_theta(fluid_θ, 290.0, 0.03)(0.0, 0.0) == 290.0
    @test consvar_from_theta(fluid_s, 290.0, 0.03)(0.0, 0.0) ≈ fluid_s(:p, :theta, :q).conservative_variable(2e7, 290.0, 0.03)
    # the parts of a model that setups use: periodic in x, walls in z
    model = (; fluid = fluid_θ, domain = Box2D(Mx = 8, Mz = 4, Hx = 1, Hz = 1, boundary = (Periodic, Bounded)),
             space = Tank2D(Lx = 2.0, Lz = 1.0, g = 9.81))
    # built-in streamfunctions vanish on walls and are periodic in periodic directions
    for ψ in (mode(model; amplitude = 1.0, kx = 2, z₀ = 0.5, δ = 0.2), noise(model; amplitude = 1.0))
        @test ψ(0.7, 0.0) == 0 && abs(ψ(0.7, 1.0)) < 1e-15
        @test ψ(0.0, 0.4) ≈ ψ(2.0, 0.4)
    end
    # canonical setups give the fields of the initialisation, with consvar from θ
    kh = kelvin_helmholtz(model; θ₀ = 300.0, Δθ = 1.0, q₀ = 0.035, Δu = 0.1)
    @test keys(kh) == (:consvar, :q, :u, :ψ)
    @test kh.consvar(0.0, 0.5) ≈ 300.0 && kh.u(0.5) == 0
    bubble = rising_bubble(model; θ₀ = 300.0, Δθ = 0.5, q₀ = 0.0)
    @test bubble.consvar(1.0, 0.25) ≈ 300.5 && bubble.consvar(0.1, 0.9) ≈ 300.0
    rest = rest_state(model; θ = 300.0, q = 0.035)
    @test rest.u(0.3) == 0 && rest.ψ(0.5, 0.5) == 0 && rest.q(0.1, 0.2) == 0.035
end
