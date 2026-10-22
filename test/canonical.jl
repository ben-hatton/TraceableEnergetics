# Canonical setups on a model. Included in a model test module (AnelasticTests, CompressibleTests), which includes
# config.jl (fluid, Lz, T0, q0, ρ0, U), helpers.jl (evaluate, all_finite, interior_Xz, interior_xZ) and the model file.
using TraceableEnergetics: TraceableEnergetics, linear_profile, setup, kelvin_helmholtz, rayleigh_benard,
    rising_bubble, lock_exchange

# round-off level of the momentum tendency: B = h - πθ - μq is a difference of O(Cp θ) numbers, divided by Δx
momentum_roundoff(model) = 100 * eps() * fluid.Cp0 * T0 / min(model.dx, model.dz)

# canonical setups, on the periodic and the closed-box configurations
setups(box) = (
    ("rest", setup(box.periodic; θ = linear_profile(box.periodic, T0, T0 + 1), q = linear_profile(box.periodic, q0, q0 - 0.002)), box.periodic),
    ("Kelvin-Helmholtz", kelvin_helmholtz(box.periodic; θ₀ = T0, Δθ = 1.0, q₀ = q0, Δu = U, ρ₀ = ρ0), box.periodic),
    ("Rayleigh-Bénard", rayleigh_benard(box.periodic; θ_bottom = T0 + 1, θ_top = T0, q₀ = q0, ρ₀ = ρ0), box.periodic),
    ("rising bubble", rising_bubble(box.closed; θ₀ = T0, Δθ = 0.5, q₀ = q0), box.closed),
    ("lock exchange", lock_exchange(box.closed; θ_left = T0, θ_right = T0 + 0.5, q_left = q0, q_right = q0), box.closed))

# every setup gives a finite state with finite tendencies; the rest state stays at rest
function test_canonical(init, tend!, periodic, closed)
    for (name, fields, m) in setups((; periodic, closed))
        s = init(m, fields)
        ds, _ = evaluate(tend!, m, s)
        @test all_finite(s) && all_finite(ds)
        if name == "rest"
            @test maximum(abs, ds.u[interior_Xz(m.domain)...]) < momentum_roundoff(m)
            @test maximum(abs, ds.w[interior_xZ(m.domain)...]) < momentum_roundoff(m)
        end
    end
end
