# Canonical setups on a model. Included in a model test module (AnelasticTests, CompressibleTests), which defines the
# configuration (Lz, T0, q0, ρ0, U) and the helpers (evaluate, all_finite, momentum_roundoff, interior_Xz, interior_xZ).
using TraceableEnergetics: TraceableEnergetics, linear_profile, rest_state, kelvin_helmholtz, rayleigh_benard,
    rising_bubble, lock_exchange

# canonical setups, on the periodic and the closed-box configurations
setups(box) = (
    ("rest", rest_state(box.periodic; θ = linear_profile(T0, T0 + 1, Lz), q = linear_profile(q0, q0 - 0.002, Lz)), box.periodic),
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
