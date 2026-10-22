# Helpers of tools/scales.jl: closed-form identities and hand values

@testset "scales" begin
    # nondimensional numbers
    @test peclet(2.0, 3.0, 0.5) ≈ reynolds(2.0, 3.0, 0.1) * prandtl(0.1, 0.5)
    @test richardson_bulk(4.0, 2.0, 3.0) ≈ 1 / froude(2.0, 2.0, 3.0)^2
    @test richardson_bulk_jump(2.0, 2.0, 3.0) ≈ richardson_bulk(2.0 / 3.0, 2.0, 3.0)
    @test rayleigh(9.81, 2e-4, 1.0, 0.1, 1e-6, 1e-7) ≈ rayleigh(9.81 * 2e-4, 0.1, 1e-6, 1e-7)
    @test atwood(1.0, 3.0) ≈ 0.5 && reduced_gravity(10.0, 1.0, 2.0) ≈ 5.0
    # stratification: density form with c² = Inf is the Boussinesq form; linear EOS ρ = ρ0 (1 - α θ + β S)
    g, ρ0, α, β, dθdz, dSdz = 9.81, 1000.0, 2e-4, 7.6e-4, 0.01, 0.002
    dρdz = ρ0 * (-α * dθdz + β * dSdz)
    @test buoyancy_frequency2_from_density(g, ρ0, dρdz) ≈ buoyancy_frequency2_from_coefficients(g, α, dθdz, β, dSdz)
    @test buoyancy_frequency2_from_density(g, ρ0, dρdz; c² = 1500.0^2) ≈ buoyancy_frequency2_from_density(g, ρ0, dρdz) - g^2 / 1500.0^2
    @test buoyancy_frequency2_from_coefficients(g, α, dθdz) ≈ g * α * dθdz
    @test buoyancy_frequency(4.0) == 2.0 && isnan(buoyancy_frequency(-1.0)) && buoyancy_period(4.0) ≈ π
    @test shear2(3.0, 4.0) == 25.0 && richardson_gradient(1.0, 4.0) ≈ 0.25 == RICHARDSON_CRITICAL && shear_ratio(4.0, 1.0) == 4.0
    # tanh shear layer u = U tanh(z/δ) over θ = Δθ/2 tanh(z/δ): Ri(0) = g α (Δθ/2δ) / (U/δ)²
    U, δ, Δθ = 0.5, 0.1, 1.0
    z = range(-δ, δ; length = 2001)
    f = reshape(U .* tanh.(z ./ δ), 1, :)
    @test vertical_gradient(f, step(z))[1, 1001] ≈ U / δ rtol = 1e-6
    @test richardson_gradient(buoyancy_frequency2_from_coefficients(g, α, Δθ / 2δ), shear2(U / δ)) ≈ g * α * (Δθ / 2δ) / (U / δ)^2
    # turbulence scales
    ν, ε, D = 1e-6, 1e-8, 1e-8
    η = kolmogorov_length(ν, ε)
    @test η ≈ (1e-18 / 1e-8)^(1 / 4)
    @test kolmogorov_velocity(ν, ε) * η / ν ≈ 1                         # Reynolds number of the Kolmogorov scale is 1
    @test kolmogorov_time(ν, ε) ≈ η / kolmogorov_velocity(ν, ε)
    @test batchelor_length(ν, ε, D) ≈ η / sqrt(schmidt(ν, D)) && batchelor_wavenumber(ν, ε, D) ≈ 1 / batchelor_length(ν, ε, D)
    N² = 1e-4
    @test ozmidov_scale(ε, N²) / η ≈ ozmidov_kolmogorov_ratio(ε, ν, N²) && buoyancy_reynolds(ε, ν, N²) ≈ 1e2
    @test taylor_microscale(ν, 0.1, ε) ≈ sqrt(15 * ν * 0.01 / ε)
    # broadcasting and type stability
    @test kolmogorov_length.(ν, [1e-8, 1e-6]) ≈ (ν^3 ./ [1e-8, 1e-6]) .^ (1 / 4)
    @inferred kolmogorov_length(1e-6, 1e-8)
    @inferred buoyancy_frequency2_from_density(9.81, 1000.0, -0.1)
end
