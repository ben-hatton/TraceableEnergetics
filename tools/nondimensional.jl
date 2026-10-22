## NONDIMENSIONAL PARAMETERS
# U velocity scale [m/s], L length scale [m], ν kinematic viscosity [m²/s], κ thermal diffusivity [m²/s],
# D scalar diffusivity [m²/s], N buoyancy frequency [1/s], g gravity [m/s²], c sound speed [m/s].

"Reynolds number `U L / ν` [1]."
reynolds(U, L, ν) = U * L / ν

"Prandtl number `ν / κ` [1]."
prandtl(ν, κ) = ν / κ

"Schmidt number `ν / D` [1], for a scalar (e.g. salinity) of diffusivity `D`."
schmidt(ν, D) = ν / D

"Péclet number `U L / κ` [1] (= Re Pr), for a scalar of diffusivity `κ`."
peclet(U, L, κ) = U * L / κ

"Froude number `U / (N L)` [1] (internal Froude number; `N` is the buoyancy frequency)."
froude(U, N, L) = U / (N * L)

"Mach number `U / c` [1]."
mach(U, c) = U / c

"Bulk Richardson number `N² L² / U²` [1], from the buoyancy frequency squared `N²` [1/s²]. Equals `1/froude²`."
richardson_bulk(N², U, L) = N² * L^2 / U^2

"Bulk Richardson number `Δb L / U²` [1], from the buoyancy jump `Δb` [m/s²] over the layer of thickness `L`."
richardson_bulk_jump(Δb, U, L) = Δb * L / U^2

"Rayleigh number `Δb H³ / (ν κ)` [1], from the buoyancy difference `Δb` [m/s², positive if unstable] over depth `H`."
rayleigh(Δb, H, ν, κ) = Δb * H^3 / (ν * κ)

"Rayleigh number for a fluid with thermal expansion coefficient `α` [1/K]: `g α ΔT H³ / (ν κ)`; `ΔT` [K] bottom minus top."
rayleigh(g, α, ΔT, H, ν, κ) = rayleigh(g * α * ΔT, H, ν, κ)

"Atwood number `(ρ₂ - ρ₁) / (ρ₂ + ρ₁)` [1], with `ρ₂` the heavier fluid."
atwood(ρ₁, ρ₂) = (ρ₂ - ρ₁) / (ρ₂ + ρ₁)

"Reduced gravity `g (ρ₂ - ρ₁) / ρ₂` [m/s²], with `ρ₂` the heavier fluid (Boussinesq reference `ρ₂`)."
reduced_gravity(g, ρ₁, ρ₂) = g * (ρ₂ - ρ₁) / ρ₂

"Aspect ratio `H / L` [1]."
aspect_ratio(H, L) = H / L
