## Initial conditions on a 2D box, independent of the model. A setup is a function of a model (for its fluid, box size
## and topology) returning the fields that initialisation needs, as a named tuple of functions:
##   consvar(x, z), q(x, z)   the tracers,
##   u(z)                     the mean flow,
##   ψ(X, Z)                  the mass streamfunction of the perturbation [kg m⁻¹ s⁻¹],
## with z = 0 at the bottom and x = 0 at the left. `initialize(model, fields)` builds the state from them: rest
## profiles, a discretely divergence-free perturbation for them, then the state. Models sharing their fluid, domain
## and space (e.g. AN2D and FC2D) get identical fields.

#####
##### Profiles, flows and streamfunctions
#####

"""
    linear_profile(bottom, top, Lz)

f(x, z), linear in z, from `bottom` at z = 0 to `top` at z = Lz.
"""
linear_profile(bottom, top, Lz) = (x, z) -> bottom + (top - bottom) * z / Lz

"""
    tanh_profile(below, above, z₀, δ)

f(x, z), a smooth step in z from `below` to `above`, centred on z₀ with thickness δ:
f = (below + above)/2 + (above - below)/2 tanh((z - z₀)/δ).
"""
tanh_profile(below, above, z₀, δ) = (x, z) -> (below + above) / 2 + (above - below) / 2 * tanh((z - z₀) / δ)

"""
    shear(Δu, z₀, δ), jet(U, z₀, δ), couette(U, Lz)

Mean flows u(z): a shear layer Δu/2 tanh((z - z₀)/δ) (Kelvin-Helmholtz), a jet U exp(-((z - z₀)/δ)²), and a linear
shear U z / Lz (e.g. between a fixed bottom wall and a top wall moving at U).
"""
shear(Δu, z₀, δ) = z -> Δu / 2 * tanh((z - z₀) / δ)
jet(U, z₀, δ) = z -> U * exp(-((z - z₀) / δ)^2)
couette(U, Lz) = z -> U * z / Lz

"""
    ψ = mode(model; amplitude, kx, z₀, δ)

Mass streamfunction ψ(X, Z) = amplitude cos(2π kx X / Lx) exp(-((Z - z₀)/δ)²) [kg m⁻¹ s⁻¹], with `kx` wavelengths
across the box, multiplied by a factor vanishing on the walls of the model's box (no flow through walls).
"""
function mode((; domain, space); amplitude, kx, z₀, δ)
    L, B = (space.Lx, space.Lz), boundary_topology(domain)
    return (X, Z) -> wall_factor(B, X, Z, L) * amplitude * cos(2π * kx * X / L[1]) * exp(-((Z - z₀) / δ)^2)
end

"""
    ψ = noise(model; amplitude, seed = 1)

Random mass streamfunction ψ(X, Z), uniform in ±amplitude [kg m⁻¹ s⁻¹] and reproducible from `seed` (a hash of the
position, wrapped in periodic directions so that the noise is periodic), multiplied by a factor vanishing on the
walls of the model's box.
"""
function noise((; domain, space); amplitude, seed = 1)
    L, B = (space.Lx, space.Lz), boundary_topology(domain)
    return (X, Z) -> wall_factor(B, X, Z, L) * amplitude *
                     (2 * (hash((position(B[1], X, L[1]), position(B[2], Z, L[2]), seed)) / typemax(UInt)) - 1)
end
position(::Periodic, x, L) = round(Int, mod(x / L, 1) * 2^30) % 2^30
position(::Bounded, x, L) = round(Int, x / L * 2^30)

wall_factor((Bx, Bz), X, Z, (Lx, Lz)) = wall_factor(Bx, X, Lx) * wall_factor(Bz, Z, Lz)
wall_factor(::Periodic, x, L) = one(x)
wall_factor(::Bounded, x, L) = sin(π * x / L)

"""
    consvar = consvar_from_theta(fluid, θ, q)

The conservative variable consvar(x, z) of `fluid` for a potential temperature θ and composition q (numbers or
functions f(x, z)), converted by ClimFluids. consvar is an entropic variable, a function of the entropy and the
composition only, so the conversion does not depend on pressure; it is evaluated at the reference pressure p0 of θ.
(A temperature profile cannot define consvar, as T also depends on pressure.)
"""
consvar_from_theta(fluid, θ, q) =
    (x, z) -> fluid(:p, :theta, :q).conservative_variable(fluid.p0, value(θ, x, z), value(q, x, z))
value(f::Number, x, z) = f
value(f, x, z) = f(x, z)

#####
##### Canonical setups, given in potential temperature θ and composition q
#####

"""
    fields = rest_state(model; θ, q)

Fluid at rest, with θ and q numbers or functions f(x, z).
"""
rest_state((; fluid); θ, q) =
    (; consvar = consvar_from_theta(fluid, θ, q), q = (x, z) -> value(q, x, z), u = z -> zero(z), ψ = (X, Z) -> zero(X))

"""
    fields = kelvin_helmholtz(model; θ₀, Δθ, q₀, Δq = 0, Δu, δ = Lz/20, kick = 0.01, kx = 1, ρ₀ = 1)

Shear layer at mid-depth over a stable tanh stratification (θ and q steps of thickness δ), with a small mode `kx`
perturbation of relative amplitude `kick` (ρ₀ is a density scale for the mass streamfunction).
"""
function kelvin_helmholtz(model; θ₀, Δθ, q₀, Δq = zero(q₀), Δu, δ = model.space.Lz / 20, kick = 0.01, kx = 1, ρ₀ = one(θ₀))
    Lz = model.space.Lz
    θ, q = tanh_profile(θ₀ - Δθ/2, θ₀ + Δθ/2, Lz/2, δ), tanh_profile(q₀ - Δq/2, q₀ + Δq/2, Lz/2, δ)
    return (; consvar = consvar_from_theta(model.fluid, θ, q), q, u = shear(Δu, Lz/2, δ),
              ψ = mode(model; amplitude = kick * ρ₀ * Δu * δ, kx, z₀ = Lz/2, δ = 2δ))
end

"""
    fields = rayleigh_benard(model; θ_bottom, θ_top, q₀, noise = 1e-3, ρ₀ = 1, seed = 1)

Conductive (linear) profile between a warm bottom and a cool top, with streamfunction noise of amplitude
`noise ρ₀ Lz` m² s⁻¹ × kg m⁻³ to trigger convection. Pair it with `Value` conditions on consvar at the walls.
"""
function rayleigh_benard(model; θ_bottom, θ_top, q₀, noise = 1e-3, ρ₀ = one(q₀), seed = 1)
    Lz = model.space.Lz
    q(x, z) = q₀
    return (; consvar = consvar_from_theta(model.fluid, linear_profile(θ_bottom, θ_top, Lz), q₀), q, u = z -> zero(z),
              ψ = TraceableEnergetics.noise(model; amplitude = noise * ρ₀ * Lz, seed))
end

"""
    fields = rising_bubble(model; θ₀, Δθ, q₀, Δq = 0, r = Lz/5, x₀ = Lx/2, z₀ = Lz/4, N² = 0)

Warm (Δθ > 0) or cold, possibly fresh/salty (Δq), bubble of radius r with a smooth (cosine) edge, in a background of
uniform stratification N² (θ = θ₀ (1 + N² z / g)).
"""
function rising_bubble(model; θ₀, Δθ, q₀, Δq = zero(q₀), r = model.space.Lz / 5, x₀ = model.space.Lx / 2,
                       z₀ = model.space.Lz / 4, N² = zero(θ₀))
    (; g) = model.space
    shape(x, z) = cos(π / 2 * min(hypot(x - x₀, z - z₀) / r, 1))^2      # 1 at the centre, 0 outside the radius
    θ(x, z) = θ₀ * (1 + N² * z / g) + Δθ * shape(x, z)
    q(x, z) = q₀ + Δq * shape(x, z)
    return (; consvar = consvar_from_theta(model.fluid, θ, q), q, u = z -> zero(z), ψ = (X, Z) -> zero(X))
end

"""
    fields = lock_exchange(model; θ_left, θ_right, q_left, q_right, δ = Lx/50)

Two fluids side by side, separated at x = Lx/2 by a front of width δ: a gravity current (use walls in x).
"""
function lock_exchange(model; θ_left, θ_right, q_left, q_right, δ = model.space.Lx / 50)
    Lx = model.space.Lx
    front(x) = (1 + tanh((x - Lx / 2) / δ)) / 2                         # 0 on the left, 1 on the right
    θ(x, z) = θ_left + (θ_right - θ_left) * front(x)
    q(x, z) = q_left + (q_right - q_left) * front(x)
    return (; consvar = consvar_from_theta(model.fluid, θ, q), q, u = z -> zero(z), ψ = (X, Z) -> zero(X))
end
